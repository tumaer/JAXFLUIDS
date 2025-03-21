from __future__ import annotations
from typing import Dict, Union, TYPE_CHECKING, Tuple, List
import copy

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.riemann_solvers.eigendecomposition import Eigendecomposition
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.stencils.reconstruction.shock_capturing.weno import WENO1
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.split_reconstruction import SplitReconstruction
from jaxfluids.config import precision

if TYPE_CHECKING:
    from jaxfluids.data_types.ml_buffers import MachineLearningSetup
    from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup, HighOrderGodunovSetup
    from jaxfluids.data_types.numerical_setup.conservatives import PositivitySetup
    from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
    from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
    from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler

Array = jax.Array

class HighOrderGodunov(ConvectiveFluxSolver):
    """The HighOrderGodunov class implements the flux calculation
    according to the high-order Godunov approach in the finite volume
    framework.

    The calculation of the fluxes consists of three steps:
    1) RECONSTRUCT STATE ON CELL FACE
    2) CONVERT PRIMITIVES TO CONSERVATIVES AND VICE VERSA
    3) SOLVE RIEMANN PROBLEM

    The reconstruction step can be done on primitive or conservative variables
    in either physical or characteristic space. The safe reconstruction guards
    against reconstruction of inadmissible states (e.g., negative pressure or
    density) by resorting to first-order upwind reconstruction in problematic
    cells.

    """
    def __init__(
            self,
            convective_fluxes_setup: ConvectiveFluxesSetup,
            positivity_setup: PositivitySetup,
            diffuse_interface_setup: DiffuseInterfaceSetup,
            material_manager: MaterialManager,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            diffuse_interface_handler: DiffuseInterfaceHandler,
            positivity_handler: PositivityHandler,
            **kwargs
            ) -> None:

        super(HighOrderGodunov, self).__init__(
            convective_fluxes_setup, material_manager, domain_information, equation_manager)

        godunov_setup: HighOrderGodunovSetup = convective_fluxes_setup.godunov

        self.reconstruction_variable = godunov_setup.reconstruction_variable
        thinc_setup = diffuse_interface_setup.thinc
        self.is_thinc_reconstruction = thinc_setup.is_thinc_reconstruction
        self.is_interpolation_limiter = positivity_setup.is_interpolation_limiter
        self.is_thinc_interpolation_limiter = positivity_setup.is_thinc_interpolation_limiter
        self.is_consistent_reconstruction = diffuse_interface_setup.is_consistent_reconstruction

        self.diffuse_interface_handler = diffuse_interface_handler
        self.positivity_handler = positivity_handler

        riemann_solver = godunov_setup.riemann_solver
        self.riemann_solver: RiemannSolver = riemann_solver(
            material_manager=material_manager,
            equation_manager=equation_manager,
            signal_speed=godunov_setup.signal_speed,
            catum_setup=godunov_setup.catum_setup)

        reconstruction_stencil = godunov_setup.reconstruction_stencil
        split_reconstruction = godunov_setup.split_reconstruction
        if split_reconstruction is not None:
            self.reconstruction_stencil = SplitReconstruction(
                nh=domain_information.nh_conservatives,
                inactive_axes=domain_information.inactive_axes,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=domain_information.get_global_cell_sizes_halos(),
                equation_information=equation_manager.equation_information,
                split_reconstruction_setup=godunov_setup.split_reconstruction)
 
        elif reconstruction_stencil is not None:
            self.reconstruction_stencil: SpatialReconstruction = reconstruction_stencil(
                nh=domain_information.nh_conservatives, 
                inactive_axes=domain_information.inactive_axes,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=domain_information.get_global_cell_sizes_halos(),)

        if self.equation_manager.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM" \
            and self.equation_manager.equation_information.active_physics.is_surface_tension:
            self.reconstruction_stencil_geometry = WENO1(
                nh=domain_information.nh_geometry,
                inactive_axes=domain_information.inactive_axes)

        if self.reconstruction_variable in ['CHAR-PRIMITIVE', 'CHAR-CONSERVATIVE']:
            if self.is_consistent_reconstruction:
                self.reconstruction_stencil_ = copy.deepcopy(self.reconstruction_stencil)
            
            self.reconstruction_stencil.set_slices_stencil()
            self.eigendecomposition = Eigendecomposition(
                material_manager=material_manager,
                stencil_size=self.reconstruction_stencil._stencil_size, 
                domain_information=domain_information,
                equation_information=equation_manager.equation_information,
                frozen_state=godunov_setup.frozen_state)

    def compute_flux_xi(
            self,
            primitives: Array,
            conservatives: Array,
            axis: int,
            curvature: Array = None,
            volume_fraction: Array = None,
            apertures: Tuple[Array] = None,
            ml_setup: MachineLearningSetup = None,
        ) -> Tuple[Array, Array, Array, int]:
        """Computes the numerical flux in a specified spatial direction.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param conservatives: Buffer of conservative variables.
        :type conservatives: Array
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis direction.
        :rtype: Array
        """

        cell_sizes = self.domain_information.get_device_cell_sizes()
        equation_type = self.equation_information.equation_type

        conservative_xi_L, conservative_xi_R, \
        primitives_xi_L, primitives_xi_R = self.reconstruct_xi(
            primitives, conservatives, cell_sizes, axis,
            curvature, ml_setup)
        
        # DIFFUSE INTERFACE SPECIFIC - THINC RECONSTRUCTION
        count_interpolation_limiter_thinc = None
        if equation_type in ("DIFFUSE-INTERFACE-4EQM", "DIFFUSE-INTERFACE-5EQM") \
            and self.is_thinc_reconstruction:

            conservative_xi_L, conservative_xi_R, \
            primitives_xi_L, primitives_xi_R \
            = self.diffuse_interface_handler.thinc_reconstruct_xi(
                conservative_xi_L, conservative_xi_R,
                primitives_xi_L, primitives_xi_R,
                conservatives, primitives, 
                curvature, axis,
                volume_fraction=volume_fraction)


        # INTERPOLATION LIMITER
        if self.is_interpolation_limiter:
            conservative_xi_L, primitives_xi_L, count_L \
            = self.positivity_handler.limit_interpolation_xi(
                primitives=primitives, primitives_xi_j=primitives_xi_L,
                j=0, cell_sizes=cell_sizes, axis=axis,
                apertures=apertures)
            conservative_xi_R, primitives_xi_R, count_R \
            = self.positivity_handler.limit_interpolation_xi(
                primitives=primitives, primitives_xi_j=primitives_xi_R,
                j=1, cell_sizes=cell_sizes, axis=axis,
                apertures=apertures)
            count_interpolation_limiter = count_L + count_R
        else:
            count_interpolation_limiter = None
            
        # # DIFFUSE INTERFACE SPECIFIC - THINC RECONSTRUCTION
        # count_interpolation_limiter_thinc = None
        # if equation_type in ("DIFFUSE-INTERFACE-4EQM", "DIFFUSE-INTERFACE-5EQM") \
        #     and self.is_thinc_reconstruction:

        #     conservative_xi_L, conservative_xi_R, \
        #     primitives_xi_L, primitives_xi_R \
        #     = self.diffuse_interface_handler.thinc_reconstruct_xi(
        #         conservative_xi_L, conservative_xi_R,
        #         primitives_xi_L, primitives_xi_R,
        #         conservatives, primitives, 
        #         curvature, axis,
        #         volume_fraction=volume_fraction)

        #     # LIMIT THINC RECONSTRUCTION
        #     # TODO combine with previous interpolation limiter
        #     if self.is_thinc_interpolation_limiter:
        #         conservative_xi_L, primitives_xi_L, count_thinc_L = \
        #             self.positivity_handler.compute_positivity_preserving_thinc_interpolation(
        #                 primitives=primitives,
        #                 primitives_xi_j=primitives_xi_L,
        #                 j=0,
        #                 cell_sizes=cell_sizes,
        #                 axis=axis)
        #         conservative_xi_R, primitives_xi_R, count_thinc_R = \
        #             self.positivity_handler.compute_positivity_preserving_thinc_interpolation(
        #                 primitives=primitives,
        #                 primitives_xi_j=primitives_xi_R,
        #                 j=1,
        #                 cell_sizes=cell_sizes,
        #                 axis=axis)
        #         count_interpolation_limiter_thinc = count_thinc_L + count_thinc_R

        # DIFFUSE INTERFACE SPECIFIC - CURVATURE RECONSTRUCTION
        if equation_type == "DIFFUSE-INTERFACE-5EQM" \
            and self.equation_information.active_physics.is_surface_tension:
            curvature_L = self.reconstruction_stencil_geometry.reconstruct_xi(curvature, axis, 0)
            curvature_R = self.reconstruction_stencil_geometry.reconstruct_xi(curvature, axis, 1)
        else:
            curvature_L, curvature_R = None, None


        # SOLVE RIEMANN PROBLEM
        fluxes_xi, u_hat, phi_hat = self.riemann_solver.solve_riemann_problem_xi(
            primitives_xi_L, primitives_xi_R,
            conservative_xi_L, conservative_xi_R,
            axis,
            curvature_L=curvature_L,
            curvature_R=curvature_R,
            ml_setup=ml_setup,
        )

        return fluxes_xi, u_hat, phi_hat, count_interpolation_limiter, \
            count_interpolation_limiter_thinc

    def reconstruct_xi(
            self,
            primitives: Array,
            conservatives: Array,
            cell_sizes: Tuple[Array],
            axis: int,
            curvature: Union[Array, None],
            ml_setup: MachineLearningSetup = None,
        ) -> Tuple[Array, Array, Array, Array]:
        """Wrapper function around various reconstruction techniques for the 
        HighOrderGodunov approach.

        1) Reconstruction of the primitive variables
        2) Reconstruction of the conservative variables
        3) Reconstruction of the characteristic variables via primitives
        4) Reconstruction of the characteristic variables via conservatives 

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param cell_sizes: Tuple of cell sizes
        :type cell_sizes: Tuple[Array]
        :param axis: Integer indicating the axis direction, i.e. (0,1,2)
        :type axis: int
        :param curvature: Buffer of the curvature, necessary for the 
            characteristic decomposition in the diffuse interface model
        :type curvature: Union[Array, None]
        :param ml_setup: Dict of neural network parameters
        :type ml_setup: MachineLearningSetup
        :return: Returns left and right reconstructed conservatives and primitives
        :rtype: Tuple[Array, Array, Array, Array]
        """

        if self.reconstruction_variable == "PRIMITIVE":
            primitives_xi_L = self.reconstruction_stencil.reconstruct_xi(
                primitives, axis=axis,
                j=0, dx=cell_sizes[axis],
                ml_setup=ml_setup
            )
            primitives_xi_R = self.reconstruction_stencil.reconstruct_xi(
                primitives, axis=axis,
                j=1, dx=cell_sizes[axis],
                ml_setup=ml_setup
            )

            conservative_xi_L = self.equation_manager.get_conservatives_from_primitives(primitives_xi_L)
            conservative_xi_R = self.equation_manager.get_conservatives_from_primitives(primitives_xi_R)

        if self.reconstruction_variable == "CONSERVATIVE":
            conservative_xi_L = self.reconstruction_stencil.reconstruct_xi(
                conservatives, axis=axis,
                j=0, dx=cell_sizes[axis],
                ml_setup=ml_setup
            )
            conservative_xi_R = self.reconstruction_stencil.reconstruct_xi(
                conservatives, axis=axis,
                j=1, dx=cell_sizes[axis],
                ml_setup=ml_setup
            )

            # TODO pass temperature
            primitives_xi_L = self.equation_manager.get_primitives_from_conservatives(conservative_xi_L)
            primitives_xi_R = self.equation_manager.get_primitives_from_conservatives(conservative_xi_R)

        if self.reconstruction_variable == "CHAR-PRIMITIVE":
            # # OLD VERSION
            # stencil_prime_window = self.eigendecomposition.get_stencil_window(primitives, axis=axis)
            # right_eigs, left_eigs = self.eigendecomposition.eigendecomposition_primitives(
            #     primitives, curvature, axis)
            # char = self.eigendecomposition.transformtochar(stencil_prime_window, left_eigs, axis=axis)
            # char_xi_L = self.reconstruction_stencil.reconstruct_xi(char, axis, 0, dx=cell_sizes[axis])
            # char_xi_R = self.reconstruction_stencil.reconstruct_xi(char, axis, 1, dx=cell_sizes[axis])
            # primitives_xi_L = self.eigendecomposition.transformtophysical(char_xi_L, right_eigs)
            # primitives_xi_R = self.eigendecomposition.transformtophysical(char_xi_R, right_eigs)

            # NEW VERSION
            stencil_prime_window = self.eigendecomposition.get_stencil_window(primitives, axis=axis)
            char = self.eigendecomposition.get_characteristics_from_primitives(
                (stencil_prime_window,), primitives, curvature, axis)[0]
            char_xi_L = self.reconstruction_stencil.reconstruct_xi(char, axis, 0, dx=cell_sizes[axis])
            char_xi_R = self.reconstruction_stencil.reconstruct_xi(char, axis, 1, dx=cell_sizes[axis])
            primitives_xi_L, primitives_xi_R = self.eigendecomposition.get_primitives_from_characteristics(
                (char_xi_L, char_xi_R,), primitives, curvature, axis)

            if self.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM" \
                and self.is_consistent_reconstruction:

                s_velocity = self.equation_information.s_velocity
                s_volume_fraction = self.equation_information.s_volume_fraction
                ids_volume_fraction = self.equation_information.ids_volume_fraction

                interface_mask = self.diffuse_interface_handler.interface_thinc.compute_interface_mask(primitives[ids_volume_fraction], axis)
                slice_L, slice_R = self.diffuse_interface_handler.interface_thinc.slices_LR[axis]

                def consistent_reconstruction(primitives_K, slice_K, K):
                    # velocities_K = self.reconstruction_stencil_.reconstruct_xi(
                    #     primitives[s_velocity],
                    #     axis=axis, j=K, dx=cell_sizes[axis],
                    #     ml_setup=ml_setup,
                    # )

                    # vf_K = self.reconstruction_stencil_.reconstruct_xi(
                    #     primitives[s_volume_fraction],
                    #     axis=axis, j=K, dx=cell_sizes[axis],
                    #     ml_setup=ml_setup,
                    # )

                    # primitives_K = primitives_K.at[s_velocity].set(
                    #     jnp.where(interface_mask[slice_K], velocities_K, primitives_K[s_velocity]))
                    # primitives_K = primitives_K.at[s_volume_fraction].set(
                    #     jnp.where(interface_mask[slice_K], vf_K, primitives_K[s_volume_fraction]))
                    
                    primitives_consistent_K = self.reconstruction_stencil_.reconstruct_xi(
                        primitives, axis=axis, j=K, dx=cell_sizes[axis],
                        ml_setup=ml_setup
                    )

                    primitives_K = jnp.where(interface_mask[slice_K], primitives_consistent_K, primitives_K)

                    return primitives_K

                primitives_xi_L = consistent_reconstruction(primitives_xi_L, slice_L, 0)
                primitives_xi_R = consistent_reconstruction(primitives_xi_R, slice_R, 1)


                # self.nhx, self.nhy, self.nhz = self.domain_information.domain_slices_conservatives
                # if axis == 0:
                #     vf_ = 0.5 * (primitives[-1,3:-4,self.nhy,self.nhz] + primitives[-1,4:-3,self.nhy,self.nhz])
                # if axis == 1:
                #     vf_ = 0.5 * (primitives[-1,self.nhx,3:-4,self.nhz] + primitives[-1,self.nhx,4:-3,self.nhz])
                # mask = jnp.where((vf_ > 1e-3) & (vf_ < 1-1e-3), 1, 0)
                # primitives_xi_L_ = self.reconstruction_stencil_.reconstruct_xi(
                #     primitives,
                #     axis=axis,
                #     j=0,
                #     dx=cell_sizes[axis],
                #     ml_setup=ml_setup,
                # )
                # primitives_xi_R_ = self.reconstruction_stencil_.reconstruct_xi(
                #     primitives,
                #     axis=axis,
                #     j=1,
                #     dx=cell_sizes[axis],
                #     ml_setup=ml_setup,
                # )
                # primitives_xi_L = mask * primitives_xi_L_ + (1 - mask) * primitives_xi_L
                # primitives_xi_R = mask * primitives_xi_R_ + (1 - mask) * primitives_xi_R

                # ids_mass_and_volume_fraction = self.equation_information.ids_mass_and_volume_fraction
                # primitives_xi_L_ = self.reconstruction_stencil_.reconstruct_xi(
                #     primitives[ids_mass_and_volume_fraction,...],
                #     axis=axis,
                #     j=0,
                #     dx=cell_sizes[axis],
                #     ml_setup=ml_setup,
                # )
                # primitives_xi_L = primitives_xi_L.at[ids_mass_and_volume_fraction,...].set(primitives_xi_L_)
                
                # primitives_xi_R_ = self.reconstruction_stencil_.reconstruct_xi(
                #     primitives[ids_mass_and_volume_fraction,...],
                #     axis=axis,
                #     j=1,
                #     dx=cell_sizes[axis],
                #     ml_setup=ml_setup,
                # )
                # primitives_xi_R = primitives_xi_R.at[ids_mass_and_volume_fraction,...].set(primitives_xi_R_)
            
            conservative_xi_L = self.equation_manager.get_conservatives_from_primitives(primitives_xi_L)
            conservative_xi_R = self.equation_manager.get_conservatives_from_primitives(primitives_xi_R)

        if self.reconstruction_variable == "CHAR-CONSERVATIVE":
            stencil_cons_window = self.eigendecomposition.get_stencil_window(conservatives, axis=axis) 
            right_eigs, left_eigs = self.eigendecomposition.eigendecomposition_conservatives(primitives, axis=axis)
            char = self.eigendecomposition.transformtochar(stencil_cons_window, left_eigs, axis=axis)

            char_xi_L = self.reconstruction_stencil.reconstruct_xi(char, axis, 0, dx=cell_sizes[axis])
            char_xi_R = self.reconstruction_stencil.reconstruct_xi(char, axis, 1, dx=cell_sizes[axis])

            conservative_xi_L = self.eigendecomposition.transformtophysical(char_xi_L, right_eigs)
            conservative_xi_R = self.eigendecomposition.transformtophysical(char_xi_R, right_eigs)
            
            # TODO pass temperature
            primitives_xi_L = self.equation_manager.get_primitives_from_conservatives(conservative_xi_L)
            primitives_xi_R = self.equation_manager.get_primitives_from_conservatives(conservative_xi_R)

        return conservative_xi_L, conservative_xi_R, primitives_xi_L, primitives_xi_R
