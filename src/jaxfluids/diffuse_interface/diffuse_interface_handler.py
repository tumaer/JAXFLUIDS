from typing import Tuple, Union

import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.diffuse_interface.diffuse_interface_compression_computer import DiffuseInterfaceCompressionComputer
from jaxfluids.diffuse_interface.diffuse_interface_geometry_calculator import DiffuseInterfaceGeometryCalculator
from jaxfluids.diffuse_interface.diffuse_interface_pde_regularization import DiffuseInterfacePDERegularizationComputer
from jaxfluids.diffuse_interface.diffuse_interface_thinc import DiffuseInterfaceTHINC
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.config import precision

class DiffuseInterfaceHandler():
    """ The DiffuseInterfaceHandler class manages computations to perform diffuse-interface
    computations.
    The main functionality includes
        - Computation of the source term in the volume fraction equation
        - Computation of source terms in momentum and energy equation due to surface tension
        - Interface compression
        - Curvature computation
        - THINC cell face reconstruction
        - Interface diffusion/sharpening flux computation

    The DiffuseInterfaceHandler holds three important members:
        - DiffuseInterfaceGeometryCalculator
        - DiffuseInterfaceCompressionComputer
        - DiffuseInterfaceTHINC
        - DiffuseInterfacePDERegularizationComputer
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            numerical_setup: NumericalSetup,
            material_manager: MaterialManager,
            unit_handler: UnitHandler,
            equation_manager: EquationManager,
            halo_manager: HaloManager,
            ) -> None:

        self.eps = precision.get_eps()

        self.halo_manager = halo_manager
        self.material_manager = material_manager
        equation_information = equation_manager.equation_information
        
        diffuse_interface_setup = numerical_setup.diffuse_interface
        self.diffuse_interface_model = diffuse_interface_setup.model
        self.is_interface_compression = diffuse_interface_setup.interface_compression.is_interface_compression
        self.is_thinc_reconstruction = diffuse_interface_setup.thinc.is_thinc_reconstruction
        self.is_diffusion_sharpening = diffuse_interface_setup.diffusion_sharpening.is_diffusion_sharpening

        # INTERFACE COMPRESSION PARAMETERS
        self.interval_compression = diffuse_interface_setup.interface_compression.interval
        self.steps_compression = diffuse_interface_setup.interface_compression.steps
        self.CFL_compression = diffuse_interface_setup.interface_compression.CFL

        self.geometry_calculator = DiffuseInterfaceGeometryCalculator(
                domain_information=domain_information,
                diffuse_interface_setup=diffuse_interface_setup,
                halo_manager=halo_manager)

        self.interface_compression_computer = \
            DiffuseInterfaceCompressionComputer(
                domain_information=domain_information,
                equation_manager=equation_manager,
                material_manager=self.material_manager,
                halo_manager=halo_manager,
                diffuse_interface_setup=diffuse_interface_setup,
                geometry_calculator=self.geometry_calculator)

        self.interface_thinc = DiffuseInterfaceTHINC(
            domain_information=domain_information,
            equation_manager=equation_manager,
            material_manager=material_manager,
            diffuse_interface_setup=diffuse_interface_setup,
            halo_manager=halo_manager)
        
        self.pde_regularization = DiffuseInterfacePDERegularizationComputer(
            domain_information=domain_information,
            equation_information=equation_information,
            material_manager=material_manager,
            diffuse_interface_setup=diffuse_interface_setup,
            geometry_calculator=self.geometry_calculator)

        self.dim = domain_information.dim
        self.active_axes_indices = domain_information.active_axes_indices
        self.nx, self.ny, self.nz = domain_information.global_number_of_cells
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nhx__, self.nhy__, self.nhz__ = domain_information.domain_slices_conservatives_to_geometry
        self.is_parallel = domain_information.is_parallel
        self.split_factors = domain_information.split_factors
        self.is_mesh_stretching = domain_information.is_mesh_stretching

        self.mass_ids = equation_information.mass_ids
        self.mass_slices = equation_information.mass_slices
        self.vel_ids = equation_information.velocity_ids
        self.vel_slices = equation_information.velocity_slices
        self.energy_ids = equation_information.energy_ids
        self.energy_slices = equation_information.energy_slices
        self.vf_ids = equation_information.vf_ids
        self.vf_slices = equation_information.vf_slices
        
        self.flux_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]
                    
    def compute_curvature(self, volume_fraction: Array) -> Array:
        """Wrapper function around geometry_caculator.compute_curvature.
        Computes the corrected curvature from volume fraction. The volume
        fraction is limited to admissible values to avoid problems with 
        machine precision.

        :param volume_fraction: Volume fraction buffer, shape=(Nx+2Nh, Ny+2Nh, Nz+2Nh)
        :type volume_fraction: Array
        :return: Curvature buffer, shape=(Nx+2Nh_geo, Ny+2Nh_geo, Nz+2Nh_geo)
        :rtype: Array
        """
        curvature = self.geometry_calculator.compute_curvature(volume_fraction)
        return curvature

    def compute_normal(self, volume_fraction: Array) -> Array:
        """Wrapper function around geometry_calculator.compute_normal. Computes 
        the normal vector for domain and geometry halo_cells. The volume
        fraction is limited to admissible values to avoid problems with 
        machine precision.

        :param volume_fraction: Volume fraction buffer, shape=(Nx+2Nh, Ny+2Nh, Nz+2Nh)
        :type volume_fraction: Array
        :return: _description_
        :rtype: Array
        """
        normal = self.geometry_calculator.compute_normal(volume_fraction)
        slice_normal_to_geometry = self.geometry_calculator.normal_to_geometry
        return normal[slice_normal_to_geometry]
    
    def compute_volume_fraction_source_term(
            self,
            u_hat_xi: Array,
            volume_fraction: Array,
            one_cell_size_xi: Union[float, Array],
            axis_index_xi: int
            ) -> Array:
        """Computes the volume fraction source term for the quasi-conservative
        volume fraction advection equation.

        s_i = alpha_i * (u^{star}_{i+1/2} - u^{star}_{i-1/2}) / dx

        :param u_hat_xi: Buffer of upwinded velocity from Riemann solver,
        defined at cell-faces
        :type u_hat_xi: Array
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: Array
        :param one_cell_size_xi: One over cell size buffer
        :type one_cell_size_xi: Union[float, Array]
        :param axis_index_xi: Index of the current axis direction
        :type axis_index_xi: int
        :return: Source term for volume fraction transport equation
        :rtype: Array
        """

        rhs_volume_fraction_contribution = one_cell_size_xi \
            * volume_fraction[...,self.nhx,self.nhy,self.nhz] \
            * (u_hat_xi[self.flux_slices[axis_index_xi][1]] \
                - u_hat_xi[self.flux_slices[axis_index_xi][0]])
        return rhs_volume_fraction_contribution
    
    def compute_surface_tension_source_term_xi(
            self,
            u_hat_xi: Array,
            alpha_hat_xi: Array,
            alpha: Array,
            curvature: Array,
            one_cell_size_xi: Union[float, Array],
            axis_index_xi: int
            ) -> Tuple[Array, Array]:
        """Computes the source term for the momentum and energy
        equation due to surface tension.

        :param u_hat_xi: Buffer of upwinded velocity from Riemann solver,
        defined at cell faces
        :type u_hat_xi: Array
        :param alpha_hat_xi: Buffer of upwinded volume fraction from Riemann solver,
        defined at cell faces
        :type alpha_hat_xi: Array
        :param alpha: Volume fraction buffer, defined at cell center
        :type alpha: Array
        :param curvature: Curvature buffer, defined at cell center
        :type curvature: Array
        :param one_cell_size_xi: One over cell size buffer
        :type one_cell_size_xi: Union[float, Array]
        :param axis_index_xi: Index of the current axis direction
        :type axis_index_xi: int
        :return: Source terms for momentum equation and energy equation, respectively
        :rtype: Tuple[Array, Array]
        """
        
        curvature = curvature[...,self.nhx_,self.nhy_,self.nhz_]
        sigma = self.material_manager.get_sigma()
        sigma_curvature_over_cell_size_xi = sigma * curvature * one_cell_size_xi
        
        # SURFACE TENSION KERNEL
        if self.geometry_calculator.surface_tension_kernel:
            # kernel = self.geometry_calculator.compute_surface_tension_kernel(
            #     alpha[...,self.nhx,self.nhy,self.nhz])
            # sigma_curvature_over_cell_size_xi *= kernel
            # kernel = self.geometry_calculator.compute_surface_tension_kernel(alpha_hat_xi)
            # alpha_hat_xi *= kernel
            alpha_hat_xi = 6.0 * alpha_hat_xi * alpha_hat_xi * (0.5 - alpha_hat_xi / 3.0)
        
        delta_alpha_hat = alpha_hat_xi[self.flux_slices[axis_index_xi][1]] \
            - alpha_hat_xi[self.flux_slices[axis_index_xi][0]]
        delta_u_hat = u_hat_xi[self.flux_slices[axis_index_xi][1]] \
            - u_hat_xi[self.flux_slices[axis_index_xi][0]]
        
        delta_alpha_hat_u_hat = alpha_hat_xi * u_hat_xi
        delta_alpha_hat_u_hat = delta_alpha_hat_u_hat[self.flux_slices[axis_index_xi][1]] \
            - delta_alpha_hat_u_hat[self.flux_slices[axis_index_xi][0]]
        
        rhs_momentum_contribution = sigma_curvature_over_cell_size_xi * delta_alpha_hat

        rhs_energy_contribution = sigma_curvature_over_cell_size_xi * (delta_alpha_hat_u_hat \
            - alpha[...,self.nhx,self.nhy,self.nhz] * delta_u_hat)

        return rhs_momentum_contribution, rhs_energy_contribution

    def get_compression_flag(
            self,
            simulation_step: jnp.int32
            ) -> bool:
        """Decides whether to perform interface compression 
        for the current simulation step based on the 
        the interface compression interval.

        :param simulation_step: _description_
        :type simulation_step: jnp.int32
        :return: _description_
        :rtype: bool
        """
        if self.interval_compression == 0:
            perform_compression = False
        else:
            if simulation_step % self.interval_compression == 0:
                perform_compression = True
            else:
                perform_compression = False
        return perform_compression

    def perform_interface_compression(
            self, 
            conservatives: Array,
            primitives: Array,
            physical_simulation_time: float,
            is_interface_compression: bool
            ) -> Array:
        """Wrapper function for interface compression.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param is_interface_compression: _description_
        :type is_interface_compression: bool
        :return: _description_
        :rtype: Array
        """
        
        if is_interface_compression: 
            conservatives, primitives = self.interface_compression_computer.perform_interface_compression(
                conservatives, primitives, 
                self.CFL_compression,
                self.steps_compression,
                physical_simulation_time)
        
        return conservatives, primitives

    def thinc_reconstruct_xi(
            self,
            conservatives_L: Array,
            conservatives_R: Array,
            primitives_L: Array,
            primitives_R: Array,
            conservatives: Array,
            primitives: Array,
            curvature: Array,
            axis: int,
            ) -> Tuple[Array, Array, Array, Array]:
        """Applies THINC reconstruction to the volume fraction 
        and (if active) corrects other quantities.

        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param curvature: _description_
        :type curvature: Array
        :param axis: _description_
        :type axis: _type_
        :return: _description_
        :rtype: Array
        """

        if self.diffuse_interface_model == "5EQM":
            volume_fraction = conservatives[self.vf_ids]
        
        normal = self.compute_normal(volume_fraction)

        # SLICE NORMAL AND CURVATURE FROM NH_GEOMETRY 
        # TO DOMAIN + 1 HALO CELL EACH SIDE
        normal = normal[self.geometry_calculator.geometry_to_domain_plus_one]
        if curvature is not None:
            curvature = curvature[self.geometry_calculator.geometry_to_domain_plus_one]

        conservatives_L, conservatives_R, primitives_L, primitives_R \
        = self.interface_thinc.reconstruct_xi(
            conservatives_L, conservatives_R,
            primitives_L, primitives_R,
            conservatives, primitives,
            normal, curvature, axis)
        return conservatives_L, conservatives_R, primitives_L, primitives_R
    
    def compute_diffusion_sharpening_flux_xi(
            self,
            conservatives: Array,
            primitives: Array,
            axis: int,
            numerical_dissipation: Array = None
            ) -> Tuple[Array, int]:
        """Computes ACDI-type interface diffusion and sharpening fluxes
        for the 4-equation or 5-equation diffuse-interface model.

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param axis: Spatial direction
        :type axis: int
        :return: _description_
        :rtype: Array
        """
        regularization_flux_xi, count_acdi_xi \
            = self.pde_regularization.compute_diffusion_sharpening_flux_xi(
                conservatives, primitives, axis, numerical_dissipation)
        
        regularization_flux_xi = self.halo_manager.boundary_condition_flux.face_flux_update(
            regularization_flux_xi, axis)

        return regularization_flux_xi, count_acdi_xi
        
    def compute_diffusion_sharpening_timestep(
            self,
            primitives: Array
            ) -> float:
        return self.pde_regularization.compute_timestep(primitives)
