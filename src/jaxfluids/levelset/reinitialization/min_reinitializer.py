from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.derivative.second_deriv_second_order_center import SecondDerivativeSecondOrderCenter
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.time_integration.RK2 import RungeKutta2
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.halos.halo_manager import HaloManager
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

class MinReinitializer(PDEBasedReinitializer):
    """Second order ENO based reinitializer with subcell fix
    according to \cite Min2010
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            reinitialization_setup: LevelsetReinitializationSetup,
            halo_cells: int,
            narrowband_setup: NarrowBandSetup,
            ) -> None:

        super(MinReinitializer, self).__init__(
            domain_information, halo_manager,
            reinitialization_setup, narrowband_setup)

        is_halos = reinitialization_setup.is_halos
        self.is_jaxforloop = reinitialization_setup.is_jaxforloop
        
        self.time_integrator = RungeKutta2(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = halo_cells if is_halos else 0)

        self.second_derivative_stencil = SecondDerivativeSecondOrderCenter(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = halo_cells + 1 if is_halos else 1)

        self.first_derivative_stencil = FirstDerivativeFirstOrderCenter(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = halo_cells if is_halos else 0)

        is_halos = self.reinitialization_setup.is_halos
        active_axes_indices = self.domain_information.active_axes_indices
        nh_conservatives = self.domain_information.nh_conservatives 
        nh_offset = self.domain_information.nh_offset
        nh = nh_offset if is_halos else nh_conservatives
        nhx, nhy, nhz = tuple(
            [jnp.s_[nh:-nh] if
            i in active_axes_indices else
            jnp.s_[:] for i in range(3)]
            )

        self.s_0 = (nhx, nhy, nhz)

        self.s_1 = tuple([
            jnp.s_[1:-1] if i in active_axes_indices
            else jnp.s_[:] for i in range(3) 
        ])

        self.s_2 = [[  
                jnp.s_[nh-1*j:-nh-1*j,nhy,nhz],
                jnp.s_[nhx,nh-1*j:-nh-1*j,nhz],
                jnp.s_[nhx,nhy,nh-1*j:-nh-1*j]
            ] for j in [1,-1]]


    def perform_reinitialization(
            self,
            levelset: Array,
            CFL: float,
            steps: int,
            mask: Array = None,
            debug: bool = False,
            **kwargs,
            ) -> Tuple[Array, float]:
        """Reinitializes the levelset buffer.

        :param levelset: _description_
        :type levelset: Array
        :param CFL: _description_
        :type CFL: float
        :param steps: _description_
        :type steps: int
        :param mask: _description_, defaults to None
        :type mask: Array, optional
        :param mask_residual: _description_, defaults to None
        :type mask_residual: Array, optional
        :param debug: _description_, defaults to False
        :type debug: bool, optional
        :return: _description_
        :rtype: Tuple[Array, float]
        """

        is_jaxforloop = self.reinitialization_setup.is_jaxforloop
        is_jaxwhileloop = self.reinitialization_setup.is_jaxwhileloop


        levelset_0 = jnp.array(levelset, copy=True)

        if mask == None:
            mask = jnp.ones_like(levelset[self.s_0], dtype=jnp.uint32)

        # DISTANCE APPROXIMATION
        distance_L, distance_R, = self.compute_distance_approximation(
            levelset_0)

        # TIMESTEP SIZE
        fictitious_timestep_size = self.compute_timestep_size(
            levelset, distance_L, distance_R, CFL)

        # REINITIALIZATION STEPS
        if is_jaxforloop:
            def _body_func(index, levelset: Array) -> Array:
                if debug:
                    levelset_in = levelset[index]
                else:
                    levelset_in = levelset
                levelset_out, _ = self.do_integration_step(levelset_in, levelset_0, distance_L,
                                                    distance_R, mask, fictitious_timestep_size)
                if debug:
                    levelset = levelset.at[index+1].set(levelset_out)
                else:
                    levelset = levelset_out
                return levelset

            if debug:
                levelset_buffer = jnp.zeros((steps+1,)+levelset.shape)
                levelset = levelset_buffer.at[0].set(levelset)
            levelset = jax.lax.fori_loop(0, steps, _body_func, levelset)
            step_count = steps

        elif is_jaxwhileloop:
            raise NotImplementedError
    
        return levelset, step_count

    def compute_residual(
            self,
            levelset: Array,
            mask: Array
            ) -> Tuple[float, float]:
        """Computes the mean and max residual.

        :param levelset: _description_
        :type levelset: Array
        :param mask: _description_
        :type mask: Array
        :return: _description_
        :rtype: _type_
        """
        is_parallel = self.domain_information.is_parallel
        distance_L, distance_R, = self.compute_distance_approximation(levelset)
        rhs = self.compute_rhs(levelset, levelset, distance_L, distance_R, mask)
        residual = jnp.abs(rhs*mask)
        mean_residual = jnp.sum(residual, axis=(-3,-2,-1))
        denominator = jnp.sum(mask, axis=(-3,-2,-1))
        max_residual = jnp.max(residual, axis=(-3,-2,-1))
        if is_parallel:
            mean_residual = jax.lax.psum(mean_residual, axis_name="i")
            denominator = jax.lax.psum(denominator, axis_name="i")
            max_residual = jax.lax.pmax(max_residual, axis_name="i")
        mean_residual = mean_residual/(denominator + 1e-30)
        return mean_residual, max_residual, rhs

    def compute_timestep_size(
            self,
            levelset: Array,
            distance_L: Array,
            distance_R: Array,
            CFL: Array
            ) -> float:
        """Computes the fictitious timestep size.

        :param levelset: _description_
        :type levelset: Array
        :param distance_L: _description_
        :type distance_L: Array
        :param distance_R: _description_
        :type distance_R: Array
        :param CFL: _description_
        :type CFL: Array
        :return: _description_
        :rtype: float
        """

        # COMPUTE TIMESTEP
        smallest_cell_size = self.domain_information.smallest_cell_size
        dxi_min = smallest_cell_size * jnp.ones_like(levelset[self.s_0])
        active_axes_indices = self.domain_information.active_axes_indices
        for axis, dxi_L, dxi_R in zip(active_axes_indices, distance_L, distance_R):
            mask_L = self.compute_directional_cut_cell_mask(levelset, axis, 0)
            mask_R = self.compute_directional_cut_cell_mask(levelset, axis, 1)
            dxi_L = dxi_L * mask_L + (1 - mask_L) * 1e10
            dxi_R = dxi_R * mask_R + (1 - mask_R) * 1e10
            dxi_min = jnp.minimum(dxi_min, jnp.minimum(dxi_L, dxi_R))
        fictitious_timestep_size = CFL * dxi_min
        return fictitious_timestep_size


    # @partial(jax.jit, static_argnums=(0))
    def do_integration_step(
            self,
            levelset: Array,
            levelset_0: Array,
            distance_L: Array,
            distance_R: Array,
            mask: Array,
            fictitious_timestep_size: float
            ) -> Tuple[Array, Array]:
        """Performs an integration step of the levelset
        reinitialization equation.

        :param levelset: _description_
        :type levelset: Array
        :param levelset_0: _description_
        :type levelset_0: Array
        :param distance_L: _description_
        :type distance_L: Array
        :param distance_R: _description_
        :type distance_R: Array
        :param mask: _description_
        :type mask: Array
        :param fictitious_timestep_size: _description_
        :type fictitious_timestep_size: float
        :return: _description_
        :rtype: Tuple[Array, Array]
        """
        residual_threshold = self.reinitialization_setup.residual_threshold
        if self.time_integrator.no_stages > 1:
            init = jnp.array(levelset, copy=True)
        for stage in range( self.time_integrator.no_stages ):
            rhs = self.compute_rhs(levelset, levelset_0, distance_L,
                                   distance_R, mask)
            residual_mask = jnp.where(jnp.abs(rhs) > residual_threshold, 1, 0)
            if stage > 0:
                levelset = self.time_integrator.prepare_buffer_for_integration(levelset, init, stage)
            levelset = self.time_integrator.integrate(levelset, rhs*residual_mask,
                                                      fictitious_timestep_size, stage)
            levelset = self.halo_manager.perform_halo_update_levelset(levelset, False, False)

        return levelset, rhs

    def compute_rhs(
            self,
            levelset: Array,
            levelset_0: Array,
            distance_L: Array,
            distance_R: Array,
            mask: Array,
            ) -> Tuple[Array, Array]:
        """Computes the right-hand-side of the
        levelset reinitialization equation.

        :param levelset: _description_
        :type levelset: Array
        :param levelset_0: _description_
        :type levelset_0: Array
        :param mask: _description_
        :type mask: Array
        :param distance_L: _description_, defaults to None
        :type distance_L: Array, optional
        :param distance_R: _description_, defaults to None
        :type distance_R: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """
        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices

        # DERIVATIVES
        derivatives_L = []
        derivatives_R = []
        for i, axis in enumerate(active_axes_indices):

            # STANDARD DERIVATIVES
            first_deriv_L = self.first_derivative_stencil.derivative_xi(levelset, cell_size, axis, 0)
            first_deriv_R = self.first_derivative_stencil.derivative_xi(levelset, cell_size, axis, 1)
            second_deriv = self.second_derivative_stencil.derivative_xi(levelset, cell_size, axis)
            second_deriv_L = self.minmod(second_deriv, jnp.roll(second_deriv, 1, axis))
            second_deriv_R = self.minmod(second_deriv, jnp.roll(second_deriv, -1, axis))
            der_L = first_deriv_L + cell_size/2.0 * second_deriv_L[self.s_1]
            der_R = first_deriv_R - cell_size/2.0 * second_deriv_R[self.s_1]

            # SUBCELL DERIVATIVES
            der_L_subcell = levelset[self.s_0]/distance_L[i] + distance_L[i]/2.0*second_deriv_L[self.s_1]
            der_R_subcell = -levelset[self.s_0]/distance_R[i] - distance_R[i]/2.0*second_deriv_R[self.s_1]

            # MASK
            mask_L = self.compute_directional_cut_cell_mask(levelset_0, axis, 0)
            mask_R = self.compute_directional_cut_cell_mask(levelset_0, axis, 1)
            der_L = der_L_subcell * mask_L + (1 - mask_L) * der_L
            der_R = der_R_subcell * mask_R + (1 - mask_R) * der_R

            derivatives_L.append(der_L)
            derivatives_R.append(der_R)

        # GODUNOV HAMILTONIAN
        sign = jnp.sign(levelset_0[self.s_0])
        godunov_hamiltonian = 0.0
        for der_L, der_R in zip(derivatives_L, derivatives_R):
            godunov_hamiltonian += jnp.maximum( jnp.maximum(0.0, sign * der_L)**2, jnp.minimum(0.0, sign * der_R)**2 )
        godunov_hamiltonian = jnp.sqrt(godunov_hamiltonian + self.eps)

        # RHS
        rhs = - sign * (godunov_hamiltonian - 1)
        rhs *= mask

        return rhs

    
    def minmod(
            self,
            array1: Array,
            array2: Array
            ) -> Array:
        """Minmod limiter

        :param array1: _description_
        :type array1: Array
        :param array2: _description_
        :type array2: Array
        :return: _description_
        :rtype: Array
        """
        mask = jnp.where(array1*array2 < 0.0, 0, 1)
        out = jnp.minimum(jnp.abs(array1), jnp.abs(array2))
        out *= mask
        return out


    def compute_distance_approximation(
            self,
            levelset: Array,
            ) -> Array:
        """Computes an approximation of the
        distance to the interface based on an 
        quadratic ENO polynomial.

        :param levelset: _description_
        :type levelset: Array
        :param axis: _description_
        :type axis: int
        :param direction: _description_
        :type direction: int
        :return: _description_
        :rtype: Array
        """

        active_axes_indices = self.domain_information.active_axes_indices
        dx = self.domain_information.smallest_cell_size

        distance_L = []
        distance_R = []

        for axis in active_axes_indices:
            
            s_2_L_xi = self.s_2[0][axis]
            s_2_R_xi = self.s_2[1][axis]

            undivided_difference = self.second_derivative_stencil.derivative_xi(levelset, 1.0, axis)
            undivided_difference_L = self.minmod(undivided_difference, jnp.roll(undivided_difference, 1, axis))
            undivided_difference_R = self.minmod(undivided_difference, jnp.roll(undivided_difference, -1, axis))

            discriminant_L = (undivided_difference_L[self.s_1]/2.0 - levelset[self.s_0] - levelset[s_2_L_xi])**2 - \
                4*levelset[self.s_0]*levelset[s_2_L_xi]
            discriminant_R = (undivided_difference_R[self.s_1]/2.0 - levelset[self.s_0] - levelset[s_2_R_xi])**2 - \
                4*levelset[self.s_0]*levelset[s_2_R_xi]

            distance_1_L = dx * (0.5 + (levelset[self.s_0] - levelset[s_2_L_xi] - 
                jnp.sign(levelset[self.s_0] - levelset[s_2_L_xi]) *
                jnp.sqrt(jnp.maximum(0.0,discriminant_L) + self.eps))/(undivided_difference_L[self.s_1] + self.eps)
                )
            distance_1_R = dx * (0.5 + (levelset[self.s_0] - levelset[s_2_R_xi] - 
                jnp.sign(levelset[self.s_0] - levelset[s_2_R_xi]) *
                jnp.sqrt(jnp.maximum(0.0,discriminant_R) + self.eps))/(undivided_difference_R[self.s_1] + self.eps)
                )

            distance_2_L = dx * jnp.abs(levelset[self.s_0])/(jnp.abs(levelset[self.s_0]) + jnp.abs(levelset[s_2_L_xi]) + self.eps)
            distance_2_R = dx * jnp.abs(levelset[self.s_0])/(jnp.abs(levelset[self.s_0]) + jnp.abs(levelset[s_2_R_xi]) + self.eps)
            
            mask_L = jnp.where(jnp.abs(undivided_difference_L[self.s_1]) > self.eps*1e5, 1, 0)
            mask_R = jnp.where(jnp.abs(undivided_difference_R[self.s_1]) > self.eps*1e5, 1, 0)

            distance_L_xi = distance_1_L * mask_L + (1 - mask_L) * distance_2_L
            distance_R_xi = distance_1_R * mask_R + (1 - mask_R) * distance_2_R

            distance_L.append(distance_L_xi)
            distance_R.append(distance_R_xi)

        return distance_L, distance_R

    def compute_directional_cut_cell_mask(
            self,
            levelset: Array,
            axis: int,
            direction: int
            ) -> Array:
        """Computes the directional cut cell mask,
        i.e., cells where the sign of the 
        levelset changes compared to the 
        neighbording cell in the specified
        axis and direction.

        :param levelset: _description_
        :type levelset: Array
        :param axis: _description_
        :type axis: int
        :param direction: _description_
        :type direction: int
        :return: _description_
        :rtype: Array
        """
        s_ = self.s_2[direction][axis]
        mask_cut_cells = jnp.where(levelset[self.s_0]*levelset[s_] < 0, 1, 0)
        return mask_cut_cells




