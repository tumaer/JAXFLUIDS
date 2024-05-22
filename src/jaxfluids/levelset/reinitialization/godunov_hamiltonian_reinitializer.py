from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.halos.halo_manager import HaloManager
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

class GodunovHamiltonianReinitializer(PDEBasedReinitializer):
    """Solves the reinitialization equation using the 
    monotone Godunov Hamiltonian \cite Bardi1991 according to 
    \cite Sussman1994. Temporal and spatial
    discretization is user specified.

    :param LevelsetReinitializer: _description_
    :type LevelsetReinitializer: _type_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            reinitialization_setup: LevelsetReinitializationSetup,
            halo_cells: int,
            narrowband_setup: NarrowBandSetup,
            ) -> None:

        super(GodunovHamiltonianReinitializer, self).__init__(
            domain_information, halo_manager,
            reinitialization_setup, narrowband_setup)

        self.halo_manager = halo_manager

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
        self.s_0 = np.s_[...,nhx,nhy,nhz]

        time_integrator = reinitialization_setup.time_integrator
        derivative_stencil = reinitialization_setup.spatial_stencil
        is_halos = reinitialization_setup.is_halos
        self.time_integrator: TimeIntegrator = time_integrator(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = halo_cells if is_halos else 0)
        self.derivative_stencil: SpatialDerivative = derivative_stencil(
            nh = self.domain_information.nh_conservatives,
            inactive_axes = self.domain_information.inactive_axes,
            offset = halo_cells if is_halos else 0)

    def perform_reinitialization(
            self,
            levelset: Array,
            CFL: float,
            steps: int,
            mask: Array = None,
            fixed_timestep = None,
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

        smallest_cell_size = self.domain_information.smallest_cell_size
        is_jaxforloop = self.reinitialization_setup.is_jaxforloop
        is_jaxwhileloop = self.reinitialization_setup.is_jaxwhileloop
        is_parallel = self.domain_information.is_parallel
        narrowband_computation = self.narrowband_setup.computation_width
        residual_threshold = self.reinitialization_setup.residual_threshold

        levelset_0 = jnp.array(levelset, copy=True)
        if fixed_timestep == None:
            fictitious_timestep_size = CFL * smallest_cell_size
        else:
            fictitious_timestep_size = fixed_timestep

        # NOTE mask indicating where to reinitialize, usually whole domain
        if mask == None:
            mask = jnp.ones_like(levelset[self.s_0], dtype=jnp.uint32)

        if is_jaxforloop:
            def _body_func(index, levelset: Array) -> Array:
                if debug:
                    levelset_in = levelset[index]
                else:
                    levelset_in = levelset
                levelset_out, _ = self.do_integration_step(levelset_in, levelset_0,
                                                           mask, fictitious_timestep_size)
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
            def _body_func(args: Tuple[Array, int, float]) -> Tuple[Array, int, float]:
                levelset, index, max_residual = args
                if debug:
                    levelset_in = levelset[index]
                else:
                    levelset_in = levelset
                levelset_out, rhs = self.do_integration_step(levelset_in, levelset_0,
                                                        mask, fictitious_timestep_size)
                normalized_levelset = jnp.abs(levelset_out[self.s_0])/smallest_cell_size
                mask_narrowband = jnp.where(normalized_levelset <= narrowband_computation, 1, 0)
                max_residual = jnp.max(jnp.abs(rhs)*mask_narrowband, axis=(-3,-2,-1)) # NOTE max residual in narrowband is used for condition
                if is_parallel:
                    max_residual = jax.lax.pmax(max_residual, axis_name="i")
                if debug:
                    levelset = levelset.at[index+1].set(levelset_out)
                else:
                    levelset = levelset_out
                args = (levelset, index+1, max_residual)
                return args

            def _cond_fun(args: Tuple[int, Array, float]) -> bool:
                _, index, max_residual = args
                condition1 = max_residual > residual_threshold
                condition2 = index < steps
                return jnp.logical_and(condition1, condition2)
            
            if debug:
                levelset_buffer = jnp.zeros((steps+1,)+levelset.shape)
                levelset = levelset_buffer.at[0].set(levelset)
            args = (levelset, 0, 1e10) # NOTE initial value for max residual for while condition is hard coded to 1e10
            args = jax.lax.while_loop(_cond_fun, _body_func, args)
            step_count = args[1]
            levelset = args[0]

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
        rhs = self.compute_rhs(levelset, levelset, mask)
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

    @partial(jax.jit, static_argnums=(0))
    def do_integration_step(
            self,
            levelset: Array,
            levelset_0: Array,
            mask: Array,
            fictitious_timestep_size: float,
            ) -> Tuple[Array, Array]:
        """Performs an integration step of the levelset
        reinitialization equation.

        :param levelset: Levelset buffer
        :type levelset: Array
        :param levelset_0: Levelset buffer at fictitious time = 0.0
        :type levelset_0: Array
        :param mask: Mask for right-hand-side
        :type mask: Array
        :param fictitious_timestep_size: Timestep size
        :type fictitious_timestep_size: float
        :return: Tuple containing integrated levelset buffer and signed distance residual
        :rtype: Tuple[Array, Array]
        """
        residual_threshold = self.reinitialization_setup.residual_threshold
        if self.time_integrator.no_stages > 1:
            init = jnp.array(levelset, copy=True)
        for stage in range( self.time_integrator.no_stages ):
            rhs = self.compute_rhs(levelset, levelset_0, mask)
            residual_mask = jnp.where(jnp.abs(rhs) > residual_threshold, 1, 0) # NOTE we only reinitialize cells where the residual threshold is exceeded
            if stage > 0:
                levelset = self.time_integrator.prepare_buffer_for_integration(levelset, init, stage)
            levelset = self.time_integrator.integrate(levelset, rhs*residual_mask, fictitious_timestep_size, stage)
            levelset = self.halo_manager.perform_halo_update_levelset(levelset, False, False)

        return levelset, rhs

    def compute_rhs(
            self,
            levelset: Array,
            levelset_0: Array,
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
        :param distance: _description_, defaults to None
        :type distance: Array, optional
        :param mask_cut_cells: _description_, defaults to None
        :type mask_cut_cells: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices
        smallest_cell_size = self.domain_information.smallest_cell_size

        # DERIVATIVES
        derivatives_L = []
        derivatives_R = []
        for axis in active_axes_indices:
            derivatives_L.append( self.derivative_stencil.derivative_xi(levelset, cell_size, axis, 0) )
            derivatives_R.append( self.derivative_stencil.derivative_xi(levelset, cell_size, axis, 1) )

        levelset_0 = levelset_0[self.s_0]

        # GODUNOV HAMILTONIAN
        sign = jnp.sign(levelset_0)
        godunov_hamiltonian = 0.0
        for der_L, der_R in zip(derivatives_L, derivatives_R):
            godunov_hamiltonian += jnp.maximum( jnp.maximum(0.0, sign * der_L)**2, jnp.minimum(0.0, sign * der_R)**2 )
        godunov_hamiltonian = jnp.sqrt(godunov_hamiltonian + 1e-10)

        # RHS
        smooth_sign = levelset_0/jnp.sqrt(levelset_0**2 + smallest_cell_size**2)
        rhs = -smooth_sign * (godunov_hamiltonian - 1.0)
        rhs *= mask

        return rhs