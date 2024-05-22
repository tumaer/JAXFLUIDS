from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.time_integration.euler import Euler
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.levelset.geometry_calculator import compute_cut_cell_mask
from jaxfluids.halos.halo_manager import HaloManager
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

class RussoReinitializer(PDEBasedReinitializer):
    """First order reinitializer with subcell fix
    according to \cite Russo2000
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            reinitialization_setup: LevelsetReinitializationSetup,
            halo_cells: int,
            narrowband_setup: NarrowBandSetup,
            ) -> None:

        super(RussoReinitializer, self).__init__(
            domain_information, halo_manager,
            reinitialization_setup, narrowband_setup)

        is_halos = reinitialization_setup.is_halos
        self.is_jaxforloop = reinitialization_setup.is_jaxforloop

        self.time_integrator = Euler(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = halo_cells if is_halos else 0)

        self.derivative_stencil = FirstDerivativeFirstOrderCenter(
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

        self.s_1 = [[  
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
            fixed_timestep: float = None,
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
        is_halos = self.reinitialization_setup.is_halos
        is_jaxwhileloop = self.reinitialization_setup.is_jaxwhileloop

        smallest_cell_size = self.domain_information.smallest_cell_size
        nh_conservatives = self.domain_information.nh_conservatives 
        nh_offset = self.domain_information.nh_offset
        nh_offset = nh_offset if is_halos else nh_conservatives

        levelset_0 = jnp.array(levelset, copy=True)
        fictitious_timestep_size = CFL * smallest_cell_size
        if fixed_timestep != None:
            fictitious_timestep_size = fixed_timestep

        if mask == None:
            mask = jnp.ones_like(levelset[self.s_0], dtype=jnp.uint32)

        mask_cut_cells = compute_cut_cell_mask(
            levelset_0, nh_offset)
        distance = self.compute_distance_approximation(levelset)

        # REINITIALIZATION STEPS
        if is_jaxforloop:
            def _body_func(index, levelset: Array) -> Array:
                if debug:
                    levelset_in = levelset[index]
                else:
                    levelset_in = levelset
                levelset_out, _ = self.do_integration_step(
                    levelset_in, levelset_0, mask, fictitious_timestep_size,
                    distance, mask_cut_cells)
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
        nh_offset = self.domain_information.nh_offset
        nh_conservatives = self.domain_information.nh_conservatives
        is_halos = self.reinitialization_setup.is_halos 
        nh_offset = nh_offset if is_halos else nh_conservatives
        mask_cut_cells = compute_cut_cell_mask(levelset, nh_offset)
        distance = self.compute_distance_approximation(levelset)
        rhs = self.compute_rhs(levelset, levelset, mask, distance, mask_cut_cells)
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
            distance: Array,
            mask_cut_cells: Array
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
            rhs = self.compute_rhs(
                levelset, levelset_0, mask, distance, mask_cut_cells)
            residual_mask = jnp.where(jnp.abs(rhs) > residual_threshold, 1, 0)
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
            distance: Array,
            mask_cut_cells: Array
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
            derivatives_L.append( self.derivative_stencil.derivative_xi(levelset, cell_size, axis, 0, levelset_0, distance) )
            derivatives_R.append( self.derivative_stencil.derivative_xi(levelset, cell_size, axis, 1, levelset_0, distance) )

        # GODUNOV HAMILTONIAN
        sign = jnp.sign(levelset_0[self.s_0])
        godunov_hamiltonian = 0.0
        for der_L, der_R in zip(derivatives_L, derivatives_R):
            godunov_hamiltonian += jnp.maximum( jnp.maximum(0.0, sign * der_L)**2, jnp.minimum(0.0, sign * der_R)**2 )
        godunov_hamiltonian = jnp.sqrt(godunov_hamiltonian + self.eps)

        # RHS
        rhs_godunov = - sign * (godunov_hamiltonian - 1)
        rhs_subcell = - 1.0/smallest_cell_size * (sign * jnp.abs(levelset[self.s_0]) - distance)
        rhs = rhs_subcell * mask_cut_cells + \
            (1 - mask_cut_cells) * rhs_godunov
        rhs *= mask

        return rhs


    def compute_distance_approximation(
            self,
            levelset: Array,
            ) -> Array:
        """Distance approximation from
        the levelset field.

        :param levelset: _description_
        :type levelset: Array
        :param nh_offset: _description_
        :type nh_offset: int
        :param smallest_cell_size: _description_
        :type smallest_cell_size: float
        :return: _description_
        :rtype: Array
        """

        smallest_cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices

        denominator_1 = 0.0
        denominator_2 = 0.0
        denominator_3 = 0.0
        for axis in active_axes_indices:
            s_L = self.s_1[0][axis]
            s_R = self.s_1[1][axis]
            denominator_1 += jnp.square(levelset[s_R] - levelset[s_L])
            denominator_2 += jnp.square(levelset[s_R] - levelset[self.s_0])
            denominator_3 += jnp.square(levelset[self.s_0] - levelset[s_L])
        denominator_1 = jnp.sqrt(denominator_1 + self.eps)/2.0
        denominator_2 = jnp.sqrt(denominator_2 + self.eps)
        denominator_3 = jnp.sqrt(denominator_3 + self.eps)
        denominator = jnp.maximum(denominator_1, denominator_2)
        denominator = jnp.maximum(denominator, denominator_3)
        denominator = jnp.maximum(denominator, self.eps)

        distance = smallest_cell_size * levelset[self.s_0]/denominator

        return distance

