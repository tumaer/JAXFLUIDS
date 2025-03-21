from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING, Dict

import jax
import jax.numpy as jnp

from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.levelset.geometry.mask_functions import compute_narrowband_mask
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

Array = jax.Array

class PDEBasedReinitializer(LevelsetReinitializer):
    """Abstract class for levelset reinitialization.
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            reinitialization_setup: LevelsetReinitializationSetup,
            narrowband_setup: NarrowBandSetup,
            ) -> None:
        
        super(PDEBasedReinitializer, self).__init__(
            domain_information, halo_manager.boundary_condition_levelset,
            reinitialization_setup, narrowband_setup)
        
        self.time_integrator: TimeIntegrator = None 
        self.halo_manager = halo_manager
        self.narrowband_width = narrowband_setup.computation_width + 1 # NOTE we compute residual in one further cell
        self.cell_size = domain_information.smallest_cell_size


    def perform_reinitialization(
            self,
            levelset: Array,
            CFL: float,
            steps: int,
            mask: Array = None,
            fixed_timestep = None,
            debug: bool = False,
            ) -> Tuple[Array, LevelsetProcedureInformation]:
        """Reinitializes the levelset buffer iteratively
        by solving the reinitialization equation to steady
        state. This is an abstract method. See child class 
        for implementation and key word arguments.
        

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Array
        """

        is_jaxwhileloop = self.reinitialization_setup.is_jaxwhileloop
        is_parallel = self.domain_information.is_parallel
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        residual_threshold = self.reinitialization_setup.residual_threshold

        levelset_0 = levelset

        if fixed_timestep == None:
            timestep_size = CFL * self.cell_size
        else:
            timestep_size = fixed_timestep

        # NOTE mask indicating where to reinitialize, usually whole domain
        if mask is None:
            mask = jnp.ones_like(levelset[nhx,nhy,nhz], dtype=jnp.uint32)

        kwargs = self.get_kwargs(levelset)

        if not is_jaxwhileloop:
            def _body_func(
                    index,
                    args: Tuple[Array, float]
                    ) -> Tuple[Array, float]:

                levelset, max_residual = args

                if debug:
                    levelset_in = levelset[index]
                else:
                    levelset_in = levelset

                levelset_out, max_residual_out = self.do_integration_step(
                    levelset_in, levelset_0, mask, timestep_size, **kwargs)

                if debug:
                    levelset = levelset.at[index+1].set(levelset_out)
                    max_residual = max_residual.at[index+1].set(max_residual_out)
                else:
                    levelset = levelset_out
                    max_residual = max_residual_out

                return levelset, max_residual

            if debug:
                levelset_buffer = jnp.zeros((steps+1,)+levelset.shape)
                levelset = levelset_buffer.at[0].set(levelset)
                residual = jnp.zeros(steps+1)
            else:
                residual = 1e10

            args = (levelset, residual)
            args = jax.lax.fori_loop(0, steps, _body_func, args)
            levelset = args[0]
            residual = args[1]
            step_count = steps

        else:
            def _body_func(
                    args: Tuple[Array, int, float]
                    ) -> Tuple[Array, int, float]:
                levelset, index, max_residual = args

                if debug:
                    levelset_in = levelset[index]
                else:
                    levelset_in = levelset

                levelset_out, max_residual = self.do_integration_step(
                    levelset_in, levelset_0, mask, timestep_size, **kwargs)
                
                if debug:
                    levelset = levelset.at[index+1].set(levelset_out)
                else:
                    levelset = levelset_out

                args = (levelset, index+1, max_residual)

                return args

            def _cond_fun(args: Tuple[Array, int, float]) -> bool:
                _, index, max_residual = args
                condition1 = max_residual > residual_threshold
                condition2 = index < steps
                return jnp.logical_and(condition1, condition2)
            
            if debug:
                levelset_buffer = jnp.zeros((steps+1,)+levelset.shape)
                levelset = levelset_buffer.at[0].set(levelset)

            args = (levelset, 0, 1e10) # NOTE initial value for max residual for while condition is hard coded to 1e10
            args = jax.lax.while_loop(_cond_fun, _body_func, args)

            levelset = args[0]
            step_count = args[1]
            residual = args[2]

        info = LevelsetProcedureInformation(
            step_count, residual, None)
        
        return levelset, info


    def do_integration_step(
            self,
            levelset: Array,
            levelset_0: Array,
            mask: Array,
            timestep_size: float,
            **kwargs: Dict[str, Array]
            ) -> Tuple[Array, float]:
        """_summary_

        :param levelset: _description_
        :type levelset: Array
        :param levelset_0: _description_
        :type levelset_0: Array
        :param mask: _description_
        :type mask: Array
        :param timestep_size: _description_
        :type timestep_size: float
        :return: _description_
        :rtype: Tuple[Array, float]
        """

        residual_threshold = self.reinitialization_setup.residual_threshold
        if self.time_integrator.no_stages > 1:
            init = jnp.array(levelset, copy=True)
        for stage in range( self.time_integrator.no_stages ):
            rhs = self.compute_rhs(levelset, levelset_0, mask, **kwargs)
            residual_mask = jnp.where(jnp.abs(rhs) > residual_threshold, 1, 0) # NOTE we only reinitialize cells where the residual threshold is exceeded
            if stage > 0:
                levelset = self.time_integrator.prepare_buffer_for_integration(levelset, init, stage)
            levelset = self.time_integrator.integrate(levelset, rhs*residual_mask, timestep_size, stage)
            levelset = self.halo_manager.perform_halo_update_levelset(levelset, False, False)

        max_residual = self.compute_residual_from_rhs(levelset, rhs)

        return levelset, max_residual

    def compute_residual_from_rhs(
            self,
            levelset: Array,
            rhs: Array
            ):
        is_parallel = self.domain_information.is_parallel
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        mask_narrowband = compute_narrowband_mask(
            levelset, self.cell_size, self.narrowband_width) 
        mask_narrowband = mask_narrowband[nhx,nhy,nhz]
        residual = jnp.abs(rhs*mask_narrowband)
        max_residual = jnp.max(residual, axis=(-3,-2,-1))
        if is_parallel:
            max_residual = jax.lax.pmax(max_residual, axis_name="i")
        return max_residual

    def compute_residual(
            self,
            levelset: Array,
            mask: Array = None,
            ) -> float:

        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        if mask is None:
            mask = jnp.ones_like(levelset[nhx,nhy,nhz], dtype=jnp.uint32)
        kwargs = self.get_kwargs(levelset)
        rhs = self.compute_rhs(levelset, levelset, mask, **kwargs)
        max_residual = self.compute_residual_from_rhs(levelset, rhs)

        return max_residual

    @abstractmethod
    def get_kwargs(self, levelset: Array, CFL: float) -> Dict[str, Array]:
        pass

    @abstractmethod
    def compute_rhs(self, *args) -> Array:
        pass