from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.halos.outer.levelset import BoundaryConditionLevelset
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.geometry_calculator import compute_cut_cell_mask
from jaxfluids.levelset.helper_functions import linear_filtering
from jaxfluids.config import precision
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup


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
        
        self.halo_manager = halo_manager

    @abstractmethod
    def perform_reinitialization(
            self,
            levelset: Array,
            CFL: float,
            steps: int,
            mask: Array = None
            ) -> Array:
        """Reinitializes the levelset buffer iteratively
        by solving the reinitialization equation to steady
        state. This is an abstract method. See child class 
        for implementation and key word arguments.
        

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Array
        """
        pass

    @abstractmethod
    def compute_residual(
            self,
            levelset: Array,
            mask: Array,
            ) -> Tuple[float, float]:
        """Computes the mean and maximum residual of the levelset
        reinitialization equation within the masked region.
        

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Array
        """
        pass