from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager
from jaxfluids.config import precision
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup import LevelsetSetup

class ConservativeMixer(ABC):
    """Parent class for mixing methods.

    :param ABC: _description_
    :type ABC: _type_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            equation_information: EquationInformation,
            levelset_setup: LevelsetSetup,
            halo_manager: HaloManager,
            material_manager: MaterialManager,
            equation_manager: EquationManager
            ) -> None:

        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.equation_information = equation_information
        self.levelset_setup = levelset_setup
        self.halo_manager = halo_manager
        self.material_manager = material_manager
        self.equation_manager = equation_manager

    @abstractmethod
    def perform_mixing(
            self,
            conservatives: Array,
            *args
            ) -> Tuple[Array, Array,
            LevelsetPositivityInformation]:
        """Performs a mixing procedure on the
        integrated conservatives buffer treating
        vanished and newly created cells.
        Returns the volume-averaged conservatives 
        which can be used to compute the integrated
        primitives. Implementation in child
        class.

        :param conservatives: _description_
        :type conservatives: Array
        :return: _description_
        :rtype: Tuple
        """