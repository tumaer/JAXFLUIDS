from abc import ABC, abstractmethod
from typing import Dict

import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.config import precision

class Mixture(ABC):
    """The Mixture class is the fundamental abstract class
    for material mixtures. Used for level-set,
    diffuse-interface methods, and homogenous
    mixtures.

    # TODO docstrings

    :param ABC: _description_
    :type ABC: _type_
    """

    def __init__(
            self,
            unit_handler: UnitHandler,
            material_properties: Dict
        ) -> None:
    
        self.eps = precision.get_eps()

        # MATERIALS DICT
        self.materials : Dict[str, Material] = {}

    @abstractmethod
    def get_thermal_conductivity(self) -> Array:
        pass

    @abstractmethod
    def get_dynamic_viscosity(self) -> Array:
        pass

    @abstractmethod
    def get_bulk_viscosity(self) -> Array:
        pass

    @abstractmethod
    def get_speed_of_sound(self) -> Array:
        pass

    @abstractmethod
    def get_pressure(self) -> Array:
        pass

    @abstractmethod
    def get_temperature(self) -> Array:
        pass

    @abstractmethod
    def get_specific_energy(self) -> Array:
        pass

    @abstractmethod
    def get_total_energy(self) -> Array:
        pass

    @abstractmethod
    def get_total_enthalpy(self) -> Array:
        pass

    @abstractmethod
    def get_psi(self) -> Array:
        pass

    @abstractmethod
    def get_grueneisen(self) -> Array:
        pass
