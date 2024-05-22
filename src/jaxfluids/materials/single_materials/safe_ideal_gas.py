from typing import List, Union
import types

import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup

class SafeIdealGas(Material):
    """Implements the safe ideal gas law, i.e., prevents
    division by zero and negative square roots"""

    def __init__(
            self,
            unit_handler: UnitHandler,
            material_setup: MaterialPropertiesSetup,
            **kwargs
        ) -> None:

        super().__init__(unit_handler, material_setup)

        self.gamma = material_setup.specific_heat_ratio
        self.R = unit_handler.non_dimensionalize(material_setup.specific_gas_constant, "specific_gas_constant")
        self.cp = self.gamma / (self.gamma - 1) * self.R
        self.pb = 0.0
        
    def get_psi(self, p, rho):
        """See base class. """
        return jnp.maximum( p, self.eps) / rho

    def get_grueneisen(self, rho):
        """See base class. """
        return self.gamma - 1

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return jnp.sqrt( self.gamma * jnp.maximum( p, self.eps) / jnp.maximum( rho, self.eps ) )

    def get_pressure(self, e: Array, rho: Array) -> Array:
        """See base class. """
        return (self.gamma - 1) * e * rho

    def get_temperature(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / ( rho * self.R + self.eps )
        
    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return jnp.maximum( p, self.eps) / jnp.maximum( rho, self.eps ) / (self.gamma - 1)

    def get_total_energy(self, p: Array, rho: Array, u: Array, v: Array, w: Array) -> Array:
        """See base class. """
        # Total energy per unit volume
        return jnp.maximum( p, self.eps) / (self.gamma - 1) + 0.5 * jnp.maximum( rho, self.eps) * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p: Array, rho: Array, u: Array, v: Array, w: Array) -> Array:
        """See base class. """
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, u, v, w) + jnp.maximum( p, self.eps)) / jnp.maximum( rho, self.eps)

    def get_specific_heat_capacity(self, T: Array) -> Union[float, Array]:
        """Calculates the specific heat coefficient per unit mass.
        [c_p] = J / kg / K

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        return self.cp
