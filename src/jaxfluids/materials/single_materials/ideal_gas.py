from typing import List, Union
import types

import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup

class IdealGas(Material):
    """Implements the ideal gas law.
    """
    def __init__(
            self,
            unit_handler: UnitHandler,
            material_setup: MaterialPropertiesSetup,
            **kwargs
            ) -> None:

        super().__init__(unit_handler, material_setup)

        self.R_universal = unit_handler.universal_gas_constant_nondim

        eos_setup = material_setup.eos.ideal_gas_setup
        self.gamma = eos_setup.specific_heat_ratio
        self.R = eos_setup.specific_gas_constant
        self.molar_mass = self.R_universal / self.R
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        self.cp_molar = self.cp / self.molar_mass
        self.pb = 0.0

        self._set_transport_properties()

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

    def get_specific_heat_ratio(self, T: Array) -> Union[float, Array]:
        """Calculates the specific heat ratio.

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        return self.gamma

    def get_psi(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / rho

    def get_grueneisen(self, rho: Array, T: Array = None) -> Array:
        """See base class. """
        return self.gamma - 1

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return jnp.sqrt(self.gamma * p / rho)

    def get_pressure(self, e: Array, rho: Array) -> Array:
        """See base class. """
        return (self.gamma - 1.0) * e * rho

    def get_temperature(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / (rho * self.R)

    def get_specific_energy(self, p:Array, rho:Array) -> Array:
        """See base class. """
        # Specific internal energy
        return p / (rho * (self.gamma - 1.0))

    def get_total_energy(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
            ) -> Array:
        """See base class. """
        # Total energy per unit volume
        # (sensible, i.e., without heat of formation)
        return p / (self.gamma - 1) + 0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
            ) -> Array:
        """See base class. """
        # Total specific enthalpy
        # (sensible, i.e., without heat of formation)
        return (self.get_total_energy(p, rho, u, v, w) + p) / rho

    def get_stagnation_temperature(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
        ) -> Array:
        T = self.get_temperature(p, rho)
        cp = self.get_specific_heat_capacity(T)
        return T + 0.5 * (u * u + v * v + w * w) / cp
