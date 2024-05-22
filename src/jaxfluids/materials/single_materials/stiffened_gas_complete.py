from typing import List, Union
import types

import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup

class StiffenedGasComplete(Material):
    """Implements the stiffened gas equation of state
    with energy correction.
    p + gamma * p_b = (gamma - 1) * rho * (e + e_b)
    RT = (p + p_b)/rho + (gamma - 1) * e_t * rho^(gamma-1)
    """
    def __init__(
            self,
            unit_handler: UnitHandler,
            material_setup: MaterialPropertiesSetup,
            **kwargs
            ) -> None:
        
        super().__init__(unit_handler, material_setup)

        eos_setup = material_setup.eos.stiffened_gas_setup
        self.gamma = eos_setup.specific_heat_ratio
        self.R = eos_setup.specific_gas_constant
        self.cp = self.gamma / (self.gamma - 1) * self.R
        self.pb = eos_setup.material_setup.background_pressure
        self.eb = eos_setup.material_setup.energy_translation_factor
        self.et = eos_setup.material_setup.thermal_energy_factor

    def get_psi(self, p: Array, rho: Array) -> Array:
        return ( p + self.gamma * self.pb ) / rho

    def get_grueneisen(self, rho: Array) -> Array:
        return self.gamma - 1

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        return jnp.sqrt( self.gamma * ( p + self.pb ) / rho )

    def get_pressure(self, e: Array, rho: Array) -> Array:
        return ( self.gamma - 1 ) * (e + self.eb) * rho - self.gamma * self.pb
    
    def get_temperature(self, p: Array, rho: Array) -> Array:
        return ( p + self.pb ) / ( rho * self.R )  + (self.gamma - 1.0) * self.et * rho**(self.gamma - 1.0) / self.R

    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        # Specific internal energy
        return ( p + self.gamma * self.pb ) / ( rho * (self.gamma - 1) ) - self.eb

    def get_total_energy(self, p:Array, rho:Array, u:Array, v:Array, w:Array) -> Array:
        # Total energy per unit volume
        return self.get_specific_energy(p, rho) * rho + 0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p:Array, rho:Array, u:Array, v:Array, w:Array) -> Array:
        # Total specific enthalpy
        return ( self.get_total_energy(p, rho, u, v, w) + p ) / rho
    
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
