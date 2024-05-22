from typing import List, Union
import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup

class StiffenedGas(Material):
    """Implements the stiffened gas equation of state.
    p = (gamma - 1) * rho * e - gamma * p_b
    c = \sqrt(gamma * (p + p_b) / rho)
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
        self.cv = self.cp / self.gamma 
        self.pb = eos_setup.background_pressure
        self.molar_mass = unit_handler.universal_gas_constant_nondim / self.R

        self._set_transport_properties()

    def get_psi(self, p: Array, rho: Array) -> Array:
        return ( p + self.gamma * self.pb ) / rho

    def get_grueneisen(self, *args, **kwargs) -> Array:
        return self.gamma - 1.0

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        return jnp.sqrt( self.gamma * ( p + self.pb ) / rho )

    def get_pressure(self, e: Array, rho: Array) -> Array:
        return ( self.gamma - 1 ) * e * rho - self.gamma * self.pb
    
    def get_temperature(self, p: Array, rho: Array) -> Array:
        # TODO
        return ( p + self.pb ) / (rho * self.R)

    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        # Specific internal energy
        # e = (p + \gamma \pi_{inf}) / (\rho (\gamma - 1))
        return (p + self.gamma * self.pb) / (rho * (self.gamma - 1.0))
    
    def get_volume_specific_energy(self, p: Array) -> Array:
        # Volume specific internal energy
        # \rho e = (p + \gamma \pi_{inf}) / (\gamma - 1)
        return (p + self.gamma * self.pb) / (self.gamma - 1.0)

    def get_specific_enthalpy(self, p: Array, rho: Array) -> Array:
        # Specific internal enthalpy
        # h = e + p / rho = (\gamma (p + \pi_{inf})) / (\rho (\gamma - 1))
        return (p + self.pb) * self.gamma / (rho * (self.gamma - 1.0))
    
    def get_volume_specific_enthalpy(self, p: Array) -> Array:
        # Volume specific internal enthalpy
        # \rho h = \rho e + p = (\gamma (p + \pi_{inf})) / (\gamma - 1)
        return (p + self.pb) * self.gamma / (self.gamma - 1.0)

    def get_total_energy(self, p:Array, rho:Array, u:Array, v:Array, w:Array) -> Array:
        # Total energy per unit volume
        return (p + self.gamma * self.pb) / (self.gamma - 1) + 0.5 * rho * (u * u + v * v + w * w)

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
    
    def get_stagnation_temperature(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
        ) -> Array:
        """Computes the stagnation temperature

        :param rho: Density buffer
        :type rho: Array
        :return: Grueneisen
        :rtype: Array
        """
        raise NotImplementedError
