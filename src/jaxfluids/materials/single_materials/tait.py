from typing import List, Union
import types

import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup

class Tait(Material):
    """Implements the tait equation of state."""
    def __init__(
            self,
            unit_handler: UnitHandler,
            material_setup: MaterialPropertiesSetup,
            **kwargs
        ) -> None:
        
        super().__init__(unit_handler, material_setup)

        tait_setup = material_setup.eos.tait_setup
        self.N = tait_setup.N
        self.B = tait_setup.B
        self.rho_ref = tait_setup.rho_ref
        self.p_ref = tait_setup.p_ref

    def get_specific_heat_capacity(self, T: Array) -> Array:
        # TODO aaron/deniz
        raise NotImplementedError

    def get_psi(self, p: Array, rho: Array) -> Array:
        return self.N * (p + self.B -self.p_ref) / rho

    def get_grueneisen(self, rho: Array) -> Array:
        return 0.0

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        return jnp.sqrt( self.N * (p + self.B) / rho)

    def get_pressure(self, e: Array, rho: Array) -> Array:
        return self.B * (jnp.power((rho / self.rho_ref), self.N) - 1.0) + self.p_ref
    
    def get_temperature(self, p: Array, rho: Array) -> Array:
        # Temperature is not defined for Tait.
        return jnp.zeros_like(p)

    def get_density(self, p: Array) -> Array:
        return self.rho_ref * jnp.power((p - self.p_ref) / self.B + 1.0, 1.0 / self.N)

    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        # Specific internal energy
        # TODO where is this coming from?
        return (p + self.B - self.p_ref) /(self.N * rho) + (self.B - self.p_ref) / rho

    def get_total_energy(self, p:Array, rho:Array, u:Array, v:Array, w:Array) -> Array:
        # Total energy per unit volume
        return ((p + self.B - self.p_ref) / self.N  + self.B - self.p_ref) + \
            0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p:Array, rho:Array, u:Array, v:Array, w:Array) -> Array:
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, u, v, w) + p) / rho
