#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from typing import List, Union
import types

import jax.numpy as jnp

from jaxfluids.materials.material import Material
from jaxfluids.unit_handler import UnitHandler

class StiffenedGas(Material):
    """Implements the stiffened gas equation of state."""
    def __init__(self, unit_handler: UnitHandler, dynamic_viscosity: Union[float, str, types.LambdaType], sutherland_parameters: List,
        bulk_viscosity: float, thermal_conductivity: Union[float, str, types.LambdaType], prandtl_number: float,
        specific_heat_ratio: float, specific_gas_constant: float, background_pressure: float, **kwargs) -> None:
        
        super().__init__(unit_handler, dynamic_viscosity, sutherland_parameters, bulk_viscosity, thermal_conductivity, prandtl_number)

        self.gamma      = specific_heat_ratio
        self.R          = unit_handler.non_dimensionalize(specific_gas_constant, "specific_gas_constant")
        self.cp         = self.gamma / (self.gamma - 1) * self.R
        self.pb         = unit_handler.non_dimensionalize(background_pressure, "pressure")

    def get_psi(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return ( p + self.gamma * self.pb ) / rho

    def get_grueneisen(self, rho: jnp.ndarray) -> jnp.ndarray:
        return self.gamma - 1

    def get_speed_of_sound(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt( self.gamma * ( p + self.pb ) / ( rho + self.eps ) )

    def get_pressure(self, e: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return ( self.gamma - 1 ) * e * rho - self.gamma * self.pb
    
    def get_temperature(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return ( p + self.pb ) / ( rho * self.R ) 

    def get_energy(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        # Specific internal energy
        return ( p + self.gamma * self.pb ) / ( ( rho + self.eps ) * (self.gamma - 1) )

    def get_total_energy(self, p:jnp.ndarray, rho:jnp.ndarray, u:jnp.ndarray, v:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        # Total energy per unit volume
        return ( p + self.gamma * self.pb ) / (self.gamma - 1) + 0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p:jnp.ndarray, rho:jnp.ndarray, u:jnp.ndarray, v:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        # Total specific enthalpy
        return ( self.get_total_energy(p, rho, u, v, w) + p ) / rho