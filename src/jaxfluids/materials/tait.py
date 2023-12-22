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

class Tait(Material):
    """Implements the tait equation of state."""
    def __init__(self, unit_handler: UnitHandler, dynamic_viscosity: Union[float, str, types.LambdaType], sutherland_parameters: List,
        bulk_viscosity: float, thermal_conductivity: Union[float, str, types.LambdaType], prandtl_number: float,
        specific_gas_constant: float, specific_heat_ratio: float = 7.15,  
        A_param: float = 1.00e+5, B_param: float = 3.31e+8, rho_0: float = 1.00e+3, **kwargs) -> None:
        
        super().__init__(unit_handler, dynamic_viscosity, sutherland_parameters, bulk_viscosity, thermal_conductivity, prandtl_number)

        self.gamma      = specific_heat_ratio

        self.A_param    = unit_handler.non_dimensionalize(A_param, "pressure")
        self.B_param    = unit_handler.non_dimensionalize(B_param, "pressure")
        self.rho_0      = unit_handler.non_dimensionalize(rho_0, "density")

    def get_psi(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return self.gamma * ( p + self.B_param -self.A_param ) / rho

    def get_grueneisen(self, rho: jnp.ndarray) -> jnp.ndarray:
        return 0.0

    def get_speed_of_sound(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt( self.gamma * ( p + self.B_param - self.A_param ) / ( rho + self.eps ) )

    def get_pressure(self, e: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        return self.A_param - self.B_param + self.B_param * (rho / self.rho_0)**self.gamma
    
    def get_temperature(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        # Temperature is not defined for Tait.
        return jnp.zeros_like(p)

    def get_energy(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        # Specific internal energy
        return ( p + self.B_param - self.A_param ) /( self.gamma * rho) + ( self.B_param - self.A_param ) / rho

    def get_total_energy(self, p:jnp.ndarray, rho:jnp.ndarray, u:jnp.ndarray, v:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        # Total energy per unit volume
        return ( ( p + self.B_param - self.A_param ) / self.gamma  + self.B_param - self.A_param ) + 0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p:jnp.ndarray, rho:jnp.ndarray, u:jnp.ndarray, v:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, u, v, w) + p) / rho
