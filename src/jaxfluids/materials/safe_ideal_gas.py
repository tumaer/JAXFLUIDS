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

class SafeIdealGas(Material):
    """Implements the safe ideal gas law, i.e., prevents division by zero and negative square roots"""
    def __init__(self, unit_handler: UnitHandler, dynamic_viscosity: Union[float, str, types.LambdaType], sutherland_parameters: List,
        bulk_viscosity: float, thermal_conductivity: Union[float, str, types.LambdaType], prandtl_number: float,
        specific_heat_ratio: float, specific_gas_constant: float, **kwargs) -> None:

        super().__init__(unit_handler, dynamic_viscosity, sutherland_parameters, bulk_viscosity, thermal_conductivity, prandtl_number)

        self.gamma      = specific_heat_ratio
        self.R          = unit_handler.non_dimensionalize(specific_gas_constant, "specific_gas_constant")
        self.cp         = self.gamma / (self.gamma - 1) * self.R

    def get_psi(self, p, rho):
        """See base class. """
        return jnp.maximum( p, self.eps) / rho

    def get_grueneisen(self, rho):
        """See base class. """
        return self.gamma - 1

    def get_speed_of_sound(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """See base class. """
        return jnp.sqrt( self.gamma * jnp.maximum( p, self.eps) / jnp.maximum( rho, self.eps ) )

    def get_pressure(self, e: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """See base class. """
        return (self.gamma - 1) * e * rho

    def get_temperature(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """See base class. """
        return p / ( rho * self.R + self.eps )
        
    def get_energy(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """See base class. """
        return jnp.maximum( p, self.eps) / jnp.maximum( rho, self.eps ) / (self.gamma - 1)

    def get_total_energy(self, p: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """See base class. """
        # Total energy per unit volume
        return jnp.maximum( p, self.eps) / (self.gamma - 1) + 0.5 * jnp.maximum( rho, self.eps) * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """See base class. """
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, u, v, w) + jnp.maximum( p, self.eps)) / jnp.maximum( rho, self.eps)