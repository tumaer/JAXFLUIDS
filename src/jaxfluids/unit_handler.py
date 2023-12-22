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

from functools import partial
from typing import Dict, Union

import jax
import jax.numpy as jnp
import numpy as np

class UnitHandler:
    """The UnitHandler class implements functionaly to solve the NSE in non-dimensional form.
    """
    def __init__(self, density_reference: float, length_reference: float, velocity_reference: float, 
        temperature_reference: float) -> None:

        self.density_reference      = density_reference
        self.length_reference       = length_reference
        self.velocity_reference     = velocity_reference
        self.temperature_reference  = temperature_reference

        self.time_reference                         = length_reference / velocity_reference
        self.pressure_reference                     = density_reference * velocity_reference**2
        self.viscosity_reference                    = density_reference * velocity_reference * length_reference
        self.thermal_conductivity_reference         = density_reference * velocity_reference**3 * length_reference / temperature_reference
        self.gravity_reference                      = velocity_reference**2 / length_reference
        self.specific_gas_constant_reference        = velocity_reference**2 / temperature_reference
        self.mass_reference                         = density_reference * length_reference**3
        self.mass_flow_reference                    = self.mass_reference / self.time_reference
        self.surface_tension_coefficient_reference  = density_reference * velocity_reference * velocity_reference * length_reference

    def non_dimensionalize_domain_size(self, domain_size: Dict) -> Dict:
        domain_size_nondim = {}
        for axis in domain_size:
            domain_size_nondim[axis] = [
                self.non_dimensionalize(domain_size[axis][0], "length"),
                self.non_dimensionalize(domain_size[axis][1], "length")
                ]
        return domain_size_nondim

    @partial(jax.jit, static_argnums=(0, 2))
    def non_dimensionalize(self, value: Union[jnp.ndarray, float], quantity: str) -> Union[jnp.ndarray, float]:
        """Non-dimensionalizes the given buffer w.r.t. the specified quantity.

        :param value: Dimensional quantity buffer
        :type value: Union[jnp.ndarray, float]
        :param quantity: Quantity name
        :type quantity: str
        :return: Non-dimensional quantity buffer
        :rtype: Union[jnp.ndarray, float]
        """
        
        # NAME CONVERSION
        if quantity == "rho":
            quantity = "density"
        if quantity in ["u", "v", "w", "velocityX", "velocityY", "velocityZ"]:
            quantity = "velocity"
        if quantity in ["momentumX", "momentumY", "momentumZ"]:
            quantity = "momentum"
        if quantity == "p":
            quantity = "pressure"
        if quantity == "T":
            quantity = "temperature"

        # PRIMES
        if quantity == "density":
            value /= self.density_reference
        elif quantity == "velocity":
            value /= self.velocity_reference
        elif quantity == "temperature":
            value /= self.temperature_reference
        elif quantity == "pressure":
            value /= self.pressure_reference

        # CONS
        elif quantity == "mass":
            value /= self.density_reference
        elif quantity == "momentum":
            value /= (self.density_reference * self.velocity_reference)
        elif quantity == "energy":
            value /= self.pressure_reference

        # MATERIAL PARAMETERS
        elif quantity == "dynamic_viscosity":
            value /= self.viscosity_reference
        elif quantity == "thermal_conductivity":
            value /= self.thermal_conductivity_reference
        elif quantity == "specific_gas_constant":
            value /= self.specific_gas_constant_reference
        elif quantity == "surface_tension_coefficient":
            value /= self.surface_tension_coefficient_reference

        # PHYSICAL QUANTITIES
        elif quantity == "gravity":
            value /= self.gravity_reference
        elif quantity == "length":
            value /= self.length_reference
        elif quantity == "time":
            value /= self.time_reference
        
        # MISC
        elif quantity == "mass":
            value /= self.mass_reference
        elif quantity == "mass_flow":
            value /= self.mass_flow_reference
        else:
            assert False, "Quantity %s is unknown" % quantity

        return value
        
    @partial(jax.jit, static_argnums=(0, 2))
    def dimensionalize(self, value: Union[jnp.ndarray, float], quantity: str) -> Union[jnp.ndarray, float]:
        """Dimensionalizes the given quantity buffer w.r.t. the specified quanty.

        :param value: Non-dimensional quantity buffer
        :type value: Union[jnp.ndarray, float]
        :param quantity: Quantity name
        :type quantity: str
        :return: Dimensional quantity buffer
        :rtype: Union[jnp.ndarray, float]
        """

        # NAME CONVERSION
        if quantity == "rho":
            quantity = "density"
        if quantity in ["u", "v", "w", "velocityX", "velocityY", "velocityZ"]:
            quantity = "velocity"
        if quantity in ["momentumX", "momentumY", "momentumZ"]:
            quantity = "momentum"
        if quantity == "p":
            quantity = "pressure"
        if quantity == "T":
            quantity = "temperature"

        # PRIMES
        if quantity == "density":
            value *= self.density_reference
        elif quantity == "velocity":
            value *= self.velocity_reference
        elif quantity == "temperature":
            value *= self.temperature_reference
        elif quantity == "pressure":
            value *= self.pressure_reference

        # CONS
        elif quantity == "mass":
            value *= self.density_reference
        elif quantity == "momentum":
            value *= (self.density_reference * self.velocity_reference)
        elif quantity == "energy":
            value *= self.pressure_reference

        # MATERIAL PARAMETERS
        elif quantity == "dynamic_viscosity":
            value *= self.viscosity_reference
        elif quantity == "thermal_conductivity":
            value *= self.thermal_conductivity_reference
        elif quantity == "specific_gas_constant":
            value *= self.specific_gas_constant_reference
        elif quantity == "surface_tension_coefficient":
            value *= self.surface_tension_coefficient_reference

        # PHYSICAL QUANTITIES
        elif quantity == "gravity":
            value *= self.gravity_reference
        elif quantity == "length":
            value *= self.length_reference
        elif quantity == "time":
            value *= self.time_reference
        
        # MISC
        elif quantity == "mass":
            value *= self.mass_reference
        elif quantity == "mass_flow":
            value *= self.mass_flow_reference
        else:
            assert False, "Quantity %s is unknown" % quantity

        return value
        
