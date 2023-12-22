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

from abc import ABC, abstractmethod
from typing import List, Union
import types

import jax.numpy as jnp
import numpy as np

from jaxfluids.unit_handler import UnitHandler

class Material(ABC):
    """The Material class implements an abstract class for a material, i.e., the computation of
    the dynamic viscosity, bulk viscosity and the thermal conductivity. The equation of states are 
    implemented in the corresonding child classes.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, unit_handler: UnitHandler, dynamic_viscosity: Union[float, str, types.LambdaType], sutherland_parameters: List,
        bulk_viscosity: float, thermal_conductivity: Union[float, str, types.LambdaType], prandtl_number: float, **kwargs) -> None:

        # MATERIAL TYPE SPECIFIC PARAMETERS
        self.gamma                  = None
        self.R                      = None
        self.cp                     = None

        # PRANDTL
        self.prandtl_number         = prandtl_number

        # DYNAMIC VISCOSITY 
        if type(dynamic_viscosity) in [float, np.float32, np.float64]:
            self.get_dynamic_viscosity = lambda T: unit_handler.non_dimensionalize(dynamic_viscosity, "dynamic_viscosity")

        elif dynamic_viscosity == "Sutherland":
            mu_0, T_0, C = sutherland_parameters
            mu_0    = unit_handler.non_dimensionalize(mu_0, "dynamic_viscosity")
            T_0     = unit_handler.non_dimensionalize(T_0, "temperature")
            C       = unit_handler.non_dimensionalize(C, "temperature")
            self.get_dynamic_viscosity = lambda T: mu_0 * ((T_0 + C)/(T + C)) * (T/T_0)**1.5
        
        elif type(dynamic_viscosity) == types.LambdaType:
            self.get_dynamic_viscosity =  lambda T: unit_handler.non_dimensionalize(dynamic_viscosity(unit_handler.dimensionalize(T, "temperature")), "dynamic_viscosity")

        else:
            assert False, "Viscosity model not implemented"

        # BULK VISCOSITY
        self.bulk_viscosity = bulk_viscosity

        # THERMAL CONDUCTIVITY
        if type(thermal_conductivity) in [float, np.float32, np.float64]:
            self.get_thermal_conductivity = lambda T: unit_handler.non_dimensionalize(thermal_conductivity, "thermal_conductivity")
        
        elif thermal_conductivity == "Prandtl":
            self.get_thermal_conductivity = lambda T: self.cp * self.get_dynamic_viscosity(T) / self.prandtl_number

        elif type(thermal_conductivity) == types.LambdaType:
            self.get_thermal_conductivity =  lambda T: unit_handler.non_dimensionalize(thermal_conductivity(unit_handler.dimensionalize(T, "temperature")), "thermal_conductivity")

        else:
            assert False, "Thermal conductivity not implemented"

    @abstractmethod
    def get_speed_of_sound(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """Computes speed of sound from pressure and density.
        c = c(p, rho)

        :param p: Pressure buffer
        :type p: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :return: Speed of sound buffer
        :rtype: jnp.ndarray
        """
        pass

    @abstractmethod
    def get_pressure(self, e: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """Computes pressure from internal energy and density.
        p = p(e, rho)

        :param e: Specific internal energy buffer
        :type e: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :return: Pressue buffer
        :rtype: jnp.ndarray
        """
        pass

    @abstractmethod
    def get_temperature(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """Computes temperature from pressure and density.
        T = T(p, rho)

        :param p: Pressure buffer
        :type p: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :return: Temperature buffer
        :rtype: jnp.ndarray
        """
        pass
    
    @abstractmethod
    def get_energy(self, p:jnp.ndarray, rho:jnp.ndarray) -> jnp.ndarray:
        """Computes specific internal energy
        e = e(p, rho)

        :param p: Pressure buffer
        :type p: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :return: Specific internal energy buffer
        :rtype: jnp.ndarray
        """
        pass

    @abstractmethod
    def get_total_energy(self, p:jnp.ndarray, rho:jnp.ndarray, u:jnp.ndarray, v:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        """Computes total energy per unit volume from pressure, density, and velocities.
        E = E(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :param u: Velocity in x direction
        :type u: jnp.ndarray
        :param v: Velocity in y direction
        :type v: jnp.ndarray
        :param w: Velocity in z direction
        :type w: jnp.ndarray
        :return: Total energy per unit volume
        :rtype: jnp.ndarray
        """
        pass

    @abstractmethod
    def get_total_enthalpy(self, p:jnp.ndarray, rho:jnp.ndarray, u:jnp.ndarray, v:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        """Computes total specific enthalpy from pressure, density, and velocities.
        H = H(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :param u: Velocity in x direction
        :type u: jnp.ndarray
        :param v: Velocity in y direction
        :type v: jnp.ndarray
        :param w: Velocity in z direction
        :type w: jnp.ndarray
        :return: Total specific enthalpy buffer
        :rtype: jnp.ndarray
        """
        pass

    @abstractmethod
    def get_psi(self, p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """Computes psi from pressure and density.
        psi = p_rho; p_rho is partial derivative of pressure wrt density.

        :param p: Pressure buffer
        :type p: jnp.ndarray
        :param rho: Density buffer
        :type rho: jnp.ndarray
        :return: Psi
        :rtype: jnp.ndarray
        """
        pass

    @abstractmethod
    def get_grueneisen(self, rho: jnp.ndarray) -> jnp.ndarray:
        """Computes the Grueneisen coefficient from density.
        Gamma = p_e / rho; p_e is partial derivative of pressure wrt internal specific energy.

        :param rho: Density buffer
        :type rho: jnp.ndarray
        :return: Grueneisen
        :rtype: jnp.ndarray
        """