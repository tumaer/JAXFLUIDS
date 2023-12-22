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
from typing import List

import jax.numpy as jnp

class SpatialDerivative(ABC):
    """Abstract parent class for the computation of spatial derivatives.

    Calculates either the first spatial derivative wrt to axis direction (derivative_xi),
    or calculates the second spatial derivative wrt to axis1 and axis2 directions (
    derivative_xi_xj). 
    """

    eps = jnp.finfo(jnp.float64).eps
    
    def __init__(self, nh: int, inactive_axis: List, offset: int = 0) -> None:
        self.n                   = nh - offset
        self.nhx                 = jnp.s_[:] if "x" in inactive_axis else jnp.s_[self.n:-self.n]    
        self.nhy                 = jnp.s_[:] if "y" in inactive_axis else jnp.s_[self.n:-self.n]    
        self.nhz                 = jnp.s_[:] if "z" in inactive_axis else jnp.s_[self.n:-self.n]

        self.eps = jnp.finfo(jnp.float64).eps

    @abstractmethod
    def derivative_xi(self, buffer: jnp.ndarray, dxi: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Calculates the derivative in the direction indicated by axis.

        :param buffer: Buffer for which the derivative will be calculated
        :type buffer: jnp.ndarray
        :param dxi: Cell sizes along axis direction
        :type dxi: jnp.ndarray
        :param axis: Spatial axis along which derivative is calculated
        :type axis: int
        :return: Buffer with numerical derivative
        :rtype: jnp.ndarray
        """
        pass

    def derivative_xi_xj(self, buffer: jnp.ndarray, dxi: jnp.ndarray, dxj: jnp.ndarray, i: int, j: int) -> jnp.ndarray:
        """Calculates the second derivative in the directions indicated by i and j.

        :param buffer: Buffer for which the second derivative will be calculated
        :type buffer: jnp.ndarray
        :param dxi: Cell sizes along i direction
        :type dxi: jnp.ndarray
        :param dxj: Cell sizes along j direction
        :type dxj: jnp.ndarray
        :param i: Spatial axis along which derivative is calculated
        :type i: int
        :param j: Spatial axis along which derivative is calculated
        :type j: int
        :return: Buffer with numerical derivative
        :rtype: jnp.ndarray
        """
        pass