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
from typing import Callable

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager

class RiemannSolver(ABC):
    """Abstract base class for Riemann solvers.

    RiemannSolver has two fundamental attributes: a material manager and a signal speed.
    The solve_riemann_problem_xi method solves the one-dimensional Riemann problem.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        self.material_manager = material_manager
        self.signal_speed = signal_speed
    
    @abstractmethod
    def solve_riemann_problem_xi(self, primes_L: jnp.ndarray, primes_R: jnp.ndarray, 
        cons_L: jnp.ndarray, cons_R: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        """Solves one-dimensional Riemann problem in the direction as specified 
        by the axis argument.

        :param primes_L: primtive variable buffer left of cell face
        :type primes_L: jnp.ndarray
        :param primes_R: primtive variable buffer right of cell face
        :type primes_R: jnp.ndarray
        :param cons_L: conservative variable buffer left of cell face
        :type cons_L: jnp.ndarray
        :param cons_R: conservative variable buffer right of cell face
        :type cons_R: jnp.ndarray
        :param axis: Spatial direction along which Riemann problem is solved.
        :type axis: int
        :return: buffer of fluxes in xi direction 
        :rtype: jnp.ndarray
        """
        pass