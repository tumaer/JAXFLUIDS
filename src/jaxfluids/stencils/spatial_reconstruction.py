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

class SpatialReconstruction(ABC):
    """This is an abstract spatial reconstruction class. SpatialReconstruction
    class implements functionality for cell face reconstruction from cell 
    averaged values. The paranet class implements the domain slices (nhx, nhy, nhz).
    The reconstruction procedure is implemented in the child classes.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, nh: int, inactive_axis: List, offset: int = 0) -> None:

        self.n                  = nh - offset
        self.nhx                = jnp.s_[:] if "x" in inactive_axis else jnp.s_[self.n:-self.n]    
        self.nhy                = jnp.s_[:] if "y" in inactive_axis else jnp.s_[self.n:-self.n]    
        self.nhz                = jnp.s_[:] if "z" in inactive_axis else jnp.s_[self.n:-self.n]

        self._stencil_size      = None

    # @abstractmethod
    def set_slices_stencil(self) -> None:
        """Sets slice objects used in eigendecomposition for flux-splitting scheme.
        In the flux-splitting scheme, each n-point stencil has to be separately 
        accessible as each stencil is transformed into characteristic space.
        """ 
        pass

    @abstractmethod
    def reconstruct_xi(self, buffer: jnp.ndarray, axis: int, j: int, dx : float = None, **kwargs) -> jnp.ndarray:
        """Reconstruction of buffer quantity along axis specified by axis. 

        :param buffer: Buffer that will be reconstructed
        :type buffer: jnp.ndarray
        :param axis: Spatial axis along which values are reconstructed
        :type axis: int
        :param j: integer which specifies whether to calculate reconstruction left (j=0) or right (j=1)
            of an interface
        :type j: int
        :param dx: cell size, defaults to None
        :type dx: float, optional
        :return: Buffer with cell face reconstructed values
        :rtype: jnp.ndarray
        """
        pass