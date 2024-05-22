from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_stencil import SpatialStencil

class SpatialDerivative(SpatialStencil):
    """Abstract base class for the computation of spatial derivatives.

    Calculates either the first spatial derivative wrt to axis direction (derivative_xi),
    or calculates the second spatial derivative wrt to axis1 and axis2 directions (
    derivative_xi_xj). 
    """
    
    def __init__(self, nh: int, inactive_axes: List, offset: int = 0) -> None:
        super(SpatialDerivative, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

    @abstractmethod
    def derivative_xi(self, buffer: Array, dxi: Array, axis: int) -> Array:
        """Calculates the derivative in the direction indicated by axis.

        :param buffer: Buffer for which the derivative will be calculated
        :type buffer: Array
        :param dxi: Cell sizes along axis direction
        :type dxi: Array
        :param axis: Spatial axis along which derivative is calculated
        :type axis: int
        :return: Buffer with numerical derivative
        :rtype: Array
        """
        pass

    def derivative_xi_xj(self, buffer: Array, dxi: Array, dxj: Array, i: int, j: int) -> Array:
        """Calculates the second derivative in the directions indicated by i and j.

        :param buffer: Buffer for which the second derivative will be calculated
        :type buffer: Array
        :param dxi: Cell sizes along i direction
        :type dxi: Array
        :param dxj: Cell sizes along j direction
        :type dxj: Array
        :param i: Spatial axis along which derivative is calculated
        :type i: int
        :param j: Spatial axis along which derivative is calculated
        :type j: int
        :return: Buffer with numerical derivative
        :rtype: Array
        """
        pass