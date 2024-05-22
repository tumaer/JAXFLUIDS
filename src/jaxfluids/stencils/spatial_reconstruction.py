from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_stencil import SpatialStencil

class SpatialReconstruction(SpatialStencil):
    """This is an abstract spatial reconstruction class. SpatialReconstruction
    class implements functionality for cell face reconstruction from cell 
    averaged values. The paranet class implements the domain slices (nhx, nhy, nhz).
    The reconstruction procedure is implemented in the child classes.
    """

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0) -> None:
        super(SpatialReconstruction, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

    def stencil_slices(self, idx_ranges) -> None:
        slices = []
        for idx_range in idx_ranges:
            slices_x, slices_y, slices_z = [], [], []
            for k in idx_range:
                slices_x.append( jnp.s_[..., k        , None:None, None:None] )
                slices_y.append( jnp.s_[..., None:None, k        , None:None] )
                slices_z.append( jnp.s_[..., None:None, None:None, k        ] )
            slices.append([slices_x, slices_y, slices_z])

        if len(idx_ranges) == 1:
            slices = slices[0] 
        self.s__ = slices

    # @abstractmethod
    def set_slices_stencil(self) -> None:
        """Sets slice objects used in eigendecomposition for flux-splitting scheme.
        In the flux-splitting scheme, each n-point stencil has to be separately 
        accessible as each stencil is transformed into characteristic space.
        """ 
        self.s_ = self.s__

    @abstractmethod
    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx : float = None, **kwargs) -> Array:
        """Reconstruction of buffer quantity along axis specified by axis. 

        :param buffer: Buffer that will be reconstructed
        :type buffer: Array
        :param axis: Spatial axis along which values are reconstructed
        :type axis: int
        :param j: integer which specifies whether to calculate reconstruction left (j=0) or right (j=1)
            of an interface
        :type j: int
        :param dx: cell size, defaults to None
        :type dx: float, optional
        :return: Buffer with cell face reconstructed values
        :rtype: Array
        """
        pass