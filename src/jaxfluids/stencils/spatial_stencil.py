from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.config import precision

class SpatialStencil(ABC):
    """This is an abstract spatial stencil class. SpatialStencil 
    implements the domain slices (nhx, nhy, nhz).
    The reconstruction procedure is implemented in the child classes.
    """

    required_halos = 0
    is_for_adaptive_mesh = False

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0) -> None:

        # self.eps = precision.get_smallest_normal()
        self.eps = precision.get_spatial_stencil_eps()
        self.n = nh - offset

        # DOMAIN + OFFSET SLICES
        self.nhx = jnp.s_[:] if "x" in inactive_axes else jnp.s_[self.n:-self.n]    
        self.nhy = jnp.s_[:] if "y" in inactive_axes else jnp.s_[self.n:-self.n]    
        self.nhz = jnp.s_[:] if "z" in inactive_axes else jnp.s_[self.n:-self.n]

        # DOMAIN + GHOST - OFFSET SLICES
        self.nhx_ = jnp.s_[:] if "x" in inactive_axes else jnp.s_[offset:-offset if offset > 0 else None]    
        self.nhy_ = jnp.s_[:] if "y" in inactive_axes else jnp.s_[offset:-offset if offset > 0 else None]    
        self.nhz_ = jnp.s_[:] if "z" in inactive_axes else jnp.s_[offset:-offset if offset > 0 else None]

        self.s_nh_ = jnp.s_[self.nhx_, self.nhy_, self.nhz_]
        self.s_nh_xi = [
            jnp.s_[...,self.nhx_, :, :],
            jnp.s_[...,:, self.nhy_, :],
            jnp.s_[...,:, :, self.nhz_],
        ]

        self.s_ = None
        self.s_mesh = None
        self._stencil_size = None

    def array_slices(self, idx_ranges: List, at_cell_center: bool = False) -> None:
        """Generates array slice objects and sets these as a member. Array slices
        are used to compute derivatives and cell-face reconstruction. Exemplary domain
        slices are:

        Example 1) WENO3-type reconstruction at i_plus_half_L
        Idx_range: (-2, -1, 0)
        At cell center: False --> i.e., evaluation at cell face

                        *----------------------------------------*
                                *------------------------------------------*
                                        *-------------------------------------------*

        Example 2) 2nd order central FD
        Idx_range: (-1, 0, 1)
        At cell center: True --> i.e., evaluation at cell center, slices are shorter by 1 element

                                *---------------------------------*
                                         *----------------------------------*
                                                *------------------------------------*
                                      |                                          |
        |   |     |         |         |     |         |     |          |         |      |        |    |    |
        | 0 | ... | n_h - 2 | n_h - 1 | n_h | n_h + 1 | ... | -n_h - 2 | -n_h -1 | -n_h | -n_h+1 |... | -1 |  
        |   |     |         |         |     |         |     |          |         |      |        |    |    |
                                      |                                          |

        Args:
            idx_ranges (List): Contains tuples of left-shift indices relative to the n_h cell
            at_cell_center (bool, optional): Indicates whether is stencil is for evaluation at 
                the cell center (True) or at cell face (False). Slices for evaluation at cell faces 
                contain one index more compared to slices which for evaluation at cell center. Defaults to False.
        """
        nhx, nhy, nhz = self.nhx, self.nhy, self.nhz
        slices = []
        slices_mesh = []
        
        for idx_range in idx_ranges:
            slices_x, slices_y, slices_z = [], [], []
            slices_x_mesh, slices_y_mesh, slices_z_mesh = [], [], []
            for k in idx_range:
                nlo = self.n + k
                assert_msg = f"Stencil with left-shift {k} is too wide for domain with {self.n} halo cells."
                assert nlo >= 0, assert_msg
                
                nhi = -self.n + k + 1 - at_cell_center
                if nhi == 0:
                    nhi = None 
                else:
                    assert_msg = f"Stencil with left-shift {k} is too wide for domain with {self.n} halo cells."
                    assert nhi < 0, assert_msg
                
                slices_x.append( jnp.s_[..., nlo:nhi, nhy    , nhz    ] )
                slices_y.append( jnp.s_[..., nhx    , nlo:nhi, nhz    ] )
                slices_z.append( jnp.s_[..., nhx    , nhy    , nlo:nhi] )

                slices_x_mesh.append( jnp.s_[..., nlo:nhi, :      , :      ] )
                slices_y_mesh.append( jnp.s_[..., :      , nlo:nhi, :      ] )
                slices_z_mesh.append( jnp.s_[..., :      , :      , nlo:nhi] )
            slices.append([slices_x, slices_y, slices_z])
            slices_mesh.append([slices_x_mesh, slices_y_mesh, slices_z_mesh])

        if len(idx_ranges) == 1:
            slices = slices[0] 
            slices_mesh = slices_mesh[0]
        self.s_ = slices
        self.s_mesh = slices_mesh