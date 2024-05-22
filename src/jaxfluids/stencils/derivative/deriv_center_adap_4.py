from typing import List

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeFourthOrderAdapCenter(SpatialDerivative):
    """4th order stencil for 1st derivative at the cell center
                   x 
    |     |     |     |     |     |
    | i-2 | i-1 |  i  | i+1 | i+2 |     
    |     |     |     |     |     |
    """
    
    required_halos = 2
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            is_mesh_stretching: List = None,
            cell_sizes: List = None
        ) -> None:
        super(DerivativeFourthOrderAdapCenter, self).__init__(nh=nh,
                                                              inactive_axes=inactive_axes,
                                                              offset=offset)

        # TODO
        assert is_mesh_stretching is not None, "is_mesh_stretching is not optional for adap stencil."
        assert cell_sizes is not None, "cell_sizes is not optional for adap stencil."

        self.order = 4
        self.array_slices([(-2,-1,0,1,2)], at_cell_center=True)

        self.coeffs = []
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                delta_x0 = cell_sizes_i[self.s_mesh[i][0]] # dx_{i-2}
                delta_x1 = cell_sizes_i[self.s_mesh[i][1]] # dx_{i-1}
                delta_x2 = cell_sizes_i[self.s_mesh[i][2]] # dx_{i}
                delta_x3 = cell_sizes_i[self.s_mesh[i][3]] # dx_{i+1}
                delta_x4 = cell_sizes_i[self.s_mesh[i][4]] # dx_{i+2}
                
                d0 = -0.5 * delta_x2 - delta_x1 - 0.5 * delta_x0
                d1 = -0.5 * delta_x2 - 0.5 * delta_x1
                d2 = 0.0
                d3 = 0.5 * delta_x2 + 0.5 * delta_x3
                d4 = 0.5 * delta_x2 + delta_x3 + 0.5 * delta_x4

                c0 = - (d1 * d2 * d3 + d1 * d2 * d4 + d1 * d3 * d4 + d2 * d3 * d4) \
                    / ((d0 - d1) * (d0 - d2) * (d0 - d3) * (d0 - d4))
                c1 = (d0 * d2 * d3 + d0 * d2 * d4 + d0 * d3 * d4 + d2 * d3 * d4) \
                    / ((d0 - d1) * (d1 - d2) * (d1 - d3) * (d1 - d4))
                c2 = - (d0 * d1 * d3 + d0 * d1 * d4 + d0 * d3 * d4 + d1 * d3 * d4) \
                    / ((d0 - d2) * (d1 - d2) * (d2 - d3) * (d2 - d4))
                c3 = (d0 * d1 * d2 + d0 * d1 * d4 + d0 * d2 * d4 + d1 * d2 * d4) \
                    / ((d0 - d3) * (d1 - d3) * (d2 - d3) * (d3 - d4))
                c4 = - (d0 * d1 * d2 + d0 * d1 * d3 + d0 * d2 * d3 + d1 * d2 * d3) \
                    / ((d0 - d4) * (d1 - d4) * (d2 - d4) * (d3 - d4))

            else:
                d = 12.0 * cell_sizes_i
                c0 = 1.0 / d
                c1 = -8.0 / d
                c2 = 0.0 / d
                c3 = 8.0 / d
                c4 = -1.0 / d
            self.coeffs.append([jnp.array(c0), jnp.array(c1), jnp.array(c2),
                                jnp.array(c3), jnp.array(c4)])

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
        ) -> Array:
        
        s1_  = self.s_[axis]
        coeffs = self.coeffs[axis]

        if coeffs[0].ndim == 4:
            c_xi = []
            device_id = jax.lax.axis_index(axis_name="i")
            for m in range(self.order + 1):
                c_xi.append(coeffs[m][device_id])
        else:
            c_xi = coeffs

        deriv_xi = c_xi[0] * buffer[s1_[0]] \
            + c_xi[1] * buffer[s1_[1]] \
            + c_xi[2] * buffer[s1_[2]] \
            + c_xi[3] * buffer[s1_[3]] \
            + c_xi[4] * buffer[s1_[4]]

        return deriv_xi
