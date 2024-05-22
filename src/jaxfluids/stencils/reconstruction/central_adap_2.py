from typing import List

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class CentralSecondOrderAdapReconstruction(SpatialReconstruction):
    """CentralSecondOrderReconstruction 

    2nd order stencil for reconstruction at the cell face
          x
    |     |     |
    |  i  | i+1 |
    |     |     |
    """

    required_halos = 1
    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            is_mesh_stretching: List = None,
            cell_sizes: List = None
        ) -> None:
        super(CentralSecondOrderAdapReconstruction, self).__init__(nh=nh,
                                                                   inactive_axes=inactive_axes,
                                                                   offset=offset)

        # TODO
        assert is_mesh_stretching is not None, "is_mesh_stretching is not optional for adap stencil."
        assert cell_sizes is not None, "cell_sizes is not optional for adap stencil."

        self.order = 2
        self.array_slices([(-1, 0)])

        self.coeffs = []
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                cell_sizes_i = cell_sizes_i[self.s_nh_xi[i]]

                # CELL SIZES
                delta_x0 = cell_sizes_i[self.s_mesh[i][0]]
                delta_x1 = cell_sizes_i[self.s_mesh[i][1]]

                # DISTANCE TO i+1/2
                d0 = - 0.5 * delta_x0
                d1 = 0.5 * delta_x1

                # COEFFICIENTS
                c0 = - d1 / (d0 - d1)
                c1 = d0 / (d0 - d1)

                c0 = delta_x1 / (delta_x0 + delta_x1)
                c1 = delta_x0 / (delta_x0 + delta_x1)

            else:
                c0 = 0.5
                c1 = 0.5
            self.coeffs.append([jnp.array(c0), jnp.array(c1)])

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
            ) -> Array:

        s1_  = self.s_[axis]
        coeffs = self.coeffs[axis]

        if coeffs[0].ndim == 4:
            c_xi = []
            device_id = jax.lax.axis_index(axis_name="i")
            for m in range(self.order):
                c_xi.append(coeffs[m][device_id])
        else:
            c_xi = coeffs

        cell_state_xi = c_xi[0] * buffer[s1_[0]] \
            + c_xi[1] * buffer[s1_[1]]

        return cell_state_xi