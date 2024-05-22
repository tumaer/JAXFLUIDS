import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class Laplacian:

    def __init__(
            self,
            derivative_stencil: SpatialDerivative,
            domain_information: DomainInformation) -> None:
        
        self.derivative_stencil: SpatialDerivative = derivative_stencil(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes,
            offset=0
        )

        self.domain_information = domain_information

    def compute_laplacian(self, buffer: Array) -> Array:

        active_axes_indices = self.domain_information.active_axes_indices
        cell_sizes = self.domain_information.get_device_cell_sizes()

        laplacian = 0.0
        for axis_index in active_axes_indices:
            laplacian += self.derivative_stencil.derivative_xi(buffer,
                                                               cell_sizes[axis_index],
                                                               axis_index)
        return laplacian
