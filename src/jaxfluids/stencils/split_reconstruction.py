from typing import Dict, List, TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.equation_information import EquationInformation
# from jaxfluids.data_types.numerical_setup.conservatives import SplitReconstructionSetup

class SplitReconstruction(SpatialReconstruction):

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            equation_information: EquationInformation,
            split_reconstruction_setup,
            **kwargs
        ) -> None:
        #TODO offset has to be passed to spatialreconstruction
        super(SplitReconstruction, self).__init__(nh=nh, inactive_axes=inactive_axes)
        
        self.primitives_slices = equation_information.material_field_slices["primitives"]
        self.reconstruction_dict = {}
        for field in self.primitives_slices:
            self.reconstruction_dict[field] = getattr(split_reconstruction_setup, field)(nh, inactive_axes, **kwargs)         

    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs
        ) -> Array:

        cell_state_xi_j = []
        for variable, variable_index in self.primitives_slices.items():
            variable_xi_j = self.reconstruction_dict[variable].reconstruct_xi(
                buffer[variable_index], axis=axis, j=j, dx=dx, **kwargs)
            cell_state_xi_j.append(variable_xi_j)

        cell_state_xi_j = jnp.concatenate(cell_state_xi_j, axis=0)
        return cell_state_xi_j