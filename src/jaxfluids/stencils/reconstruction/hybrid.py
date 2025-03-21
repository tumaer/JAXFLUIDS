from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_z import WENO5Z
from jaxfluids.stencils.reconstruction.central.central_6 import CentralSixthOrderReconstruction
from jaxfluids.shock_sensor.ducros import Ducros
from jaxfluids.equation_information import EquationInformation

Array = jax.Array

class HybridReconstruction(SpatialReconstruction):
    """ Hybrid WENO-Central reconstruction """
    
    required_halos = max(WENO5Z.required_halos, CentralSixthOrderReconstruction.required_halos)

    def __init__(self,
            nh,
            inactive_axes,
            equation_information : EquationInformation,
            domain_information: DomainInformation,
            offset: int = 0,
            **kwargs
            ):
        
        super().__init__(nh, inactive_axes, offset)
        self.upwind_stencil = WENO5Z(nh, inactive_axes, offset)
        self.central_stencil = CentralSixthOrderReconstruction(nh, inactive_axes)
        self.shock_sensor = Ducros(domain_information)
        self.s_velocity = equation_information.s_velocity 
        self.pressure_slices = equation_information.s_energy
        self.nh = domain_information.nh_conservatives
        
    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int, 
            j: int,
            **kwargs
        ) -> Array:
        vels = buffer[self.s_velocity]
        pressure = buffer[self.pressure_slices]
        cell_state_weno = self.weno5_z.reconstruct_xi(buffer, axis, j)
        cell_state_central = self.central_6.reconstruct_xi(buffer, axis)

        fs = self.shock_sensor.compute_sensor_function(vels, axis)

        cell_state = cell_state_central + fs * (cell_state_weno - cell_state_central)


        return cell_state

        
