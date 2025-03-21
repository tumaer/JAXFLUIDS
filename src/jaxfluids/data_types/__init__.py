from typing import Dict, NamedTuple, Tuple

import jax

from jaxfluids.data_types.buffers import SimulationBuffers, TimeControlVariables, ForcingParameters
from jaxfluids.data_types.information import StepInformation, FlowStatistics

Array = jax.Array

class JaxFluidsBuffers(NamedTuple):
    simulation_buffers: SimulationBuffers
    time_control_variables: TimeControlVariables
    forcing_parameters: ForcingParameters 
    step_information: StepInformation

class JaxFluidsData(NamedTuple):
    cell_centers: Tuple[Array]
    cell_sizes: Tuple[Array]
    times: Array
    data: Dict[str, Array]