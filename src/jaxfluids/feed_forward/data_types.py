from typing import Dict, NamedTuple
from jaxfluids.data_types.buffers import SimulationBuffers, TimeControlVariables, ForcingParameters
from jaxfluids.data_types.ml_buffers import ParametersSetup

class FeedForwardSetup(NamedTuple):
    outer_steps: int
    inner_steps: int = 1,
    is_scan: bool = True
    is_checkpoint_inner_step: bool = True
    is_checkpoint_integration_step: bool = False
    is_include_t0: bool = True
    is_include_halos: bool = True

class ScanFields(NamedTuple):
    simulation_buffers: SimulationBuffers
    time_control_variables: TimeControlVariables
    forcing_parameters: ForcingParameters
    ml_parameters: ParametersSetup