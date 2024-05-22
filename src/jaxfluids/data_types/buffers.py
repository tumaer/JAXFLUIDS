from collections import namedtuple
from typing import Dict, NamedTuple

import jax.numpy as jnp
from jax import Array

class MaterialFieldBuffers(NamedTuple):
    """Contains the conserved and primitive 
    variables associated with the
    fluid materials.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    conservatives: Array = None
    primitives: Array = None

class LevelsetFieldBuffers(NamedTuple):
    """Contains the levelset,
    the associated volume fraction and apertures,
    and the interface velocity.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    levelset: Array = None
    volume_fraction: Array = None
    apertures: Array = None
    interface_velocity: Array = None
    interface_pressure: Array = None

class SimulationBuffers(NamedTuple):
    """Contains all buffers that
    are advanced in time, i.e., 
    material fields, levelset related fields
    buffers.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    material_fields: MaterialFieldBuffers = None
    levelset_fields: LevelsetFieldBuffers = None

class TimeControlVariables(NamedTuple):
    """Contains time control variables, i.e.,
    the current physical simulation time, the
    current simulation step

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    physical_simulation_time: float = None
    simulation_step: int = None
    physical_timestep_size: float = None

class IntegrationBuffers(NamedTuple):
    """Contains buffers related the fields
    that are integrated. This container
    is used to store:
     - the actual buffers
     - the corresponding right hand side
     - the correspondong initial stage
        buffers

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    conservatives: Array = None
    levelset: Array = None
    interface_velocity: Array = None

MassFlowControllerParameters = namedtuple("MassFlowControllerParameters",
    ["current_error", "integral_error"])
    
class ForcingParameters(NamedTuple):
    """Contains parameters for forcings
    - mass flow: PID controller parameters
    - hit: reference energy spectrum.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    mass_flow_controller_params: MassFlowControllerParameters = None
    hit_ek_ref: Array = None
    # TODO aaron, why is temperature_target, forcing cutoff etc not in here???

class ForcingBuffers(NamedTuple):
    """Contains the forcing buffers that
    are added to the right hand side.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    mass_flow_force: Array = None
    hit_force: Array = None
    temperature_force: Array = None
    acoustic_force: Array = None
    custom_force: Array = None

class MLParameters(NamedTuple):
    """Contains neural network functions and 
    neural network parameters. Networks must 
    be an immutable dict.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    networks: Dict = None
    parameters: Dict = None
