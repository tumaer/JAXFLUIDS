from collections import namedtuple
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array

class MaterialFieldBuffers(NamedTuple):
    """Contains the conserved and primitive 
    variables associated with the
    fluid materials.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    conservatives: Array = None
    primitives: Array = None
    temperature: Array = None

class LevelsetSolidCellIndicesField(NamedTuple):
    indices: Tuple[Array] = None
    mask: Array = None 

class LevelsetSolidCellIndices(NamedTuple):
    interface: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    extension_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    extension_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()

    mixing_source_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ii_0_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ii_1_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ii_2_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ij_01_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ij_02_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ij_12_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ijk_fluid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()

    mixing_source_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ii_0_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ii_1_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ii_2_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ij_01_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ij_02_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ij_12_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()
    mixing_target_ijk_solid: LevelsetSolidCellIndicesField = LevelsetSolidCellIndicesField()

class LevelsetSolidInterfaceState(NamedTuple):
    primitives: Array = None
    solid_temperature: Array = None

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
    solid_cell_indices: LevelsetSolidCellIndices = LevelsetSolidCellIndices()
    interface_state_ip: LevelsetSolidInterfaceState = None

class SolidFieldBuffers(NamedTuple):
    velocity: Array = None
    energy: Array = None
    temperature: Array = None

class SimulationBuffers(NamedTuple):
    # TODO should we register SimulationsBuffers
    # as a custom pytree node? jax.tree_util.register_pytree_node_class
    """Contains all buffers that
    are advanced in time, i.e., 
    material fields, levelset related fields 

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    material_fields: MaterialFieldBuffers = None
    levelset_fields: LevelsetFieldBuffers = None
    solid_fields: SolidFieldBuffers = None
    # boundary_fields: BoundaryFieldBuffers = None

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
    fixed_time_step_size: float = None
    end_time: float = None
    end_step: int = None

class EulerIntegrationBuffers(NamedTuple):
    conservatives: Array = None
    levelset: Array = None
    solid_velocity: Array = None
    solid_energy: Array = None

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
    euler_buffers: EulerIntegrationBuffers = None

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
    solid_temperature_force: Array = None
    acoustic_force: Array = None
    custom_force: Array = None
    sponge_layer_force: Array = None
    enthalpy_damping_force: Array = None

class ControlFlowParameters(NamedTuple):
    perform_reinitialization: bool = False
    perform_compression: bool = False
    is_cumulative_statistics: bool = False
    is_logging_statistics: bool = False
    is_feed_foward: bool = False