from typing import NamedTuple, List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.data_types.statistics import FlowStatistics

Array = jax.Array

class MassFlowForcingInformation(NamedTuple):
    current_value: float = None
    target_value: float = None
    force_scalar: float = None

class TemperatureForcingInformation(NamedTuple):
    current_error_fluid: float = None
    current_error_solid: float = None

class SpongeLayerForcingInformation(NamedTuple):
    error: float = None

class ForcingInformation(NamedTuple):
    """Contains information on the forcings.
    - mass flow: current and target value and
    scalar value of the force
    - temperature: current error

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    
    mass_flow: MassFlowForcingInformation = None
    temperature: TemperatureForcingInformation = None
    sponge_layer: SpongeLayerForcingInformation = None

class LevelsetPositivityInformation(NamedTuple):
    """Contains information for the mixing
    procedure within the levelset 
    algorithm


    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    mixing_invalid_cell_count: int = None
    extension_invalid_cell_count: int = None

class DiscretizationCounter(NamedTuple):
    acdi: int = None
    thinc: int = None

class PositivityCounter(NamedTuple):
    interpolation_limiter: int = None
    thinc_limiter: int = None
    flux_limiter: int = None
    acdi_limiter: int = None
    volume_fraction_limiter: int = None

class PositivityStateInformation(NamedTuple):
    """Contains minimum and maximum values
    for critical state variables.
    I.e., pressure, density, and volume-fraction.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    min_pressure: float = None
    min_density: float = None
    min_temperature: float = None
    min_alpharho: float = None
    min_alpha: float = None
    max_alpha: float = None
    positivity_counter: PositivityCounter = None
    discretization_counter: DiscretizationCounter = None
    levelset_fluid: LevelsetPositivityInformation = None
    levelset_solid: LevelsetPositivityInformation = None

class LevelsetProcedureInformation(NamedTuple):
    steps: int = None
    max_residual: float = None
    mean_residual: float = None

class LevelsetResidualInformation(NamedTuple):
    """Contains the residuals of 
    the extension and reinitialization
    procedure within the levelset 
    algorithm.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    reinitialization: LevelsetProcedureInformation
    primitive_extension: LevelsetProcedureInformation
    interface_extension: LevelsetProcedureInformation
    solids_extension: LevelsetProcedureInformation

class StepInformation(NamedTuple):
    """Contains simulation information
    on positivity states,
    levelset residuals, forcing status,
    and turbulent statistics.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    positivity: Tuple[PositivityStateInformation] = []
    levelset: Tuple[LevelsetResidualInformation] = []
    forcing_info: ForcingInformation = None
    statistics: FlowStatistics = None

class WallClockTimes(NamedTuple):
    """Contains the wall clock times per
    simulation step.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    step: float = 0.0
    step_per_cell: float = 0.0
    mean_step: float = 0.0
    mean_step_per_cell: float = 0.0
