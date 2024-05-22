from typing import NamedTuple, List
import jax.numpy as jnp
from jax import Array

class MassFlowForcingInformation(NamedTuple):
    """MassFlowForcingInformation contains information regarding the mass flow forcing.
    - current_value: current mass flow
    - target_value: user-specified target mass flow
    - force_scalar: current scalar mass flow forcing 

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    current_value: float = None
    target_value: float = None
    force_scalar: float = None

class TemperatureForcingInformation(NamedTuple):
    """_summary_

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    current_error: float = None

class ForcingInformation(NamedTuple):
    """ForcingInformation contains information on the forcings.
    - mass flow: current and target value and
    scalar value of the force
    - temperature: current error

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    
    mass_flow: MassFlowForcingInformation = None
    temperature: TemperatureForcingInformation = None

class LevelsetPositivityInformation(NamedTuple):
    """LevelsetPositivityInformation contains information for the mixing
    procedure within the levelset algorithm


    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    mixing_invalid_cell_count: int = None
    extension_invalid_cell_count: int = None

class PositivityStateInformation(NamedTuple):
    """PositivityStateInformation contains minimum and maximum values
    for critical state variables.
    I.e., pressure, density, and volume-fraction.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    min_pressure: float = None
    min_density: float = None
    min_alpharho: float = None
    min_alpha: float = None
    max_alpha: float = None
    positivity_count_flux: jnp.int32 = None
    positivity_count_interpolation: jnp.int32 = None
    positivity_count_thinc: jnp.int32 = None
    positivity_count_acdi: jnp.int32 = None
    vf_correction_count: jnp.int32 = None
    count_acdi: jnp.int32 = None
    count_thinc: jnp.int32 = None
    levelset: LevelsetPositivityInformation = None

class LevelsetResidualInformation(NamedTuple):
    """LevelsetResidualInformation contains the residuals of 
    the extension and reinitialization
    procedure within the levelset 
    algorithm.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    reinitialization_residual_mean: float = None
    reinitialization_residual_max: float = None
    reinitialization_steps: int = None
    prime_extension_residual_mean: float = None
    prime_extension_residual_max: float = None
    prime_extension_steps: int = None
    interface_quantity_extension_residual_mean: float = None
    interface_quantity_extension_residual_max: float = None
    interface_quantity_extension_steps: int = None

class HITStatisticsLogging(NamedTuple):
    """HITStatisticsLogging contains statistics of HIT simulations
    which are used for logging.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    rho_bulk: float
    pressure_bulk: float
    temperature_bulk: float
    rho_rms: float
    pressure_rms: float
    temperature_rms: float
    u_rms: float
    mach_rms: float

class ChannelStatisticsLogging(NamedTuple):
    """ChannelStatisticsLogging contains statistics
    of turbulent channel simulations
    which are used for logging.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    rho_bulk: float
    pressure_bulk: float
    temperature_bulk: float
    u_bulk: float
    mach_bulk: float
    reynolds_tau: float
    reynolds_bulk: float
    delta_x_plus: float
    delta_y_plus_min: float
    delta_y_plus_max: float
    delta_z_plus: float

class BoundaryLayerStatisticsLogging(NamedTuple):
    """BoundaryLayerStatisticsLogging contains statistics
    of turbulent boundary layer simulations
    which are used for logging.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    l_plus: float
    delta_0: float
    delta_1: float
    delta_2: float
    reynolds_tau: float
    delta_x_plus: float
    delta_y_plus_min: float
    delta_y_plus_edge: float
    delta_z_plus: float


class StatisticsLogging(NamedTuple):
    hit_statistics: HITStatisticsLogging = None
    channel_statistics: ChannelStatisticsLogging = None
    boundarylayer_statistics: BoundaryLayerStatisticsLogging = None

class HITStatisticsCumulative(NamedTuple):
    number_sample_steps: int
    number_sample_points: int
    density_T: float
    pressure_T: float
    temperature_T: float
    c_T: float
    rhop_rhop_S: float
    pp_pp_S: float
    Tp_Tp_S: float
    machp_machp_S: float
    up_up_S: float
    vp_vp_S: float
    wp_wp_S: float

class ChannelStatisticsCumulative(NamedTuple):
    number_sample_steps: int
    number_sample_points: int
    U_T: Array
    V_T: Array
    W_T: Array
    density_T: Array
    pressure_T: Array
    T_T: Array
    c_T: Array
    mach_T: Array
    pp_pp_S: Array
    rhop_rhop_S: Array
    machp_machp_S: Array
    up_up_S: Array
    vp_vp_S: Array
    wp_wp_S: Array
    up_vp_S: Array
    up_wp_S: Array
    vp_wp_S: Array
    Tp_Tp_S: Array
    vp_Tp_S: Array

class BoundaryLayerStatisticsCumulative(NamedTuple):
    number_sample_steps: int
    number_sample_points: int
    U_T: Array
    V_T: Array
    W_T: Array
    density_T: Array
    pressure_T: Array
    T_T: Array
    c_T: Array
    mach_T: Array
    pp_pp_S: Array
    rhop_rhop_S: Array
    machp_machp_S: Array
    up_up_S: Array
    vp_vp_S: Array
    wp_wp_S: Array
    up_vp_S: Array
    up_wp_S: Array
    vp_wp_S: Array
    Tp_Tp_S: Array
    vp_Tp_S: Array

class StatisticsCumulative(NamedTuple):
    hit_statistics: HITStatisticsCumulative = None
    channel_statistics: ChannelStatisticsCumulative = None
    boundarylayer_statistics: BoundaryLayerStatisticsCumulative = None

class TurbulentStatisticsInformation(NamedTuple):
    logging: StatisticsLogging
    cumulative: StatisticsCumulative

class StepInformation(NamedTuple):
    """Contains simulation information
    on positivity states,
    levelset residuals, forcing status,
    and turbulent statistics.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    positivity_state_info_list: List[PositivityStateInformation] = []
    levelset_residuals_info_list: List[LevelsetResidualInformation] = []
    levelset_positivity_info_list: List[LevelsetPositivityInformation] = []
    forcing_info: ForcingInformation = None

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
