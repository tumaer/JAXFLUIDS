from typing import Dict, NamedTuple, List, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


class HITStatisticsLogging(NamedTuple):
    rho_bulk: float = None
    pressure_bulk: float = None
    temperature_bulk: float = None
    rho_rms: float = None
    pressure_rms: float = None
    temperature_rms: float = None
    u_rms: float = None
    mach_rms: float = None

class ChannelStatisticsLogging(NamedTuple):
    rho_bulk: float = None
    pressure_bulk: float = None
    temperature_bulk: float = None
    u_bulk: float = None
    mach_bulk: float = None
    mach_0: float = None
    reynolds_tau: float = None
    reynolds_bulk: float = None
    reynolds_0: float = None
    delta_x_plus: float = None
    delta_y_plus_min: float = None
    delta_y_plus_max: float = None
    delta_z_plus: float = None

class BoundaryLayerStatisticsLogging(NamedTuple):
    l_plus: float = None
    delta_0: float = None
    delta_1: float = None
    delta_2: float = None
    reynolds_tau: float = None
    delta_x_plus: float = None
    delta_y_plus_min: float = None
    delta_y_plus_edge: float = None
    delta_z_plus: float = None

class StatisticsLogging(NamedTuple):
    hit: HITStatisticsLogging = None
    channel: ChannelStatisticsLogging = None
    boundary_layer: BoundaryLayerStatisticsLogging = None


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
    density_mean: Array
    velX_mean: Array
    velY_mean: Array
    velZ_mean: Array
    pressure_mean: Array
    temperature_mean: Array
    speed_of_sound_mean: Array
    mach_mean: Array
    pp_pp_square: Array
    rhop_rhop_square: Array
    machp_machp_square: Array
    up_up_square: Array
    vp_vp_square: Array
    wp_wp_square: Array
    up_vp_square: Array
    up_wp_square: Array
    vp_wp_square: Array
    Tp_Tp_square: Array
    vp_Tp_square: Array

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

class Metrics(NamedTuple):
    sampling_dt: float = 0.0
    next_sampling_time: float = 0.0
    sample_steps: int = 0
    total_sample_points: int = 0
    total_sample_weights: float | Array = 0.0
    reynolds_means: Dict[str,Array] = None
    favre_means: Dict[str,Array] = None
    reynolds_covs: Dict[str,Array] = None
    favre_covs: Dict[str,Array] = None

class StatisticsCumulative(NamedTuple):
    metrics: Metrics = None
    hit: Dict = None
    channel: Dict = None
    boundary_layer: Dict = None


class TurbulenceStatisticsInformation(NamedTuple):
    logging: StatisticsLogging = None
    cumulative: StatisticsCumulative = None

class FlowStatistics(NamedTuple):
    turbulence: TurbulenceStatisticsInformation = TurbulenceStatisticsInformation()