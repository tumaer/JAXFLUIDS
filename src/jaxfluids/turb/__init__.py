from jaxfluids.turb.statistics.turb_stats_manager_online import TurbulentOnlineStatisticsManager
from jaxfluids.turb.statistics.turb_stats_manager_postprocess import TurbulentStatisticsManager

__all__ = (
    "TurbulentOnlineStatisticsManager",
    "TurbulentStatisticsManager"
)

TUPLE_TURB_INIT_CONDITIONS = (
    "HIT", "CHANNEL", "DUCT",
    "TGV"
)

TUPLE_HIT_ENERGY_SPECTRUM = (
    "EXPONENTIAL", "KOLMOGOROV", "BOX"
)

TUPLE_HIT_IC_TYPE = (
    "IC1", "IC2", "IC3", "IC4"
)

TUPLE_VELOCITY_PROFILES_CHANNEL = (
    "LAMINAR", "TURBULENT"
)
