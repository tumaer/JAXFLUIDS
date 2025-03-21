from typing import NamedTuple

class TurbulenceStatisticsSetup(NamedTuple):
    is_cumulative: bool
    is_logging: bool
    case: str
    start_sampling: float
    sampling_dt: float
    sampling_step: int
    save_dt: int
    streamwise_measure_position: float


class StatisticsSetup(NamedTuple):
    turbulence: TurbulenceStatisticsSetup