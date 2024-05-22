from typing import NamedTuple

class TurbulenceStatisticsSetup(NamedTuple):
    is_active: bool
    turbulence_case: str
    start_sampling: float
    sampling_dt: float
    sampling_step: int
    save_dt: int
    streamwise_measure_position: float