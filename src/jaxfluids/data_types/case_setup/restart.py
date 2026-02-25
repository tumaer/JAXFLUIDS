from typing import NamedTuple

class RestartSetup(NamedTuple):
    is_restart: bool
    file_path: str
    is_reset_time: bool
    time: float
    is_equal_decomposition_multihost: bool
    is_interpolate: bool
    numerical_setup_path: str
    case_setup_path: str
