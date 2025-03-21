from typing import NamedTuple

class RestartSetup(NamedTuple):
    flag: bool
    file_path: str
    use_time: bool
    time: float
    is_equal_decomposition_multihost: bool
    is_interpolate: bool
    numerical_setup_path: str
    case_setup_path: str
