from typing import NamedTuple

class RestartSetup(NamedTuple):
    flag: bool
    file_path: str
    use_time: bool
    time: float
    is_equal_decomposition: bool