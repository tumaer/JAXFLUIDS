from typing import NamedTuple
import numpy as np

class GeneralSetup(NamedTuple):
    case_name: str
    end_time: float
    end_step: int
    save_path: str
    save_dt: float
    save_step: int
    save_timestamps: np.ndarray
    save_start: float
