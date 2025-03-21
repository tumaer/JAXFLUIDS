from typing import NamedTuple, Tuple

class OptimizerSetup(NamedTuple):
    optimizer: str
    scheduler: str
    init_value: float
    end_value: float = None
    power: int = None
    transition_begin: int = None
    transition_steps: int = None
    decay_rate: float = None
    boundaries_and_scales: Tuple[Tuple[int, float]] = None