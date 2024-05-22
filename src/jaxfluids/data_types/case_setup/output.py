from typing import Tuple, NamedTuple

class OutputQuantitiesSetup(NamedTuple):
    primitives: Tuple[str]
    conservatives: Tuple[str]
    real_fluid: Tuple[str]
    levelset: Tuple[str]
    miscellaneous: Tuple[str]
    forcings: Tuple[str]
