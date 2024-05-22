from typing import NamedTuple

class NondimensionalizationParameters(NamedTuple):
    density_reference: float = 1.0
    length_reference: float = 1.0
    velocity_reference: float = 1.0
    temperature_reference: float = 1.0
