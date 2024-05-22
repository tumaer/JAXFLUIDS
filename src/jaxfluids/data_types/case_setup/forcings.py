from collections import namedtuple
from typing import NamedTuple, Callable

import jax.numpy as jnp
from jax import Array

class AcousticForcingSetup(NamedTuple):
    type: str
    axis: str
    plane_value: float
    forcing: Callable

class GeometricSourceSetup(NamedTuple):
    symmetry_type: str
    symmetry_axis: str
    
class ForcingSetup(NamedTuple):
    gravity: Array
    mass_flow_target: Callable
    mass_flow_direction: str
    temperature_target: Callable
    hit_forcing_cutoff: int
    geometric_source: GeometricSourceSetup
    acoustic_forcing: AcousticForcingSetup
    custom_forcing: NamedTuple