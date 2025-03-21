from collections import namedtuple
from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp

Array = jax.Array

class MassFlowForcingSetup(NamedTuple):
    target_value: Callable
    direction: str

class TemperatureForcingSetup(NamedTuple):
    target_value: Callable
    solid_target_value: Callable
    solid_target_mask: Callable

class AcousticForcingSetup(NamedTuple):
    type: str
    axis: str
    plane_value: float
    forcing: Callable

class GeometricSourceSetup(NamedTuple):
    symmetry_type: str
    symmetry_axis: str

class SpongeLayerSetup(NamedTuple):
    primitives: NamedTuple
    strength: Callable

class EnthalpyDampingSetup(NamedTuple):
    type: str
    alpha: float
    H_infty: float

class ForcingSetup(NamedTuple):
    gravity: Array
    mass_flow_forcing: MassFlowForcingSetup
    temperature_forcing: TemperatureForcingSetup
    hit_forcing_cutoff: int
    geometric_source: GeometricSourceSetup
    acoustic_forcing: AcousticForcingSetup
    custom_forcing: NamedTuple
    sponge_layer: SpongeLayerSetup
    enthalpy_damping: EnthalpyDampingSetup