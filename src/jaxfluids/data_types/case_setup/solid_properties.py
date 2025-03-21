from typing import NamedTuple, Tuple, Callable

class VelocityCallable(NamedTuple):
    u: Callable = None
    v: Callable = None
    w: Callable = None

class SolidVelocityBlock(NamedTuple):
    velocity_callable: VelocityCallable
    bounding_domain_callable: Callable

class SolidVelocitySetup(NamedTuple):
    blocks: Tuple[SolidVelocityBlock]
    velocity_callable: VelocityCallable
    is_blocks: bool
    is_callable: bool

class SolidPropertiesSetup(NamedTuple):
    velocity: SolidVelocitySetup
    temperature: Callable
    density: float
    thermal_conductivity: Callable
    specific_heat_capacity: float