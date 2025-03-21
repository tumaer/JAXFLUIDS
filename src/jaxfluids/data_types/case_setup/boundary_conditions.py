from typing import NamedTuple, Tuple, Callable, List

import jax
import jax.numpy as jnp

Array = jax.Array

class PrimitivesTable(NamedTuple):
    primitives: Array = None
    axis_values: Array = None
    axis_name: Array = None

class VelocityCallable(NamedTuple):
    u: Callable = None
    v: Callable = None
    w: Callable = None

class WallMassTransferSetup(NamedTuple):
    primitives_callable: NamedTuple = None
    bounding_domain_callable: Callable = None

class SimpleInflowSetup(NamedTuple):
    primitives_callable: NamedTuple = None

class SimpleOutflowSetup(NamedTuple):
    primitives_callable: NamedTuple = None

class VelocityTuple(NamedTuple):
    u: float
    v: float
    w: float

class BoundaryConditionsFace(NamedTuple):
    boundary_type: str = None
    bounding_domain_callable: Callable = None
    primitives_callable: NamedTuple = None
    levelset_callable: Callable = None
    wall_velocity_callable: VelocityCallable = None
    wall_temperature_callable: Callable = None
    wall_mass_transfer: WallMassTransferSetup = None
    primitives_table: PrimitivesTable = None
    simple_inflow: SimpleInflowSetup = None
    simple_outflow: SimpleOutflowSetup = None
    temperature_callable: Callable = None # NOTE for coupled solids

class BoundaryConditionsField(NamedTuple):
    east: Tuple[BoundaryConditionsFace] = None
    west: Tuple[BoundaryConditionsFace] = None
    north: Tuple[BoundaryConditionsFace] = None
    south: Tuple[BoundaryConditionsFace] = None
    top: Tuple[BoundaryConditionsFace] = None
    bottom: Tuple[BoundaryConditionsFace] = None

class BoundaryConditionSetup(NamedTuple):
    primitives: BoundaryConditionsField = None
    levelset: BoundaryConditionsField = None
    solids: BoundaryConditionsField = None