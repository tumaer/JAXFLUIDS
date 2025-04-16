from typing import Any, Dict, NamedTuple, Tuple
import jax
import jax.numpy as jnp 

class ConvectiveFluxesSetup(NamedTuple):
    flux_function: Any = None
    cell_face_reconstruction: Any = None

class LevelSetSetup(NamedTuple):
    fluid_fluid: Any = None
    fluid_solid: Any = None

class BoundaryConditionsFace(NamedTuple):
    bounding_domain: Any = None
    primitives: Any = None
    levelset: Any = None
    wall_velocity: Any = None
    wall_temperature: Any = None
    wall_mass_transfer: Any = None

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

class BaseSetup(NamedTuple):
    convective_fluxes: ConvectiveFluxesSetup = None
    levelset: LevelSetSetup = None
    boundary_conditions: BoundaryConditionSetup = None

class ParametersSetup(BaseSetup):
    pass

class CallablesSetup(BaseSetup):
    pass

class MachineLearningSetup(NamedTuple):
    callables: CallablesSetup = None
    parameters: ParametersSetup = None

def combine_callables_and_params(
        callables_setup: CallablesSetup,
        parameters_setup: ParametersSetup
    ) -> MachineLearningSetup:
    return MachineLearningSetup(
        callables_setup,
        parameters_setup
    )