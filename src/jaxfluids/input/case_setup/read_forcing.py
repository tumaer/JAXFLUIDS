from typing import Dict
import warnings

import numpy as np
from jaxfluids.data_types.case_setup.forcings import *
from jaxfluids.data_types.case_setup import DomainSetup
from jaxfluids.data_types.case_setup import GetPrimitivesCallable
from jaxfluids.data_types.numerical_setup import NumericalSetup, ActiveForcingsSetup, ActivePhysicsSetup
from jaxfluids.equation_information import EquationInformation
from jaxfluids.forcing import TUPLE_ACOUSTIC_FORCING
from jaxfluids.input.setup_reader import create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value, get_path_to_key, loop_fields
from jaxfluids.solvers import TUPLE_GEOMETRIC_SOURCES
from jaxfluids.unit_handler import UnitHandler

FORCING_KEYS = (
    "gravity", "mass_flow", "temperature",
    "hit_forcing_cutoff", "geometric_source",
    "acoustic_forcing", "custom_forcing",
    "sponge_layer", "enthalpy_damping"
)

def read_forcing_setup(
        case_setup_dict: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler,
        domain_setup: DomainSetup
        ) -> ForcingSetup:
    """Reads the forcing properties in the
    case setup .json file and creates
    the corresponding containers.
    """

    active_physics = numerical_setup.active_physics
    active_forcings = numerical_setup.active_forcings

    is_optional = not any([getattr(active_forcings, field) for field in active_forcings._fields])
    forcings_setup: Dict = get_setup_value(
        case_setup_dict, "forcings", "forcings", dict,
        default_value={}, is_optional=is_optional)

    for key in forcings_setup:
        if key not in FORCING_KEYS:
            path = get_path_to_key("forcings", key)
            warning_string = (
                "While reading the case setup file, "
                f"the following unknown key was encountered: {path}. "
                "This key will be ignored. "
            )
            warnings.warn(warning_string, RuntimeWarning)

    gravity = read_gravity(
        forcings_setup, active_physics,
        unit_handler)

    geometric_source = read_geometric_source(
        forcings_setup, active_physics,
        domain_setup)

    mass_flow_forcing_setup = read_mass_flow_forcing(
        forcings_setup, active_forcings,
        domain_setup, unit_handler)

    temperature_forcing_setup = read_temperature_forcing(
        forcings_setup, active_forcings,
        domain_setup, unit_handler)

    hit_forcing_cutoff = read_hit_forcing(
        forcings_setup, active_forcings)

    acoustic_forcing_setup = read_acoustic_forcing(
        forcings_setup, active_forcings,
        domain_setup, unit_handler)

    custom_forcing_callable = read_custom_forcing(
        forcings_setup, active_forcings,
        domain_setup, equation_information,
        unit_handler)

    sponge_layer_setup = read_sponge_layer(
        forcings_setup, active_forcings,
        domain_setup, equation_information,
        unit_handler)

    enthalpy_damping_setup = read_enthalpy_damping(
        forcings_setup, active_forcings)

    forcing_setup = ForcingSetup(
        gravity=gravity,
        mass_flow_forcing=mass_flow_forcing_setup,
        temperature_forcing=temperature_forcing_setup,
        hit_forcing_cutoff=hit_forcing_cutoff,
        geometric_source=geometric_source,
        acoustic_forcing=acoustic_forcing_setup,
        custom_forcing=custom_forcing_callable,
        sponge_layer=sponge_layer_setup,
        enthalpy_damping=enthalpy_damping_setup)

    return forcing_setup


def read_gravity(
        forcings_setup: Dict,
        active_physics: ActivePhysicsSetup,
        unit_handler: UnitHandler
        ) -> Array:
    
    is_optional = not active_physics.is_volume_force
    path = get_path_to_key("forcings", "gravity")
    gravity = get_setup_value(
        forcings_setup, "gravity", path, list,
        is_optional=is_optional, default_value=(0.0,0.0,0.0))
    gravity = np.array(gravity)
    gravity = unit_handler.non_dimensionalize(gravity, "gravity")

    return gravity

def read_mass_flow_forcing(
        forcings_setup: Dict,
        active_forcings: ActiveForcingsSetup,
        domain_setup: DomainSetup,
        unit_handler: UnitHandler
        ) -> MassFlowForcingSetup:

    is_optional = not active_forcings.is_mass_flow_forcing
    base_path = get_path_to_key("forcings", "mass_flow")
    mass_flow_setup = get_setup_value(
        forcings_setup, "mass_flow", base_path, dict,
        default_value={}, is_optional=is_optional)

    path = get_path_to_key(base_path, "target_value")
    callable_case_setup = get_setup_value(
        mass_flow_setup, "target_value", path, (float, str),
        is_optional=is_optional, default_value=0.0)
    input_argument_labels = tuple(["t"])
    input_argument_units = tuple(["time"])
    target_value = create_wrapper_for_callable(
        callable_case_setup, input_argument_units,
        input_argument_labels, "mass_flow", path,
        perform_nondim=True, unit_handler=unit_handler)

    path = get_path_to_key(base_path, "direction")
    direction = get_setup_value(
        mass_flow_setup, "direction", path, str,
        is_optional=is_optional, default_value="x",
        possible_string_values=domain_setup.active_axes)

    mass_flow_forcing = MassFlowForcingSetup(
        target_value, direction
    )

    return mass_flow_forcing

def read_temperature_forcing(
        forcings_setup: Dict,
        active_forcings: ActiveForcingsSetup,
        domain_setup: DomainSetup,
        unit_handler: UnitHandler
        ) -> TemperatureForcingSetup:
    
    active_axes = domain_setup.active_axes
    dim = domain_setup.dim

    input_argument_labels = active_axes + ("t",)
    input_argument_units = tuple(["length"] * dim + ["time"])

    is_temperature_forcing = active_forcings.is_temperature_forcing
    is_turb_hit_forcing = active_forcings.is_turb_hit_forcing
    is_solid_temperature_forcing = active_forcings.is_solid_temperature_forcing
    is_optional = not any((
        is_temperature_forcing,
        is_turb_hit_forcing,
        is_solid_temperature_forcing))

    base_path = get_path_to_key("forcings", "temperature")
    temperature_setup = get_setup_value(
        forcings_setup, "temperature", base_path, dict,
        default_value={}, is_optional=is_optional)

    # NOTE Fluid temperature forcing
    flag = not any((is_temperature_forcing, is_turb_hit_forcing))
    path = get_path_to_key(base_path, "target_value")
    callable_case_setup = get_setup_value(
        temperature_setup, "target_value", path, (float, str),
        is_optional=flag, default_value=1.0,
        numerical_value_condition=(">", 0.0))
    temperature_target_wrapper = create_wrapper_for_callable(
        callable_case_setup, input_argument_units,
        input_argument_labels, "temperature", path, perform_nondim=True,
        unit_handler=unit_handler, is_scalar=True)
    
    # NOTE Solid temperature forcing
    path = get_path_to_key(base_path, "solid_target_value")
    callable_case_setup = get_setup_value(
        temperature_setup, "solid_target_value", path, (float, str),
        is_optional=not is_solid_temperature_forcing, default_value=1.0,
        numerical_value_condition=(">", 0.0))
    solid_temperature_target_wrapper = create_wrapper_for_callable(
        callable_case_setup, input_argument_units,
        input_argument_labels, "temperature", path, perform_nondim=True,
        unit_handler=unit_handler, is_scalar=True)
    
    path = get_path_to_key(base_path, "solid_target_mask")
    callable_case_setup = get_setup_value(
        temperature_setup, "solid_target_mask", path, (float, str),
        is_optional=True, default_value=1.0,
        numerical_value_condition=(">", 0.0))
    solid_temperature_target_mask_wrapper = create_wrapper_for_callable(
        callable_case_setup, input_argument_units,
        input_argument_labels, "None", path, perform_nondim=True,
        unit_handler=unit_handler, is_scalar=True)

    temperature_forcing_setup = TemperatureForcingSetup(
        temperature_target_wrapper,
        solid_temperature_target_wrapper,
        solid_temperature_target_mask_wrapper,)

    return temperature_forcing_setup

def read_hit_forcing(
        forcings_setup: Dict,
        active_forcings: ActiveForcingsSetup
        ) -> int:

    is_optional = not active_forcings.is_turb_hit_forcing
    path = get_path_to_key("forcings", "hit_forcing_cutoff")
    hit_forcing_cutoff = get_setup_value(
        forcings_setup, "hit_forcing_cutoff", path, int,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">", 0.0))
    
    return hit_forcing_cutoff

def read_geometric_source(
        forcings_setup: Dict,
        active_physics: ActivePhysicsSetup,
        domain_setup: DomainSetup
        ) -> GeometricSourceSetup | None:

    is_optional = not active_physics.is_geometric_source
    base_path = get_path_to_key("forcings", "geometric_source")
    geometric_source_setup = get_setup_value(
        forcings_setup, "geometric_source", base_path,
        dict, is_optional=is_optional)
    
    if not is_optional:
        path = get_path_to_key(base_path, "symmetry_type")
        symmetry_type = get_setup_value(
            geometric_source_setup, "symmetry_type", path, str,
            is_optional=False, possible_string_values=TUPLE_GEOMETRIC_SOURCES)
        
        path = get_path_to_key(base_path, "symmetry_axis")
        symmetry_axis = get_setup_value(
            geometric_source_setup, "symmetry_axis", path, str,
            is_optional=False, possible_string_values=domain_setup.active_axes)
        
        geometric_source = GeometricSourceSetup(symmetry_type, symmetry_axis)
    
    else:
        geometric_source = None
    
    return geometric_source

def read_acoustic_forcing(
        forcings_setup: Dict, 
        active_forcings: ActiveForcingsSetup,
        domain_setup: DomainSetup,
        unit_handler: UnitHandler
        ) -> AcousticForcingSetup:

    is_optional = not active_forcings.is_acoustic_forcing
    base_path = get_path_to_key("forcings", "acoustic_forcing")
    acoustic_forcing_case_setup = get_setup_value(
        forcings_setup, "acoustic_forcing", base_path, dict,
        default_value={}, is_optional=is_optional)

    path = get_path_to_key(base_path, "type")
    acoustic_forcing_type = get_setup_value(
        acoustic_forcing_case_setup, "type", path, str,
        is_optional=is_optional, default_value="PLANAR",
        possible_string_values=TUPLE_ACOUSTIC_FORCING)

    path = get_path_to_key(base_path, "axis")
    acoustic_forcing_axis = get_setup_value(
        acoustic_forcing_case_setup, "axis", path, str,
        is_optional=is_optional, default_value="x",
        possible_string_values=domain_setup.active_axes)
    
    path = get_path_to_key(base_path, "plane_value")
    acoustic_forcing_plane_value = get_setup_value(
        acoustic_forcing_case_setup, "plane_value", path, float,
        is_optional=is_optional, default_value=0.0)
    if not is_optional:
        acoustic_forcing_plane_value = unit_handler.non_dimensionalize(
            acoustic_forcing_plane_value, "length")

    path = get_path_to_key(base_path, "forcing")
    acoustic_forcing_callable_case_setup = get_setup_value(
        acoustic_forcing_case_setup, "forcing", path, (float, str),
        is_optional=is_optional, default_value=0.0)
    if not is_optional:
        input_argument_labels = tuple(["t"])
        input_argument_units = tuple(["time"])
        acoustic_forcing_wrapper = create_wrapper_for_callable(
            acoustic_forcing_callable_case_setup, input_argument_units,
            input_argument_labels, "pressure", path,
            perform_nondim=True, unit_handler=unit_handler)
    else:
        acoustic_forcing_wrapper = None

    acoustic_forcing_setup = AcousticForcingSetup(
        type=acoustic_forcing_type,
        axis=acoustic_forcing_axis,
        plane_value=acoustic_forcing_plane_value,
        forcing=acoustic_forcing_wrapper)
    
    return acoustic_forcing_setup

def read_custom_forcing(
        forcings_setup: Dict,
        active_forcings: ActiveForcingsSetup,
        domain_setup: DomainSetup,
        equation_information: EquationInformation,
        unit_handler: UnitHandler
        ) -> Callable:

    is_optional = not active_forcings.is_custom_forcing
    base_path = get_path_to_key("forcings", "custom_forcing")
    custom_forcing_setup = get_setup_value(
        forcings_setup, "custom_forcing",
        base_path, dict, is_optional=is_optional,
        default_value=None)

    if custom_forcing_setup is not None:
        active_axes = domain_setup.active_axes
        dim = domain_setup.dim

        input_argument_labels = tuple(active_axes) + ("t",)
        input_argument_units = tuple(["length"] * dim + ["time"])

        custom_forcing_callables_dict = {}
        for prime_state in equation_information.primes_tuple:
            path = get_path_to_key(base_path, prime_state)
            prime_state_case_setup = get_setup_value(
                custom_forcing_setup, prime_state, path, (float, str),
                is_optional=False)
            # For velocities, we need to normalize by momentum 
            output_unit = f"rho{prime_state}" if prime_state in ("u", "v", "w") else prime_state
            forcing_wrapper = create_wrapper_for_callable(
                prime_state_case_setup, input_argument_units,
                input_argument_labels, output_unit,
                path, perform_nondim=True, is_temporal_derivative=True,
                unit_handler=unit_handler)
            custom_forcing_callables_dict[prime_state] = forcing_wrapper
        custom_forcing_callable = GetPrimitivesCallable(custom_forcing_callables_dict)
    else:
        custom_forcing_callable = None

    return custom_forcing_callable

def read_sponge_layer(
        forcings_setup: Dict, 
        active_forcings: ActiveForcingsSetup,
        domain_setup: DomainSetup,
        equation_information: EquationInformation,
        unit_handler: UnitHandler
    ) -> SpongeLayerSetup | None:

    is_optional = not active_forcings.is_sponge_layer_forcing
    base_path = get_path_to_key("forcings", "sponge_layer")
    sponge_layer_setup_dict = get_setup_value(
        forcings_setup, "sponge_layer",
        base_path, dict, is_optional=is_optional,
        default_value=None)
    
    if sponge_layer_setup_dict is not None:
        active_axes = domain_setup.active_axes
        dim = domain_setup.dim

        input_argument_labels = tuple(active_axes) + ("t",)
        input_argument_units = tuple(["length"] * dim + ["time"])

        path_primitives = get_path_to_key(base_path, "primitives")
        primitives_dict = get_setup_value(
            sponge_layer_setup_dict, "primitives",
            base_path, dict, is_optional=False)

        primitives_callable_dict = {}
        for prime_state in equation_information.primes_tuple:
            path = get_path_to_key(path_primitives, prime_state)
            prime_state_case_setup = get_setup_value(
                primitives_dict, prime_state, path, (float, str),
                is_optional=False)
            forcing_wrapper = create_wrapper_for_callable(
                prime_state_case_setup, input_argument_units,
                input_argument_labels, prime_state,
                path, perform_nondim=True, unit_handler=unit_handler)
            primitives_callable_dict[prime_state] = forcing_wrapper
        primitives_callable = GetPrimitivesCallable(primitives_callable_dict)

        path_strength = get_path_to_key(base_path, "strength")
        strength_case_setup = get_setup_value(
            sponge_layer_setup_dict, "strength", path_strength, (float, str),
            is_optional=False)
        strength_callable = create_wrapper_for_callable(
            strength_case_setup, input_argument_units,
            input_argument_labels, "None",
            path_strength, perform_nondim=True,
            unit_handler=unit_handler)
        
        sponge_layer_setup = SpongeLayerSetup(
            primitives_callable, strength_callable)
    else:
        sponge_layer_setup = None

    return sponge_layer_setup

def read_enthalpy_damping(
        forcings_setup: Dict, 
        active_forcings: ActiveForcingsSetup
        ) -> EnthalpyDampingSetup | None:
    
    is_optional = not active_forcings.is_enthalpy_damping
    base_path = get_path_to_key("forcings", "enthalpy_damping")
    enthalpy_damping_setup_dict: Dict = get_setup_value(
        forcings_setup, "enthalpy_damping",
        base_path, dict, is_optional=is_optional,
        default_value=None) 

    if enthalpy_damping_setup_dict is not None:
        
        path = get_path_to_key(base_path, "type")
        enthalpy_damping_type = get_setup_value(
            enthalpy_damping_setup_dict, "type", path, str,
            is_optional=False)
        
        path = get_path_to_key(base_path, "alpha")
        alpha = get_setup_value(
            enthalpy_damping_setup_dict, "alpha", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        path = get_path_to_key(base_path, "H_infty")
        H_infty = get_setup_value(
            enthalpy_damping_setup_dict, "H_infty", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        enthalpy_damping_setup = EnthalpyDampingSetup(
            enthalpy_damping_type, alpha, H_infty)

    else:
        enthalpy_damping_setup = None

    return enthalpy_damping_setup
