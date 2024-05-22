from typing import Dict
import numpy as np
from jaxfluids.data_types.case_setup.forcings import *
from jaxfluids.data_types.case_setup import DomainSetup
from jaxfluids.data_types.case_setup import GetPrimitivesCallable
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.equation_information import EquationInformation
from jaxfluids.forcing import TUPLE_ACOUSTIC_FORCING
from jaxfluids.input.setup_reader import create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value, get_path_to_key, loop_fields
from jaxfluids.solvers import TUPLE_GEOMETRIC_SOURCES
from jaxfluids.unit_handler import UnitHandler

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
    active_axes = domain_setup.active_axes
    dim = domain_setup.dim

    basepath = "forcings"

    active_physics = numerical_setup.active_physics
    active_forcings = numerical_setup.active_forcings

    is_optional = not any([getattr(active_forcings, field) for field in active_forcings._fields])
    forcings_setup = get_setup_value(
        case_setup_dict, "forcings", basepath, dict,
        default_value={}, is_optional=is_optional)

    # Gravity
    is_optional = not active_physics.is_volume_force
    path = get_path_to_key(basepath, "gravity")
    gravity = get_setup_value(
        forcings_setup, "gravity", path, list,
        is_optional=is_optional, default_value=(0.0,0.0,0.0))
    gravity = np.array(gravity)
    gravity = unit_handler.non_dimensionalize(gravity, "gravity")

    # Geometric source
    is_optional = not active_physics.is_geometric_source
    path_to_geometric_source = get_path_to_key(basepath, "geometric_source")
    geometric_source_setup = get_setup_value(
        forcings_setup, "geometric_source", path_to_geometric_source, dict,
        default_value=GeometricSourceSetup, is_optional=is_optional)
    
    if not is_optional:
        path = get_path_to_key(path_to_geometric_source, "symmetry_type")
        symmetry_type = get_setup_value(
            geometric_source_setup, "symmetry_type", path, str,
            is_optional=False, possible_string_values=TUPLE_GEOMETRIC_SOURCES)
        
        path = get_path_to_key(path_to_geometric_source, "symmetry_axis")
        symmetry_axis = get_setup_value(
            geometric_source_setup, "symmetry_axis", path, str,
            is_optional=False, possible_string_values=active_axes)
        
        geometric_source = GeometricSourceSetup(symmetry_type, symmetry_axis)
    
    else:
        geometric_source = None

    # Mass flow forcing
    is_optional = not active_forcings.is_mass_flow_forcing
    path = get_path_to_key(basepath, "mass_flow_target")
    mass_flow_target_callable_case_setup = get_setup_value(
        forcings_setup, "mass_flow_target", path, (float, str),
        is_optional=is_optional, default_value=0.0)
    input_argument_labels = tuple(["t"])
    input_argument_units = tuple(["time"])
    mass_flow_target_wrapper = create_wrapper_for_callable(
        mass_flow_target_callable_case_setup, input_argument_units,
        input_argument_labels, "mass_flow", path,
        perform_nondim=True, unit_handler=unit_handler)

    path = get_path_to_key(basepath, "mass_flow_direction")
    mass_flow_direction = get_setup_value(
        forcings_setup, "mass_flow_direction", path, str,
        is_optional=is_optional, default_value=0,
        possible_string_values=active_axes)

    is_temperature_forcing = active_forcings.is_temperature_forcing
    is_turb_hit_forcing = active_forcings.is_turb_hit_forcing
    flag = not any((is_temperature_forcing, is_turb_hit_forcing))
    path = get_path_to_key(basepath, "temperature_target")
    temperature_target_callable_case_setup = get_setup_value(
        forcings_setup, "temperature_target", path, (float, str),
        is_optional=flag, default_value=1.0,
        numerical_value_condition=(">", 0.0))
    input_argument_labels = active_axes + ("t",)
    input_argument_units = tuple(["length"] * dim + ["time"])
    temperature_target_wrapper = create_wrapper_for_callable(
        temperature_target_callable_case_setup, input_argument_units,
        input_argument_labels, "temperature", path, perform_nondim=True,
        unit_handler=unit_handler, is_scalar=True)

    is_optional = not active_forcings.is_turb_hit_forcing
    path = get_path_to_key(basepath, "hit_forcing_cutoff")
    hit_forcing_cutoff = get_setup_value(
        forcings_setup, "hit_forcing_cutoff", path, int,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">", 0.0))

    # Acoustic forcing
    is_optional = not active_forcings.is_acoustic_forcing
    acoustic_forcing_basepath = get_path_to_key(basepath, "acoustic_forcing")
    acoustic_forcing_case_setup = get_setup_value(
        forcings_setup, "acoustic_forcing", acoustic_forcing_basepath, dict,
        default_value={}, is_optional=is_optional)

    path = get_path_to_key(acoustic_forcing_basepath, "type")
    acoustic_forcing_type = get_setup_value(
        acoustic_forcing_case_setup, "type", path, str,
        is_optional=is_optional, default_value="PLANAR",
        possible_string_values=TUPLE_ACOUSTIC_FORCING)

    path = get_path_to_key(acoustic_forcing_basepath, "axis")
    acoustic_forcing_axis = get_setup_value(
        acoustic_forcing_case_setup, "axis", path, str,
        is_optional=is_optional, default_value="x",
        possible_string_values=active_axes)
    
    path = get_path_to_key(acoustic_forcing_basepath, "plane_value")
    acoustic_forcing_plane_value = get_setup_value(
        acoustic_forcing_case_setup, "plane_value", path, float,
        is_optional=is_optional, default_value=0.0)
    if not is_optional:
        acoustic_forcing_plane_value = unit_handler.non_dimensionalize(
            acoustic_forcing_plane_value, "length")

    path = get_path_to_key(acoustic_forcing_basepath, "forcing")
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

    # Custom forcing
    is_optional = not active_forcings.is_custom_forcing
    path_custom_forcing = get_path_to_key(basepath, "custom_forcing")
    custom_forcing_setup = get_setup_value(
        forcings_setup, "custom_forcing",
        path_custom_forcing, dict, is_optional=is_optional,
        default_value=None)

    if custom_forcing_setup is not None:
        active_axes = domain_setup.active_axes
        dim = domain_setup.dim

        input_argument_labels = tuple(active_axes) + ("t",)
        input_argument_units = tuple(["length"] * dim + ["time"])

        custom_forcing_callables_dict = {}
        for prime_state in equation_information.primes_tuple:
            path = get_path_to_key(path_custom_forcing, prime_state)
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

    forcing_setup = ForcingSetup(
        gravity=gravity,
        mass_flow_target=mass_flow_target_wrapper,
        mass_flow_direction=mass_flow_direction,
        temperature_target=temperature_target_wrapper,
        hit_forcing_cutoff=hit_forcing_cutoff,
        geometric_source=geometric_source,
        acoustic_forcing=acoustic_forcing_setup,
        custom_forcing=custom_forcing_callable)

    return forcing_setup
