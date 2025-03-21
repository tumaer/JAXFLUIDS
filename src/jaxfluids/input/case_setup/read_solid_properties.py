from typing import Dict
import warnings

from jaxfluids.data_types.case_setup.solid_properties import *
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup import DomainSetup
from jaxfluids.equation_information import EquationInformation
from jaxfluids.input.setup_reader import get_path_to_key, create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value, loop_fields
from jaxfluids.unit_handler import UnitHandler

def read_solid_properties_setup(
        case_setup_dict: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler,
        domain_setup: DomainSetup
        ) -> SolidPropertiesSetup:
    """Reads the solid properties in the
    case setup .json file and creates
    the corresponding containers.

    :return: _description_
    :rtype: SolidPropertiesSetup
    """

    active_axes = domain_setup.active_axes
    dim = domain_setup.dim
    input_argument_labels = active_axes + ("t",)
    input_argument_units = tuple(["length"] * dim + ["time"])

    levelset_model = equation_information.levelset_model
    solid_coupling = equation_information.solid_coupling

    basepath = "solid_properties"

    solid_coupling = equation_information.solid_coupling
    if any((solid_coupling.dynamic, solid_coupling.thermal)):
        is_optional = False
    else:
        is_optional = True

    solid_properties_case_setup = get_setup_value(
        case_setup_dict, "solid_properties", basepath, dict,
        is_optional=is_optional, default_value={})

    # SOLID VELOCITY
    is_optional = False if solid_coupling.dynamic == "ONE-WAY" else True
    path_to_velocity = get_path_to_key(basepath, "velocity")
    solid_velocity_case_setup = get_setup_value(
        solid_properties_case_setup, "velocity", path_to_velocity, (list, dict),
        is_optional=is_optional, default_value=None)

    if solid_velocity_case_setup is not None:

        def _read_velocity_callable(
                velocity_callable_case_setup: Dict,
                basepath: str
                ) -> VelocityCallable:
            """Wrapper to read the velocity callable
            from the case setup .json file and create jaxfluids
            container.

            :param solid_velocity_case_setup: _description_
            :type solid_velocity_case_setup: Dict
            :param path_to_velocity: _description_
            :type path_to_velocity: str
            :return: _description_
            :rtype: VelocityCallable
            """
            velocity_callables_dict = {}
            for velocity_xi in ("u","v","w"):
                path = get_path_to_key(basepath, velocity_xi)
                velocity_xi_callable_case_setup = get_setup_value(
                    velocity_callable_case_setup, velocity_xi, path, (float, str),
                    is_optional=False)
                velocity_wrapper = create_wrapper_for_callable(
                    velocity_xi_callable_case_setup, input_argument_units,
                    input_argument_labels, "velocity", path, 
                    perform_nondim=True, unit_handler=unit_handler)
                velocity_callables_dict[velocity_xi] = velocity_wrapper
            velocity_callable = VelocityCallable(**velocity_callables_dict)
            return velocity_callable

        if isinstance(solid_velocity_case_setup, list):
            solid_velocity_block_list = []
            for solid_velocity_block_case in solid_velocity_case_setup:
                velocity_callable = _read_velocity_callable(
                    solid_velocity_block_case, path_to_velocity)
                path = get_path_to_key(path_to_velocity, "bounding_domain")
                bounding_domain_callable_case_setup = get_setup_value(
                    solid_velocity_block_case, "bounding_domain", path, (float, str),
                    is_optional=False)
                bounding_domain_callable = create_wrapper_for_callable(
                    bounding_domain_callable_case_setup,
                    input_argument_units, input_argument_labels,
                    "None", path, perform_nondim=True,
                    unit_handler=unit_handler)
                solid_velocity_block = SolidVelocityBlock(
                    velocity_callable, bounding_domain_callable)
                solid_velocity_block_list.append(solid_velocity_block)
            solid_velocity_blocks = tuple(solid_velocity_block_list)
            is_blocks = True
            is_callable = False
            velocity_callable = None

        else:
            velocity_callable = _read_velocity_callable(
                solid_velocity_case_setup, path_to_velocity)
            solid_velocity_blocks = None
            is_callable = True
            is_blocks = False
        
        solid_velocity_setup = SolidVelocitySetup(
            solid_velocity_blocks,
            velocity_callable,
            is_blocks, is_callable)
        
    else:

        solid_velocity_setup = SolidVelocitySetup(
            None, None, False, False)

    if solid_coupling.dynamic == "TWO-WAY":
        warning_string = (f"solid velocity will not be used.")
        warnings.warn(warning_string, RuntimeWarning)

    # SOLID TEMPERATURE
    is_optional = False if solid_coupling.thermal == "ONE-WAY" else True
    path = get_path_to_key(basepath, "temperature")
    solid_temperature_case_setup = get_setup_value(
        solid_properties_case_setup, "temperature", path, (str, float),
        is_optional=is_optional, default_value=1.0)

    solid_temperature = create_wrapper_for_callable(
        solid_temperature_case_setup, input_argument_units, input_argument_labels,
        "temperature", path, perform_nondim=True, unit_handler=unit_handler)

    if solid_coupling.thermal == "TWO-WAY":
        warning_string = (f"solid velocity will not be used.")
        warnings.warn(warning_string, RuntimeWarning)

    # SOLID DENSITY
    if any((solid_coupling.dynamic == "TWO-WAY", solid_coupling.thermal == "TWO-WAY")):
        is_optional = False
    else:
        is_optional = True
    path = get_path_to_key(basepath, "density")
    solid_density_case_setup = get_setup_value(
        solid_properties_case_setup, "density", path, float,
        is_optional=is_optional, default_value=1.0,
        numerical_value_condition=(">", 0.0))
    solid_density = unit_handler.non_dimensionalize(solid_density_case_setup, "density")

    # THERMAL CONDUCTIVITY
    input_argument_labels = active_axes + ("T",)
    input_argument_units = tuple(["length"] * dim + ["temperature"])

    is_optional = False if solid_coupling.thermal == "TWO-WAY" else True
    path = get_path_to_key(basepath, "thermal_conductivity")
    thermal_conductivity_case_setup = get_setup_value(
        solid_properties_case_setup, "thermal_conductivity", path, (float, str),
        is_optional=is_optional, default_value=1.0,
        numerical_value_condition=(">=", 0.0))
    thermal_conductivity_callable = create_wrapper_for_callable(
        thermal_conductivity_case_setup,
        input_argument_units, input_argument_labels,
        "thermal_conductivity", path, perform_nondim=True,
        unit_handler=unit_handler, is_scalar=True)
    

    # SPECIFIC HEAT CAPACITY
    is_optional = False if solid_coupling.thermal == "TWO-WAY" else True
    path = get_path_to_key(basepath, "specific_heat_capacity")
    specific_heat_capacity_case_setup = get_setup_value(
        solid_properties_case_setup, "specific_heat_capacity", path, float,
        is_optional=is_optional, default_value=1.0,
        numerical_value_condition=(">", 0.0))
    specific_heat_capacity = unit_handler.non_dimensionalize(specific_heat_capacity_case_setup, "specific_heat_capacity")

    solid_properties = SolidPropertiesSetup(
        solid_velocity_setup, solid_temperature,
        solid_density, thermal_conductivity_callable,
        specific_heat_capacity)

    return solid_properties
