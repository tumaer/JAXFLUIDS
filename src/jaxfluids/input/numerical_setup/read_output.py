from typing import Dict

from jaxfluids.stencils import DICT_FIRST_DERIVATIVE_CENTER
from jaxfluids.data_types.numerical_setup.output import *
from jaxfluids.data_types.numerical_setup import ConservativesSetup
from jaxfluids.input.numerical_setup import get_setup_value, loop_fields, get_path_to_key
from jaxfluids.io_utils import TUPLE_LOGGING_MODES

def read_output_setup(
        numerical_setup_dict: Dict,
        conservatives_setup: ConservativesSetup
        ) -> OutputSetup:

    basepath = "output"

    output_dict = get_setup_value(
        numerical_setup_dict, "output", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "derivative_stencil")
    derivative_stencil_str = get_setup_value(
        output_dict, "derivative_stencil", path, str,
        is_optional=True, default_value="CENTRAL4",
        possible_string_values=tuple(DICT_FIRST_DERIVATIVE_CENTER.keys()))
    derivative_stencil = DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_str]
    halo_cells = conservatives_setup.halo_cells
    required_halos = derivative_stencil.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but the derivative "
        f"stencil for the output quantities required at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string

    path = get_path_to_key(basepath, "is_active")
    is_active = get_setup_value(
        output_dict, "is_active", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_domain")
    is_domain = get_setup_value(
        output_dict, "is_domain", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_wall_clock_times")
    is_wall_clock_times = get_setup_value(
        output_dict, "is_wall_clock_times", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_xdmf")
    is_xdmf = get_setup_value(
        output_dict, "is_xdmf", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_parallel_filesystem")
    is_parallel_filesystem = get_setup_value(
        output_dict, "is_parallel_filesystem", path, bool,
        is_optional=True, default_value=True)

    path = get_path_to_key(basepath, "is_metadata")
    is_metadata = get_setup_value(
        output_dict, "is_metadata", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_time")
    is_time = get_setup_value(
        output_dict, "is_time", path, bool,
        is_optional=True, default_value=True)

    logging_setup = read_logging(output_dict)

    output_setup = OutputSetup(
        is_active, is_domain, is_wall_clock_times,
        derivative_stencil, is_xdmf, is_parallel_filesystem,
        is_metadata, is_time, logging_setup)

    return output_setup


def read_logging(output_dict: Dict) -> LoggingSetup:

    basepath = get_path_to_key("output", "logging")
    logging_dict = get_setup_value(
        output_dict, "logging", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "level")
    level = get_setup_value(
        logging_dict, "level", path, str,
        is_optional=True, default_value="INFO",
        possible_string_values=TUPLE_LOGGING_MODES)

    path = get_path_to_key(basepath, "frequency")
    frequency = get_setup_value(
        logging_dict, "frequency", path, int,
        is_optional=True, default_value=1,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "is_positivity")
    is_positivity = get_setup_value(
        logging_dict, "is_positivity", path, bool,
        is_optional=True, default_value=True)

    path = get_path_to_key(basepath, "is_levelset_residuals")
    is_levelset_residuals = get_setup_value(
        logging_dict, "is_levelset_residuals", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_only_last_stage")
    is_only_last_stage = get_setup_value(
        logging_dict, "is_only_last_stage", path, bool,
        is_optional=True, default_value=True)
    
    logging_setup = LoggingSetup(
        level, frequency, is_positivity,
        is_levelset_residuals, is_only_last_stage
    )

    return logging_setup

    