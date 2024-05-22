from typing import Dict, Any

from jaxfluids.turb.statistics import TUPLE_TURBULENCE_CASES
from jaxfluids.data_types.numerical_setup.turbulence_statistics import TurbulenceStatisticsSetup
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.input.numerical_setup import get_setup_value, get_path_to_key


def read_turbulence_statistics_setup(
        numerical_setup_dict: Dict,
        unit_handler: UnitHandler
        ) -> TurbulenceStatisticsSetup:
    
    basepath = "turbulence_statistics"

    turbulence_statistics_dict = get_setup_value(
        numerical_setup_dict, "turbulence_statistics", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "is_active")
    is_active = get_setup_value(
        turbulence_statistics_dict, "is_active", path, bool,
        is_optional=True, default_value=False)

    is_optional = True if not is_active else False    
    path = get_path_to_key(basepath, "turbulence_case")
    turbulence_case = get_setup_value(
        turbulence_statistics_dict, "turbulence_case", path,
        str, is_optional=is_optional, 
        possible_string_values=TUPLE_TURBULENCE_CASES)

    path = get_path_to_key(basepath, "start_sampling")
    start_sampling = get_setup_value(
        turbulence_statistics_dict, "start_sampling", path, float,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">=", 0.0))
    
    # TODO DENIZ DEFAULT VALUES ? 
    path = get_path_to_key(basepath, "sampling_dt")
    sampling_dt = get_setup_value(
        turbulence_statistics_dict, "sampling_dt", path, float,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">", 0.0))
    
    path = get_path_to_key(basepath, "sampling_step")
    sampling_step = get_setup_value(
        turbulence_statistics_dict, "sampling_step", path, int,
        is_optional=is_optional, default_value=0,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "save_dt")
    save_dt = get_setup_value(
        turbulence_statistics_dict, "save_dt", path, float,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">", 0.0))

    start_sampling = unit_handler.non_dimensionalize(start_sampling, "time")
    sampling_dt = unit_handler.non_dimensionalize(sampling_dt, "time")
    save_dt = unit_handler.non_dimensionalize(save_dt, "time")

    is_optional = False if turbulence_case == "BOUNDARYLAYER" else True
    path = get_path_to_key(basepath, "streamwise_measure_position")
    streamwise_measure_position = get_setup_value(
        turbulence_statistics_dict, "streamwise_measure_position", path, float,
        is_optional=is_optional, default_value=0.0, numerical_value_condition=(">", 0.0))

    turbulence_statistics_setup = TurbulenceStatisticsSetup(
        is_active, turbulence_case, start_sampling,
        sampling_dt, sampling_step, save_dt, streamwise_measure_position)

    return turbulence_statistics_setup
    