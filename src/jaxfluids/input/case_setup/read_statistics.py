from typing import Dict, Any

from jaxfluids.turbulence.statistics import TUPLE_TURBULENCE_CASES
from jaxfluids.data_types.case_setup.statistics import StatisticsSetup, TurbulenceStatisticsSetup
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.input.case_setup import get_setup_value, get_path_to_key


def read_statistics_setup(
        case_setup_dict: Dict,
        unit_handler: UnitHandler
        ) -> StatisticsSetup:
    
    basepath = "statistics"
    statistics_dict: Dict = get_setup_value(
        case_setup_dict, "statistics", basepath,
        dict, is_optional=True, default_value={})

    turbulence_statistics = read_turbulence_statistics_setup(
        statistics_dict, unit_handler)

    statistics_setup = StatisticsSetup(turbulence_statistics)
    
    return statistics_setup

def read_turbulence_statistics_setup(
        statistics_dict: Dict,
        unit_handler: UnitHandler
        ) -> TurbulenceStatisticsSetup:
    
    basepath = get_path_to_key("statistics", "turbulence")

    turbulence_statistics_dict = get_setup_value(
        statistics_dict, "turbulence", basepath,
        dict, is_optional=True, default_value={})

    path = get_path_to_key(basepath, "is_cumulative")
    is_cumulative = get_setup_value(
        turbulence_statistics_dict, "is_cumulative", path, bool,
        is_optional=True, default_value=False)

    path = get_path_to_key(basepath, "is_logging")
    is_logging = get_setup_value(
        turbulence_statistics_dict, "is_logging", path, bool,
        is_optional=True, default_value=False)

    path = get_path_to_key(basepath, "case")
    case = get_setup_value(
        turbulence_statistics_dict, "case", path,
        str, is_optional=not any((is_cumulative, is_logging)), 
        possible_string_values=TUPLE_TURBULENCE_CASES)

    path = get_path_to_key(basepath, "start_sampling")
    start_sampling = get_setup_value(
        turbulence_statistics_dict, "start_sampling", path, float,
        is_optional=not is_cumulative, default_value=0.0,
        numerical_value_condition=(">=", 0.0))
    start_sampling = unit_handler.non_dimensionalize(start_sampling, "time")
    
    # TODO DENIZ DEFAULT VALUES ? 
    path = get_path_to_key(basepath, "sampling_dt")
    sampling_dt = get_setup_value(
        turbulence_statistics_dict, "sampling_dt", path, float,
        is_optional=not is_cumulative, default_value=0.0,
        numerical_value_condition=(">", 0.0))
    sampling_dt = unit_handler.non_dimensionalize(sampling_dt, "time")

    path = get_path_to_key(basepath, "sampling_step")
    sampling_step = get_setup_value(
        turbulence_statistics_dict, "sampling_step", path, int,
        is_optional=True, default_value=0,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "save_dt")
    save_dt = get_setup_value(
        turbulence_statistics_dict, "save_dt", path, float,
        is_optional=not is_cumulative, default_value=0.0,
        numerical_value_condition=(">", 0.0))
    save_dt = unit_handler.non_dimensionalize(save_dt, "time")    

    is_optional = False if case == "BOUNDARYLAYER" else True
    path = get_path_to_key(basepath, "streamwise_measure_position")
    streamwise_measure_position = get_setup_value(
        turbulence_statistics_dict, "streamwise_measure_position", path, float,
        is_optional=is_optional, default_value=0.0, numerical_value_condition=(">", 0.0))

    turbulence_statistics_setup = TurbulenceStatisticsSetup(
        is_cumulative, is_logging, case, start_sampling,
        sampling_dt, sampling_step, save_dt, streamwise_measure_position)

    return turbulence_statistics_setup
    