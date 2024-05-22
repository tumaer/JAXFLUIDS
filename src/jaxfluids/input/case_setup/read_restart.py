from typing import Dict
from jaxfluids.data_types.case_setup import RestartSetup
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.input.case_setup import get_setup_value, get_path_to_key
from jaxfluids.unit_handler import UnitHandler

def read_restart_setup(
        case_setup_dict: Dict,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler
        ) -> RestartSetup:
    
    basepath = "restart"
    restart_dict: Dict = get_setup_value(
        case_setup_dict, "restart", basepath,
        dict, is_optional=True, default_value={})
    is_optional = not "restart" in case_setup_dict

    path = get_path_to_key(basepath, "flag")
    flag = get_setup_value(
        restart_dict, "flag", path,
        bool, is_optional=is_optional, default_value=False)

    is_optional = not flag
    path = get_path_to_key(basepath, "file_path")
    file_path = get_setup_value(
        restart_dict, "file_path", path,
        str, is_optional=is_optional, default_value="")

    path = get_path_to_key(basepath, "use_time")
    use_time = get_setup_value(
        restart_dict, "use_time", path,
        bool, is_optional=True, default_value=False)
    
    path = get_path_to_key(basepath, "is_equal_decomposition")
    is_equal_decomposition = get_setup_value(
        restart_dict, "is_equal_decomposition", path,
        bool, is_optional=True, default_value=False)

    is_optional = not use_time
    path = get_path_to_key(basepath, "time")
    time = get_setup_value(
        restart_dict, "time", path, float,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">=", 0.0))
    time = unit_handler.non_dimensionalize(time, "time")

    restart_setup = RestartSetup(
        flag,
        file_path,
        use_time,
        time,
        is_equal_decomposition)

    return restart_setup
