import os
from typing import Dict

import jax.numpy as jnp
import numpy as np

from jaxfluids.data_types.case_setup import GeneralSetup
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.input.setup_reader import SetupReader, get_path_to_key, create_wrapper_for_callable, assert_case
from jaxfluids.input.case_setup import get_setup_value, loop_fields
from jaxfluids.unit_handler import UnitHandler

def read_general_setup(
        case_setup_dict: Dict,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler
        ) -> GeneralSetup:
    """Reads the case setup and initializes
    general settings and the level-set interaction type.
    """

    basepath = "general"
    general_dict: Dict = get_setup_value(
        case_setup_dict, "general", basepath,
        dict, is_optional=False)

    path = get_path_to_key(basepath, "case_name")
    case_name = get_setup_value(
        general_dict, "case_name", path,
        str, is_optional=False)

    path = get_path_to_key(basepath, "end_time")
    end_time = get_setup_value(
        general_dict, "end_time", path, float,
        is_optional=True, default_value=False,
        numerical_value_condition=(">=", 0.0))

    path = get_path_to_key(basepath, "end_step")
    end_step = get_setup_value(
        general_dict, "end_step", path, int,
        is_optional=True, default_value=False,
        numerical_value_condition=(">=", 0))

    assert_str = "Either end_time or end_step must be given."
    assert_case(isinstance(end_step, int) or isinstance(end_time, float), assert_str)

    is_double = numerical_setup.precision.is_double_precision_compute
    dtype_int = jnp.int64 if is_double else jnp.int32
    dtype_float = jnp.float64 if is_double else jnp.float32
    if isinstance(end_time, bool):
        end_time = jnp.finfo(dtype_float).max
    else:
        end_time = unit_handler.non_dimensionalize(end_time, "time")
    if isinstance(end_step, bool):
        end_step = jnp.iinfo(dtype_int).max

    path = get_path_to_key(basepath, "save_path")
    save_path = get_setup_value(
        general_dict, "save_path", path, str,
        is_optional=True, default_value="./results")

    path = get_path_to_key(basepath, "save_dt")
    save_dt = get_setup_value(
        general_dict, "save_dt", path, float,
        is_optional=True, default_value=False,
        numerical_value_condition=(">", 0.0))
    if save_dt:
        save_dt = unit_handler.non_dimensionalize(save_dt, "time")

    path = get_path_to_key(basepath, "save_step")
    save_step = get_setup_value(
        general_dict, "save_step", path, int,
        is_optional=True, default_value=False,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "save_timestamps")
    save_timestamps = get_setup_value(
        general_dict, "save_timestamps", path, list,
        is_optional=True, default_value=False)
    if save_timestamps == []:
        save_timestamps = False
    if save_timestamps:
        save_timestamps = [
            unit_handler.non_dimensionalize(time_stamp, "time") for time_stamp in save_timestamps]
        save_timestamps = np.sort(save_timestamps)

    assert_str = "Either save_dt, save_step or save_timestamps must be given."
    assert_case(any((save_dt, save_step, save_timestamps)), assert_str)

    path = get_path_to_key(basepath, "save_start")
    save_start = get_setup_value(
        general_dict, "save_start", path, float,
        is_optional=True, default_value=0.0,
        numerical_value_condition=(">=", 0.0))
    save_start = unit_handler.non_dimensionalize(save_start, "time")

    general_setup = GeneralSetup(
        case_name=case_name,
        end_time=end_time,
        end_step=end_step,
        save_path=save_path,
        save_dt=save_dt,
        save_step=save_step,
        save_timestamps=save_timestamps,
        save_start=save_start)

    return general_setup
