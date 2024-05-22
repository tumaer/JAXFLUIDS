from typing import Dict
from jaxfluids.input.setup_reader import get_path_to_key, create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value, loop_fields
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import *

def read_ideal_gas(
        eos_setup: Dict,
        eos_path: str,
        unit_handler: UnitHandler
        ) -> IdealGasSetup:
    path = get_path_to_key(eos_path, "specific_heat_ratio")
    specific_heat_ratio = get_setup_value(
        eos_setup, "specific_heat_ratio", path, float,
        is_optional=False, numerical_value_condition=(">", 0.0))

    path = get_path_to_key(eos_path, "specific_gas_constant")
    specific_gas_constant = get_setup_value(
        eos_setup, "specific_gas_constant", path, float,
        is_optional=False, numerical_value_condition=(">", 0.0))
    specific_gas_constant = unit_handler.non_dimensionalize(
        specific_gas_constant, "specific_gas_constant")

    ideal_gas_setup = IdealGasSetup(
        specific_heat_ratio=specific_heat_ratio,
        specific_gas_constant=specific_gas_constant)
    
    return ideal_gas_setup