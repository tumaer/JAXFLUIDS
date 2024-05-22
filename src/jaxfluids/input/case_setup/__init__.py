from typing import Dict, Tuple, Any, NamedTuple

from jaxfluids.unit_handler import UnitHandler
from jaxfluids.input.setup_reader import _get_setup_value, _loop_fields, get_path_to_key

SETUP = "case"

def get_setup_value(
        setup_dict: Dict,
        key: str,
        absolute_path: str,
        possible_data_types: Tuple,
        is_optional: bool,
        default_value: Any = None,
        possible_string_values: Tuple[str] = None,
        numerical_value_condition = None,
        ) -> Any:
    """Wrapper for the get setup value function implemented
    in the setup reader base class.

    :param setup_dict: _description_
    :type setup_dict: Dict
    :param key: _description_
    :type key: str
    :param absolute_path: _description_
    :type absolute_path: str
    :param possible_data_types: _description_
    :type possible_data_types: Tuple
    :param default_value: _description_, defaults to None
    :type default_value: Any, optional
    :param is_optional: _description_, defaults to False
    :type is_optional: bool, optional
    :param possible_string_values: _description_, defaults to None
    :type possible_string_values: Tuple[str], optional
    :param numerical_value_condition: _description_, defaults to None
    :type numerical_value_condition: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    return _get_setup_value(SETUP, setup_dict, key, absolute_path,
                            possible_data_types, is_optional, default_value,
                            possible_string_values, numerical_value_condition)

def loop_fields(
        parameters_tuple: NamedTuple,
        setup_dict: Dict,
        basepath: str,
        unit_handler: UnitHandler = None,
        unit: str = None,
        numerical_value_condition: Tuple = None,
        unit_exceptions: Dict = {}
        ) -> NamedTuple:
    """Wrapper for the get loop fields function implemented
    in the setup reader base class.

    :param parameters_tuple: _description_
    :type parameters_tuple: NamedTuple
    :param setup_dict: _description_
    :type setup_dict: Dict
    :param basepath: _description_
    :type basepath: str
    :return: _description_
    :rtype: NamedTuple
    """
    return _loop_fields(SETUP, parameters_tuple, setup_dict,
                        basepath, unit_handler, unit,
                        numerical_value_condition,
                        unit_exceptions)
