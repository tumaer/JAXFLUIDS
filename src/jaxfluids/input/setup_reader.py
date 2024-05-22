from typing import Any, Dict, Tuple, Union, Callable, NamedTuple, List

import jax.numpy as jnp
from jax import Array
from jaxfluids.unit_handler import UnitHandler

class SetupReader:

    def __init__(self, unit_handler: UnitHandler):
        self.unit_handler = unit_handler

def get_path_to_key(*args):
    return "/".join(args)

def _get_setup_value(
        setup: str,
        setup_dict: Dict,
        key: str,
        absolute_path: str,
        possible_data_types: Tuple,
        is_optional: bool,
        default_value: Any = None,
        possible_string_values: Tuple[str] = None,
        numerical_value_condition: Tuple = None,
        ) -> Any:
    """Retrieves the specified key from the 
    setup dictionary. Performs
    consistency checks, i.e., asserts if
    there is a key error, the value
    has a wrong data type or is not
    possible.

    :param setup: _description_
    :type setup: str
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
    :type numerical_value_condition: Tuple, optional
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Dict
    """

    def check_value(setup_value):
        """Checks if the setup value is of possible
        type, possible string value and fulfills the 
        float condition

        :param setup_value: _description_
        :type setup_value: _type_
        :raises NotImplementedError: _description_
        """
        flag = False
        assert_string = (
            f"Consistency error in {setup:s} setup file. "
            f"Key {absolute_path} must be of types {possible_data_types}, "
            f"but is of type {type(setup_value)}.")
        assert isinstance(setup_value, possible_data_types), assert_string

        if possible_string_values is not None and isinstance(setup_value, str):
            assert_string = (
                f"Consistency error in {setup:s} setup file. "
                f"Value of {absolute_path:s} must be in {possible_string_values} "
                "if value is of type str.")
            assert setup_value in possible_string_values, assert_string

        if numerical_value_condition != None and isinstance(setup_value, (int, float)):
            condition = numerical_value_condition[0]
            value = numerical_value_condition[1]
            if condition not in [">", "<", ">=", "<="]:
                raise NotImplementedError
            if isinstance(value, int):
                condition_str = f"{str(setup_value):s}" + condition + f"{value:d}"
            elif isinstance(value, float):
                condition_str = f"{str(setup_value):s}" + condition + f"{value:f}"
            else:
                raise NotImplementedError
            assert_string = (
                f"Consistency error in {setup:s} setup file. "
                f"Value of {absolute_path} must be {condition:s} {str(value):s}.")
            flag = eval(condition_str)
            assert flag, assert_string

    if key in setup_dict:
        setup_value = setup_dict[key]
        if not is_optional:
            check_value(setup_value)
        else:
            if setup_value != None:
                check_value(setup_value)
            else:
                setup_value = default_value

    else:
        if is_optional:
            setup_value = default_value
        else:
            assert_string = (
                f"Consistency error in {setup:s} setup file. "
                f"Key {key:s} is not optional, but missing {absolute_path:s}.")
            assert False, assert_string

    return setup_value

def create_wrapper_for_callable(
        value_case_setup: Union[float,str],
        input_units: Tuple[str],
        input_labels: Tuple[str],
        output_unit: str,
        absolute_path: str,
        perform_nondim: bool,
        is_spatial_derivative: bool = False,
        is_temporal_derivative: bool = False,
        unit_handler: UnitHandler = None,
        is_scalar: bool = False
        ) -> Callable:
    """Generates a wrapper function for
    an input value from the case setup .json file.
    Performs consistency checks, i.e., asserts if
    the provided input value has wrong type
    or specifies a lambda function with wrong
    input arguments. Optionally performs 
    dimensionalization of the input arguments
    and non dimensionalization of the output value.
    Optionally returns values as scalars instead
    of field buffers, if the value is a float.

    :param value_case_setup: _description_
    :type value_case_setup: Union[float,str]
    :param input_units: _description_
    :type input_units: Tuple[str]
    :param input_labels: _description_
    :type input_labels: Tuple[str]
    :param output_unit: _description_
    :type output_unit: str
    :param setup_dict: _description_
    :type setup_dict: str
    :param perform_nondim: _description_
    :type perform_nondim: bool
    :param is_spatial_derivative: _description_, defaults to False
    :type is_spatial_derivative: bool, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: Callable
    """

    if isinstance(value_case_setup, float):
        is_callable = False
    elif isinstance(value_case_setup, str):
        value_case_setup: Callable = eval(value_case_setup)
        varnames = value_case_setup.__code__.co_varnames
        assert_string_varnames = (
            "Consistency error in case setup file. " 
            "Input argument labels of lambda for "
            f"{absolute_path} must be {input_labels}.")
        assert varnames == input_labels, assert_string_varnames
        is_callable = True
    else:
        assert_string = (
            "Consistency error in case setup file. "
            f"Value of {absolute_path} must be float or string that " 
            "specifies a lambda function.")
        assert False, assert_string
    
    is_scalar = all((is_scalar, isinstance(value_case_setup, float)))

    def wrapper(*args) -> Array:
        if not len(args) == len(input_units):
            raise RuntimeError

        if perform_nondim and is_callable:
            nondim_args = []
            for arg_i, unit in zip(args, input_units):
                nondim_arg_i = unit_handler.dimensionalize(
                    arg_i, unit)
                nondim_args.append(nondim_arg_i)
        else:
            nondim_args = args

        if is_callable:
            output = value_case_setup(*nondim_args)
        else:
            if not is_scalar:
                output = jnp.ones_like(args[0])
                output *= value_case_setup
            else:
                output = value_case_setup

        if perform_nondim:
            output = unit_handler.non_dimensionalize(
                output, output_unit, None,
                is_spatial_derivative=is_spatial_derivative,
                is_temporal_derivative=is_temporal_derivative)

        return output
    
    return wrapper


def _loop_fields(
        setup: str,
        parameters_tuple: NamedTuple,
        setup_dict: Dict,
        basepath: str,
        unit_handler: UnitHandler = None,
        unit: str = None,
        numerical_value_condition: Tuple = None,
        unit_exceptions: Dict = {}
        ) -> NamedTuple:
    """Wrapper that reads values from a setup dictionary and creates
    the corresponding jaxfluids container.

    :param ParametersTuple: _description_
    :type ParametersTuple: NamedTuple
    :param setup_dict: _description_
    :type setup_dict: Dict
    :param basepath: _description_
    :type basepath: str
    :return: _description_
    :rtype: NamedTuple
    """
    get_path_to_key = lambda *args: "/".join(args)

    parameters_setup = {}
    for field, typehints in parameters_tuple.__annotations__.items():

        if field in parameters_tuple._field_defaults.keys():
            is_optional = True
            default_value = parameters_tuple._field_defaults[field]
        else:
            is_optional = False
            default_value = None

        path = get_path_to_key(basepath, field)
        setup_value = _get_setup_value(
            setup, setup_dict, field, path, typehints,
            is_optional=is_optional, default_value=default_value)
        
        if numerical_value_condition != None:
            condition = numerical_value_condition[0]
            condition_value = numerical_value_condition[1]
            if condition not in [">", "<", ">=", "<="]:
                raise NotImplementedError
            if isinstance(condition_value, int):
                condition_str = f"{str(setup_value):s}" + condition + f"{condition_value:d}"
            elif isinstance(condition_value, float):
                condition_str = f"{str(setup_value):s}" + condition + f"{condition_value:f}"
            else:
                raise NotImplementedError
            assert_string = (
                f"Consistency error in {setup:s} setup file. "
                f"Value of {path} must be {condition:s} {str(condition_value):s}.")
            flag = eval(condition_str)
            assert flag, assert_string

        if unit_handler is not None and unit is not None:
            if field in unit_exceptions:
                setup_value = unit_handler.non_dimensionalize(setup_value, unit_exceptions[field])
            else:
                setup_value = unit_handler.non_dimensionalize(setup_value, unit)

        parameters_setup[field] = setup_value

    parameters_setup = parameters_tuple(**parameters_setup)

    return parameters_setup

def _loop_fields_new(
        setup: str,
        parameters_tuple: NamedTuple,
        setup_dict: Dict,
        basepath: str,
        unit_handler: UnitHandler = None,
        unit: str = None,
        numerical_value_condition: Tuple = None
        ) -> NamedTuple:
    """Wrapper that reads values from a setup dictionary and creates
    the corresponding jaxfluids container.

    :param ParametersTuple: _description_
    :type ParametersTuple: NamedTuple
    :param setup_dict: _description_
    :type setup_dict: Dict
    :param basepath: _description_
    :type basepath: str
    :return: _description_
    :rtype: NamedTuple
    """
    get_path_to_key = lambda *args: "/".join(args)

    parameters_setup = {}
    for i, (field, typehints) in enumerate(parameters_tuple.__annotations__.items()):

        if field in parameters_tuple._field_defaults.keys():
            is_optional = True
            default_value = parameters_tuple._field_defaults[field]
        else:
            is_optional = False
            default_value = None

        path = get_path_to_key(basepath, field)
        setup_value = _get_setup_value(
            setup, setup_dict, field, path, typehints,
            is_optional=is_optional, default_value=default_value)
        
        if numerical_value_condition != None:
            condition = numerical_value_condition[0]
            condition_value = numerical_value_condition[1]
            if condition not in [">", "<", ">=", "<="]:
                raise NotImplementedError
            if isinstance(condition_value, int):
                condition_str = f"{str(setup_value):s}" + condition + f"{condition_value:d}"
            elif isinstance(condition_value, float):
                condition_str = f"{str(setup_value):s}" + condition + f"{condition_value:f}"
            else:
                raise NotImplementedError
            assert_string = (
                f"Consistency error in {setup:s} setup file. "
                f"Value of {path} must be {condition:s} {str(condition_value):s}.")
            flag = eval(condition_str)
            assert flag, assert_string

        if unit_handler is not None and unit is not None:
            if isinstance(unit, (list, tuple)):
                unit_ = unit[i]
            else:
                unit_ = unit
            if unit_ is not None:
                setup_value = unit_handler.non_dimensionalize(setup_value, unit_)

        parameters_setup[field] = setup_value

    parameters_setup = parameters_tuple(**parameters_setup)

    return parameters_setup