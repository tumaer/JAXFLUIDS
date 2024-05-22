import collections
from typing import Dict, NamedTuple, Tuple

import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.case_setup import GetPrimitivesCallable, DomainSetup
from jaxfluids.data_types.case_setup.initial_conditions import *
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.input.setup_reader import SetupReader, get_path_to_key, create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value, loop_fields
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.turb import TUPLE_HIT_ENERGY_SPECTRUM, TUPLE_HIT_IC_TYPE, \
    TUPLE_TURB_INIT_CONDITIONS, TUPLE_VELOCITY_PROFILES_CHANNEL

def read_initial_condition_setup(
        case_setup_dict: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler,
        domain_setup: DomainSetup
        ) -> InitialConditionSetup:
    """Reads the initial condition from the
    case setup .json file and initializes
    the corresponding jaxfluids container.

    :return: _description_
    :rtype: _type_
    """

    basepath = "initial_condition"
    initial_condition_case_setup: Dict = get_setup_value(
        case_setup_dict, "initial_condition", basepath, dict,
        is_optional=False)

    if "turbulent" in initial_condition_case_setup.keys():
        initial_condition_turbulent = read_turbulent(
            initial_condition_case_setup, unit_handler)
        is_turb_init = True
    else:
        initial_condition_turbulent = None
        is_turb_init = False

    if not is_turb_init:
        initial_condition_primitives = read_primitives(
            initial_condition_case_setup, equation_information,
            domain_setup, unit_handler)
    else:
        initial_condition_primitives = None

    if equation_information.levelset_model:
        initial_condition_levelset = read_levelset(
            initial_condition_case_setup, equation_information,
            domain_setup, unit_handler)
    else:
        initial_condition_levelset = None

    # SOLID VELOCITY
    if equation_information.levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
        initial_condition_solid_velocity = read_solid_velocity(
            initial_condition_case_setup, equation_information, domain_setup,
            unit_handler)
    else:
        initial_condition_solid_velocity = None

    initial_condition_setup = InitialConditionSetup(
        initial_condition_primitives,
        initial_condition_levelset,
        initial_condition_solid_velocity,
        initial_condition_turbulent,
        is_turb_init)

    return initial_condition_setup

def read_turbulent(
        initial_condition_case_setup: Dict,
        unit_handler: UnitHandler
        ) -> InitialConditionTurbulent:

    path_to_turbulent = get_path_to_key("initial_condition", "turbulent")

    turbulent_case_setup: Dict = get_setup_value(
        initial_condition_case_setup, "turbulent", path_to_turbulent, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(path_to_turbulent, "case")
    case = get_setup_value(
        turbulent_case_setup, "case", path, str,
        is_optional=False, default_value=None,
        possible_string_values=TUPLE_TURB_INIT_CONDITIONS)

    path = get_path_to_key(path_to_turbulent, "random_seed")
    random_seed = get_setup_value(
        turbulent_case_setup, "random_seed", path, int,
        is_optional=True, default_value=0,
        numerical_value_condition=(">=", 0))

    path_parameters = get_path_to_key(path_to_turbulent, "parameters")
    parameters_case_setup = get_setup_value(
        turbulent_case_setup, "parameters", path_parameters, dict,
        is_optional=False, default_value={})
    
    parameters = None
    if case == "HIT":
        path = get_path_to_key(path_parameters, "T_ref")
        T_ref = get_setup_value(
            parameters_case_setup, "T_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        T_ref = unit_handler.non_dimensionalize(T_ref, "temperature")

        path = get_path_to_key(path_parameters, "rho_ref")
        rho_ref = get_setup_value(
            parameters_case_setup, "rho_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        rho_ref = unit_handler.non_dimensionalize(rho_ref, "density")
        
        path = get_path_to_key(path_parameters, "ma_target")
        ma_target = get_setup_value(
            parameters_case_setup, "ma_target", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        path = get_path_to_key(path_parameters, "energy_spectrum")
        energy_spectrum = get_setup_value(
            parameters_case_setup, "energy_spectrum", path, str,
            is_optional=False, possible_string_values=TUPLE_HIT_ENERGY_SPECTRUM)
        
        path = get_path_to_key(path_parameters, "ic_type")
        ic_type = get_setup_value(
            parameters_case_setup, "ic_type", path, str,
            is_optional=False, possible_string_values=TUPLE_HIT_IC_TYPE)
        
        path = get_path_to_key(path_parameters, "xi_0")
        xi_0 = get_setup_value(
            parameters_case_setup, "xi_0", path, int,
            is_optional=False, numerical_value_condition=(">=", 0))
        
        path = get_path_to_key(path_parameters, "xi_1")
        xi_1 = get_setup_value(
            parameters_case_setup, "xi_1", path, int,
            is_optional=True, default_value=16,
            numerical_value_condition=(">=", 0))
        
        path = get_path_to_key(path_parameters, "is_velocity_spectral")
        is_velocity_spectral = get_setup_value(
            parameters_case_setup, "is_velocity_spectral", path, bool,
            is_optional=True, default_value=False)
        
        parameters = HITParameters(
            T_ref, rho_ref, energy_spectrum, ma_target,
            ic_type, xi_0, xi_1, is_velocity_spectral)

    elif case == "CHANNEL":
        path = get_path_to_key(path_parameters, "velocity_profile")
        velocity_profile = get_setup_value(
            parameters_case_setup, "velocity_profile", path, str,
            is_optional=False, possible_string_values=TUPLE_VELOCITY_PROFILES_CHANNEL)

        path = get_path_to_key(path_parameters, "U_ref")
        U_ref = get_setup_value(
            parameters_case_setup, "U_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        U_ref = unit_handler.non_dimensionalize(U_ref, "velocity")
        
        path = get_path_to_key(path_parameters, "rho_ref")
        rho_ref = get_setup_value(
            parameters_case_setup, "rho_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        rho_ref = unit_handler.non_dimensionalize(rho_ref, "density")
        
        path = get_path_to_key(path_parameters, "T_ref")
        T_ref = get_setup_value(
            parameters_case_setup, "T_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        T_ref = unit_handler.non_dimensionalize(T_ref, "temperature")

        path = get_path_to_key(path_parameters, "noise_level")
        noise_level = get_setup_value(
            parameters_case_setup, "noise_level", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        parameters = ChannelParameters(
            velocity_profile, U_ref, rho_ref,
            T_ref, noise_level)

    elif case == "DUCT":
        path = get_path_to_key(path_parameters, "velocity_profile")
        velocity_profile = get_setup_value(
            parameters_case_setup, "velocity_profile", path, str,
            is_optional=False, possible_string_values=TUPLE_VELOCITY_PROFILES_CHANNEL)

        path = get_path_to_key(path_parameters, "U_ref")
        U_ref = get_setup_value(
            parameters_case_setup, "U_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        U_ref = unit_handler.non_dimensionalize(U_ref, "velocity")
        
        path = get_path_to_key(path_parameters, "rho_ref")
        rho_ref = get_setup_value(
            parameters_case_setup, "rho_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        rho_ref = unit_handler.non_dimensionalize(rho_ref, "density")
        
        path = get_path_to_key(path_parameters, "T_ref")
        T_ref = get_setup_value(
            parameters_case_setup, "T_ref", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        T_ref = unit_handler.non_dimensionalize(T_ref, "temperature")

        path = get_path_to_key(path_parameters, "noise_level")
        noise_level = get_setup_value(
            parameters_case_setup, "noise_level", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        parameters = DuctParameters(
            velocity_profile, U_ref, rho_ref,
            T_ref, noise_level)

    elif case == "BOUNDARYLAYER":

        path = get_path_to_key(path_parameters, "T_e")
        T_e = get_setup_value(
            parameters_case_setup, "T_e", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))

        path = get_path_to_key(path_parameters, "rho_e")
        rho_e = get_setup_value(
            parameters_case_setup, "rho_e", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        path = get_path_to_key(path_parameters, "U_e")
        U_e = get_setup_value(
            parameters_case_setup, "U_e", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        path = get_path_to_key(path_parameters, "mu_e")
        mu_e = get_setup_value(
            parameters_case_setup, "mu_e", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))
        
        path = get_path_to_key(path_parameters, "x_position")
        x_position = get_setup_value(
            parameters_case_setup, "x_position", path, float,
            is_optional=False, numerical_value_condition=(">", 0.0))

        T_e = unit_handler.non_dimensionalize(T_e, "temperature")
        rho_e = unit_handler.non_dimensionalize(rho_e, "density")
        U_e = unit_handler.non_dimensionalize(U_e, "velocity")
        mu_e = unit_handler.non_dimensionalize(mu_e, "dynamic_viscosity")
        x_position = unit_handler.non_dimensionalize(x_position, "length")

        parameters = BoundaryLayerParameters(
            T_e, rho_e, U_e, mu_e, x_position)

    elif case == "TGV":
        raise NotImplementedError
    
    else:
        raise NotImplementedError

    initial_condition_turbulent = InitialConditionTurbulent(
        case, random_seed, parameters)

    return initial_condition_turbulent

def read_primitives(
        initial_condition_case_setup: Dict,
        equation_information: EquationInformation,
        domain_setup: DomainSetup,
        unit_handler: UnitHandler
        ) -> NamedTuple:

    active_axes = domain_setup.active_axes
    dim = domain_setup.dim

    input_argument_labels = tuple(active_axes)
    input_argument_units = tuple(["length"] * dim)

    if "primitives" in initial_condition_case_setup.keys():
        initial_condition_primitives_case_setup = initial_condition_case_setup["primitives"]
        path_to_primes = get_path_to_key("initial_condition", "primitives")
    else:
        initial_condition_primitives_case_setup = initial_condition_case_setup
        path_to_primes = "initial_condition"

    def _read_primititves_callables(
            initial_condition_primitives_case_setup: Dict,
            primes_tuple: Tuple,
            path: str
            ) -> NamedTuple:
        """Wrapper to read the primitives callable
        from the case setup .json file and create jaxfluids
        container.

        :param initial_condition_primitives_case_setup: [description]
        :type initial_condition_primitives_case_setup: Dict
        :param primes_tuple: [description]
        :type primes_tuple: Tuple
        :param path: [description]
        :type path: str
        :return: [description]
        :rtype: NamedTuple
        """

        primitives_callables_dict = {}
        for prime_state in primes_tuple:
            path_prime = get_path_to_key(path, prime_state)
            prime_state_case_setup = get_setup_value(
                initial_condition_primitives_case_setup, prime_state,
                path_prime, (float, str), is_optional=False)
            prime_wrapper = create_wrapper_for_callable(
                prime_state_case_setup, input_argument_units,
                input_argument_labels, prime_state, path_prime,
                perform_nondim=True, unit_handler=unit_handler)
            primitives_callables_dict[prime_state] = prime_wrapper
        primitives_callable = GetPrimitivesCallable(primitives_callables_dict)
        return primitives_callable

    if equation_information.levelset_model == "FLUID-FLUID":
        initial_condition_primitives_dict = {}
        for phase in InitialConditionPrimitivesLevelset._fields:
            path_phase = get_path_to_key(path_to_primes, phase)
            initial_condition_primitives_phase_case_setup = get_setup_value(
                initial_condition_primitives_case_setup, phase, path_phase, dict,
                is_optional=False)
            primitives_callable = _read_primititves_callables(
                initial_condition_primitives_phase_case_setup,
                equation_information.primes_tuple, path_phase)
            initial_condition_primitives_dict[phase] = primitives_callable
        initial_condition_primitives = InitialConditionPrimitivesLevelset(
            **initial_condition_primitives_dict)

    elif equation_information.diffuse_interface_model in ("5EQM",):
        init_cond_primitive_keys = list(initial_condition_primitives_case_setup.keys())

        if collections.Counter(init_cond_primitive_keys) \
            == collections.Counter(equation_information.primes_tuple):
            # Initialize via (alpharho_0, alpharho_1, u, v, w, p, alpha_0)
            initial_condition_primitives = _read_primititves_callables(
                initial_condition_primitives_case_setup,
                equation_information.primes_tuple, path_to_primes)
        elif collections.Counter(init_cond_primitive_keys) \
            == collections.Counter(equation_information.primes_tuple_):
            # Initialize via (rho_0, rho_1, u, v, w, p, alpha_0)
            initial_condition_primitives = _read_primititves_callables(
                initial_condition_primitives_case_setup,
                equation_information.primes_tuple_, path_to_primes)
        else:
            assert_string = (
                "Consistency error in case setup file. "
                "Reading initial conditions for diffuse interface method "
                "requires initialization via (alpharho_i, u, p, alpha_i) "
                "or (rho_i, u, p, alpha_i). "
                f"Neither was found valid, instead {init_cond_primitive_keys} was found.")
            assert False, assert_string 

    else:
        initial_condition_primitives = _read_primititves_callables(
            initial_condition_primitives_case_setup,
            equation_information.primes_tuple, path_to_primes)

    return initial_condition_primitives

def read_levelset(
        initial_condition_case_setup: Dict,
        equation_information: EquationInformation,
        domain_setup: DomainSetup,
        unit_handler: UnitHandler) -> NamedTuple:

    active_axes = domain_setup.active_axes
    dim = domain_setup.dim

    input_argument_labels = tuple(active_axes)
    input_argument_units = tuple(["length"] * dim)

    levelset_blocks = None
    levelset_callable = None
    NACA_profile = None
    diamond_airfoil_params = None
    h5_file_path = None

    is_blocks = False
    is_callable = False
    is_NACA = False
    is_h5_file = False

    path_to_levelset = get_path_to_key("initial_condition", "levelset")

    initial_condition_levelset_case_setup = get_setup_value(
        initial_condition_case_setup, "levelset", path_to_levelset,
        (str, list), is_optional=False)
    
    if isinstance(initial_condition_levelset_case_setup, list):
        is_blocks = True
        levelset_blocks_list = []
        for levelset_block_case_setup in initial_condition_levelset_case_setup:
            path = get_path_to_key(path_to_levelset, "shape")
            shape = get_setup_value(levelset_block_case_setup, "shape", path, str,
                                    is_optional=False)

            path = get_path_to_key(path_to_levelset, "parameters")
            parameters_case_setup = get_setup_value(
                levelset_block_case_setup, "parameters", path, dict,
                is_optional=False)
            ParametersTuple = GetInitialLevelsetBlockParametersTuple(shape)
            parameters = loop_fields(ParametersTuple, parameters_case_setup,
                                     path, unit_exceptions={"angle_of_attack": "None"})
            path = get_path_to_key(path_to_levelset, "bounding_domain")

            bounding_domain_case_setup = get_setup_value(
                levelset_block_case_setup, "bounding_domain", path, str,
                is_optional=False)
            bounding_domain_callable = create_wrapper_for_callable(
                bounding_domain_case_setup, input_argument_units,
                input_argument_labels, None, path, 
                perform_nondim=False, unit_handler=unit_handler)
            
            levelset_block = InitialLevelsetBlock(
                shape, parameters, bounding_domain_callable)
            levelset_blocks_list.append(levelset_block)

        levelset_blocks = tuple(levelset_blocks_list)

    elif isinstance(initial_condition_levelset_case_setup, str):
        if "lambda" in initial_condition_levelset_case_setup:
            is_callable = True
            levelset_callable = create_wrapper_for_callable(
                initial_condition_levelset_case_setup,
                input_argument_units, input_argument_labels,
                "length", path_to_levelset, 
                perform_nondim=True, unit_handler=unit_handler)
        if "NACA" in initial_condition_levelset_case_setup:
            is_NACA = True
            NACA_profile = initial_condition_levelset_case_setup
        if initial_condition_levelset_case_setup.endswith("h5"):
            is_h5_file = True
            h5_file_path = initial_condition_levelset_case_setup

    else:
        raise NotImplementedError

    initial_condition_levelset = InitialConditionLevelset(
        levelset_blocks, levelset_callable,
        h5_file_path, NACA_profile, is_blocks,
        is_callable, is_NACA, is_h5_file)

    return initial_condition_levelset
    
def read_solid_velocity(
        initial_condition_case_setup: Dict,
        equation_information: EquationInformation,
        domain_setup: DomainSetup,
        unit_handler: UnitHandler) -> VelocityCallable:

    active_axes = domain_setup.active_axes
    dim = domain_setup.dim

    input_argument_labels = tuple(active_axes)
    input_argument_units = tuple(["length"] * dim)

    path_solid_velocity = get_path_to_key("initial_condition", "solid_velocity")
    solid_velocity_case_setup = get_setup_value(
        initial_condition_case_setup, "solid_velocity",
        path_solid_velocity, dict, is_optional=False)

    solid_velocity_callables_dict = {}
    for velocity_xi in ["u","v","w"]:
        path = get_path_to_key(path_solid_velocity, velocity_xi)
        wall_velocity_xi_case_setup = get_setup_value(
            solid_velocity_case_setup, velocity_xi, path, (float, str),
            is_optional=False)
        velocity_wrapper = create_wrapper_for_callable(
            wall_velocity_xi_case_setup, input_argument_units,
            input_argument_labels, "velocity", path,
            perform_nondim=True, unit_handler=unit_handler)
        solid_velocity_callables_dict[velocity_xi] = velocity_wrapper
    solid_velocity_callable = VelocityCallable(**solid_velocity_callables_dict)
    return solid_velocity_callable
