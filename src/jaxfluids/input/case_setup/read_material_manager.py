from typing import Dict, Tuple, Any

import numpy as np

from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup.material_properties import *
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials import DICT_MATERIAL, DYNAMIC_VISCOSITY_MODELS, HEAT_CAPACITY_MODELS, \
    THERMAL_CONDUCTIVITY_MODELS, MASS_DIFFUSIVITY_MODELS
from jaxfluids.input.setup_reader import get_path_to_key, create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value, loop_fields

def read_material_manager_setup(
        case_setup_dict: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler,
        ) -> MaterialManagerSetup:
    """Reads the case setup and initializes material properties.

    Four different options are available:
    1) Single-phase
    2) Level-set mixture
    3) Diffuse-interface mixture

    :return: _description_
    :rtype: MaterialManagerSetup
    """
    basepath = "material_properties"
    material_properties_case_setup = get_setup_value(
        case_setup_dict, "material_properties", basepath, dict,
        is_optional=False)

    single_material = None
    levelset_mixture = None
    diffuse_mixture = None

    active_physics = numerical_setup.active_physics

    # FLUID-FLUID LEVEL-SET
    if equation_information.levelset_model == "FLUID-FLUID":
        path_positive = get_path_to_key(basepath, "positive")
        positive_fluid = read_material(
            material_properties_case_setup["positive"], path_positive,
            equation_information, numerical_setup, unit_handler)

        path_negative = get_path_to_key(basepath, "positive")
        negative_fluid = read_material(
            material_properties_case_setup["negative"], path_negative,
            equation_information, numerical_setup, unit_handler)

        # PAIRING PROPERTIES
        pairing_properties = read_pairing_properties(
            material_properties_case_setup,
            equation_information,
            numerical_setup,
            unit_handler)

        levelset_mixture = LevelsetMixtureSetup(
            positive_fluid,
            negative_fluid,
            pairing_properties)

    # DIFFUSE-INTERFACE MIXTURE
    elif equation_information.diffuse_interface_model:
        diffuse_fluids_dict = {}
        for fluid in equation_information.fluid_names:
            path_fluid = get_path_to_key(basepath, fluid)
            fluid_case_setup = get_setup_value(
                material_properties_case_setup,
                fluid, path_fluid, (dict, str),
                is_optional=False)
            diffuse_fluids_dict[fluid] = read_material(
                fluid_case_setup, path_fluid,
                equation_information, numerical_setup,
                unit_handler)
        diffuse_fluids = GetFluids(diffuse_fluids_dict)

        # PAIRING PROPERTIES
        pairing_properties = read_pairing_properties(
            material_properties_case_setup,
            equation_information,
            numerical_setup,
            unit_handler)

        diffuse_mixture = DiffuseMixtureSetup(diffuse_fluids, pairing_properties)

    # SINGLE-PHASE
    else:
        single_material = read_material(
            material_properties_case_setup, basepath,
            equation_information, numerical_setup, unit_handler)

    material_manager_setup = MaterialManagerSetup(
        single_material=single_material,
        levelset_mixture=levelset_mixture,
        diffuse_mixture=diffuse_mixture)
    return material_manager_setup

def read_material(
        fluid_case_setup: Dict,
        basepath: str,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler) -> MaterialPropertiesSetup:
    """Reads material properties from a dict, non-dimensionalizes
    all quantities and returns the corresponding MaterialPropertiesSetup.
    Material properties consist of
    1) Equation of state properties
    2) Transport properties.

    :param material_properties_dict: _description_
    :type material_properties_dict: Dict
    :return: _description_
    :rtype: MaterialPropertiesSetup
    """
    eos_properties_setup = read_equation_of_state(
        basepath, fluid_case_setup, 
        numerical_setup, unit_handler)
    transport_properties_setup = read_transport_properties(
        basepath, fluid_case_setup,
        equation_information, numerical_setup, unit_handler)

    material_properties_setup = MaterialPropertiesSetup(
        eos=eos_properties_setup,
        transport=transport_properties_setup)

    return material_properties_setup

def read_equation_of_state(
        basepath: str,
        fluid_case_setup: Dict,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler
        ) -> EquationOfStatePropertiesSetup:
    """Reads the equation of state setup from
    the case setup.

    :param basepath: _description_
    :type basepath: str
    :param fluid_case_setup: _description_
    :type fluid_case_setup: Dict
    :param unit_handler: _description_
    :type unit_handler: UnitHandler
    :raises NotImplementedError: _description_
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: EquationOfStatePropertiesSetup
    """

    # EQUATION OF STATE
    eos_path = get_path_to_key(basepath, "equation_of_state")
    eos_setup = get_setup_value(
        fluid_case_setup, "equation_of_state", eos_path, dict,
        is_optional=False)

    path = get_path_to_key(eos_path, "model")
    eos_model = get_setup_value(
        eos_setup, "model", path, str, is_optional=False,
        possible_string_values=tuple(DICT_MATERIAL.keys()))

    assert_str = (
        "Consistency error in case setup file. "
        f"Chosen EOS model '{eos_model}' is not implemented. "
        f"Please choose from the following models: {list(DICT_MATERIAL.keys())}")
    assert eos_model in DICT_MATERIAL, assert_str

    ideal_gas_setup, stiffened_gas_setup, tait_setup = None, None, None

    if eos_model == "IdealGas":
        ideal_gas_setup = read_ideal_gas(
            eos_setup, eos_path, unit_handler)

    elif eos_model in ("StiffenedGas", "StiffenedGasComplete"):
        stiffened_gas_setup = read_stiffened_gas(
            eos_setup, eos_path, eos_model, unit_handler)

    elif eos_model == "Tait":
        tait_setup = read_tait(eos_setup, eos_path, unit_handler)

    else:
        raise NotImplementedError
        
    eos_setup = EquationOfStatePropertiesSetup(
        model=eos_model,
        ideal_gas_setup=ideal_gas_setup,
        stiffened_gas_setup=stiffened_gas_setup,
        tait_setup=tait_setup)
    
    return eos_setup

def read_transport_properties(
        basepath: str,
        fluid_case_setup: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler,
        ) -> TransportPropertiesSetup:
    is_viscous_flux = numerical_setup.active_physics.is_viscous_flux
    is_heat_flux = numerical_setup.active_physics.is_heat_flux

    is_viscosity_optional = not (is_viscous_flux or is_heat_flux)
    is_thermal_conductivity_optional = not is_heat_flux
    is_transport_properties_optional = all(
        (is_viscosity_optional,
        is_thermal_conductivity_optional))

    transport_properties_path = get_path_to_key(basepath, "transport")
    transport_properties_setup = get_setup_value(
        fluid_case_setup, "transport", transport_properties_path, dict,
        default_value={}, is_optional=is_transport_properties_optional)

    # DYNAMIC VISCOSITY
    path_dynamic_viscosity = get_path_to_key(
        transport_properties_path, "dynamic_viscosity")
    dynamic_viscosity_setup = get_setup_value(
        transport_properties_setup, "dynamic_viscosity", path_dynamic_viscosity,
        dict, default_value={}, is_optional=is_viscosity_optional)

    path = get_path_to_key(path_dynamic_viscosity, "model")
    dynamic_viscosity_model = get_setup_value(
        dynamic_viscosity_setup, "model", path, str,
        is_optional=is_viscosity_optional, default_value="CUSTOM",
        possible_string_values=DYNAMIC_VISCOSITY_MODELS)

    dynamic_viscosity_value = None
    sutherland_parameters = None
    if dynamic_viscosity_model == "CUSTOM":
        input_argument_labels = tuple(["T"])
        input_argument_units = tuple(["temperature"])
        path = get_path_to_key(path_dynamic_viscosity, "value")
        dynamic_viscosity_value_case_setup = get_setup_value(
            dynamic_viscosity_setup, "value", path, (float, str),
            default_value=0.0, is_optional=is_viscosity_optional,
            numerical_value_condition=(">=", 0.0))
        dynamic_viscosity_value = create_wrapper_for_callable(
            dynamic_viscosity_value_case_setup, input_argument_units,
            input_argument_labels, "dynamic_viscosity", path, 
            perform_nondim=True, unit_handler=unit_handler,
            is_scalar=True)

    elif dynamic_viscosity_model == "SUTHERLAND":
        sutherland_parameters_path = get_path_to_key(
            path_dynamic_viscosity, "sutherland_parameters")
        sutherland_parameters_case_setup = get_setup_value(
            dynamic_viscosity_setup, "sutherland_parameters",
            sutherland_parameters_path, list, is_optional=False)

        viscosity_ref = unit_handler.non_dimensionalize(
            sutherland_parameters_case_setup[0], "dynamic_viscosity")
        T_ref = unit_handler.non_dimensionalize(
            sutherland_parameters_case_setup[1], "temperature")
        constant = unit_handler.non_dimensionalize(
            sutherland_parameters_case_setup[2], "temperature")

        sutherland_parameters = SutherlandParameters(
            viscosity_ref, T_ref, constant)

    else:
        raise NotImplementedError

    dynamic_viscosity_setup = DynamicViscositySetup(
        dynamic_viscosity_model,
        dynamic_viscosity_value,
        sutherland_parameters)

    # BULK VISCOSITY
    path = get_path_to_key(transport_properties_path, "bulk_viscosity")
    bulk_viscosity = get_setup_value(
        transport_properties_setup, "bulk_viscosity", path, float,
        default_value=0.0, is_optional=is_viscosity_optional,
        numerical_value_condition=(">=", 0.0))
    bulk_viscosity = unit_handler.non_dimensionalize(
        bulk_viscosity, "dynamic_viscosity")

    # THERMAL CONDUCTIVITY
    path_thermal_conductivity = get_path_to_key(
        transport_properties_path, "thermal_conductivity")
    thermal_conductivity_setup = get_setup_value(
        transport_properties_setup, "thermal_conductivity", path_thermal_conductivity,
        dict, default_value={}, is_optional=is_thermal_conductivity_optional)

    path = get_path_to_key(path_thermal_conductivity, "model")
    thermal_conductivity_model = get_setup_value(
        thermal_conductivity_setup, "model", path, str,
        default_value="CUSTOM", is_optional=is_thermal_conductivity_optional,
        possible_string_values=THERMAL_CONDUCTIVITY_MODELS)

    thermal_conductivity_value = None
    prandtl_number = None
    sutherland_parameters = None
    if thermal_conductivity_model == "CUSTOM":
        path = get_path_to_key(path_thermal_conductivity, "value")
        thermal_conductivity_value_case_setup = get_setup_value(
            thermal_conductivity_setup, "value", path, (float, str),
            is_optional=is_thermal_conductivity_optional,
            default_value=0.0, numerical_value_condition=(">=", 0.0))

        input_argument_labels = tuple(["T"])
        input_argument_units = tuple(["temperature"])
        thermal_conductivity_value = create_wrapper_for_callable(
            thermal_conductivity_value_case_setup, input_argument_units,
            input_argument_labels, "thermal_conductivity", path, 
            perform_nondim=True, unit_handler=unit_handler, is_scalar=True)

    elif thermal_conductivity_model == "PRANDTL":
        path = get_path_to_key(path_thermal_conductivity, "prandtl_number")
        prandtl_number = get_setup_value(
            thermal_conductivity_setup, "prandtl_number", path, float,
            is_optional=False)

    elif thermal_conductivity_model == "SUTHERLAND":
        sutherland_parameters_path = get_path_to_key(
            path_thermal_conductivity, "sutherland_parameters")
        sutherland_parameters_case_setup = get_setup_value(
            thermal_conductivity_setup, "sutherland_parameters",
            sutherland_parameters_path, list, is_optional=False)

        thermal_conductivity_ref = unit_handler.non_dimensionalize(
            sutherland_parameters_case_setup[0], "thermal_conductivity")
        T_ref = unit_handler.non_dimensionalize(
            sutherland_parameters_case_setup[1], "temperature")
        constant = unit_handler.non_dimensionalize(
            sutherland_parameters_case_setup[2], "temperature")

        sutherland_parameters = SutherlandParameters(
            thermal_conductivity_ref, T_ref, constant)

    elif thermal_conductivity_model == "CHAPMAN-ENSKOG":
        is_chapman_enskog_optional = False

    else:
        raise NotImplementedError

    thermal_conductivity_setup = ThermalConductivitySetup(
        thermal_conductivity_model,
        thermal_conductivity_value,
        prandtl_number,
        sutherland_parameters)

    transport_properties_setup = TransportPropertiesSetup(
        dynamic_viscosity=dynamic_viscosity_setup,
        bulk_viscosity=bulk_viscosity,
        thermal_conductivity=thermal_conductivity_setup)

    return transport_properties_setup

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

def read_stiffened_gas(
        eos_setup: Dict,
        eos_path: str,
        eos_model: str,
        unit_handler: UnitHandler
        ) -> StiffenedGasSetup:

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

    path = get_path_to_key(eos_path, "background_pressure")
    background_pressure = get_setup_value(
        eos_setup, "background_pressure", path, float,
        is_optional=False, numerical_value_condition=(">=", 0.0))
    background_pressure = unit_handler.non_dimensionalize(
        background_pressure, "pressure")

    is_optional = eos_model != "StiffenedGasComplete"
    path = get_path_to_key(eos_path, "energy_translation_factor")
    energy_translation_factor = get_setup_value(
        eos_setup, "energy_translation_factor", path, float,
        default_value=0.0, is_optional=is_optional)
    energy_translation_factor = unit_handler.non_dimensionalize(
        energy_translation_factor, "energy_translation_factor")

    path = get_path_to_key(eos_path, "thermal_energy_factor")
    thermal_energy_factor = get_setup_value(
        eos_setup, "thermal_energy_factor", path, float,
        default_value=0.0, is_optional=is_optional)
    thermal_energy_factor = unit_handler.non_dimensionalize(
        thermal_energy_factor, "thermal_energy_factor")

    stiffened_gas_setup = StiffenedGasSetup(
        specific_heat_ratio=specific_heat_ratio,
        specific_gas_constant=specific_gas_constant,
        background_pressure=background_pressure,
        energy_translation_factor=energy_translation_factor,
        thermal_energy_factor=thermal_energy_factor)
    
    return stiffened_gas_setup

def read_tait(
        eos_setup: Dict,
        eos_path: str,
        unit_handler: UnitHandler
        ) -> TaitSetup:

    path = get_path_to_key(eos_path, "B")
    B_param = get_setup_value(
        eos_setup, "B_param", path, float,
        is_optional=False, numerical_value_condition=(">", 0.0))
    B_param = unit_handler.non_dimensionalize(
        B_param, "pressure")

    path = get_path_to_key(eos_path, "N")
    N_param = get_setup_value(
        eos_setup, "N_param", path, float, is_optional=False,
        numerical_value_condition=(">", 0.0))

    path = get_path_to_key(eos_path, "rho_ref")
    rho_ref = get_setup_value(
        eos_setup, "rho_ref", path, float, is_optional=False,
        numerical_value_condition=(">", 0.0))
    rho_ref = unit_handler.non_dimensionalize(rho_ref, "density")
    
    path = get_path_to_key(eos_path, "p_ref")
    p_ref = get_setup_value(
        eos_setup, "p_ref", path, float, is_optional=False,
        numerical_value_condition=(">", 0.0))
    p_ref = unit_handler.non_dimensionalize(p_ref, "pressure")

    tait_setup = TaitSetup(B_param, N_param, rho_ref, p_ref)
    
    return tait_setup

def read_pairing_properties(
        material_properties_case_setup: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler
        ) -> MaterialPairingProperties:

    basepath = "material_properties"
    active_physics = numerical_setup.active_physics
    is_optional = not active_physics.is_surface_tension

    path_pairing = get_path_to_key(basepath, "pairing")
    pairing_case_setup = get_setup_value(
        material_properties_case_setup, "pairing",
        path_pairing, dict, is_optional=is_optional, default_value={})

    path = get_path_to_key(basepath, "surface_tension_coefficient")
    surface_tension_setup = get_setup_value(
        pairing_case_setup, "surface_tension_coefficient",
        path, float, is_optional=is_optional, default_value=0.0)
    surface_tension_coefficient = unit_handler.non_dimensionalize(
        surface_tension_setup, "surface_tension_coefficient")

    pairing_properties = MaterialPairingProperties(surface_tension_coefficient)

    return pairing_properties
