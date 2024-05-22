from typing import Dict, Tuple, Any

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup import CaseSetup
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_information import (EquationInformation, AVAILABLE_QUANTITIES,
                                            TUPLE_PRIMES, TUPLE_CHEMICAL_COMPONENTS)
from jaxfluids.input.setup_reader import SetupReader
from jaxfluids.input.case_setup.read_boundary_conditions import read_boundary_condition_setup
from jaxfluids.input.case_setup.read_domain import read_domain_setup
from jaxfluids.input.case_setup.read_forcing import read_forcing_setup
from jaxfluids.input.case_setup.read_general import read_general_setup
from jaxfluids.input.case_setup.read_initial_conditions import read_initial_condition_setup
from jaxfluids.input.case_setup.read_material_manager import read_material_manager_setup
from jaxfluids.input.case_setup.read_restart import read_restart_setup
from jaxfluids.input.case_setup.read_solid_properties import read_solid_properties_setup
from jaxfluids.input.case_setup.read_output_quantities import read_output_quantities
from jaxfluids.input.case_setup import get_setup_value, get_path_to_key

class CaseSetupReader(SetupReader):

    def __init__(
            self,
            unit_handler: UnitHandler,
            ) -> None:

        super(CaseSetupReader, self).__init__(unit_handler)

    def initialize_case_setup(
            self,
            case_setup_dict: Dict,
            numerical_setup: NumericalSetup
            ) -> Tuple[CaseSetup, EquationInformation]:

        equation_information = self.get_equation_information(
            case_setup_dict, numerical_setup)

        general_setup = read_general_setup(
            case_setup_dict, numerical_setup,
            self.unit_handler)
        restart_setup = read_restart_setup(
            case_setup_dict, numerical_setup,
            self.unit_handler)
        domain_setup = read_domain_setup(
            case_setup_dict, numerical_setup,
            self.unit_handler)
        boundary_condition_setup = read_boundary_condition_setup(
            case_setup_dict, equation_information,
            numerical_setup, self.unit_handler, domain_setup)
        initial_condition_setup = read_initial_condition_setup(
            case_setup_dict, equation_information, numerical_setup,
            self.unit_handler, domain_setup)
        material_manager_setup = read_material_manager_setup(
            case_setup_dict, equation_information,
            numerical_setup, self.unit_handler)
        solid_properties_setup = read_solid_properties_setup(
            case_setup_dict, equation_information,
            numerical_setup, self.unit_handler, domain_setup)
        forcing_setup = read_forcing_setup(
            case_setup_dict, equation_information,
            numerical_setup, self.unit_handler, domain_setup)
        output_quantities_setup = read_output_quantities(
            case_setup_dict, numerical_setup)

        case_setup = CaseSetup(
            general_setup=general_setup,
            restart_setup=restart_setup,
            domain_setup=domain_setup,
            boundary_condition_setup=boundary_condition_setup,
            initial_condition_setup=initial_condition_setup,
            material_manager_setup=material_manager_setup,
            solid_properties_setup=solid_properties_setup,
            forcing_setup=forcing_setup,
            output_quantities_setup=output_quantities_setup)

        return case_setup, equation_information

    def get_equation_information(
            self,
            case_setup_dict: Dict,
            numerical_setup: NumericalSetup
            ) -> EquationInformation:
        """Initializes an EquationInformation based on
        the numerical setup .

        :return: _description_
        :rtype: EquationInformation
        """
        # DEFAULT VALUES FOR SINGLE PHASE
        fluid_names = ("fluid_0",)
        primes_tuple = TUPLE_PRIMES

        # LEVELSET INTERFACE INTERACTION TYPE
        levelset_model = numerical_setup.levelset.model
        if levelset_model == "FLUID-FLUID":
            fluid_names = ("positive", "negative")

        # DIFFUSE INTERFACE MODEL
        diffuse_interface_model = numerical_setup.diffuse_interface.model
        if diffuse_interface_model:
            basepath = "material_properties"
            material_properties_case_setup = get_setup_value(
                case_setup_dict, "material_properties", basepath, dict,
                is_optional=False)
            path = get_path_to_key(basepath, "fluid_names")
            fluid_names = get_setup_value(
                material_properties_case_setup, "fluid_names", path, list,
                is_optional=False)
            no_fluids = len(fluid_names)
            alpharho_tuple = tuple([f"alpharho_{i:d}" for i in range(no_fluids)])
            alpha_tuple = tuple([f"alpha_{i:d}" for i in range(no_fluids - 1)])
            primes_tuple = tuple([p for p in TUPLE_PRIMES if p != "rho"])
            if diffuse_interface_model == "5EQM":
                primes_tuple = alpharho_tuple + primes_tuple + alpha_tuple
            else:
                raise NotImplementedError
            AVAILABLE_QUANTITIES["primitives"] += alpharho_tuple + alpha_tuple
            AVAILABLE_QUANTITIES["conservatives"] += alpharho_tuple + alpha_tuple

        equation_information = EquationInformation(
            primes_tuple=primes_tuple,
            fluid_names=fluid_names,
            levelset_model=levelset_model,
            diffuse_interface_model=diffuse_interface_model,
            active_physics=numerical_setup.active_physics,
            active_forcings=numerical_setup.active_forcings)

        return equation_information
