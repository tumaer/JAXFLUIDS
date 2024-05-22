from typing import Dict, Any

from jaxfluids.input.setup_reader import SetupReader
from jaxfluids.data_types.numerical_setup import *
from jaxfluids.unit_handler import UnitHandler

from jaxfluids.input.numerical_setup.read_conservatives import read_conservatives_setup
from jaxfluids.input.numerical_setup.read_active_physics import read_active_physics_setup
from jaxfluids.input.numerical_setup.read_levelset import read_levelset_setup
from jaxfluids.input.numerical_setup.read_diffuse_interface import read_diffuse_interface_setup
from jaxfluids.input.numerical_setup.read_active_forcings import read_active_forcings_setup
from jaxfluids.input.numerical_setup.read_turbulence_statistics import read_turbulence_statistics_setup
from jaxfluids.input.numerical_setup.read_output import read_output_setup
from jaxfluids.input.numerical_setup.read_precision import read_precision_setup

class NumericalSetupReader(SetupReader):

    def __init__(
            self,
            unit_handler: UnitHandler
            ) -> None:

        super(NumericalSetupReader, self).__init__(unit_handler)

    def initialize_numerical_setup(self, numerical_setup_dict: Dict) -> NumericalSetup:

        active_physics_setup = read_active_physics_setup(numerical_setup_dict)
        active_forcings_setup = read_active_forcings_setup(numerical_setup_dict)
        conservatives_setup = read_conservatives_setup(numerical_setup_dict,
                                                       active_physics_setup,
                                                       self.unit_handler)
        levelset_setup = read_levelset_setup(numerical_setup_dict,
                                             self.unit_handler,
                                             conservatives_setup)
        diffuse_interface_setup = read_diffuse_interface_setup(numerical_setup_dict,
                                                               conservatives_setup,
                                                               self.unit_handler)
        turbulence_statistics_setup = read_turbulence_statistics_setup(numerical_setup_dict, self.unit_handler)
        output_setup = read_output_setup(numerical_setup_dict, conservatives_setup)
        precision_setup = read_precision_setup(numerical_setup_dict)

        numerical_setup = NumericalSetup(
            conservatives_setup,
            levelset_setup,
            diffuse_interface_setup,
            active_physics_setup,
            active_forcings_setup,
            turbulence_statistics_setup,
            output_setup,
            precision_setup)

        self.sanity_check(numerical_setup)

        return numerical_setup
    
    
    def sanity_check(self, numerical_setup: NumericalSetup) -> None:

        levelset_model = numerical_setup.levelset.model
        diffuse_interface_model = numerical_setup.diffuse_interface.model

        if levelset_model:
            model_tuple = ("diffuse_interface",)
            for model_str in model_tuple:
                model_setup = getattr(numerical_setup, model_str)
                assert_string = (
                    "Consistency error in numerical setup file. "
                    f"Levelset and {model_str:s} model can not be used "
                    "simultaneously."
                )
                assert not model_setup.model, assert_string

            is_interpolation_limiter = numerical_setup.conservatives.positivity.is_interpolation_limiter
            assert_string = (
                "Consistency error in numerical setup file. "
                f"Active levelset requires active interpolation limiter. "
                "In the numerical setup, set positivity/is_interpolation_limiter true."
            )
            assert is_interpolation_limiter, assert_string


        if diffuse_interface_model:
            model_tuple = ("levelset",)
            for model_str in model_tuple:
                model_setup = getattr(numerical_setup, model_str)
                assert_string = (
                    "Consistency error in numerical setup file. "
                    f"Diffuse interface and {model_str:s} model can not be used "
                    "simultaneously."
                )
                assert not model_setup.model, assert_string
            
            if numerical_setup.conservatives.positivity.is_thinc_interpolation_limiter:
                assert_string = (
                    "Consistency error in numerical setup file. "
                    "THINC reconstruction limiter is active, but THINC is turned off."
                )
                assert numerical_setup.diffuse_interface.thinc.is_thinc_reconstruction, assert_string
            
            if numerical_setup.conservatives.positivity.is_acdi_flux_limiter:
                assert_string = (
                    "Consistency error in numerical setup file. "
                    "ACDI flux limiter is active, but ACDI flux is turned off."
                )
                assert numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening, assert_string
            
