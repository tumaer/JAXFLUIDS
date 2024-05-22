from typing import Dict

from jaxfluids.data_types.numerical_setup.conservatives import PositivitySetup
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup
from jaxfluids.solvers import TUPLE_POSITIVITY_FIXES
from jaxfluids.input.numerical_setup import get_setup_value, get_path_to_key

def read_positivity_setup(
        conservatives_dict: Dict,
        ) -> PositivitySetup:

    basepath = "conservatives"
    default_value = False
    
    path_positivity = get_path_to_key(basepath, "positivity")    
    positivity_dict = get_setup_value(
        conservatives_dict, "positivity", path_positivity, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(path_positivity, "flux_limiter")
    flux_limiter = get_setup_value(
        positivity_dict, "flux_limiter", path, str,
        is_optional=True, default_value=default_value, 
        possible_string_values=TUPLE_POSITIVITY_FIXES)
        
    path = get_path_to_key(path_positivity, "is_interpolation_limiter")
    is_interpolation_limiter = get_setup_value(
        positivity_dict, "is_interpolation_limiter", path, bool,
        is_optional=True, default_value=default_value)

    path = get_path_to_key(path_positivity, "is_thinc_interpolation_limiter")
    is_thinc_interpolation_limiter = get_setup_value(
        positivity_dict, "is_thinc_interpolation_limiter", path, bool,
        is_optional=True, default_value=default_value)
    
    path = get_path_to_key(path_positivity, "is_volume_fraction_limiter")
    is_volume_fraction_limiter = get_setup_value(
        positivity_dict, "is_volume_fraction_limiter", path, bool,
        is_optional=True, default_value=default_value)
    
    path = get_path_to_key(path_positivity, "is_acdi_flux_limiter")
    is_acdi_flux_limiter = get_setup_value(
        positivity_dict, "is_acdi_flux_limiter", path, bool,
        is_optional=True, default_value=default_value)
    
    path = get_path_to_key(path_positivity, "is_logging")
    is_logging = get_setup_value(
        positivity_dict, "is_logging", path, bool,
        is_optional=True, default_value=False)
    
    positivity_setup = PositivitySetup(
        flux_limiter,
        is_interpolation_limiter,
        is_thinc_interpolation_limiter,
        is_volume_fraction_limiter,
        is_acdi_flux_limiter,
        is_logging
    )

    return positivity_setup