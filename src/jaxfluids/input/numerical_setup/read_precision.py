from typing import Dict

from jaxfluids.data_types.numerical_setup.precision import *
from jaxfluids.input.numerical_setup import get_setup_value, get_path_to_key

def read_precision_setup(
        numerical_setup_dict: dict
        ) -> PrecisionSetup:

    basepath = "precision"

    precision_dict = get_setup_value(
        numerical_setup_dict, "precision", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "is_double_precision_compute")
    is_double_precision_compute = get_setup_value(
        precision_dict, "is_double_precision_compute", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "is_double_precision_output")
    is_double_precision_output = get_setup_value(
        precision_dict, "is_double_precision_output", path, bool,
        is_optional=True, default_value=True)

    path = get_path_to_key(basepath, "epsilon")
    epsilon = get_setup_value(
        precision_dict, "epsilon", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">", 0.0))
    
    path = get_path_to_key(basepath, "smallest_normal")
    smallest_normal = get_setup_value(
        precision_dict, "smallest_normal", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">", 0.0))

    path = get_path_to_key(basepath, "fmax")
    fmax = get_setup_value(
        precision_dict, "fmax", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">", 0.0))

    path = get_path_to_key(basepath, "spatial_stencil_epsilon")
    spatial_stencil_epsilon = get_setup_value(
        precision_dict, "spatial_stencil_epsilon", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">", 0.0))

    path = get_path_to_key(basepath, "interpolation_limiter_epsilon")
    interpolation_limiter_epsilon_dict = get_setup_value(
        precision_dict, "interpolation_limiter_epsilon", path, dict,
        is_optional=True, default_value={})
    interpolation_limiter_epsilon = read_epsilon(interpolation_limiter_epsilon_dict, path)

    path = get_path_to_key(basepath, "flux_limiter_epsilon")
    flux_limiter_epsilon_dict = get_setup_value(
        precision_dict, "flux_limiter_epsilon", path, dict,
        is_optional=True, default_value={})
    flux_limiter_epsilon = read_epsilon(flux_limiter_epsilon_dict, path)

    path = get_path_to_key(basepath, "thinc_limiter_epsilon")
    thinc_limiter_epsilon_dict = get_setup_value(
        precision_dict, "thinc_limiter_epsilon", path, dict,
        is_optional=True, default_value={})
    thinc_limiter_epsilon = read_epsilon(thinc_limiter_epsilon_dict, path)

    precision_setup = PrecisionSetup(
        is_double_precision_compute,
        is_double_precision_output,
        epsilon, smallest_normal,
        fmax, spatial_stencil_epsilon,
        interpolation_limiter_epsilon,
        flux_limiter_epsilon,
        thinc_limiter_epsilon)

    return precision_setup

def read_epsilon(epsilon_dict: Dict, basepath: str):

    path = get_path_to_key(basepath, "density")
    density = get_setup_value(
        epsilon_dict, "density", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">=", 0.0))

    path = get_path_to_key(basepath, "pressure")
    pressure = get_setup_value(
        epsilon_dict, "pressure", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">=", 0.0))
    
    path = get_path_to_key(basepath, "volume_fraction")
    volume_fraction = get_setup_value(
        epsilon_dict, "volume_fraction", path, float,
        is_optional=True, default_value=None,
        numerical_value_condition=(">=", 0.0))

    epsilon = Epsilons(density, pressure, volume_fraction)

    return epsilon