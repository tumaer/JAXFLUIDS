from typing import Dict

from jaxfluids.data_types.numerical_setup import CavitationSetup
from jaxfluids.input.numerical_setup import get_setup_value, get_path_to_key

def read_cavitation_setup(numerical_setup_dict: Dict) -> CavitationSetup:

    basepath = "cavitation"

    cavitation_dict = get_setup_value(
        numerical_setup_dict, "cavitation", basepath, dict,
        is_optional=True, default_value={})

    # MODEL
    path = get_path_to_key(basepath, "model")
    model = get_setup_value(
        cavitation_dict, "model", path, str,
        is_optional=True, default_value=False)
    # TODO INTRODUCE CAVITATION MODULE
    
    cavitation_setup = CavitationSetup(
        model)

    return cavitation_setup
    
