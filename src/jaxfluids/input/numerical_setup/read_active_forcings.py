from typing import Dict

from jaxfluids.data_types.numerical_setup import ActiveForcingsSetup
from jaxfluids.input.numerical_setup import get_setup_value, loop_fields

def read_active_forcings_setup(numerical_setup_dict: Dict) -> ActiveForcingsSetup:
    basepath = "active_forcings"
    active_forcings_setup = get_setup_value(
        numerical_setup_dict, "active_forcings", basepath,
        dict, is_optional=True, default_value={})
    active_forcings_setup = loop_fields(ActiveForcingsSetup,
                                        active_forcings_setup, basepath)
    return active_forcings_setup
    



