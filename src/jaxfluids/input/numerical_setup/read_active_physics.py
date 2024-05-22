from typing import Dict

from jaxfluids.data_types.numerical_setup import ActivePhysicsSetup
from jaxfluids.input.numerical_setup import get_setup_value, loop_fields

def read_active_physics_setup(numerical_setup_dict: Dict) -> ActivePhysicsSetup:
    basepath = "active_physics"
    active_physics_dict = get_setup_value(
        numerical_setup_dict, "active_physics", basepath,
        dict, is_optional=False)
    active_physics_setup = loop_fields(ActivePhysicsSetup, active_physics_dict, basepath)
    return active_physics_setup

