from typing import NamedTuple

class ActiveForcingsSetup(NamedTuple):
    """NamedTuple describing which external forcings are active.
    """
    is_mass_flow_forcing: bool = False
    is_temperature_forcing: bool = False
    is_solid_temperature_forcing: bool = False
    is_turb_hit_forcing: bool = False
    is_acoustic_forcing: bool = False
    is_custom_forcing: bool = False
    is_sponge_layer_forcing: bool = False
    is_enthalpy_damping: bool = False