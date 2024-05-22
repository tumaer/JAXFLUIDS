from typing import NamedTuple

class ActiveForcingsSetup(NamedTuple):
    """ActiveForcingsSetup describes which external forcing 
    terms are active. 

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    is_mass_flow_forcing: bool = False
    is_temperature_forcing: bool = False
    is_turb_hit_forcing: bool = False
    is_acoustic_forcing: bool = False
    is_custom_forcing: bool = False
