from typing import NamedTuple

class ActivePhysicsSetup(NamedTuple):
    """ActivePhysicsSetup describes which physical contributions
    in the Navier-Stokes equations are active. 

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    is_convective_flux: bool
    is_viscous_flux: bool = False
    is_heat_flux: bool = False
    is_volume_force: bool = False
    is_surface_tension: bool = False
    is_geometric_source: bool = False