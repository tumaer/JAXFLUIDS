from typing import NamedTuple

class ActivePhysicsSetup(NamedTuple):
    """NamedTuple describing which fluxes and source terms
    are active.
    """
    is_convective_flux: bool
    is_viscous_flux: bool = False
    is_heat_flux: bool = False
    is_volume_force: bool = False
    is_surface_tension: bool = False
    is_geometric_source: bool = False
    is_viscous_heat_production: bool = True