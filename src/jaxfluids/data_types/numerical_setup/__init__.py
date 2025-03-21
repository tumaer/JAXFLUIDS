from typing import NamedTuple

from jaxfluids.data_types.numerical_setup.active_forcings import ActiveForcingsSetup
from jaxfluids.data_types.numerical_setup.active_physics import ActivePhysicsSetup
from jaxfluids.data_types.numerical_setup.cavitation import CavitationSetup
from jaxfluids.data_types.numerical_setup.conservatives import ConservativesSetup
from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup, SolidCouplingSetup
from jaxfluids.data_types.numerical_setup.output import OutputSetup
from jaxfluids.data_types.numerical_setup.precision import PrecisionSetup


class NumericalSetup(NamedTuple):
    conservatives: ConservativesSetup
    levelset: LevelsetSetup
    diffuse_interface: DiffuseInterfaceSetup
    cavitation: CavitationSetup
    active_physics: ActivePhysicsSetup
    active_forcings: ActiveForcingsSetup
    output: OutputSetup
    precision: PrecisionSetup
