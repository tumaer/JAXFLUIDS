from jaxfluids.solvers.riemann_solvers.LaxFriedrichs import LaxFriedrichs
from jaxfluids.solvers.riemann_solvers.HLL import HLL
from jaxfluids.solvers.riemann_solvers.HLLC import HLLC
from jaxfluids.solvers.riemann_solvers.HLLC_simplealpha import HLLC_SIMPLEALPHA
from jaxfluids.solvers.riemann_solvers.HLLCLM import HLLCLM
from jaxfluids.solvers.riemann_solvers.Rusanov import Rusanov
from jaxfluids.solvers.riemann_solvers.AUSMP import AUSMP
from jaxfluids.solvers.riemann_solvers.CATUM import CATUM

from jaxfluids.solvers.riemann_solvers.signal_speeds import (
    signal_speed_Arithmetic, signal_speed_Rusanov, 
    signal_speed_Davis, signal_speed_Davis_2,
    signal_speed_Einfeldt, signal_speed_Toro)


DICT_RIEMANN_SOLVER ={
    'LAX-FRIEDRICHS': LaxFriedrichs,
    'HLL': HLL,
    'HLLC': HLLC,
    'HLLC_SIMPLEALPHA': HLLC_SIMPLEALPHA,
    'HLLC-LM': HLLCLM,
    'RUSANOV': Rusanov,
    'AUSMP': AUSMP,
    'CATUM': CATUM,
}

DICT_SIGNAL_SPEEDS ={
    'ARITHMETIC': signal_speed_Arithmetic,
    'RUSANOV': signal_speed_Rusanov,
    'DAVIS': signal_speed_Davis,
    'DAVIS2': signal_speed_Davis_2,
    'EINFELDT': signal_speed_Einfeldt,
    'TORO': signal_speed_Toro,
}

TUPLE_CATUM_TRANSPORT_VELOCITIES = (
    "EGERER", "SCHMIDT", "SEZAL", "MIHATSCH", "KYRIAZIS")
