from jaxfluids.solvers.convective_fluxes.high_order_godunov import HighOrderGodunov
from jaxfluids.solvers.convective_fluxes.flux_splitting_scheme import FluxSplittingScheme
from jaxfluids.iles.ALDM import ALDM
from jaxfluids.solvers.convective_fluxes.central_scheme import CentralScheme


DICT_CONVECTIVE_SOLVER = {
    "GODUNOV"           : HighOrderGodunov, 
    "FLUX-SPLITTING"    : FluxSplittingScheme, 
    "ALDM"              : ALDM,
    "CENTRAL"           : CentralScheme,
}

TUPLE_CONVECTIVE_SOLVER = tuple(DICT_CONVECTIVE_SOLVER.keys())