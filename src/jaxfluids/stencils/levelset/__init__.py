from typing import Dict

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

from jaxfluids.stencils.levelset.houc3 import HOUC3
from jaxfluids.stencils.levelset.houc5 import HOUC5
from jaxfluids.stencils.levelset.houc7 import HOUC7
from jaxfluids.stencils.levelset.houc9 import HOUC9
from jaxfluids.stencils.levelset.weno3_js_hj import WENO3HJ
from jaxfluids.stencils.levelset.weno5_js_hj import WENO5HJ
from jaxfluids.stencils.levelset.weno7_js_hj import WENO7HJ
from jaxfluids.stencils.levelset.weno9_js_hj import WENO9HJ
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.stencils.levelset.first_deriv_second_order_center import FirstDerivativeSecondOrder

DICT_DERIVATIVE_LEVELSET_ADVECTION: Dict[str, SpatialDerivative] = {
    "HOUC3": HOUC3,
    "HOUC5": HOUC5,
    "HOUC7": HOUC7,
    "HOUC9": HOUC9,
}

DICT_DERIVATIVE_REINITIALIZATION: Dict[str, SpatialDerivative] = {
    "WENO3HJ": WENO3HJ,
    "WENO5HJ": WENO5HJ,
    "WENO7HJ": WENO7HJ,
    "WENO9HJ": WENO9HJ,
}