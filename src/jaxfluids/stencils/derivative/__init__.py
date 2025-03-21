from typing import Dict

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

from jaxfluids.stencils.derivative.deriv_center_2 import DerivativeSecondOrderCenter
from jaxfluids.stencils.derivative.deriv_center_adap_2 import DerivativeSecondOrderAdapCenter
from jaxfluids.stencils.derivative.deriv_center_4 import DerivativeFourthOrderCenter
from jaxfluids.stencils.derivative.deriv_center_adap_4 import DerivativeFourthOrderAdapCenter
from jaxfluids.stencils.derivative.deriv_center_6 import DerivativeSixthOrderCenter
from jaxfluids.stencils.derivative.deriv_center_adap_6 import DerivativeSixthOrderAdapCenter
from jaxfluids.stencils.derivative.deriv_center_8 import DerivativeEighthOrderCenter
from jaxfluids.stencils.derivative.deriv_center_adap_8 import DerivativeEighthOrderAdapCenter

from jaxfluids.stencils.derivative.second_deriv_second_order_center import SecondDerivativeSecondOrderCenter
from jaxfluids.stencils.derivative.second_deriv_fourth_order_center import SecondDerivativeFourthOrderCenter

from jaxfluids.stencils.derivative.deriv_face_2 import DerivativeSecondOrderFace
from jaxfluids.stencils.derivative.deriv_face_adap_2 import DerivativeSecondOrderAdapFace
from jaxfluids.stencils.derivative.deriv_face_4 import DerivativeFourthOrderFace
from jaxfluids.stencils.derivative.deriv_face_adap_4 import DerivativeFourthOrderAdapFace
from jaxfluids.stencils.derivative.deriv_face_6 import DerivativeSixthOrderFace
from jaxfluids.stencils.derivative.deriv_face_adap_6 import DerivativeSixthOrderAdapFace
from jaxfluids.stencils.derivative.deriv_face_8 import DerivativeEighthOrderFace
from jaxfluids.stencils.derivative.deriv_face_adap_8 import DerivativeEighthOrderAdapFace


DICT_FIRST_DERIVATIVE_CENTER: Dict[str, SpatialDerivative] = {
    "CENTRAL2":         DerivativeSecondOrderCenter,
    "CENTRAL2-ADAP":    DerivativeSecondOrderAdapCenter,
    "CENTRAL4":         DerivativeFourthOrderCenter,
    "CENTRAL4-ADAP":    DerivativeFourthOrderAdapCenter,
    "CENTRAL6":         DerivativeSixthOrderCenter,
    "CENTRAL6-ADAP":    DerivativeSixthOrderAdapCenter,
    "CENTRAL8":         DerivativeEighthOrderCenter,
    "CENTRAL8-ADAP":    DerivativeEighthOrderAdapCenter,
}

DICT_SECOND_DERIVATIVE_CENTER: Dict[str, SpatialDerivative] = {
    "CENTRAL2":  SecondDerivativeSecondOrderCenter,
    "CENTRAL4":  SecondDerivativeFourthOrderCenter,
}

DICT_DERIVATIVE_FACE: Dict[str, SpatialDerivative] = {
    "CENTRAL2":         DerivativeSecondOrderFace,
    "CENTRAL2-ADAP":    DerivativeSecondOrderAdapFace,
    "CENTRAL4":         DerivativeFourthOrderFace, 
    "CENTRAL4-ADAP":    DerivativeFourthOrderAdapFace,
    "CENTRAL6":         DerivativeSixthOrderFace,
    "CENTRAL6-ADAP":    DerivativeSixthOrderAdapFace,
    "CENTRAL8":         DerivativeEighthOrderFace,
    "CENTRAL8-ADAP":    DerivativeEighthOrderAdapFace,
}