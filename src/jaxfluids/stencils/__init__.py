from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.reconstruction.weno1_js import WENO1
from jaxfluids.stencils.reconstruction.weno3_js import WENO3
from jaxfluids.stencils.reconstruction.weno3_js_adap import WENO3ADAP
from jaxfluids.stencils.reconstruction.weno3_n import WENO3N
from jaxfluids.stencils.reconstruction.weno3_nn_opt1 import WENO3NNOPT1
from jaxfluids.stencils.reconstruction.weno3_nn_opt2 import WENO3NNOPT2
from jaxfluids.stencils.reconstruction.weno3_z import WENO3Z
from jaxfluids.stencils.reconstruction.weno3_z_adap import WENO3ZADAP
from jaxfluids.stencils.reconstruction.weno3_fp import WENO3FP
from jaxfluids.stencils.reconstruction.weno5_js import WENO5
from jaxfluids.stencils.reconstruction.weno5_js_adap import WENO5ADAP
from jaxfluids.stencils.reconstruction.weno5_z import WENO5Z
from jaxfluids.stencils.reconstruction.weno5_z_adap import WENO5ZADAP
from jaxfluids.stencils.reconstruction.weno6_cu import WENO6CU
from jaxfluids.stencils.reconstruction.weno6_cu_adap import WENO6CUADAP
from jaxfluids.stencils.reconstruction.weno6_cum1 import WENO6CUM1
from jaxfluids.stencils.reconstruction.weno6_cum2 import WENO6CUM2
from jaxfluids.stencils.reconstruction.weno6_cumd import WENO6CUMD
from jaxfluids.stencils.reconstruction.weno7_js import WENO7
from jaxfluids.stencils.reconstruction.weno9_js import WENO9
from jaxfluids.stencils.reconstruction.teno5 import TENO5
from jaxfluids.stencils.reconstruction.teno5_a import TENO5A
from jaxfluids.stencils.reconstruction.teno5_adap import TENO5ADAP
from jaxfluids.stencils.reconstruction.teno6 import TENO6
from jaxfluids.stencils.reconstruction.teno6_adap import TENO6ADAP
from jaxfluids.stencils.reconstruction.teno6_a import TENO6A
from jaxfluids.stencils.reconstruction.teno6_a_adap import TENO6AADAP
from jaxfluids.stencils.reconstruction.teno8 import TENO8
from jaxfluids.stencils.reconstruction.teno8_a import TENO8A
from jaxfluids.stencils.reconstruction.weno_aldm import WENOALDM
from jaxfluids.stencils.reconstruction.muscl3 import (
    MUSCL3, KOREN, MC, MINMOD, SUPERBEE, VANALBADA, VANLEER
)
from jaxfluids.stencils.reconstruction.muscl3_adap import (
    KORENADAP, MCADAP, MINMODADAP, SUPERBEEADAP, VANALBADAADAP, VANLEERADAP
)

DICT_SPATIAL_RECONSTRUCTION: Dict[str, SpatialReconstruction] = {
    "WENO1":            WENO1,
    "WENO3-JS":         WENO3,
    "WENO3-JS-ADAP":    WENO3ADAP,
    "WENO3-N":          WENO3N,
    "WENO3-NN-OPT1":    WENO3NNOPT1,
    "WENO3-NN-OPT2":    WENO3NNOPT2,
    "WENO3-Z":          WENO3Z,
    "WENO3-Z-ADAP":     WENO3ZADAP,
    "WENO3-FP":         WENO3FP,
    "WENO5-JS":         WENO5,
    "WENO5-JS-ADAP":    WENO5ADAP,
    "WENO5-Z":          WENO5Z,
    "WENO5-Z-ADAP":     WENO5ZADAP,
    "WENO6-CU":         WENO6CU,
    "WENO6-CU-ADAP":    WENO6CUADAP,
    "WENO6-CUM1":       WENO6CUM1,
    "WENO6-CUM2":       WENO6CUM2,
    "WENO6-CUMD":       WENO6CUMD,
    "WENO7-JS":         WENO7,
    "WENO9-JS":         WENO9,
    "TENO5":            TENO5,
    "TENO5-A":          TENO5A,
    "TENO5-ADAP":       TENO5ADAP,
    "TENO6":            TENO6,
    "TENO6-ADAP":       TENO6ADAP,
    "TENO6-A":          TENO6A,
    "TENO6-A-ADAP":     TENO6AADAP,
    "TENO8":            TENO8,
    "TENO8-A":          TENO8A,
    # "MUSCL3":           MUSCL3,
    "KOREN":            KOREN,
    "MC":               MC,
    "MINMOD":           MINMOD,
    "SUPERBEE":         SUPERBEE,
    "VANALBADA":        VANALBADA,
    "VANLEER":          VANLEER,
    "KOREN-ADAP":       KORENADAP,
    "MC-ADAP":          MCADAP,
    "MINMOD-ADAP":      MINMODADAP,
    "SUPERBEE-ADAP":    SUPERBEEADAP,
    "VANALBADA-ADAP":   VANALBADAADAP,
    "VANLEER-ADAP":     VANLEERADAP,
}


from jaxfluids.stencils.reconstruction.central_2 import CentralSecondOrderReconstruction
from jaxfluids.stencils.reconstruction.central_adap_2 import CentralSecondOrderAdapReconstruction
from jaxfluids.stencils.reconstruction.central_4 import CentralFourthOrderReconstruction
from jaxfluids.stencils.reconstruction.central_adap_4 import CentralFourthOrderAdapReconstruction
from jaxfluids.stencils.reconstruction.central_6 import CentralSixthOrderReconstruction
from jaxfluids.stencils.reconstruction.central_adap_6 import CentralSixthOrderAdapReconstruction
from jaxfluids.stencils.reconstruction.central_8 import CentralEighthOrderReconstruction
from jaxfluids.stencils.reconstruction.central_adap_8 import CentralEighthOrderAdapReconstruction

DICT_CENTRAL_RECONSTRUCTION: Dict[str, SpatialReconstruction] = {
    "CENTRAL2":         CentralSecondOrderReconstruction,
    "CENTRAL2_ADAP":    CentralSecondOrderAdapReconstruction,
    "CENTRAL4":         CentralFourthOrderReconstruction,
    "CENTRAL4_ADAP":    CentralFourthOrderAdapReconstruction,
    "CENTRAL6":         CentralSixthOrderReconstruction,
    "CENTRAL6_ADAP":    CentralSixthOrderAdapReconstruction,
    "CENTRAL8":         CentralEighthOrderReconstruction,
    "CENTRAL8_ADAP":    CentralEighthOrderAdapReconstruction,
}

DICT_SPATIAL_RECONSTRUCTION["CENTRAL2"] = CentralSecondOrderReconstruction


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
    "CENTRAL2_ADAP":    DerivativeSecondOrderAdapCenter,
    "CENTRAL4":         DerivativeFourthOrderCenter,
    "CENTRAL4_ADAP":    DerivativeFourthOrderAdapCenter,
    "CENTRAL6":         DerivativeSixthOrderCenter,
    "CENTRAL6_ADAP":    DerivativeSixthOrderAdapCenter,
    "CENTRAL8":         DerivativeEighthOrderCenter,
    "CENTRAL8_ADAP":    DerivativeEighthOrderAdapCenter,
}

DICT_SECOND_DERIVATIVE_CENTER: Dict[str, SpatialDerivative] = {
    "CENTRAL2":  SecondDerivativeSecondOrderCenter,
    "CENTRAL4":  SecondDerivativeFourthOrderCenter,
}

DICT_DERIVATIVE_FACE: Dict[str, SpatialDerivative] = {
    "CENTRAL2":         DerivativeSecondOrderFace,
    "CENTRAL2_ADAP":    DerivativeSecondOrderAdapFace,
    "CENTRAL4":         DerivativeFourthOrderFace, 
    "CENTRAL4_ADAP":    DerivativeFourthOrderAdapFace,
    "CENTRAL6":         DerivativeSixthOrderFace,
    "CENTRAL6_ADAP":    DerivativeSixthOrderAdapFace,
    "CENTRAL8":         DerivativeEighthOrderFace,
    "CENTRAL8_ADAP":    DerivativeEighthOrderAdapFace,
}


from jaxfluids.stencils.levelset.houc3 import HOUC3
from jaxfluids.stencils.levelset.houc5 import HOUC5
from jaxfluids.stencils.levelset.houc7 import HOUC7
from jaxfluids.stencils.levelset.weno3_deriv import WENO3DERIV
from jaxfluids.stencils.levelset.weno5_deriv import WENO5DERIV
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.stencils.levelset.first_deriv_second_order_center import FirstDerivativeSecondOrder

DICT_DERIVATIVE_QUANTITY_EXTENDER: Dict[str, SpatialDerivative] = {
    "FIRSTORDER":   FirstDerivativeFirstOrderCenter, 
    "SECONDORDER":  FirstDerivativeSecondOrder, 
}

DICT_DERIVATIVE_LEVELSET_ADVECTION: Dict[str, SpatialDerivative] = {
    "HOUC3": HOUC3,
    "HOUC5": HOUC5,
    "HOUC7": HOUC7
}

DICT_DERIVATIVE_REINITIALIZATION: Dict[str, SpatialDerivative] = {
    "FIRSTORDER": FirstDerivativeFirstOrderCenter,
    "WENO3DERIV": WENO3DERIV,
    "WENO5DERIV": WENO5DERIV,
}