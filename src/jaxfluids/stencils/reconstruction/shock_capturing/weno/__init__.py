from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno1_js import WENO1
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_js import WENO3JS
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_js_adap import WENO3ADAP
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_n import WENO3N
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_nn_opt1 import WENO3NNOPT1
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_nn_opt2 import WENO3NNOPT2
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_z import WENO3Z
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_z_adap import WENO3ZADAP
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_fp import WENO3FP
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_js import WENO5JS
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_js_adap import WENO5ADAP
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_z import WENO5Z
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_z_adap import WENO5ZADAP
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno6_cu import WENO6CU
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno6_cu_adap import WENO6CUADAP
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno6_cum1 import WENO6CUM1
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno6_cum2 import WENO6CUM2
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno7_js import WENO7JS
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno9_js import WENO9JS
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno_aldm import WENOALDM

WENO_CONST_DICT: Dict[str, SpatialReconstruction] = {
    "WENO1":            WENO1,
    "WENO3-JS":         WENO3JS,
    "WENO3-N":          WENO3N,
    "WENO3-NN-OPT1":    WENO3NNOPT1,
    "WENO3-NN-OPT2":    WENO3NNOPT2,
    "WENO3-Z":          WENO3Z,
    "WENO3-FP":         WENO3FP,
    "WENO5-JS":         WENO5JS,
    "WENO5-Z":          WENO5Z,
    "WENO6-CU":         WENO6CU,
    "WENO6-CUM1":       WENO6CUM1,
    "WENO6-CUM2":       WENO6CUM2,
    "WENO7-JS":         WENO7JS,
    "WENO9-JS":         WENO9JS,
}

WENO_ADAP_DICT: Dict[str, SpatialReconstruction] = {
    "WENO3-JS-ADAP":    WENO3ADAP,
    "WENO3-Z-ADAP":     WENO3ZADAP,
    "WENO5-JS-ADAP":    WENO5ADAP,
    "WENO5-Z-ADAP":     WENO5ZADAP,
    "WENO6-CU-ADAP":    WENO6CUADAP,
}

WENO_DICT = {
    **WENO_CONST_DICT,
    **WENO_ADAP_DICT,
}