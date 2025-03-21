from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

from jaxfluids.stencils.reconstruction.shock_capturing.muscl.muscl3 import (
    KOREN, MC, MINMOD, SUPERBEE, VANALBADA, VANLEER
)
from jaxfluids.stencils.reconstruction.shock_capturing.muscl.muscl3_adap import (
    KORENADAP, MCADAP, MINMODADAP, SUPERBEEADAP, VANALBADAADAP, VANLEERADAP
)

from jaxfluids.stencils.reconstruction.shock_capturing.muscl.minmod_ad import MINMODAD
from jaxfluids.stencils.reconstruction.shock_capturing.muscl.minmod_ad_adap import MINMODADADAP

MUSCL_DICT: Dict[str, SpatialReconstruction] = {
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
    "MINMOD-AD":        MINMODAD,
    "MINMOD-AD-ADAP":   MINMODADADAP
}