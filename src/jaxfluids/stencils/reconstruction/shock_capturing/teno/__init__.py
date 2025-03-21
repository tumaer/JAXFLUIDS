from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno5 import TENO5
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno5_a import TENO5A
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno5_adap import TENO5ADAP
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno6 import TENO6
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno6_adap import TENO6ADAP
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno6_a import TENO6A
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno6_a_adap import TENO6AADAP
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno8 import TENO8
from jaxfluids.stencils.reconstruction.shock_capturing.teno.teno8_a import TENO8A

TENO_DICT: Dict[str, SpatialReconstruction] = {
    "TENO5":            TENO5,
    "TENO5-A":          TENO5A,
    "TENO5-ADAP":       TENO5ADAP,
    "TENO6":            TENO6,
    "TENO6-ADAP":       TENO6ADAP,
    "TENO6-A":          TENO6A,
    "TENO6-A-ADAP":     TENO6AADAP,
    "TENO8":            TENO8,
    "TENO8-A":          TENO8A,
}