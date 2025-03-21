from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

from jaxfluids.stencils.reconstruction.shock_capturing.muscl import MUSCL_DICT
from jaxfluids.stencils.reconstruction.shock_capturing.teno import TENO_DICT
from jaxfluids.stencils.reconstruction.shock_capturing.weno import WENO_DICT

SHOCK_CAPTURING_DICT: Dict[str, SpatialReconstruction] = {
    **MUSCL_DICT,
    **TENO_DICT,
    **WENO_DICT
}