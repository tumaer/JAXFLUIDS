from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

from jaxfluids.stencils.reconstruction import SHOCK_CAPTURING_DICT, CENTRAL_RECONSTRUCTION_DICT
from jaxfluids.stencils.derivative import (
    DICT_DERIVATIVE_FACE, 
    DICT_FIRST_DERIVATIVE_CENTER,
    DICT_SECOND_DERIVATIVE_CENTER)
from jaxfluids.stencils.levelset import (
    DICT_DERIVATIVE_LEVELSET_ADVECTION,
    DICT_DERIVATIVE_REINITIALIZATION)

DICT_SPATIAL_RECONSTRUCTION: Dict[str, SpatialReconstruction] = {
    **SHOCK_CAPTURING_DICT, 
}

DICT_SPATIAL_RECONSTRUCTION["CENTRAL2"] = CENTRAL_RECONSTRUCTION_DICT["CENTRAL2"]