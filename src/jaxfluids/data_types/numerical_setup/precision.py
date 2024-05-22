from typing import NamedTuple, Tuple

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class Epsilons(NamedTuple):
    density: float
    pressure: float
    volume_fraction: float

class PrecisionSetup(NamedTuple):
    is_double_precision_compute: bool
    is_double_precision_output: bool
    epsilon: float
    smallest_normal: float
    fmax: float
    spatial_stencil_epsilon: float
    interpolation_limiter_epsilons: Epsilons
    flux_limiter_epsilons: Epsilons
    thinc_limiter_epsilons: Epsilons
