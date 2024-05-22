from jaxfluids.iles.ALDM_WENO1 import ALDM_WENO1
from jaxfluids.iles.ALDM_WENO3 import ALDM_WENO3
from jaxfluids.iles.ALDM_WENO5 import ALDM_WENO5

TUPLE_SMOOTHNESS_MEASURE = (
    "TV", "WENO"
)

TUPLE_WALL_DAMPING = (
    "COHERENTSTRUCTURE", "VANDRIEST"
)

TUPLE_SHOCK_SENSOR = (
    "DUCROS"
)