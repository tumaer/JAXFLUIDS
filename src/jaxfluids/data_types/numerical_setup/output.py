from typing import NamedTuple, Tuple

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class LoggingSetup(NamedTuple):
    level: str
    frequency: int
    is_positivity: bool
    is_levelset_residuals: bool
    is_only_last_stage: bool

class OutputSetup(NamedTuple):
    is_active: bool
    is_domain: bool
    is_wall_clock_times: bool
    derivative_stencil: SpatialDerivative
    is_xdmf: bool
    is_parallel_filesystem: bool
    is_metadata: bool
    is_time: bool
    is_sync_hosts: bool
    logging: LoggingSetup

