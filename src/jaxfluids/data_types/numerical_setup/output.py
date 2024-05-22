from typing import NamedTuple, Tuple

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class LoggingSetup(NamedTuple):
    """LoggingSetup describes how often and what is being logged.
    This includes:
    - level: The logging level specifies whether output is being
        logged to the terminal and/or to the output file
    - frequency: Interval (in integration steps) at which logging
        is done
    - is_positivity: Whether positivity related information is logged
    - is_levelset_residuals: Whether level-set residuals are logged
    - is_only_last_stage: Whether only information of the last 
        Runge-Kutta sub-stage is logged. Otherwise information from 
        all sub-stages is logged.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    level: str
    frequency: int
    is_positivity: bool
    is_levelset_residuals: bool
    is_only_last_stage: bool

class OutputSetup(NamedTuple):
    """OutputSetup specifies what output is written.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    is_active: bool
    is_domain: bool
    is_wall_clock_times: bool
    derivative_stencil: SpatialDerivative
    is_xdmf: bool
    is_parallel_filesystem: bool
    is_metadata: bool
    is_time: bool
    logging: LoggingSetup
