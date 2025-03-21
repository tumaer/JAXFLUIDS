from jaxfluids.turbulence.statistics.online.turbulence_statistics_computer import TurbulenceStatisticsComputer
from jaxfluids.turbulence.statistics.online.hit_statistics_computer import HITStatisticsComputer
from jaxfluids.turbulence.statistics.online.channel_statistics_computer import ChannelStatisticsComputer
from jaxfluids.turbulence.statistics.online.boundary_layer_statistics_computer import BoundaryLayerStatisticsComputer

DICT_TURBULENCE_STATISTICS_COMPUTER = {
    "HIT": HITStatisticsComputer,
    "CHANNEL": ChannelStatisticsComputer,
    "BOUNDARY_LAYER": BoundaryLayerStatisticsComputer 
}