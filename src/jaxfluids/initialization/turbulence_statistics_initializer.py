import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.data_types.statistics import TurbulenceStatisticsInformation
from jaxfluids.turbulence.statistics.online import DICT_TURBULENCE_STATISTICS_COMPUTER, TurbulenceStatisticsComputer
from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.data_types.buffers import MaterialFieldBuffers

Array = jax.Array


class TurbulenceStatisticsInitializer:

    def __init__(
        self,
        domain_information: DomainInformation,
        material_manager: MaterialManager,
        turbulence_statistics_setup: TurbulenceStatisticsSetup,
        ) -> None:
        
        self.domain_information = domain_information
        self.material_manager = material_manager
        self.turbulence_statistics_setup = turbulence_statistics_setup
        turbulence_case = turbulence_statistics_setup.case

        self.turbulence_online_statistics_computer: TurbulenceStatisticsComputer \
        = DICT_TURBULENCE_STATISTICS_COMPUTER[turbulence_case](
            turbulence_statistics_setup,
            domain_information,
            material_manager)

    def initialize(self, material_fields: MaterialFieldBuffers,
        ) -> TurbulenceStatisticsInformation:
    
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        primitives = material_fields.primitives[...,nhx,nhy,nhz]

        turbulent_statistics = self.turbulence_online_statistics_computer.initialize_statistics()
        turbulent_statistics = self.turbulence_online_statistics_computer.compute_turbulent_statistics(
            primitives, turbulent_statistics, False,
            self.turbulence_statistics_setup.is_logging)

        return turbulent_statistics
