from jax import Array

from jaxfluids.data_types.buffers import IntegrationBuffers, EulerIntegrationBuffers
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.levelset.helper_functions import transform_to_conserved

def get_integration_buffers(
        conservatives: Array,
        levelset: Array,
        volume_fraction: Array,
        solid_velocity: Array,
        solid_energy: Array,
        domain_information: DomainInformation,
        equation_information: EquationInformation
    ) -> IntegrationBuffers:
        
        levelset_model = equation_information.levelset_model
        solid_coupling = equation_information.solid_coupling

        if levelset_model:
            init_conservatives = transform_to_conserved(
                conservatives,
                volume_fraction,
                domain_information,
                levelset_model
            )
        else:
            init_conservatives = conservatives
        
        if equation_information.is_moving_levelset:
            init_levelset = levelset
        else:
            init_levelset = None
            
        if solid_coupling.dynamic == "TWO-WAY":
            init_solid_velocity = solid_velocity
        else:
            init_solid_velocity = None

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

        else:
            init_solid_energy = None

        euler_integration_buffers = EulerIntegrationBuffers(
            init_conservatives,
            init_levelset,
            init_solid_velocity,
            init_solid_energy
        )

        integration_buffers = IntegrationBuffers(
            euler_integration_buffers,
        )

        return integration_buffers
