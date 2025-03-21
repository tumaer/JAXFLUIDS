import jax

from jaxfluids.data_types.case_setup import CaseSetup
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.buffers import (
    MaterialFieldBuffers, LevelsetFieldBuffers, 
    SolidFieldBuffers, TimeControlVariables
)
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.time_integration.time_step_size import compute_time_step_size

class TimeControlInitializer:

    def __init__(
            self,
            case_setup: CaseSetup,
            numerical_setup: NumericalSetup,
            domain_information: DomainInformation,
            equation_information: EquationInformation,
            material_manager: MaterialManager,
            solid_properties_manager: SolidPropertiesManager,
        ) -> None:
        
        self.case_setup = case_setup
        self.numerical_setup = numerical_setup
        self.domain_information = domain_information
        self.equation_information = equation_information
        self.material_manager = material_manager
        self.solid_properties_manager = solid_properties_manager

    def initialize(
            self,
            physical_simulation_time: float,
            simulation_step: int,
            material_fields: MaterialFieldBuffers,
            levelset_fields: LevelsetFieldBuffers,
            solid_fields: SolidFieldBuffers
        ) -> TimeControlVariables:
        
        is_parallel = self.domain_information.is_parallel
        if is_parallel:
            compute_time_step_size_fn = jax.pmap(
                compute_time_step_size,
                in_axes=(0,0,0,0,0,None,None,None,None,None),
                out_axes=(None),
                static_broadcasted_argnums=(5,6,7,8,9),
                axis_name="i"
            )
        else:
            compute_time_step_size_fn = compute_time_step_size

        physical_timestep_size = compute_time_step_size_fn(
            material_fields.primitives,
            material_fields.temperature,
            levelset_fields.levelset,
            levelset_fields.volume_fraction,
            solid_fields.temperature,
            self.domain_information,
            self.equation_information,
            self.material_manager,
            self.solid_properties_manager,
            self.numerical_setup
        )

        fixed_time_step_size = self.numerical_setup.conservatives.time_integration.fixed_timestep
        end_time = self.case_setup.general_setup.end_time
        end_step = self.case_setup.general_setup.end_step

        time_control_variables = TimeControlVariables(
            physical_simulation_time,
            simulation_step,
            physical_timestep_size,
            fixed_time_step_size,
            end_time,
            end_step
        )
        
        return time_control_variables