import types
from typing import Union, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.input.input_manager import InputManager
from jaxfluids.initialization.material_fields_initializer import MaterialFieldsInitializer
from jaxfluids.initialization.levelset_initializer import LevelsetInitializer
from jaxfluids.initialization.forcings_initializer import ForcingsInitializer
from jaxfluids.initialization.time_control_initializer import TimeControlInitializer
from jaxfluids.initialization.turbulence_statistics_initializer import TurbulenceStatisticsInitializer
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.solvers.positivity.positivity_handler import get_positvity_state_info
from jaxfluids.initialization.solid_initializer import SolidsInitializer
from jaxfluids.data_types.buffers import (SimulationBuffers, 
    MaterialFieldBuffers, LevelsetFieldBuffers, ForcingParameters, TimeControlVariables, 
    SolidFieldBuffers)
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.domain.helper_functions import reassemble_buffer, reassemble_buffer_np
from jaxfluids.data_types.information import (StepInformation, PositivityCounter, LevelsetResidualInformation,
                                              LevelsetProcedureInformation, PositivityStateInformation,
                                              DiscretizationCounter, LevelsetPositivityInformation, FlowStatistics)
from jaxfluids.data_types.statistics import TurbulenceStatisticsInformation
from jaxfluids.data_types.ml_buffers import (
    MachineLearningSetup, combine_callables_and_params,
    ParametersSetup, CallablesSetup)
from jaxfluids.data_types.ml_buffers import (MachineLearningSetup, combine_callables_and_params)
from jaxfluids.levelset.extension.material_fields.extension_handler import ghost_cell_extension_material_fields
from jaxfluids.levelset.fluid_fluid.interface_quantities import compute_interface_quantities
from jaxfluids.levelset.fluid_solid.interface_quantities import compute_thermal_interface_state

Array = jax.Array

class InitializationManager:
    """The InitializationManager class implements functionality to create a dictionary of initial buffers that is 
    passed to the simulate() method of the SimulationManager class. The initialization() method returns this
    dictionary. The initial buffers are created in one of the following ways: 
    1) From a restart file that is specified in the case setup.
    2) From turbulent initial condition parameters that are specified in the case setup
    3) From the initial primitive buffer that is passed to the initialization() method
    4) From the initial conditions for primitive variables specified in case setup
    Note that if multiple of the above are provided, the priority is 1) - 4).
    """

    def __init__(
            self,
            input_manager: InputManager
            ) -> None:
        
        assert_str = "InitializationManager requires an InputManager object as input."
        assert isinstance(input_manager, InputManager), assert_str

        self.numerical_setup = input_manager.numerical_setup
        self.case_setup = input_manager.case_setup

        unit_handler = input_manager.unit_handler
        halo_manager = input_manager.halo_manager
        self.material_manager = input_manager.material_manager
        self.equation_manager = input_manager.equation_manager
        self.domain_information = input_manager.domain_information
        self.equation_information = input_manager.equation_information
        self.solid_properties_manager = input_manager.solid_properties_manager

        self.material_fields_initializer = MaterialFieldsInitializer(
            case_setup=self.case_setup,
            numerical_setup=self.numerical_setup,
            domain_information=self.domain_information,
            unit_handler=unit_handler,
            equation_manager=self.equation_manager,
            material_manager=self.material_manager,
            halo_manager=halo_manager
        )

        levelset_model = self.equation_information.levelset_model

        if levelset_model:
            self.levelset_initializer = LevelsetInitializer(
                numerical_setup = self.numerical_setup,
                domain_information = self.domain_information,
                unit_handler = unit_handler,
                equation_manager = self.equation_manager,
                material_manager = self.material_manager,
                halo_manager = halo_manager,
                initial_condition_levelset = input_manager.case_setup.initial_condition_setup.levelset,
                restart_setup = input_manager.case_setup.restart_setup,
                solid_properties_manager = input_manager.solid_properties_manager
                )
        
        solid_coupling = self.numerical_setup.levelset.solid_coupling
        if any((solid_coupling.thermal == "TWO-WAY", solid_coupling.dynamic == "TWO-WAY")):

            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError
            else:
                extender_solids = None

            self.solids_initializer = SolidsInitializer(
                numerical_setup=self.numerical_setup,
                domain_information=self.domain_information,
                unit_handler=unit_handler,
                equation_information=input_manager.equation_information,
                halo_manager=halo_manager,
                extender=extender_solids,
                initial_condition_solids=input_manager.case_setup.initial_condition_setup.solids,
                restart_setup=input_manager.case_setup.restart_setup,
                solid_properties_setup=input_manager.case_setup.solid_properties_setup)
        else:
            extender_solids = None

        if input_manager.numerical_setup.active_forcings:
            self.forcings_initializer = ForcingsInitializer(
                numerical_setup = self.numerical_setup,
                domain_information = self.domain_information,
                material_manager = self.material_manager,
                restart_setup = input_manager.case_setup.restart_setup)

        self.time_control_initializer = TimeControlInitializer(
            case_setup=self.case_setup,
            numerical_setup=self.numerical_setup,
            domain_information=self.domain_information,
            equation_information=self.equation_information,
            material_manager=self.material_manager,
            solid_properties_manager=self.solid_properties_manager
        )       


    def initialization(
            self,
            user_prime_init: Union[np.ndarray, Array] = None,
            user_time_init: float = None,
            user_levelset_init: Union[np.ndarray, Array] = None,
            user_solid_interface_velocity_init: Union[np.ndarray, Array] = None,
            user_restart_file_path: str = None,
            ml_parameters: ParametersSetup = ParametersSetup(),
            ml_callables: CallablesSetup = CallablesSetup()
        ) -> JaxFluidsBuffers:
        """Creates a buffer dictionary containing the initial buffers
        for the material fields, time control variables,
        levelset related fields, and forcings.

        :param user_prime_init: _description_, defaults to None
        :type user_prime_init: Union[np.ndarray, Array], optional
        :param user_levelset_init: _description_, defaults to None
        :type user_levelset_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: JaxFluidsBuffers
        """

        is_parallel = self.domain_information.is_parallel

        ml_setup = combine_callables_and_params(ml_callables, ml_parameters)

        # MATERIAL FIELDS
        (
            material_fields,
            time_control_variables
        ) = self.material_fields_initializer.initialize(
            user_prime_init, user_time_init, user_restart_file_path, ml_setup
        )

        # LEVELSET
        if self.equation_information.levelset_model:
            levelset_fields = self.levelset_initializer.initialize(
                user_levelset_init, user_solid_interface_velocity_init,
                user_restart_file_path
            )
        else:
            levelset_fields = LevelsetFieldBuffers()

        # SOLIDS
        solid_coupling = self.equation_information.solid_coupling
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        if any((solid_coupling.dynamic == "TWO-WAY",)):
            solid_fields = self.solids_initializer.initialize(
                levelset_fields,
                user_restart_file_path=user_restart_file_path
            )
        else:
            solid_fields = SolidFieldBuffers()

        if self.equation_information.levelset_model:
            if is_parallel:
                extension_and_interface_quantities_fn = jax.pmap(
                    self.extension_and_interface_quantities,
                    axis_name="i",
                    in_axes=(0,0,0,None,None),
                    out_axes=(0,0,0,None,None,None)
                )
            else:
                extension_and_interface_quantities_fn = self.extension_and_interface_quantities
                
            (
                material_fields,
                levelset_fields,
                solid_fields,
                levelset_residual_info,
                levelset_positivity_fluid_info,
                levelset_positivity_solids_info, 
            ) = extension_and_interface_quantities_fn(
                material_fields,
                levelset_fields,
                solid_fields,
                time_control_variables,
                ml_setup
            )

        else:
            levelset_residual_info = None
            levelset_positivity_fluid_info = None
            levelset_positivity_solids_info = None

        # FORCINGS
        active_forcings = self.numerical_setup.active_forcings
        if any(active_forcings._asdict().values()):
            forcing_parameters = self.forcings_initializer.initialize(
                material_fields,
                user_restart_file_path=user_restart_file_path
            )
        else:
            forcing_parameters = ForcingParameters()

        # TIME CONTROL VARIABLES
        time_control_variables = self.time_control_initializer.initialize(
            time_control_variables.physical_simulation_time,
            time_control_variables.simulation_step,
            material_fields,
            levelset_fields,
            solid_fields
        )

        # FLOW STATISTICS
        # TURBULENCE STATISTICS
        turbulence_statistics_setup = self.case_setup.statistics_setup.turbulence
        is_turbulence_statistics = any((
            turbulence_statistics_setup.is_logging,
            turbulence_statistics_setup.is_cumulative))
        if is_turbulence_statistics:
            turbulence_statistics_initializer = TurbulenceStatisticsInitializer(
                self.domain_information, self.material_manager, 
                turbulence_statistics_setup)
            if is_parallel:
                turbulence_statistics = jax.pmap(
                    turbulence_statistics_initializer.initialize, axis_name="i",
                    in_axes=0, out_axes=None)(
                    material_fields)
            else:
                turbulence_statistics = turbulence_statistics_initializer.initialize(
                    material_fields)
        else:
            turbulence_statistics = TurbulenceStatisticsInformation()

        flow_statistics = FlowStatistics(turbulence_statistics)


        def wrapper_positivity_info(
                primitives: Array,
                temperature: Array,
                volume_fraction: Array
                ) -> PositivityStateInformation:
            """Pmap wrapper for positivity state info.

            :param material_fields: _description_
            :type material_fields: MaterialFieldBuffers
            :param levelset_fields: _description_
            :type levelset_fields: LevelsetFieldBuffers
            :return: _description_
            :rtype: PositivityStateInformation
            """
            positivity_state_info = get_positvity_state_info(
                primitives, temperature,
                PositivityCounter(0,0,0,0,0),
                DiscretizationCounter(0,0),
                volume_fraction,
                levelset_positivity_fluid_info,
                levelset_positivity_solids_info,
                self.material_manager, self.equation_information,
                self.domain_information)
            return positivity_state_info

        if self.domain_information.is_parallel:
            positivity_state_info = jax.pmap(wrapper_positivity_info, in_axes=(0,0,0), out_axes=None, axis_name="i")(
                material_fields.primitives, material_fields.temperature, levelset_fields.volume_fraction)
        else:
            positivity_state_info = wrapper_positivity_info(
                material_fields.primitives, material_fields.temperature, levelset_fields.volume_fraction)
        
        step_information = StepInformation(
            tuple([positivity_state_info]),
            tuple([levelset_residual_info]),
            statistics=flow_statistics)

        simulation_buffers = SimulationBuffers(
            material_fields, levelset_fields,
            solid_fields)
    
        jxf_buffers = JaxFluidsBuffers(
            simulation_buffers,
            time_control_variables,
            forcing_parameters,
            step_information
        )

        return jxf_buffers
        

    def extension_and_interface_quantities(
            self,
            material_fields: MaterialFieldBuffers,
            levelset_fields: LevelsetFieldBuffers,
            solid_fields: SolidFieldBuffers,
            time_control_variables: TimeControlVariables,
            ml_setup: MachineLearningSetup
        ) -> Tuple[MaterialFieldBuffers, LevelsetFieldBuffers,
                   SolidFieldBuffers, LevelsetResidualInformation,
                   LevelsetPositivityInformation,
                   LevelsetPositivityInformation]:
        """Wrapper to perform ghost cell extension
        and interface quantity computation.

        :param material_fields: _description_
        :type material_fields: MaterialFieldBuffers
        :param levelset_fields: _description_
        :type levelset_fields: LevelsetFieldBuffers
        :return: _description_
        :rtype: Tuple[MaterialFieldBuffers, LevelsetFieldBuffers]
        """

        primitives = material_fields.primitives
        conservatives = material_fields.conservatives
        temperature = material_fields.temperature

        levelset = levelset_fields.levelset
        volume_fraction = levelset_fields.volume_fraction
        apertures = levelset_fields.apertures
        
        extension_cells_indices_fluid = levelset_fields.solid_cell_indices.extension_fluid
        extension_cells_indices_solid = levelset_fields.solid_cell_indices.extension_solid

        solid_temperature = solid_fields.temperature
        solid_velocity = solid_fields.velocity

        physical_simulation_time = time_control_variables.physical_simulation_time

        levelset_model = self.equation_information.levelset_model

        levelset_setup = self.numerical_setup.levelset
        extension_setup = levelset_setup.extension
        narrowband_setup = levelset_setup.narrowband
        method_extension_primitives = extension_setup.primitives.method
        method_extension_solids = extension_setup.solids.method

        extender_primes = self.levelset_initializer.extender_primes
        extender_interface = self.levelset_initializer.extender_interface
        geometry_calculator = self.levelset_initializer.geometry_calculator
        halo_manager = self.levelset_initializer.halo_manager

        solid_coupling = self.equation_information.solid_coupling

        fill_edge_halos_material = halo_manager.fill_edge_halos_material
        fill_vertex_halos_material = halo_manager.fill_vertex_halos_material

        normal = geometry_calculator.compute_normal(levelset)

        # interpolation based extension requires all halo regions
        if method_extension_primitives == "INTERPOLATION" and \
        not all((fill_vertex_halos_material, fill_edge_halos_material)):
            primitives = halo_manager.perform_halo_update_material(
                primitives, physical_simulation_time, True, True, None, False,
                ml_setup=ml_setup)
            
        # interface heat flux/temperature required for extension for conjugate heat
        # do we want to introduce new buffer that carries the interpolated interface states
        # because currently perform the interpolation twice
        if solid_coupling.thermal == "TWO-WAY" and any((method_extension_primitives == "INTERPOLATION",
                                                        method_extension_solids == "INTERPOLATION")):
            raise NotImplementedError
        else:
            interface_heat_flux = None
            interface_temperature = None

        (
            conservatives,
            primitives,
            invalid_cell_count_extension,
            residual_info_prime 
        ) = ghost_cell_extension_material_fields(
            conservatives, primitives, levelset, volume_fraction,
            normal, solid_temperature, solid_velocity,
            interface_heat_flux, interface_temperature,
            None, physical_simulation_time, extension_setup.primitives,
            narrowband_setup, extension_cells_indices_fluid,
            extender_primes, self.equation_manager,
            self.solid_properties_manager,
            is_initialization=True,
            ml_setup=ml_setup
            )
        
        primitives, conservatives = halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, fill_edge_halos_material,
            fill_vertex_halos_material, conservatives, ml_setup=ml_setup)
        if self.equation_information.is_compute_temperature:
            temperature = self.material_manager.get_temperature(primitives)
            temperature = halo_manager.perform_outer_halo_update_temperature(
                temperature, physical_simulation_time)
        else:
            temperature = None

        material_fields = MaterialFieldBuffers(
            conservatives, primitives, temperature)
        
        if levelset_model == "FLUID-FLUID":
            curvature = geometry_calculator.compute_curvature(levelset)
            interface_velocity, interface_pressure, residual_info_interface = \
            compute_interface_quantities(
                primitives, levelset, volume_fraction, normal, curvature, 
                self.material_manager, extender_interface,
                extension_setup.interface.iterative, narrowband_setup,
                is_initialization=True, ml_setup=ml_setup)
            levelset_fields = LevelsetFieldBuffers(
                levelset, volume_fraction, apertures, interface_velocity,
                interface_pressure)
        else:
            residual_info_interface = None

        levelset_positivity_fluid_info = LevelsetPositivityInformation(
            None, invalid_cell_count_extension)

        max_residual = self.levelset_initializer.reinitializer.compute_residual(
            levelset_fields.levelset)
    
        if solid_coupling.thermal == "TWO-WAY":

            raise NotImplementedError

        else:

            levelset_positivity_solids_info = None
            residual_info_solid = None

        levelset_residual_info = LevelsetResidualInformation(
            LevelsetProcedureInformation(0, max_residual, None),
            residual_info_prime,
            residual_info_interface,
            residual_info_solid,
            )

        return material_fields, levelset_fields, solid_fields, levelset_residual_info, \
            levelset_positivity_fluid_info, levelset_positivity_solids_info

