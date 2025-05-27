from functools import partial
import time
from typing import List, Tuple, Union, Dict

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.callbacks.base_callback import Callback
from jaxfluids.data_types.buffers import (
    ForcingBuffers, IntegrationBuffers, IntegrationBuffers,
    SimulationBuffers, MaterialFieldBuffers, LevelsetFieldBuffers, ForcingParameters,
    TimeControlVariables, SolidFieldBuffers, LevelsetSolidCellIndices,
    ControlFlowParameters
)
from jaxfluids.data_types.ml_buffers import MachineLearningSetup, combine_callables_and_params, CallablesSetup, ParametersSetup
from jaxfluids.data_types.information import (
    LevelsetResidualInformation, StepInformation,
    WallClockTimes
)
from jaxfluids.data_types.statistics import FlowStatistics
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup import CaseSetup
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.feed_forward.feed_forward import configure_multistep, initialize_fields_feed_forward
from jaxfluids.feed_forward.data_types import FeedForwardSetup
from jaxfluids.forcing.forcing import Forcing
from jaxfluids.input.input_manager import InputManager
from jaxfluids.io_utils.output_writer import OutputWriter
from jaxfluids.io_utils.logger import Logger
from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks, compute_cut_cell_mask_sign_change_based
from jaxfluids.levelset.helper_functions import transform_to_conserved
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.solvers.space_solver import SpaceSolver
from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler, get_positvity_state_info
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.turbulence.statistics.online import DICT_TURBULENCE_STATISTICS_COMPUTER, TurbulenceStatisticsComputer
from jax.experimental.multihost_utils import sync_global_devices
from jaxfluids.config import precision
from jaxfluids.materials import DICT_MATERIAL
from jaxfluids.time_integration.time_step_size import compute_time_step_size
from jaxfluids.time_integration.helper_functions import get_integration_buffers

Array = jax.Array

class SimulationManager:
    """ The SimulationManager is the top-level class in JAX-FLUIDS. It
    provides functionality to perform conventional CFD simulations
    as well as end-to-end optimization of ML models.

    The most important methods of the SimulationManager are:

    1) simulate()               -   Performs conventional CFD simulation.
                                    Output
    2) feedforward()            -   Feedforward of a batch of data, i.e.,
        advances a batch of initial conditions in time for a fixed amount of steps
    3) do_integration_step()    -   Performs a single integration step
    """

    def __init__(
            self,
            input_manager: InputManager,
            callbacks: Union[Callback, List[Callback]] = None
            ) -> None:

        assert_str = "SimulationManager requires an InputManager object as input."
        assert isinstance(input_manager, InputManager), assert_str

        self.eps = precision.get_eps()

        self.input_manager = input_manager
        self.case_setup: CaseSetup = input_manager.case_setup
        self.numerical_setup: NumericalSetup = input_manager.numerical_setup

        self.unit_handler = input_manager.unit_handler # ONLY OUTPUT WRITER SHOULD NEED THIS NOW
        self.material_manager = input_manager.material_manager
        self.equation_manager = input_manager.equation_manager
        self.domain_information = input_manager.domain_information
        self.halo_manager = input_manager.halo_manager
        self.equation_information = input_manager.equation_information
        self.solid_properties_manager = input_manager.solid_properties_manager

        # TIME INTEGRATION
        time_integrator = self.numerical_setup.conservatives.time_integration.integrator
        self.time_integrator: TimeIntegrator = time_integrator(
            nh=self.domain_information.nh_conservatives,
            inactive_axes=self.domain_information.inactive_axes)
        
        # LEVELSET HANDLER
        if self.equation_information.levelset_model:
            self.levelset_handler = LevelsetHandler(
                domain_information=self.domain_information,
                numerical_setup=self.numerical_setup,
                material_manager=self.material_manager,
                equation_manager=self.equation_manager,
                halo_manager=self.halo_manager,
                solid_properties_manager=self.solid_properties_manager)
        else:
            self.levelset_handler = None

        # DIFFUSE INTERFACE HANDLER
        if self.equation_information.diffuse_interface_model:
            self.diffuse_interface_handler = DiffuseInterfaceHandler(
                domain_information=self.domain_information,
                numerical_setup=self.numerical_setup,
                material_manager=self.material_manager,
                unit_handler=self.unit_handler,
                equation_manager=self.equation_manager,
                halo_manager=self.halo_manager)
        else:
            self.diffuse_interface_handler = None

        # POSITIVITY HANDLER
        self.positivity_handler = PositivityHandler(
            domain_information=self.domain_information,
            material_manager=self.material_manager,
            equation_manager=self.equation_manager,
            halo_manager=self.halo_manager,
            numerical_setup=self.numerical_setup,
            levelset_handler=self.levelset_handler,
            diffuse_interface_handler=self.diffuse_interface_handler,
        )

        # SPACE SOLVER
        self.space_solver = SpaceSolver(
            domain_information=self.domain_information,
            material_manager=self.material_manager,
            equation_manager=self.equation_manager,
            halo_manager=self.halo_manager,
            numerical_setup=self.numerical_setup,
            gravity=self.case_setup.forcing_setup.gravity,
            geometric_source=self.case_setup.forcing_setup.geometric_source,
            levelset_handler=self.levelset_handler,
            diffuse_interface_handler=self.diffuse_interface_handler,
            positivity_handler=self.positivity_handler)

        # TURBULENCE STATISTICS
        turbulence_statistics_setup = self.case_setup.statistics_setup.turbulence
        self.is_turbulence_statistics = any((
            turbulence_statistics_setup.is_cumulative, 
            turbulence_statistics_setup.is_logging))
        if self.is_turbulence_statistics:
            self.turbulence_online_statistics_computer: TurbulenceStatisticsComputer \
            = DICT_TURBULENCE_STATISTICS_COMPUTER[turbulence_statistics_setup.case](
                turbulence_statistics_setup=turbulence_statistics_setup,
                domain_information=self.domain_information,
                material_manager=self.material_manager)

        # FORCINGS
        if self.numerical_setup.active_forcings:
            self.forcings_computer = Forcing(
                domain_information=self.domain_information,
                equation_manager=self.equation_manager,
                material_manager=self.material_manager,
                solid_properties_manager=self.solid_properties_manager,
                unit_handler=self.unit_handler,
                forcing_setup=self.case_setup.forcing_setup)
            
        # OUTPUT WRITER
        self.output_writer = OutputWriter(
            input_manager=input_manager,
            unit_handler=self.unit_handler,
            domain_information=self.domain_information,
            equation_information=self.equation_information,
            material_manager=self.material_manager,
            levelset_handler=self.levelset_handler)

        # LOGGER
        self.logger = Logger(
            case_setup=self.case_setup,
            numerical_setup=self.numerical_setup,
            jax_backend=jax.default_backend(),
            is_multihost=self.domain_information.is_multihost)

        # CALLBACKS INIT
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callbacks = callbacks or []
        for cb in self.callbacks:
            assert isinstance(cb, Callback)
            cb.init_callback(sim_manager=self)

    def simulate(
            self,
            jxf_buffers: JaxFluidsBuffers,
            ml_parameters: ParametersSetup = ParametersSetup(),
            ml_callables: CallablesSetup = CallablesSetup(),
        ) -> int:
        """Performs a conventional CFD simulation.

        :param jxf_buffers: _description_
        :type jxf_buffers: JaxFluidsBuffers
        :param ml_parameters: _description_, defaults to ParametersSetup()
        :type ml_parameters: ParametersSetup, optional
        :param ml_callables: _description_, defaults to CallablesSetup()
        :type ml_callables: CallablesSetup, optional
        :return: _description_
        :rtype: int
        """

        self.initialize(jxf_buffers)
        return_value = self.advance(
            jxf_buffers,
            ml_parameters,
            ml_callables
        )

        return return_value

    def initialize(self, jxf_buffers: JaxFluidsBuffers) -> None:
        """Initializes the simulation, i.e., creates the
        output directory, logs the numerical and case setup,
        and writes the initial output.

        :param jxf_buffers: _description_
        :type jxf_buffers: JaxFluidsBuffers
        :return: _description_
        :rtype: JaxFluidsBuffers
        """

        
        simulation_buffers = jxf_buffers.simulation_buffers
        time_control_variables = jxf_buffers.time_control_variables
        forcing_parameters = jxf_buffers.forcing_parameters
        step_information = jxf_buffers.step_information

        self.sanity_check(simulation_buffers, time_control_variables, forcing_parameters)

        # CREATE OUTPUT FOLDER, CASE SETUP AND NUMERICAL SETUP
        # TODO has to be done before call to simulate, otherwise race condition
        # with other jobs.
        # self.output_writer.create_folder()
        save_path_case, save_path_domain, save_path_statistics = self.output_writer.configure_output_writer()

        # CONFIGURE LOGGER AND LOG NUMERICAL SETUP AND CASE SETUP
        self.logger.configure_logger(save_path_case)
        self.logger.log_initialization()
        self.logger.log_numerical_setup_and_case_setup(
            *self.input_manager.info(),
            self.domain_information.info()
        )

        # WRITE INITIAL OUTPUT
        self.output_writer.set_simulation_start_time(time_control_variables.physical_simulation_time)

        # LOG SIMULATION START
        self.logger.log_sim_start()

        # LOG T0
        self.logger.log_initial_time_step(
            time_control_variables, step_information,
            self.unit_handler.time_reference)

        self.output_writer.write_output(
            simulation_buffers, time_control_variables,
            WallClockTimes(), forcing_parameters, 
            force_output=True)

    def sanity_check(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters = None
        ) -> None:
        """Very light weight initial sanity check of inputs to simulate.
        #TODO should we expand this?

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_, defaults to None
        :type forcing_parameters: ForcingParameters, optional
        """

        assert_string = "No simulation buffer provided to simulate()."
        assert simulation_buffers is not None, assert_string

        assert_string = "No time control variables provided to simulate()."
        assert time_control_variables is not None, assert_string

        if any((self.numerical_setup.active_forcings.is_mass_flow_forcing,
                self.numerical_setup.active_forcings.is_turb_hit_forcing)):
            assert_string = ("Mass flow forcing or turbulent HIT forcing is active "
                             "but no forcing parameters were provided to simulate().")
            is_all_forcing_parameters_none = all(map(lambda x: x is None, forcing_parameters))
            assert forcing_parameters is not None and not is_all_forcing_parameters_none, \
                assert_string

    def advance(
            self,
            jxf_buffers: JaxFluidsBuffers,
            ml_parameters: ParametersSetup,
            ml_callables: CallablesSetup,
        ) -> bool:
        """Advances the initial buffers in time.
        """

        # START LOOP
        start_loop = self.synchronize_and_clock(
            jxf_buffers.simulation_buffers.material_fields.primitives)

        # CALLBACK on_simulation_start
        callback_dict = {}
        jxf_buffers, callback_dict = self._callback(
            "on_simulation_start",
            jxf_buffers=jxf_buffers,
            callback_dict=callback_dict
        )

        time_control_variables = jxf_buffers.time_control_variables
        physical_simulation_time = time_control_variables.physical_simulation_time
        simulation_step = time_control_variables.simulation_step

        wall_clock_times = WallClockTimes()

        while (
            physical_simulation_time < time_control_variables.end_time 
            and simulation_step < time_control_variables.end_step
        ):

            start_step = self.synchronize_and_clock(
                jxf_buffers.simulation_buffers.material_fields.primitives)

            control_flow_params = self.compute_control_flow_params(
                time_control_variables, jxf_buffers.step_information)

            # NOTE CALLBACK
            jxf_buffers, callback_dict = self._callback(
                "before_step_start",
                jxf_buffers=jxf_buffers,
                callback_dict=callback_dict
            )

            # PERFORM INTEGRATION STEP
            jxf_buffers, callback_dict_step = self.do_integration_step(
                jxf_buffers,
                control_flow_params,
                ml_parameters,
                ml_callables
            )

            # NOTE CALLBACK - AFTER_STEP_END
            # This callback receives the callback_dict_step
            jxf_buffers, callback_dict = self._callback(
                "after_step_end",
                jxf_buffers=jxf_buffers,
                callback_dict=callback_dict,
                callback_dict_step=callback_dict_step
            )
            
            # NOTE UNPACK JAX FLUIDS BUFFERS
            simulation_buffers = jxf_buffers.simulation_buffers
            time_control_variables = jxf_buffers.time_control_variables
            forcing_parameters = jxf_buffers.forcing_parameters
            step_information = jxf_buffers.step_information

            # CLOCK INTEGRATION STEP
            end_step = self.synchronize_and_clock(
                simulation_buffers.material_fields.primitives)
            wall_clock_step = end_step - start_step

            # COMPUTE WALL CLOCK TIMES FOR TIME STEP
            wall_clock_times = self.compute_wall_clock_time(
                wall_clock_step,
                wall_clock_times,
                time_control_variables.simulation_step
            )

            # LOG TERMINAL END TIME STEP
            self.logger.log_end_time_step(
                time_control_variables,
                step_information,
                wall_clock_times,
                self.unit_handler.time_reference
            )

            # WRITE H5 OUTPUT
            self.output_writer.write_output(
                simulation_buffers,
                time_control_variables,
                wall_clock_times,
                forcing_parameters,
                flow_statistics=step_information.statistics
            )

            # UNPACK FOR WHILE LOOP
            physical_simulation_time = time_control_variables.physical_simulation_time
            simulation_step = time_control_variables.simulation_step

        # CALLBACK on_simulation_end
        jxf_buffers, callback_dict = self._callback(
            "on_simulation_end",
            jxf_buffers=jxf_buffers,
            callback_dict=callback_dict
        )

        # UNPACK JAX FLUIDS BUFFERS
        simulation_buffers = jxf_buffers.simulation_buffers
        time_control_variables = jxf_buffers.time_control_variables
        forcing_parameters = jxf_buffers.forcing_parameters
        step_information = jxf_buffers.step_information

        # FINAL OUTPUT
        self.output_writer.write_output(
            simulation_buffers,
            time_control_variables,
            wall_clock_times,
            forcing_parameters,
            force_output=True,
            simulation_finish=True,
            flow_statistics=step_information.statistics
        )

        # LOG SIMULATION FINISH
        end_loop = self.synchronize_and_clock(
            simulation_buffers.material_fields.primitives)
        self.logger.log_sim_finish(end_loop - start_loop)

        return bool(physical_simulation_time >= time_control_variables.end_time)

    def compute_wall_clock_time(
            self,
            wall_clock_step: float,
            wall_clock_times: WallClockTimes,
            simulation_step: int
        ) -> WallClockTimes:
        """Computes the instantaneous 
        and mean wall clock time for the
        a single simulation steps.

        :param wall_clock_step: _description_
        :type wall_clock_step: float
        :param simulation_step: _description_
        :type simulation_step: int
        :param wall_clock_times: _description_
        :type wall_clock_times: WallClockTimes
        :return: _description_
        :rtype: WallClockTimes
        """
        offset = 10
        cells_per_device = self.domain_information.cells_per_device
        if simulation_step >= offset:
            mean_wall_clock_step = wall_clock_times.mean_step
            mean_wall_clock_step_cell = wall_clock_times.mean_step_per_cell
            wall_clock_step_cell = wall_clock_step / cells_per_device
            mean_wall_clock_step = (wall_clock_step + mean_wall_clock_step * (simulation_step - offset)) / (simulation_step - offset + 1)
            mean_wall_clock_step_cell = (wall_clock_step_cell + mean_wall_clock_step_cell * (simulation_step - offset)) / (simulation_step - offset + 1)
        else:
            wall_clock_step_cell = wall_clock_step / cells_per_device
            mean_wall_clock_step = wall_clock_step
            mean_wall_clock_step_cell = wall_clock_step_cell

        wall_clock_times = WallClockTimes(
            wall_clock_step, wall_clock_step_cell,
            mean_wall_clock_step, mean_wall_clock_step_cell
            )
            
        return wall_clock_times

    def synchronize_and_clock(
            self,
            buffer: Array
        ) -> float:
        """Synchronizes jax and python by blocking
        python until the input buffer is ready.
        For multi-host simulations, subsequently
        performs a all-reduce operation to
        synchronize all hosts. 

        :param buffer: _description_, defaults to True
        :type buffer: _type_, optional
        :return: _description_
        :rtype: float
        """
        buffer.block_until_ready()
        if self.numerical_setup.output.is_sync_hosts:
            sync_global_devices("complete")
        return time.time()

    def compute_control_flow_params(
            self,
            time_control_variables: TimeControlVariables,
            step_information: StepInformation
        ) -> ControlFlowParameters:

        # COMPUTE REINITIALIZATION FLAG
        if self.equation_information.levelset_model:
            perform_reinitialization = self.levelset_handler.get_reinitialization_flag(
                time_control_variables.simulation_step)
        else:
            perform_reinitialization = None

        # COMPUTE INTERFACE COMPRESSION FLAG
        if self.equation_information.diffuse_interface_model:
            perform_compression = self.diffuse_interface_handler.get_compression_flag(
                time_control_variables.simulation_step)
        else:
            perform_compression = None

        # COMPUTE STATISTICS FLAG
        if self.is_turbulence_statistics:
            turbulence_statistics = step_information.statistics.turbulence
            logging_frequency = self.logger.logging_frequency

            (
                is_cumulative_statistics,
                is_logging_statistics,
            ) = self.turbulence_online_statistics_computer.get_statistics_flags(
                turbulence_statistics, 
                logging_frequency, 
                time_control_variables
            )
        else:
            is_cumulative_statistics = is_logging_statistics = None

        is_feed_forward = False

        control_flow_params = ControlFlowParameters(
            perform_reinitialization,
            perform_compression,
            is_cumulative_statistics,
            is_logging_statistics,
            is_feed_forward
        )

        return control_flow_params

    def _do_integration_step(
            self,
            jxf_buffers: JaxFluidsBuffers,
            control_flow_params: ControlFlowParameters,
            ml_parameters: ParametersSetup,
            ml_callables: CallablesSetup,
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Performs an integration step.
        1) Compute timestep size
        2) Compute forcings
        3) Do Runge Kutta stages
        4) Compute simulation information, i.e.,
            positivity state, turbulence statistics
            etc.


        :param jxf_buffers: _description_
        :type jxf_buffers: JaxFluidsBuffers
        :param control_flow_params: _description_
        :type control_flow_params: ControlFlowParameters
        :param ml_parameters: _description_
        :type ml_parameters: ParametersSetup
        :param ml_callables: _description_
        :type ml_callables: CallablesSetup
        :return: _description_
        :rtype: Tuple[JaxFluidsBuffers, Dict]
        """

        # CALLBACK on_step_start
        callback_dict = {}
        jxf_buffers, callback_dict = self._callback(
            "on_step_start",
            jxf_buffers=jxf_buffers,
            callback_dict=callback_dict
        )

        # UNPACK JAXFLUIDS BUFFERS
        simulation_buffers = jxf_buffers.simulation_buffers
        time_control_variables = jxf_buffers.time_control_variables
        forcing_parameters = jxf_buffers.forcing_parameters
        step_information = jxf_buffers.step_information

        material_fields = simulation_buffers.material_fields
        levelset_fields = simulation_buffers.levelset_fields
        solid_fields = simulation_buffers.solid_fields
        statistics = step_information.statistics
        ml_setup = combine_callables_and_params(ml_callables, ml_parameters)

        # COMPUTE FORCINGS
        active_forcings = self.numerical_setup.active_forcings
        if any(active_forcings._asdict().values()):
            forcing_buffers, forcing_parameters, forcing_infos = self.forcings_computer.compute_forcings(
                simulation_buffers,
                time_control_variables,
                forcing_parameters,
                self.do_runge_kutta_stages,
                ml_setup=ml_setup
            )
        else:
            forcing_buffers, forcing_infos = None, None

        # PERFORM INTEGRATION STEP CFD
        (
            material_fields,
            time_control_variables,
            levelset_fields,
            solid_fields,
            step_information
        ) = self.do_runge_kutta_stages(
            material_fields, time_control_variables,
            levelset_fields, solid_fields,
            forcing_buffers, control_flow_params,
            ml_setup
        )

        # COMPUTE TIMESTEP 
        if not control_flow_params.is_feed_foward:
            physical_timestep_size = compute_time_step_size(
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
            
            time_control_variables = time_control_variables._replace(
                physical_timestep_size=physical_timestep_size
            )

        # COMPUTE TURBULENCE STATISTICS
        if self.is_turbulence_statistics:
            nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
            turbulent_statistics = self.turbulence_online_statistics_computer.compute_turbulent_statistics(
                simulation_buffers.material_fields.primitives[...,nhx,nhy,nhz],
                statistics.turbulence,
                control_flow_params.is_cumulative_statistics,
                control_flow_params.is_logging_statistics)
            statistics = FlowStatistics(turbulent_statistics)
        else:
            statistics = FlowStatistics()

        # CREATE CONTAINERS
        simulation_buffers = SimulationBuffers(
            material_fields, levelset_fields,
            solid_fields)

        step_information = StepInformation(
            step_information.positivity,
            step_information.levelset,
            forcing_info=forcing_infos,
            statistics=statistics
        )

        jxf_buffers = JaxFluidsBuffers(
            simulation_buffers,
            time_control_variables,
            forcing_parameters,
            step_information
        )
        
        # CALLBACK on_step_end
        jxf_buffers, callback_dict = self._callback(
            "on_step_end",
            jxf_buffers=jxf_buffers,
            callback_dict=callback_dict
        )

        return jxf_buffers, callback_dict

    def do_runge_kutta_stages(
            self,
            material_fields: MaterialFieldBuffers,
            time_control_variables: TimeControlVariables,
            levelset_fields: LevelsetFieldBuffers = None,
            solid_fields: SolidFieldBuffers = None,
            forcing_buffers: ForcingBuffers = None, 
            control_flow_params: ControlFlowParameters = None,
            ml_setup: MachineLearningSetup = None,
        ) -> Tuple[MaterialFieldBuffers, TimeControlVariables,
            LevelsetFieldBuffers, SolidFieldBuffers,
            StepInformation]:
        """Performs the Runge Kutta stages. For twophase
        levelset simulations a single RK stage consists
        of the following:
        1) Compute right-hand-side buffers
        2) Integrate buffers
        3) Treat integrated levelset
            - Perform reinitialization (only last RK stage)
            - Perform halo update
            - Perform interface reconstruction
        4) Treat integrated material fields
            - Perform conservative mixing
            - Perform extension procedure on primitives
        5) Halo update

        :param material_fields: _description_
        :type material_fields: MaterialFieldBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param levelset_fields: _description_, defaults to None
        :type levelset_fields: LevelsetFieldBuffers, optional
        :param solid_fields: _description_, defaults to None
        :type solid_fields: SolidFieldBuffers, optional
        :param forcing_buffers: _description_, defaults to None
        :type forcing_buffers: ForcingBuffers, optional
        :param control_flow_params: _description_, defaults to None
        :type control_flow_params: ControlFlowParameters, optional
        :return: _description_
        :rtype: Tuple[MaterialFieldBuffers, TimeControlVariables, LevelsetFieldBuffers, SolidFieldBuffers, StepInformation]
        """

        equation_type = self.equation_information.equation_type
        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset
        diffuse_interface_model = self.equation_information.diffuse_interface_model
        solid_coupling = self.equation_information.solid_coupling
        extension_setup_primitives = self.numerical_setup.levelset.extension.primitives
        extension_setup_solids = self.numerical_setup.levelset.extension.solids

        is_positivity_logging = self.numerical_setup.output.logging.is_positivity
        is_levelset_residuals_logging = self.numerical_setup.output.logging.is_levelset_residuals
        is_only_last_stage_logging = self.numerical_setup.output.logging.is_only_last_stage

        is_feed_forward = control_flow_params.is_feed_foward

        # UNPACK BUFFERS
        conservatives = material_fields.conservatives
        primitives = material_fields.primitives
        temperature = material_fields.temperature

        physical_simulation_time = time_control_variables.physical_simulation_time
        physical_timestep_size = time_control_variables.physical_timestep_size
        simulation_step = time_control_variables.simulation_step

        levelset = levelset_fields.levelset
        apertures = levelset_fields.apertures
        volume_fraction = levelset_fields.volume_fraction

        # NOTE these are cell indices used for static fluid solid simulations
        solid_cell_indices = levelset_fields.solid_cell_indices
        interface_cells = solid_cell_indices.interface
        extension_cells_fluid = solid_cell_indices.extension_fluid
        extension_cells_solid = solid_cell_indices.extension_solid

        interface_velocity = levelset_fields.interface_velocity
        interface_pressure = levelset_fields.interface_pressure

        solid_velocity = solid_fields.velocity
        solid_temperature = solid_fields.temperature
        solid_energy = solid_fields.energy

        # NOTE CREATE INTEGRATION BUFFERS
        integration_buffers = get_integration_buffers(
            conservatives,
            levelset,
            volume_fraction,
            solid_velocity,
            solid_energy,
            self.domain_information,
            self.equation_information
        )
        initial_stage_buffers = integration_buffers if self.time_integrator.no_stages > 1 else None
        
        current_time_stage = physical_simulation_time

        levelset_residuals_info_list = []
        positivity_state_info_list = []

        # LOOP STAGES
        for stage in range(self.time_integrator.no_stages):
            
            is_logging_stage = True
            if is_only_last_stage_logging:
                if stage != self.time_integrator.no_stages - 1:
                    is_logging_stage = False

            # TODO
            # # CALLBACK on_stage_start
            # conservatives, primitives = self._callback("on_stage_start", conservatives=conservatives, primitives=primitives,
            #     physical_timestep_size=physical_timestep_size, physical_simulation_time=physical_simulation_time, levelset=levelset,
            #     volume_fraction=volume_fraction, apertures=apertures, forcings=forcing_parameters, 
            #     ml_setup=ml_setup)

            # RIGHT HAND SIDE
            (
                rhs_buffers,
                positivity_counter,
                discretization_counter
            ) = self.space_solver.compute_rhs(
                conservatives,
                primitives,
                temperature,
                current_time_stage,
                physical_timestep_size,
                levelset,
                volume_fraction,
                apertures,
                interface_velocity,
                interface_pressure,
                solid_velocity,
                solid_temperature,
                interface_cells,
                forcing_buffers,
                ml_setup,
                is_feed_forward
            )

            # PERFORM STAGE INTEGRATION
            integration_buffers = self.time_integrator.perform_stage_integration(
                integration_buffers,
                rhs_buffers,
                initial_stage_buffers,
                physical_timestep_size,
                stage,
                self.equation_information
            )

            # UNPACK INTEGRATED BUFFERS
            integration_euler_buffers = integration_buffers.euler_buffers
            conservatives = integration_euler_buffers.conservatives
            
            levelset = integration_euler_buffers.levelset if is_moving_levelset else levelset
            solid_velocity = integration_euler_buffers.solid_velocity if solid_coupling.dynamic == "TWO-WAY" else solid_velocity
            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError
            else:
                solid_energy = solid_energy

            # POSITIVITY
            diffuse_interface_model = self.equation_information.diffuse_interface_model
            is_volume_fraction_limiter = self.numerical_setup.conservatives.positivity.is_volume_fraction_limiter

            if diffuse_interface_model and is_volume_fraction_limiter:
                conservatives, vf_correction_count = self.positivity_handler.correct_volume_fraction(conservatives)
                positivity_counter = positivity_counter._replace(volume_fraction_limiter=vf_correction_count)

            # UPDATE STAGE TIME
            increment_factor = self.time_integrator.timestep_increment_factor[stage]
            current_time_stage = physical_simulation_time + physical_timestep_size * increment_factor

            # TREAD INTEGRATED LEVELSET FIELDS
            if is_moving_levelset:
                is_last_stage = stage == self.time_integrator.no_stages - 1
                perform_reinitialization = control_flow_params.perform_reinitialization
                (
                    levelset,
                    volume_fraction_new,
                    apertures,
                    info_reinit
                ) = self.levelset_handler.treat_integrated_levelset(
                    levelset,
                    perform_reinitialization,
                    is_last_stage
                )

            else:
                volume_fraction_new = volume_fraction
                info_reinit = None

            # TREAT INTEGRATED MATERIAL
            if levelset_model:

                # NOTE mixing for fluid, returns volume averaged conservatives and
                # corresponding primitives
                (
                    primitives,
                    conservatives,
                    mixing_invalid_cells_fluid,
                    mixing_invalid_cell_count_fluid,
                ) = self.levelset_handler.mixing_material_fields(
                    conservatives, primitives, levelset, volume_fraction_new,
                    volume_fraction, current_time_stage,
                    solid_cell_indices, ml_setup
                )

                # NOTE mixing for solid, returns volume averaged solid energy
                # and corresponding solid temperature
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError
                
                volume_fraction = volume_fraction_new

                # NOTE at this point, real part of buffers contain volume averaged states of new timestep
                # we need to update ghost cells now

                # NOTE for fluid solid conjugate heat, we need interface heat flux
                # for extension procedure, if we use interpolation based extension
                if solid_coupling.thermal == "TWO-WAY" and any(
                    method == "INTERPOLATION" for method
                    in (extension_setup_primitives.method, extension_setup_solids.method)
                ):
                    raise NotImplementedError
                else:
                    interface_heat_flux = None
                    interface_temperature = None

                (
                    conservatives,
                    primitives,
                    extension_invalid_cell_count_fluid,
                    info_prime_extension
                ) = self.levelset_handler.extension_material_fields(
                    conservatives, primitives, levelset,
                    volume_fraction, current_time_stage,
                    mixing_invalid_cells_fluid,
                    solid_temperature, solid_velocity,
                    interface_heat_flux, interface_temperature,
                    extension_cells_fluid,
                    ml_setup=ml_setup
                )

                levelset_positivity_fluid_info = LevelsetPositivityInformation(
                    mixing_invalid_cell_count_fluid,
                    extension_invalid_cell_count_fluid)
            
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError

                else:
                    levelset_positivity_solid_info = None
                    info_solids_extension = None

            else:

                primitives = self.equation_manager.get_primitives_from_conservatives(
                    conservatives)
                levelset_positivity_fluid_info = None
                info_prime_extension = None
                levelset_positivity_solid_info = None
                info_solids_extension = None

            # DIFFUSE INTERFACE COMPRESSION
            is_interface_compression = self.numerical_setup.diffuse_interface.interface_compression.is_interface_compression
            diffuse_interface_model = self.equation_information.diffuse_interface_model

            if diffuse_interface_model and is_interface_compression:
                perform_compression = control_flow_params.perform_compression
                interface_compression_flag = stage == self.time_integrator.no_stages - 1 and perform_compression
                conservatives, primitives = self.diffuse_interface_handler.perform_interface_compression(
                    conservatives, primitives, current_time_stage, interface_compression_flag)

            # HALO UPDATE
            fill_edge_halos_material = self.halo_manager.fill_edge_halos_material
            fill_vertex_halos_material = self.halo_manager.fill_vertex_halos_material
            primitives, conservatives = self.halo_manager.perform_halo_update_material(
                primitives,
                current_time_stage, 
                fill_edge_halos_material,
                fill_vertex_halos_material, 
                conservatives,
                ml_setup=ml_setup
            )
            if self.equation_information.is_compute_temperature: 
                temperature = self.material_manager.get_temperature(primitives)
                temperature = self.halo_manager.perform_outer_halo_update_temperature(
                    temperature, current_time_stage)
            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

            # INTERFACE QUANTITIES
            if equation_type == "TWO-PHASE-LS":
                fluid_fluid_handler = self.levelset_handler.fluid_fluid_handler
                (
                    interface_velocity,
                    interface_pressure,
                    info_interface_extension
                ) = fluid_fluid_handler.compute_interface_quantities(
                    primitives, levelset, volume_fraction,
                    interface_velocity, interface_pressure,
                    ml_setup=ml_setup
                )
            else:
                info_interface_extension = None

            # LEVELSET RESIDUALS INFO
            condition = all((levelset_model, is_levelset_residuals_logging,
                             not is_feed_forward, is_logging_stage))
            if condition:
                levelset_residuals_info = LevelsetResidualInformation(
                    info_reinit, info_prime_extension,
                    info_interface_extension, info_solids_extension
                    )
                levelset_residuals_info_list.append(levelset_residuals_info)

            # POSITIVITY STATE INFO
            condition = all((is_positivity_logging, not is_feed_forward,
                             is_logging_stage))
            if condition:
                positivity_state_info = get_positvity_state_info(
                    primitives, temperature,
                    positivity_counter, discretization_counter, 
                    volume_fraction, levelset_positivity_fluid_info,
                    levelset_positivity_solid_info,
                    self.material_manager, self.equation_information,
                    self.domain_information)
                positivity_state_info_list.append(positivity_state_info)

            # NOTE CREATE INTEGRATION FOR NEXT STAGE            
            if stage < self.time_integrator.no_stages - 1:
                integration_buffers = get_integration_buffers(
                    conservatives,
                    levelset,
                    volume_fraction,
                    solid_velocity,
                    solid_energy,
                    self.domain_information,
                    self.equation_information
                )

            # TODO
            # # CALLBACK on_stage_end
            # conservatives, primitives = self._callback("on_stage_end", conservatives=conservatives, primitives=primitives,
            #     physical_timestep_size=physical_timestep_size, physical_simulation_time=current_time_stage, levelset=levelset,
            #     volume_fraction=volume_fraction, apertures=apertures, forcings=forcings, 
            #     ml_setup=ml_setup)

        # INCREMENT PHYSICAL SIMULATION TIME
        physical_simulation_time += physical_timestep_size
        simulation_step += 1

        # CREATE CONTAINERS
        material_fields = MaterialFieldBuffers(
            conservatives, primitives, temperature)

        time_control_variables = time_control_variables._replace(
            simulation_step=simulation_step,
            physical_simulation_time=physical_simulation_time
        )
    
        solid_fields = SolidFieldBuffers(
            solid_velocity, solid_energy, solid_temperature)

        levelset_fields = LevelsetFieldBuffers(
            levelset, volume_fraction, apertures,
            interface_velocity, interface_pressure,
            solid_cell_indices)

        step_information = StepInformation(
            tuple(positivity_state_info_list),
            tuple(levelset_residuals_info_list),
        )

        return (
            material_fields,
            time_control_variables,
            levelset_fields, 
            solid_fields,
            step_information
        )

    def _callback(
            self,
            hook_name: str,
            jxf_buffers: JaxFluidsBuffers = None,
            callback_dict: Dict = None,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Executes the hook_name method of all callbacks. 

        :param hook_name: _description_
        :type hook_name: str
        :param jxf_buffers: _description_, defaults to None
        :type jxf_buffers: JaxFluidsBuffers, optional
        :param callback_dict: _description_, defaults to None
        :type callback_dict: Dict, optional
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Tuple[JaxFluidsBuffers, Dict]
        """

        if hook_name in ("on_step_start", "on_step_end"):
            for cb in self.callbacks:
                fn = getattr(cb, hook_name)
                jxf_buffers, callback_dict = fn(
                    jxf_buffers=jxf_buffers,
                    callback_dict=callback_dict,
                    **kwargs
                )
                
            return jxf_buffers, callback_dict
        
        elif hook_name in (
            "on_simulation_start",
            "on_simulation_end",
            "before_step_start",
            "after_step_end"
        ):
            # NOTE these callbacks can have state
            for cb in self.callbacks:
                fn = getattr(cb, hook_name)
                jxf_buffers, callback_dict = fn(
                    jxf_buffers=jxf_buffers,
                    callback_dict=callback_dict,
                    **kwargs
                )

            return jxf_buffers, callback_dict

        elif hook_name in ("on_stage_start", "on_stage_end"):
            for cb in self.callbacks:
                fn = getattr(cb, hook_name)
                _, callback_dict = fn(**kwargs)

            return None, None

        else:
            raise NotImplementedError


    ### WRAPPER FUNCTIONS ###
    def do_integration_step(
            self,
            jxf_buffers: JaxFluidsBuffers,
            control_flow_params: ControlFlowParameters,
            ml_parameters: ParametersSetup,
            ml_callables: CallablesSetup,
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Wrapper for the _do_integration_step function 
        that specifies single (jit) or multi (pmap) 
        GPU execution of the integration step.
        For argument description see base function.
        """
        if self.domain_information.is_parallel:
            return self._do_integration_step_pmap(
                    jxf_buffers,
                    control_flow_params,
                    ml_parameters,
                    ml_callables
                )
        else:
            return self._do_integration_step_jit(
                    jxf_buffers,
                    control_flow_params,
                    ml_parameters,
                    ml_callables
                )

    # JIT AND PMAP WRAPPER FOR DO INTEGRATION STEP
    @partial(
        jax.pmap,
        static_broadcasted_argnums=(0, 2, 4),
        in_axes=(None, JaxFluidsBuffers(0, None, None, None), None, None, None),
        out_axes=(JaxFluidsBuffers(0, None, None, None), None),
        axis_name="i"
    )
    def _do_integration_step_pmap(
            self,
            jxf_buffers: JaxFluidsBuffers,
            control_flow_params: ControlFlowParameters,
            ml_parameters: ParametersSetup,
            ml_callables: CallablesSetup,
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Pmap wrapper for the _do_integration_step function.
        For argument description see base function.
        """
        return self._do_integration_step(
                jxf_buffers,
                control_flow_params,
                ml_parameters,
                ml_callables
            )

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def _do_integration_step_jit(
            self,
            jxf_buffers: JaxFluidsBuffers,
            control_flow_params: ControlFlowParameters,
            ml_parameters: ParametersSetup,
            ml_callables: CallablesSetup,
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Jit wrapper for the _do_integration_step function.
        For argument description see base function.
        """
        return self._do_integration_step(
                jxf_buffers,
                control_flow_params,
                ml_parameters,
                ml_callables
            )
        
    def feed_forward(
            self, 
            batch_primes_init: Array, 
            physical_timestep_size: Array, 
            t_start: Array,
            feed_forward_setup: FeedForwardSetup,
            batch_levelset_init: Array = None,
            batch_solid_temperature_init: Array = None,
            batch_solid_interface_velocity_init: Array = None,
            ml_parameters: ParametersSetup = ParametersSetup(),
            ml_callables: CallablesSetup = CallablesSetup(),
        ) -> Tuple[Tuple[Array], Array]:
        """Vectorized version of the _feed_forward() method.

        :param batch_primes_init: batch of initial primitive variable buffers
        :type batch_primes_init: Array
        :param batch_levelset_init: batch of initial levelset buffers
        :type batch_levelset_init: Array
        :param n_steps: Number of integration steps
        :type n_steps: int
        :param physical_timestep_size: batch of physical time step sizes
        :type physical_timestep_size: Array
        :param t_start: batch of physical start times
        :type t_start: Array
        :param output_freq: Frequency in time steps for output, defaults to 1
        :type output_freq: int, optional
        :param ml_networks_dict: NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        return jax.vmap(
                self._feed_forward,
                in_axes=(0, 0, 0, None, 0, 0, 0, None, None),
                out_axes=(0, 0)
            )(
                batch_primes_init,
                physical_timestep_size,
                t_start,
                feed_forward_setup,
                batch_levelset_init,
                batch_solid_temperature_init,
                batch_solid_interface_velocity_init,
                ml_parameters,
                ml_callables
            )

    def _feed_forward(
            self, 
            primes_init: Array, 
            physical_timestep_size: float, 
            t_start: float, 
            feed_forward_setup: FeedForwardSetup,
            levelset_init: Array = None,
            solid_temperature_init: Array = None,
            solid_velocity_init: Array = None,   
            ml_parameters: ParametersSetup = ParametersSetup(),
            ml_callables: CallablesSetup = CallablesSetup(),
        ) -> Tuple[Tuple[Array], Array]:
        """Advances the initial buffers in time for a fixed amount of steps and returns the
        entire trajectory. This function is differentiable and
        must therefore be used to end-to-end optimize ML models within the JAX-FLUIDS simulator.

        :param primes_init: Initial primitive variables buffer
        :type primes_init: Array
        :param levelset_init: Initial levelset buffer
        :type levelset_init: Array
        :param n_steps: Number of time steps
        :type n_steps: int
        :param physical_timestep_size: Physical time step size
        :type physical_timestep_size: float
        :param t_start: Physical start time
        :type t_start: float
        :param output_freq: Frequency in time steps for output, defaults to 1
        :type output_freq: int, optional
        :param ml_parameters: _description_, defaults to None
        :type ml_parameters: MachineLearningParametersSetup, optional
        :param ml_callables: _description_, defaults to None
        :type ml_callables: MachineLearningCallablesSetup, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        def post_process_fn(
                simulation_buffers: SimulationBuffers,
            ) -> Dict[str, Array]:
            # TODO should we add time to this???
            # TODO @dbezgin should be user input???
            nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
            nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
            material_fields = simulation_buffers.material_fields
            levelset_fields = simulation_buffers.levelset_fields
            solid_fields = simulation_buffers.solid_fields

            primitives = material_fields.primitives
            volume_fraction = levelset_fields.volume_fraction
            levelset = levelset_fields.levelset
            solid_temperature = solid_fields.temperature

            is_include_halos = feed_forward_setup.is_include_halos

            quantity_list = self.equation_information.primitive_quantities
            primitives = self.unit_handler.dimensionalize(primitives, "specified", quantity_list)

            feed_forward_output = {
                "primitives": primitives if is_include_halos else primitives[...,nhx,nhy,nhz]
            }

            if self.equation_information.levelset_model:
                
                # NOTE level-set and volume fraction only required
                # at every outer step when level-set is moving
                is_moving_levelset = self.equation_information.is_moving_levelset
                if is_moving_levelset:
                    levelset = self.unit_handler.dimensionalize(levelset, "length")
                    feed_forward_output["levelset"] = levelset if is_include_halos else levelset[nhx,nhy,nhz]
                    feed_forward_output["volume_fraction"] = volume_fraction if is_include_halos else volume_fraction[nhx_,nhy_,nhz_]

                solid_coupling = self.equation_information.solid_coupling
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError
                
            return feed_forward_output

        ml_setup = combine_callables_and_params(ml_callables, ml_parameters)

        multistep = configure_multistep(
            do_integration_step_fn=self._do_integration_step,
            post_process_fn=post_process_fn,
            feed_forward_setup=feed_forward_setup,
            ml_callables=ml_callables)

        simulation_buffers, time_control_variables, forcing_parameters = initialize_fields_feed_forward(
            sim_manager=self, primes_init=primes_init,
            physical_timestep_size=physical_timestep_size, t_start=t_start,
            levelset_init=levelset_init,
            solid_temperature_init=solid_temperature_init,
            solid_velocity_init=solid_velocity_init,
            ml_setup=ml_setup
        )

        solution_array, times_array = multistep(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters)

        times_array = self.unit_handler.dimensionalize(times_array, "time")

        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset
        if levelset_model:
            if not is_moving_levelset:
                nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
                nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry

                levelset = simulation_buffers.levelset_fields.levelset
                volume_fraction = simulation_buffers.levelset_fields.volume_fraction
                levelset = self.unit_handler.dimensionalize(levelset, "length")

                is_include_halos = feed_forward_setup.is_include_halos
                solution_array["levelset"] = levelset if is_include_halos else levelset[nhx,nhy,nhz]
                solution_array["volume_fraction"] = volume_fraction if is_include_halos else volume_fraction[nhx_,nhy_,nhz_]


        return solution_array, times_array

