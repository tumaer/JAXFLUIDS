from functools import partial
import time
from typing import List, Tuple, Union, Dict

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.callbacks.base_callback import Callback
from jaxfluids.data_types.buffers import ForcingBuffers, IntegrationBuffers, IntegrationBuffers, \
    SimulationBuffers, MaterialFieldBuffers, LevelsetFieldBuffers, ForcingParameters, \
    TimeControlVariables
from jaxfluids.data_types.information import LevelsetResidualInformation, StepInformation, \
    WallClockTimes, TurbulentStatisticsInformation
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup import CaseSetup
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.feedforward_utils import configure_multistep, initialize_fields_for_feedforward
from jaxfluids.forcing.forcing import Forcing
from jaxfluids.input.input_manager import InputManager
from jaxfluids.io_utils.output_writer import OutputWriter
from jaxfluids.io_utils.logger import Logger
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks, compute_cut_cell_mask
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.space_solver import SpaceSolver
from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler
from jaxfluids.time_integration.BDF import BDF_Solver
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.turb.statistics.turb_stats_manager_online import TurbulentOnlineStatisticsManager
from jaxfluids.turb.statistics.turb_stats_manager_postprocess import TurbulentStatisticsManager, \
    turbulent_statistics_for_logging
from jaxfluids.config import precision
from jaxfluids.materials import DICT_MATERIAL

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

        # TIME INTEGRATION
        self.fixed_timestep = self.numerical_setup.conservatives.time_integration.fixed_timestep
        self.end_step = self.case_setup.general_setup.end_step
        self.end_time = self.case_setup.general_setup.end_time

        time_integrator = self.numerical_setup.conservatives.time_integration.integrator
        self.time_integrator: TimeIntegrator = time_integrator(
            nh = self.domain_information.nh_conservatives,
            inactive_axes = self.domain_information.inactive_axes)
        
        # LEVELSET HANDLER
        if self.equation_information.levelset_model:
            self.levelset_handler = LevelsetHandler(
                domain_information=self.domain_information,
                numerical_setup=self.numerical_setup,
                material_manager=self.material_manager,
                equation_manager=self.equation_manager,
                halo_manager=self.halo_manager,
                solid_properties=self.case_setup.solid_properties_setup)
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
            diffuse_interface_handler=self.diffuse_interface_handler)

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

        self.numerical_setup.active_physics.is_geometric_source
        # FORCINGS
        if self.numerical_setup.active_forcings:
            self.forcings_computer = Forcing(
                domain_information=self.domain_information,
                equation_information=self.equation_information,
                material_manager=self.material_manager,
                unit_handler=self.unit_handler,
                forcing_setup=self.case_setup.forcing_setup,
                active_forcings_setups=self.numerical_setup.active_forcings,
                active_physics=self.numerical_setup.active_physics)
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
            numerical_setup=self.numerical_setup,
            jax_backend=jax.default_backend(),
            is_multihost=self.domain_information.is_multihost)

        # CALLBACKS INIT
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callbacks = callbacks or []
        for cb in self.callbacks:
            assert isinstance(cb, Callback)
            cb.init_callback(
                domain_information  = self.domain_information,
                material_manager    = self.material_manager,
                halo_manager        = self.halo_manager,
                logger              = self.logger,
                output_writer       = self.output_writer)

    def simulate(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters = ForcingParameters(),
            ml_parameters_dict: Dict = None,
            ml_networks_dict: Dict = None,
            ) -> int:
        """Performs a conventional CFD simulation.

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_
        :type forcing_parameters: ForcingParameters
        :return: _description_
        :rtype: _type_
        """
        self.initialize(
            simulation_buffers,
            time_control_variables,
            forcing_parameters)
        return_value = self.advance(
            simulation_buffers,
            time_control_variables,
            forcing_parameters,
            ml_parameters_dict,
            ml_networks_dict)
        return return_value

    def initialize(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters = None
            ) -> None:
        """ Initializes the simulation, i.e., creates the
        output directory, logs the numerical and case setup,
        and writes the initial output.

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_, defaults to None
        :type forcing_parameters: ForcingParameters, optional
        """

        self.sanity_check(simulation_buffers, time_control_variables, forcing_parameters)

        # CREATE OUTPUT FOLDER, CASE SETUP AND NUMERICAL SETUP
        save_path_case, save_path_domain, save_path_statistics \
            = self.output_writer.configure_output_writer()

        # CONFIGURE LOGGER AND LOG NUMERICAL SETUP AND CASE SETUP
        self.logger.configure_logger(save_path_case)
        self.logger.log_initialization()
        self.logger.log_numerical_setup_and_case_setup(*self.input_manager.info())

        # WRITE INITIAL OUTPUT
        self.output_writer.set_simulation_start_time(time_control_variables.physical_simulation_time)

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
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters = None,
            ml_parameters_dict: Dict = None,
            ml_networks_dict = None,
            ) -> None:
        """Advances the initial buffers in time.

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_, defaults to None
        :type forcing_parameters: ForcingParameters, optional
        :return: _description_
        :rtype: _type_
        """

        # LOG SIMULATION START
        self.logger.log_sim_start()

        # START LOOP
        start_loop = self.synchronize_and_clock(
            simulation_buffers.material_fields.primitives)

        # CALLBACK on_simulation_start
        # buffer_dictionary = self._callback("on_simulation_start",
        #   buffer_dictionary=buffer_dictionary)

        physical_simulation_time = time_control_variables.physical_simulation_time
        simulation_step = time_control_variables.simulation_step

        wall_clock_times = WallClockTimes()

        while physical_simulation_time < self.end_time and \
            simulation_step < self.end_step:

            start_step = self.synchronize_and_clock(
                simulation_buffers.material_fields.primitives)

            # COMPUTE REINITIALIZATION FLAG
            if self.equation_information.levelset_model:
                perform_reinitialization = \
                self.levelset_handler.get_reinitialization_flag(
                    time_control_variables.simulation_step)
            else:
                perform_reinitialization = None

            # COMPUTE INTERFACE COMPRESSION FLAG
            if self.equation_information.diffuse_interface_model:
                perform_compression = \
                    self.diffuse_interface_handler.get_compression_flag(
                        time_control_variables.simulation_step)
            else:
                perform_compression = None
            
            # PERFORM INTEGRATION STEP
            simulation_buffers, time_control_variables, \
            forcing_parameters, step_information = \
            self.do_integration_step(
                simulation_buffers,
                time_control_variables,
                forcing_parameters,
                perform_reinitialization,
                perform_compression,
                ml_parameters_dict,
                ml_networks_dict)

            # CLOCK INTEGRATION STEP
            end_step = self.synchronize_and_clock(
                simulation_buffers.material_fields.primitives)
            wall_clock_step = end_step - start_step

            # COMPUTE WALL CLOCK TIMES FOR TIME STEP
            wall_clock_times = self.compute_wall_clock_time(
                wall_clock_step, wall_clock_times,
                time_control_variables.simulation_step)

            # LOG TERMINAL END TIME STEP
            self.logger.log_end_time_step(
                time_control_variables, step_information,
                wall_clock_times, self.unit_handler.time_reference)

            # WRITE H5 OUTPUT
            self.output_writer.write_output(
                simulation_buffers, time_control_variables,
                wall_clock_times, forcing_parameters)

            # UNPACK FOR WHILE LOOP
            physical_simulation_time = time_control_variables.physical_simulation_time
            simulation_step = time_control_variables.simulation_step

        # CALLBACK on_simulation_end
        # buffer_dictionary = self._callback("on_simulation_end",
        #   buffer_dictionary=buffer_dictionary)

        # FINAL OUTPUT
        self.output_writer.write_output(
            simulation_buffers, time_control_variables,
            wall_clock_times, forcing_parameters,
            force_output=True, simulation_finish=True)

        # LOG SIMULATION FINISH
        end_loop = self.synchronize_and_clock(
            simulation_buffers.material_fields.primitives)
        self.logger.log_sim_finish(end_loop - start_loop)

        return bool(physical_simulation_time >= self.end_time)

    def compute_wall_clock_time(
            self,
            wall_clock_step: float,
            wall_clock_times: WallClockTimes,
            simulation_step: jnp.int32
            ) -> WallClockTimes:
        """Computes the instantaneous 
        and mean wall clock time for the
        a single simulation steps.

        :param wall_clock_step: _description_
        :type wall_clock_step: float
        :param simulation_step: _description_
        :type simulation_step: jnp.int32
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
            mean_wall_clock_step, mean_wall_clock_step_cell)
            
        return wall_clock_times

    def synchronize_and_clock(
            self,
            buffer: Array,
            all_reduce: bool = False
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
        if self.domain_information.is_multihost and all_reduce:
            local_device_count = self.domain_information.local_device_count
            host_sync_buffer = np.ones(local_device_count)
            host_sync_buffer = jax.pmap(lambda x: jax.lax.psum(x, axis_name="i"),
                axis_name="i")(host_sync_buffer)
            host_sync_buffer.block_until_ready()
        return time.time()

    def _do_integration_step(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            perform_reinitialization: bool,
            perform_compression: bool,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            is_feedforward: bool = False,
            ) -> Tuple[SimulationBuffers, TimeControlVariables,
            ForcingParameters, StepInformation]:
        """Performs an integration step.
        1) Compute timestep size
        2) Compute forcings
        3) Do Runge Kutta stages
        4) Compute simulation information, i.e.,
            positivity state, turbulent statistics
            etc.

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_
        :type forcing_parameters: ForcingParameters
        :param perform_reinitialization: _description_
        :type perform_reinitialization: bool
        :param ml_parameters_dict: _description_, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: _description_, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[SimulationBuffers, TimeControlVariables, ForcingParameters, StepInformation]
        """
        material_fields = simulation_buffers.material_fields
        levelset_fields = simulation_buffers.levelset_fields

        if not is_feedforward:
            # COMPUTE TIMESTEP 
            physical_timestep_size = self.compute_timestep(
                material_fields.primitives,
                levelset_fields.levelset,
                levelset_fields.volume_fraction)
            
            time_control_variables = TimeControlVariables(
                time_control_variables.physical_simulation_time,
                time_control_variables.simulation_step,
                physical_timestep_size)

        # COMPUTE FORCINGS
        active_forcings = self.numerical_setup.active_forcings
        if any(active_forcings._asdict().values()):
            forcing_buffers, forcing_parameters, \
            forcing_infos = self.forcings_computer.compute_forcings(
                simulation_buffers, time_control_variables,
                forcing_parameters, self.do_runge_kutta_stages,
                ml_parameters_dict=ml_parameters_dict,
                ml_networks_dict=ml_networks_dict)
        else:
            forcing_buffers, forcing_infos = None, None

        # PERFORM INTEGRATION STEP
        material_fields, time_control_variables, \
        levelset_fields, \
        step_information = self.do_runge_kutta_stages(
            material_fields, time_control_variables,
            levelset_fields,
            forcing_buffers, perform_reinitialization,
            perform_compression,
            ml_parameters_dict, ml_networks_dict,
            is_feedforward)

        # CREATE CONTAINERS
        simulation_buffers = SimulationBuffers(
            material_fields, levelset_fields)

        step_information = StepInformation(
            step_information.positivity_state_info_list,
            step_information.levelset_residuals_info_list,
            step_information.levelset_positivity_info_list,
            forcing_info=forcing_infos)

        return simulation_buffers, time_control_variables, \
            forcing_parameters, step_information

    def do_runge_kutta_stages(
            self,
            material_fields: MaterialFieldBuffers,
            time_control_variables: TimeControlVariables,
            levelset_fields: LevelsetFieldBuffers = None,
            forcing_buffers: ForcingBuffers = None, 
            perform_reinitialization: bool = False,
            perform_compression: bool = False,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            is_feedforward: bool = False
            ) -> Tuple[MaterialFieldBuffers, TimeControlVariables,
            LevelsetFieldBuffers, StepInformation]:
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
        :param levelset_fields: _description_
        :type levelset_fields: LevelsetFieldBuffers
        :param forcing_buffers: _description_, defaults to None
        :type forcing_buffers: ForcingBuffers, optional
        :param perform_reinitialization: _description_, defaults to False
        :type perform_reinitialization: bool, optional
        :param ml_parameters_dict: _description_, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: _description_, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Returns MaterialFieldBuffers, TimeControlVariables,
            LevelsetRelatedFieldBuffers, LevelSetResiduals,
            PositivityCount, VolumeFractionCorrectionCount
        :rtype: Tuple[MaterialFieldBuffers, TimeControlVariables,
            LevelsetFieldBuffers, ParticleBuffers, LevelsetResidualInformation,
            jnp.int32]
        """

        equation_type = self.equation_information.equation_type
        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset
        diffuse_interface_model = self.equation_information.diffuse_interface_model
        is_positivity_logging = self.numerical_setup.output.logging.is_positivity
        is_levelset_residuals_logging = self.numerical_setup.output.logging.is_levelset_residuals
        is_only_last_stage_logging = self.numerical_setup.output.logging.is_only_last_stage

        conservatives = material_fields.conservatives
        primitives = material_fields.primitives

        physical_simulation_time = time_control_variables.physical_simulation_time
        physical_timestep_size = time_control_variables.physical_timestep_size
        simulation_step = time_control_variables.simulation_step

        levelset = levelset_fields.levelset
        apertures = levelset_fields.apertures
        volume_fraction = levelset_fields.volume_fraction
        interface_velocity = levelset_fields.interface_velocity
        interface_pressure = levelset_fields.interface_pressure

        integration_buffers = IntegrationBuffers(
            conservatives, levelset,
            interface_velocity)

        initial_stage_buffers = self.get_initial_buffers_for_stage_integration(
            integration_buffers, volume_fraction)
        
        current_time_stage = physical_simulation_time

        levelset_residuals_info_list = []
        positivity_state_info_list = []

        # LOOP STAGES
        for stage in range( self.time_integrator.no_stages ):

            is_logging_stage = True
            if is_only_last_stage_logging:
                if stage != self.time_integrator.no_stages - 1:
                    is_logging_stage = False

            # RIGHT HAND SIDE
            rhs_buffers, positivity_count_flux, \
            positivity_count_interpolation, \
            positivity_count_thinc, \
            positivity_count_acdi, \
            count_acdi \
            = self.space_solver.compute_rhs(
                conservatives, primitives, current_time_stage,
                physical_timestep_size, levelset, volume_fraction,
                apertures, interface_velocity, interface_pressure,
                forcing_buffers, ml_parameters_dict, ml_networks_dict,
                is_feedforward)

            # PERFORM STAGE INTEGRATION
            integration_buffers = self.perform_stage_integration(
                integration_buffers, rhs_buffers, initial_stage_buffers,
                physical_timestep_size, stage, volume_fraction)

            # UNPACK INTEGRATED BUFFERS
            conservatives = integration_buffers.conservatives
            levelset = integration_buffers.levelset
            interface_velocity = integration_buffers.interface_velocity

            # POSITIVITY
            if self.numerical_setup.conservatives.positivity.is_volume_fraction_limiter \
                and self.equation_information.diffuse_interface_model:
                conservatives, vf_correction_count \
                = self.positivity_handler.correct_volume_fraction(conservatives)
            else:
                vf_correction_count = None

            # UPDATE STAGE TIME
            increment_factor = self.time_integrator.timestep_increment_factor[stage]
            current_time_stage = physical_simulation_time + physical_timestep_size * increment_factor
        
            # REINITIALIZE LEVELSET AND PERFORM INTERFACE RECONSTRUCTION
            if is_moving_levelset:
                is_last_stage = stage == self.time_integrator.no_stages - 1
                levelset, volume_fraction_new, apertures, reinitialization_step_count \
                = self.levelset_handler.treat_integrated_levelset(levelset, perform_reinitialization,
                                                                  is_last_stage)
            else:
                volume_fraction_new = volume_fraction
                reinitialization_step_count = 0

            # MIX CONSERVATIVES - COMPUTE PRIMITIVES - GHOST CELL EXTENSION
            if levelset_model:
                conservatives, primitives, levelset_positivity_info, \
                prime_extension_step_count = self.levelset_handler.treat_integrated_material_fields(
                    conservatives, primitives, levelset,
                    volume_fraction_new, volume_fraction,
                    current_time_stage, interface_velocity)
                volume_fraction = volume_fraction_new
            else:
                primitives = self.equation_manager.get_primitives_from_conservatives(
                    conservatives)
                levelset_positivity_info = None

            # DIFFUSE INTERFACE COMPRESSION
            if self.equation_information.diffuse_interface_model \
                and self.numerical_setup.diffuse_interface.interface_compression.is_interface_compression:
                interface_compression_flag = stage == self.time_integrator.no_stages - 1 \
                    and perform_compression
                conservatives, primitives = self.diffuse_interface_handler.perform_interface_compression(
                    conservatives, primitives, current_time_stage, interface_compression_flag)

            # MATERIAL HALO UPDATE
            active_physics = self.numerical_setup.active_physics
            is_viscous_flux = active_physics.is_viscous_flux
            primitives, conservatives = \
            self.halo_manager.perform_halo_update_material(
                primitives, current_time_stage, is_viscous_flux,
                False, conservatives)

            # INTERFACE QUANTITIES AND RESIDUAL INFO
            if equation_type == "TWO-PHASE-LS":
                interface_velocity, interface_pressure, interface_extension_step_count = \
                self.levelset_handler.compute_interface_quantities(
                    primitives, levelset, volume_fraction, interface_velocity, interface_pressure)
                if is_levelset_residuals_logging and not is_feedforward and is_logging_stage:
                    levelset_residuals_info = self.levelset_handler.compute_residuals(
                        primitives, volume_fraction, levelset,
                        interface_velocity, interface_pressure,
                        reinitialization_step_count, prime_extension_step_count,
                        interface_extension_step_count)
                    levelset_residuals_info_list.append(levelset_residuals_info)
            elif equation_type == "SINGLE-PHASE-SOLID-LS":
                if is_levelset_residuals_logging and not is_feedforward and is_logging_stage:
                    levelset_residuals_info = self.levelset_handler.compute_residuals(
                        primitives, volume_fraction, levelset, None, None,
                        reinitialization_step_count, prime_extension_step_count)
                    levelset_residuals_info_list.append(levelset_residuals_info)

            # POSITIVITY STATE INFO
            if is_positivity_logging and not is_feedforward and is_logging_stage:
                positivity_state_info = self.positivity_handler.get_positvity_state_info(
                    primitives, positivity_count_flux, positivity_count_interpolation,
                    vf_correction_count, positivity_count_thinc, positivity_count_acdi,
                    count_acdi, volume_fraction, levelset_positivity_info)
                positivity_state_info_list.append(positivity_state_info)
            
            integration_buffers = IntegrationBuffers(
                conservatives, levelset, interface_velocity)

        # INCREMENT PHYSICAL SIMULATION TIME
        physical_simulation_time += physical_timestep_size
        simulation_step += 1

        # CREATE CONTAINERS
        material_fields = MaterialFieldBuffers(
            conservatives, primitives)

        time_control_variables = TimeControlVariables(
            physical_simulation_time, simulation_step,
            physical_timestep_size)

        levelset_fields = LevelsetFieldBuffers(
            levelset, volume_fraction, apertures,
            interface_velocity, interface_pressure)

        step_information = StepInformation(
            positivity_state_info_list, levelset_residuals_info_list)

        return material_fields, time_control_variables, \
            levelset_fields, step_information

    def get_initial_buffers_for_stage_integration(
            self,
            integration_buffers: IntegrationBuffers,
            volume_fraction: Array,
            ) -> IntegrationBuffers:
        """Creates the initial stage buffers required
        for later stages within the runge kutta scheme.

        :param integration_buffers: _description_
        :type integration_buffers: IntegrationBuffers
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :return: _description_
        :rtype: IntegrationBuffers
        """

        conservatives = integration_buffers.conservatives
        levelset = integration_buffers.levelset
        interface_velocity = integration_buffers.interface_velocity

        if self.time_integrator.no_stages > 1:

            if self.equation_information.levelset_model:
                init_conservatives = self.levelset_handler.transform_to_conservatives(
                    conservatives, volume_fraction)
            else:
                init_conservatives = conservatives
            
            if self.equation_information.is_moving_levelset:
                init_levelset = levelset
            else:
                init_levelset = None
                
            if self.equation_information.levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
                init_solid_interface_velocity = interface_velocity
            else:
                init_solid_interface_velocity = None
        
            initial_stage_buffers = IntegrationBuffers(
                init_conservatives, init_levelset,
                init_solid_interface_velocity)
        
        else:
            initial_stage_buffers = None

        return initial_stage_buffers
    
    def perform_stage_integration(
            self,
            integration_buffers: IntegrationBuffers,
            rhs_buffers: IntegrationBuffers,
            initial_stage_buffers: IntegrationBuffers,
            physical_timestep_size: float,
            stage: jnp.int32,
            volume_fraction: Array
            ) -> IntegrationBuffers:
        """Performs a stage integration step.
        1) Transform volume-averaged conservatives
            to actual conservatives (only for levelset
            simulations)
        2) Compute stage buffer
        3) Integrate

        :param integration_buffers: _description_
        :type integration_buffers: IntegrationBuffers
        :param initial_stage_buffers: _description_
        :type initial_stage_buffers: IntegrationBuffers
        :param stage: _description_
        :type stage: jnp.int32
        :return: _description_
        :rtype: IntegrationBuffers
        """

        conservatives = integration_buffers.conservatives
        levelset = integration_buffers.levelset
        interface_velocity = integration_buffers.interface_velocity

        rhs_conservatives = rhs_buffers.conservatives
        rhs_levelset = rhs_buffers.levelset
        rhs_solid_interface_velocity = rhs_buffers.interface_velocity

        # TRANSFORM TO REAL CONSERVATIVES
        if self.equation_information.levelset_model:
            conservatives = self.levelset_handler.transform_to_conservatives(
                conservatives, volume_fraction)

        # PREPARE BUFFERS FOR STAGE INTEGRATION
        if stage > 0:

            initial_conservatives = initial_stage_buffers.conservatives
            initial_levelset = initial_stage_buffers.levelset
            initial_solid_velocity = initial_stage_buffers.interface_velocity

            conservatives = self.time_integrator.prepare_buffer_for_integration(
                conservatives, initial_conservatives, stage)
            if self.equation_information.is_moving_levelset:
                levelset = self.time_integrator.prepare_buffer_for_integration(
                    levelset, initial_levelset, stage)
            if self.equation_information.levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
                interface_velocity = self.time_integrator.prepare_buffer_for_integration(
                    interface_velocity, initial_solid_velocity, stage)

        # PERFORM INTEGRATION
        conservatives = self.time_integrator.integrate(
            conservatives, rhs_conservatives,
            physical_timestep_size, stage)

        if self.equation_information.is_moving_levelset:
            levelset = self.time_integrator.integrate(
                levelset, rhs_levelset,
                physical_timestep_size, stage)

        if self.equation_information.levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            interface_velocity = self.time_integrator.integrate(
                interface_velocity, rhs_solid_interface_velocity,
                physical_timestep_size, stage) 

        # CREATE CONTAINER
        integration_buffers = IntegrationBuffers(
            conservatives, levelset,
            interface_velocity)

        return integration_buffers

    def compute_timestep(
            self,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array
            ) -> jnp.float32:
        """Computes the physical time step size
        depending on the active physics.

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :return: _description_
        :rtype: jnp.float32
        """
        if self.fixed_timestep:

            dt = self.fixed_timestep

        else:

            # DOMAIN INFORMATION
            nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
            nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
            active_axes_indices = self.domain_information.active_axes_indices
            active_physics_setup = self.numerical_setup.active_physics

            equation_type = self.equation_information.equation_type
            levelset_model = self.equation_information.levelset_model

            # First min over cell_sizes in i direction, then min over axes.
            # Necessary for mesh stretching.
            min_cell_size = self.domain_information.smallest_cell_size

            energy_ids = self.equation_information.energy_ids
            vf_slices = self.equation_information.vf_slices

            alpha = None
            density = self.material_manager.get_density(primitives[...,nhx,nhy,nhz])
            pressure = primitives[energy_ids,...,nhx,nhy,nhz]

            if equation_type == "DIFFUSE-INTERFACE-5EQM":
                alpha = primitives[(vf_slices,) + (...,nhx,nhy,nhz)]

            # COMPUTE TEMPERATURE
            # ONLY IN THE DOMAIN, I.E., WITHOUT HALOS
            if active_physics_setup.is_viscous_flux \
                or active_physics_setup.is_heat_flux:
                temperature = self.material_manager.get_temperature(
                    primitives[...,nhx,nhy,nhz], pressure,
                    density, volume_fractions=alpha)

            # COMPUTE MASKS
            if levelset_model:
                mask_real = compute_fluid_masks(volume_fraction, levelset_model)
                nh_offset = self.domain_information.nh_offset
                mask_cut_cells = compute_cut_cell_mask(levelset, nh_offset)
                mask_real *= (1 - mask_cut_cells)

            speed_of_sound = self.material_manager.get_speed_of_sound(
                primitives[...,nhx,nhy,nhz], pressure, density,
                volume_fractions=alpha)

            abs_velocity = 0.0
            for i in self.equation_information.velocity_ids:
                abs_velocity += (jnp.abs(primitives[i,...,nhx,nhy,nhz]) + speed_of_sound)
                if levelset_model:
                    abs_velocity *= mask_real[..., nhx_,nhy_,nhz_]
            dt = min_cell_size / ( jnp.max(abs_velocity) + self.eps )

            # VISCOUS CONTRIBUTION
            if active_physics_setup.is_viscous_flux:
                const = 3.0 / 14.0
                kinematic_viscosity = self.material_manager.get_dynamic_viscosity(
                    temperature,
                    primitives[...,nhx,nhy,nhz],
                    ) / density
                if levelset_model:
                    kinematic_viscosity = kinematic_viscosity * mask_real[..., nhx_,nhy_,nhz_]
                dt_viscous = const * (min_cell_size * min_cell_size) / (jnp.max(kinematic_viscosity) + self.eps)
                dt = jnp.minimum(dt, dt_viscous)

            # HEAT TRANSFER CONTRIBUTION
            if active_physics_setup.is_heat_flux:
                const = 0.1
                cp = self.material_manager.get_specific_heat_capacity(
                    temperature, primitives[...,nhx,nhy,nhz])
                thermal_diffusivity = self.material_manager.get_thermal_conductivity(
                    temperature,
                    primitives[...,nhx,nhy,nhz],
                    ) / (density * cp)
                if levelset_model:
                    thermal_diffusivity = thermal_diffusivity * mask_real[..., nhx_,nhy_,nhz_]
                dt_thermal = const * (min_cell_size * min_cell_size) / (jnp.max(thermal_diffusivity) + self.eps)
                dt = jnp.minimum(dt, dt_thermal)

            # DIFFUSION SHARPENING CONTRIBUTION
            if self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening:
                dt_diffusion_sharpening = \
                    self.diffuse_interface_handler.compute_diffusion_sharpening_timestep(
                        primitives)
                dt = jnp.minimum(dt, dt_diffusion_sharpening)

            # PARALLEL
            if self.domain_information.is_parallel:
                dt = jax.lax.pmin(dt, axis_name="i")

            CFL = self.numerical_setup.conservatives.time_integration.CFL
            dt *= CFL

        return dt

    def _callback(
            self,
            hook_name: str,
            buffer_dictionary: Dict = None,
            conservatives: Array = None,
            primitives: Array = None,
            **kwargs
            ) -> Union[Dict, Tuple[Array, Array]]:
        """Executes the hook_name method of all callbacks. 

        :param hook_name: Str indentifier of the callback routine.
        :type hook_name: str
        """

        if hook_name in ("on_simulation_start", "on_simulation_end", "on_step_start", "on_step_end"):
            for cb in self.callbacks:
                fn = getattr(cb, hook_name)
                buffer_dictionary = fn(buffer_dictionary, **kwargs)
                
            return buffer_dictionary
        
        elif hook_name in ("on_stage_start", "on_stage_end"):
            for cb in self.callbacks:
                fn = getattr(cb, hook_name)
                conservatives, primitives = fn(conservatives, primitives, **kwargs)

            return conservatives, primitives

        else:
            raise NotImplementedError


    ### WRAPPER FUNCTIONS ###
    def do_integration_step(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            perform_reinitialization: bool,
            perform_compression: bool,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            ) -> Tuple[SimulationBuffers, TimeControlVariables,
            ForcingParameters, StepInformation]:
        """Wrapper for the _do_integration_step function 
        that specifies single (jit) or multi (pmap) 
        GPU execution of the integration step.
        For argument description see base function.
        """
        if self.domain_information.is_parallel:
            return self._do_integration_step_pmap(
                    simulation_buffers,
                    time_control_variables,
                    forcing_parameters,
                    perform_reinitialization,
                    perform_compression,
                    ml_parameters_dict,
                    ml_networks_dict)
        else:
            return self._do_integration_step_jit(
                    simulation_buffers,
                    time_control_variables,
                    forcing_parameters,
                    perform_reinitialization,
                    perform_compression,
                    ml_parameters_dict,
                    ml_networks_dict)

    # JIT AND PMAP WRAPPER FOR DO INTEGRATION STEP
    # TODO why is ml_parameters_dict mapped?
    @partial(jax.pmap,
        static_broadcasted_argnums=(0,4,5,7),
        in_axes=(None,0,None,None,None,None,None,None),
        out_axes=(0,None,None,None),
        axis_name="i")
    def _do_integration_step_pmap(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            perform_reinitialization: bool,
            perform_compression: bool,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            ) -> Tuple[SimulationBuffers, TimeControlVariables,
            ForcingParameters, StepInformation]:
        """Pmap wrapper for the _do_integration_step function.
        For argument description see base function.
        """
        return self._do_integration_step(
                simulation_buffers,
                time_control_variables,
                forcing_parameters,
                perform_reinitialization,
                perform_compression,
                ml_parameters_dict,
                ml_networks_dict)

    @partial(jax.jit, static_argnums=(0,4,5,7))
    def _do_integration_step_jit(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            perform_reinitialization: bool,
            perform_compression: bool,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            ) -> Tuple[SimulationBuffers, TimeControlVariables,
            ForcingParameters, StepInformation]:
        """Jit wrapper for the _do_integration_step function.
        For argument description see base function.
        """
        return self._do_integration_step(
                simulation_buffers,
                time_control_variables,
                forcing_parameters,
                perform_reinitialization,
                perform_compression,
                ml_parameters_dict,
                ml_networks_dict)
        
    def feed_forward(
            self, 
            batch_primes_init: Array, 
            physical_timestep_size: Array, 
            t_start: float, 
            outer_steps: int, 
            inner_steps: int = 1,
            is_scan: bool = False,
            is_checkpoint: bool = True,
            is_include_t0: bool = True,
            batch_levelset_init: Array = None,
            batch_solid_interface_velocity_init: Array = None,
            ml_parameters_dict: Union[Dict, None] = None, 
            ml_networks_dict: Union[Dict, None] = None
        ) -> Tuple[Array, Array]:
        """Vectorized version of the _feed_forward() method.

        :param batch_primes_init: batch of initial primitive variable buffers
        :type batch_primes_init: Array
        :param batch_levelset_init: batch of initial levelset buffers
        :type batch_levelset_init: Array
        :param n_steps: Number of integration steps
        :type n_steps: int
        :param physical_timestep_size: Physical time step size
        :type physical_timestep_size: float
        :param t_start: Physical start time
        :type t_start: float
        :param output_freq: Frequency in time steps for output, defaults to 1
        :type output_freq: int, optional
        :param ml_parameters_dict: NN weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        return jax.vmap(
                self._feed_forward,
                in_axes=(0,0,None,None,None,None,None,None,0,0,None,None),
                out_axes=(0,0,))(
            batch_primes_init,
            physical_timestep_size,
            t_start,
            outer_steps,
            inner_steps,
            is_scan,
            is_checkpoint,
            is_include_t0,
            batch_levelset_init,
            batch_solid_interface_velocity_init,
            ml_parameters_dict,
            ml_networks_dict)

    def _feed_forward(
            self, 
            primes_init: Array, 
            physical_timestep_size: float, 
            t_start: float, 
            outer_steps: int, 
            inner_steps: int = 1,
            is_scan: bool = False,
            is_checkpoint: bool = True,
            is_include_t0: bool = True,
            levelset_init: Array = None,   
            solid_interface_velocity_init: Array = None,   
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None
        ) -> Tuple[Array, Array]:
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
        :param ml_parameters_dict: _description_, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: _description_, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        def post_process_fn(simulation_buffers: SimulationBuffers
            ) -> Tuple[Array]:
            # TODO @dbezgin should be user input???
            nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
            nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
            material_fields = simulation_buffers.material_fields
            levelset_fields = simulation_buffers.levelset_fields

            primitives = material_fields.primitives
            volume_fraction = levelset_fields.volume_fraction
            levelset = levelset_fields.levelset

            if self.equation_information.levelset_model:
                out_buffer = (
                    primitives[...,nhx,nhy,nhz], 
                    levelset[nhx,nhy,nhz],
                    volume_fraction[nhx_,nhy_,nhz_],)
            else:
                out_buffer = (primitives[:,nhx,nhy,nhz],)
            return out_buffer

        multistep = configure_multistep(
            do_integration_step_fn=self._do_integration_step,
            post_process_fn=post_process_fn,
            outer_steps=outer_steps, inner_steps=inner_steps,
            is_scan=is_scan, is_checkpoint=is_checkpoint,
            is_include_t0=is_include_t0, ml_networks_dict=ml_networks_dict)

        simulation_buffers, time_control_variables, \
        forcing_parameters = initialize_fields_for_feedforward(
            sim_manager=self, primes_init=primes_init,
            physical_timestep_size=physical_timestep_size, t_start=t_start,
            levelset_init=levelset_init,
            solid_interface_velocity_init=solid_interface_velocity_init)

        solution_array, times_array = multistep(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters_dict)

        return solution_array, times_array