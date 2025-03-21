from datetime import datetime
import logging
from platform import python_version
import os
import textwrap
from typing import Dict, List

import git

import jax
from jax import version as jax_version
from jaxlib import version as jaxlib_version
import jax.numpy as jnp

import jaxfluids
from jaxfluids.data_types.buffers import TimeControlVariables
from jaxfluids.data_types.information import StepInformation, WallClockTimes
from jaxfluids.data_types.case_setup import CaseSetup
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.io_utils.helper_functions import prepate_turbulent_statistics_for_logging

Array = jax.Array


class Logger:
    """Logger for the JAX-FLUIDS solver.
    Logs information during the simulation to file and/or screen."""

    def __init__(
            self,
            case_setup: CaseSetup,
            numerical_setup: NumericalSetup,
            logger_name: str = "",
            jax_backend: str = None,
            is_multihost: bool = False
            ) -> None:

        self.logger_name = logger_name
        # TODO do we need to pass numerical setup or maybe 
        # passing logging_setup is sufficient and pass 
        # numerical setup to the methods separately
        self.case_setup = case_setup
        self.numerical_setup = numerical_setup
        self.is_multihost = is_multihost

        self.level_dict = {
            "DEBUG": logging.DEBUG, 
            "INFO": logging.INFO, 
            "WARNING": logging.WARNING, 
            "ERROR": logging.ERROR, 
            "NONE": logging.CRITICAL}

        self.is_streamoutput = True
        logging_level = numerical_setup.output.logging.level
        if logging_level in ["DEBUG_TO_FILE", "INFO_TO_FILE"]:
            logging_level = logging_level[:-8]
            self.is_streamoutput = False

        self.logging_level = self.level_dict[logging_level]
        self.logging_frequency = numerical_setup.output.logging.frequency

        self.python_version = python_version()
        self.jax_version = jax_version.__version__
        self.jaxlib_version = jaxlib_version.__version__
        self.jaxfluids_version = jaxfluids.__version__

        try:
            repo = git.Repo(
                path=os.path.abspath(__file__),
                search_parent_directories=True)
            self.git_sha = repo.head.object.hexsha
        except git.exc.InvalidGitRepositoryError:
            self.git_sha = "None"
        except:
            self.git_sha = "None"

        self.today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        self.process_id = str(os.getpid())
        self.jax_backend = jax_backend

    def configure_logger(self, log_path: str) -> None:
        """Configures the logger. Sets up formatter, file and 
        stream handler. 

        :param log_path: Path to which logs are saved.
        :type log_path: str
        """
        logger = logging.getLogger(self.logger_name)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.logging_level)
        formatter = logging.Formatter('%(message)s')

        process_id = jax.process_index()
        if self.is_streamoutput and process_id == 0:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.logging_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        
        if self.is_multihost:
            output_log_name = "output_proc%d.log" % process_id
        else:
            output_log_name = "output.log"
        file_handler = logging.FileHandler(os.path.join(os.path.abspath(log_path), output_log_name))
        file_handler.setLevel(self.logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def log_jax_fluids(self) -> None:
        self.logger.info("*                                                                              *")
        self.logger.info("*          _     _    __  __        _____  _      _   _  ___  ____   ____      *")
        self.logger.info("*         | |   / \   \ \/ /       |  ___|| |    | | | ||_ _||  _ \ / ___|     *")
        self.logger.info("*      _  | |  / _ \   \  /  _____ | |_   | |    | | | | | | | | | |\___ \     *")
        self.logger.info("*     | |_| | / ___ \  /  \ |_____||  _|  | |___ | |_| | | | | |_| | ___) |    *")
        self.logger.info("*      \___/ /_/   \_\/_/\_\       |_|    |_____| \___/ |___||____/ |____/     *")
        self.logger.info("*                                                                              *")

    def log_copyright(self) -> None:
        self.logger.info("*------------------------------------------------------------------------------*")
        self.logger.info("* JAX-FLUIDS -                                                                 *")
        self.logger.info("*                                                                              *")
        self.logger.info("* A fully-differentiable CFD solver for compressible two-phase flows.          *") 
        self.logger.info("*                                                                              *")
        self.logger.info("* MIT License                                                                  *")
        self.logger.info("* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *")
        self.logger.info("*                                                                              *")
        self.logger.info("* Permission is hereby granted, free of charge, to any person obtaining a copy *")
        self.logger.info("* of this software and associated documentation files (the 'Software'),        *")
        self.logger.info("* to deal in the Software without restriction, including without limitation    *")
        self.logger.info("* the rights to use, copy, modify, merge, publish, distribute, sublicense,     *")
        self.logger.info("* and/or sell copies of the Software, and to permit persons to whom            *")
        self.logger.info("* the Software is furnished to do so, subject to the following conditions:     *")
        self.logger.info("*                                                                              *")
        self.logger.info("* The above copyright notice and this permission notice shall be included in   *")
        self.logger.info("* all copies or substantial portions of the Software.                          *")
        self.logger.info("*                                                                              *")
        self.logger.info("* THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   *")
        self.logger.info("* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     *")
        self.logger.info("* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  *")
        self.logger.info("* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       *")
        self.logger.info("* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,              *")
        self.logger.info("* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE           *")
        self.logger.info("* OR OTHER DEALINGS IN THE SOFTWARE.                                           *")
        self.logger.info("*                                                                              *")
        self.logger.info("*------------------------------------------------------------------------------*")
        self.logger.info("*                                                                              *")
        self.logger.info("* CONTACT                                                                      *")
        self.logger.info("*                                                                              *")
        self.logger.info("* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *")               
        self.logger.info("*                                                                              *")
        self.logger.info("*------------------------------------------------------------------------------*")
        self.logger.info("*                                                                              *")
        self.logger.info("* Munich, April 15th, 2022                                                     *")
        self.logger.info("*                                                                              *")
        self.logger.info("*------------------------------------------------------------------------------*")

    def log_initialization(self) -> None:
        """Logs the initialization of the SimulationManager.
        """
        self.hline()
        self.nline()
        self.log_jax_fluids()
        self.logger.info(f"*{'By BB - ML@AER':^78}*")
        self.nline()
        self.log_copyright()
        self.nline()

    def log_sim_start(self) -> None:
        """Logs the simulation start.
        """
        self.hline()
        self.nline()
        self.log_jax_fluids()
        self.nline()
        self.hline()
        self.nline()
        self.logger.info(f"*{'PYTHON Version: ' + self.python_version:^78}*")
        self.logger.info(f"*{'JAX Version: ' + self.jax_version:^78}*")
        self.logger.info(f"*{'JAXLIB Version: ' + self.jaxlib_version:^78}*")
        self.logger.info(f"*{'JAX-Fluids Version: ' + self.jaxfluids_version:^78}*")
        self.logger.info(f"*{'GIT Commit: ' + self.git_sha:^78}*")
        self.logger.info(f"*{'DATE & TIME: ' + self.today:^78}*")
        self.logger.info(f"*{'PROCESS ID: ' + self.process_id:^78}*")
        if self.jax_backend is not None:
            self.logger.info(f"*{'JAX BACKEND: ' + self.jax_backend:^78}*")
        self.nline()
        self.hline()

    def log_sim_finish(self, end_time: float) -> None:
        """Logs the simulation end.

        :param end_time: Final simulation time.
        :type end_time: float
        """
        self.hline()
        self.nline()
        self.logger.info(f"*{'SIMULATION FINISHED SUCCESSFULLY':^78}*")
        simulation_time_str = f"SIMULATION TIME {end_time:.3e}s"
        self.logger.info(f"*{simulation_time_str:^78}*")
        self.log_jax_fluids()
        self.nline()
        self.hline()

        self._shutdown_logger()

    def _shutdown_logger(self) -> None:
        """Shutsdown logger.
        Closes handlers and removes them from logger.
        """
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
    def log_numerical_setup_and_case_setup(
            self,
            numerical_setup_dict: Dict,
            case_setup_dict: Dict,
            mesh_info_dict: Dict
        ) -> None:
        """Logs numerical setup and input file.

        :param numerical_setup_dict: Dictionary which contains information on the numerical setup.
        :type numerical_setup_dict: Dict
        :param case_setup_dict: Dictionary which contains information on the case setup.
        :type case_setup_dict: Dict
        """
        def log_dict(log_input, level: int = 0,
                     width: int = 80, indent: int = 2,
                     left_indent_level0: int = 4) -> None:
            left_space = left_indent_level0 + indent * level
            available_space = width - (left_space + 2) 

            key_list = list(log_input.keys())
            if key_list != list():
                max_key_len = max([len(key) for key in key_list])
            else:
                max_key_len = 0
            for key, item in log_input.items():
                if isinstance(item, dict):
                    self.logger.info(f"*{'':<{left_space}}{key.upper():<{available_space}}*")
                    log_dict(item, level+1, width)
                    if level == 0:
                        self.logger.info(f"*{'':<{width - 2}}*")
                elif isinstance(item, list) and item != []:
                    if isinstance(item[0], dict):
                        self.logger.info(f"*{'':<{left_space}}{key.upper():<{available_space}}*")
                        for list_item in item:
                            log_dict(list_item, level+1, width)
                            self.logger.info(f"*{'':<{width - 2}}*")
                    else:
                        log_item(left_space, key, item, available_space, max_key_len)
                else:
                    log_item(left_space, key, item, available_space, max_key_len)

        def log_item(left_space, key, item, available_space, max_key_len) -> None:
            available_space_key = (max_key_len // 4 + 1) * 4
            available_space_item = max(0, available_space - available_space_key - 5)
            log_str = str(item) if not isinstance(item, str) else item
            len_str = len(log_str)
            if len_str > available_space_item:
                self.logger.info(f"*{'':<{left_space}}{key.upper():<{available_space_key}}:    {'':<{available_space_item}}*")
                log_str_wrap = textwrap.wrap(log_str, available_space - 8)
                for i, line in enumerate(log_str_wrap):
                    indent = 2 if i > 0 else 0
                    self.logger.info(f"*{'':<{left_space}}{'':<{4 + indent}}{line:<{available_space - 8}}{'':<{4 - indent}}*")
                self.logger.info(f"*{'':<{left_space + available_space}}*")
            else:
                self.logger.info(f"*{'':<{left_space}}{key.upper():<{available_space_key}}:    {log_str:<{available_space_item}}*")

        # LOG NUMERICAL SETUP
        self.nline()
        self.logger.info(f"*{'NUMERICAL SETUP':^78}*")
        self.nline()
        log_dict(numerical_setup_dict, width=80)     
        self.nline()
        self.hline()

        # LOG CASE SETUP
        self.nline()
        self.logger.info(f"*{'CASE SETUP':^78}*")
        self.nline()
        log_dict(case_setup_dict, width=80)        
        self.nline()
        self.hline()

        # LOG MESH INFORMATION
        self.nline()
        self.logger.info(f"*{'MESH INFORMATION':^78}*")
        self.nline()
        log_dict(mesh_info_dict, width=80)        
        self.nline()

    def log_list(self, input_list: List) -> None:
        """Logs every line in input_list.

        :param input_list: List of strings to be printed at the start
            of an integration step. 
        :type input_list: List
        """
        self.nline()
        for line in input_list:
            self.logger.info(f"*    {line:<74}*")
        self.nline()


    def log_dict(self, input_dict: Dict, dict_name: str = None) -> None:
        """Logs every key-value pair in input_dict:

        Args:
            input_dict (Dict): [description]
        """
        self.nline()
        if dict_name:
            self.logger.info(f"*    {dict_name:<74}*")
        for key, value in input_dict.items():
            out_str = f"{key:<20}    = {value:4.3e}"
            self.logger.info(f"*    {out_str:<74}*")
        self.nline()

    def log_initial_time_step(
        self,
        time_control_variables: TimeControlVariables,
        step_information: StepInformation,
        time_reference: float,
        ) -> None:

        # TIME CONTROL
        physical_simulation_time_dimensional = time_control_variables.physical_simulation_time * time_reference
        physical_timestep_size_dimensional = time_control_variables.physical_timestep_size * time_reference
        print_list = [
            "TIME CONTROL",
            f"CURRENT TIME                       = {physical_simulation_time_dimensional:4.5e}",
            f"CURRENT DT                         = {physical_timestep_size_dimensional:4.5e}",
            f"CURRENT STEP                       = {time_control_variables.simulation_step:6d}"
        ]
        self.log_list(print_list)

        # POSITVITY STATE
        diffuse_interface_model = self.numerical_setup.diffuse_interface.model
        logging_setup = self.numerical_setup.output.logging
        is_positivity = logging_setup.is_positivity
        is_levelset_residuals = logging_setup.is_levelset_residuals

        levelset_residuals = step_information.levelset[0]
        positivity_state = step_information.positivity[0]
        statistics = step_information.statistics

        if is_positivity:
            print_list = [f"POSITIVITY STATE"]
            if diffuse_interface_model:
                print_list += [ f"MIN ALPHARHO                       = {positivity_state.min_alpharho:8.7e}" ]
                if diffuse_interface_model == "5EQM":
                    print_list += [ f"MIN ALPHA                          = {positivity_state.min_alpha:8.7e}" ]
                    print_list += [ f"MAX ALPHA                          = {positivity_state.max_alpha:8.7e}" ]
            print_list += [ f"MIN DENSITY                        = {positivity_state.min_density:4.4e}" ]
            print_list += [ f"MIN PRESSURE                       = {positivity_state.min_pressure:4.4e}" ]
            self.log_list(print_list)
            
        # LEVELSET STATE
        levelset_setup = self.numerical_setup.levelset
        levelset_model = levelset_setup.model
        solid_coupling = levelset_setup.solid_coupling
        is_interpolate_invalid_cells_extension_primitives = levelset_setup.extension.primitives.iterative.is_interpolate_invalid_cells
        is_interpolate_invalid_cells_extension_solids = levelset_setup.extension.solids.iterative.is_interpolate_invalid_cells
        is_use_iterative_procedure_extension_primitives = levelset_setup.extension.primitives.method == "ITERATIVE"
        is_use_iterative_procedure_extension_solids = levelset_setup.extension.solids.method == "ITERATIVE"
        is_log = levelset_model and (is_levelset_residuals or is_positivity)
        
        if is_log:
            primitives_extension_info = levelset_residuals.primitive_extension
            if primitives_extension_info is not None:
                prime_extension_mean = primitives_extension_info.mean_residual

            extension_invalid_cell_count_fluid = positivity_state.levelset_fluid.extension_invalid_cell_count

            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

            interface_extension_info = levelset_residuals.interface_extension
            if interface_extension_info is not None:
                interface_extension_mean = interface_extension_info.mean_residual

            reinitialization_info = levelset_residuals.reinitialization
            if reinitialization_info is not None:
                reinitialzation_max = reinitialization_info.max_residual

            solid_temperature_extension_info = levelset_residuals.solids_extension
            if solid_temperature_extension_info is not None:
                solid_temperature_extension_residual_mean = solid_temperature_extension_info.mean_residual

            print_list = [f"LEVELSET STATE"]
            print_list += ["MATERIAL"]
            if levelset_model == "FLUID-FLUID":
                if is_interpolate_invalid_cells_extension_primitives:
                    print_list += [ f"  EXTENSION INVALID CELLS          =  {extension_invalid_cell_count_fluid[0]:10d} / {extension_invalid_cell_count_fluid[1]:10d}" ]
            else:
                if is_interpolate_invalid_cells_extension_primitives and is_use_iterative_procedure_extension_primitives:
                    print_list += [ f"  EXTENSION INVALID CELLS          =  {extension_invalid_cell_count_fluid:10d}" ]
            if is_use_iterative_procedure_extension_primitives:
                print_list += [ f"  PRIMITIVE EXTENSION RESIDUAL     = {prime_extension_mean:4.5e}" ]
            if interface_extension_info is not None:
                print_list += [ f"  INTERFACE EXTENSION RESIDUAL     = {interface_extension_mean:4.5e}" ]

            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

            if reinitialization_info is not None:
                print_list += ["LEVELSET"]
                print_list += [ f"  REINITIALIZATION RESIDUAL        = {reinitialzation_max:4.5e}" ]
            
            self.log_list(print_list)

        # TURBULENCE STATISTICS
        if self.case_setup.statistics_setup.turbulence.is_logging:
            self.log_list(
                prepate_turbulent_statistics_for_logging(statistics.turbulence.logging)
            )

        self.hline()


    def log_end_time_step(
            self,
            time_control_variables: TimeControlVariables,
            step_information: StepInformation,
            wall_clock_times: WallClockTimes,
            time_reference: float,
            ) -> None:
        """Logs information at the end of an integration step.

        :param info_list: List of strings to be printed at the end
            of an integration step.
        :type info_list: List
        """

        if time_control_variables.simulation_step % self.logging_frequency == 0:

            active_forcings = self.numerical_setup.active_forcings
            forcing_infos = step_information.forcing_info
            levelset_residuals_info_list = step_information.levelset
            positivity_state_info_list = step_information.positivity
            statistics = step_information.statistics
            
            # TIME CONTROL
            physical_simulation_time_dimensional = time_control_variables.physical_simulation_time * time_reference
            physical_timestep_size_dimensional = time_control_variables.physical_timestep_size * time_reference
            print_list = [
                "TIME CONTROL",
                f"CURRENT TIME                       = {physical_simulation_time_dimensional:4.5e}",
                f"CURRENT DT                         = {physical_timestep_size_dimensional:4.5e}",
                f"CURRENT STEP                       = {time_control_variables.simulation_step:6d}",
                f"WALL CLOCK TIMESTEP                = {wall_clock_times.step:4.5e}",
                f"WALL CLOCK TIMESTEP CELL           = {wall_clock_times.step_per_cell:4.5e}",
                f"MEAN WALL CLOCK TIMESTEP CELL      = {wall_clock_times.mean_step_per_cell:4.5e}",
            ]
            self.log_list(print_list)
            
            # FORCINGS
            if forcing_infos is not None:
                if forcing_infos.mass_flow != None:
                    mass_flow_target = forcing_infos.mass_flow.target_value
                    mass_flow_current = forcing_infos.mass_flow.current_value
                    mass_flow_forcing_scalar = forcing_infos.mass_flow.force_scalar
                    self.log_list([
                        "MASS FLOW CONTROL",
                        f"TARGET VALUE   = {mass_flow_target:4.5e}",
                        f"CURRENT VALUE  = {mass_flow_current:4.5e}",
                        f"FORCE SCALAR   = {mass_flow_forcing_scalar:4.5e}"
                    ])

                if active_forcings.is_temperature_forcing or active_forcings.is_solid_temperature_forcing:
                    temperature_error_fluid = forcing_infos.temperature.current_error_fluid
                    temperature_error_solid = forcing_infos.temperature.current_error_solid
                    print_list = ["TEMPERATURE CONTROL"]
                    if temperature_error_fluid is not None:
                        print_list += [f"ERROR FLUID  = {temperature_error_fluid:4.5e}"]
                    if temperature_error_solid is not None:
                        print_list += [f"ERROR SOLID  = {temperature_error_solid:4.5e}"]
                    self.log_list(print_list)

            # TURBULENCE STATISTICS
            if self.case_setup.statistics_setup.turbulence.is_logging:
                self.log_list(prepate_turbulent_statistics_for_logging(statistics.turbulence.logging))

            # POSITVITY STATE
            diffuse_interface_model = self.numerical_setup.diffuse_interface.model
            logging_setup = self.numerical_setup.output.logging
            positivity_setup = self.numerical_setup.conservatives.positivity
            is_positivity = logging_setup.is_positivity
            is_only_last_stage = logging_setup.is_only_last_stage

            if is_positivity:
                for stage, positivity_state in enumerate(positivity_state_info_list):
                    if is_only_last_stage:
                        print_list = [f"POSITIVITY STATE"]
                    else:
                        print_list = [f"POSITIVITY STATE STAGE {stage:d}"]
                    positivity_counter = positivity_state.positivity_counter
                    if positivity_setup.is_interpolation_limiter:
                        print_list += [ f"COUNT INTERPOLATION LIMITER        = {positivity_counter.interpolation_limiter:d}" ]
                    if positivity_setup.is_thinc_interpolation_limiter:
                        print_list += [ f"COUNT INTERPOLATION LIMITER THINC  = {positivity_counter.thinc_limiter:d}" ]
                    if positivity_setup.flux_limiter:
                        print_list += [ f"COUNT FLUX LIMITER                 = {positivity_counter.flux_limiter:d}" ]
                    if diffuse_interface_model:
                        if positivity_setup.is_volume_fraction_limiter:
                            print_list += [ f"COUNT VOLUME FRACTION LIMITER      = {positivity_counter.volume_fraction_limiter:d}" ]
                        if positivity_setup.is_acdi_flux_limiter:
                            print_list += [ f"COUNT CDI/ACDI FLUX                = {positivity_state.discretization_counter.acdi:d}"]
                            print_list += [ f"COUNT CDI/ACDI FLUX LIMITER        = {positivity_counter.acdi_limiter:d}"]
                    if diffuse_interface_model:
                        print_list += [ f"MIN ALPHARHO                       = {positivity_state.min_alpharho:8.7e}" ]
                        if diffuse_interface_model == "5EQM":
                            print_list += [ f"MIN ALPHA                          = {positivity_state.min_alpha:8.7e}" ]
                            print_list += [ f"MAX ALPHA                          = {positivity_state.max_alpha:8.7e}" ]
                    print_list += [ f"MIN DENSITY                        = {positivity_state.min_density:4.4e}" ]
                    print_list += [ f"MIN PRESSURE                       = {positivity_state.min_pressure:4.4e}" ]
                    self.log_list(print_list)

            # LEVELSET STATE
            levelset_setup = self.numerical_setup.levelset
            levelset_model = levelset_setup.model
            is_levelset_residuals = logging_setup.is_levelset_residuals
            is_log = levelset_model and (is_levelset_residuals or is_positivity)
            solid_coupling = levelset_setup.solid_coupling
            is_interpolate_invalid_cells_extension_primitives = levelset_setup.extension.primitives.iterative.is_interpolate_invalid_cells
            is_interpolate_invalid_cells_extension_solids = levelset_setup.extension.solids.iterative.is_interpolate_invalid_cells
            is_use_iterative_procedure_extension_primitives = levelset_setup.extension.primitives.method == "ITERATIVE" 
            is_use_iterative_procedure_extension_solids = levelset_setup.extension.solids.method == "ITERATIVE"

            if is_log:
                for stage, (levelset_residuals, positivity_state) in \
                enumerate(zip(levelset_residuals_info_list, positivity_state_info_list)):
                    
                    levelset_fluid = positivity_state.levelset_fluid
                    mixing_invalid_cells_count_fluid = levelset_fluid.mixing_invalid_cell_count
                    extension_invalid_cell_count_fluid = levelset_fluid.extension_invalid_cell_count

                    primitives_extension_info = levelset_residuals.primitive_extension
                    if primitives_extension_info is not None:
                        prime_extension_mean = primitives_extension_info.mean_residual
                        prime_extension_steps = primitives_extension_info.steps

                    levelset_solid = positivity_state.levelset_solid
                    if levelset_solid is not None:
                        mixing_invalid_cells_count_solid = levelset_solid.mixing_invalid_cell_count
                        extension_invalid_cell_count_solid = levelset_solid.extension_invalid_cell_count

                    interface_extension_info = levelset_residuals.interface_extension
                    if interface_extension_info is not None:
                        interface_extension_mean = interface_extension_info.mean_residual
                        interface_quantity_extension_steps = interface_extension_info.steps

                    reinitialization_info = levelset_residuals.reinitialization
                    if reinitialization_info is not None:
                        reinitialzation_max = reinitialization_info.max_residual
                        reinitialization_steps = reinitialization_info.steps

                    solid_temperature_extension_info = levelset_residuals.solids_extension
                    if solid_temperature_extension_info is not None:
                        solid_temperature_extension_residual_mean = solid_temperature_extension_info.mean_residual
                        solid_temperature_extension_steps = solid_temperature_extension_info.steps

                    if is_only_last_stage:
                        print_list = [f"LEVELSET STATE"]
                    else:
                        print_list = [f"LEVELSET STATE STAGE {stage:d}"]
                    
                    print_list += ["MATERIAL"]
                    if levelset_model == "FLUID-FLUID":
                        print_list += [ f"  MIXING INVALID CELLS             =  {mixing_invalid_cells_count_fluid[0]:10d} / {mixing_invalid_cells_count_fluid[1]:10d}" ]
                        if is_interpolate_invalid_cells_extension_primitives and is_use_iterative_procedure_extension_primitives:
                            print_list += [ f"  EXTENSION INVALID CELLS          =  {extension_invalid_cell_count_fluid[0]:10d} / {extension_invalid_cell_count_fluid[1]:10d}" ]
                    else:
                        print_list += [ f"  MIXING INVALID CELLS             =  {mixing_invalid_cells_count_fluid:10d}" ]
                        if is_interpolate_invalid_cells_extension_primitives and is_use_iterative_procedure_extension_primitives:
                            print_list += [ f"  EXTENSION INVALID CELLS          =  {extension_invalid_cell_count_fluid:10d}" ]

                    if is_use_iterative_procedure_extension_primitives:
                        print_list += [ f"  PRIMITIVE EXTENSION STEPS        = {prime_extension_steps:11d}" ]
                        print_list += [ f"  PRIMITIVE EXTENSION RESIDUAL     = {prime_extension_mean:4.5e}" ]

                    if interface_extension_info is not None:
                        print_list += [ f"  INTERFACE EXTENSION STEPS        = {interface_quantity_extension_steps:11d}" ]
                        print_list += [ f"  INTERFACE EXTENSION RESIDUAL     = {interface_extension_mean:4.5e}" ]

                    if solid_coupling.thermal == "TWO-WAY":
                        raise NotImplementedError

                    if reinitialization_info is not None:
                        print_list += ["LEVELSET"]
                        print_list += [ f"  REINITIALIZATION STEPS           = {reinitialization_steps:11d}" ]
                        print_list += [ f"  REINITIALIZATION RESIDUAL        = {reinitialzation_max:4.5e}" ]

                    self.log_list(print_list)

            self.hline()

    def hline(self) -> None:
        """Inserts a dashed horizontal line in log.
        """
        hline_str = "*" + "-" * 78 + "*"
        self.logger.info(hline_str)

    def nline(self) -> None:
        """Inserts a line break in log.
        """
        self.logger.info(f"{'*':<40}{'*':>40}")

    def log_line(self, line: str) -> None:
        self.logger.info(f"*{line:<78}*")

    def log(self, line: str) -> None:
        self.logger.info(line)
