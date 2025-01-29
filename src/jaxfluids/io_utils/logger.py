from datetime import datetime
import logging
from platform import python_version
import os
import textwrap
from typing import Dict, List

import git

from jax import version as jax_version
from jaxlib import version as jaxlib_version
import jax

import jaxfluids
from jaxfluids.data_types.buffers import TimeControlVariables
from jaxfluids.data_types.information import StepInformation, WallClockTimes, \
    TurbulentStatisticsInformation, ChannelStatisticsLogging, HITStatisticsLogging
from jaxfluids.data_types.numerical_setup import NumericalSetup

class Logger:
    """Logger for the JAX-FLUIDS solver.
    Logs information during the simulation to file and/or screen."""

    def __init__(
            self,
            numerical_setup: NumericalSetup,
            logger_name: str = "",
            jax_backend: str = None,
            is_multihost: bool = False
            ) -> None:

        self.logger_name = logger_name
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
            case_setup_dict: Dict
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
                elif isinstance(item, list):
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
        
    def log_turbulent_stats_at_start(self, turb_stats_dict: Dict) -> None:
        """Logs the turbulent statistics of the initial turbulent flow field.

        :param turb_stats_dict: Dictionary which information on turbulent statistics.
        :type turb_stats_dict: Dict
        """
        self.nline()
        self.logger.info(f"*{'TURBULENT STATISTICS':^78}*")
        self.nline()
        for key, item in turb_stats_dict.items():
            self.logger.info(f"*    {key:<74}*")
            for subkey, subitem in item.items():
                self.logger.info(f"*        {subkey:<20} = {subitem:<47.3e}*")       
            self.nline()
        self.nline()
        self.hline()

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

    def log_end_time_step(
            self,
            time_control_variables: TimeControlVariables,
            step_information: StepInformation,
            wall_clock_times: WallClockTimes,
            time_reference: float,
            turbulent_statistics: TurbulentStatisticsInformation = None
            ) -> None:
        """Logs information at the end of an integration step.

        :param info_list: List of strings to be printed at the end
            of an integration step.
        :type info_list: List
        """

        if time_control_variables.simulation_step % self.logging_frequency == 0:

            forcing_infos = step_information.forcing_info
            levelset_residuals_info_list = step_information.levelset_residuals_info_list
            positivity_state_info_list = step_information.positivity_state_info_list
            
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

                if forcing_infos.temperature is not None:
                    temperature_error = forcing_infos.temperature.current_error
                    self.log_list([
                        "TEMPERATURE CONTROL",
                        f"ERROR  = {temperature_error:4.5e}",
                    ])

            # TURBULENT STATS
            if turbulent_statistics is not None:
                statistics_logging = turbulent_statistics.logging

                if statistics_logging.hit_statistics is not None:
                    hit_statistics: HITStatisticsLogging = statistics_logging.hit_statistics
                    density_mean = hit_statistics.rho_bulk
                    pressure_mean = hit_statistics.pressure_bulk
                    temperature_mean = hit_statistics.temperature_bulk
                    density_rms = hit_statistics.rho_rms
                    pressure_rms = hit_statistics.pressure_rms
                    temperature_rms = hit_statistics.temperature_rms
                    mach_turbulent = hit_statistics.mach_rms
                    velocity_rms = hit_statistics.u_rms
                    self.log_list([
                        "TURBULENT STATISTICS - HIT",
                        f"MEAN DENSITY               = {density_mean:4.4e}",
                        f"MEAN PRESSURE              = {pressure_mean:4.4e}",
                        f"MEAN TEMPERATURE           = {temperature_mean:4.4e}",                        
                        f"TURBULENT MACH RMS         = {mach_turbulent:4.4e}",
                        f"VELOCITY RMS               = {velocity_rms:4.4e}",
                        f"DENSITY RMS                = {density_rms:4.4e}",
                        f"PRESSURE RMS               = {pressure_rms:4.4e}",
                        f"TEMPERATURE RMS            = {temperature_rms:4.4e}",
                    ])
                
                elif statistics_logging.channel_statistics is not None:
                    channel_statistics: ChannelStatisticsLogging = statistics_logging.channel_statistics
                    rho_bulk = channel_statistics.rho_bulk
                    temperature_bulk = channel_statistics.temperature_bulk
                    u_bulk = channel_statistics.u_bulk
                    mach_bulk = channel_statistics.mach_bulk
                    reynolds_tau = channel_statistics.reynolds_tau
                    reynolds_bulk = channel_statistics.reynolds_bulk
                    delta_x_plus = channel_statistics.delta_x_plus
                    delta_y_plus_min = channel_statistics.delta_y_plus_min
                    delta_y_plus_max = channel_statistics.delta_y_plus_max
                    delta_z_plus = channel_statistics.delta_z_plus
                    self.log_list([
                        "TURBULENT STATISTICS - CHANNEL",
                        f"DENSITY BULK               = {rho_bulk:4.4e}",
                        f"TEMPERATURE BULK           = {temperature_bulk:4.4e}",
                        f"VELOCITY BULK              = {u_bulk:4.4e}",
                        f"MACH NUBMER BULK           = {mach_bulk:4.4e}",
                        f"REYNOLDS NUMBER TAU        = {reynolds_tau:4.4e}",
                        f"REYNOLDS NUMBER BULK       = {reynolds_bulk:4.4e}",
                        f"DELTA X+                   = {delta_x_plus:4.4e}",
                        f"DELTA Y+ MIN               = {delta_y_plus_min:4.4e}",
                        f"DELTA Y+ MAX               = {delta_y_plus_max:4.4e}",
                        f"DELTA Z+                   = {delta_z_plus:4.4e}",
                    ])

            # REACTION KINETICS
            # TODO combustion

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
                    if positivity_setup.is_interpolation_limiter:
                        print_list += [ f"COUNT INTERPOLATION LIMITER        = {positivity_state.positivity_count_interpolation:d}" ]
                    if positivity_setup.is_thinc_interpolation_limiter:
                        print_list += [ f"COUNT INTERPOLATION LIMITER THINC  = {positivity_state.positivity_count_thinc:d}" ]
                    if positivity_setup.flux_limiter:
                        print_list += [ f"COUNT FLUX LIMITER                 = {positivity_state.positivity_count_flux:d}" ]
                    if diffuse_interface_model:
                        if positivity_setup.is_volume_fraction_limiter:
                            print_list += [ f"COUNT VOLUME FRACTION LIMITER      = {positivity_state.vf_correction_count:d}" ]
                        if positivity_setup.is_acdi_flux_limiter:
                            print_list += [ f"COUNT CDI/ACDI FLUX                = {positivity_state.count_acdi:d}"]
                            print_list += [ f"COUNT CDI/ACDI FLUX LIMITER        = {positivity_state.positivity_count_acdi:d}"]
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

            log_any = any((is_levelset_residuals, is_positivity))
            
            if log_any:
                for stage, (levelset_residuals, positivity_state) in \
                enumerate(zip(levelset_residuals_info_list, positivity_state_info_list)):
                    prime_extension_mean = levelset_residuals.prime_extension_residual_mean
                    prime_extension_max = levelset_residuals.prime_extension_residual_max
                    prime_extension_steps = levelset_residuals.prime_extension_steps
                    interface_extension_mean = levelset_residuals.interface_quantity_extension_residual_mean
                    interface_extension_max = levelset_residuals.interface_quantity_extension_residual_max
                    interface_quantity_extension_steps = levelset_residuals.interface_quantity_extension_steps
                    reinitialzation_mean = levelset_residuals.reinitialization_residual_mean
                    reinitialzation_max = levelset_residuals.reinitialization_residual_max
                    reinitialization_steps = levelset_residuals.reinitialization_steps

                    mixing_invalid_cells_count = positivity_state.levelset.mixing_invalid_cell_count
                    extension_invalid_cell_count = positivity_state.levelset.extension_invalid_cell_count

                    if is_only_last_stage:
                        print_list = [f"LEVELSET STATE"]
                    else:
                        print_list = [f"LEVELSET STATE STAGE {stage:d}"]
                    if is_positivity:
                        if levelset_model == "FLUID-FLUID":
                            print_list += [ f"MIXING INVALID CELLS               =  {mixing_invalid_cells_count[0]:10d} / {mixing_invalid_cells_count[1]:11d}" ]
                            print_list += [ f"EXTENSION INVALID CELLS            =  {extension_invalid_cell_count[0]:10d} / {extension_invalid_cell_count[1]:11d}" ]
                        else:
                            print_list += [ f"MIXING INVALID CELLS               =  {mixing_invalid_cells_count:11d}" ]
                            print_list += [ f"EXTENSION INVALID CELLS            =  {extension_invalid_cell_count:11d}" ]
                    if is_levelset_residuals:
                        if not isinstance(prime_extension_steps, type(None)):
                            print_list += [ f"STEPS EXTENSION PRIMES             = {prime_extension_steps:25d}" ]
                        if not isinstance(interface_quantity_extension_steps, type(None)) and levelset_model == "FLUID-FLUID":
                            print_list += [ f"STEPS EXTENSION INTERFACE          = {interface_quantity_extension_steps:25d}" ]
                        if not isinstance(reinitialization_steps, type(None)):
                            print_list += [ f"STEPS REINITIALIZATION             = {reinitialization_steps:25d}" ]
                        print_list += [ f"RESIDUAL EXTENSION PRIMES          = {prime_extension_mean:4.5e} / {prime_extension_max:4.5e}" ]
                        if levelset_model == "FLUID-FLUID":
                            print_list += [ f"RESIDUAL EXTENSION INTERFACE       = {interface_extension_mean:4.5e} / {interface_extension_max:4.5e}" ]
                        print_list += [ f"RESIDUAL REINITIALIZATION          = {reinitialzation_mean:4.5e} / {reinitialzation_max:4.5e}" ]

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
