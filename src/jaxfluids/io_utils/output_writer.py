import os
import json
from typing import Dict, Tuple, Union
from functools import partial
import math

import h5py
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.io_utils.hdf5_writer import HDF5Writer
from jaxfluids.io_utils.statistics_writer import StatisticsWriter
from jaxfluids.io_utils.xdmf_writer import XDMFWriter
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.input.input_manager import InputManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.equation_information import EquationInformation
from jaxfluids.data_types.buffers import ForcingParameters, SimulationBuffers, TimeControlVariables
from jaxfluids.data_types.information import WallClockTimes, TurbulentStatisticsInformation
from jaxfluids.parallel.helper_functions import synchronize_jf

class OutputWriter:
    """Output writer for JAX-FLUIDS. The OutputWriter class can write h5 and xdmf 
    files. h5 and xdmf files can be visualized in paraview. Xdmf output is activated
    via the is_xdmf_file flag under the output keyword in the numerical setup.

    If the xdmf option is activated, a single xdmf file is written for each h5 file.
    Additionally, at the end of the simulation, an time series xdmf file is written
    which summarizes the whole simulation. This enables loading the entire timeseries
    into Paraview.
    """
    def __init__(
            self,
            input_manager: InputManager,
            unit_handler: UnitHandler,
            domain_information: DomainInformation,
            equation_information: EquationInformation,
            material_manager: MaterialManager,
            levelset_handler: LevelsetHandler
        ) -> None:

        numerical_setup = input_manager.numerical_setup
        case_setup = input_manager.case_setup
        
        self.is_active = numerical_setup.output.is_active
        self.is_active_turbulent_statistics = numerical_setup.turbulence_statistics.is_active

        general_setup = case_setup.general_setup
        self.case_name = general_setup.case_name

        save_start = general_setup.save_start
        self.save_dt = general_setup.save_dt
        self.next_output_time = save_start
        # If save_start = 0.0, then add self.save_dt,
        # since output at t=0.0 is enforced anyway.
        if save_start == 0.0:
            self.next_output_time += self.save_dt
        self.save_step = general_setup.save_step
        self.is_save_step = self.save_step > 0
        self.save_timestamps = general_setup.save_timestamps
        self.is_save_timestamps = isinstance(self.save_timestamps, np.ndarray)
        if self.is_save_timestamps:
            self.number_output_timestamps = len(self.save_timestamps)
            self.next_output_timestamp_index = 0

        self.save_path = general_setup.save_path
        self.save_path_case, self.save_path_domain, \
            self.save_path_statistics = None, None, None
        # self.save_path_case, self.save_path_domain, \
        # self.save_path_statistics = self.get_folder_name()

        self.case_setup_dict = input_manager.case_setup_dict
        self.numerical_setup_dict = input_manager.numerical_setup_dict

        self.is_xdmf = numerical_setup.output.is_xdmf
        self.eps_time = 1e-12 if numerical_setup.precision.is_double_precision_compute \
            else 1e-6
        self.is_parallel_filesystem = numerical_setup.output.is_parallel_filesystem
  
        is_double = numerical_setup.precision.is_double_precision_output
        quantities_setup = case_setup.output_quantities_setup

        equation_type = equation_information.equation_type
        if equation_type == "TWO-PHASE-LS":
            nh_offset = domain_information.nh_offset
        else:
            nh_offset = 0
        derivative_stencil = numerical_setup.output.derivative_stencil
        derivative_stencil = derivative_stencil(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes,
            is_mesh_stretching=domain_information.is_mesh_stretching,
            cell_sizes=domain_information.get_global_cell_sizes_halos(),
            offset=nh_offset)
        self.hdf5_writer = HDF5Writer(
            domain_information=domain_information,
            unit_handler=unit_handler,
            material_manager=material_manager,
            levelset_handler=levelset_handler,
            derivative_stencil=derivative_stencil,
            quantities_setup=quantities_setup,
            is_double=is_double,
            output_setup=numerical_setup.output)

        if self.is_xdmf:
            self.xdmf_writer = XDMFWriter(
                domain_information=domain_information,
                unit_handler=unit_handler,
                equation_information=equation_information,
                quantities_setup=quantities_setup,
                is_double=is_double,)
        
        if self.is_active_turbulent_statistics:
            turbulence_statistics_setup = numerical_setup.turbulence_statistics
            self.next_output_time_statistics = turbulence_statistics_setup.start_sampling
            self.save_dt_statistics = turbulence_statistics_setup.save_dt
            self.statistics_writer = StatisticsWriter(
                turbulence_statistics_setup=turbulence_statistics_setup,
                domain_information=domain_information,
                unit_handler=unit_handler,
                is_double=is_double,)
            
        self.is_multihost = domain_information.is_multihost
        self.is_parallel = domain_information.is_parallel
        self.process_id = domain_information.process_id
        self.global_device_count = domain_information.global_device_count
        self.local_device_count = domain_information.local_device_count

    def configure_output_writer(self) -> Tuple[str, str, str]:
        self.save_path_case, self.save_path_domain, \
        self.save_path_statistics = self.get_folder_name()

        # SYNCHRONIZING
        synchronize_jf(self.is_multihost)
        
        if self.is_multihost and self.is_parallel_filesystem:
            if self.process_id == 0:
                self.create_folder()

            # SYNCHRONIZING
            synchronize_jf(self.is_multihost)

        else:
            self.create_folder()
        
        self.hdf5_writer.set_save_path_domain(self.save_path_domain)
        if self.is_xdmf:
            self.xdmf_writer.set_save_path_domain(self.save_path_domain)
        if self.is_active_turbulent_statistics:
            self.statistics_writer.set_save_path_statistics(self.save_path_statistics)
        
        return self.save_path_case, self.save_path_domain, \
            self.save_path_statistics

    def create_folder(self) -> None:
        """Sets up a folder for the simulation. Dumps the numerical setup and
        case setup into the simulation folder. Creates an output folder 'domain'
        within the simulation folder into which simulation output is saved. If
        turbulence statistics are active, creates a folder 'statistics' within
        the simulatiion folder into which statistics output is saved.

        simulation_folder
        ---- Numerical setup
        ---- Case setup
        ---- domain
        ---- statistics (only if turbulence statistics is active)
        """
        os.mkdir(self.save_path_case)
        os.mkdir(self.save_path_domain)
        if self.is_active_turbulent_statistics:
            os.mkdir(self.save_path_statistics)

        with open(os.path.join(self.save_path_case, self.case_name + ".json"), "w") as json_file:
            json.dump(self.case_setup_dict, json_file, ensure_ascii=False, indent=4)
        with open(os.path.join(self.save_path_case, "numerical_setup.json"), "w") as json_file:
            json.dump(self.numerical_setup_dict, json_file, ensure_ascii=False, indent=4)
        
    def get_folder_name(self) -> Tuple[str, str]:
        """Returns a name for the simulation folder based on the case name.

        :return: Path to the simulation folder and path to domain folder within
            simulation folder.
        :rtype: Tuple[str, str]
        """

        case_name_folder = self.case_name

        if not os.path.exists(self.save_path):
            if self.is_multihost and self.is_parallel_filesystem:
                if self.process_id == 0:
                    os.makedirs(self.save_path, exist_ok=True)

                # SYNCHRONIZING
                synchronize_jf(self.is_multihost)

            else:
                os.makedirs(self.save_path, exist_ok=True)

        create_directory = True
        i = 1
        while create_directory:
            if os.path.exists(os.path.join(self.save_path, case_name_folder)):
                case_name_folder = self.case_name + "-%d" % i
                i += 1
            else:
                save_path_case = os.path.join(self.save_path, case_name_folder)
                save_path_domain = os.path.join(save_path_case, "domain")
                save_path_statistics = os.path.join(save_path_case, "statistics")
                create_directory = False

        return save_path_case, save_path_domain, save_path_statistics

    def set_simulation_start_time(
            self,
            start_time: float
        ) -> None:
        if self.next_output_time:
            self.next_output_time += start_time

        if self.is_save_timestamps:
            self.next_output_timestamp_index = np.searchsorted(self.save_timestamps, start_time)

    def write_output(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            wall_clock_times: WallClockTimes,
            forcing_parameters: ForcingParameters = None,
            force_output: bool = False,
            simulation_finish: bool = False
            ) -> None:
        """Writes h5 and (optional) xdmf output.
        
        1) Calls hdf5_writer.write_file()
        2) Calls xdmf_writer.write_file()

        :param buffer_dictionary: Dictionary with flow field buffers
        :type buffer_dictionary: Dict[str, Dict[str, Union[Array, float]]]
        :param force_output: Flag which forces an output.
        :type force_output: bool
        :param simulation_finish: Flag that indicates the simulation finish -> 
            then timeseries xdmf is written, defaults to False
        :type simulation_finish: bool, optional
        """
        if self.is_active:
            physical_simulation_time = time_control_variables.physical_simulation_time
            simulation_step = time_control_variables.simulation_step
            
            if force_output:
                self._write_output(simulation_buffers, time_control_variables,
                                           wall_clock_times, forcing_parameters,
                                           is_write_step=self.is_save_step)

            else:
                is_write_output = False
                if self.save_dt:
                    diff = physical_simulation_time - self.next_output_time
                    if diff >= -self.eps_time:
                        self.next_output_time += self.save_dt
                        is_write_output = True

                if self.is_save_timestamps and self.next_output_timestamp_index < self.number_output_timestamps:
                    next_output_timestamp = self.save_timestamps[self.next_output_timestamp_index]
                    diff = physical_simulation_time - next_output_timestamp
                    if diff >= -self.eps_time:
                        self.next_output_timestamp_index += 1
                        is_write_output = True

                if is_write_output:
                    self._write_output(simulation_buffers, time_control_variables,
                                       wall_clock_times, forcing_parameters)

                if self.save_step:
                    if simulation_step % self.save_step == 0:
                        self._write_output(simulation_buffers, time_control_variables,
                                           wall_clock_times, forcing_parameters, True)

            if simulation_finish and self.is_xdmf:
                self.xdmf_writer.write_timeseries()

    def _write_output(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            wall_clock_times: WallClockTimes,
            forcing_parameters: ForcingParameters = None,
            is_write_step: bool = False
            ) -> None:
        self.hdf5_writer.write_file(
            simulation_buffers, time_control_variables,
            wall_clock_times, forcing_parameters, is_write_step)
        if self.is_xdmf:
            self.xdmf_writer.write_file(time_control_variables,
                                        is_write_step)

    def write_turbulent_statistics(
            self, 
            turbulent_statistics: TurbulentStatisticsInformation,
            time_control_variables: TimeControlVariables,
            force_output: bool = False,
            ) -> None:
        
        if self.is_active_turbulent_statistics:
            physical_simulation_time = time_control_variables.physical_simulation_time
            simulation_step = time_control_variables.simulation_step
            
            if self.save_dt_statistics:
                diff = physical_simulation_time - self.next_output_time_statistics
                if diff >= -self.eps_time or force_output:
                    self.statistics_writer.write_statistics(
                        turbulent_statistics, time_control_variables,)
                    if not force_output:
                        self.next_output_time_statistics += self.save_dt_statistics
