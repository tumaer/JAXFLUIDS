import os
import json
from typing import Dict, Tuple, Union
from functools import partial

import h5py
import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.buffers import ForcingParameters, SimulationBuffers, \
    MaterialFieldBuffers, LevelsetFieldBuffers, TimeControlVariables
from jaxfluids.data_types.information import TurbulentStatisticsInformation, \
    ChannelStatisticsCumulative, HITStatisticsCumulative
from jaxfluids.data_types.numerical_setup.turbulence_statistics import TurbulenceStatisticsSetup

class StatisticsWriter:
    """Writes turbulent statistics to h5 file.

    # TODO turbulent case should be member
    """
    def __init__(
        self,
        turbulence_statistics_setup: TurbulenceStatisticsSetup,
        domain_information: DomainInformation,
        unit_handler: UnitHandler,
        is_double: bool,
        ) -> None:
        
        # MEMBER
        self.turbulence_statistics_setup = turbulence_statistics_setup
        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.is_double = is_double
        self.save_path_statistics = None

        self.turbulence_case = turbulence_statistics_setup.turbulence_case

        self.num_digits_output = 10

    def set_save_path_statistics(self, save_path_statistics: str) -> None:
        self.save_path_statistics = save_path_statistics

    def write_statistics(
            self,
            turbulent_statistics: TurbulentStatisticsInformation,
            time_control_variables: TimeControlVariables,
            ) -> None:
        """Saves turbulent statistics to h5 file.

        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        """

        physical_simulation_time = time_control_variables.physical_simulation_time
        cumulative_turbulent_statistics = turbulent_statistics.cumulative

        dtype = "f8" if self.is_double else "f4"

        current_time = self.unit_handler.dimensionalize(physical_simulation_time, "time")
        filename = f"statistics_{current_time:.{self.num_digits_output}f}.h5"

        with h5py.File(os.path.join(self.save_path_statistics, filename), "w") as h5file:

            # CURRENT TIME
            h5file.create_dataset(name="time", data=current_time, dtype=dtype)

            # DOMAIN INFORMATION
            cell_centers = self.domain_information.get_local_cell_centers()
            cell_faces = self.domain_information.get_local_cell_faces()
            cell_sizes = self.domain_information.get_local_cell_sizes()

            cell_centers_h5 = []
            cell_faces_h5 = []
            cell_sizes_h5 = []
            for i in range(3):
                xi = self.unit_handler.dimensionalize(cell_centers[i], "length")
                fxi = self.unit_handler.dimensionalize(cell_faces[i], "length")
                dxi = self.unit_handler.dimensionalize(cell_sizes[i], "length")
                cell_centers_h5.append(xi)
                cell_faces_h5.append(fxi)
                cell_sizes_h5.append(dxi)

            dim = self.domain_information.dim
            split_factors = self.domain_information.split_factors
            is_parallel = self.domain_information.is_parallel
            is_multihost = self.domain_information.is_multihost
            process_id = self.domain_information.process_id
            local_device_count = self.domain_information.local_device_count
            global_device_count = self.domain_information.global_device_count

            # META DATA
            grp_meta = h5file.create_group(name="metadata")

            h5file.create_dataset(name="metadata/is_parallel", data=is_parallel)
            h5file.create_dataset(name="metadata/is_multihost", data=is_multihost)
            h5file.create_dataset(name="metadata/process_id", data=process_id)
            h5file.create_dataset(name="metadata/local_device_count", data=local_device_count)
            h5file.create_dataset(name="metadata/global_device_count", data=global_device_count)

            grp_meta.create_dataset(name="is_double_precision", data=self.is_double)

            # DOMAIN
            grp_d = h5file.create_group(name="domain")
            grp_d.create_dataset(name="dim", data=dim)
            grp_d.create_dataset(name="gridX", data=cell_centers[0], dtype=dtype)
            grp_d.create_dataset(name="gridY", data=cell_centers[1], dtype=dtype)
            grp_d.create_dataset(name="gridZ", data=cell_centers[2], dtype=dtype)
            grp_d.create_dataset(name="split_factors", data=jnp.array(split_factors), dtype="i8")

            # STATISTICS
            grp_meta.create_dataset(name="turbulence_case", data=self.turbulence_case)
            if self.turbulence_case == "CHANNEL":
                self._write_channel_statistics(h5file,
                                               cumulative_turbulent_statistics.channel_statistics,
                                               dtype)
        
            if self.turbulence_case == "HIT":
                self._write_hit_statistsics(h5file,
                                            cumulative_turbulent_statistics.hit_statistics,
                                            dtype)

    def _write_hit_statistsics(
            self,
            h5file: h5py.File,
            data: HITStatisticsCumulative,
            dtype: str
            ) -> None:
        # SAMPLES
        grp_s = h5file.create_group(name="samples")
        grp_s.create_dataset(name="number_sample_steps", data=data.number_sample_steps, dtype=dtype)
        grp_s.create_dataset(name="number_sample_points", data=data.number_sample_points, dtype=dtype)
        
        one_sp = 1.0 / data.number_sample_points
        
        # MEANS
        grp_m = h5file.create_group(name="means")
        grp_m.create_dataset(name="density", data=one_sp * data.density_T, dtype=dtype)
        grp_m.create_dataset(name="pressure", data=one_sp * data.pressure_T, dtype=dtype)
        grp_m.create_dataset(name="temperature", data=one_sp * data.temperature_T, dtype=dtype)
        grp_m.create_dataset(name="speed_of_sound", data=one_sp * data.c_T, dtype=dtype)

        # FLUCTUATIONS
        grp_f = h5file.create_group(name="fluctuations")
        grp_f.create_dataset(name="rhop_rhop", data=one_sp * data.rhop_rhop_S, dtype=dtype)
        grp_f.create_dataset(name="pp_pp", data=one_sp * data.pp_pp_S, dtype=dtype)
        grp_f.create_dataset(name="Tp_Tp", data=one_sp * data.Tp_Tp_S, dtype=dtype)
        grp_f.create_dataset(name="up_up", data=one_sp * data.up_up_S, dtype=dtype)
        grp_f.create_dataset(name="vp_vp", data=one_sp * data.vp_vp_S, dtype=dtype)
        grp_f.create_dataset(name="wp_wp", data=one_sp * data.wp_wp_S, dtype=dtype)
        grp_f.create_dataset(name="Mp_Mp", data=one_sp * data.machp_machp_S, dtype=dtype)

    def _write_channel_statistics(
            self, 
            h5file: h5py.File,
            data: ChannelStatisticsCumulative,
            dtype: str
            ) -> None:
        # SAMPLES
        grp_s = h5file.create_group(name="samples")
        grp_s.create_dataset(name="number_sample_steps", data=data.number_sample_steps, dtype=dtype)
        grp_s.create_dataset(name="number_sample_points", data=data.number_sample_points, dtype=dtype)
        
        one_sp = 1.0 / data.number_sample_points

        # MEANS
        grp_m = h5file.create_group(name="means")
        grp_m.create_dataset(name="velocityX", data=one_sp * jnp.squeeze(data.U_T), dtype=dtype)
        grp_m.create_dataset(name="velocityY", data=one_sp * jnp.squeeze(data.V_T), dtype=dtype)
        grp_m.create_dataset(name="velocityZ", data=one_sp * jnp.squeeze(data.W_T), dtype=dtype)
        grp_m.create_dataset(name="density", data=one_sp * jnp.squeeze(data.density_T), dtype=dtype)
        grp_m.create_dataset(name="pressure", data=one_sp * jnp.squeeze(data.pressure_T), dtype=dtype)
        grp_m.create_dataset(name="temperature", data=one_sp * jnp.squeeze(data.T_T), dtype=dtype)
        grp_m.create_dataset(name="mach_number", data=one_sp * jnp.squeeze(data.mach_T), dtype=dtype)
        grp_m.create_dataset(name="speed_of_sound", data=one_sp * jnp.squeeze(data.c_T), dtype=dtype)
        
        # FLUCTUATIONS
        grp_f = h5file.create_group(name="fluctuations")
        grp_f.create_dataset(name="Mp_Mp", data=one_sp * jnp.squeeze(data.machp_machp_S), dtype=dtype)
        grp_f.create_dataset(name="pp_pp", data=one_sp * jnp.squeeze(data.pp_pp_S), dtype=dtype)
        grp_f.create_dataset(name="rhop_rhop", data=one_sp * jnp.squeeze(data.rhop_rhop_S), dtype=dtype)
        grp_f.create_dataset(name="up_up", data=one_sp * jnp.squeeze(data.up_up_S), dtype=dtype)
        grp_f.create_dataset(name="vp_vp", data=one_sp * jnp.squeeze(data.vp_vp_S), dtype=dtype)
        grp_f.create_dataset(name="wp_wp", data=one_sp * jnp.squeeze(data.wp_wp_S), dtype=dtype)
        grp_f.create_dataset(name="up_vp", data=one_sp * jnp.squeeze(data.up_vp_S), dtype=dtype)
        grp_f.create_dataset(name="up_wp", data=one_sp * jnp.squeeze(data.up_wp_S), dtype=dtype)
        grp_f.create_dataset(name="vp_wp", data=one_sp * jnp.squeeze(data.vp_wp_S), dtype=dtype)
        grp_f.create_dataset(name="Tp_Tp", data=one_sp * jnp.squeeze(data.Tp_Tp_S), dtype=dtype)
        grp_f.create_dataset(name="vp_Tp", data=one_sp * jnp.squeeze(data.vp_Tp_S), dtype=dtype)
    
    def _write_duct_statistics(self):
        raise NotImplementedError
    
    def _write_boundary_layer_statistics(self):
        raise NotImplementedError
    
