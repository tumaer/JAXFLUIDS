import os
import json
from typing import Dict, Tuple, Union
from functools import partial

import h5py
import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.buffers import ForcingParameters, SimulationBuffers, \
    MaterialFieldBuffers, LevelsetFieldBuffers, TimeControlVariables
from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.data_types.statistics import TurbulenceStatisticsInformation

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

        self.turbulence_case = turbulence_statistics_setup.case

        self.num_digits_output = 10

    def set_save_path_statistics(self, save_path_statistics: str) -> None:
        self.save_path_statistics = save_path_statistics

    def write_statistics(
            self,
            turbulence_statistics: TurbulenceStatisticsInformation,
            time_control_variables: TimeControlVariables,
            ) -> None:
        """Saves turbulence statistics to h5 file.

        :param turbulence_statistics: _description_
        :type turbulence_statistics: TurbulenceStatisticsInformation
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :raises NotImplementedError: _description_
        """

        physical_simulation_time = time_control_variables.physical_simulation_time
        turbulence_statistics_cumulative = turbulence_statistics.cumulative

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
                case_statistics = turbulence_statistics_cumulative.channel
            elif self.turbulence_case == "HIT":
                case_statistics = turbulence_statistics_cumulative.hit                
            elif self.turbulence_case == "BOUNDARY_LAYER":
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            self._write_turbulence_statistics(h5file, case_statistics, dtype)

    def _write_turbulence_statistics(
            self, 
            h5file: h5py.File,
            data: Dict,
            dtype: str
            ) -> None:
        
        for key1 in data:
            grp = h5file.create_group(name=key1)
            sub_data = data[key1]
            for key2 in sub_data:
                grp.create_dataset(name=key2, data=sub_data[key2], dtype=dtype)

    