from typing import Union, Callable, Dict, List, Tuple
import warnings
import os

import jax
import jax.numpy as jnp
import json
import numpy as np
import h5py

from jaxfluids.input.input_manager import InputManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.initialization.helper_functions import (
    create_field_buffer, get_load_function, interpolate, get_h5file_list,
    parse_restart_files)
from jaxfluids.data_types.buffers import (
    LevelsetFieldBuffers, SolidFieldBuffers)
from jaxfluids.equation_information import EquationInformation
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionSolids
from jaxfluids.data_types.case_setup.restart import RestartSetup
from jaxfluids.data_types.case_setup.solid_properties import SolidPropertiesSetup
from jaxfluids.data_types.information import LevelsetProcedureInformation, LevelsetPositivityInformation

Array = jax.Array

class SolidsInitializer:
    """The LevelsetInitializer implements functionality
    to create initial buffers for the levelset 
    and all related fields, e.g., apertures,
    volume fraction, solid interface velocity
    """
    def __init__(
            self,
            numerical_setup: NumericalSetup,
            domain_information: DomainInformation,
            unit_handler: UnitHandler,
            equation_information: EquationInformation,
            halo_manager: HaloManager,
            extender: IterativeExtender,
            initial_condition_solids: InitialConditionSolids,
            restart_setup: RestartSetup,
            solid_properties_setup: SolidPropertiesSetup,
            ) -> None:

        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.halo_manager = halo_manager
        self.equation_information = equation_information
        self.numerical_setup = numerical_setup
        self.restart_setup = restart_setup
        self.initial_condition_solids = initial_condition_solids
        self.is_double_precision = numerical_setup.precision.is_double_precision_compute
        self.extender = extender

        self.solid_properties_manager = SolidPropertiesManager(
            domain_information = domain_information,
            solid_properties_setup = solid_properties_setup,
            )

    def initialize(
            self,
            levelset_fields: LevelsetFieldBuffers,
            user_solid_velocity_init: Union[np.ndarray, Array] = None,
            user_solid_temperature_init: Union[np.ndarray, Array] = None,
            user_restart_file_path: str = None
            ) -> Tuple[SolidFieldBuffers,
                       LevelsetPositivityInformation,
                       LevelsetProcedureInformation]:
        """Initializes the levelset related field buffers. Peforms ghost cell
        treatment on the material fields.

        :param user_solid_velocity_init: _description_, defaults to None
        :type user_solid_velocity_init: Union[np.ndarray, Array], optional
        :param user_solid_temperature_init: _description_, defaults to None
        :type user_solid_temperature_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: Tuple[SolidFieldBuffers]
        """

        is_restart = self.restart_setup.flag
        is_parallel = self.domain_information.is_parallel
        cell_centers = self.domain_information.get_local_cell_centers()

        if is_restart:
            solid_fields = self.from_restart_file(user_restart_file_path)
            
        elif user_solid_velocity_init is not None or user_solid_temperature_init is not None:
            raise NotImplementedError
            self.from_user_specified_buffer()

        else:
            if is_parallel:
                solid_fields = jax.pmap(
                    self.from_solids_initial_condition, axis_name="i")(
                    cell_centers)
            else:
                solid_fields = self.from_solids_initial_condition(
                    cell_centers)

        return solid_fields

    def create_solid_fields(
            self,
            temperature_np: Array,
            velocity_np: Array,
            ) -> SolidFieldBuffers:

        device_number_of_cells = self.domain_information.device_number_of_cells
        nh = self.domain_information.nh_conservatives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        dtype = jnp.float64 if self.is_double_precision else jnp.float32
    
        solid_coupling = self.equation_information.solid_coupling

        if solid_coupling.dynamic == "TWO-WAY":
            buffer = create_field_buffer(nh, device_number_of_cells, dtype, 3)
            velocity = buffer.at[...,nhx,nhy,nhz].set(velocity_np)

        else:
            velocity = None  

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

        else:
            energy = None
            temperature = None

        solid_fields = SolidFieldBuffers(velocity, energy, temperature)

        return solid_fields
    

    def from_restart_file(self, user_restart_file_path: str | None) -> SolidFieldBuffers:

        restart_file_path = self.restart_setup.file_path if user_restart_file_path is None else user_restart_file_path
        restart_time = self.restart_setup.time
        use_restart_time = self.restart_setup.use_time
        is_interpolate = self.restart_setup.is_interpolate

        split_factors = self.domain_information.split_factors
        is_parallel = self.domain_information.is_parallel
        is_multihost = self.domain_information.is_multihost
        local_device_count = self.domain_information.local_device_count
        process_id = self.domain_information.process_id
        is_equal_decomposition_multihost = self.restart_setup.is_equal_decomposition_multihost

        restart_file_path = parse_restart_files(restart_file_path)
        h5file_list = get_h5file_list(restart_file_path, process_id,
                                      is_equal_decomposition_multihost)
        h5file = h5file_list[0]

        if use_restart_time:
            physical_simulation_time = restart_time
        else:
            physical_simulation_time = h5file["time"][()]
        physical_simulation_time = self.unit_handler.non_dimensionalize(physical_simulation_time, "time")

        is_parallel_restart = h5file["metadata"]["is_parallel"][()]
        split_factors_restart = h5file["domain"]["split_factors"][:]

        solid_coupling = self.equation_information.solid_coupling

        # SANITY CHECK
        solids_keys = h5file["metadata"]["available_quantities"]["solids"][:].astype("U")

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

        if solid_coupling.dynamic == "TWO-WAY":
            assert_str = "Solid velocity not in restart file %s." % restart_file_path
            assert "velocity" in solids_keys, assert_str


        load_function = get_load_function(is_parallel, is_parallel_restart,
                                          split_factors, split_factors_restart)

        load_function_inputs = {
            "h5file": h5file_list,
            "split_factors": split_factors,
            "split_factors_restart": split_factors_restart,
            }

        # LOAD DATA FROM H5 INTO NUMPY ARRAYS
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        else:
            temperature = None
            
        if solid_coupling.dynamic == "TWO-WAY":
            velocity = load_function("solids/velocity", **load_function_inputs)
        else:
            velocity = None

        if is_interpolate:
            case_setup_restart = json.load(open(self.restart_setup.case_setup_path))
            numerical_setup_restart = json.load(open(self.restart_setup.numerical_setup_path))
            for axis_name, axis_id in DomainInformation.axis_to_axis_id.items():
                case_setup_restart["domain"]["decomposition"][f"split_{axis_name}"] = split_factors[axis_id]
            input_manager_restart = InputManager(case_setup_restart, numerical_setup_restart)

            local_cell_centers = self.domain_information.get_local_cell_centers()
            local_cell_centers = tuple([self.unit_handler.dimensionalize(xi, "length") for xi in local_cell_centers])
            dtype = np.float64 if self.is_double_precision else np.float32

        if is_parallel:
            if is_multihost and not is_equal_decomposition_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError
                if solid_coupling.dynamic == "TWO-WAY":
                    velocity = velocity[s_]

            if is_interpolate:
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError

                if solid_coupling.dynamic == "TWO-WAY":
                    raise NotImplementedError

            solid_fields = jax.pmap(self.create_solid_fields, axis_name="i")(
                temperature, velocity)
        else:
            if is_interpolate:
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError

                if solid_coupling.dynamic == "TWO-WAY":
                    raise NotImplementedError

            solid_fields = self.create_solid_fields(
                temperature, velocity)
        
        return solid_fields
            

    def from_user_specified_buffer(self) -> SolidFieldBuffers:
        raise NotImplementedError # TODO

    def from_solids_initial_condition(
            self,
            cell_centers: List,
            ) -> SolidFieldBuffers:
        """Computes the levelset
        buffers from the primitive initial
        condition provided in the case setup .json
        file.


        :param cell_centers: _description_
        :type cell_centers: List
        :return: _description_
        :rtype: Tuple
        """

        mesh_grid = self.domain_information.compute_device_mesh_grid()

        solid_coupling = self.equation_information.solid_coupling

        if solid_coupling.dynamic == "TWO-WAY":
            initial_velocity = self.initial_condition_solids.velocity
            velocity_list = []
            for field in initial_velocity._fields:
                velocity_callable = getattr(initial_velocity, field)
                velocity = velocity_callable(*mesh_grid)
                velocity_list.append(velocity)
            velocity = jnp.stack(velocity_list)
        else:
            velocity = None

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

        else:
            temperature = None

        solid_fields = self.create_solid_fields(temperature, velocity)

        return solid_fields