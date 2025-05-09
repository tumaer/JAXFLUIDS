from typing import Union, Callable, Dict, List, Tuple
import warnings
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import h5py

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.levelset.creation.levelset_creator import LevelsetCreator
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.data_types.information import LevelsetPositivityInformation, LevelsetResidualInformation
from jaxfluids.initialization.helper_functions import create_field_buffer, get_load_function, get_h5file_list, parse_restart_files
from jaxfluids.initialization.cell_index_marker import compute_solid_cell_indices
from jaxfluids.levelset.geometry.mask_functions import compute_narrowband_mask, compute_fluid_masks
from jaxfluids.domain.helper_functions import split_buffer_np
from jaxfluids.data_types.buffers import (
    LevelsetFieldBuffers, MaterialFieldBuffers, TimeControlVariables,
    SolidFieldBuffers, LevelsetSolidCellIndices)
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionLevelset
from jaxfluids.data_types.case_setup.restart import RestartSetup
from jaxfluids.levelset.residual_computer import ResidualComputer
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.data_types.information import LevelsetResidualInformation

Array = jax.Array

class LevelsetInitializer:
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
            equation_manager: EquationManager,
            material_manager: MaterialManager,
            halo_manager: HaloManager,
            initial_condition_levelset: InitialConditionLevelset,
            restart_setup: RestartSetup,
            solid_properties_manager: SolidPropertiesManager
            ) -> None:

        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager
        self.solid_properties_manager = solid_properties_manager

        self.numerical_setup = numerical_setup
        self.restart_setup = restart_setup
        self.initial_condition_levelset = initial_condition_levelset
        self.is_double_precision = numerical_setup.precision.is_double_precision_compute

        levelset_setup = numerical_setup.levelset

        self.geometry_calculator = GeometryCalculator(
            domain_information = domain_information,
            geometry_setup = numerical_setup.levelset.geometry,
            halo_cells_geometry = levelset_setup.halo_cells,
            narrowband_computation = levelset_setup.narrowband.computation_width
            )

        self.extender_primes = IterativeExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            is_jaxwhileloop = False,
            residual_threshold = 1e-16,
            extension_quantity = "primitives")
        
        self.extender_interface = IterativeExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            is_jaxwhileloop = False,
            residual_threshold = 1e-16,
            extension_quantity = "interface")

        narrowband_setup = levelset_setup.narrowband

        reinitialization_setup = levelset_setup.reinitialization_startup
        reinitializer = reinitialization_setup.type
        self.reinitializer: PDEBasedReinitializer = reinitializer(
            domain_information = domain_information,
            halo_manager = halo_manager,
            reinitialization_setup = reinitialization_setup,
            narrowband_setup = narrowband_setup)

        self.levelset_creator = LevelsetCreator(
                domain_information = self.domain_information,
                unit_handler = unit_handler,
                initial_condition_levelset = initial_condition_levelset,
                is_double_precision = numerical_setup.precision.is_double_precision_compute)

    def initialize(
            self,
            user_levelset_init: Union[np.ndarray, Array] = None,
            user_solid_interface_velocity_init: Union[np.ndarray, Array] = None,
            user_restart_file_path: str | None = None
        ) -> LevelsetFieldBuffers:
        """Initializes the levelset related field buffers. Peforms ghost cell
        treatment on the material fields.

        :param user_levelset_init: _description_, defaults to None
        :type user_levelset_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: Dict[str, Array]
        """

        is_restart = self.restart_setup.flag
        is_interpolate = self.restart_setup.is_interpolate
        is_parallel = self.domain_information.is_parallel
        is_h5_file = self.initial_condition_levelset.is_h5_file

        if is_restart and is_interpolate:
            is_restart = False

            warning_string = (
                "For level-set simulations, the level-set field cannot be interpolated "
                "upon restart. However, in the case setup is_restart and is_interpolate "
                "are both set True. Default behavior is to overwrite is_restart for level-set "
                "and to attempt to initialize level-set field from the "
                "corresponding initial condition.")
            warnings.warn(warning_string, RuntimeWarning)

        if is_restart:
            levelset_fields = self.from_restart_file(user_restart_file_path)

        elif is_h5_file:
            levelset_fields = self.from_h5_file()
            
        elif user_levelset_init is not None:
            levelset_fields = self.from_user_specified_buffer(
                user_levelset_init, user_solid_interface_velocity_init)

        else:
            cell_centers = self.domain_information.get_local_cell_centers()
            if is_parallel:
                levelset_fields = jax.pmap(
                    self.from_levelset_initial_condition, axis_name="i")(
                    cell_centers)
            else:
                levelset_fields = self.from_levelset_initial_condition(
                    cell_centers)

        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset
        if "FLUID-SOLID" in levelset_model and not is_moving_levelset:
            solid_cell_indices = compute_solid_cell_indices(
                levelset_fields, self.geometry_calculator,
                self.domain_information, self.numerical_setup.levelset)
            levelset_fields = LevelsetFieldBuffers(
                levelset = levelset_fields.levelset,
                volume_fraction = levelset_fields.volume_fraction,
                apertures = levelset_fields.apertures,
                solid_cell_indices = solid_cell_indices)

        return levelset_fields

  
    def create_levelset_fields(
            self,
            levelset_np: Array,
            interface_velocity_np: Array = None
            ) -> LevelsetFieldBuffers:
        """Wrapper preparing the levelset fields from
        numpy buffers, i.e., creates corresponding jax.numpy buffers,
        performing non dimensionalization, halo update,
        interface reconstruction.

        :param levelset_np: _description_
        :type levelset_np: Array
        :param interface_velocity_np: _description_
        :type interface_velocity_np: Array
        :return: _description_
        :rtype: LevelsetFieldBuffers
        """

        nh = self.domain_information.nh_conservatives
        device_number_of_cells = self.domain_information.device_number_of_cells
        split_factors = self.domain_information.split_factors
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        perform_cutoff = self.numerical_setup.levelset.narrowband.perform_cutoff
        dtype = jnp.float64 if self.is_double_precision else jnp.float32
        
        levelset = create_field_buffer(nh, device_number_of_cells, dtype)
        levelset = levelset.at[...,nhx,nhy,nhz].set(levelset_np)
        levelset = self.unit_handler.non_dimensionalize(levelset, "length")
        if perform_cutoff:
            levelset = self.reinitializer.set_levelset_cutoff(levelset)
        levelset = self.halo_manager.perform_halo_update_levelset(levelset, True, True)
        volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(levelset)

        if interface_velocity_np is not None:
            interface_velocity = create_field_buffer(nh, device_number_of_cells, dtype, 3)
            interface_velocity = interface_velocity.at[...,nhx,nhy,nhz].set(interface_velocity_np)
        else:
            interface_velocity = None

        levelset_fields = LevelsetFieldBuffers(
            levelset, volume_fraction, apertures,
            interface_velocity)
        
        return levelset_fields

    def from_restart_file(self, user_restart_file_path: str | None) -> LevelsetFieldBuffers:
        """Creates the initial levelset related field
        buffers from a restart file.

        :return: _description_
        :rtype: Dict[str, Array]
        """
        
        # DOMAIN INFORMATION
        split_factors = self.domain_information.split_factors
        is_parallel = self.domain_information.is_parallel
        is_multihost = self.domain_information.is_multihost
        local_device_count = self.domain_information.local_device_count
        process_id = self.domain_information.process_id
        levelset_model = self.equation_information.levelset_model
        restart_file_path = self.restart_setup.file_path if user_restart_file_path is None else user_restart_file_path
        is_equal_decomposition_multihost = self.restart_setup.is_equal_decomposition_multihost

        restart_file_path = parse_restart_files(restart_file_path)
        h5file_list = get_h5file_list(restart_file_path, process_id,
                                      is_equal_decomposition_multihost)
        h5file = h5file_list[0]

        is_parallel_restart = h5file["metadata"]["is_parallel"][()]
        split_factors_restart = h5file["domain"]["split_factors"][:]

        # SANITY CHECK
        levelset_keys = h5file["metadata"]["available_quantities"]["levelset"][:].astype("U")
        assert "levelset" in levelset_keys, "Levelset not in restart file %s." % restart_file_path

        load_function = get_load_function(is_parallel, is_parallel_restart,
                                          split_factors, split_factors_restart)

        load_function_inputs = {
            "h5file": h5file_list,
            "split_factors": split_factors,
            "split_factors_restart": split_factors_restart,
            }

        # LOAD DATA FROM H5 INTO NUMPY ARRAYS
        levelset = load_function("levelset/levelset", **load_function_inputs)

        if is_parallel:
            if is_multihost and not is_equal_decomposition_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                levelset = levelset[s_]
            levelset_fields = jax.pmap(self.create_levelset_fields, axis_name="i")(
                levelset)
        else:
            levelset_fields = self.create_levelset_fields(
                levelset)

        return levelset_fields

    def from_user_specified_buffer(
            self,
            user_levelset_init: Array,
            user_solid_interface_velocity_init: Array,
            ) -> LevelsetFieldBuffers:
        """Creates the initial levelset related
        field buffers from user specified buffers.

        :param user_levelset_init: _description_
        :type user_levelset_init: Array
        :param user_solid_interface_velocity_init: _description_, defaults to None
        :type user_solid_interface_velocity_init: Array, optional
        :return: _description_
        :rtype: Dict
        """

        # DOMAIN INFORMATION
        split_factors = self.domain_information.split_factors
        is_parallel = self.domain_information.is_parallel
        is_multihost = self.domain_information.is_multihost
        process_id = self.domain_information.process_id
        local_device_count = self.domain_information.local_device_count
        levelset_model = self.equation_information.levelset_model
        number_of_cells = self.domain_information.global_number_of_cells
        
        assert_string = ("Initial user levelset buffer does not have the a shape that "
                         "is consistent with the present case setup file.")
        assert tuple(number_of_cells) == user_levelset_init.shape, assert_string
            
        if is_parallel:
            user_levelset_init = split_buffer_np(user_levelset_init, split_factors)
            if is_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                user_levelset_init = user_levelset_init[s_]
            levelset_fields = jax.pmap(self.create_levelset_fields, axis_name="i")(
                user_levelset_init)
        else:
            levelset_fields = self.create_levelset_fields(
                user_levelset_init)

        return levelset_fields

    def from_h5_file(self) -> LevelsetFieldBuffers:
        """Initializes the levelset from the
        h5file specified in the case setup .json
        file.

        :return: _description_
        :rtype: LevelsetFieldBuffers
        """
        is_parallel = self.domain_information.is_parallel
        split_factors = self.domain_information.split_factors
        h5_file_path = self.initial_condition_levelset.h5_file_path
        is_multihost = self.domain_information.is_multihost
        process_id = self.domain_information.process_id
        local_device_count = self.domain_information.local_device_count

        with h5py.File(h5_file_path, "r") as h5file:
            levelset_init = h5file["levelset"][:]

        if is_parallel:
            levelset_init = split_buffer_np(levelset_init, split_factors)
            if is_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                levelset_init = levelset_init[s_]
            levelset_fields = jax.pmap(self.create_levelset_fields, axis_name="i")(levelset_init)
        else:
            levelset_fields = self.create_levelset_fields(levelset_init)

        return levelset_fields   

    def from_levelset_initial_condition(
            self,
            cell_centers: List,
            ) -> LevelsetFieldBuffers:
        """Computes the levelset
        buffers from the primitive initial
        condition provided in the case setup .json
        file.


        :param cell_centers: _description_
        :type cell_centers: List
        :return: _description_
        :rtype: Tuple
        """

        # DOMAIN INFORMATION
        device_number_of_cells = self.domain_information.device_number_of_cells
        nh = self.domain_information.nh_conservatives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        active_axes_indices = self.domain_information.active_axes_indices
        dtype = jnp.float64 if self.is_double_precision else jnp.float32
        
        mesh_grid = self.domain_information.compute_device_mesh_grid()

        # CREATE LEVELSET
        levelset = create_field_buffer(nh, device_number_of_cells, dtype)
        levelset = self.levelset_creator.create_levelset(levelset, mesh_grid)

        # REINITIALIZE LEVELSET
        levelset_setup = self.numerical_setup.levelset
        CFL = levelset_setup.reinitialization_startup.CFL
        steps = levelset_setup.reinitialization_startup.steps
        if steps > 1:
            levelset = self.halo_manager.perform_halo_update_levelset(levelset, True, True)
            levelset, _ = self.reinitializer.perform_reinitialization(
                levelset, CFL, steps)

        # CUTOFF AND HALOS
        perform_cutoff = levelset_setup.narrowband.perform_cutoff
        if perform_cutoff:
            levelset = self.reinitializer.set_levelset_cutoff(levelset)
        levelset = self.halo_manager.perform_halo_update_levelset(levelset, True, True)

        # INTERFACE RECONSTRUCTION
        volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(levelset)

        levelset_fields = LevelsetFieldBuffers(
            levelset, volume_fraction, apertures)

        return levelset_fields