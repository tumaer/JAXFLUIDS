from typing import Union, Callable, Dict, List, Tuple
import warnings
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
import h5py

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.geometry_calculator import GeometryCalculator
from jaxfluids.levelset.ghost_cell_handler import GhostCellHandler
from jaxfluids.levelset.interface_quantity_computer import InterfaceQuantityComputer
from jaxfluids.levelset.creation.levelset_creator import LevelsetCreator
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.initialization.helper_functions import create_field_buffer, get_load_function
from jaxfluids.domain.helper_functions import split_buffer_np
from jaxfluids.data_types.buffers import LevelsetFieldBuffers, MaterialFieldBuffers, TimeControlVariables
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup.solid_properties import SolidPropertiesSetup
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionLevelset
from jaxfluids.data_types.case_setup.initial_conditions import VelocityCallable
from jaxfluids.data_types.case_setup.restart import RestartSetup
from jaxfluids.levelset.residual_computer import ResidualComputer
from jaxfluids.levelset.quantity_extender import QuantityExtender
from jaxfluids.data_types.information import LevelsetResidualInformation

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
            initial_condition_solid_velocity: VelocityCallable,
            solid_properties: SolidPropertiesSetup,
            restart_setup: RestartSetup
            ) -> None:

        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager
        self.numerical_setup = numerical_setup
        self.restart_setup = restart_setup
        self.initial_condition_levelset = initial_condition_levelset
        self.initial_condition_solid_velocity = initial_condition_solid_velocity
        self.is_double_precision = numerical_setup.precision.is_double_precision_compute

        self.geometry_calculator = GeometryCalculator(
            domain_information = domain_information,
            levelset_setup = numerical_setup.levelset)

        extender_primes = QuantityExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            extension_setup = numerical_setup.levelset.extension,
            narrowband_setup = numerical_setup.levelset.narrowband,
            extension_quantity = "primitives")
        
        extender_interface = QuantityExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            extension_setup = numerical_setup.levelset.extension,
            narrowband_setup = numerical_setup.levelset.narrowband,
            extension_quantity = "interface")

        self.ghost_cell_handler = GhostCellHandler(
            domain_information = domain_information,
            halo_manager = halo_manager,
            extender_primes = extender_primes,
            equation_manager = equation_manager,
            levelset_setup = numerical_setup.levelset)
        
        self.interface_quantity_computer = InterfaceQuantityComputer(
            domain_information = domain_information,
            material_manager = material_manager,
            solid_properties = solid_properties,
            extender_interface = extender_interface,
            numerical_setup = numerical_setup)

        levelset_setup = numerical_setup.levelset

        reinitialization_setup = levelset_setup.reinitialization_startup
        nh_geometry = levelset_setup.halo_cells
        narrowband_setup = levelset_setup.narrowband
        reinitializer = reinitialization_setup.type

        self.reinitializer: LevelsetReinitializer = reinitializer(
            domain_information = domain_information,
            halo_manager = halo_manager,
            reinitialization_setup = reinitialization_setup,
            halo_cells = nh_geometry,
            narrowband_setup = narrowband_setup)

        self.levelset_creator = LevelsetCreator(
                domain_information = self.domain_information,
                unit_handler = unit_handler,
                initial_condition_levelset = initial_condition_levelset,
                is_double_precision = numerical_setup.precision.is_double_precision_compute)
        
        self.residual_computer = ResidualComputer(
            domain_information = domain_information,
            levelset_reinitializer = self.reinitializer,
            extender_interface = extender_interface,
            extender_primes = extender_primes,
            levelset_setup = levelset_setup)

    def initialize(
            self,
            material_fields: MaterialFieldBuffers,
            time_control_variables: TimeControlVariables,
            user_levelset_init: Union[np.ndarray, Array] = None,
            user_solid_interface_velocity_init: Union[np.ndarray, Array] = None,
            ) -> Tuple[MaterialFieldBuffers, LevelsetFieldBuffers,
                       LevelsetResidualInformation]:
        """Initializes the levelset related field buffers. Peforms ghost cell
        treatment on the material fields.

        :param user_levelset_init: _description_, defaults to None
        :type user_levelset_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: Dict[str, Array]
        """

        is_restart = self.restart_setup.flag
        is_parallel = self.domain_information.is_parallel
        is_h5_file = self.initial_condition_levelset.is_h5_file

        if is_restart:
            levelset_fields = self.from_restart_file()

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

        if is_parallel:
            material_fields, levelset_fields = jax.pmap(
                self.ghost_cell_treatment_and_interface_quantities, axis_name="i", in_axes=(0,0,None))(
                material_fields, levelset_fields, time_control_variables)
        else:
            material_fields, levelset_fields = self.ghost_cell_treatment_and_interface_quantities(
                material_fields, levelset_fields, time_control_variables)

        return levelset_fields, material_fields

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

    def from_restart_file(self) -> LevelsetFieldBuffers:
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
        restart_file_path = self.restart_setup.file_path
        is_equal_decomposition = self.restart_setup.is_equal_decomposition

        # LOAD H5 FILE
        h5file_basename = os.path.split(restart_file_path)[-1]
        if "proc" in h5file_basename:
            if is_equal_decomposition:
                h5file = h5py.File(restart_file_path, "r")
                h5file_list = [h5file]
            else:
                h5file_path = os.path.split(restart_file_path)[0]
                time_string = h5file_basename.split("_")[-1]
                h5file_names = [file for file in os.listdir(h5file_path) if time_string in file]
                h5file_list = []
                for i in range(len(h5file_names)):
                    file_name = f"data_proc{i:d}_{time_string:s}"
                    file_path = os.path.join(h5file_path, file_name)
                    h5file = h5py.File(file_path, "r")
                    h5file_list.append(h5file)
        else:
            h5file = h5py.File(restart_file_path, "r")
            h5file_list = [h5file]

        is_parallel_restart = h5file["metadata"]["is_parallel"][()]
        split_factors_restart = h5file["domain"]["split_factors"][:]

        # SANITY CHECK
        levelset_keys = h5file["metadata"]["available_quantities"]["levelset"][:].astype("U")
        assert "levelset" in levelset_keys, "Levelset not in restart file %s." % restart_file_path
        if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            assert "interface_velocity" in levelset_keys, "Solid interface velocity not in restart file %s." % restart_file_path

        load_function = get_load_function(is_parallel, is_parallel_restart,
                                          split_factors, split_factors_restart)

        load_function_inputs = {
            "h5file": h5file_list,
            "split_factors": split_factors,
            "split_factors_restart": split_factors_restart,
            }

        # LOAD DATA FROM H5 INTO NUMPY ARRAYS
        levelset = load_function("levelset/levelset", **load_function_inputs)
        if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            interface_velocity = load_function("levelset/interface_velocity", **load_function_inputs, is_vector_buffer=True)
        else:
            interface_velocity = None

        if is_parallel:
            if is_multihost and not is_equal_decomposition:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                levelset = levelset[s_]
            levelset_fields = jax.pmap(self.create_levelset_fields, axis_name="i")(
                levelset, interface_velocity)
        else:
            levelset_fields = self.create_levelset_fields(
                levelset, interface_velocity)

        return levelset_fields

    def from_user_specified_buffer(
            self,
            user_levelset_init: Array,
            user_solid_interface_velocity_init: Array = None,
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
        

        if user_solid_interface_velocity_init is not None:
            assert_string = ("User specified initial solid interface velocity provided, "
                             "however, the levelset model is not FLUID-SOLID-DYNAMIC-COUPLED.")
            assert levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED", assert_string
            assert_string = ("Initial user solid interface velocity buffer "
                            "does not have the a shape that is consistent with the present "
                            "case setup file.")
            assert (3,) +tuple(number_of_cells) == user_solid_interface_velocity_init.shape, assert_string
            
        if is_parallel:
            user_levelset_init = split_buffer_np(user_levelset_init, split_factors)
            if user_solid_interface_velocity_init is not None:
                user_solid_interface_velocity_init = split_buffer_np(user_solid_interface_velocity_init,
                                                                     split_factors)
            if is_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                user_levelset_init = user_levelset_init[s_]
                if user_solid_interface_velocity_init is not None:
                    user_solid_interface_velocity_init = user_solid_interface_velocity_init[s_]
            levelset_fields = jax.pmap(self.create_levelset_fields, axis_name="i")(
                user_levelset_init, user_solid_interface_velocity_init)
        else:
            levelset_fields = self.create_levelset_fields(
                user_levelset_init, user_solid_interface_velocity_init)

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
            levelset = self.reinitializer.perform_reinitialization(
                levelset, CFL, steps)

        # CUTOFF AND HALOS
        perform_cutoff = levelset_setup.narrowband.perform_cutoff
        if perform_cutoff:
            levelset = self.reinitializer.set_levelset_cutoff(levelset)
        levelset = self.halo_manager.perform_halo_update_levelset(levelset, True, True)

        # INTERFACE RECONSTRUCTION
        volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(levelset)

        # INITIAL SOLID VELOCITY FOR COUPLED INTERACTION
        if self.equation_information.levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            buffer = create_field_buffer(nh, device_number_of_cells, dtype, 3)
            solid_velocity = []
            for field in self.initial_condition_solid_velocity._fields:
                velocity_callable = getattr(self.initial_condition_solid_velocity, field)
                velocity = velocity_callable(*mesh_grid)
                solid_velocity.append(velocity)
            solid_velocity = jnp.stack(solid_velocity)
            solid_velocity = buffer.at[...,nhx,nhy,nhz].set(solid_velocity)
        else:
            solid_velocity = None
        
        levelset_fields = LevelsetFieldBuffers(
            levelset, volume_fraction, apertures,
            solid_velocity)

        return levelset_fields

    def ghost_cell_treatment_and_interface_quantities(
            self,
            material_fields: MaterialFieldBuffers,
            levelset_fields: LevelsetFieldBuffers,
            time_control_variables: TimeControlVariables
            ) -> Tuple[MaterialFieldBuffers,
                    LevelsetFieldBuffers]:
        """Wrapper to perform ghost cell treatment
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
        levelset = levelset_fields.levelset
        volume_fraction = levelset_fields.volume_fraction
        apertures = levelset_fields.apertures
        interface_velocity = levelset_fields.interface_velocity
        physical_simulation_time = time_control_variables.physical_simulation_time
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        fill_edge_halos = self.numerical_setup.active_physics.is_viscous_flux

        levelset_model = self.equation_information.levelset_model
        if levelset_model == "FLUID-SOLID-DYNAMIC":
            solid_velocity = self.interface_quantity_computer.compute_solid_velocity(physical_simulation_time)
        elif levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            solid_velocity = levelset_fields.interface_velocity
            solid_velocity = solid_velocity[...,nhx,nhy,nhz]
        else:
            solid_velocity = None
        normal = self.geometry_calculator.compute_normal(levelset)
        conservatives, primitives, _, _ = self.ghost_cell_handler.perform_ghost_cell_treatment(
            conservatives, primitives, levelset, volume_fraction,
            physical_simulation_time, normal, solid_velocity,
            CFL=0.5, steps=100)

        primitives, conservatives = self.halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, fill_edge_halos, False, conservatives)
        material_fields = MaterialFieldBuffers(
            conservatives, primitives)
        
        if levelset_model == "FLUID-FLUID":
            curvature = self.geometry_calculator.compute_curvature(levelset)
            interface_velocity, interface_pressure, _ = \
            self.interface_quantity_computer.compute_interface_quantities(
                primitives, levelset, volume_fraction, normal, curvature, steps=100, CFL=0.5)
            levelset_fields = LevelsetFieldBuffers(
                levelset, volume_fraction, apertures, interface_velocity,
                interface_pressure)
            
        return material_fields, levelset_fields