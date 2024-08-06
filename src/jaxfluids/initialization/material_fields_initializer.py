import types
from typing import Union, Dict, List, Tuple, NamedTuple, Callable
import warnings
import os

import h5py
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.turb.initialization.turb_init_manager import TurbulentInitializationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.domain.helper_functions import split_buffer_np
from jaxfluids.initialization.helper_functions import create_field_buffer, get_load_function, expand_buffers
from jaxfluids.data_types.buffers import MaterialFieldBuffers, TimeControlVariables
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionSetup
from jaxfluids.data_types.case_setup.restart import RestartSetup

class MaterialFieldsInitializer:
    """The MaterialFieldsInitializer implements functionality
    to create initial buffers for the conservative and
    primitive variables.
    """
    def __init__(
            self,
            domain_information: DomainInformation,
            unit_handler: UnitHandler,
            equation_manager: EquationManager,
            material_manager: MaterialManager,
            halo_manager: HaloManager,
            initial_condition: InitialConditionSetup,
            restart_setup: RestartSetup,
            is_double_precision: bool
            ) -> None:

        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.material_manager = material_manager
        self.halo_manager = halo_manager
        self.restart_setup = restart_setup
        
        self.initial_condition_primitives = initial_condition.primitives
        self.is_turb_init = initial_condition.is_turbulent
        self.is_double_precision = is_double_precision
        self.is_viscous_flux = self.equation_information.active_physics.is_viscous_flux

        if self.is_turb_init:
            self.turb_init_condition = TurbulentInitializationManager(
                domain_information=self.domain_information,
                material_manager=material_manager,
                initial_condition_turbulent=initial_condition.turbulent)

    def initialize(
            self,
            user_prime_init: Union[np.ndarray, Array] = None,
            user_time_init: float = None
            ) -> Tuple[MaterialFieldBuffers, TimeControlVariables]:
        """Initializes the material field buffers.

        :param user_prime_init: _description_, defaults to None
        :type user_prime_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: Dict[str, Array]
        """

        is_restart = self.restart_setup.flag
        is_parallel = self.domain_information.is_parallel
        cell_centers = self.domain_information.get_local_cell_centers()

        physical_simulation_time = 0.0
        simulation_step = 0

        if is_restart:
            material_fields, physical_simulation_time = self.from_restart_file()

        elif user_prime_init is not None:
            material_fields, physical_simulation_time \
                = self.from_user_specified_buffer(user_prime_init, user_time_init)
            
        elif self.is_turb_init:
            if is_parallel:
                material_fields = jax.pmap(
                    self.from_turbulent_initial_condition,
                    axis_name="i")(cell_centers)
            else:
                material_fields = self.from_turbulent_initial_condition(
                    cell_centers)

        else:
            if is_parallel:
                material_fields = jax.pmap(
                    self.from_primitive_initial_condition,
                    axis_name="i")(cell_centers)
            else:
                material_fields = self.from_primitive_initial_condition(
                    cell_centers)

        time_control_variables = TimeControlVariables(
            physical_simulation_time, simulation_step)

        return material_fields, time_control_variables

    def create_material_fields(
            self,
            primitives_np: Array,
            physical_simulation_time: float
            ) -> MaterialFieldBuffers:
        """Prepares the material fields given a
        numpy primitive buffer, i.e.,
        creates the corresponding jax.numpy buffer,
        performs non dimensionalization, halo update,
        and computes conservative buffer.

        :param primitives_np: _description_
        :type primitives_np: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: MaterialFieldBuffers
        """

        nh = self.domain_information.nh_conservatives
        device_number_of_cells = self.domain_information.device_number_of_cells
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        split_factors = self.domain_information.split_factors
        no_primes = self.equation_information.no_primes
        equation_type = self.equation_information.equation_type
        dtype = np.float64 if self.is_double_precision else np.float32

        velocity_ids = self.equation_information.velocity_ids
        inactive_axes_indices = self.domain_information.inactive_axes_indices
        inactive_velocity_indices = [velocity_ids[i] for i in range(3) if i in inactive_axes_indices]
        prime_indices = []
        for i in range(no_primes):
            if i not in inactive_velocity_indices:
                prime_indices.append(i)

        leading_dim = (5,2) if equation_type == "TWO-PHASE-LS" else no_primes
        primitives = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim)
        primitives = primitives.at[prime_indices,...,nhx,nhy,nhz].set(primitives_np)
        quantity_list = self.equation_information.primitive_quantities
        primitives = self.unit_handler.non_dimensionalize(primitives, "specified", quantity_list)
        conservatives = self.equation_manager.get_conservatives_from_primitives(primitives)
        primitives, conservatives = self.halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, self.is_viscous_flux,
            False, conservatives)
        material_fields = MaterialFieldBuffers(
            conservatives, primitives)
        return material_fields
    
    def from_restart_file(self) -> MaterialFieldBuffers:
        """Initializes the material field buffers
        from a restart .h5 file.

        :return: _description_
        :rtype: _type_
        """
        
        restart_file_path = self.restart_setup.file_path
        restart_time = self.restart_setup.time
        use_restart_time = self.restart_setup.use_time
        is_equal_decomposition = self.restart_setup.is_equal_decomposition

        # DOMAIN INFORMATION
        number_of_cells = self.domain_information.global_number_of_cells
        is_parallel = self.domain_information.is_parallel
        is_multihost = self.domain_information.is_multihost
        host_count = self.domain_information.host_count
        split_factors = self.domain_information.split_factors
        local_device_count = self.domain_information.local_device_count
        process_id = self.domain_information.process_id
        dim = self.domain_information.dim

        # EQUATION INFORMATION
        equation_type = self.equation_information.equation_type
        levelset_model = self.equation_information.levelset_model
        diffuse_interface_model = self.equation_information.diffuse_interface_model

        if os.path.isfile(restart_file_path):
            assert_string = (f"{restart_file_path} is not a valid h5-file. "
                             "For restarting a JAX-Fluids simulation, restart_file_path "
                             "must either point to a valid h5-file or to an output folder "
                             "containing a valid h5-file.")
            assert restart_file_path.endswith(".h5"), assert_string

        elif os.path.isdir(restart_file_path):
            # IF restart_file_path is a folder, try to find last data_*.h5 checkpoint
            warning_string = (f"Restart file path {restart_file_path} points "
            "to a folder and not to a file. By default, the simulation is "
            "restarted from the latest existing checkpoint file in the given folder.")
            warnings.warn(warning_string, RuntimeWarning)

            files = []
            is_multihost_restart = []
            for file in os.listdir(restart_file_path):
                if file.endswith("h5"):
                    if "nan" in file:
                        assert_string = (
                            f"Trying to restart from given folder {restart_file_path}. "
                            "However, a nan file was found. Aborting default restart.")
                        assert False, assert_string

                    if file.startswith("data_proc"):
                        if file.startswith("data_proc0"):
                            files.append(file)
                    elif file.startswith("data_"):
                        files.append(file)
                    else:
                        assert_string = (
                            f"Trying to restart from given folder {restart_file_path}. "
                            "However, no data_*.h5 or data_proc*.h5 files found. "
                            "Aborting default restart.")
                        assert False, assert_string
                    is_multihost_restart.append(file.startswith("data_proc"))

            assert_string = (f"Trying to restart from given folder {restart_file_path}. "
                             "However, no suitable h5 files were found. "
                             "Aborting default restart.")
            assert len(files) > 0, assert_string

            is_multihost_restart = all(is_multihost_restart)
            if is_multihost_restart:
                times = [float(os.path.splitext(file)[0][11:]) for file in files]
            else:
                times = [float(os.path.splitext(file)[0][5:]) for file in files]

            indices = np.argsort(np.array(times))
            last_file = np.array(files)[indices][-1]
            restart_file_path = os.path.join(restart_file_path, last_file)

        else:
            assert_string = (
                "restart_file_path must be an existing regular file or an existing directory. "
                f"However, {restart_file_path} is neither of the above.")
            assert False, assert_string

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

        if use_restart_time:
            physical_simulation_time = restart_time
        else:
            physical_simulation_time = h5file["time"][()]
        physical_simulation_time = self.unit_handler.non_dimensionalize(physical_simulation_time, "time")

        # SANITY CHECK
        primes_restart = h5file["metadata"]["available_quantities"]["primitives"][:].astype("U")
        dim_restart = h5file["domain"]["dim"][()]
        split_factors_restart = h5file["domain"]["split_factors"][:]
        number_of_cells_restart = []
        for i, axis in enumerate(["X", "Y", "Z"]):
            cells_xi = h5file["domain"][f"grid{axis}"]
            split_xi = split_factors_restart[i]
            if cells_xi.size == 1:
                number_of_cells_xi = 1
            else:
                number_of_cells_xi = cells_xi.shape[-1]*split_xi
            number_of_cells_restart.append(number_of_cells_xi)

        is_parallel_restart = h5file["metadata"]["is_parallel"][()]
        levelset_model_restart = h5file["metadata"]["levelset_model"][()]
        diffuse_interface_model_restart = h5file["metadata"]["diffuse_interface_model"][()]
        levelset_model_restart = levelset_model_restart.decode("utf-8") if \
            type(levelset_model_restart) != np.bool_ else levelset_model_restart
        diffuse_interface_model_restart = diffuse_interface_model_restart.decode("utf-8") if \
            type(diffuse_interface_model_restart) != np.bool_ else diffuse_interface_model_restart

        assert_string = (f"Dimension of restart file {restart_file_path} does "
                         "not match case setup file.")
        assert dim == dim_restart, assert_string

        assert_string = (f"Number of cells of restart file {restart_file_path} do "
                         "not match case setup file.")
        assert (np.array(number_of_cells) == number_of_cells_restart).all(), assert_string
        
        assert_string = (f"Levelset model of restart file {restart_file_path} does"
                         "not match numerical setup file. ")
        assert levelset_model_restart == levelset_model, assert_string
        
        assert_string = (f"Diffuse interface model of restart file {restart_file_path} "
                         "does not match numerical setup file.")
        assert diffuse_interface_model_restart == diffuse_interface_model, assert_string
        
        load_function = get_load_function(is_parallel, is_parallel_restart,
                                          split_factors, split_factors_restart)

        load_function_inputs = {
            "h5file": h5file_list,
            "split_factors": split_factors,
            "split_factors_restart": split_factors_restart,
            }

        axis_expand = 0 if not is_parallel else 1

        # SINGLE-PHASE
        if equation_type in ["SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS"]:
            for quantity in ["density", "velocity", "pressure"]:
                assert quantity in primes_restart, f"{quantity:s} not in restart file {restart_file_path:s}."
            density = load_function("primitives/density", **load_function_inputs)
            pressure = load_function("primitives/pressure", **load_function_inputs)
            velocity = load_function("primitives/velocity", **load_function_inputs, is_vector_buffer=True)
            density, pressure = expand_buffers(density, pressure, axis=axis_expand)
            primitives = np.concatenate([density, velocity, pressure], axis=axis_expand)

        # TWO-PHASE LS
        elif equation_type == "TWO-PHASE-LS":
            for quantity in ["density", "density", "velocity"]:
                assert quantity in primes_restart, f"{quantity:s} not in restart file {restart_file_path:s}."
            density_0 = load_function("primitives/density_0", **load_function_inputs)
            density_1 = load_function("primitives/density_1", **load_function_inputs) 
            pressure_0 = load_function("primitives/pressure_0", **load_function_inputs) 
            pressure_1 = load_function("primitives/pressure_1", **load_function_inputs) 
            velocity_0 = load_function("primitives/velocity_0", **load_function_inputs, is_vector_buffer=True)
            velocity_1 = load_function("primitives/velocity_1", **load_function_inputs, is_vector_buffer=True)
            density_0, density_1, pressure_0, pressure_1 = expand_buffers(
                density_0, density_1, pressure_0, pressure_1, axis=axis_expand)
            primitives_0 = np.concatenate([density_0, velocity_0, pressure_0], axis=axis_expand)
            primitives_1 = np.concatenate([density_1, velocity_1, pressure_1], axis=axis_expand)
            primitives = np.stack([primitives_0, primitives_1], axis=axis_expand+1)
                            
        # DIFFUSE-INTERFACE-5EQM
        elif equation_type == "DIFFUSE-INTERFACE-5EQM":
            no_fluids = self.equation_information.no_fluids
            alpharho_list = [ "alpharho_%i" %ii for ii in range(no_fluids)  ]
            alpha_list = [ "alpha_%i" %ii for ii in range(no_fluids - 1) ]
            for quantity in ["velocity", "pressure"] + alpharho_list + alpha_list:
                assert quantity in primes_restart, f"{quantity:s} not in restart file {restart_file_path:s}."
            alpharho = []
            for i in range(no_fluids):
                alpharho_i = load_function("primitives/alpharho_%i" % i, **load_function_inputs)
                alpharho.append(alpharho_i)
            pressure = load_function("primitives/pressure", **load_function_inputs)
            velocity = load_function("primitives/velocity", **load_function_inputs, is_vector_buffer=True)
            alpha = []
            for i in range(no_fluids - 1):
                alpha_i = load_function("primitives/alpha_%i" % i, **load_function_inputs)
                alpha.append(alpha_i)
            alpharho = expand_buffers(*alpharho, axis=axis_expand)
            alpha = expand_buffers(*alpha, axis=axis_expand)
            pressure = np.expand_dims(pressure, axis=axis_expand)
            primitives = np.concatenate([*alpharho, velocity, pressure, *alpha], axis=axis_expand)

        else:
            raise NotImplementedError

        if is_parallel:
            if is_multihost and not is_equal_decomposition:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                primitives = primitives[s_]
            material_fields = jax.pmap(self.create_material_fields, axis_name="i", in_axes=(0,None))(
                primitives, physical_simulation_time)
        else:
            material_fields = self.create_material_fields(primitives, physical_simulation_time)

        return material_fields, physical_simulation_time

    def from_user_specified_buffer(
            self,
            user_prime_init: Array,
            user_time_init: float
            ) -> MaterialFieldBuffers:
        """Initializes the simulations from buffers provided by the user.

        :param user_prime_init: _description_
        :type user_prime_init: Array
        :param user_levelset_init: _description_
        :type user_levelset_init: Array
        :return: _description_
        :rtype: _type_
        """

        is_parallel = self.domain_information.is_parallel
        is_multihost = self.domain_information.is_multihost
        process_id = self.domain_information.process_id
        local_device_count = self.domain_information.local_device_count
        split_factors = self.domain_information.split_factors
        number_of_cells = self.domain_information.global_number_of_cells
        no_primes = self.equation_information.no_primes
        equation_type = self.equation_information.equation_type
        dim = self.domain_information.dim

        if equation_type == "TWO-PHASE-LS":
            buffer_shape = tuple([no_primes-3+dim,2,*number_of_cells])
        elif equation_type in ("SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS", "DIFFUSE-INTERFACE-5EQM"):
            buffer_shape = tuple([no_primes-3+dim,*number_of_cells])
        else:
            raise NotImplementedError

        assert_string = ("Given initial user primitive buffer has shape "
                         f"{user_prime_init.shape} which is not "
                         "consistent with the present "
                         "case setup file. The required shape is "
                         f"{buffer_shape}.")
        assert buffer_shape == user_prime_init.shape, assert_string

        if isinstance(user_time_init, float):
            physical_simulation_time = self.unit_handler.non_dimensionalize(user_time_init, "time")
        else:
            physical_simulation_time = 0.0

        if is_parallel:
            primitives = split_buffer_np(user_prime_init, split_factors)
            if is_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                primitives = primitives[s_]
            material_fields = jax.pmap(self.create_material_fields, axis_name="i", in_axes=(0,None))(
                primitives, physical_simulation_time)
        else:
            material_fields = self.create_material_fields(user_prime_init, physical_simulation_time)

        
        return material_fields, physical_simulation_time        

    def from_turbulent_initial_condition(
            self,
            cell_centers: List
            ) -> MaterialFieldBuffers:
        """Initializes the material field
        buffers from turbulent initial conditions.

        :param cell_centers: _description_
        :type cell_centers: List
        :return: _description_
        :rtype: MaterialFieldBuffers
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nh = self.domain_information.nh_conservatives
        split_factors = self.domain_information.split_factors
        device_number_of_cells = self.domain_information.device_number_of_cells
        active_axes_indices = self.domain_information.active_axes_indices
        no_primes = self.equation_information.no_primes

        dtype = np.float64 if self.is_double_precision else np.float32

        mesh_grid = self.domain_information.compute_device_mesh_grid()

        primitives_init = self.turb_init_condition.get_turbulent_initial_condition(
            mesh_grid)

        primitives = create_field_buffer(nh, device_number_of_cells, dtype, no_primes)
        primitives = primitives.at[...,nhx,nhy,nhz].set(primitives_init)
        conservatives = self.equation_manager.get_conservatives_from_primitives(primitives)
        primitives, conservatives = self.halo_manager.perform_halo_update_material(
            primitives, 0.0, self.is_viscous_flux, False, conservatives)

        material_fields = MaterialFieldBuffers(
            conservatives=conservatives,
            primitives=primitives)
        
        return material_fields

    def from_primitive_initial_condition(
            self,
            cell_centers: List
            ) -> MaterialFieldBuffers:
        """Computes the conservative and primitive
        variable buffers from the primitive initial
        condition provided in the case setup .json
        file.

        :param cell_centers: _description_
        :type cell_centers: List
        :return: _description_
        :rtype: MaterialFieldBuffers
        """
        # DOMAIN/EQUATION INFORMATION
        nh = self.domain_information.nh_conservatives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        device_number_of_cells = self.domain_information.device_number_of_cells

        no_primes = self.equation_information.no_primes
        equation_type = self.equation_information.equation_type
        dtype = np.float64 if self.is_double_precision else np.float32

        mesh_grid = self.domain_information.compute_device_mesh_grid()

        def _read_prime_state(
                initial_condition: NamedTuple
                ) -> Array:
            """Wrapper to evaluate the 
            primitive initial condition.

            :param initial_condition: _description_
            :type initial_condition: NamedTuple
            :return: _description_
            :rtype: Array
            """
            primes_init_list = []
            for prime_state in initial_condition._fields:
                prime_callable: Callable = getattr(
                    initial_condition, prime_state)
                prime_init = prime_callable(*mesh_grid)
                primes_init_list.append(prime_init)
            primes_init = jnp.stack(primes_init_list)
            return primes_init

        if equation_type == "TWO-PHASE-LS":
            leading_dim = (5,2)
            primitives = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim)
            for i, phase in enumerate(self.initial_condition_primitives._fields):
                initial_condition_phase: NamedTuple = getattr(
                    self.initial_condition_primitives, phase)
                primes_init = _read_prime_state(initial_condition_phase)
                s_ = jnp.s_[...,i,nhx,nhy,nhz]
                primitives = primitives.at[s_].set(primes_init)

        else:
            leading_dim = no_primes
            primes_init = _read_prime_state(self.initial_condition_primitives)

            if equation_type in ("DIFFUSE-INTERFACE-5EQM",):

                if self.initial_condition_primitives._fields == self.equation_information.primes_tuple_:
                    # If initial_condition via (rho_1, rho_2, u, v, w, p, alpha_0),
                    # then overwrite rho_i with alpharho_i = rho_i * alpha_i
                    mass_slices = self.equation_information.mass_slices
                    vf_slices = self.equation_information.vf_slices
                    vf_full_vector = jnp.concatenate([
                        primes_init[vf_slices],
                        1.0 - jnp.sum(primes_init[vf_slices], axis=0, keepdims=True)
                    ], axis=0)
                    primes_init = primes_init.at[mass_slices].mul(vf_full_vector)

            else:
                pass

            primitives = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim)
            primitives = primitives.at[...,nhx,nhy,nhz].set(primes_init)

        conservatives = self.equation_manager.get_conservatives_from_primitives(primitives)

        primitives, conservatives = \
        self.halo_manager.perform_halo_update_material(
            primitives, 0.0, self.is_viscous_flux,
            False, conservatives)
        
        material_fields = MaterialFieldBuffers(
            conservatives, primitives)

        return material_fields