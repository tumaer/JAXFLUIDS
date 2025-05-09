import types
from typing import Union, Dict, List, Tuple, NamedTuple, Callable
import warnings
import os
import sys

import h5py
import jax
import jax.numpy as jnp
import json
import numpy as np

from jaxfluids.input.input_manager import InputManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.turbulence.initialization.turb_init_manager import TurbulenceInitializationManager
from jaxfluids.cavitation.cavitation_init_cond import CavitationInitializationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.domain.helper_functions import split_buffer_np
from jaxfluids.initialization.helper_functions import (
     create_field_buffer, get_load_function, expand_buffers,
     get_h5file_list, interpolate, parse_restart_files)
from jaxfluids.data_types.buffers import MaterialFieldBuffers, TimeControlVariables
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionSetup
from jaxfluids.data_types.case_setup.restart import RestartSetup
from jaxfluids.data_types.case_setup import CaseSetup
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

Array = jax.Array

class MaterialFieldsInitializer:
    """The MaterialFieldsInitializer implements functionality
    to create initial buffers for the conservative and
    primitive variables.
    """
    def __init__(
            self,
            case_setup: CaseSetup,
            numerical_setup: NumericalSetup,
            domain_information: DomainInformation,
            unit_handler: UnitHandler,
            equation_manager: EquationManager,
            material_manager: MaterialManager,
            halo_manager: HaloManager,
        ) -> None:

        self.case_setup = case_setup
        self.numerical_setup = numerical_setup
        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.material_manager = material_manager
        self.halo_manager = halo_manager
        
        initial_condition = self.case_setup.initial_condition_setup
        self.initial_condition_primitives = initial_condition.primitives
        self.is_turbulence_init = initial_condition.is_turbulent
        self.is_cavitation_init = initial_condition.is_cavitation
        self.is_double_precision = numerical_setup.precision.is_double_precision_compute

        if self.is_turbulence_init:
            self.turbulence_init_condition = TurbulenceInitializationManager(
                domain_information=self.domain_information,
                material_manager=material_manager,
                initial_condition_turbulent=initial_condition.turbulent)

        if self.is_cavitation_init:
            self.cavitation_init_condition = CavitationInitializationManager(
                domain_information=self.domain_information,
                material_manager=material_manager,
                initial_condition_cavitation=initial_condition.cavitation)

    def initialize(
            self,
            user_prime_init: Union[np.ndarray, Array] = None,
            user_time_init: float = None,
            user_restart_file_path: str = None,
            ml_setup: MachineLearningSetup = None
        ) -> Tuple[MaterialFieldBuffers, TimeControlVariables]:
        """Initializes the material field buffers.

        :param user_prime_init: _description_, defaults to None
        :type user_prime_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: Dict[str, Array]
        """
        restart_setup = self.case_setup.restart_setup
        is_restart = restart_setup.flag
        is_parallel = self.domain_information.is_parallel
        cell_centers = self.domain_information.get_local_cell_centers()

        physical_simulation_time = 0.0
        simulation_step = 0

        if is_restart:
            material_fields, physical_simulation_time = self.from_restart_file(
                user_restart_file_path, ml_setup)

        elif user_prime_init is not None:
            material_fields, physical_simulation_time = self.from_user_specified_buffer(
                user_prime_init,
                user_time_init,
                ml_setup
            )
            
        elif self.is_turbulence_init:
            if is_parallel:
                material_fields = jax.pmap(
                    self.from_turbulent_initial_condition,
                    axis_name="i")(cell_centers)
            else:
                material_fields = self.from_turbulent_initial_condition(cell_centers)

        elif self.is_cavitation_init:
            if is_parallel:
                material_fields = jax.pmap(
                    self.from_cavitation_initial_condition,
                    axis_name="i")(cell_centers)
            else:
                material_fields = self.from_cavitation_initial_condition(
                    cell_centers)

        else:
            if is_parallel:
                material_fields = jax.pmap(
                    self.from_primitive_initial_condition,
                    axis_name="i"
                )(cell_centers, ml_setup)
            else:
                material_fields = self.from_primitive_initial_condition(
                    cell_centers, ml_setup
                )

        time_control_variables = TimeControlVariables(
            physical_simulation_time, simulation_step
        )

        return material_fields, time_control_variables

    def create_material_fields(
            self,
            primitives_np: Array,
            physical_simulation_time: float,
            ml_setup: MachineLearningSetup
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

        fill_edge_halos = self.halo_manager.fill_edge_halos_material
        fill_vertex_halos = self.halo_manager.fill_vertex_halos_material

        ids_velocity = self.equation_information.ids_velocity
        inactive_axes_indices = self.domain_information.inactive_axes_indices
        inactive_velocity_indices = [ids_velocity[i] for i in range(3) if i in inactive_axes_indices]
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
            primitives, physical_simulation_time, fill_edge_halos,
            fill_vertex_halos, conservatives, ml_setup=ml_setup)
        
        if self.equation_information.is_compute_temperature:
            temperature = self.material_manager.get_temperature(primitives)
            temperature = self.halo_manager.perform_outer_halo_update_temperature(
                temperature, physical_simulation_time)
        else:
            temperature = None
        
        material_fields = MaterialFieldBuffers(
            conservatives, primitives, temperature)
        
        return material_fields
    
    def from_restart_file(
            self,
            user_restart_file_path: str | None,
            ml_setup: MachineLearningSetup
        ) -> MaterialFieldBuffers:
        """Initializes the material field buffers
        from a restart .h5 file.

        :return: _description_
        :rtype: _type_
        """
        restart_setup = self.case_setup.restart_setup
        restart_file_path = restart_setup.file_path if user_restart_file_path is None else user_restart_file_path
        restart_time = restart_setup.time
        use_restart_time = restart_setup.use_time
        is_interpolate = restart_setup.is_interpolate
        is_equal_decomposition_multihost = restart_setup.is_equal_decomposition_multihost

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

        restart_file_path = parse_restart_files(restart_file_path)
        h5file_list = get_h5file_list(restart_file_path, process_id,
                                      is_equal_decomposition_multihost)
        h5file = h5file_list[0]
        
        if use_restart_time:
            physical_simulation_time = restart_time
        else:
            physical_simulation_time = h5file["time"][()]
        physical_simulation_time = self.unit_handler.non_dimensionalize(physical_simulation_time, "time")

        # SANITY CHECK
        primes_restart = h5file["metadata"]["available_quantities"]["primitives"][:].astype("U")
        dim_restart = h5file["domain"]["dim"][()]
        active_axes_indices = h5file["domain"]["active_axes_indices"][:]
        split_factors_restart = h5file["domain"]["split_factors"][:]
        number_of_cells_restart = []

        for i, axis in enumerate(["X", "Y", "Z"]):
            cells_xi = h5file["domain"][f"grid{axis}"]
            split_xi = split_factors_restart[i]
            if i in active_axes_indices:
                number_of_cells_xi = cells_xi.shape[-1] * split_xi
            else:
                number_of_cells_xi = 1
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

        if not is_interpolate:
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
        if equation_type == "SINGLE-PHASE":
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

        if is_interpolate:
            restart_setup = self.case_setup.restart_setup
            case_setup_restart = json.load(open(restart_setup.case_setup_path))
            numerical_setup_restart = json.load(open(restart_setup.numerical_setup_path))
            for axis_name, axis_id in DomainInformation.axis_to_axis_id.items():
                case_setup_restart["domain"]["decomposition"][f"split_{axis_name}"] = split_factors[axis_id]
            input_manager_restart = InputManager(case_setup_restart, numerical_setup_restart)

            local_cell_centers = self.domain_information.get_local_cell_centers()
            local_cell_centers = tuple([self.unit_handler.dimensionalize(xi, "length") for xi in local_cell_centers])
            dtype = np.float64 if self.is_double_precision else np.float32

        if is_parallel:
            if is_multihost and not is_equal_decomposition_multihost:
                s_ = jnp.s_[process_id*local_device_count:(process_id+1)*local_device_count]
                primitives = primitives[s_]
            
            if is_interpolate:
                primitives = jax.pmap(interpolate, axis_name="i", 
                                      in_axes=(None,0,0,None,None,None,None,None),
                                      static_broadcasted_argnums=(0,4,5,6,7)
                                      )("MATERIAL",
                                        local_cell_centers, primitives, 
                                        physical_simulation_time,
                                        input_manager_restart.domain_information,
                                        input_manager_restart.equation_information,
                                        input_manager_restart.halo_manager,
                                        dtype)

            material_fields = jax.pmap(
                self.create_material_fields,
                axis_name="i",
                in_axes=(0, None, None)
            )(
                primitives,
                physical_simulation_time,
                ml_setup
            )
        else:
            if is_interpolate:
                primitives = interpolate("MATERIAL",
                                         local_cell_centers, primitives, 
                                         physical_simulation_time,
                                         input_manager_restart.domain_information,
                                         input_manager_restart.equation_information,
                                         input_manager_restart.halo_manager,
                                         dtype)
            material_fields = self.create_material_fields(
                primitives,
                physical_simulation_time,
                ml_setup
            )


        return material_fields, physical_simulation_time

    def from_user_specified_buffer(
            self,
            user_prime_init: Array,
            user_time_init: float,
            ml_setup: MachineLearningSetup
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
        elif equation_type in ("SINGLE-PHASE",
                               "DIFFUSE-INTERFACE-4EQM", "DIFFUSE-INTERFACE-5EQM",
                               ):
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
            material_fields = jax.pmap(
                self.create_material_fields,
                axis_name="i",
                in_axes=(0, None, None)
            )(
                primitives,
                physical_simulation_time,
                ml_setup
            )
        else:
            material_fields = self.create_material_fields(
                user_prime_init,
                physical_simulation_time,
                ml_setup
            )

        
        return material_fields, physical_simulation_time        

    def from_turbulent_initial_condition(self, cell_centers: Array) -> MaterialFieldBuffers:
        """Initializes the material field
        buffers from turbulent initial conditions.

        :return: _description_
        :rtype: MaterialFieldBuffers
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nh = self.domain_information.nh_conservatives
        split_factors = self.domain_information.split_factors
        device_number_of_cells = self.domain_information.device_number_of_cells
        active_axes_indices = self.domain_information.active_axes_indices
        no_primes = self.equation_information.no_primes

        fill_edge_halos = self.halo_manager.fill_edge_halos_material
        fill_vertex_halos = self.halo_manager.fill_vertex_halos_material

        dtype = np.float64 if self.is_double_precision else np.float32

        mesh_grid = self.domain_information.compute_device_mesh_grid()

        primitives_init = self.turbulence_init_condition.get_turbulent_initial_condition(
            mesh_grid)

        primitives = create_field_buffer(nh, device_number_of_cells, dtype, no_primes)
        primitives = primitives.at[...,nhx,nhy,nhz].set(primitives_init)
        conservatives = self.equation_manager.get_conservatives_from_primitives(primitives)
        primitives, conservatives = self.halo_manager.perform_halo_update_material(
            primitives, 0.0, fill_edge_halos, 
            fill_vertex_halos, conservatives)

        if self.equation_information.is_compute_temperature:
            temperature = self.material_manager.get_temperature(primitives)
            temperature = self.halo_manager.perform_outer_halo_update_temperature(
                temperature, 0.0)
        else:
            temperature = None

        material_fields = MaterialFieldBuffers(
            conservatives, primitives, temperature)
        
        return material_fields

    def from_cavitation_initial_condition(
            self,
            cell_centers: List
            ) -> MaterialFieldBuffers:
        """Initializes the material field
        buffers from cavitation initial conditions.

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

        fill_edge_halos = self.halo_manager.fill_edge_halos_material
        fill_vertex_halos = self.halo_manager.fill_vertex_halos_material

        dtype = np.float64 if self.is_double_precision else np.float32

        mesh_grid = self.domain_information.compute_device_mesh_grid()

        primitives_init = self.cavitation_init_condition.get_cavitation_initial_condition(
            mesh_grid)
        primitives = create_field_buffer(nh, device_number_of_cells, dtype, no_primes)
        primitives = primitives.at[...,nhx,nhy,nhz].set(primitives_init)
        conservatives = self.equation_manager.get_conservatives_from_primitives(primitives)
        primitives, conservatives = self.halo_manager.perform_halo_update_material(
            primitives, 0.0, fill_edge_halos, 
            fill_vertex_halos, conservatives)

        if self.equation_information.is_compute_temperature:
            temperature = self.material_manager.get_temperature(primitives)
            temperature = self.halo_manager.perform_outer_halo_update_temperature(
                temperature, 0.0)
        else:
            temperature = None

        material_fields = MaterialFieldBuffers(
            conservatives, primitives,temperature)
        return material_fields

    def from_primitive_initial_condition(
            self,
            cell_centers: List,
            ml_setup: MachineLearningSetup = None
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
                    s_mass = self.equation_information.s_mass
                    s_volume_fraction = self.equation_information.s_volume_fraction
                    vf_full_vector = jnp.concatenate([
                        primes_init[s_volume_fraction],
                        1.0 - jnp.sum(primes_init[s_volume_fraction], axis=0, keepdims=True)
                    ], axis=0)
                    primes_init = primes_init.at[s_mass].mul(vf_full_vector)

            else:
                pass

            primitives = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim)
            primitives = primitives.at[...,nhx,nhy,nhz].set(primes_init)

        conservatives = self.equation_manager.get_conservatives_from_primitives(primitives)

        fill_edge_halos = self.halo_manager.fill_edge_halos_material
        fill_vertex_halos = self.halo_manager.fill_vertex_halos_material

        primitives, conservatives = self.halo_manager.perform_halo_update_material(
            primitives, 0.0, fill_edge_halos,
            fill_vertex_halos, conservatives,
            ml_setup=ml_setup
        )
        
        if self.equation_information.is_compute_temperature:
            temperature = self.material_manager.get_temperature(primitives)
            temperature = self.halo_manager.perform_outer_halo_update_temperature(
                temperature, 0.0)
        else:
            temperature = None

        material_fields = MaterialFieldBuffers(
            conservatives, primitives, temperature)
        
        return material_fields
    