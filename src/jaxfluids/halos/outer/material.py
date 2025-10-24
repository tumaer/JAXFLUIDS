import types
from typing import Callable, Union, Dict, List, Tuple, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition, get_signs_symmetry
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace, \
    VelocityCallable, WallMassTransferSetup, PrimitivesTable
from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.halos.outer.helper_function import get_derivative_stencils_linear_extrapolation
from jaxfluids.domain import EDGE_LOCATIONS, VERTEX_LOCATIONS, AXES, FACE_LOCATIONS

Array = jax.Array

class BoundaryConditionMaterial(BoundaryCondition):
    """ The BoundaryConditionMaterial class implements functionality
    to enforce user-specified outer boundary conditions for the material fields.
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            boundary_conditions: BoundaryConditionsField
            ) -> None:

        super().__init__(domain_information, boundary_conditions)

        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.material_manager = self.equation_manager.material_manager

        # UPWIND DIRECTION FOR NEUMANN DERIVATIVE STENCIL
        self.upwind_difference_sign = {
            "east"  : -1, "west"  :  1,
            "north" : -1, "south" :  1,
            "top"   : -1, "bottom":  1 }

        no_primes = self.equation_information.no_primes
        equation_type = self.equation_information.equation_type
        vel_indices = self.equation_information.ids_velocity
        self.face_signs_symmetry, self.edge_signs_symmetry, \
        self.vertex_signs_symmetry = get_signs_symmetry(
            no_primes, equation_type, vel_indices)

        active_face_locations = self.domain_information.active_face_locations

        # LINEAR EXTRAPOLATION BOUNDARY CONDITION
        self.is_linear_extrapolation = False
        for face_location in active_face_locations:
            boundary_conditions_face_tuple: Tuple[BoundaryConditionsFace] \
                = getattr(boundary_conditions, face_location)

            for i, boundary_conditions_face in enumerate(boundary_conditions_face_tuple):
                boundary_type = boundary_conditions_face.boundary_type

                if boundary_type == "LINEAREXTRAPOLATION":
                    self.is_linear_extrapolation = True

        if self.is_linear_extrapolation:
            self.derivative_upwind: SpatialDerivative = None
            self.derivative_downwind: SpatialDerivative = None
            
 
        # INITIALIZE OPPOSITION CONTROL
        self.opposition_control_data = {}
        for face_location in active_face_locations:
            boundary_conditions_face_tuple: Tuple[BoundaryConditionsFace] \
                = getattr(boundary_conditions, face_location)
            for i, boundary_conditions_face in enumerate(boundary_conditions_face_tuple):            
                boundary_type = boundary_conditions_face.boundary_type
                if boundary_type == "OPPOSITIONCONTROLWALL":
                    assert len(boundary_conditions_face_tuple) == 1
                    assert face_location in ("south", "north")

                    opposition_control_setup = boundary_conditions_face.opposition_control
                    distance_sensing_plane = opposition_control_setup.distance_sensing_plane
                    amplitude = opposition_control_setup.amplitude

                    y_cc = np.squeeze(self.domain_information.get_global_cell_centers()[1])
                    wall_location = self.domain_information.domain_size[1][0 if face_location == "south" else 1]
                    wall_normal_distance = y_cc - wall_location
                    sign = -1 if face_location == "south" else 1
                    index_sensing_plane = np.argmin(np.abs(wall_normal_distance + sign * distance_sensing_plane), axis=-1)
                    self.opposition_control_data[face_location] = {"idx": index_sensing_plane, "A": amplitude}

    def face_halo_update(
            self,
            primitives: Array,
            physical_simulation_time: float,
            conservatives: Array = None,
            ml_setup: MachineLearningSetup = None
        ) -> Tuple[Array, Array]:
        """Fills the face halo cells of the primitive and
        variable buffer. If conservatives is passed,
        then the corresponding conservatives are
        also computed and returned. 

        :param primitives: _description_
        :type primitives: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param conservatives: _description_, defaults to None
        :type conservatives: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        if conservatives is not None:
            compute_conservatives = True
        else:
            compute_conservatives = False

        def update_loop(
                primitives: Array, physical_simulation_time: 
                float, conservatives: Array = None,
            ) -> Tuple[Array, Array]:
            
            active_face_locations = self.domain_information.active_face_locations
            for face_location in active_face_locations:

                boundary_conditions_face_tuple: Tuple[BoundaryConditionsFace] = \
                getattr(self.boundary_conditions, face_location)

                if len(boundary_conditions_face_tuple) > 1:
                    multiple_types_at_face = True
                else:
                    multiple_types_at_face = False

                for bc_id, boundary_conditions_face in enumerate(boundary_conditions_face_tuple):

                    boundary_type = boundary_conditions_face.boundary_type
                    

                    if boundary_type in ["SYMMETRY", "PERIODIC", "ZEROGRADIENT"]:
                        halos_primes = self.miscellaneous(
                            primitives, boundary_type, face_location)
                        
                    elif boundary_type == "LINEAREXTRAPOLATION":
                        halos_primes = self.linear_extrapolation(
                            primitives, boundary_type, face_location
                        )

                    elif boundary_type in ["ISOTHERMALWALL", "WALL"]:
                        wall_velocity_callable = boundary_conditions_face.wall_velocity_callable
                        halos_primes = self.wall(
                            primitives, face_location,
                            wall_velocity_callable,
                            physical_simulation_time)

                    elif boundary_type in ["ISOTHERMALMASSTRANSFERWALL", "MASSTRANSFERWALL"]:
                        wall_velocity_callable = boundary_conditions_face.wall_velocity_callable
                        wall_mass_transfer = boundary_conditions_face.wall_mass_transfer
                        halos_primes = self.masstransferwall(
                            primitives, face_location, wall_velocity_callable,
                            wall_mass_transfer, physical_simulation_time)
                    
                    elif boundary_type == "MASSTRANSFERWALL_PARAMETERIZED":
                        wall_velocity_callable = boundary_conditions_face.wall_velocity_callable
                        bounding_domain_callable = boundary_conditions_face.wall_mass_transfer.bounding_domain_callable

                        wall_mass_transfer_callable = getattr(
                            ml_setup.callables.boundary_conditions.primitives,
                            face_location)[bc_id].wall_mass_transfer
                        
                        wall_mass_transfer_parameters = getattr(
                            ml_setup.parameters.boundary_conditions.primitives,
                            face_location)[bc_id].wall_mass_transfer
                                                    
                        halos_primes = self.masstransferwall_parameterized(
                            primitives, face_location,
                            wall_velocity_callable,
                            wall_mass_transfer_callable,
                            wall_mass_transfer_parameters,
                            bounding_domain_callable,
                            physical_simulation_time
                        )

                    elif boundary_type == "DIRICHLET":
                        primitives_callable = boundary_conditions_face.primitives_callable
                        primitives_table = boundary_conditions_face.primitives_table
                        halos_primes = self.dirichlet(
                            face_location, primitives_callable,
                            physical_simulation_time, primitives_table)
                    
                    elif boundary_type == "DIRICHLET_PARAMETERIZED":
                        primitives_callable = getattr(
                            ml_setup.callables.boundary_conditions.primitives,
                            face_location)[bc_id].primitives
                        primitives_parameters = getattr(
                            ml_setup.parameters.boundary_conditions.primitives,
                            face_location)[bc_id].primitives
                        
                        halos_primes = self.dirichlet_parameterized(
                            face_location, primitives_callable,
                            primitives_parameters, physical_simulation_time
                        )

                    elif boundary_type == "NEUMANN":
                        primitives_callable = boundary_conditions_face.primitives_callable
                        halos_primes = self.neumann(
                            primitives, face_location, primitives_callable,
                            physical_simulation_time)

                    elif boundary_type == "SIMPLE_INFLOW":
                        primitives_callable = boundary_conditions_face.primitives_callable
                        halos_primes = self.simple_inflow(
                            primitives, face_location, primitives_callable,
                            physical_simulation_time)

                    elif boundary_type == "SIMPLE_OUTFLOW":
                        primitives_callable = boundary_conditions_face.primitives_callable
                        halos_primes = self.simple_outflow(
                            primitives, face_location, primitives_callable,
                            physical_simulation_time)

                    elif boundary_type in ("OPPOSITIONCONTROLWALL", "ISOTHERMALOPPOSITIONCONTROLWALL"):
                        halos_primes = self.opposition_control(
                            primitives, conservatives, face_location
                        )

                    else:
                        raise NotImplementedError

                    if compute_conservatives:
                        halos_cons = self.equation_manager.get_conservatives_from_primitives(
                            halos_primes)
                    
                    if multiple_types_at_face:
                        meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
                            face_location)
                        bounding_domain_callable = boundary_conditions_face.bounding_domain_callable
                        bounding_domain_mask = bounding_domain_callable(*meshgrid)
                        for axis in axes_to_expand:
                            bounding_domain_mask = jnp.expand_dims(bounding_domain_mask, axis)
                    else:
                        bounding_domain_mask = 1.0

                    if self.domain_information.is_parallel:
                        device_id = jax.lax.axis_index(axis_name="i")
                        device_mask = self.face_halo_mask
                        device_mask = device_mask[face_location][device_id]
                        mask = bounding_domain_mask * device_mask
                    else:
                        mask = bounding_domain_mask

                    slices_fill = self.halo_slices.face_slices_conservatives[face_location]
                    primitives = primitives.at[slices_fill].mul(1 - mask)
                    primitives = primitives.at[slices_fill].add(halos_primes * mask)
                    if compute_conservatives:
                        conservatives = conservatives.at[slices_fill].mul(1 - mask)
                        conservatives = conservatives.at[slices_fill].add(halos_cons * mask)

            return primitives, conservatives
                
        primitives, conservatives = update_loop(
            primitives, physical_simulation_time, 
            conservatives
        )

        if compute_conservatives:
            return primitives, conservatives
        else:
            return primitives

    def edge_halo_update(
            self,
            primitives: Array,
            conservatives: Array = None,
            ) -> Tuple[Array, Array]:
        """Fills the edge halo cells of the
        conservative and primitive buffer.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        if conservatives != None:
            compute_conservatives = True
        else:
            compute_conservatives = False

        is_parallel = self.domain_information.is_parallel
        active_edge_locations = self.domain_information.active_edge_locations
        for edge_location in active_edge_locations:

            edge_boundary_types = self.edge_boundary_types[edge_location]

            halos_primes = self.compute_edge_halos(
                primitives, edge_location, edge_boundary_types)
            if compute_conservatives:
                halos_cons = self.equation_manager.get_conservatives_from_primitives(halos_primes)

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.edge_halo_mask[edge_location][device_id]
            else:
                mask = 1

            slices_fill = self.halo_slices.edge_slices_conservatives[edge_location]
            primitives = primitives.at[slices_fill].mul(1 - mask)
            primitives = primitives.at[slices_fill].add(halos_primes * mask)
            if compute_conservatives:
                conservatives = conservatives.at[slices_fill].mul(1 - mask)
                conservatives = conservatives.at[slices_fill].add(halos_cons * mask)

        if compute_conservatives:
            return primitives, conservatives
        else:
            return primitives

    def compute_edge_halos(
            self,
            primitives: Array,
            edge_location: str, 
            edge_boundary_types: str
            ) -> Array:
        """Computes the edge halo cells for the
        specified edge location and boundary 
        types.

        :param primitives: _description_
        :type primitives: Array
        :param edge_location: _description_
        :type edge_location: str
        :param edge_boundary_types: _description_
        :type edge_boundary_types: str
        :return: _description_
        :rtype: Array
        """

        edge_slices = self.halo_slices.edge_slices_conservatives

        if edge_boundary_types == "ANY_ANY":
            location_retrieve_1 = edge_location + "_10"
            location_retrieve_2 = edge_location + "_01"
            slice_retrieve_1 = edge_slices[location_retrieve_1]
            slice_retrieve_2 = edge_slices[location_retrieve_2]
            halos_primes = 0.5 * (primitives[slice_retrieve_1] + primitives[slice_retrieve_2])
            
        else:
            location_retrieve = self.edge_types_to_location_retrieve[edge_location][edge_boundary_types]
            slice_retrieve = edge_slices[location_retrieve]
            halos_primes = primitives[slice_retrieve]
            if "SYMMETRY" in edge_boundary_types:
                s_ = self.edge_flip_slices_symmetry[edge_location][edge_boundary_types]
                signs = self.edge_signs_symmetry[edge_location][edge_boundary_types]
                if signs.shape[0] == halos_primes.shape[0]:
                    halos_primes = halos_primes[s_]
                    halos_primes *= signs

        return halos_primes

    def vertex_halo_update(
            self,
            primitives: Array,
            conservatives: Array = None,
            ) -> Tuple[Array]:
        """Updates the vertex halos
        of the material field buffers.

        :param primitives: _description_
        :type primitives: Array
        :param conservatives: _description_, defaults to None
        :type conservatives: Array, optional
        :return: _description_
        :rtype: Tuple[Array]
        """
        
        if conservatives != None:
            compute_conservatives = True
        else:
            compute_conservatives = False

        is_parallel = self.domain_information.is_parallel
        for vertex_location in VERTEX_LOCATIONS:

            vertex_boundary_types = self.vertex_boundary_types[vertex_location]

            halos_primes = self.compute_vertex_halos(
                primitives, vertex_location, vertex_boundary_types)

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.vertex_halo_mask[vertex_location][device_id]
            else:
                mask = 1

            slices_fill = self.halo_slices.vertex_slices_conservatives[vertex_location]
            primitives = primitives.at[slices_fill].mul(1 - mask)
            primitives = primitives.at[slices_fill].add(halos_primes * mask)
            
            if compute_conservatives:
                halos_cons = self.equation_manager.get_conservatives_from_primitives(halos_primes)
                conservatives = conservatives.at[slices_fill].mul(1 - mask)
                conservatives = conservatives.at[slices_fill].add(halos_cons * mask)

        if compute_conservatives:
            return primitives, conservatives
        else:
            return primitives

    def compute_vertex_halos(
            self,
            primitives: Array,
            vertex_location: str, 
            vertex_boundary_types: str
            ) -> Array:
        """Computes the vertex halos
        for the specified vertex location
        and boundary types.

        :param primitives: _description_
        :type primitives: Array
        :param vertex_location: _description_
        :type vertex_location: str
        :param vertex_boundary_types: _description_
        :type vertex_boundary_types: str
        :return: _description_
        :rtype: Array
        """
        vertex_slices = self.halo_slices.vertex_slices_conservatives

        if vertex_boundary_types == "ANY_ANY_ANY":
            location_retrieve_1 = vertex_location + "_100"
            location_retrieve_2 = vertex_location + "_010"
            location_retrieve_3 = vertex_location + "_001"
            slice_retrieve_1 = vertex_slices[location_retrieve_1]
            slice_retrieve_2 = vertex_slices[location_retrieve_2]
            slice_retrieve_3 = vertex_slices[location_retrieve_3]
            halos_primes = 1.0/3.0 * (primitives[slice_retrieve_1] + primitives[slice_retrieve_2]
                                      + primitives[slice_retrieve_3])
        else:
            location_retrieve = self.vertex_types_to_location_retrieve[vertex_location][vertex_boundary_types]
            slice_retrieve = vertex_slices[location_retrieve]
            halos_primes = primitives[slice_retrieve]
            if "SYMMETRY" in vertex_boundary_types:
                s_ = self.vertex_flip_slices_symmetry[vertex_location][vertex_boundary_types]
                signs = self.vertex_signs_symmetry[vertex_location][vertex_boundary_types]
                if signs.shape[0] == halos_primes.shape[0]:
                    halos_primes = halos_primes[s_]
                    halos_primes *= signs

        return halos_primes

    def wall(
            self, 
            primitives: Array,
            face_location: str, 
            wall_velocity_callable: VelocityCallable,
            physical_simulation_time: float, 
            ) -> Array:
        """Computes the primitive halos
        for wall boundaries.

        :param primitives: _description_
        :type primitives: Array
        :param face_location: _description_
        :type face_location: str
        :param wall_velocity_callable: _description_
        :type wall_velocity_callable: VelocityCallable
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """

        meshgrid, axes_to_expand = \
        self.get_boundary_coordinates_at_location(
            face_location)

        wall_velocity_list = []
        for i, velocity in enumerate(wall_velocity_callable._fields):
            wall_velocity_xi_callable: Callable = getattr(wall_velocity_callable, velocity)
            wall_velocity = wall_velocity_xi_callable(*meshgrid, physical_simulation_time)
            for axis_index in axes_to_expand:
                wall_velocity = jnp.expand_dims(wall_velocity, axis_index)
            wall_velocity_list.append(wall_velocity)

        vel_slices = self.equation_information.s_velocity
        s_mass = self.equation_information.s_mass
        s_energy = self.equation_information.s_energy

        velocity = primitives[vel_slices]
        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]
        u_halo = 2 * wall_velocity_list[0] - velocity[(jnp.s_[0:1],) + slices_retrieve]
        v_halo = 2 * wall_velocity_list[1] - velocity[(jnp.s_[1:2],) + slices_retrieve]
        w_halo = 2 * wall_velocity_list[2] - velocity[(jnp.s_[2:3],) + slices_retrieve]

        halos_primes = jnp.concatenate([
            primitives[(s_mass,) + slices_retrieve],
            u_halo,
            v_halo,
            w_halo,
            primitives[(s_energy,) + slices_retrieve]
        ], axis=0)

        diffuse_interface_model = self.equation_information.diffuse_interface_model
        if diffuse_interface_model:
            s_volume_fraction = self.equation_information.s_volume_fraction
            halos_primes = jnp.concatenate([
                halos_primes,
                primitives[(s_volume_fraction,) + slices_retrieve]
            ], axis=0)

        return halos_primes

    def masstransferwall(
            self, 
            primitives: Array,
            face_location: str, 
            wall_velocity_callable: VelocityCallable,
            wall_mass_transfer: WallMassTransferSetup,
            physical_simulation_time: float, 
            ) -> Array:
        """Computes the primitive halos for 
        wall boundaries with mass transfer.
        Within the mass transfer region,
        density and velocity is specified
        at the wall boundary while
        pressure is constantly extrapolated
        from the domain.

        :param primitives: _description_
        :type primitives: Array
        :param face_location: _description_
        :type face_location: str
        :param velocity_functions: _description_
        :type velocity_functions: Dict
        :param mass_transfer_functions: _description_
        :type mass_transfer_functions: Dict
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """

        levelset_model = self.equation_information.levelset_model
        diffuse_interface_model = self.equation_information.diffuse_interface_model

        halos_primes_wall = self.wall(
            primitives, face_location, wall_velocity_callable,
            physical_simulation_time)
        
        meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
            face_location)

        halos_primes_list = []
        primes_tuple = self.equation_information.primes_tuple
        primes_tuple = [state for state in primes_tuple if state != "p"]
        primitives_callable = wall_mass_transfer.primitives_callable
        for prime_state in primes_tuple:
            prime_state_callable = getattr(primitives_callable, prime_state)
            halos = prime_state_callable(*meshgrid, physical_simulation_time)
            for axis in axes_to_expand:
                halos = jnp.expand_dims(halos, axis)
            halos_primes_list.append(halos)
        halos_primes = jnp.stack(halos_primes_list, axis=0) 

        if levelset_model == "FLUID-FLUID":
            halos_primes = jnp.stack([ # TODO INTRODUCE MASS TRANSFER WALL FOR BOTH NEGATIVE AND POSITIVE
                halos_primes, halos_primes], axis=1)

        face_location_to_axis_index = self.domain_information.face_location_to_axis_index
        axis_index = face_location_to_axis_index[face_location]
        nh = self.domain_information.nh_conservatives
        no_fluids = self.equation_information.no_fluids

        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]

        s_energy = self.equation_information.s_energy
        s_mass = self.equation_information.s_mass
        vel_slices = self.equation_information.s_velocity
        s_vel = (vel_slices,) + slices_retrieve

        halos_mass = halos_primes[s_mass]
        halos_mass = jnp.repeat(halos_mass, nh, axis=-3+axis_index)
        halos_velocity = halos_primes[vel_slices]
        halos_velocity = 2 * halos_velocity - primitives[s_vel]
        if diffuse_interface_model:
            halos_vf = halos_primes[-no_fluids+1:]
            halos_vf = jnp.repeat(halos_vf, nh, axis=-3+axis_index)
        slices_retrieve = self.face_slices_retrieve_conservatives["ZEROGRADIENT"][face_location]
        halos_pressure = primitives[(s_energy,)+slices_retrieve]
        halos_pressure = jnp.repeat(halos_pressure, nh, axis=-3+axis_index)
        
        if diffuse_interface_model:
            halos_primes = jnp.concatenate([
                halos_mass,
                halos_velocity,
                halos_pressure,
                halos_vf
            ], axis=0)
        else:
            halos_primes = jnp.concatenate([
                halos_mass,
                halos_velocity,
                halos_pressure
            ], axis=0)
        
        bounding_domain_callable = wall_mass_transfer.bounding_domain_callable
        mask = bounding_domain_callable(*meshgrid)
        for axis in axes_to_expand:
            mask = jnp.expand_dims(mask, axis)
        halos_primes = halos_primes * mask + (1 - mask) * halos_primes_wall
        return halos_primes

    def masstransferwall_parameterized(
            self, 
            primitives: Array,
            face_location: str, 
            wall_velocity_callable: VelocityCallable,
            wall_mass_transfer_callable: Callable,
            wall_mass_transfer_parameters: Callable,
            bounding_domain_callable: Callable,
            physical_simulation_time: float, 
        ) -> Array:
        """Computes the primitive halos for 
        wall boundaries with mass transfer.
        Within the mass transfer region,
        density and velocity is specified
        at the wall boundary while
        pressure is constantly extrapolated
        from the domain.

        :param primitives: _description_
        :type primitives: Array
        :param face_location: _description_
        :type face_location: str
        :param velocity_functions: _description_
        :type velocity_functions: Dict
        :param mass_transfer_functions: _description_
        :type mass_transfer_functions: Dict
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """

        levelset_model = self.equation_information.levelset_model
        diffuse_interface_model = self.equation_information.diffuse_interface_model

        # NOTE compute halo cells for section of the wall
        # where no mass transfer occurs
        halos_primes_wall = self.wall(
            primitives, face_location, wall_velocity_callable,
            physical_simulation_time)
        
        # NOTE compute halo cells for section of the wall
        # where mass transfer occurs
        meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
            face_location)

        halos_primes_list = []
        primes_tuple = self.equation_information.primes_tuple
        primes_tuple = [state for state in primes_tuple if state != "p"]
        for prime_state in primes_tuple:
            prime_state_callable = getattr(wall_mass_transfer_callable, prime_state)
            prime_state_parameters = getattr(wall_mass_transfer_parameters, prime_state)
            halos = prime_state_callable(*meshgrid, physical_simulation_time, prime_state_parameters)
            for axis in axes_to_expand:
                halos = jnp.expand_dims(halos, axis)
            halos_primes_list.append(halos)
        halos_primes = jnp.stack(halos_primes_list, axis=0) 

        if levelset_model == "FLUID-FLUID":
            halos_primes = jnp.stack([ # TODO INTRODUCE MASS TRANSFER WALL FOR BOTH NEGATIVE AND POSITIVE
                halos_primes, halos_primes], axis=1)

        face_location_to_axis_index = self.domain_information.face_location_to_axis_index
        axis_index = face_location_to_axis_index[face_location]
        nh = self.domain_information.nh_conservatives
        no_fluids = self.equation_information.no_fluids

        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]

        s_energy = self.equation_information.s_energy
        s_mass = self.equation_information.s_mass
        vel_slices = self.equation_information.s_velocity
        s_vel = (vel_slices,) + slices_retrieve

        halos_mass = halos_primes[s_mass]
        halos_mass = jnp.repeat(halos_mass, nh, axis=-3+axis_index)
        halos_velocity = halos_primes[vel_slices]
        halos_velocity = 2 * halos_velocity - primitives[s_vel]
        if diffuse_interface_model:
            halos_vf = halos_primes[-no_fluids+1:]
            halos_vf = jnp.repeat(halos_vf, nh, axis=-3+axis_index)
        slices_retrieve = self.face_slices_retrieve_conservatives["ZEROGRADIENT"][face_location]
        halos_pressure = primitives[(s_energy,)+slices_retrieve]
        halos_pressure = jnp.repeat(halos_pressure, nh, axis=-3+axis_index)
        
        halos_primes = [halos_mass, halos_velocity, halos_pressure]   
        if diffuse_interface_model:
            halos_primes += halos_vf
        halos_primes = jnp.concatenate(halos_primes, axis=0)
        
        # NOTE combine halo cells w/o mass transfer with halo cells
        # w/ mass transfer
        mask = bounding_domain_callable(*meshgrid)
        for axis in axes_to_expand:
            mask = jnp.expand_dims(mask, axis)
        halos_primes = halos_primes * mask + (1 - mask) * halos_primes_wall

        return halos_primes

    def dirichlet(
            self,
            face_location: str,
            primitives_callable: NamedTuple,
            physical_simulation_time: float,
            primitives_table: PrimitivesTable
        ) -> Array:
        """Computes the halo cells from
        a DIRICHLET condition.

        :param face_location: _description_
        :type face_location: str
        :param primitives_callable: _description_
        :type primitives_callable: NamedTuple
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """

        levelset_model = self.equation_information.levelset_model

        if primitives_table != None:

            if levelset_model == "FLUID-FLUID":
                raise NotImplementedError
            
            no_primes = self.equation_information.no_primes
            axis_to_axis_id = self.domain_information.axis_to_axis_id
            cell_centers = self.domain_information.get_device_cell_centers()

            primitives_values_table = primitives_table.primitives
            if no_primes != len(primitives_values_table):
                raise RuntimeError
            
            axis_name_table = primitives_table.axis_name
            axis_values_table = primitives_table.axis_values

            axis_index = axis_to_axis_id[axis_name_table]
            cell_centers_xi = cell_centers[axis_index]

            halos_list = []
            for i in range(no_primes):
                left_value = primitives_values_table[i,0]
                right_value = primitives_values_table[i,-1]
                halos = jnp.interp(cell_centers_xi, axis_values_table,
                                   primitives_values_table[i], left_value, right_value)
                halos_list.append(halos)
            halos_primes = jnp.stack(halos_list)

        else:
            meshgrid, axes_to_expand = \
            self.get_boundary_coordinates_at_location(
                face_location)
            halos_primes_list = []
            for prime_state in primitives_callable._fields:
                prime_callable: Callable = getattr(primitives_callable, prime_state)
                halos = prime_callable(*meshgrid, physical_simulation_time)
                # TODO we should also allow scalar return values of the boundary callable
                for axis in axes_to_expand:
                    halos = jnp.expand_dims(halos, axis)
                halos_primes_list.append(halos)
            halos_primes = jnp.stack(halos_primes_list, axis=0)
            if levelset_model == "FLUID-FLUID": # TODO INTRODUCE DIRICHLET BOUNDARIES FOR BOTH POSITIVE AND NEGATIVE
                halos_primes = jnp.stack([halos_primes, halos_primes], axis=1)
    
        return halos_primes

    def dirichlet_parameterized(
            self,
            face_location: str,
            primitives_callable: NamedTuple,
            primitives_parameters: NamedTuple,
            physical_simulation_time: float,
    ) -> Array:

        levelset_model = self.equation_information.levelset_model

        meshgrid, axes_to_expand = \
        self.get_boundary_coordinates_at_location(
            face_location)
        halos_primes_list = []
        for prime_state in primitives_callable._fields:
            prime_callable: Callable = getattr(primitives_callable, prime_state)
            prime_parameters = getattr(primitives_parameters, prime_state)
            halos = prime_callable(*meshgrid, physical_simulation_time, prime_parameters)
            for axis in axes_to_expand:
                halos = jnp.expand_dims(halos, axis)
            halos_primes_list.append(halos)
        halos_primes = jnp.stack(halos_primes_list, axis=0)
        if levelset_model == "FLUID-FLUID": # TODO INTRODUCE DIRICHLET BOUNDARIES FOR BOTH POSITIVE AND NEGATIVE
            halos_primes = jnp.stack([halos_primes, halos_primes], axis=1)
    
        return halos_primes

    def neumann(
            self,
            primitives: Array,
            face_location: str,
            primitives_callable: NamedTuple,
            physical_simulation_time: float
            ) -> Array:
        """Computes primitive halos for NEUMANN
        boundary conditions. 

        :param primitives: _description_
        :type primitives: Array
        :param face_location: _description_
        :type face_location: str
        :param functions: _description_
        :type functions: Union[Callable, float]
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param slice_retrieve: _description_
        :type slice_retrieve: Tuple
        :return: _description_
        :rtype: Array
        """
        
        meshgrid, axes_to_expand = \
        self.get_boundary_coordinates_at_location(
            face_location)

        slices_retrieve = self.face_slices_retrieve_conservatives["NEUMANN"][face_location]
        dx = self.get_cell_size_at_face(face_location)

        halos_primes_list = []
        for i, prime_state in enumerate(primitives_callable._fields):
            prime_callable: Callable = getattr(primitives_callable, prime_state)
            neumann_value = prime_callable(*meshgrid, physical_simulation_time)
            for axis in axes_to_expand:
                neumann_value = jnp.expand_dims(neumann_value, axis)
            neumann_value *= self.upwind_difference_sign[face_location]
            neumann_value *= dx
            halos = primitives[(i,) + slices_retrieve] + neumann_value 
            halos_primes_list.append(halos)
        halos_primes = jnp.stack(halos_primes_list, axis=0)

        return halos_primes

    def miscellaneous(
            self,
            primitives: Array,
            boundary_type: str,
            face_location: str,
            )-> Array:
        """Computes the primitive halo cells
        for PERIODIC, ZEROGRADIENT or 
        SYMMETRY boundary conditions

        :param primitives: _description_
        :type primitives: Array
        :param boundary_type: _description_
        :type boundary_type: str
        :param face_location: _description_
        :type face_location: str
        :return: _description_
        :rtype: Array
        """

        slices_retrieve = self.face_slices_retrieve_conservatives[boundary_type][face_location]
        halos_primes = primitives[slices_retrieve]

        if boundary_type == "SYMMETRY":
            halos_primes *= self.face_signs_symmetry[face_location]

        return halos_primes

    def opposition_control(
            self,
            primitives: Array,
            conservatives: Array,
            face_location: str, 
        ):

        vel_slices = self.equation_information.s_velocity
        s_mass = self.equation_information.s_mass
        s_energy = self.equation_information.s_energy
        ids_velocity = self.equation_information.ids_velocity

        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]
        wall_normal_index = self.domain_information.face_location_to_axis_index[face_location]
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives

        idx_sensing_plane = self.opposition_control_data[face_location]["idx"]
        amplitude = self.opposition_control_data[face_location]["A"]

        conservatives = conservatives[..., nhx, nhy, nhz]
        mass_flux = conservatives[ids_velocity[wall_normal_index], :, idx_sensing_plane:idx_sensing_plane+1, :]
        mass_flux = -amplitude * mass_flux
        mean_mass_flux = jnp.mean(mass_flux, axis=(-1,-3), keepdims=True)
        if self.domain_information.is_parallel:
            mean_mass_flux = jax.lax.pmean(mean_mass_flux, axis_name="i")

        slices_wall_adjacent = self.face_slices_retrieve_conservatives["ZEROGRADIENT"][face_location]
        density_wall = self.material_manager.get_density(primitives[slices_wall_adjacent])
        wall_normal_velocity = (mass_flux - mean_mass_flux) / density_wall

        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]
        velocity = primitives[vel_slices]
        u_halo = - velocity[(jnp.s_[0:1],) + slices_retrieve]
        v_halo = 2 * wall_normal_velocity - velocity[(jnp.s_[1:2],) + slices_retrieve]
        w_halo = - velocity[(jnp.s_[2:3],) + slices_retrieve]

        halos_primes = jnp.concatenate([
            primitives[(s_mass,) + slices_retrieve],
            u_halo,
            v_halo,
            w_halo,
            primitives[(s_energy,) + slices_retrieve]
        ], axis=0)

        diffuse_interface_model = self.equation_information.diffuse_interface_model
        if diffuse_interface_model:
            s_volume_fraction = self.equation_information.s_volume_fraction
            halos_primes = jnp.concatenate([
                halos_primes,
                primitives[(s_volume_fraction,) + slices_retrieve]
            ], axis=0)

        return halos_primes


    def simple_inflow(
            self,
            primitives: Array,
            face_location: str,
            primitives_callable: NamedTuple,
            physical_simulation_time: float, 
        ) -> Array:

        levelset_model = self.equation_information.levelset_model
        diffuse_interface_model = self.equation_information.diffuse_interface_model

        s_mass = self.equation_information.s_mass
        vel_slices = self.equation_information.s_velocity
        s_energy = self.equation_information.s_energy
        s_volume_fraction = self.equation_information.s_volume_fraction
        no_fluids = self.equation_information.no_fluids

        meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
            face_location)

        # Retrieve pressure from inside the domain
        slices_retrieve = self.face_slices_retrieve_conservatives["ZEROGRADIENT"][face_location]
        halos_pressure = primitives[(s_energy,) + slices_retrieve]

        # Evaluate inflow callables, i.e., 
        # density and velocity (and volume fraction)
        halos_primes_list = []
        primes_tuple = self.equation_information.primes_tuple
        primes_tuple = [state for state in primes_tuple if state != "p"]
        for prime_state in primes_tuple:
            prime_state_callable = getattr(primitives_callable, prime_state)
            halos = prime_state_callable(*meshgrid, physical_simulation_time)
            for axis in axes_to_expand:
                halos = jnp.expand_dims(halos, axis)
            halos_primes_list.append(halos)
        halos_primes = jnp.stack(halos_primes_list, axis=0) 

        if levelset_model == "FLUID-FLUID":
            halos_primes = jnp.stack([halos_primes, halos_primes], axis=1)

        # Concatenate inflow callables with pressure
        if diffuse_interface_model:
            halos_primes = jnp.concatenate([
                halos_primes[s_mass],
                halos_primes[vel_slices],
                halos_pressure,
                halos_primes[-no_fluids+1:]
            ], axis=0)
        else:
            halos_primes = jnp.concatenate([
                halos_primes[s_mass],
                halos_primes[vel_slices],
                halos_pressure
            ], axis=0)

        return halos_primes


    def simple_outflow(
            self,
            primitives: Array,
            face_location: str,
            primitives_callable: NamedTuple,
            physical_simulation_time: float, 
        ) -> Array:

        s_energy = self.equation_information.s_energy

        meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(face_location)

        # Retrieve all variables from inside the domain
        slices_retrieve = self.face_slices_retrieve_conservatives["ZEROGRADIENT"][face_location]
        halos_primes = primitives[slices_retrieve]

        # Evaluate pressure callable
        pressure_callable = getattr(primitives_callable, "p")
        halos_pressure = pressure_callable(*meshgrid, physical_simulation_time)
        for axis in axes_to_expand:
            halos_pressure = jnp.expand_dims(halos_pressure, axis)
        
        # Overwrite pressure with pressure callable
        halos_primes = halos_primes.at[s_energy].set(halos_pressure)

        return halos_primes


    def linear_extrapolation(
            self,
            primitives: Array,
            boundary_type: str,
            face_location: str,
            ) -> Array:
        
        axis = self.domain_information.face_location_to_axis_index[face_location]
        cell_sizes = self.domain_information.get_device_cell_sizes()
        cell_centers_halos = self.domain_information.get_device_cell_centers_halos()[axis]

        slices_retrieve, slices_retrieve_deriv, slice_cc, slice_cc_halos \
            = self.face_slices_retrieve_conservatives[boundary_type][face_location]

        if face_location in ("east", "north", "top"):
            dW_dx = self.derivative_upwind.derivative_xi(primitives, cell_sizes[axis], axis)
        elif face_location in ("west", "south", "bottom"):
            dW_dx = self.derivative_downwind.derivative_xi(primitives, cell_sizes[axis], axis)
        else:
            raise NotImplementedError

        dW_dx = dW_dx[slices_retrieve_deriv]
        dx = cell_centers_halos[slice_cc_halos] - cell_centers_halos[slice_cc]
        halos_primes = primitives[slices_retrieve] + dx * dW_dx

        return halos_primes

    # NOTE @aaron introduce temperature boundary condition class ?
    def face_halo_update_temperature(
            self,
            temperature: Array,
            physical_simulation_time: float,
            ) -> Array:
        """Fills the face halos of the temperature
        buffer.

        :param temperature: _description_
        :type temperature: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """

        is_parallel = self.domain_information.is_parallel
        active_face_locations = self.domain_information.active_face_locations
        for face_location in active_face_locations:

            boundary_conditions_face_tuple: Tuple[BoundaryConditionsFace] = \
            getattr(self.boundary_conditions, face_location)
            if len(boundary_conditions_face_tuple) > 1:
                multiple_types_at_face = True
            else:
                multiple_types_at_face = False

            for boundary_conditions_face in boundary_conditions_face_tuple:

                boundary_type = boundary_conditions_face.boundary_type
                if boundary_type in ["ISOTHERMALWALL", "ISOTHERMALMASSTRANSFERWALL"]:
                    wall_temperature_callable = boundary_conditions_face.wall_temperature_callable
                    halos = self.wall_temperature(
                        temperature, face_location,
                        wall_temperature_callable,
                        physical_simulation_time)

                else:
                    continue

                if multiple_types_at_face:
                    meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
                        face_location)
                    bounding_domain_callable = boundary_conditions_face.bounding_domain_callable
                    bounding_domain_mask = bounding_domain_callable(*meshgrid)
                    for axis in axes_to_expand:
                        bounding_domain_mask = jnp.expand_dims(bounding_domain_mask, axis)
                else:
                    bounding_domain_mask = 1.0

                slices_fill = self.halo_slices.face_slices_conservatives[face_location]

                if is_parallel:
                    device_id = jax.lax.axis_index(axis_name="i")
                    device_mask = self.face_halo_mask
                    device_mask = device_mask[face_location][device_id]
                    mask = bounding_domain_mask * device_mask
                else:
                    mask = bounding_domain_mask

                temperature = temperature.at[slices_fill].mul(1 - mask)
                temperature = temperature.at[slices_fill].add(halos * mask)

        return temperature
    
    def wall_temperature(
            self, 
            temperature: Array,
            face_location: str, 
            wall_temperature_callable: Callable,
            physical_simulation_time: float, 
            ) -> Array:
        """Computes the temperature halos for
        isothermal wall boundaries.

        :param temperature: _description_
        :type temperature: Array
        :param face_location: _description_
        :type face_location: str
        :param temperature_functions: _description_
        :type temperature_functions: Dict
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """

        meshgrid, axes_to_expand = \
        self.get_boundary_coordinates_at_location(
            face_location)
        
        wall_temperature = wall_temperature_callable(
            *meshgrid, physical_simulation_time)
        
        for axis in axes_to_expand:
            wall_temperature = jnp.expand_dims(wall_temperature, axis)

        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]
        halos_temperature = 2 * wall_temperature - temperature[slices_retrieve]

        return halos_temperature
    
    def get_cell_size_at_face(
            self,
            face_location
            ) -> float:
        """Gets the cell size at the present
        face location.

        :param face_location: _description_
        :type face_location: _type_
        :return: _description_
        :rtype: float
        """
        is_parallel = self.domain_information.is_parallel
        cell_sizes = self.domain_information.get_global_cell_sizes()
        indices = {
            "east": -1, "west": 0,
            "north": -1, "south": 0,
            "top": -1, "bottom": 0,
        }
        face_location_to_axis_index = self.domain_information.face_location_to_axis_index
        mesh_stretching = self.domain_information.is_mesh_stretching
        axis_index = face_location_to_axis_index[face_location]
        dx = cell_sizes[axis_index]
        if mesh_stretching[axis_index]:
            index = indices[face_location]
            dx = jnp.squeeze(dx)[index]
        return dx
    
    def set_stencils_linear_extrapolation(self) -> None:
        self.derivative_upwind, self.derivative_downwind \
            = get_derivative_stencils_linear_extrapolation(self.domain_information)
