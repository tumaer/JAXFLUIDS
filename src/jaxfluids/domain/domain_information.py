from typing import List, Tuple, Dict, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.helper_functions import flatten_subdomain_dimensions, \
    split_cell_centers_xi, split_cell_sizes_xi, split_subdomain_dimensions
from jaxfluids.domain.mesh_creation import *
from jaxfluids.data_types.case_setup.domain import \
    DomainSetup, AxisSetup, MeshStretchingSetup
from jaxfluids.domain import AXES, AXES_INDICES, FACE_LOCATIONS, \
    EDGE_LOCATIONS, VERTEX_LOCATIONS

class DomainInformation:
    """The DomainInformation class holds information
    about the computational domain, e.g., mesh, number of cells,
    extension in each spatial direction, active axis,
    domain slice objects etc..
    """
    # LOCATION TO AXIS TRANSFORMATIONS

    axis_to_axis_id = {"x": 0, "y": 1, "z": 2}
    axis_id_to_axis = {0: "x", 1: "y", 2: "z"}
    
    axis_to_velocity = {"x": "u", "y": "v", "z": "w"}
    velocity_to_axis = {"u": "x", "v": "y", "w": "z"}

    face_location_to_axis = {
        "east": "x", "west": "x",
        "north": "y", "south": "y",
        "top": "z", "bottom": "z"
    }
    face_location_to_axis_index = {
        "east": 0, "west": 0,
        "north": 1, "south": 1,
        "top": 2, "bottom": 2
    }
    axis_to_face_locations = {
        "x": ("west", "east"),
        "y": ("south", "north"),
        "z": ("bottom", "top")
    }
    face_location_to_axis_side = {
        "west": 0, "east": -1,
        "south": 0, "north": -1,
        "bottom": 0, "top": -1,
    }
    face_to_opposite_face = {
        "west": "east", "east": "west",
        "south": "north", "north": "south",
        "bottom": "top", "top": "bottom",
    }
    
    def __init__(
            self,
            domain_setup: DomainSetup,
            nh_conservatives: int,
            nh_geometry: int,
            ) -> None:

        # EDGE HALO SLICES
        nh = nh_conservatives
        nx = domain_setup.x.cells
        ny = domain_setup.y.cells
        nz = domain_setup.z.cells
        self.dim = sum(1 if n > 1 else 0 for n in [nx,ny,nz])
        global_domain_size = (
            np.array(domain_setup.x.range),
            np.array(domain_setup.y.range),
            np.array(domain_setup.z.range)
            )

        # GLOBAL MESH
        global_cell_centers = []
        global_cell_sizes = []
        global_cell_faces = []
        is_mesh_stretching = []
        global_number_of_cells = (nx,ny,nz)
        global_number_of_cell_faces = (nx+1,ny+1,nz+1)

        mesh_functions_dict = {
            "HOMOGENEOUS": homogeneous,
            "CHANNEL": channel,
            "BOUNDARY_LAYER": boundary_layer,
            "PIECEWISE": piecewise
        } 

        for axis in AXES:
            axis_setup: AxisSetup = getattr(domain_setup, axis)

        for axis_index, axis in enumerate(AXES):

            axis_setup: AxisSetup = getattr(domain_setup, axis)
            stretching_type = axis_setup.stretching.type
            is_mesh_stretching.append(stretching_type is not False)

            if stretching_type == False:
                mesh_type = "HOMOGENEOUS"
            else:
                mesh_type = stretching_type

            mesh_function: Callable = mesh_functions_dict[mesh_type]

            mesh_function_inputs = {
                "axis": axis_index,
                "nxi": global_number_of_cells[axis_index],
                "domain_size_xi": global_domain_size[axis_index],
                "stretching_setup": axis_setup.stretching,
            }

            cell_centers_xi, cell_faces_xi, cell_sizes_xi = mesh_function(
                **mesh_function_inputs)
            
            global_cell_centers.append( jnp.array(cell_centers_xi) )
            global_cell_sizes.append( jnp.array(cell_sizes_xi) )
            global_cell_faces.append( jnp.array(cell_faces_xi) )

        global_cell_centers = tuple(global_cell_centers)
        global_cell_sizes = tuple(global_cell_sizes)
        global_cell_faces = tuple(global_cell_faces)
        self.is_mesh_stretching = tuple(is_mesh_stretching)
        self.global_number_of_cells = tuple(global_number_of_cells)
        self.global_number_of_cell_faces = tuple(global_number_of_cell_faces)

        # ACTIVE AXIS AND LOCATIONS
        self.active_axes = domain_setup.active_axes
        self.inactive_axes = domain_setup.inactive_axes
        self.active_axes_indices = domain_setup.active_axes_indices
        self.inactive_axes_indices = domain_setup.inactive_axes_indices
        
        active_velocities = []
        for axis in self.active_axes:
            active_velocities.append(self.axis_to_velocity[axis])
        self.active_velocities = tuple(active_velocities)

        active_face_locations = []
        for face_location in FACE_LOCATIONS:
            if self.face_location_to_axis_index[face_location] \
                in self.active_axes_indices:
                active_face_locations.append(face_location)
        self.active_face_locations = tuple(active_face_locations)

        active_edge_locations = []
        for edge_location in EDGE_LOCATIONS:
            corner_location_list = edge_location.split("_")
            loc_1 = corner_location_list[0]
            loc_2 = corner_location_list[1]
            if self.face_location_to_axis_index[loc_1] in self.active_axes_indices and \
                self.face_location_to_axis_index[loc_2] in self.active_axes_indices:
                active_edge_locations.append(edge_location)
        self.active_edge_locations = tuple(active_edge_locations)

        self.smallest_cell_size = jnp.min(jnp.array([jnp.min(global_cell_sizes[i]) for i in self.active_axes_indices]))
        self.largest_cell_size = jnp.max(jnp.array([jnp.max(global_cell_sizes[i]) for i in self.active_axes_indices]))
        
        # DOMAIN SLICES
        self.nh_conservatives = nh_conservatives     
        self.domain_slices_conservatives = tuple(
            [jnp.s_[nh_conservatives:-nh_conservatives] if
            i in self.active_axes_indices else
            jnp.s_[:] for i in range(3)])
        if nh_geometry != None:
            self.nh_geometry = nh_geometry
            self.domain_slices_geometry = tuple(
                [jnp.s_[nh_geometry:-nh_geometry] if
                i in self.active_axes_indices else
                jnp.s_[:] for i in range(3)])

            self.nh_offset = nh_conservatives - nh_geometry
            self.domain_slices_conservatives_to_geometry = tuple(
                [jnp.s_[self.nh_offset:-self.nh_offset] if
                i in self.active_axes_indices else
                jnp.s_[:] for i in range(3)])

            nhx_geometry_1D = jnp.s_[:] if "x" in self.inactive_axes else jnp.s_[self.nh_offset:-self.nh_offset if self.nh_offset > 0 else None]    
            nhy_geometry_1D = jnp.s_[:] if "y" in self.inactive_axes else jnp.s_[self.nh_offset:-self.nh_offset if self.nh_offset > 0 else None]    
            nhz_geometry_1D = jnp.s_[:] if "z" in self.inactive_axes else jnp.s_[self.nh_offset:-self.nh_offset if self.nh_offset > 0 else None]
            self.domain_slices_geometry_1D = (
                jnp.s_[...,nhx_geometry_1D, :, :],
                jnp.s_[...,:, nhy_geometry_1D, :],
                jnp.s_[...,:, :, nhz_geometry_1D],)
        else:
            self.nh_geometry = None
            self.domain_slices_geometry = (None, None, None)
            self.domain_slices_conservatives_to_geometry = (None, None, None)

        # DOMAIN DECOMPOSITION
        self.split_factors = (domain_setup.decomposition.split_x,
                              domain_setup.decomposition.split_y,
                              domain_setup.decomposition.split_z,)
        self.no_subdomains = np.prod(np.array(self.split_factors))
        self.is_parallel = self.no_subdomains > 1

        self.global_device_count = jax.device_count()
        self.local_device_count = jax.local_device_count()
        self.is_multihost = self.global_device_count > self.local_device_count
        self.host_count = self.global_device_count // self.local_device_count
        self.process_id = jax.process_index()

        self.device_number_of_cells = tuple([int(n/s) for n, s in zip(global_number_of_cells, self.split_factors)])
        self.cells_per_device = int(np.prod(self.device_number_of_cells))
        self.subdomain_ids_flat = np.arange(self.no_subdomains, dtype=jnp.int32)
        self.subdomain_ids_grid = split_subdomain_dimensions(self.subdomain_ids_flat, self.split_factors)    

        # NOTE N_g global device count, N_l local device count, N_x device cells in x direction
        # NOTE global_cell_centers (N_g, N_x, 1, 1)
        # NOTE local_cell_centers (N_l, N_x, 1, 1)
        # NOTE global_cell_sizes mesh stretching (N_g, N_x, 1, 1), global_cell_sizes no mesh stretching (N_g, 1, 1, 1)
        # NOTE local_cell_sizes mesh stretching (N_l, N_x, 1, 1), global_cell_sizes no mesh stretching (N_l, 1, 1, 1)
        # NOTE global_cell_faces (N_g, N_x+1, 1, 1)
        # NOTE local_cell_faces (N_l, N_x+1, 1, 1)
        # NOTE global_domain_size (N_g, 2)
        # NOTE local_domain_size (N_l, 2)

        # NOTE local members are used as input for pmapped functions
        # NOTE global members are sliced from within a pmapped function using jax.lax.axis_index()
        # NOTE for single host runs, global and local members are equal
        # NOTE for single device runs, members have no leading dimension

        if self.is_parallel:
            global_cell_centers, global_cell_sizes = self.split_cell_centers_and_cell_sizes(
                global_cell_centers, global_cell_sizes)
            self.__global_cell_centers: Tuple[Array] = global_cell_centers
            self.__global_cell_sizes: Tuple[Array] = global_cell_sizes
            local_cell_centers = self.get_local_cell_centers()
            local_cell_sizes = self.get_local_cell_sizes()
            global_cell_faces, global_domain_size = jax.pmap(
                self.compute_local_cell_faces, axis_name="i",
                out_axes=(None,None))(local_cell_centers, local_cell_sizes)
            self.__global_cell_faces: Tuple[Array] = global_cell_faces
            self.__global_domain_size: Tuple[Array] = global_domain_size

        else:
            self.__global_domain_size: Tuple[Array] = global_domain_size
            self.__global_cell_centers: Tuple[Array] = global_cell_centers
            self.__global_cell_sizes: Tuple[Array] = global_cell_sizes
            self.__global_cell_faces: Tuple[Array] = global_cell_faces

        self.__global_one_cell_sizes: Tuple[Array] = tuple([1.0/dxi for dxi in global_cell_sizes])

        # NOTE these members are set in the InputManager constructor, since the HaloManager is required to compute the halos first
        self.__global_cell_centers_halos: Tuple[Array] = None
        self.__global_cell_centers_difference: Tuple[Array] = None
        self.__global_cell_sizes_halos: Tuple[Array] = None
        self.__global_cell_sizes_halos_geometry: Tuple[Array] = None
        self.__global_one_cell_sizes_halos: Tuple[Array] = None
        self.__global_one_cell_sizes_halos_geometry: Tuple[Array] = None


    # GET GLOBAL MESH
    def get_global_cell_centers(self) -> Tuple[Array]:
        return self.__global_cell_centers

    def get_global_cell_centers_halos(self) -> Tuple[Array]:
        return self.__global_cell_centers_halos

    def get_global_cell_sizes(self) -> Tuple[Array]:
        return self.__global_cell_sizes
    
    def get_global_cell_sizes_halos(self) -> Tuple[Array]:
        return self.__global_cell_sizes_halos
    
    def get_global_cell_sizes_halos_geometry(self) -> Tuple[Array]:
        return self.__global_cell_sizes_halos_geometry
    
    def get_global_domain_size (self) -> Tuple[Array]:
        return self.__global_domain_size
    
    def get_global_cell_faces (self) -> Tuple[Array]:
        return self.__global_cell_faces


    # GET LOCAL MESH
    def get_local_quantity(self, global_quantity_tuple: Tuple[Array]) -> None:
        if self.is_parallel and self.is_multihost:
            s_ = jnp.s_[self.process_id*self.local_device_count:(self.process_id+1)*self.local_device_count]
            local_quantity_tuple = []
            for axis_index in range(3):
                quantity_xi = global_quantity_tuple[axis_index][s_]
                local_quantity_tuple.append(quantity_xi)
        else:
            local_quantity_tuple = global_quantity_tuple
        return tuple(local_quantity_tuple)
    
    def get_local_cell_centers(self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_cell_centers)

    def get_local_cell_centers_halos(self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_cell_centers_halos)

    def get_local_cell_sizes(self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_cell_sizes)

    def get_local_cell_sizes_halos(self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_cell_sizes_halos)
    
    def get_local_cell_sizes_halos_geometry(self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_cell_sizes_halos_geometry)

    def get_local_cell_faces (self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_cell_faces)

    def get_local_domain_size (self) -> Tuple[Array]:
        return self.get_local_quantity(self.__global_domain_size)




    def compute_device_mesh_grid(self) -> Tuple[Array]:
        """Returns the mesh grid of the active axes for
        the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        device_cell_centers = self.get_device_cell_centers()
        device_cell_centers = [xi.flatten() for xi in device_cell_centers]
        mesh_grid = jnp.meshgrid(*device_cell_centers, indexing="ij")
        mesh_grid = tuple([mesh_grid[i] for i in self.active_axes_indices])
        return mesh_grid
    
    def get_device_cell_centers(self) -> Tuple[Array]:
        """Gets the cell centers corresponding to the 
        present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_cell_centers = [xi[device_id] for xi in self.__global_cell_centers]
        else:
            device_cell_centers = self.__global_cell_centers
        return device_cell_centers
    
    def get_device_cell_centers_halos(self) -> Tuple[Array]:
        """Gets the cell centers with halos corresponding to the 
        present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_cell_centers_halos = [
                jnp.array(xi)[device_id] for xi in self.__global_cell_centers_halos]
        else:
            device_cell_centers_halos = self.__global_cell_centers_halos
        return device_cell_centers_halos
    
    def get_device_cell_centers_difference(self) -> Tuple[Array]:
        """Gets the cell centers difference corresponding to the 
        present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_cell_centers_difference = [
                jnp.array(xi)[device_id] for xi in self.__global_cell_centers_difference]
        else:
            device_cell_centers_difference = self.__global_cell_centers_difference
        return device_cell_centers_difference

    def get_device_cell_faces(self) -> Tuple[Array]:
        """Gets the cell faces corresponding to the 
        present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_cell_faces = [xi[device_id] for xi in self.__global_cell_faces]
        else:
            device_cell_faces = self.__global_cell_faces
        return device_cell_faces

    def get_device_cell_sizes(self) -> Tuple[Array]:
        """Gets the cell sizes corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_cell_sizes = [dxi[device_id] for dxi in self.__global_cell_sizes]
        else:
            device_cell_sizes = self.__global_cell_sizes
        return device_cell_sizes

    def get_device_cell_sizes_halos(self) -> Tuple[Array]:
        """Gets the cell sizes halos corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        device_cell_sizes_halos = []
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_cell_sizes_halos = [dxi[device_id] for dxi in self.__global_cell_sizes_halos]
        else:
            device_cell_sizes_halos = self.__global_cell_sizes_halos
        return device_cell_sizes_halos
    
    def get_device_domain_size(self) -> Dict[str, Array]:
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_domain_size = [size_xi[device_id] for size_xi in self.__global_domain_size]
        else:
            device_domain_size = self.__global_domain_size
        return device_domain_size
    
    def get_device_one_cell_sizes(self) -> Tuple[Array]:
        """Gets the one cell sizes corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_one_cell_sizes = [dxi[device_id] for dxi in self.__global_one_cell_sizes]
        else:
            device_one_cell_sizes = self.__global_one_cell_sizes
        return device_one_cell_sizes
    
    def get_device_one_cell_sizes_halos(self) -> Tuple[Array]:
        """Gets the one cell sizes with halos corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_one_cell_sizes_halos = [dxi[device_id] for dxi in self.__global_one_cell_sizes_halos]
        else:
            device_one_cell_sizes_halos = self.__global_one_cell_sizes_halos
        return device_one_cell_sizes_halos

    def get_device_one_cell_sizes_halos_geometry(self) -> Tuple[Array]:
        """Gets the one cell sizes with halos corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            device_one_cell_sizes_halos_geometry = [dxi[device_id] for dxi in self.__global_one_cell_sizes_halos_geometry]
        else:
            device_one_cell_sizes_halos_geometry = self.__global_one_cell_sizes_halos_geometry
        return device_one_cell_sizes_halos_geometry

    def get_device_cell_face_areas(self) -> Tuple[Array]:
        """Gets the cell face areas corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        dx,dy,dz = self.get_device_cell_sizes()
        if self.dim == 3:
            device_cell_face_areas = tuple([dy*dz, dx*dz, dy*dx])
        elif self.dim == 2:
            device_cell_face_areas = tuple([dy, dx, dz])
        else:
            device_cell_face_areas = tuple([1.0, 1.0, 1.0])
        return device_cell_face_areas

    def get_device_cell_volume(self) -> Array:
        """Gets the cell volume corresponding to the present device.

        :return: _description_
        :rtype: Tuple[Array]
        """
        device_cell_sizes = self.get_device_cell_sizes()
        device_cell_volume: Array = 1.0
        for i in self.active_axes_indices:
            device_cell_volume *= device_cell_sizes[i]
        return device_cell_volume





    # SPLIT MESH
    def split_cell_centers_and_cell_sizes(
            self,
            cell_centers: Tuple[Array],
            cell_sizes: Tuple[Array],
            ) -> Tuple:
        """Splits the cell centers according
        to the domain decomposition.

        :param global_cell_centers: _description_
        :type global_cell_centers: Tuple
        :return: _description_
        :rtype: Tuple
        """
        global_cell_centers = []
        global_cell_sizes = []
        for i in range(3):
            xi = cell_centers[i]
            dxi = cell_sizes[i]
            xi_split = split_cell_centers_xi(xi, self.split_factors, i)
            dxi_split = split_cell_sizes_xi(dxi, self.split_factors, self.is_mesh_stretching, i)
            global_cell_centers.append(xi_split)
            global_cell_sizes.append(dxi_split)
        return tuple(global_cell_centers), tuple(global_cell_sizes)
        
    def compute_local_cell_faces(
            self,
            local_cell_centers: Tuple[Array],
            local_cell_sizes: Tuple[Array],
            ) -> Tuple[Tuple[Array], Tuple[Array]]:
        """Computes the local cell faces.

        :param domain_size: _description_
        :type domain_size: Dict
        :return: _description_
        :rtype: Dict
        """
        domain_size_split = []
        cell_faces_split = []
        for axis_index, axis in zip(AXES_INDICES, AXES):
            xi = local_cell_centers[axis_index]
            dxi = local_cell_sizes[axis_index]
            xi = xi.flatten()
            dxi = dxi.flatten()
            if not self.is_mesh_stretching[axis_index]:
                dxi = jnp.ones_like(xi)*dxi
            lower_bound_xi = xi[0] - dxi[0]/2.0
            cell_faces_xi = lower_bound_xi + jnp.cumsum(dxi)
            cell_faces_xi = jnp.concatenate([jnp.array([lower_bound_xi]), cell_faces_xi])
            domain_size_xi = jnp.array([cell_faces_xi[0], cell_faces_xi[-1]])
            shape = np.roll(np.array([-1,1,1]), axis_index)
            cell_faces_xi = cell_faces_xi.reshape(shape)

            cell_faces_xi = jax.lax.all_gather(cell_faces_xi, axis_name="i")
            domain_size_xi = jax.lax.all_gather(domain_size_xi, axis_name="i")

            cell_faces_split.append(cell_faces_xi)
            domain_size_split.append(domain_size_xi)
        return tuple(cell_faces_split), tuple(domain_size_split)





    # SET HALO CELLS
    def set_global_cell_sizes_with_halos(self, global_cell_sizes_halos: Tuple[Array]) -> None:
        """Setter function for the cell sizes with halos.

        :param cell_sizes_halos: _description_
        :type cell_sizes_halos: Tuple[Array]
        """
        self.__global_cell_sizes_halos = global_cell_sizes_halos
        self.__global_one_cell_sizes_halos = tuple(1.0 / dxi for dxi in global_cell_sizes_halos)

        if self.nh_geometry is not None:
            global_cell_sizes_halos_geometry = []
            for axis_index in range(3):
                cell_sizes_halos_xi = global_cell_sizes_halos[axis_index]
                if self.is_mesh_stretching[axis_index]:
                    cell_sizes_halos_geometry_xi = cell_sizes_halos_xi[self.domain_slices_geometry_1D[axis_index]]
                else:
                    cell_sizes_halos_geometry_xi = cell_sizes_halos_xi

                global_cell_sizes_halos_geometry.append(cell_sizes_halos_geometry_xi)

            self.__global_cell_sizes_halos_geometry = tuple(global_cell_sizes_halos_geometry)
            self.__global_one_cell_sizes_halos_geometry = tuple(1.0 / dxi for dxi in global_cell_sizes_halos_geometry)

    def set_global_cell_centers_with_halos(
            self,
            global_cell_centers_halos: Tuple[Array],
            global_cell_centers_difference: Tuple[Array]
            ) -> None:
        """Setter function for the cell centers with halos.

        :param cell_centers_halos: _description_
        :type cell_centers_halos: Tuple[Array]
        """
        self.__global_cell_centers_halos = global_cell_centers_halos
        self.__global_cell_centers_difference = global_cell_centers_difference