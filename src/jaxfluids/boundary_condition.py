#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

import types
from typing import Callable, Union, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.utilities import get_conservatives_from_primitives

class BoundaryCondition:
    """ The BoundaryCondition class implements functionality to enforce user-
    specified boundary conditions. Boundary conditions are enforced on the 
    primitive variables and the level-set field (for two-phase simulations
    only).

    Boundary conditions for the primitive variables:
    1) Periodic 
    2) Symmetric
    3) No-slip walls
    4) Dirichlet
    5) Neumann

    Boundary conditions for the level-set field:
    1) Periodic
    2) Symmetry
    3) Zero-Gradient
    """

    def __init__(self, domain_information: DomainInformation,  material_manager: MaterialManager, unit_handler: UnitHandler,
            boundary_types: Dict, wall_velocity_functions: Dict, dirichlet_functions: Dict, neumann_functions: Dict, levelset_type: str) -> None:

        self.material_manager   = material_manager
        self.unit_handler       = unit_handler
        self.levelset_type      = levelset_type

        if self.levelset_type != None:
            self.boundary_types_primes      = boundary_types["primes"]
            self.boundary_types_levelset    = boundary_types["levelset"]
        else:
            self.boundary_types_primes      = boundary_types

        self.wall_velocity_functions    = wall_velocity_functions
        self.dirichlet_functions        = dirichlet_functions
        self.neumann_functions          = neumann_functions

        self.dim                = domain_information.dim
        self.number_of_cells    = np.array(domain_information.number_of_cells)
        self.nh                 = domain_information.nh_conservatives

        self.cell_sizes = { "east"  : domain_information.cell_sizes[0],   "west" : domain_information.cell_sizes[0],
                            "north" : domain_information.cell_sizes[1],  "south" : domain_information.cell_sizes[1], 
                            "top"   : domain_information.cell_sizes[2], "bottom" : domain_information.cell_sizes[2] }

        self.coordinates_plane = { 
            "east"  : { "y": domain_information.cell_centers[1], "z": domain_information.cell_centers[2]},
            "west"  : { "y": domain_information.cell_centers[1], "z": domain_information.cell_centers[2]},
            "north" : { "x": domain_information.cell_centers[0], "z": domain_information.cell_centers[2]},
            "south" : { "x": domain_information.cell_centers[0], "z": domain_information.cell_centers[2]},
            "top"   : { "x": domain_information.cell_centers[0], "y": domain_information.cell_centers[1]},
            "bottom": { "x": domain_information.cell_centers[0], "y": domain_information.cell_centers[1]}
        }

        self.inactive_axis                      = domain_information.inactive_axis
        self.active_axis                        = domain_information.active_axis
        self.location_to_axis                   = { "east"  : "x", "west"  : "x", "north" : "y", "south" : "y", "top"   : "z", "bottom": "z" }
        self.spatial_axis_to_index              = {"x":  0, "y":  1, "z":  2}
        self.spatial_axis_to_index_for_slices   = {"x": -3, "y": -2, "z": -1}

        # SLICE OBJECTS
        nh                      = domain_information.nh_conservatives
        nhx, nhy, nhz           = domain_information.domain_slices_conservatives
        
        self.slices_fill = {
            
            "east"  : jnp.s_[..., -nh:, nhy, nhz],
            "west"  : jnp.s_[..., :nh, nhy, nhz],
            "north" : jnp.s_[..., nhx, -nh:, nhz],
            "south" : jnp.s_[..., nhx, :nh, nhz],
            "top"   : jnp.s_[..., nhx, nhy, -nh:],
            "bottom": jnp.s_[..., nhx, nhy, :nh],
        }

        self.slices_retrieve = {

            "periodic" : {
                "east"      :   jnp.s_[..., nh:2*nh, nhy, nhz], 
                "west"      :   jnp.s_[..., -2*nh:-nh, nhy, nhz], 
                "north"     :   jnp.s_[..., nhx, nh:2*nh, nhz], 
                "south"     :   jnp.s_[..., nhx, -2*nh:-nh, nhz], 
                "top"       :   jnp.s_[..., nhx, nhy, nh:2*nh], 
                "bottom"    :   jnp.s_[..., nhx, nhy, -2*nh:-nh], 
            },

            "symmetry" : {
                "east"      :   jnp.s_[..., -nh-1:-2*nh-1:-1, nhy, nhz], 
                "west"      :   jnp.s_[..., 2*nh-1:nh-1:-1, nhy, nhz], 
                "north"     :   jnp.s_[..., nhx, -nh-1:-2*nh-1:-1, nhz], 
                "south"     :   jnp.s_[..., nhx, 2*nh-1:nh-1:-1, nhz], 
                "top"       :   jnp.s_[..., nhx, nhy, -nh-1:-2*nh-1:-1], 
                "bottom"    :   jnp.s_[..., nhx, nhy, 2*nh-1:nh-1:-1], 
            },

            "neumann" : {
                "east"      :   jnp.s_[..., -nh-1:-nh, nhy, nhz], 
                "west"      :   jnp.s_[..., nh:nh+1, nhy, nhz], 
                "north"     :   jnp.s_[..., nhx, -nh-1:-nh, nhz], 
                "south"     :   jnp.s_[..., nhx, nh:nh+1, nhz], 
                "top"       :   jnp.s_[..., nhx, nhy, -nh-1:-nh], 
                "bottom"    :   jnp.s_[..., nhx, nhy, nh:nh+1], 
            },

        }

        # MEMBER FOR SYMMETRY
        self.symmetry_indices = { "x": ([0,2,3,4], 1), "y": ([0,1,3,4], 2), "z": ([0,1,2,4], 3) }
        
        # MEMBER FOR NEUMANN
        self.upwind_difference_sign = { "east"  : -1, "west"  :  1, "north" : -1, "south" :  1, "top"   : -1, "bottom":  1 }

        # BOUNDARY TYPES IN CORNERS
        self.corners = self.assign_corners()

        # CORNER FILL SLICES
        self.corner_slices_fill = {
            "west_south"    : jnp.s_[..., :nh, :nh, nhz],
            "west_north"    : jnp.s_[..., :nh, -nh:, nhz],
            "east_south"    : jnp.s_[..., -nh:, :nh, nhz], 
            "east_north"    : jnp.s_[..., -nh:, -nh:, nhz],

            "bottom_south"  : jnp.s_[..., nhx, :nh, :nh],  
            "bottom_north"  : jnp.s_[..., nhx, -nh:, :nh],
            "top_south"     : jnp.s_[..., nhx, :nh, -nh:],
            "top_north"     : jnp.s_[..., nhx, -nh:, -nh:],

            "bottom_east"   : jnp.s_[..., -nh:, nhy, :nh],
            "bottom_west"   : jnp.s_[..., :nh, nhy, :nh],
            "top_east"      : jnp.s_[..., -nh:, nhy, -nh:],
            "top_west"      : jnp.s_[..., :nh, nhy, -nh:]
        }

        # CORNER RETRIEVE SLICES - NUMBERING CLOCKWISE FROM FILL BLOCK ALONG CORRESPONDING AXIS
        self.corner_slices_retrieve = {
            
            # WEST EAST NORTH SOUTH RETRIEVES
            "west_south_0" : jnp.s_[..., :nh, nh:2*nh, nhz],
            "west_south_1" : jnp.s_[..., nh:2*nh, nh:2*nh, nhz],
            "west_south_2" : jnp.s_[..., nh:2*nh, :nh, nhz],

            "west_north_0" : jnp.s_[..., nh:2*nh, -nh:, nhz],
            "west_north_1" : jnp.s_[..., nh:2*nh, -2*nh:-nh, nhz],
            "west_north_2" : jnp.s_[..., :nh, -2*nh:-nh, nhz],

            "east_south_0" : jnp.s_[..., -2*nh:-nh, :nh, nhz],
            "east_south_1" : jnp.s_[..., -2*nh:-nh, nh:2*nh, nhz],
            "east_south_2" : jnp.s_[..., -nh:, nh:2*nh, nhz],

            "east_north_0" : jnp.s_[..., -nh:, -2*nh:-nh, nhz],
            "east_north_1" : jnp.s_[..., -2*nh:-nh, -2*nh:-nh, nhz],
            "east_north_2" : jnp.s_[..., -2*nh:-nh, -nh:, nhz],

            # BOTTOM TOP NORTH SOUTH RETRIEVES
            "bottom_south_0" : jnp.s_[..., nhx, nh:2*nh, :nh],
            "bottom_south_1" : jnp.s_[..., nhx, nh:2*nh, nh:2*nh],
            "bottom_south_2" : jnp.s_[..., nhx, :nh, nh:2*nh],

            "bottom_north_0" : jnp.s_[..., nhx, -nh:, nh:2*nh],
            "bottom_north_1" : jnp.s_[..., nhx, -2*nh:-nh, nh:2*nh],
            "bottom_north_2" : jnp.s_[..., nhx, -2*nh:-nh, :nh],

            "top_south_0" : jnp.s_[..., nhx, :nh, -2*nh:-nh],
            "top_south_1" : jnp.s_[..., nhx, nh:2*nh, -2*nh:-nh],
            "top_south_2" : jnp.s_[..., nhx, nh:2*nh, -nh:],

            "top_north_0" : jnp.s_[..., nhx, -2*nh:-nh, -nh:],
            "top_north_1" : jnp.s_[..., nhx, -2*nh:-nh, -2*nh:-nh],
            "top_north_2" : jnp.s_[..., nhx, -nh:, -2*nh:-nh],

            # BOTTOM TOP WEST EAST RETRIEVES
            "bottom_west_0" : jnp.s_[..., :nh, nhy, nh:2*nh],
            "bottom_west_1" : jnp.s_[..., nh:2*nh, nhy, nh:2*nh],
            "bottom_west_2" : jnp.s_[..., nh:2*nh, nhy, :nh],

            "bottom_east_0" : jnp.s_[..., -2*nh:-nh, nhy, :nh],
            "bottom_east_1" : jnp.s_[..., -2*nh:-nh, nhy, nh:2*nh],
            "bottom_east_2" : jnp.s_[..., -nh:, nhy, nh:2*nh],

            "top_east_0" : jnp.s_[...,  -nh:, nhy, -2*nh:-nh], 
            "top_east_1" : jnp.s_[...,  -2*nh:-nh, nhy, -2*nh:-nh],
            "top_east_2" : jnp.s_[...,  -2*nh:-nh, nhy, -nh:],

            "top_west_0" : jnp.s_[..., nh:2*nh, nhy, -nh:],
            "top_west_1" : jnp.s_[..., nh:2*nh, nhy, -2*nh:-nh],
            "top_west_2" : jnp.s_[..., :nh, nhy, -2*nh:-nh]

        }

        # CORNER COMBINATIONS
        self.corner_combinations = {

            # WEST EAST NORTH SOUTH COMBS
            "west_south": {

                "periodic_periodic":    ( "east_north_1", [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "west_south_1", [-1,-1, 1] ),

                "symmetry_any":         ( "west_south_2", [-1, 1, 1] ),
                "any_symmetry":         ( "west_south_0", [ 1,-1, 1] ),

                "periodic_any":         ( "east_south_0", [ 1, 1, 1] ),
                "any_periodic":         ( "west_north_2", [ 1, 1, 1] ),

                "any_any":  [("west_south_0", [ 1, 1, 1]), ("west_south_2", [ 1, 1, 1])],

            },

            "west_north": {

                "periodic_periodic":    ( "east_south_1", [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "west_north_1", [-1,-1, 1] ),

                "symmetry_any":         ( "west_north_0", [-1, 1, 1] ),
                "any_symmetry":         ( "west_north_2", [ 1,-1, 1] ),

                "periodic_any":         ( "east_north_2", [ 1, 1, 1] ),
                "any_periodic":         ( "west_south_0", [ 1, 1, 1] ),

                "any_any":  [("west_north_0", [ 1, 1, 1]), ("west_north_2", [ 1, 1, 1])],

            },

            "east_north": {

                "periodic_periodic":    ( "west_south_1", [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "east_north_1", [-1,-1, 1] ),

                "symmetry_any":         ( "east_north_2", [-1, 1, 1] ),
                "any_symmetry":         ( "east_north_0", [ 1,-1, 1] ),

                "periodic_any":         ( "west_north_0", [ 1, 1, 1] ),
                "any_periodic":         ( "east_south_2", [ 1, 1, 1] ),

                "any_any":  [("east_north_0", [ 1, 1, 1]), ("east_north_2", [ 1, 1, 1])],

            },

            "east_south": {

                "periodic_periodic":    ( "west_north_1", [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "east_south_1", [-1,-1, 1] ),

                "symmetry_any":         ( "east_south_0", [-1, 1, 1] ),
                "any_symmetry":         ( "east_south_2", [ 1,-1, 1] ),

                "periodic_any":         ( "west_south_2", [ 1, 1, 1] ),
                "any_periodic":         ( "east_north_0", [ 1, 1, 1] ),

                "any_any":  [("east_south_0", [ 1, 1, 1]), ("east_south_2", [ 1, 1, 1])],

            },

            # BOTTOM TOP NORTH SOUTH COMBS
            "bottom_south": {

                "periodic_periodic":    ( "top_north_1",    [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "bottom_south_1", [ 1,-1,-1] ),

                "symmetry_any":         ( "bottom_south_2", [ 1, 1,-1] ),
                "any_symmetry":         ( "bottom_south_0", [ 1,-1, 1] ),

                "periodic_any":         ( "top_south_0",    [ 1, 1, 1] ),
                "any_periodic":         ( "bottom_north_2", [ 1, 1, 1] ),

                "any_any":  [ ("bottom_south_0", [ 1, 1, 1]), ("bottom_south_2", [ 1, 1, 1]) ],

            },

            "bottom_north": {

                "periodic_periodic":    ( "top_south_1",     [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "bottom_north_1",  [ 1,-1,-1] ),

                "symmetry_any":         ( "bottom_north_0",  [ 1, 1,-1] ),
                "any_symmetry":         ( "bottom_north_2",  [ 1,-1, 1] ),

                "periodic_any":         ( "top_north_2",     [ 1, 1, 1] ),
                "any_periodic":         ( "bottom_south_0",  [ 1, 1, 1] ),

                "any_any":  [ ("bottom_north_0", [ 1, 1, 1]), ("bottom_north_2", [ 1, 1, 1]) ],

            },

            "top_north": {

                "periodic_periodic":    ( "bottom_south_1",  [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "top_north_1",     [ 1,-1,-1] ),

                "symmetry_any":         ( "top_north_2",     [ 1, 1,-1] ),
                "any_symmetry":         ( "top_north_0",     [ 1,-1, 1] ),

                "periodic_any":         ( "bottom_north_0",  [ 1, 1, 1] ),
                "any_periodic":         ( "top_south_2",     [ 1, 1, 1] ),

                "any_any":  [ ("top_north_0", [ 1, 1, 1]), ("top_north_2", [ 1, 1, 1]) ],

            },

            "top_south": {

                "periodic_periodic":    ( "bottom_north_1",  [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "top_south_1",     [ 1,-1,-1] ),

                "symmetry_any":         ( "top_south_0",     [ 1, 1,-1] ),
                "any_symmetry":         ( "top_south_2",     [ 1,-1, 1] ),

                "periodic_any":         ( "bottom_south_2",  [ 1, 1, 1] ),
                "any_periodic":         ( "top_north_0",     [ 1, 1, 1] ),

                "any_any":  [ ("top_south_0", [ 1, 1, 1]), ("top_south_2", [ 1, 1, 1]) ],

            },

            # BOTTOM TOP WEST EAST COMBS
            "bottom_west": {

                "periodic_periodic":    ( "top_east_1",     [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "bottom_west_1",  [-1, 1,-1] ),

                "symmetry_any":         ( "bottom_west_0",  [ 1, 1,-1] ),
                "any_symmetry":         ( "bottom_west_2",  [-1, 1, 1] ),

                "periodic_any":         ( "top_west_2",     [ 1, 1, 1] ),
                "any_periodic":         ( "bottom_east_0",  [ 1, 1, 1] ),

                "any_any":  [ ("bottom_west_0", [ 1, 1, 1]), ("bottom_west_2", [ 1, 1, 1]) ],

            },

            "bottom_east": {

                "periodic_periodic":    ( "top_west_1",     [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "bottom_east_1",  [-1, 1,-1] ),

                "symmetry_any":         ( "bottom_east_2",  [ 1, 1,-1] ),
                "any_symmetry":         ( "bottom_east_0",  [-1, 1, 1] ),

                "periodic_any":         ( "top_east_0",     [ 1, 1, 1] ),
                "any_periodic":         ( "bottom_west_2",  [ 1, 1, 1] ),

                "any_any":  [ ("bottom_east_0", [ 1, 1, 1]), ("bottom_east_2", [ 1, 1, 1]) ],

            },

            "top_east": {

                "periodic_periodic":    ( "bottom_west_1",  [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "top_east_1",     [-1, 1,-1] ),

                "symmetry_any":         ( "top_east_0",     [ 1, 1,-1] ),
                "any_symmetry":         ( "top_east_2",     [-1, 1, 1] ),

                "periodic_any":         ( "bottom_east_2",  [ 1, 1, 1] ),
                "any_periodic":         ( "top_west_0",     [ 1, 1, 1] ),

                "any_any":  [ ("top_east_0", [ 1, 1, 1]), ("top_east_2", [ 1, 1, 1]) ],

            },

            "top_west": {

                "periodic_periodic":    ( "bottom_east_1",  [ 1, 1, 1] ),
                "symmetry_symmetry":    ( "top_west_1",     [-1, 1,-1] ),

                "symmetry_any":         ( "top_west_2",     [ 1, 1,-1] ),
                "any_symmetry":         ( "top_west_0",     [-1, 1, 1] ),

                "periodic_any":         ( "bottom_west_0",  [ 1, 1, 1] ),
                "any_periodic":         ( "top_east_2",     [ 1, 1, 1] ),

                "any_any":  [ ("top_west_0", [ 1, 1, 1]), ("top_west_2", [ 1, 1, 1]) ],

            }

        }


    def assign_corners(self) -> Dict:
        """Identifies the boundary type pairs at the corners (2D) / edges (3D) of 
        computational domain. This is necessary to fill the halo cells that are located at
        the diagonal extension of the domain.

        :return: Dictionary containing the boundary type pairs at each boundary corner/edge location
        :rtype: Dict
        """

        locations = ["west_south", "west_north", "east_north", "east_south", "bottom_south", "bottom_north", "top_south", "top_north", "bottom_east", "bottom_west", "top_east", "top_west"]
        indices = {

            "west_south"    :   {"west"     :  0, "south" :  0},
            "west_north"    :   {"west"     : -1, "north" :  0},
            "east_south"    :   {"east"     :  0, "south" : -1},
            "east_north"    :   {"east"     : -1, "north" : -1},

            "bottom_south"  :   {"bottom"   :  0, "south" :  0}, 
            "bottom_north"  :   {"bottom"   : -1, "north" :  0},
            "top_south"     :   {"top"      :  0, "south" : -1},
            "top_north"     :   {"top"      : -1, "north" : -1},

            "bottom_east"   :   {"bottom"   : -1, "east" :  0},
            "bottom_west"   :   {"bottom"   :  0, "west" :  0},
            "top_east"      :   {"top"      : -1, "east" : -1},
            "top_west"      :   {"top"      :  0, "west" : -1},

        }

        boundary_types = {}
        boundary_types["primes"] = self.boundary_types_primes
        boundary_types["levelset"] = self.boundary_types_levelset if self.levelset_type != None else None

        corners = {"primes": {}, "levelset": {}} if self.levelset_type != None else {"primes": {}}
        
        for key in corners:
            for location in locations:
                b_type = []
                for loc in location.split("_"):
                    boundary_type = boundary_types[key][loc]
                    if type(boundary_type) == list:
                        b_type.append(boundary_type[0][indices[location][loc]])
                    else:
                        b_type.append(boundary_type)
                b_type = "_".join(b_type)
                corners[key][location] = b_type

            for corner, combinations in corners[key].items():
                boundary1, boundary2 = combinations.split("_")
                if np.array([bound in ["dirichlet", "neumann", "wall"] for bound in [boundary1, boundary2]]).all():
                    boundary1, boundary2 = "any", "any"
                elif boundary1 in ["dirichlet", "neumann", "wall", "symmetry"] and boundary2 == "periodic":
                    boundary1 = "any"
                elif boundary1 in ["dirichlet", "neumann", "wall", "periodic"] and boundary2 == "symmetry":
                    boundary1 = "any"
                elif boundary2 in ["dirichlet", "neumann", "wall", "symmetry"] and boundary1 == "periodic":
                    boundary2 = "any"
                elif boundary2 in ["dirichlet", "neumann", "wall", "periodic"] and boundary1 == "symmetry":
                    boundary2 = "any"
                corners[key][corner] = "_".join([boundary1, boundary2])

        return corners


    def fill_boundary_primes(self, cons: jnp.ndarray, primes: jnp.ndarray,
            current_time: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fills the halo cells of the primitive and conservative variable buffers.

        :param cons: Buffer of conservative variables
        :type cons: jnp.ndarray
        :param primes: Buffer of primitive variables
        :type primes: jnp.ndarray
        :param current_time: Current physical simulation time
        :type current_time: float
        :return: Primitive and conservative variable buffer with filled halo cells
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        # FILL BOUNDARIES
        for boundary_location, boundary_types in self.boundary_types_primes.items():

            if type(boundary_types) == list:
                b_types     = boundary_types[0]
                b_ranges    = boundary_types[1]
            else:
                b_types = [boundary_types]
                b_ranges = [(0.0, 1.0)]

            wall_counter        = 0
            dirichlet_counter   = 0
            neumann_counter     = 0

            for b_type, b_range in zip(b_types, b_ranges):

                if b_type == "symmetry":
                    cons, primes = self.symmetry(cons, primes, boundary_location, b_range)

                if b_type == "periodic":
                    cons, primes = self.periodic(cons, primes, boundary_location, b_range)

                if b_type == "wall":
                    if type(self.wall_velocity_functions[boundary_location]) == list:
                        function = self.wall_velocity_functions[boundary_location][wall_counter]
                    else:
                        function = self.wall_velocity_functions[boundary_location]
                    cons, primes = self.wall(cons, primes, boundary_location, function, current_time, b_range)
                    wall_counter += 1

                if b_type == "dirichlet":
                    if type(self.dirichlet_functions[boundary_location]) == list:
                        function = self.dirichlet_functions[boundary_location][dirichlet_counter]
                    else:
                        function = self.dirichlet_functions[boundary_location]
                    cons, primes = self.dirichlet(cons, primes, boundary_location, function, current_time, b_range)
                    dirichlet_counter += 1

                if b_type == "neumann":
                    if type(self.neumann_functions[boundary_location]) == list:
                        function = self.neumann_functions[boundary_location][neumann_counter]
                    else:
                        function = self.neumann_functions[boundary_location]
                    cons, primes = self.neumann(cons, primes, boundary_location, function, current_time, b_range)
                    neumann_counter += 1


                if b_type == "inactive":
                    continue

        # FILL DOMAIN CORNERS
        velocities  = self.fill_corners_primes(primes[1:4])
        primes      = primes.at[1:4].set(velocities)
        cons        = get_conservatives_from_primitives(primes, self.material_manager)

        return cons, primes

    def fill_boundary_levelset(self, levelset: jnp.ndarray) -> jnp.ndarray:
        """Fills the levelset buffer halo cells.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :return: Levelset buffer with filled halo cells
        :rtype: jnp.ndarray
        """

        for boundary_location, boundary_type in self.boundary_types_levelset.items():
            if boundary_type == "inactive":
                continue
            elif boundary_type in ["symmetry", "periodic", "neumann"]:
                slices_retrieve = self.slices_retrieve[boundary_type][boundary_location]
                slices_fill = self.slices_fill[boundary_location]
                levelset = levelset.at[slices_fill].set(levelset[slices_retrieve])

        levelset = self.fill_corners_levelset(levelset)

        return levelset

    def fill_corners_levelset(self, levelset: jnp.ndarray) -> jnp.ndarray:
        """Fills the levelset buffer halo cells that are located at the diagional extension of the domain.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :return: Levelset buffer with filled halo cells at the corners
        :rtype: jnp.ndarray
        """

        for location_fill, combinations in self.corners["levelset"].items():

            if "inactive" in combinations.split("_"):
                continue

            if combinations == "any_any":

                block1, block2 = self.corner_combinations[location_fill][combinations]

                location_retrieve1, flip1 = block1
                location_retrieve2, flip2 = block2

                slice_fill = self.corner_slices_fill[location_fill]
                
                slice_retrieve1 = self.corner_slices_retrieve[location_retrieve1]
                slice_retrieve2 = self.corner_slices_retrieve[location_retrieve2]

                halo = 0.5 * (levelset[slice_retrieve1][::flip1[0], ::flip1[1], ::flip1[2]] + levelset[slice_retrieve2][::flip2[0], ::flip2[1], ::flip2[2]])
                levelset = levelset.at[slice_fill].set(halo)

            else:

                location_retrieve, flip = self.corner_combinations[location_fill][combinations]

                slice_fill     = self.corner_slices_fill[location_fill]
                slice_retrieve = self.corner_slices_retrieve[location_retrieve]

                levelset = levelset.at[slice_fill].set(levelset[slice_retrieve][..., ::flip[0], ::flip[1], ::flip[2]])

        return levelset    

    def fill_corners_primes(self, primes: jnp.ndarray) -> jnp.ndarray:
        """Fills the prime buffer halo cells that are located at the diagonal extension of the domain.

        :param primes: Buffer of the primitive variables
        :type primes: jnp.ndarray
        :return: Primitive variables buffer with filled halo cells at the corners
        :rtype: jnp.ndarray
        """

        for location_fill, combinations in self.corners["primes"].items():

            if "inactive" in combinations.split("_"):
                continue

            # FOR ANY - ANY, WE FILL THE CORNER USING THE AVERAGE OF THE ADJESCENT SQUARES
            if combinations == "any_any":

                block1, block2 = self.corner_combinations[location_fill][combinations]

                location_retrieve1, flip1 = block1
                location_retrieve2, flip2 = block2

                slice_fill = self.corner_slices_fill[location_fill]
                
                slice_retrieve1 = self.corner_slices_retrieve[location_retrieve1]
                slice_retrieve2 = self.corner_slices_retrieve[location_retrieve2]

                halo = 0.5 * (primes[slice_retrieve1][..., ::flip1[0], ::flip1[1], ::flip1[2]] + primes[slice_retrieve2][..., ::flip2[0], ::flip2[1], ::flip2[2]])
                primes = primes.at[slice_fill].set(halo)                

            else:

                # FOR SYMMETRY COMBINATIONS, THE SIGN OF CORRESPONDING VELOCITY MUST BE CHANGED
                if "symmetry" in combinations.split("_"):
                    
                    location_retrieve, flip = self.corner_combinations[location_fill][combinations]

                    slice_fill     = self.corner_slices_fill[location_fill]
                    slice_retrieve = self.corner_slices_retrieve[location_retrieve]
                    
                    indices_flip    = list(np.where(np.array(flip) == -1)[0])
                    indices_noflip  = [i for i in range(3) if i not in indices_flip]

                    fill_flip   = (indices_flip,) + slice_fill
                    fill_noflip = (indices_noflip,) + slice_fill

                    retrieve_flip   = (indices_flip,) + slice_retrieve
                    retrieve_noflip = (indices_noflip,) + slice_retrieve

                    primes = primes.at[fill_flip].set(-primes[retrieve_flip][..., ::flip[0], ::flip[1], ::flip[2]])         
                    primes = primes.at[fill_noflip].set(primes[retrieve_noflip][..., ::flip[0], ::flip[1], ::flip[2]]) 

                else:

                    location_retrieve, flip = self.corner_combinations[location_fill][combinations]

                    slice_fill     = self.corner_slices_fill[location_fill]
                    slice_retrieve = self.corner_slices_retrieve[location_retrieve]

                    primes = primes.at[slice_fill].set(primes[slice_retrieve][..., ::flip[0], ::flip[1], ::flip[2]])

        return primes

    def wall(self, cons: jnp.ndarray, primes: jnp.ndarray, location: str,
            functions: Dict, current_time: float, b_range: List) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fills the halo cells of the primitive and conservative variable buffer at the specified
        location according to the no-slip wall boundary condition.

        :param cons: Conservative variable buffer
        :type cons: jnp.ndarray
        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param location: Boundary location
        :type location: str
        :param functions: Wall velocity functions
        :type functions: Dict
        :param current_time: Current physical simulation time
        :type current_time: float
        :param b_range: List containing the spatial range of the boundary at the specified location
        :type b_range: List
        :return: Primitive and conservative variable buffers with filled halos at specified location
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        if self.dim == 2:
            slices_retrieve = self.get_slices_retrieve(location, "symmetry", b_range)
            slices_fill = self.get_slices_fill(location, b_range)
        else:
            slices_retrieve = self.slices_retrieve["symmetry"][location]
            slices_fill = self.slices_fill[location]

        wall_velocity = {}

        for velocity in ["u", "v", "w"]:
            if velocity in functions.keys():
                if type(functions[velocity]) == types.LambdaType:
                    wall_velocity[velocity] = functions[velocity](self.unit_handler.dimensionalize(current_time, "time"))
                else:
                    wall_velocity[velocity] = functions[velocity]
                wall_velocity[velocity] = self.unit_handler.non_dimensionalize(wall_velocity[velocity], "velocity")
            else:
                wall_velocity[velocity] = 0.0

        u_halo = 2 * wall_velocity["u"] - primes[(jnp.s_[1:2],) + slices_retrieve]
        v_halo = 2 * wall_velocity["v"] - primes[(jnp.s_[2:3],) + slices_retrieve]
        w_halo = 2 * wall_velocity["w"] - primes[(jnp.s_[3:4],) + slices_retrieve]

        halos_prime = jnp.vstack([      primes[(jnp.s_[0:1],) + slices_retrieve],
                                        u_halo,
                                        v_halo,
                                        w_halo, 
                                        primes[(jnp.s_[4:5],) + slices_retrieve]      ])
        

        halos_cons = get_conservatives_from_primitives(halos_prime, self.material_manager)

        cons     = cons.at[slices_fill].set(halos_cons)
        primes   = primes.at[slices_fill].set(halos_prime)

        return cons, primes

    def dirichlet(self, cons: jnp.ndarray, primes: jnp.ndarray, location: str,
            functions: Union[Callable, float], current_time: float, b_range: List) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fills the halo cells of the primitive and conservative variable buffer at the specified location 
        according to the dirichlet boundary condition.


        :param cons: Conservative variable buffer
        :type cons: jnp.ndarray
        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param location: Boundary location
        :type location: str
        :param functions: Dirichlet functions
        :type functions: Union[Callable, float]
        :param current_time: Current physical simulation time
        :type current_time: float
        :param b_range: List containing the spatial range of the boundary at the specified location
        :type b_range: List
        :return: Primitive and conservative variable buffers with filled halos at specified location
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        # GET SLICE OBJECTS
        if self.dim == 2:
            slices_fill = self.get_slices_fill(location, b_range)
        else:
            slices_fill = self.slices_fill[location]

        # COMPUTE PRESENT COORDINATES
        coordinates         = [self.coordinates_plane[location].get(axis)[int(b_range[0]*self.number_of_cells[self.spatial_axis_to_index[axis]]):int(b_range[1]*self.number_of_cells[self.spatial_axis_to_index[axis]])] for axis in self.active_axis if self.coordinates_plane[location].get(axis) != None]
        coordinates_name    = [axis for axis in self.active_axis if self.coordinates_plane[location].get(axis) != None]
        axis_to_expand      = [axis for axis in ["x", "y", "z"] if axis not in coordinates_name]

        # DIMENSIONALIZE FOR LAMBDA FUNCTION
        mesh_grid           = jnp.meshgrid(*[self.unit_handler.dimensionalize(coord, "length") for coord in coordinates], indexing="ij")
        current_time        = self.unit_handler.dimensionalize(current_time, "time")

        # EVALUATE LAMBDAS
        halos_prime_list    = []
        for prime_state in functions:
            func = functions[prime_state]
            if type(func) in [float, np.float64, np.float32]:
                halos = func*jnp.ones(mesh_grid[0].shape) if self.dim != 1 else func
            elif type(func) == types.LambdaType:
                halos = func(*mesh_grid, current_time)
            else:
                assert False, "Dirichlet boundary values must be lambda function or python/numpy float"
            for ax in axis_to_expand:
                halos = jnp.expand_dims(halos, self.spatial_axis_to_index[ax])
            halos = self.unit_handler.non_dimensionalize(halos, prime_state)
            halos_prime_list.append(halos)

        # STACK
        halos_prime = jnp.stack(halos_prime_list, axis=0)

        if self.levelset_type == "FLUID-FLUID":
            halos_prime = jnp.stack([halos_prime, halos_prime], axis=1)

        # COMPUTE CONSERVATIVES
        halos_cons = get_conservatives_from_primitives(halos_prime, self.material_manager)
        
        # FILL
        primes = primes.at[slices_fill].set(halos_prime) 
        cons = cons.at[slices_fill].set(halos_cons) 

        return cons, primes

    def neumann(self, cons: jnp.ndarray, primes: jnp.ndarray, location: str,
            functions: Union[Callable, float], current_time: float, b_range: List) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fills the halo cells of the primitive and conservative variable buffer at the specified
        location according to the neumann boundary condition. 

        :param cons: Conservative variable buffer
        :type cons: jnp.ndarray
        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param location: Boundary location
        :type location: str
        :param functions: Neumann functions
        :type functions: Union[Callable, float]
        :param current_time: Current physical simulation time
        :type current_time: float
        :param b_range: List containing the spatial range of the boundary at the specified location
        :type b_range: List
        :return: Primitive and conservative variable buffers with filled halos at specified location
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        
        # GET SLICE OBJECTS
        if self.dim == 2:
            slices_retrieve = self.get_slices_retrieve(location, "neumann", b_range)
            slices_fill = self.get_slices_fill(location, b_range)
        else:
            slices_retrieve = self.slices_retrieve["neumann"][location]
            slices_fill = self.slices_fill[location]

        # COMPUTE PRESENT COORDINATES
        coordinates = [self.coordinates_plane[location].get(axis)[int(b_range[0]*self.number_of_cells[self.spatial_axis_to_index[axis]]):int(b_range[1]*self.number_of_cells[self.spatial_axis_to_index[axis]])] for axis in self.active_axis if self.coordinates_plane[location].get(axis) != None]
        coordinates_name = [axis for axis in self.active_axis if self.coordinates_plane[location].get(axis) != None]
        axis_to_expand = [axis for axis in ["x", "y", "z"] if axis not in coordinates_name]

        # DIMENSIONALIZE FOR LAMBDA FUNCTION
        mesh_grid           = jnp.meshgrid(*[self.unit_handler.dimensionalize(coord, "length") for coord in coordinates], indexing="ij")
        current_time        = self.unit_handler.dimensionalize(current_time, "time")

        # EVALUATE LAMBDAS
        halos_prime_list = []
        for i, prime_state in enumerate(functions):
            func = functions[prime_state]
            if type(func) in [float, np.float64, np.float32]:
                neumann_value = func*jnp.ones(mesh_grid[0].shape) if self.dim != 1 else func
            elif type(func) == types.LambdaType:
                neumann_value = func(*mesh_grid, current_time)
            else:
                assert False, "Neumann boundary values must be lambda function or python/numpy float"
            for axis in axis_to_expand:
                neumann_value = jnp.expand_dims(neumann_value, self.spatial_axis_to_index[axis])
            neumann_value = self.unit_handler.non_dimensionalize(neumann_value, prime_state)
            neumann_value = self.unit_handler.dimensionalize(neumann_value, "length")

            halos  = primes[(i,) + slices_retrieve] + self.upwind_difference_sign[location] * neumann_value * self.cell_sizes[location]
            halos_prime_list.append(halos)

        # STACK
        halos_prime = jnp.stack(halos_prime_list, axis=0)

        # COMPUTE CONSERVATIVES
        halos_cons = get_conservatives_from_primitives(halos_prime, self.material_manager)

        # FILL
        primes = primes.at[slices_fill].set(halos_prime) 
        cons = cons.at[slices_fill].set(halos_cons)  

        return cons, primes

    def symmetry(self, cons: jnp.ndarray, primes: jnp.ndarray,
            location: str, b_range: List) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fills the halo cells of the primitive and conservative variable buffer
        at the specified location according to the symmetric boundary condition.

        :param cons: Conservative variable buffer
        :type cons: jnp.ndarray
        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param location: Boundary location
        :type location: str
        :param b_range: List containing the spatial range of the boundary at the specified location
        :type b_range: List
        :return: Primitive and conservative variable buffers with filled halos at specified location
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        if self.dim == 2:
            slices_retrieve = self.get_slices_retrieve(location, "symmetry", b_range)
            slices_fill = self.get_slices_fill(location, b_range)
        else:
            slices_retrieve = self.slices_retrieve["symmetry"][location]
            slices_fill = self.slices_fill[location]

        axis = self.location_to_axis[location]

        cons = cons.at[(self.symmetry_indices[axis][0], ) + slices_fill].set(cons[(self.symmetry_indices[axis][0], ) + slices_retrieve])
        cons = cons.at[(self.symmetry_indices[axis][1], ) + slices_fill].set(-cons[(self.symmetry_indices[axis][1], ) + slices_retrieve])
        primes = primes.at[(self.symmetry_indices[axis][0], ) + slices_fill].set(primes[(self.symmetry_indices[axis][0], ) + slices_retrieve])
        primes = primes.at[(self.symmetry_indices[axis][1], ) + slices_fill].set(-primes[(self.symmetry_indices[axis][1], ) + slices_retrieve])

        return cons, primes

    def periodic(self, cons: jnp.ndarray, primes: jnp.ndarray, location: str, b_range: List) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fills the halos of the conservative and primitive variable buffer at
        the specified location according to the periodic boundary condition.

        :param cons: Conservative variable buffer
        :type cons: jnp.ndarray
        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param location: Boundary location
        :type location: str
        :param b_range: List containing the spatial range of the boundary at the specified location
        :type b_range: List
        :return: Primitive and conservative variable buffers with filled halos at specified location
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        if self.dim == 2:
            slices_retrieve = self.get_slices_retrieve(location, "periodic", b_range)
            slices_fill = self.get_slices_fill(location, b_range)
        else:
            slices_retrieve = self.slices_retrieve["periodic"][location]
            slices_fill = self.slices_fill[location]

        cons = cons.at[slices_fill].set(cons[slices_retrieve])
        primes = primes.at[slices_fill].set(primes[slices_retrieve])

        return cons, primes

    def get_slices_fill(self, location: str, b_range: List) -> Tuple:
        """Computes the slice objects to fill the halos depending on the range at the specified boundary location.

        :param location: Boundary location
        :type location: str
        :param b_range: List containing the spatial range of the boundary at the specified location 
        :type b_range: List
        :return: Slice objects
        :rtype: Tuple
        """
        # 2D only
        axis = [axis for axis in self.active_axis if self.coordinates_plane[location].get(axis) != None][0]
        a = self.nh + int(b_range[0] * self.number_of_cells[self.spatial_axis_to_index[axis]])
        b = self.nh + int(b_range[1] * self.number_of_cells[self.spatial_axis_to_index[axis]])
        slices = list(self.slices_fill[location])
        slices[self.spatial_axis_to_index_for_slices[axis]] = jnp.s_[a:b]
        return tuple(slices)

    def get_slices_retrieve(self, location: str, boundary_type: str, b_range: List) -> Tuple:
        """Computes the slice objects to retrieve the values depending on the range at the specified boundary location.

        :param location: Boundary location
        :type location: str
        :param boundary_type: Boundary location
        :type boundary_type: str
        :param b_range: List containing the spatial range of the boundary at the specified location 
        :type b_range: List
        :return: Slice objects
        :rtype: Tuple
        """
        # 2D only
        axis = [axis for axis in self.active_axis if self.coordinates_plane[location].get(axis) != None][0]
        a = self.nh + int(b_range[0] * self.number_of_cells[self.spatial_axis_to_index[axis]])
        b = self.nh + int(b_range[1] * self.number_of_cells[self.spatial_axis_to_index[axis]])
        slices = list(self.slices_retrieve[boundary_type][location])
        slices[self.spatial_axis_to_index_for_slices[axis]] = jnp.s_[a:b]
        return tuple(slices)

    def symmetry_levelset(self, levelset: jnp.ndarray, location: str):
        slices_retrieve = self.slices_retrieve["symmetry"][location]
        slices_fill = self.slices_fill[location]
        levelset = levelset.at[slices_fill].set(levelset[slices_retrieve])
        return levelset
        
    def periodic_levelset(self, levelset: jnp.ndarray, location: str):
        slices_retrieve = self.slices_retrieve["periodic"][location]
        slices_fill = self.slices_fill[location]
        levelset = levelset.at[slices_fill].set(levelset[slices_retrieve])
        return levelset