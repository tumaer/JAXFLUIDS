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

from typing import List, Union, Dict
import types

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.unit_handler import UnitHandler

class LevelsetCreator:
    """The LevelsetCreator implements functionality to create initial levelset fields. The initial
    levelset field in one of two ways:
    1) Lambda function via case setup file
    2) List of building blocks. A single building block includes a shape
        and a lambda function for the bounding domain.
    """

    def __init__(self, domain_information: DomainInformation, unit_handler: UnitHandler,
        initial_levelset: Union[str, List], narrow_band_cutoff: int) -> None:

        self.unit_handler       = unit_handler
        self.initial_levelset   = initial_levelset
        self.narrow_band_cutoff = narrow_band_cutoff

        self.cell_centers               = domain_information.cell_centers
        self.cell_sizes                 = domain_information.cell_sizes
        self.nx, self.ny, self.nz       = domain_information.number_of_cells
        self.nhx, self.nhy, self.nhz    = domain_information.domain_slices_conservatives
        self.nh                         = domain_information.nh_conservatives         
        self.active_axis_indices        = domain_information.active_axis_indices
        self.smallest_cell_size         = jnp.min(jnp.array([self.cell_sizes[i] for i in self.active_axis_indices]))

        self.shape_function_dict: Dict[str, types.LambdaType] = {
            "circle": self.get_circle,
            "square": self.get_rectangle,
            "rounded_square": self.get_rectangle,
            "square": self.get_rectangle,
            "rounded_rectangle": self.get_rectangle,
            "sphere": self.get_sphere
            }

    def get_circle(self, radius: float, position: List) -> jnp.ndarray:
        """Creates the levelset field for a circle.

        :param radius: Radius
        :type radius: float
        :param position: Center position
        :type position: List
        :return: Levelset buffer
        :rtype: jnp.ndarray
        """
        mesh_grid   = [jnp.meshgrid(*self.cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
        for i in range(len(mesh_grid)):
            mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")
        levelset    = - radius + jnp.sqrt((mesh_grid[0] - position[0])**2 + (mesh_grid[1] - position[1])**2)
        return levelset

    def get_rectangle(self, length: float, position: List, height : float = None, radius: float = None) -> jnp.ndarray:
        """Creates the levelset field for a rectangle. If the radius argument is specified, the rectangle corners will 
        be rounded using that radius. If the height argument is not specified, a square will be created.

        :param length: Length
        :type length: float
        :param position: Center position
        :type position: List
        :param height: Height, defaults to None
        :type height: float, optional
        :param radius: Radius of the corners, defaults to None
        :type radius: float, optional
        :return: Leveset buffer
        :rtype: jnp.ndarray
        """
        mesh_grid = [jnp.meshgrid(*self.cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
        for i in range(len(mesh_grid)):
            mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")

        if height == None:
            height = length

        edge_tr = jnp.array([position[0] + length/2.0, position[1] + height/2.0])
        edge_tl = jnp.array([position[0] - length/2.0, position[1] + height/2.0])

        edge_bl = jnp.array([position[0] - length/2.0, position[1] - height/2.0])
        edge_br = jnp.array([position[0] + length/2.0, position[1] - height/2.0])

        line_1_slope    = (edge_tr[1] - edge_bl[1])/(edge_tr[0] - edge_bl[0])
        line_1_offset   = edge_tr[1] - line_1_slope * edge_tr[0]

        line_2_slope    = (edge_br[1] - edge_tl[1])/(edge_br[0] - edge_tl[0])
        line_2_offset   = edge_br[1] - line_2_slope * edge_br[0]

        levelset  =   (mesh_grid[1] - (position[1] + height/2.0)) * ((mesh_grid[1] > line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] > line_2_slope * mesh_grid[0] + line_2_offset))
        levelset += - (mesh_grid[1] - (position[1] - height/2.0)) * ((mesh_grid[1] < line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] < line_2_slope * mesh_grid[0] + line_2_offset))
        levelset +=   (mesh_grid[0] - (position[0] + length/2.0)) * ((mesh_grid[1] <= line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] >= line_2_slope * mesh_grid[0] + line_2_offset))
        levelset += - (mesh_grid[0] - (position[0] - length/2.0)) * ((mesh_grid[1] >= line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] <= line_2_slope * mesh_grid[0] + line_2_offset))

        if radius:

            levelset *= jnp.invert((mesh_grid[0] > edge_tr[0] - radius) & (mesh_grid[1] > edge_tr[1] - radius))
            levelset *= jnp.invert((mesh_grid[0] < edge_tl[0] + radius) & (mesh_grid[1] > edge_tl[1] - radius))
            levelset *= jnp.invert((mesh_grid[0] < edge_bl[0] + radius) & (mesh_grid[1] < edge_bl[1] + radius))
            levelset *= jnp.invert((mesh_grid[0] > edge_br[0] - radius) & (mesh_grid[1] < edge_br[1] + radius))
            
            levelset += self.get_circle(radius, edge_tr + jnp.array([-1.0, -1.0]) * radius) * ((mesh_grid[0] > edge_tr[0] - radius) & (mesh_grid[1] > edge_tr[1] - radius))
            levelset += self.get_circle(radius, edge_tl + jnp.array([ 1.0, -1.0]) * radius) * ((mesh_grid[0] < edge_tl[0] + radius) & (mesh_grid[1] > edge_tl[1] - radius))
            levelset += self.get_circle(radius, edge_bl + jnp.array([ 1.0,  1.0]) * radius) * ((mesh_grid[0] < edge_bl[0] + radius) & (mesh_grid[1] < edge_bl[1] + radius))
            levelset += self.get_circle(radius, edge_br + jnp.array([-1.0,  1.0]) * radius) * ((mesh_grid[0] > edge_br[0] - radius) & (mesh_grid[1] < edge_br[1] + radius))

        return levelset

    def get_sphere(self, radius: float, position: float) -> jnp.ndarray:
        """Creates the levelset field for a sphere.

        :param radius: Radius
        :type radius: float
        :param position: Center position
        :type position: float
        :return: _description_
        :rtype: jnp.ndarray
        """

        mesh_grid = [jnp.meshgrid(*self.cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
        for i in range(len(mesh_grid)):
            mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")
        levelset = - radius + jnp.sqrt(sum([(mesh_grid[i] - position[i])**2 for i in range(3)]))
        
        return levelset

    def create_levelset(self) -> jnp.ndarray:
        """Creates the levelset field either from the user defined lambda or from the user defined building blocks.

        :return: Levelset buffer
        :rtype: jnp.ndarray
        """

        # CREATE BUFFER
        levelset_cutoff     = self.narrow_band_cutoff * self.smallest_cell_size
        levelset_buffer     = levelset_cutoff*jnp.ones((self.nx + 2*self.nh if self.nx > 1 else self.nx, self.ny + 2*self.nh if self.ny > 1 else self.ny, self.nz + 2*self.nh if self.nz > 1 else self.nz))

        # INPUT FOR LAMBDAS
        mesh_grid           = [jnp.meshgrid(*self.cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
        for i in range(len(mesh_grid)):
            mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")

        # FROM LAMBDA FUNCTION
        if type(self.initial_levelset) == types.LambdaType:
            
            levelset        = self.initial_levelset(*mesh_grid)
            levelset        = self.unit_handler.non_dimensionalize(levelset, "length")
            levelset_buffer = levelset_buffer.at[self.nhx, self.nhy, self.nhz].set(levelset)

        # FROM BUILDING BLOCKS
        else:

            for levelset_object in self.initial_levelset:
                
                levelset            = self.shape_function_dict[levelset_object["shape"]](**levelset_object["parameters"])
                levelset            = self.unit_handler.non_dimensionalize(levelset, "length")
                bounding_domain     = levelset_object["bounding_domain"]
                mask                = bounding_domain(*mesh_grid)
                levelset_buffer     = levelset_buffer.at[self.nhx, self.nhy, self.nhz].mul(1.0 - mask)
                levelset_buffer     = levelset_buffer.at[self.nhx, self.nhy, self.nhz].add(levelset * mask)                         

        return levelset_buffer