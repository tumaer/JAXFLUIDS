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

from typing import Dict, List, Tuple

import jax.numpy as jnp

class DomainInformation:
    """This DomainInformation class holds information about the computational domain, e.g., 
    mesh, number of cells, extension in each spatial direction, active axis, domain slice objects etc..
    """

    
    def __init__(self, dim: int, nx: int, ny: int, nz: int, nh_conservatives: int, nh_geometry: int, domain_size: Dict) -> None:

        self.dim                = dim
        self.cell_centers       = []
        self.cell_faces         = []
        self.cell_sizes         = []
        self.number_of_cells    = []

        # COMPUTE MESH
        number_of_cells = {"x": nx, "y": ny, "z": nz}
        for axis in ["x", "y", "z"]:
            cell_centers_xi, cell_faces_xi, cell_sizes_xi = self.compute_mesh_xi(number_of_cells[axis], domain_size[axis])
            self.cell_centers.append( cell_centers_xi )
            self.cell_faces.append( cell_faces_xi )
            self.cell_sizes.append( cell_sizes_xi )
            self.number_of_cells.append( number_of_cells[axis] )

        # ACTIVE AXIS
        self.active_axis            = []
        self.inactive_axis          = []
        self.active_axis_indices    = []
        self.inactive_axis_indices  = []
        for i, axis in enumerate(["x", "y", "z"]):
            self.active_axis.append( axis ) if self.number_of_cells[i] > 1 else None
            self.inactive_axis.append( axis ) if self.number_of_cells[i] == 1 else None
            self.active_axis_indices.append( i ) if self.number_of_cells[i] > 1 else None
            self.inactive_axis_indices.append( i ) if self.number_of_cells[i] == 1 else None

        # DOMAIN SLICES
        self.nh_conservatives                           = nh_conservatives
        nhx                                             = jnp.s_[:] if "x" in self.inactive_axis else jnp.s_[nh_conservatives:-nh_conservatives]    
        nhy                                             = jnp.s_[:] if "y" in self.inactive_axis else jnp.s_[nh_conservatives:-nh_conservatives]    
        nhz                                             = jnp.s_[:] if "z" in self.inactive_axis else jnp.s_[nh_conservatives:-nh_conservatives]
        self.domain_slices_conservatives                = [nhx, nhy, nhz]
        if nh_geometry != None:
            self.nh_geometry                                = nh_geometry
            nhx                                             = jnp.s_[:] if "x" in self.inactive_axis else jnp.s_[nh_geometry:-nh_geometry]    
            nhy                                             = jnp.s_[:] if "y" in self.inactive_axis else jnp.s_[nh_geometry:-nh_geometry]    
            nhz                                             = jnp.s_[:] if "z" in self.inactive_axis else jnp.s_[nh_geometry:-nh_geometry]
            self.domain_slices_geometry                     = [nhx, nhy, nhz]   
            offset                                          = nh_conservatives - nh_geometry
            nhx                                             = jnp.s_[:] if "x" in self.inactive_axis else jnp.s_[offset:-offset]    
            nhy                                             = jnp.s_[:] if "y" in self.inactive_axis else jnp.s_[offset:-offset]    
            nhz                                             = jnp.s_[:] if "z" in self.inactive_axis else jnp.s_[offset:-offset]
            self.domain_slices_conservatives_to_geometry    = [nhx, nhy, nhz]
        else:
            self.domain_slices_geometry                     = None, None, None   
            self.domain_slices_conservatives_to_geometry    = None, None, None
        
        self.resolution         = jnp.prod(jnp.array(self.number_of_cells))

    def compute_mesh_xi(self, nxi: int, domain_size_xi: List) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Computes cell center coordinates, cell face coordinates and cell sizes in the specified direction

        :param nxi: Number of cells in xi direction
        :type nxi: int
        :param domain_size_xi: Domain size in xi direction
        :type domain_size_xi: List
        :return: Cell center coordinates, cell face coordinates and cell sizes in xi direction
        :rtype: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        """
        cell_sizes_xi   = (domain_size_xi[1] - domain_size_xi[0]) / nxi
        cell_centers_xi = jnp.linspace(domain_size_xi[0] + cell_sizes_xi/2, domain_size_xi[1] - cell_sizes_xi/2, nxi)
        cell_faces_xi   = jnp.linspace(domain_size_xi[0], domain_size_xi[1], nxi+1)
        return cell_centers_xi, cell_faces_xi, cell_sizes_xi