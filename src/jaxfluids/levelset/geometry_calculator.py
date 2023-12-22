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

from typing import Tuple, List

import jax.numpy as jnp
import numpy as np

from jaxfluids.domain_information import DomainInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class GeometryCalculator:
    """The GeometryCalculator class implements functionality to compute geometrical quantities that
    are required for two-phase simulations, i.e., volume fraction, apertures, interface normal and
    interface curvature. The volume fraction and the apertures are computed by linear interpolation
    of the levelset function. Interface normal and curvature are computed with user specified finite difference
    stencils.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, domain_information: DomainInformation, first_derivative_stencil: SpatialDerivative,
        second_derivative_stencil: SpatialDerivative, subcell_reconstruction: bool) -> None:

        self.subcell_reconstruction         = subcell_reconstruction

        self.dim                            = domain_information.dim
        self.cell_centers                   = domain_information.cell_centers
        self.cell_sizes                     = domain_information.cell_sizes
        self.n                              = domain_information.nh_conservatives - domain_information.nh_geometry
        self.nhx__, self.nhy__, self.nhz__  = domain_information.domain_slices_conservatives_to_geometry
        self.nhx_, self.nhy_, self.nhz_     = domain_information.domain_slices_geometry
        self.nhx, self.nhy, self.nhz        = domain_information.domain_slices_conservatives
        self.active_axis_indices            = domain_information.active_axis_indices
        self.inactive_axis_indices          = domain_information.inactive_axis_indices
        self.active_axis                    = domain_information.active_axis
        self.inactive_axis                  = domain_information.inactive_axis

        self.corner_values_subcell_shape    = tuple([(domain_information.number_of_cells[i] + 2*domain_information.nh_geometry)*2 + 1 if i in self.active_axis_indices else 1 for i in range(3)])

        self.derivative_stencil = first_derivative_stencil
        self.second_derivative_stencil = second_derivative_stencil

        # CELL FACE AREA AND CELL VOLUME
        cell_sizes_IR = jnp.array(self.cell_sizes) * 0.5 if self.subcell_reconstruction else jnp.array(self.cell_sizes)
        if self.dim == 3:
            self.cell_face_area      = [cell_sizes_IR[1]*cell_sizes_IR[2], cell_sizes_IR[0]*cell_sizes_IR[2], cell_sizes_IR[0]*cell_sizes_IR[1]]
        elif self.dim == 2:
            self.cell_face_area      = [cell_sizes_IR[1], cell_sizes_IR[0], cell_sizes_IR[2]]
        else:
            self.cell_face_area      = [1.0, 1.0, 1.0]
        self.cell_volume = jnp.prod(jnp.array([cell_sizes_IR[i] for i in self.active_axis_indices]))

        # SLICE OBJECTS
        # 1D
        if "".join(domain_information.active_axis) == "x":
            self.single_cell_interpolation_slices = [
                np.s_[self.n-1:-self.n, self.nhy__, self.nhz__],
                np.s_[self.n:-self.n+1, self.nhy__, self.nhz__],
            ]
            self.levelset_center_value_slices = [
                np.s_[:-1,:,:], np.s_[1:,:,:]
            ]   
        elif "".join(domain_information.active_axis) == "y":
            self.single_cell_interpolation_slices = [
                np.s_[self.nhx__, self.n-1:-self.n, self.nhz__],
                np.s_[self.nhx__, self.n:-self.n+1, self.nhz__],
            ]
            self.levelset_center_value_slices = [
                np.s_[:,:-1,:], np.s_[:,1:,:]
            ]   
        elif "".join(domain_information.active_axis) == "z":
            self.single_cell_interpolation_slices = [
                np.s_[self.nhx__, self.nhy__, self.n-1:-self.n],
                np.s_[self.nhx__, self.nhy__, self.n:-self.n+1],
            ]
            self.levelset_center_value_slices = [
                np.s_[:,:,:-1], np.s_[:,:,1:]
            ]   

        # 2D
        elif "".join(domain_information.active_axis) == "xy":
            self.single_cell_interpolation_slices = [
                np.s_[self.n-1:-self.n, self.n-1:-self.n, self.nhz__],
                np.s_[self.n:-self.n+1, self.n-1:-self.n, self.nhz__],
                np.s_[self.n-1:-self.n, self.n:-self.n+1, self.nhz__],
                np.s_[self.n:-self.n+1, self.n:-self.n+1, self.nhz__],
            ]
            self.subcell_interpolation_slices = [
                [np.s_[self.n-1:-self.n, self.nhy__, self.nhz__], np.s_[self.n:-self.n+1, self.nhy__, self.nhz__]],
                [np.s_[self.nhx__, self.n-1:-self.n, self.nhz__], np.s_[self.nhx__, self.n:-self.n+1, self.nhz__]]
            ]
            self.set_subcell_buffer_slices = [ np.s_[::2,1::2,:], np.s_[1::2,::2,:], np.s_[::2,::2,:], np.s_[1::2,1::2,:] ]
            self.levelset_center_value_slices = [ np.s_[:-1,:-1,:], np.s_[1:,:-1,:], np.s_[:-1,1:,:], np.s_[1:,1:,:] ]    
            self.volume_fraction_subcell_interpolation_slices   = [ np.s_[::2,::2,:], np.s_[1::2,::2,:], np.s_[::2,1::2,:], np.s_[1::2,1::2,:] ]    
            self.aperture_subcell_interpolation_slices          = [
                [np.s_[::2,::2,:], np.s_[::2,1::2,:]],
                [np.s_[::2,::2,:], np.s_[1::2,::2,:]],
                [None]
            ]
            self.rearrange_for_aperture_lambdas_slices = [
                [np.s_[:,:-1,:], np.s_[:,1:,:]],
                [np.s_[:-1,:,:], np.s_[1:,:,:]],
                [None],
            ]
        elif "".join(domain_information.active_axis) == "xz":
            self.single_cell_interpolation_slices = [
                np.s_[self.n-1:-self.n, self.nhy__, self.n-1:-self.n],
                np.s_[self.n:-self.n+1, self.nhy__, self.n-1:-self.n],
                np.s_[self.n-1:-self.n, self.nhy__, self.n:-self.n+1],
                np.s_[self.n:-self.n+1, self.nhy__, self.n:-self.n+1],
            ]
            self.subcell_interpolation_slices = [
                [np.s_[self.n-1:-self.n, self.nhy__, self.nhz__], np.s_[self.n:-self.n+1, self.nhy__, self.nhz__]],
                [np.s_[self.nhx__, self.nhy__, self.n-1:-self.n], np.s_[self.nhx__, self.nhy__, self.n:-self.n+1]]
            ]
            self.set_subcell_buffer_slices = [ np.s_[::2,:,1::2], np.s_[1::2,:,::2], np.s_[::2,:,::2], np.s_[1::2,:,1::2] ]
            self.levelset_center_value_slices = [ np.s_[:-1,:,:-1], np.s_[1:,:,:-1], np.s_[:-1,:,1:], np.s_[1:,:,1:] ]    
            self.volume_fraction_subcell_interpolation_slices   = [ np.s_[::2,:,::2], np.s_[1::2,:,::2], np.s_[::2,:,1::2], np.s_[1::2,:,1::2] ]    
            self.aperture_subcell_interpolation_slices          = [
                [np.s_[::2,:,::2], np.s_[::2,:,1::2]],
                [None],
                [np.s_[::2,:,::2], np.s_[1::2,:,::2]]
            ]
            self.rearrange_for_aperture_lambdas_slices = [
                [np.s_[:,:,:-1], np.s_[:,:,1:]],
                [None],
                [np.s_[:-1,:,:], np.s_[1:,:,:]],
            ]
        elif "".join(domain_information.active_axis) == "yz":
            self.single_cell_interpolation_slices = [
                np.s_[self.nhx__, self.n-1:-self.n, self.n-1:-self.n],
                np.s_[self.nhx__, self.n:-self.n+1, self.n-1:-self.n],
                np.s_[self.nhx__, self.n-1:-self.n, self.n:-self.n+1],
                np.s_[self.nhx__, self.n:-self.n+1, self.n:-self.n+1],
            ]
            self.subcell_interpolation_slices = [
                [np.s_[self.nhx__, self.n-1:-self.n, self.nhz__], np.s_[self.nhx__, self.n:-self.n+1, self.nhz__]],
                [np.s_[self.nhx__, self.nhy__, self.n-1:-self.n], np.s_[self.nhx__, self.nhy__, self.n:-self.n+1]]
            ]
            self.set_subcell_buffer_slices = [ np.s_[:,::2,1::2], np.s_[:,1::2,::2], np.s_[:,::2,::2], np.s_[:,1::2,1::2] ]
            self.levelset_center_value_slices = [ np.s_[:,:-1,:-1], np.s_[:,1:,:-1], np.s_[:,:-1,1:], np.s_[:,1:,1:] ]    
            self.volume_fraction_subcell_interpolation_slices   = [ np.s_[:,::2,::2], np.s_[:,1::2,::2], np.s_[:,::2,1::2], np.s_[:,1::2,1::2] ]    
            self.aperture_subcell_interpolation_slices          = [
                [None],
                [np.s_[:,::2,::2], np.s_[:,::2,1::2]],
                [np.s_[:,::2,::2], np.s_[:,1::2,::2]]
            ]
            self.rearrange_for_aperture_lambdas_slices = [
                [None],
                [np.s_[:,:,:-1], np.s_[:,:,1:]],
                [np.s_[:,:-1,:], np.s_[:,1:,:]],
            ]
        
        # 3D
        elif "".join(domain_information.active_axis) == "xyz":

            # SLICES TO COMPUTE THE CORNER VALUES OF THE SINGLE CELL
            self.single_cell_interpolation_slices = [
                np.s_[self.n-1:-self.n, self.n-1:-self.n, self.n-1:-self.n],
                np.s_[self.n:-self.n+1, self.n-1:-self.n, self.n-1:-self.n],
                np.s_[self.n-1:-self.n, self.n:-self.n+1, self.n-1:-self.n],
                np.s_[self.n:-self.n+1, self.n:-self.n+1, self.n-1:-self.n],
                np.s_[self.n-1:-self.n, self.n-1:-self.n, self.n:-self.n+1],
                np.s_[self.n:-self.n+1, self.n-1:-self.n, self.n:-self.n+1],
                np.s_[self.n-1:-self.n, self.n:-self.n+1, self.n:-self.n+1],
                np.s_[self.n:-self.n+1, self.n:-self.n+1, self.n:-self.n+1],
            ]

            # SLICES TO COMPUTE THE CORNER VALUES OF THE SUBCELL AND SET THEM IN THE BUFFER 
            self.subcell_interpolation_slices = [
                [   np.s_[self.n-1:-self.n,self.nhy__,self.nhz__],          np.s_[self.n:-self.n+1,self.nhy__,self.nhz__]       ],
                [   np.s_[self.nhx__,self.n-1:-self.n,self.nhz__],          np.s_[self.nhx__,self.n:-self.n+1,self.nhz__]       ],
                [   np.s_[self.nhx__,self.nhy__,self.n-1:-self.n],          np.s_[self.nhx__,self.nhy__,self.n:-self.n+1]       ],
                [   np.s_[self.n-1:-self.n,self.n-1:-self.n,self.nhz__],    np.s_[self.n:-self.n+1,self.n-1:-self.n,self.nhz__], 
                    np.s_[self.n-1:-self.n,self.n:-self.n+1,self.nhz__],    np.s_[self.n:-self.n+1,self.n:-self.n+1,self.nhz__] ],
                [   np.s_[self.n-1:-self.n,self.nhy__,self.n-1:-self.n],    np.s_[self.n:-self.n+1,self.nhy__,self.n-1:-self.n],
                    np.s_[self.n-1:-self.n,self.nhy__,self.n:-self.n+1],    np.s_[self.n:-self.n+1,self.nhy__,self.n:-self.n+1] ],
                [   np.s_[self.nhx__,self.n-1:-self.n,self.n-1:-self.n],    np.s_[self.nhx__,self.n:-self.n+1,self.n-1:-self.n],
                    np.s_[self.nhx__,self.n-1:-self.n,self.n:-self.n+1],    np.s_[self.nhx__,self.n:-self.n+1,self.n:-self.n+1] ]
            ]   
            self.set_subcell_buffer_slices = [
                np.s_[::2,1::2,1::2], np.s_[1::2, ::2,1::2], np.s_[1::2,1::2,::2],
                np.s_[::2, ::2,1::2], np.s_[::2 ,1::2, ::2], np.s_[1::2,::2, ::2],
                np.s_[::2, ::2, ::2], np.s_[1::2,1::2,1::2]
                ]

            # SLICES TO INTERPOLATE THE LEVELSET CENTER VALUE (NEEDED FOR VOLUME FRACTION RECONSTRUCTION) AND SUBCELL AVERAGING
            self.levelset_center_value_slices = [
                np.s_[:-1,:-1,:-1], np.s_[1:,:-1,:-1], np.s_[:-1,1:,:-1], np.s_[1:,1:,:-1],
                np.s_[:-1,:-1, 1:], np.s_[1:,:-1, 1:], np.s_[:-1,1:, 1:], np.s_[1:,1:, 1:] ]    
            self.volume_fraction_subcell_interpolation_slices = [
                np.s_[  ::2,  ::2, ::2], np.s_[1::2,  ::2, ::2], np.s_[  ::2,1::2, ::2], np.s_[1::2,1::2, ::2], 
                np.s_[:-1:2,:-1:2,1::2], np.s_[1::2,:-1:2,1::2], np.s_[:-1:2,1::2,1::2], np.s_[1::2,1::2,1::2]
            ]    
            self.aperture_subcell_interpolation_slices = [
                [np.s_[::2,::2,::2], np.s_[::2,1::2,::2], np.s_[::2,::2,1::2], np.s_[::2,1::2,1::2]],
                [np.s_[::2,::2,::2], np.s_[1::2,::2,::2], np.s_[::2,::2,1::2], np.s_[1::2,::2,1::2]],
                [np.s_[::2,::2,::2], np.s_[1::2,::2,::2], np.s_[::2,1::2,::2], np.s_[1::2,1::2,::2]],
            ]

            # SLICES TO REARRANGE THE CORNER VALUES BUFFER IN ORDER TO COMPUTE THE APERTURES USING THE LAMBDA EXPRESSIONS BELOW
            self.rearrange_for_aperture_lambdas_slices = [
                [np.s_[:,:-1,:-1], np.s_[:,1:,:-1], np.s_[:,1:,1:], np.s_[:,:-1,1:]],
                [np.s_[:-1,:,:-1], np.s_[1:,:,:-1], np.s_[1:,:,1:], np.s_[:-1,:,1:]],
                [np.s_[:-1,:-1,:], np.s_[:-1,1:,:], np.s_[1:,1:,:], np.s_[1:,:-1,:]]
            ]

        self.axis_slices_apertures= [
            [np.s_[1:,:,:], np.s_[:-1,:,:]],
            [np.s_[:,1:,:], np.s_[:,:-1,:]],
            [np.s_[:,:,1:], np.s_[:,:,:-1]],
        ]

        # DIMIENSIONAL DEPENDEND FACTORS
        self.interpolation_factor           = [0.5, 0.25, 0.125]
        self.volume_reconstruction_factor   = [1.0, 1.0/2.0, 1.0/3.0]
    
        # BIT TO INT
        self.bitset_to_int = lambda bitset : jnp.sum(jnp.array([bitset[i]*2**i for i in range(len(bitset))]), axis=0)

        # 2D APERTURE FUNCTIONS
        self.PP = lambda corner_values: jnp.ones(corner_values.shape[1:])
        self.MM = lambda corner_values: jnp.zeros(corner_values.shape[1:])
        self.PM = lambda corner_values: corner_values[0] / ( corner_values[0] + corner_values[1] + self.eps)
        self.MP = lambda corner_values: corner_values[1] / ( corner_values[1] + corner_values[0] + self.eps)

        # 3D APERTURE FUNCTIONS 
        self.PPPP = lambda corner_values: jnp.ones(corner_values.shape[1:])
        self.MMMM = lambda corner_values: jnp.zeros(corner_values.shape[1:])

        self.PMMM = lambda corner_values: 0.5 * ( corner_values[0] / ( corner_values[0] + corner_values[3] + self.eps ) * corner_values[0] / ( corner_values[0] + corner_values[1] + self.eps ) )
        self.MPMM = lambda corner_values: 0.5 * ( corner_values[1] / ( corner_values[1] + corner_values[0] + self.eps ) * corner_values[1] / ( corner_values[1] + corner_values[2] + self.eps ) )
        self.MMPM = lambda corner_values: 0.5 * ( corner_values[2] / ( corner_values[2] + corner_values[1] + self.eps ) * corner_values[2] / ( corner_values[2] + corner_values[3] + self.eps ) )
        self.MMMP = lambda corner_values: 0.5 * ( corner_values[3] / ( corner_values[3] + corner_values[1] + self.eps ) * corner_values[3] / ( corner_values[3] + corner_values[2] + self.eps ) )
        self.PPMM = lambda corner_values: 0.5 * ( corner_values[0] / ( corner_values[0] + corner_values[3] + self.eps ) + corner_values[1] / ( corner_values[1] + corner_values[2] + self.eps ) )
        self.PMMP = lambda corner_values: 0.5 * ( corner_values[0] / ( corner_values[0] + corner_values[1] + self.eps ) + corner_values[3] / ( corner_values[3] + corner_values[2] + self.eps ) )

        self.MPPP = lambda corner_values: 1.0 - self.PMMM(corner_values)
        self.PMPP = lambda corner_values: 1.0 - self.MPMM(corner_values)
        self.PPMP = lambda corner_values: 1.0 - self.MMPM(corner_values)
        self.PPPM = lambda corner_values: 1.0 - self.MMMP(corner_values)
        self.MMPP = lambda corner_values: 1.0 - self.PPMM(corner_values)
        self.MPPM = lambda corner_values: 1.0 - self.PMMP(corner_values)

        self.PMPM = lambda corner_values: (jnp.sum(corner_values, axis=0) > 0.0) * (1.0 - (self.MPMM(corner_values) + self.MMMP(corner_values))) + (np.sum(corner_values, axis=0) <= 0.0) * (self.PMMM(corner_values) + self.MMPM(corner_values))
        self.MPMP = lambda corner_values: (jnp.sum(corner_values, axis=0) > 0.0) * (1.0 - (self.PMMM(corner_values) + self.MMPM(corner_values))) + (np.sum(corner_values, axis=0) <= 0.0) * (self.MPMM(corner_values) + self.MMMP(corner_values))

        self.index_pairs = [(0,1), (0,2), (1,2)]

    def linear_interface_reconstruction(self, levelset: jnp.ndarray) -> Tuple[jnp.ndarray, List]:
        """Computes the volume fraction and the apertures assuming a linear interface within each cell.

        :param levelset: Leveset buffer
        :type levelset: jnp.ndarray
        :return: Tuple of volume fraction and apertures
        :rtype: Tuple[jnp.ndarray, List]
        """
        corner_values = self.compute_corner_values(levelset)
        apertures = []
        for i in range(3):
            apertures.append( self.compute_apertures(corner_values, i) if i in self.active_axis_indices else None )
        volume_fraction = self.compute_volume_fraction(corner_values, apertures)
        if self.subcell_reconstruction:
            volume_fraction = self.interpolation_factor[self.dim - 1] * sum([volume_fraction[slices] for slices in self.volume_fraction_subcell_interpolation_slices])
            for i in self.active_axis_indices:
                apertures[i] = 2 * self.interpolation_factor[self.dim - 1] * sum([apertures[i][slices] for slices in self.aperture_subcell_interpolation_slices[i]])
        return volume_fraction, apertures

    def compute_corner_values(self, levelset: jnp.ndarray) -> jnp.ndarray:
        """Linear interpolation of the levelset values at the cell center to the corners of the cells.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :return: Levelset values at the corners of cells
        :rtype: jnp.ndarray
        """
        factor          = self.interpolation_factor[self.dim - 1]
        corner_values   = factor * sum([levelset[slices] for slices in self.single_cell_interpolation_slices])
        if self.subcell_reconstruction:
            corner_values_subcell   = jnp.zeros(self.corner_values_subcell_shape)
            interpolations = []
            for interpolation_slices, set_slices in zip(self.subcell_interpolation_slices, self.set_subcell_buffer_slices):
                factor = 1./len(interpolation_slices)
                interpolations = factor * sum([levelset[s_] for s_ in interpolation_slices])
                corner_values_subcell = corner_values_subcell.at[set_slices].set(interpolations)
            corner_values_subcell = corner_values_subcell.at[self.set_subcell_buffer_slices[-2]].set(corner_values)
            corner_values_subcell = corner_values_subcell.at[self.set_subcell_buffer_slices[-1]].set(levelset[self.nhx__,self.nhy__,self.nhz__])
            corner_values         = corner_values_subcell
        corner_values = corner_values * jnp.where(jnp.abs(corner_values) < self.eps, 0.0, 1.0) 
        return corner_values

    def compute_apertures(self, corner_values: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes the apertures in axis direction.

        :param corner_values: Levelset values at cell corners
        :type corner_values: jnp.ndarray
        :param axis: spatial axis
        :type axis: int
        :return: Apertures in axis direction
        :rtype: jnp.ndarray
        """

        if self.dim == 1:
            apertures = jnp.where(corner_values > 0.0, 1.0, 0.0)

        elif self.dim == 2:
            corner_values_for_apertures         = jnp.stack([corner_values[self.rearrange_for_aperture_lambdas_slices[axis][0]], corner_values[self.rearrange_for_aperture_lambdas_slices[axis][1]]], axis=0)
            corner_values_for_apertures_sign    = jnp.where(corner_values_for_apertures > 0.0, 1, 0)
            apertures           = self.MM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 0, 1.0, 0.0) \
                                + self.PM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 1, 1.0, 0.0) \
                                + self.MP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 2, 1.0, 0.0) \
                                + self.PP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 3, 1.0, 0.0)
        
        elif self.dim == 3:
            corner_values_for_apertures         = jnp.stack([  corner_values[self.rearrange_for_aperture_lambdas_slices[axis][0]],
                                                        corner_values[self.rearrange_for_aperture_lambdas_slices[axis][1]],
                                                        corner_values[self.rearrange_for_aperture_lambdas_slices[axis][2]],
                                                        corner_values[self.rearrange_for_aperture_lambdas_slices[axis][3]]    ], axis=0)
            corner_values_for_apertures_sign    = jnp.where(corner_values_for_apertures > 0.0, 1, 0) 
            apertures           = self.MMMM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 0 , 1.0, 0.0) \
                                + self.PMMM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 1 , 1.0, 0.0) \
                                + self.MPMM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 2 , 1.0, 0.0) \
                                + self.PPMM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 3 , 1.0, 0.0) \
                                + self.MMPM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 4 , 1.0, 0.0) \
                                + self.PMPM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 5 , 1.0, 0.0) \
                                + self.MPPM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 6 , 1.0, 0.0) \
                                + self.PPPM(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 7 , 1.0, 0.0) \
                                + self.MMMP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 8 , 1.0, 0.0) \
                                + self.PMMP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 9 , 1.0, 0.0) \
                                + self.MPMP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 10, 1.0, 0.0) \
                                + self.PPMP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 11, 1.0, 0.0) \
                                + self.MMPP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 12, 1.0, 0.0) \
                                + self.PMPP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 13, 1.0, 0.0) \
                                + self.MPPP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 14, 1.0, 0.0) \
                                + self.PPPP(jnp.abs(corner_values_for_apertures)) * jnp.where(self.bitset_to_int(corner_values_for_apertures_sign) == 15, 1.0, 0.0)  
        
        apertures = jnp.clip(apertures, 0.0, 1.0)
        return apertures
        
    def compute_volume_fraction(self, corner_values: jnp.ndarray, apertures: List) -> jnp.ndarray:
        """Computes the volume fraction.

        :param corner_values: Levelset values at cell corners
        :type corner_values: jnp.ndarray
        :param apertures: Apertures
        :type apertures: List
        :return: Volume fraction
        :rtype: jnp.ndarray
        """
        levelset_center_value   = self.interpolation_factor[self.dim - 1] * sum([corner_values[slices] for slices in self.levelset_center_value_slices])
        interface_length        = jnp.sqrt(sum([((apertures[i][self.axis_slices_apertures[i][0]] - apertures[i][self.axis_slices_apertures[i][1]]) * self.cell_face_area[i] )**2 for i in self.active_axis_indices]))
        volume_fraction         = self.volume_reconstruction_factor[self.dim -1] * ( 0.5 * sum([apertures[i][self.axis_slices_apertures[i][0]] + apertures[i][self.axis_slices_apertures[i][1]] for  i in self.active_axis_indices]) + \
                                    interface_length * levelset_center_value / self.cell_volume )
        volume_fraction         = jnp.clip(volume_fraction, 0.0, 1.0)
        return volume_fraction

    def compute_normal(self, levelset: jnp.ndarray) -> jnp.ndarray:
        """Computes the normal with the stencil specified in the numerical setup.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :return: Normal buffer
        :rtype: jnp.ndarray
        """
        normal = []
        for i in range(3):
            normal.append( self.derivative_stencil.derivative_xi(levelset, self.cell_sizes[i], i) if i in self.active_axis_indices else jnp.zeros(levelset[self.nhx__,self.nhy__,self.nhz__].shape) )
        normal = jnp.stack(normal, axis=0)
        normal = normal / jnp.sqrt( jnp.sum( jnp.square(normal), axis=0 ) + self.eps )
        return normal

    def compute_curvature(self, levelset: jnp.ndarray) -> jnp.ndarray:
        """Computes the curvature with the stencil specified in the numerical setup.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :return: Curvature buffer
        :rtype: jnp.ndarray
        """

        # GRADIENT
        gradient = []
        for i in range(3):
            gradient.append( self.derivative_stencil.derivative_xi(levelset, self.cell_sizes[i], i) if i in self.active_axis_indices else jnp.zeros(levelset[self.nhx__,self.nhy__,self.nhz__].shape) )
        gradient = jnp.stack(gradient, axis=0)

        # SECOND DERIVATIVES
        second_derivative_ii = []
        for i in range(3):
            second_derivative_ii.append( self.second_derivative_stencil.derivative_xi(levelset, self.cell_sizes[i], i) if i in self.active_axis_indices else jnp.zeros(levelset[self.nhx__,self.nhy__,self.nhz__].shape) )
        second_derivative_ij = []
        for (i,j) in self.index_pairs:
            second_derivative_ij.append( self.second_derivative_stencil.derivative_xi_xj(levelset, self.cell_sizes[i], self.cell_sizes[j], i, j) if i in self.active_axis_indices and j in self.active_axis_indices else jnp.zeros(levelset[self.nhx__,self.nhy__,self.nhz__].shape) )
        
        gradient_length = jnp.sqrt( jnp.sum( jnp.square(gradient), axis=0) + self.eps )
        
        curvature = (second_derivative_ii[0] + second_derivative_ii[1] + second_derivative_ii[2]) / gradient_length - \
                    (gradient[0]**2 * second_derivative_ii[0] + gradient[1]**2 * second_derivative_ii[1] + gradient[2]**2 * second_derivative_ii[2] + \
                    2 * (gradient[0] * gradient[1] * second_derivative_ij[0] + gradient[0] * gradient[2] * second_derivative_ij[1] + gradient[1] * gradient[2] * second_derivative_ij[2])) / gradient_length**3
        
        curvature = self.dim * curvature / (self.dim - levelset[self.nhx__,self.nhy__,self.nhz__]*curvature)

        return curvature