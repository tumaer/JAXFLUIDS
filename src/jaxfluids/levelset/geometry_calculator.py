from __future__ import annotations
from functools import partial
from typing import Callable, Tuple, List, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.derivative.deriv_center_2 import DerivativeSecondOrderCenter
from jaxfluids.config import precision
from jaxfluids.math.power_functions import cubed, squared
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup import LevelsetSetup

class GeometryCalculator:
    """The GeometryCalculator class implements functionality
    to compute geometrical quantities that are required for
    two-phase simulations, i.e., volume fraction, apertures,
    interface normal and interface curvature. The volume fraction
    and the apertures are computed by linear interpolation
    of the levelset function. Interface normal and curvature
    are computed with user specified finite difference
    stencils.
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            levelset_setup: LevelsetSetup
            ) -> None:

        self.eps = precision.get_eps()

        geometry_setup = levelset_setup.geometry
        self.narrowband_computation = levelset_setup.narrowband.computation_width
        derivative_stencil_normal = geometry_setup.derivative_stencil_normal
        derivative_stencil_curvature = geometry_setup.derivative_stencil_curvature

        self.derivative_stencil_normal: SpatialDerivative = derivative_stencil_normal(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = levelset_setup.halo_cells)

        nh_stencil = derivative_stencil_curvature.required_halos
        self.derivative_stencil_curvature_1: SpatialDerivative = derivative_stencil_curvature(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes,
            offset = levelset_setup.halo_cells + nh_stencil)
        self.derivative_stencil_curvature_2: SpatialDerivative = derivative_stencil_curvature(
            nh = levelset_setup.halo_cells + nh_stencil,
            inactive_axes = domain_information.inactive_axes,
            offset = levelset_setup.halo_cells)

        self.subcell_reconstruction = geometry_setup.subcell_reconstruction

        self.domain_information = domain_information

        self.dim = domain_information.dim
        self.n = domain_information.nh_conservatives - domain_information.nh_geometry
        self.nhx__, self.nhy__, self.nhz__ = domain_information.domain_slices_conservatives_to_geometry
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.active_axes_indices = domain_information.active_axes_indices
        self.inactive_axes_indices = domain_information.inactive_axes_indices
        self.active_axes = domain_information.active_axes
        self.inactive_axes = domain_information.inactive_axes

        number_of_cells = self.domain_information.device_number_of_cells
        nh_geometry = self.domain_information.nh_geometry
        self.corner_values_subcell_shape = []
        for i in range(3):
            if i in self.active_axes_indices:
                cells = int((number_of_cells[i] + 2*nh_geometry)*2) + 1
                self.corner_values_subcell_shape.append(cells)
            else:
                self.corner_values_subcell_shape.append(1)

        # LEVELSET MUST BE ON FINEST GRID WHERE DX=DY=DZ !!!
        dx_min = self.domain_information.smallest_cell_size
        self.cell_size = dx_min


        # SLICE OBJECTS
        # 1D
        if "".join(domain_information.active_axes) == "x":
            self.single_cell_interpolation_slices = [
                jnp.s_[self.n-1:-self.n, self.nhy__, self.nhz__],
                jnp.s_[self.n:-self.n+1, self.nhy__, self.nhz__],
            ]
            self.levelset_center_value_slices = [
                jnp.s_[:-1,:,:], jnp.s_[1:,:,:]
            ]   
        elif "".join(domain_information.active_axes) == "y":
            self.single_cell_interpolation_slices = [
                jnp.s_[self.nhx__, self.n-1:-self.n, self.nhz__],
                jnp.s_[self.nhx__, self.n:-self.n+1, self.nhz__],
            ]
            self.levelset_center_value_slices = [
                jnp.s_[:,:-1,:], jnp.s_[:,1:,:]
            ]   
        elif "".join(domain_information.active_axes) == "z":
            self.single_cell_interpolation_slices = [
                jnp.s_[self.nhx__, self.nhy__, self.n-1:-self.n],
                jnp.s_[self.nhx__, self.nhy__, self.n:-self.n+1],
            ]
            self.levelset_center_value_slices = [
                jnp.s_[:,:,:-1], jnp.s_[:,:,1:]
            ]   

        # 2D
        elif "".join(domain_information.active_axes) == "xy":
            self.single_cell_interpolation_slices = [
                jnp.s_[self.n-1:-self.n, self.n-1:-self.n, self.nhz__],
                jnp.s_[self.n:-self.n+1, self.n-1:-self.n, self.nhz__],
                jnp.s_[self.n-1:-self.n, self.n:-self.n+1, self.nhz__],
                jnp.s_[self.n:-self.n+1, self.n:-self.n+1, self.nhz__],
            ]
            self.subcell_interpolation_slices = [
                [jnp.s_[self.n-1:-self.n, self.nhy__, self.nhz__], jnp.s_[self.n:-self.n+1, self.nhy__, self.nhz__]],
                [jnp.s_[self.nhx__, self.n-1:-self.n, self.nhz__], jnp.s_[self.nhx__, self.n:-self.n+1, self.nhz__]]
            ]
            self.set_subcell_buffer_slices = [ jnp.s_[::2,1::2,:], jnp.s_[1::2,::2,:], jnp.s_[::2,::2,:], jnp.s_[1::2,1::2,:] ]
            self.levelset_center_value_slices = [ jnp.s_[:-1,:-1,:], jnp.s_[1:,:-1,:], jnp.s_[:-1,1:,:], jnp.s_[1:,1:,:] ]    
            self.volume_fraction_subcell_interpolation_slices   = [ jnp.s_[::2,::2,:], jnp.s_[1::2,::2,:], jnp.s_[::2,1::2,:], jnp.s_[1::2,1::2,:] ]    
            self.aperture_subcell_interpolation_slices          = [
                [jnp.s_[::2,::2,:], jnp.s_[::2,1::2,:]],
                [jnp.s_[::2,::2,:], jnp.s_[1::2,::2,:]],
                [None]
            ]
            self.rearrange_for_aperture_lambdas_slices = [
                [jnp.s_[:,:-1,:], jnp.s_[:,1:,:]],
                [jnp.s_[:-1,:,:], jnp.s_[1:,:,:]],
                [None],
            ]
        elif "".join(domain_information.active_axes) == "xz":
            self.single_cell_interpolation_slices = [
                jnp.s_[self.n-1:-self.n, self.nhy__, self.n-1:-self.n],
                jnp.s_[self.n:-self.n+1, self.nhy__, self.n-1:-self.n],
                jnp.s_[self.n-1:-self.n, self.nhy__, self.n:-self.n+1],
                jnp.s_[self.n:-self.n+1, self.nhy__, self.n:-self.n+1],
            ]
            self.subcell_interpolation_slices = [
                [jnp.s_[self.n-1:-self.n, self.nhy__, self.nhz__], jnp.s_[self.n:-self.n+1, self.nhy__, self.nhz__]],
                [jnp.s_[self.nhx__, self.nhy__, self.n-1:-self.n], jnp.s_[self.nhx__, self.nhy__, self.n:-self.n+1]]
            ]
            self.set_subcell_buffer_slices = [ jnp.s_[::2,:,1::2], jnp.s_[1::2,:,::2], jnp.s_[::2,:,::2], jnp.s_[1::2,:,1::2] ]
            self.levelset_center_value_slices = [ jnp.s_[:-1,:,:-1], jnp.s_[1:,:,:-1], jnp.s_[:-1,:,1:], jnp.s_[1:,:,1:] ]    
            self.volume_fraction_subcell_interpolation_slices   = [ jnp.s_[::2,:,::2], jnp.s_[1::2,:,::2], jnp.s_[::2,:,1::2], jnp.s_[1::2,:,1::2] ]    
            self.aperture_subcell_interpolation_slices          = [
                [jnp.s_[::2,:,::2], jnp.s_[::2,:,1::2]],
                [None],
                [jnp.s_[::2,:,::2], jnp.s_[1::2,:,::2]]
            ]
            self.rearrange_for_aperture_lambdas_slices = [
                [jnp.s_[:,:,:-1], jnp.s_[:,:,1:]],
                [None],
                [jnp.s_[:-1,:,:], jnp.s_[1:,:,:]],
            ]
        elif "".join(domain_information.active_axes) == "yz":
            self.single_cell_interpolation_slices = [
                jnp.s_[self.nhx__, self.n-1:-self.n, self.n-1:-self.n],
                jnp.s_[self.nhx__, self.n:-self.n+1, self.n-1:-self.n],
                jnp.s_[self.nhx__, self.n-1:-self.n, self.n:-self.n+1],
                jnp.s_[self.nhx__, self.n:-self.n+1, self.n:-self.n+1],
            ]
            self.subcell_interpolation_slices = [
                [jnp.s_[self.nhx__, self.n-1:-self.n, self.nhz__], jnp.s_[self.nhx__, self.n:-self.n+1, self.nhz__]],
                [jnp.s_[self.nhx__, self.nhy__, self.n-1:-self.n], jnp.s_[self.nhx__, self.nhy__, self.n:-self.n+1]]
            ]
            self.set_subcell_buffer_slices = [ jnp.s_[:,::2,1::2], jnp.s_[:,1::2,::2], jnp.s_[:,::2,::2], jnp.s_[:,1::2,1::2] ]
            self.levelset_center_value_slices = [ jnp.s_[:,:-1,:-1], jnp.s_[:,1:,:-1], jnp.s_[:,:-1,1:], jnp.s_[:,1:,1:] ]    
            self.volume_fraction_subcell_interpolation_slices   = [ jnp.s_[:,::2,::2], jnp.s_[:,1::2,::2], jnp.s_[:,::2,1::2], jnp.s_[:,1::2,1::2] ]    
            self.aperture_subcell_interpolation_slices          = [
                [None],
                [jnp.s_[:,::2,::2], jnp.s_[:,::2,1::2]],
                [jnp.s_[:,::2,::2], jnp.s_[:,1::2,::2]]
            ]
            self.rearrange_for_aperture_lambdas_slices = [
                [None],
                [jnp.s_[:,:,:-1], jnp.s_[:,:,1:]],
                [jnp.s_[:,:-1,:], jnp.s_[:,1:,:]],
            ]
        
        # 3D
        elif "".join(domain_information.active_axes) == "xyz":

            # SLICES TO COMPUTE THE CORNER
            # VALUES OF THE SINGLE CELL
            self.single_cell_interpolation_slices = [
                jnp.s_[self.n-1:-self.n, self.n-1:-self.n, self.n-1:-self.n],
                jnp.s_[self.n:-self.n+1, self.n-1:-self.n, self.n-1:-self.n],
                jnp.s_[self.n-1:-self.n, self.n:-self.n+1, self.n-1:-self.n],
                jnp.s_[self.n:-self.n+1, self.n:-self.n+1, self.n-1:-self.n],
                jnp.s_[self.n-1:-self.n, self.n-1:-self.n, self.n:-self.n+1],
                jnp.s_[self.n:-self.n+1, self.n-1:-self.n, self.n:-self.n+1],
                jnp.s_[self.n-1:-self.n, self.n:-self.n+1, self.n:-self.n+1],
                jnp.s_[self.n:-self.n+1, self.n:-self.n+1, self.n:-self.n+1],
            ]

            # SLICES TO COMPUTE THE CORNER VALUES
            # OF THE SUBCELL AND SET THEM IN THE BUFFER 
            self.subcell_interpolation_slices = [
                [   jnp.s_[self.n-1:-self.n,self.nhy__,self.nhz__],          jnp.s_[self.n:-self.n+1,self.nhy__,self.nhz__]       ],
                [   jnp.s_[self.nhx__,self.n-1:-self.n,self.nhz__],          jnp.s_[self.nhx__,self.n:-self.n+1,self.nhz__]       ],
                [   jnp.s_[self.nhx__,self.nhy__,self.n-1:-self.n],          jnp.s_[self.nhx__,self.nhy__,self.n:-self.n+1]       ],
                [   jnp.s_[self.n-1:-self.n,self.n-1:-self.n,self.nhz__],    jnp.s_[self.n:-self.n+1,self.n-1:-self.n,self.nhz__], 
                    jnp.s_[self.n-1:-self.n,self.n:-self.n+1,self.nhz__],    jnp.s_[self.n:-self.n+1,self.n:-self.n+1,self.nhz__] ],
                [   jnp.s_[self.n-1:-self.n,self.nhy__,self.n-1:-self.n],    jnp.s_[self.n:-self.n+1,self.nhy__,self.n-1:-self.n],
                    jnp.s_[self.n-1:-self.n,self.nhy__,self.n:-self.n+1],    jnp.s_[self.n:-self.n+1,self.nhy__,self.n:-self.n+1] ],
                [   jnp.s_[self.nhx__,self.n-1:-self.n,self.n-1:-self.n],    jnp.s_[self.nhx__,self.n:-self.n+1,self.n-1:-self.n],
                    jnp.s_[self.nhx__,self.n-1:-self.n,self.n:-self.n+1],    jnp.s_[self.nhx__,self.n:-self.n+1,self.n:-self.n+1] ]
            ]   
            self.set_subcell_buffer_slices = [
                jnp.s_[::2,1::2,1::2], jnp.s_[1::2, ::2,1::2], jnp.s_[1::2,1::2,::2],
                jnp.s_[::2, ::2,1::2], jnp.s_[::2 ,1::2, ::2], jnp.s_[1::2,::2, ::2],
                jnp.s_[::2, ::2, ::2], jnp.s_[1::2,1::2,1::2]
                ]

            # SLICES TO INTERPOLATE THE LEVELSET CENTER VALUE
            # (NEEDED FOR VOLUME FRACTION RECONSTRUCTION) AND SUBCELL AVERAGING
            self.levelset_center_value_slices = [
                jnp.s_[:-1,:-1,:-1], jnp.s_[1:,:-1,:-1], jnp.s_[:-1,1:,:-1], jnp.s_[1:,1:,:-1],
                jnp.s_[:-1,:-1, 1:], jnp.s_[1:,:-1, 1:], jnp.s_[:-1,1:, 1:], jnp.s_[1:,1:, 1:] ]    
            self.volume_fraction_subcell_interpolation_slices = [
                jnp.s_[  ::2,  ::2, ::2], jnp.s_[1::2,  ::2, ::2], jnp.s_[  ::2,1::2, ::2], jnp.s_[1::2,1::2, ::2], 
                jnp.s_[:-1:2,:-1:2,1::2], jnp.s_[1::2,:-1:2,1::2], jnp.s_[:-1:2,1::2,1::2], jnp.s_[1::2,1::2,1::2]
            ]    
            self.aperture_subcell_interpolation_slices = [
                [jnp.s_[::2,::2,::2], jnp.s_[::2,1::2,::2], jnp.s_[::2,::2,1::2], jnp.s_[::2,1::2,1::2]],
                [jnp.s_[::2,::2,::2], jnp.s_[1::2,::2,::2], jnp.s_[::2,::2,1::2], jnp.s_[1::2,::2,1::2]],
                [jnp.s_[::2,::2,::2], jnp.s_[1::2,::2,::2], jnp.s_[::2,1::2,::2], jnp.s_[1::2,1::2,::2]],
            ]

            # SLICES TO REARRANGE THE CORNER VALUES BUFFER TO COMPUTE THE APERTURES
            self.rearrange_for_aperture_lambdas_slices = [
                [jnp.s_[:,:-1,:-1], jnp.s_[:,1:,:-1], jnp.s_[:,1:,1:], jnp.s_[:,:-1,1:]],
                [jnp.s_[:-1,:,:-1], jnp.s_[1:,:,:-1], jnp.s_[1:,:,1:], jnp.s_[:-1,:,1:]],
                [jnp.s_[:-1,:-1,:], jnp.s_[:-1,1:,:], jnp.s_[1:,1:,:], jnp.s_[1:,:-1,:]]
            ]

        self.axis_slices_apertures= [
            [jnp.s_[1:,:,:], jnp.s_[:-1,:,:]],
            [jnp.s_[:,1:,:], jnp.s_[:,:-1,:]],
            [jnp.s_[:,:,1:], jnp.s_[:,:,:-1]],
        ]

        # DIMENSIONAL DEPENDEND FACTORS
        self.interpolation_factor = [0.5, 0.25, 0.125]
        self.volume_reconstruction_factor = [1.0, 0.5, 1.0/3.0]

        # 2D APERTURE FUNCTIONS
        if self.dim == 2:

            def PP(corner_values, axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                return jnp.ones_like(corner_values[s_0])

            def MM(corner_values, axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                return jnp.zeros_like(corner_values[s_0])

            def PM(corner_values, axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                s_1 = self.rearrange_for_aperture_lambdas_slices[axis][1]
                return corner_values[s_0] / ( corner_values[s_0] + corner_values[s_1] + self.eps )

            def MP(corner_values, axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                s_1 = self.rearrange_for_aperture_lambdas_slices[axis][1]
                return corner_values[s_1] / ( corner_values[s_0] + corner_values[s_1] + self.eps )

            self.aperture_functions = [MM, PM, MP, PP]

        # 3D APERTURE FUNCTIONS 
        if self.dim == 3:
            
            def get_slices(axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                s_1 = self.rearrange_for_aperture_lambdas_slices[axis][1]
                s_2 = self.rearrange_for_aperture_lambdas_slices[axis][2]
                s_3 = self.rearrange_for_aperture_lambdas_slices[axis][3]
                return s_0, s_1, s_2, s_3

            def PPPP(corner_values, axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                return jnp.ones_like(corner_values[s_0])

            def MMMM(corner_values, axis):
                s_0 = self.rearrange_for_aperture_lambdas_slices[axis][0]
                return jnp.zeros_like(corner_values[s_0])
            
            def PMMM(corner_values, axis):
                s_0, s_1, _, s_3 = get_slices(axis)
                aperture_0 = corner_values[s_0] / ( corner_values[s_0] + corner_values[s_3] + self.eps )
                aperture_1 = corner_values[s_0] / ( corner_values[s_0] + corner_values[s_1] + self.eps )
                aperture = 0.5 * (aperture_0 * aperture_1)
                return aperture

            def MPMM(corner_values, axis):
                s_0, s_1, s_2, _ = get_slices(axis)
                aperture_0 = corner_values[s_1] / ( corner_values[s_1] + corner_values[s_0] + self.eps )
                aperture_1 = corner_values[s_1] / ( corner_values[s_1] + corner_values[s_2] + self.eps )
                aperture = 0.5 * (aperture_0 * aperture_1)
                return aperture
     
            def MMPM(corner_values, axis):
                _, s_1, s_2, s_3 = get_slices(axis)
                aperture_0 = corner_values[s_2] / ( corner_values[s_2] + corner_values[s_1] + self.eps )
                aperture_1 = corner_values[s_2] / ( corner_values[s_2] + corner_values[s_3] + self.eps )
                aperture = 0.5 * (aperture_0 * aperture_1)
                return aperture

            def MMMP(corner_values, axis):
                s_0, _, s_2, s_3 = get_slices(axis)
                aperture_0 = corner_values[s_3] / ( corner_values[s_3] + corner_values[s_2] + self.eps )
                aperture_1 = corner_values[s_3] / ( corner_values[s_3] + corner_values[s_0] + self.eps )
                aperture = 0.5 * (aperture_0 * aperture_1)
                return aperture
            
            def PPMM(corner_values, axis):
                s_0, s_1, s_2, s_3 = get_slices(axis)
                aperture_0 = corner_values[s_0] / ( corner_values[s_0] + corner_values[s_3] + self.eps )
                aperture_1 = corner_values[s_1] / ( corner_values[s_1] + corner_values[s_2] + self.eps )
                aperture = 0.5 * (aperture_0 + aperture_1)
                return aperture
            
            def PMMP(corner_values, axis):
                s_0, s_1, s_2, s_3 = get_slices(axis)
                aperture_0 = corner_values[s_0] / ( corner_values[s_0] + corner_values[s_1] + self.eps )
                aperture_1 = corner_values[s_3] / ( corner_values[s_3] + corner_values[s_2] + self.eps )
                aperture = 0.5 * (aperture_0 + aperture_1)
                return aperture

            def MPPP(corner_values, axis):
                return 1.0 - PMMM(corner_values, axis)
            def PMPP(corner_values, axis):
                return 1.0 - MPMM(corner_values, axis)
            def PPMP(corner_values, axis):
                return 1.0 - MMPM(corner_values, axis)
            def PPPM(corner_values, axis):
                return 1.0 - MMMP(corner_values, axis)
            def MMPP(corner_values, axis):
                return 1.0 - PPMM(corner_values, axis)
            def MPPM(corner_values, axis):
                return 1.0 - PMMP(corner_values, axis)

            def PMPM(corner_values, axis):
                center_value = 0.0
                for s_ in get_slices(axis):
                    center_value += corner_values[s_]
                aperture_0 = 1.0 - (MPMM(corner_values, axis) + MMMP(corner_values, axis))
                aperture_1 = PMMM(corner_values, axis) + MMPM(corner_values, axis)
                aperture_0 *= (center_value > 0.0)
                aperture_1 *= (center_value <= 0.0)
                return aperture_0 + aperture_1

            def MPMP(corner_values, axis):
                center_value = 0.0
                for s_ in get_slices(axis):
                    center_value += corner_values[s_]
                aperture_0 = 1.0 - (PMMM(corner_values, axis) + MMPM(corner_values, axis))
                aperture_1 = MPMM(corner_values, axis) + MMMP(corner_values, axis)
                aperture_0 *= (center_value > 0.0)
                aperture_1 *= (center_value <= 0.0)
                return aperture_0 + aperture_1
                
            self.aperture_functions = [
                MMMM, PMMM, MPMM, PPMM,
                MMPM, PMPM, MPPM, PPPM,
                MMMP, PMMP, MPMP, PPMP,
                MMPP, PMPP, MPPP, PPPP,
            ]

        # TO COMPUTE CROSS SECOND DERIVATVES
        self.index_pairs = [(0,1), (0,2), (1,2)]

    def interface_reconstruction(
            self,
            levelset: Array
            ) -> Tuple[Array, Tuple]:
        """Computes the volume fraction and the apertures
        assuming a linear interface within each cell.

        :param levelset: Leveset buffer
        :type levelset: Array
        :return: Tuple of volume fraction and apertures
        :rtype: Tuple[Array, Tuple]
        """
        corner_values = self.compute_corner_values(levelset)
        apertures = []
        for i in range(3):
            if i in self.active_axes_indices:
                aperture_xi = self.compute_apertures(corner_values, i)
            else:
                aperture_xi = None
            apertures.append(aperture_xi)
        volume_fraction = self.compute_volume_fraction(corner_values, apertures, levelset)

        if self.subcell_reconstruction:
            factor = self.interpolation_factor[self.dim - 1]
            volume_fraction_subcell = 0.0
            for s_ in self.volume_fraction_subcell_interpolation_slices:
                volume_fraction_subcell += volume_fraction[s_]
            volume_fraction_subcell *= factor
            volume_fraction = volume_fraction_subcell
            for i in self.active_axes_indices:
                aperture_subcell = 0.0
                for s_ in self.aperture_subcell_interpolation_slices[i]:
                    aperture_xi = apertures[i]
                    aperture_subcell += aperture_xi[s_]
                aperture_subcell *= 2*factor
                apertures[i] = aperture_subcell
        
        return volume_fraction, tuple(apertures)

    def compute_corner_values(
            self,
            levelset: Array
            ) -> Array:
        """Linear interpolation of the levelset values
        at the cell center to the corners of the cells.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Levelset values at the corners of cells
        :rtype: Array
        """
        factor = self.interpolation_factor[self.dim - 1]
        corner_values = 0.0
        for s_ in self.single_cell_interpolation_slices:
            corner_values += levelset[s_]
        corner_values *= factor
        if self.subcell_reconstruction: # TODO AARON THIS COSTS TOO MUCH MEMORY ESPECIALLY IN 3D
            corner_values_subcell = jnp.zeros(self.corner_values_subcell_shape)
            interpolations = []
            for interpolation_slices, set_slices in zip(self.subcell_interpolation_slices, self.set_subcell_buffer_slices):
                factor = 1./len(interpolation_slices)
                interpolations = factor * sum([levelset[s_] for s_ in interpolation_slices])
                corner_values_subcell = corner_values_subcell.at[set_slices].set(interpolations)
            corner_values_subcell = corner_values_subcell.at[self.set_subcell_buffer_slices[-2]].set(corner_values)
            corner_values_subcell = corner_values_subcell.at[self.set_subcell_buffer_slices[-1]].set(levelset[self.nhx__,self.nhy__,self.nhz__])
            corner_values = corner_values_subcell
        return corner_values

    def compute_apertures(
            self,
            corner_values: Array,
            axis: int
            ) -> Array:
        """Computes the apertures in axis direction.

        :param corner_values: Levelset values at cell corners
        :type corner_values: Array
        :param axis: spatial axis
        :type axis: int
        :return: Apertures in axis direction
        :rtype: Array
        """

        if self.dim == 1:
            apertures = jnp.where(corner_values > 0.0, 1.0, 0.0)

        else:
            cut_cell_types = 0
            for i, s_ in enumerate(self.rearrange_for_aperture_lambdas_slices[axis]):
                sign = corner_values[s_] > 0.0
                cut_cell_types += sign*2**i
            apertures = 0.0
            for i, aperture_function in enumerate(self.aperture_functions):
                aperture_temp = aperture_function(jnp.abs(corner_values), axis)
                apertures += aperture_temp * (cut_cell_types == i)

        apertures = jnp.clip(apertures, 0.0, 1.0)

        return apertures
        
    def compute_volume_fraction(
            self,
            corner_values: Array,
            apertures: Tuple,
            levelset: Array
            ) -> Array:
        """Computes the volume fraction.

        :param corner_values: Levelset values at cell corners
        :type corner_values: Array
        :param apertures: Apertures
        :type apertures: Tuple
        :return: Volume fraction
        :rtype: Array
        """

        dx = self.cell_size
        if self.subcell_reconstruction:
            dx = self.cell_size * 0.5
        
        if self.dim == 2:
            dV = dx*dx
        elif self.dim == 3:
            dV = dx*dx*dx

        if self.dim == 1:
            nhx__,nhy__,nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
            volume_fraction = 0.5 + levelset[nhx__,nhy__,nhz__]/dx

        else:
            levelset_center_value = 0.0
            for s_ in self.levelset_center_value_slices:
                levelset_center_value += corner_values[s_]
            levelset_center_value *= self.interpolation_factor[self.dim-1]

            interface_length = self.compute_interface_length(apertures)
            volume_fraction = 0.0
            for axis in self.active_axes_indices:
                s_0 = self.axis_slices_apertures[axis][0]
                s_1 = self.axis_slices_apertures[axis][1]
                aperture_i = apertures[axis]
                volume_fraction += 0.5 * (aperture_i[s_0] + aperture_i[s_1])
            volume_fraction += interface_length * levelset_center_value / dV
            volume_fraction *= self.volume_reconstruction_factor[self.dim -1]

        volume_fraction = jnp.clip(volume_fraction, 0.0, 1.0)

        # print(jnp.squeeze((interface_length * levelset_center_value / dV)[:,0,0]))
        # print(jnp.squeeze(interface_length[:,0,0]))
        # print(jnp.squeeze(volume_fraction[:,0,0]))
        # exit()

        return volume_fraction

    def compute_interface_length(
            self,
            apertures: Tuple,
            ) -> Array:
        """Computes the interface
        segment length.

        :param apertures: _description_
        :type apertures: Tuple
        :return: _description_
        :rtype: Array
        """
        dx = self.cell_size
        if self.dim == 1:
            dA = 1.0
        elif self.dim == 2:
            dA = dx
        elif self.dim == 3:
            dA = dx*dx
        interface_length = 0.0
        for axis in self.active_axes_indices:
            s_0 = self.axis_slices_apertures[axis][0]
            s_1 = self.axis_slices_apertures[axis][1]
            aperture_i = apertures[axis]
            interface_length += squared((aperture_i[s_0] - aperture_i[s_1])*dA)
        interface_length = jnp.sqrt(interface_length + 1e-20) # TODO 
        return interface_length

    def compute_normal(
            self,
            levelset: Array
            ) -> Array:
        """Computes the unit normal with the
        stencil specified in the numerical setup.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Normal buffer
        :rtype: Array
        """
        gradient = self.compute_gradient(levelset)
        gradient_length = jnp.sqrt(jnp.sum(gradient*gradient, axis=0) + 1e-20) # TODO EPS
        normal = gradient/gradient_length
        return normal

    def compute_gradient(
            self,
            levelset: Array
            ) -> Array:
        """Computes the gradient of the levelset.

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Array
        """
        shape = levelset[self.nhx__,self.nhy__,self.nhz__].shape
        gradient = []
        for axis in range(3):
            if axis in self.active_axes_indices:
                derivative_xi = self.derivative_stencil_normal.derivative_xi(
                    levelset, self.cell_size, axis)
            else:
                derivative_xi = jnp.zeros(shape)
            gradient.append(derivative_xi)
        gradient = jnp.stack(gradient, axis=0)
        return gradient

    def compute_curvature(self, levelset: Array) -> Array:
        """Computes the curvature with the stencil specified in the numerical setup.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Curvature buffer
        :rtype: Array
        """
        gradient = []
        for axis in self.active_axes_indices:
            derivative_xi = self.derivative_stencil_curvature_1.derivative_xi(
                levelset, self.cell_size, axis)
            gradient.append(derivative_xi)
        gradient = jnp.stack(gradient, axis=0)
        gradient_length = jnp.sqrt(jnp.sum(gradient*gradient, axis=0) + 1e-30)
        normal = gradient/gradient_length

        dx = self.domain_information.smallest_cell_size
        nhx__,nhy__,nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        mask = jnp.where(jnp.abs(levelset[nhx__,nhy__,nhz__])/dx < self.narrowband_computation, 1, 0)

        curvature = []
        for i, axis in enumerate(self.active_axes_indices):
            derivative_xi = self.derivative_stencil_curvature_2.derivative_xi(
                normal[i], self.cell_size, axis)
            curvature.append(derivative_xi)
        curvature = sum(curvature)
        curvature *= mask
        curvature = self.dim * curvature / (self.dim - levelset[self.nhx__,self.nhy__,self.nhz__]*curvature)

        return curvature

def compute_fluid_masks(
        volume_fraction: Array,
        levelset_model: bool
        ) -> Tuple[Array, Array]:
    """Computes the real fluid mask, i.e., 
    cells where the volume fraction is > 0.
    If the levelset model is FLUID-FLUID, 
    the corresponding buffer for both phases
    is returned.

    :param volume_fraction: _description_
    :type volume_fraction: Array
    :param levelset_model: _description_
    :type levelset_model: bool
    :return: _description_
    :rtype: Tuple[Array, Array]
    """

    if levelset_model == "FLUID-FLUID":
        mask_positive = jnp.where( volume_fraction > 0.0, 1, 0 )
        mask_negative = jnp.where( 1.0 - volume_fraction > 0.0, 1, 0 )
        mask_real = jnp.stack([mask_positive, mask_negative], axis=0)
    else:
        mask_positive = jnp.where( volume_fraction > 0.0, 1, 0 )
        mask_real = mask_positive

    return mask_real


def compute_cut_cell_mask(
        levelset: Array,
        nh_offset: int
        ) -> Array:
    """Computes the cut cell mask, i.e., cells
    with different levelset signs compared to
    neighboring cells within the 3x3x3 stencil.

    :param levelset: _description_
    :type levelset: Array
    :param nh_offset: _description_
    :type nh_offset: int
    :return: _description_
    :rtype: Array
    """

    shape = levelset.shape
    active_axes_indices = [i for i in range(3) if shape[i] > 1]
    nhx, nhy, nhz = tuple(
        [jnp.s_[nh_offset:-nh_offset] if
        i in active_axes_indices else
        jnp.s_[:] for i in range(3)]
        )
    index_pairs = [(0,1), (0,2), (1,2)]
    active_planes = [] 
    for i, pair in enumerate(index_pairs):
        if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
            active_planes.append(i)
    dim = len(active_axes_indices)
    nh = nh_offset
    
    s_0 = (nhx,nhy,nhz)

    # II
    s_ii_list = [
        [
            jnp.s_[nh-1:-nh-1,nhy,nhz],
            jnp.s_[nh+1:-nh+1,nhy,nhz],
        ],
        [
            jnp.s_[nhx,nh-1:-nh-1,nhz],
            jnp.s_[nhx,nh+1:-nh+1,nhz],
        ],
        [
            jnp.s_[nhx,nhy,nh-1:-nh-1],
            jnp.s_[nhx,nhy,nh+1:-nh+1]
        ]
    ]
    
    # IJ
    s_ij_list = [
        [  
            jnp.s_[nh-1:-nh-1,nh-1:-nh-1,nhz],
            jnp.s_[nh-1:-nh-1,nh+1:-nh+1,nhz],
            jnp.s_[nh+1:-nh+1,nh-1:-nh-1,nhz],
            jnp.s_[nh+1:-nh+1,nh+1:-nh+1,nhz],
        ],
        [
            jnp.s_[nh-1:-nh-1,nhy,nh-1:-nh-1],
            jnp.s_[nh-1:-nh-1,nhy,nh+1:-nh+1],
            jnp.s_[nh+1:-nh+1,nhy,nh-1:-nh-1],
            jnp.s_[nh+1:-nh+1,nhy,nh+1:-nh+1],
        ],
        [
            jnp.s_[nhx,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[nhx,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[nhx,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[nhx,nh+1:-nh+1,nh+1:-nh+1],
        ]
    ]

    # IJK
    s_ijk_list = [  
            jnp.s_[nh-1:-nh-1,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[nh-1:-nh-1,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[nh-1:-nh-1,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[nh-1:-nh-1,nh+1:-nh+1,nh+1:-nh+1],
            jnp.s_[nh+1:-nh+1,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[nh+1:-nh+1,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[nh+1:-nh+1,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[nh+1:-nh+1,nh+1:-nh+1,nh+1:-nh+1],
        ]
    
    
    mask_cut_cells = jnp.zeros_like(levelset[s_0], dtype=jnp.uint32)
    for axis in active_axes_indices:
        for s_ii in s_ii_list[axis]:
            mask_cut_cells_temp = jnp.where(levelset[s_0]*levelset[s_ii] < 0, 1, 0)
            mask_cut_cells = jnp.maximum(mask_cut_cells, mask_cut_cells_temp)
    
    if dim > 1:
        for i in active_planes:
            for s_ij in s_ij_list[i]:
                mask_cut_cells_temp = jnp.where(levelset[s_0]*levelset[s_ij] < 0, 1, 0)
                mask_cut_cells = jnp.maximum(mask_cut_cells_temp, mask_cut_cells)

    if dim == 3:
        for s_ijk in s_ijk_list:
            mask_cut_cells_temp = jnp.where(levelset[s_0]*levelset[s_ijk] < 0, 1, 0)
            mask_cut_cells = jnp.maximum(mask_cut_cells_temp, mask_cut_cells)

    return mask_cut_cells