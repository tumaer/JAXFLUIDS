from __future__ import annotations
from typing import Callable, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.config import precision
from jaxfluids.math.power_functions import cubed, squared
from jaxfluids.levelset.geometry.nn_interface_reconstruction import load_nn, nn_eval, prepare_levelset, compute_mean_apertures
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetGeometryComputationSetup

Array = jax.Array

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
            geometry_setup: LevelsetGeometryComputationSetup,
            halo_cells_geometry: int,
            narrowband_computation: int
            ) -> None:

        self.eps = precision.get_eps()

        self.narrowband_computation = narrowband_computation
        derivative_stencil_normal = geometry_setup.derivative_stencil_normal
        derivative_stencil_curvature = geometry_setup.derivative_stencil_curvature

        self.interface_reconstruction_method = geometry_setup.interface_reconstruction_method
        self.symmetries_nn = geometry_setup.symmetries_nn
        path_nn = geometry_setup.path_nn
        
        if self.interface_reconstruction_method == "NEURALNETWORK":
            self.nn_params_vf, self.nn_params_ap = load_nn(path_nn)
        else:
            self.nn_params_vf, self.nn_params_ap = None, None

        self.derivative_stencil_normal: SpatialDerivative = derivative_stencil_normal(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes,
            offset=halo_cells_geometry)

        nh_stencil = derivative_stencil_curvature.required_halos
        self.derivative_stencil_curvature_1: SpatialDerivative = derivative_stencil_curvature(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes,
            offset=halo_cells_geometry + nh_stencil)
        self.derivative_stencil_curvature_2: SpatialDerivative = derivative_stencil_curvature(
            nh=halo_cells_geometry + nh_stencil,
            inactive_axes=domain_information.inactive_axes,
            offset=halo_cells_geometry)

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


        if self.interface_reconstruction_method == "MARCHINGSQUARES":
                
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

        elif self.interface_reconstruction_method == "NEURALNETWORK":
            # NOTE this is only avaiable in 2D 
            nx,ny,nz = self.domain_information.device_number_of_cells
            nh = self.domain_information.nh_conservatives
            nh_ = self.domain_information.nh_geometry
            offset = nh - (nh_+1)
            dx = self.domain_information.smallest_cell_size
            levelset = prepare_levelset(levelset/dx, offset, self.symmetries_nn)
            if self.symmetries_nn > 1:
                apertures = jax.vmap(nn_eval, in_axes=(None,0), out_axes=(0))(self.nn_params_ap, levelset)
                apertures = apertures.reshape(self.symmetries_nn,nx+2*(nh_+1),ny+2*(nh_+1),1,4)
                apertures = compute_mean_apertures(apertures, self.symmetries_nn)
                volume_fraction = jax.vmap(nn_eval, in_axes=(None,0), out_axes=(0))(self.nn_params_vf, levelset)
                volume_fraction = jnp.mean(volume_fraction, axis=0)
                volume_fraction = volume_fraction.reshape(nx+2*(nh_+1),ny+2*(nh_+1),1)
                volume_fraction = volume_fraction[1:-1,1:-1,:]
            else:
                apertures = nn_eval(self.nn_params_ap, levelset)
                apertures = apertures.reshape(nx+2*(nh_+1),ny+2*(nh_+1),1,4)
                apertures = compute_mean_apertures(apertures, self.symmetries_nn)
                volume_fraction = nn_eval(self.nn_params_vf, levelset)
                volume_fraction = volume_fraction.reshape(nx+2*(nh_+1),ny+2*(nh_+1),1)
                volume_fraction = volume_fraction[1:-1,1:-1,:]

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

        corner_values *= jnp.abs(corner_values) > self.eps

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
            # NOTE convention, if interface lies on cell face, aperture is 0
            apertures = jnp.where(corner_values > 0.0, 1.0, 0.0)

        else:
            cut_cell_types = 0
            for i, s_ in enumerate(self.rearrange_for_aperture_lambdas_slices[axis]):
                # NOTE convention, if interface lies on cell face, aperture is 0
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
        
        if self.dim == 1:
            dV = dx
        elif self.dim == 2:
            dV = dx*dx
        elif self.dim == 3:
            dV = dx*dx*dx

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
        volume_fraction *= self.volume_reconstruction_factor[self.dim-1]

        volume_fraction = jnp.clip(volume_fraction, 0.0, 1.0)

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
        # interface_length = jnp.sqrt(interface_length + 1e-20) # TODO custom derivative with epsilon
        interface_length = jnp.sqrt(interface_length)
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
        gradient_length = jnp.sqrt(jnp.sum(gradient*gradient, axis=0) + 1e-30) # TODO EPS
        normal = gradient/gradient_length
        return normal

    def compute_normal_apertures_based(
        self, 
        apertures: Tuple[Array],
        ) -> Array:
        active_axes_indices = self.domain_information.active_axes_indices
        nx,ny,nz = self.domain_information.device_number_of_cells
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nh_ = self.domain_information.nh_geometry

        aperture_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]
        delta_apertures = []
        for i in range(3):
            if i in active_axes_indices:
                s0,s1 = aperture_slices[i]
                apertures_xi = apertures[i]
                delta_aperture_xi = apertures_xi[s0] - apertures_xi[s1]
            else:
                delta_aperture_xi = jnp.zeros((
                    nx+2*nh_ if 0 in active_axes_indices else 1,
                    ny+2*nh_ if 1 in active_axes_indices else 1,
                    nz+2*nh_ if 2 in active_axes_indices else 1
                    ))
            delta_apertures.append(delta_aperture_xi)
        delta_apertures = jnp.stack(delta_apertures,axis=0)
        normal_aperture_based = delta_apertures/(jnp.linalg.norm(delta_apertures, axis=0, ord=2) + 1e-100)
        return normal_aperture_based

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
        mask = jnp.where(jnp.abs(levelset[nhx__,nhy__,nhz__])/dx <= self.narrowband_computation, 1, 0)

        curvature = []
        for i, axis in enumerate(self.active_axes_indices):
            derivative_xi = self.derivative_stencil_curvature_2.derivative_xi(
                normal[i], self.cell_size, axis)
            curvature.append(derivative_xi)
        curvature = sum(curvature)
        curvature *= mask
        curvature = self.dim * curvature / (self.dim - levelset[self.nhx__,self.nhy__,self.nhz__]*curvature)

        return curvature