import jax.numpy as jnp
from jax import Array
import numpy as np
import matplotlib.pyplot as plt

from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.stencils import DICT_FIRST_DERIVATIVE_CENTER
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.diffuse_interface.helper_functions import smoothed_interface_function
from jaxfluids.config import precision

class DiffuseInterfaceGeometryCalculator:
    """The DiffuseInterfaceGeometryCalculator class implements functionality
    to compute geometrical quantities that are required for two-phase 
    simulations with the diffuse-interface method, i.e., interface normals and 
    curvature. Interface normal and curvature are computed with user-specified
    finite difference stencils.
    """
    
    def __init__(
            self,
            domain_information: DomainInformation,
            diffuse_interface_setup: DiffuseInterfaceSetup,
            halo_manager: HaloManager
            ) -> None:

        self.eps = precision.get_eps()
        # TODO eps_ad
        self.eps_ad = 1e-20

        self.halo_manager = halo_manager

        self.dim = domain_information.dim
        self.domain_information = domain_information
        self.smallest_cell_size = domain_information.smallest_cell_size
        self.n = domain_information.nh_conservatives - domain_information.nh_geometry
        self.nhx__, self.nhy__, self.nhz__ = domain_information.domain_slices_conservatives_to_geometry
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.active_axes_indices = domain_information.active_axes_indices
        self.inactive_axes_indices = domain_information.inactive_axes_indices
        self.active_axes = domain_information.active_axes
        self.inactive_axes = domain_information.inactive_axes
        
        nx, ny, nz = domain_information.global_number_of_cells
        sx, sy, sz = domain_information.split_factors
        self.normal_at_cell_face_shape = self.shape_fluxes = [
                (int(nx/sx+1), int(ny/sy), int(nz/sz)),
                (int(nx/sx), int(ny/sy+1), int(nz/sz)),
                (int(nx/sx), int(ny/sy), int(nz/sz+1))]
        
        geometry_calculation_setup = diffuse_interface_setup.geometry_calculation
        self.steps_curvature = geometry_calculation_setup.steps_curvature
        self.is_correct_curvature = self.steps_curvature > 0
        self.alpha = geometry_calculation_setup.interface_smoothing
        self.volume_fraction_mapping = geometry_calculation_setup.volume_fraction_mapping
        self.surface_tension_kernel = geometry_calculation_setup.surface_tension_kernel
        
        derivative_stencil_curvature = geometry_calculation_setup.derivative_stencil_curvature
        derivative_stencil_face = geometry_calculation_setup.derivative_stencil_face
        derivative_stencil_center = geometry_calculation_setup.derivative_stencil_center
        reconstruction_stencil = geometry_calculation_setup.reconstruction_stencil
        
        # Number of ghost cells of the normal buffer used for calculating
        # uncorrected curvature
        offset_normal_cc = 2 if derivative_stencil_curvature in \
            [DICT_FIRST_DERIVATIVE_CENTER[rec] for rec in ["CENTRAL4", "CENTRAL4_ADAP"]] \
            else 1
        
        nh_conservatives = domain_information.nh_conservatives
        nh_geometry = domain_information.nh_geometry
        is_mesh_stretching = domain_information.is_mesh_stretching
        cell_sizes_with_halos = domain_information.get_global_cell_sizes_halos()
        cell_sizes_with_halos_geometry = domain_information.get_global_cell_sizes_halos_geometry()

        # STENCIL FOR CURVATURE CALCULATION AT CELL CENTER
        # TODO probably will throw a bug since cell_sizes_halos has more
        # halo cells than nh_geometry + offset_normal_cc
        cell_sizes_with_halos_geometry_plus_offset = []
        nh_geometry_plus_offset = nh_conservatives - (nh_geometry + offset_normal_cc)
        for i in range(3):
            if i in self.active_axes_indices and domain_information.is_mesh_stretching[i]:
                if i == 0:
                    s_ = jnp.s_[...,nh_geometry_plus_offset:-nh_geometry_plus_offset,:,:]
                elif i == 1:
                    s_ = jnp.s_[...,:,nh_geometry_plus_offset:-nh_geometry_plus_offset,:]
                elif i == 2:
                    s_ = jnp.s_[...,:,:,nh_geometry_plus_offset:-nh_geometry_plus_offset]
                cell_sizes_with_halos_geometry_plus_offset.append(cell_sizes_with_halos[i][s_])
            else:
                cell_sizes_with_halos_geometry_plus_offset.append(cell_sizes_with_halos[i])

        self._first_derivative_stencil_curvature : SpatialDerivative = \
            derivative_stencil_curvature(
                nh=nh_geometry + offset_normal_cc, 
                inactive_axes=self.inactive_axes,
                offset=nh_geometry,
                is_mesh_stretching=is_mesh_stretching,
                cell_sizes=cell_sizes_with_halos_geometry_plus_offset,)
            
        # STENCIL FOR GRADIENT AT CELL-CENTER CALCULATION
        self._first_derivative_stencil_cell_center_1 : SpatialDerivative = derivative_stencil_center(
            nh=nh_conservatives, 
            inactive_axes=self.inactive_axes,
            offset=nh_geometry + offset_normal_cc,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_with_halos,)

        # STENCILS FOR GRADIENT ON CELL-FACE CALCULATION
        # Stencil for direct derivative evaluation on cell-face
        self._first_derivative_stencil_cell_face : SpatialDerivative = \
            derivative_stencil_face(
                nh=nh_conservatives,
                inactive_axes=self.inactive_axes,
                offset=0,
                is_mesh_stretching=is_mesh_stretching,
                cell_sizes=cell_sizes_with_halos,)
        
        if derivative_stencil_center in \
            [DICT_FIRST_DERIVATIVE_CENTER[rec] for rec in ["CENTRAL2", "CENTRAL2_ADAP"]]:
            offset_gradient_cell_face = 1
        elif derivative_stencil_center in \
            [DICT_FIRST_DERIVATIVE_CENTER[rec] for rec in ["CENTRAL4", "CENTRAL4_ADAP"]]:
            offset_gradient_cell_face = 2
        else:
            raise NotImplementedError
        
        # Stencil for derivative evaluation on cell-center
        self._first_derivative_stencil_cell_center_2 : SpatialDerivative = \
            derivative_stencil_center(
                nh=nh_conservatives,
                inactive_axes=self.inactive_axes,
                offset=offset_gradient_cell_face,
                is_mesh_stretching=is_mesh_stretching,
                cell_sizes=cell_sizes_with_halos,)

        # Stencil for derivative reconstruction cell center 
        # (including offset) to cell face
        self._reconstruct_stencil_cell_face : SpatialReconstruction = \
            reconstruction_stencil(
                nh=nh_conservatives,
                inactive_axes=self.inactive_axes,
                offset=nh_conservatives-offset_gradient_cell_face,
                is_mesh_stretching=is_mesh_stretching,
                cell_sizes=cell_sizes_with_halos,)
            
        # self._reconstruct_stencil_cell_face : SpatialReconstruction = \
        #     reconstruction_stencil(
        #         nh=offset_gradient_cell_face,
        #         inactive_axes=self.inactive_axes,
        #         offset=0)
        
        # STANDALONE DERIVATIVE STENCILS
        self.first_derivative_cell_center: SpatialDerivative = derivative_stencil_center(
            nh=nh_conservatives,
            inactive_axes=self.inactive_axes,
            offset=0,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_with_halos,)
        
        self.first_derivative_stencil_cell_face : SpatialDerivative = derivative_stencil_face(
            nh=nh_conservatives,
            inactive_axes=self.inactive_axes,
            offset=0,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_with_halos,)
        
        # STANDALONE RECONSTRUCTION STENCILS
        # Reconstruct from conservatives to cell face
        self.reconstruction_stencil_conservatives: SpatialReconstruction = \
            reconstruction_stencil(
                nh=nh_conservatives,
                inactive_axes=self.inactive_axes,
                offset=0,
                is_mesh_stretching=is_mesh_stretching,
                cell_sizes=cell_sizes_with_halos,)
        
        # Reconstruct from geometry to cell face
        # TODO BUG cell_sizes_with_halos_geometry required
        self.reconstruction_stencil_geometry: SpatialReconstruction = \
            reconstruction_stencil(
                nh=nh_geometry,
                inactive_axes=self.inactive_axes,
                offset=0,
                is_mesh_stretching=is_mesh_stretching,
                cell_sizes=cell_sizes_with_halos_geometry,)

        # SLICE OBJECTS
        s_geometry_to_domain_plus_one = jnp.s_[(nh_geometry-1):-(nh_geometry-1)] if nh_geometry > 1 \
            else jnp.s_[:]
        nh_normal = nh_geometry + offset_normal_cc
        s_normal = jnp.s_[nh_normal:-nh_normal]
        nh_conservatives_to_normal = nh_conservatives - nh_normal
        s_nh_cons_to_normal = jnp.s_[nh_conservatives_to_normal:-nh_conservatives_to_normal]
        s_normal_to_geometry = jnp.s_[offset_normal_cc:-offset_normal_cc]
        # 1D
        if "".join(domain_information.active_axes) == "x":
            self.domain_slices_normal = jnp.s_[...,s_normal,:,:]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,s_nh_cons_to_normal,:,:]
            self.geometry_to_domain_plus_one = jnp.s_[
                ...,s_geometry_to_domain_plus_one,:,:]
            self.normal_to_geometry = jnp.s_[...,s_normal_to_geometry,:,:]
 
            self.no_neighbors_curvature_correction = 3
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:-2,:,:], jnp.s_[1:-1,:,:], jnp.s_[2:,:,:]]
            self.curvature_correction_center_slice = jnp.s_[1:-1,:,:]
            
        elif "".join(domain_information.active_axes) == "y":
            self.domain_slices_normal = jnp.s_[...,:,s_normal,:]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,:,s_nh_cons_to_normal,:]
            self.geometry_to_domain_plus_one = jnp.s_[...,:,s_geometry_to_domain_plus_one,:]
            self.normal_to_geometry = jnp.s_[...,:,s_normal_to_geometry,:]
        
            self.no_neighbors_curvature_correction = 3
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:,:-2,:], jnp.s_[:,1:-1,:], jnp.s_[:,2:,:]]
            self.curvature_correction_center_slice = jnp.s_[:,1:-1,:]

        elif "".join(domain_information.active_axes) == "z":
            self.domain_slices_normal = jnp.s_[...,:,:,s_normal]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,:,:,s_nh_cons_to_normal]
            self.geometry_to_domain_plus_one = jnp.s_[...,:,:,s_geometry_to_domain_plus_one]
            self.normal_to_geometry = jnp.s_[...,:,:,s_normal_to_geometry]

            self.no_neighbors_curvature_correction = 3
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:,:,:-2], jnp.s_[:,:,1:-1], jnp.s_[:,:,2:]]
            self.curvature_correction_center_slice = jnp.s_[:,:,1:-1]

        # 2D
        elif "".join(domain_information.active_axes) == "xy":
            self.domain_slices_normal = jnp.s_[...,s_normal,s_normal,:]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,s_nh_cons_to_normal,s_nh_cons_to_normal,:]
            self.geometry_to_domain_plus_one = jnp.s_[
                ...,s_geometry_to_domain_plus_one,
                s_geometry_to_domain_plus_one,:]
            self.normal_to_geometry = jnp.s_[
                ...,s_normal_to_geometry,s_normal_to_geometry,:]
            self.no_neighbors_curvature_correction = 9
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:-2, :-2,:], jnp.s_[1:-1, :-2,:], jnp.s_[2:, :-2,:],
                jnp.s_[:-2,1:-1,:], jnp.s_[1:-1,1:-1,:], jnp.s_[2:,1:-1,:],
                jnp.s_[:-2,2:  ,:], jnp.s_[1:-1,2:  ,:], jnp.s_[2:,2:  ,:]]
            self.curvature_correction_center_slice = jnp.s_[...,1:-1,1:-1,:]

        elif "".join(domain_information.active_axes) == "xz":
            self.domain_slices_normal = jnp.s_[...,s_normal,:,s_normal]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,s_nh_cons_to_normal,:,s_nh_cons_to_normal]
            self.geometry_to_domain_plus_one = jnp.s_[
                ...,s_geometry_to_domain_plus_one,
                :,s_geometry_to_domain_plus_one]
            self.normal_to_geometry = jnp.s_[
                ...,s_normal_to_geometry,:,s_normal_to_geometry]
            self.no_neighbors_curvature_correction = 9
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:-2,:, :-2], jnp.s_[1:-1,:, :-2], jnp.s_[2:,:, :-2],
                jnp.s_[:-2,:,1:-1], jnp.s_[1:-1,:,1:-1], jnp.s_[2:,:,1:-1],
                jnp.s_[:-2,:,2:  ], jnp.s_[1:-1,:,2:  ], jnp.s_[2:,:,2:  ]]
            self.curvature_correction_center_slice = jnp.s_[1:-1,:,1:-1]
        
        elif "".join(domain_information.active_axes) == "yz":
            self.domain_slices_normal = jnp.s_[...,:,s_normal,s_normal]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,:,s_nh_cons_to_normal,s_nh_cons_to_normal]
            self.geometry_to_domain_plus_one = jnp.s_[
                ...,:,
                s_geometry_to_domain_plus_one,
                s_geometry_to_domain_plus_one]
            self.normal_to_geometry = jnp.s_[
                ...,:,s_normal_to_geometry,s_normal_to_geometry]
            self.no_neighbors_curvature_correction = 9
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:,:-2, :-2], jnp.s_[:,1:-1, :-2], jnp.s_[:,2:, :-2],
                jnp.s_[:,:-2,1:-1], jnp.s_[:,1:-1,1:-1], jnp.s_[:,2:,1:-1],
                jnp.s_[:,:-2,2:  ], jnp.s_[:,1:-1,2:  ], jnp.s_[:,2:,2:  ]]
            self.curvature_correction_center_slice = jnp.s_[:,1:-1,1:-1]
        
        # 3D
        elif "".join(domain_information.active_axes) == "xyz":
            self.domain_slices_normal = jnp.s_[
                ...,s_normal,s_normal,s_normal]
            self.domain_slices_conservatives_to_normal = jnp.s_[
                ...,s_nh_cons_to_normal,s_nh_cons_to_normal,s_nh_cons_to_normal]
            self.geometry_to_domain_plus_one = jnp.s_[
                ...,
                s_geometry_to_domain_plus_one,
                s_geometry_to_domain_plus_one,
                s_geometry_to_domain_plus_one]
            self.normal_to_geometry = jnp.s_[
                ...,s_normal_to_geometry,s_normal_to_geometry,s_normal_to_geometry]

            self.no_neighbors_curvature_correction = 27
            self.curvature_correction_neighbor_slices = [
                jnp.s_[:-2, :-2, :-2], jnp.s_[1:-1, :-2, :-2], jnp.s_[2:, :-2, :-2],
                jnp.s_[:-2,1:-1, :-2], jnp.s_[1:-1,1:-1, :-2], jnp.s_[2:,1:-1, :-2],
                jnp.s_[:-2,2:  , :-2], jnp.s_[1:-1,2:  , :-2], jnp.s_[2:,2:  , :-2], 
                jnp.s_[:-2, :-2,1:-1], jnp.s_[1:-1, :-2,1:-1], jnp.s_[2:, :-2,1:-1],
                jnp.s_[:-2,1:-1,1:-1], jnp.s_[1:-1,1:-1,1:-1], jnp.s_[2:,1:-1,1:-1],
                jnp.s_[:-2,2:  ,1:-1], jnp.s_[1:-1,2:  ,1:-1], jnp.s_[2:,2:  ,1:-1],
                jnp.s_[:-2, :-2,2:  ], jnp.s_[1:-1, :-2,2:  ], jnp.s_[2:, :-2,2:  ],
                jnp.s_[:-2,1:-1,2:  ], jnp.s_[1:-1,1:-1,2:  ], jnp.s_[2:,1:-1,2:  ],
                jnp.s_[:-2,2:  ,2:  ], jnp.s_[1:-1,2:  ,2:  ], jnp.s_[2:,2:  ,2:  ]]
            self.curvature_correction_center_slice = jnp.s_[1:-1,1:-1,1:-1]
    
    def compute_normal(self, volume_fraction: Array) -> Array:
        """Wrapper around self.compute_normal_(). Computes the normal
        at cell centers with the stencil specified in the numerical setup.
        Volume fraction field is smoothed before derivative is taken.

        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: Array
        :return: Normal buffer
        :rtype: Array
        """
        volume_fraction = self.limit_volume_fraction(volume_fraction)
        if self.volume_fraction_mapping == None:
            vf_projected = volume_fraction
        elif self.volume_fraction_mapping == "SMOOTHING":
            vf_projected = smoothed_interface_function(volume_fraction, self.alpha)
        elif self.volume_fraction_mapping == "SIGNED-DISTANCE":
            vf_projected = self.compute_signed_distance_function(volume_fraction)
        else:
            raise NotImplementedError
        return self.compute_normal_(vf_projected, location="CENTER") 
    
    def compute_normal_(
            self,
            volume_fraction: Array,
            location: str,
            axis: int = None
            ) -> Array:
        """Computes the normal based on the volume fraction field at
        either the cell center or at the cell face in axis-direction.
        Note: It is usually a good idea, to not compute the normal
        directly from the volume fraction field, but rather from a
        pre-processed volume fraction field (either smoothed or
        convert to a pseudo signed distance function).

        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param location: _description_
        :type location: str
        :param axis: _description_, defaults to None
        :type axis: int, optional
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """        
        gradient = self.compute_gradient(volume_fraction, location, axis)
        return gradient / jnp.sqrt(jnp.sum(gradient * gradient, axis=0) + self.eps_ad)
    
    def compute_gradient(
            self,
            buffer: Array,
            location: str,
            axis: int = None
            ) -> Array:
        """Wrapper function for computing the gradient of a quantity
        at either the cell center or at the cell face in axis-direction.

        :param buffer: _description_
        :type buffer: Array
        :param location: _description_
        :type location: str
        :param axis: _description_, defaults to None
        :type axis: int, optional
        :return: _description_
        :rtype: Array
        """
        if location == "CENTER":
            gradient = self.compute_gradient_at_cell_center(buffer)
        elif location == "FACE":
            gradient = self.compute_gradient_at_cell_face_xi(buffer, axis)
        else:
            raise NotImplementedError
        return gradient

    def compute_gradient_at_cell_center(self, buffer: Array) -> Array:
        cell_sizes = self.domain_information.get_device_cell_sizes()
        gradient = []
        for axis in range(3):
            if axis in self.active_axes_indices:
                derivative_axis = self._first_derivative_stencil_cell_center_1.derivative_xi(
                    buffer, cell_sizes[axis], axis)
            else:
                derivative_axis = jnp.zeros_like(
                    buffer[self.domain_slices_conservatives_to_normal])
            gradient.append(derivative_axis)
        return jnp.stack(gradient, axis=0)

    def compute_gradient_at_cell_face_xi(
            self,
            buffer: Array,
            axis: int,
            ) -> Array:
        cell_sizes = self.domain_information.get_device_cell_sizes()
        gradient = []
        for axis_j in range(3):
            if axis_j in self.active_axes_indices:
                if axis_j == axis:
                    derivative_axis_j = self._first_derivative_stencil_cell_face.derivative_xi(
                        buffer, cell_sizes[axis_j], axis=axis_j)
                else:
                    derivative_axis_j_at_cell_center = self._first_derivative_stencil_cell_center_2.derivative_xi(
                        buffer, cell_sizes[axis_j], axis=axis_j)
                    derivative_axis_j = self._reconstruct_stencil_cell_face.reconstruct_xi(
                        derivative_axis_j_at_cell_center, axis=axis)
            else:
                derivative_axis_j = jnp.zeros(self.normal_at_cell_face_shape[axis])
            gradient.append(derivative_axis_j)
        return jnp.stack(gradient, axis=0)
    
    def compute_curvature(self, volume_fraction: Array) -> Array:
        """Computes the corrected curvature (at cell centers)
        with the stencil specified in the numerical setup. The
        raw curvature  is computed from a smoothed volume fraction
        field and is then iteratively corrected.
        
        curvature = div(normal)

        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: Array
        :return: Curvature buffer
        :rtype: Array
        """
        cell_sizes = self.domain_information.get_device_cell_sizes()
        normal = self.compute_normal(volume_fraction)

        # vf_projected = smoothed_interface_function(volume_fraction, self.alpha)
        # normal = self.compute_normal_(vf_projected, location="CENTER") 

        # normal = self.compute_normal_(volume_fraction, location="CENTER") 

        # CURAVTURE = - DIVERGENCE(NORMAL)
        # compute the curvature as the negative 
        # divergence of the normals
        curvature = - sum(self._first_derivative_stencil_curvature.derivative_xi(
            normal[i], cell_sizes[i], i) for i in self.active_axes_indices)

        # ITERATIVE CORRECTION
        if self.is_correct_curvature:
            curvature = self.correct_curvature(
                curvature, 
                volume_fraction[self.nhx__,self.nhy__,self.nhz__])
        return curvature
    
    def correct_curvature(
            self,
            curvature: Array,
            volume_fraction: Array,
            ) -> Array:
        """Iteratively corrects the given curvature value
        by averaging the curvature with neighboring cells.
        
        \kappa^{m+1} = \sum w_l kappa^{m}_l / \sum w_l
        
        w_l = (phi * (1 - phi))^2

        :param curvature: _description_
        :type curvature: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :return: _description_
        :rtype: Array
        """

        curvature_weights = self.compute_curvature_weights(volume_fraction)
        for _ in range(self.steps_curvature):
            # fig, ax = plt.subplots()
            # ax.imshow(jnp.squeeze(curvature).T, origin="lower")
            # plt.show()
            correction = self._correct_curvature_step(curvature, curvature_weights)
            # print(correction.shape)
            curvature = curvature.at[...,self.nhx_,self.nhy_,self.nhz_].set(
                correction[self.curvature_correction_center_slice])
            # fig, ax = plt.subplots(ncols=2)
            # ax[0].imshow(jnp.squeeze(curvature).T, origin="lower")
            # ax[1].imshow(jnp.squeeze(curvature[self.nhx_,self.nhy_,self.nhz_]).T, origin="lower")

            curvature = self.halo_manager.perform_halo_update_diffuse_curvature(curvature)
            # curvature = curvature.at[:2,:,:].set(0.0)
            # curvature = curvature.at[-2:,:,:].set(0.0)
            # curvature = curvature.at[:,:2,:].set(0.0)
            # curvature = curvature.at[:,-2:,:].set(0.0)
            
            # fig, ax = plt.subplots(ncols=2)
            # ax[0].imshow(jnp.squeeze(curvature).T, origin="lower")
            # ax[1].imshow(jnp.squeeze(curvature[self.nhx_,self.nhy_,self.nhz_]).T, origin="lower")
            # plt.show()
            # exit()
        # print(curvature.shape)
        return curvature
            
    def _correct_curvature_step(
            self,
            curvature: Array,
            curvature_weights: Array,
            ) -> Array:
        """_summary_

        :param curvature: _description_
        :type curvature: Array
        :param curvature_weights: _description_
        :type curvature_weights: Array
        :return: _description_
        :rtype: Array
        """
        weighted_curvature = 0
        sum_weights = 0
        for i in range(self.no_neighbors_curvature_correction):
            s_ = self.curvature_correction_neighbor_slices[i]
            weighted_curvature += curvature_weights[s_] * curvature[s_]
            sum_weights += curvature_weights[s_]
        # curvature = weighted_curvature / (sum_weights + self.eps)
        # curvature = weighted_curvature / (sum_weights)
        curvature = weighted_curvature / (sum_weights + 1e-100)
        return curvature
    
    def compute_curvature_weights(self, volume_fraction: Array) -> Array:
        """
        curvature_weight = alpha * (1 - alpha)

        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: Array
        :return: Curvature weight buffer
        :rtype: Array
        """
        curvature_weight = volume_fraction * (1.0 - volume_fraction)
        return curvature_weight * curvature_weight

    def limit_volume_fraction(self, volume_fraction: Array) -> Array:
        volume_fraction = jnp.minimum(
            jnp.maximum(volume_fraction, self.eps), 1.0-self.eps)
        return volume_fraction
        
    def compute_smoothed_volume_fraction(
            self,
            volume_fraction: Array, 
            vf_power: float,
            ) -> Array:
        """Computes a smoothed interface function based on the volume fraction
        field. See Shukla et al. 2010.

        psi = phi**vf_power / (phi**vf_power + (1 - phi)**vf_power)

        :param volume_fraction: Buffer of volume fraction
        :type volume_fraction: Array
        :param vf_power: Scalar parameter
        :type vf_power: float
        :return: Smoothed interface function
        :rtype: Array
        """
        vf_power_alpha = jnp.power(volume_fraction, vf_power)
        return vf_power_alpha / (vf_power_alpha + jnp.power(1.0 - volume_fraction, vf_power))
    
    def compute_signed_distance_function(
            self,
            phi: Array,
            ) -> Array:
        """Computes an (approximate) signed distance function
        from a given volume fraction field.
        
        \psi / \epsilon = log((\phi + 1e-100) / (1.0 - \phi + 1e-100))

        :param phi: _description_
        :type phi: Array
        :return: _description_
        :rtype: Array
        """
        # TODO eps
        # return jnp.log((phi + self.eps) / (1.0 - phi + self.eps))
        return jnp.log((phi + 1e-100) / (1.0 - phi + 1e-100))

    def compute_surface_tension_kernel(
            self,
            volume_fraction: Array
            ) -> Array:
        
        if self.surface_tension_kernel == "QUADRATIC":
            kernel = 6.0 * volume_fraction * (1.0 - volume_fraction)
        
        else:
            raise NotImplementedError

        return kernel