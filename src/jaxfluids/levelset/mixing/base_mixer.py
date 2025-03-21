from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.geometry.mask_functions import (
    compute_cut_cell_mask_sign_change_based, compute_cut_cell_mask_value_based)
from jaxfluids.levelset.mixing.helper_functions import move_source_to_target_ii, \
    move_source_to_target_ij, move_source_to_target_ijk, move_target_to_source_ii, \
    move_target_to_source_ij, move_target_to_source_ijk
from jaxfluids.data_types.buffers import LevelsetSolidCellIndices
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetMixingFieldSetup

Array = jax.Array

class Mixer(ABC):
    """Parent class for mixing methods.

    :param ABC: _description_
    :type ABC: _type_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            mixing_setup: LevelsetMixingFieldSetup
            ) -> None:

        # self.eps = precision.get_eps()
        self.eps = 1e-20

        self.domain_information = domain_information
        self.mixing_setup = mixing_setup

        active_axes_indices = domain_information.active_axes_indices
        index_pairs = [(0,1), (0,2), (1,2)]
        self.index_pairs_mixing = [] 
        for pair in index_pairs:
            if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
                self.index_pairs_mixing.append(pair)

        nhx,nhy,nhz = domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
        nh = domain_information.nh_conservatives
        nh_ = domain_information.nh_geometry

        active_axes_indices = domain_information.active_axes_indices
        self.slices_nh = (...,)
        for i in range(3):
            if i in active_axes_indices:
                self.slices_nh += (jnp.s_[nh-1:-nh+1],)
            else:
                self.slices_nh += (jnp.s_[:],)

        self.slices_nh_ = (...,)
        for i in range(3):
            if i in active_axes_indices:
                self.slices_nh_ += (jnp.s_[nh_-1:-nh_+1],)
            else:
                self.slices_nh_ += (jnp.s_[:],)

        self.s1 = (...,)
        for i in range(3):
            if i in active_axes_indices:
                self.s1 += (jnp.s_[1:-1],)
            else:
                self.s1 += (jnp.s_[:],)

    def compute_source_cells(
            self,
            volume_fraction_new: Array,
            volume_fraction_old: Array,
            levelset: Array
            ) -> Tuple[Array]:
        """Computes the mask for the cells that
        require mixing, i.e., vanished cells, newly
        created cells and cell below the volume
        fraction and/or density threshold.

        :param volume_fraction_new: _description_
        :type volume_fraction_new: Array
        :param volume_fraction_old: _description_
        :type volume_fraction_old: Array
        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Tuple[Array]
        """
        nh_offset = self.domain_information.nh_offset
        dx = self.domain_information.smallest_cell_size
        volume_fraction_threshold = self.mixing_setup.volume_fraction_threshold

        # NOTE sign changed based cut cells, necessary for
        # cases when interface is located directly on cell face
        mask_cut_cells = compute_cut_cell_mask_sign_change_based(levelset, nh_offset)
        # mask_cut_cells = compute_cut_cell_mask_value_based(levelset, dx)
        threshold_cells = volume_fraction_new < volume_fraction_threshold
        threshold_cells *= mask_cut_cells

        new_cells = (volume_fraction_new > 0.0) & (volume_fraction_old == 0.0)
        vanished_cells = (volume_fraction_new == 0.0) & (volume_fraction_old > 0.0)

        return threshold_cells, vanished_cells, new_cells    

    def compute_mixing_weights(
            self,
            volume_fraction: Array, 
            normal: Array,
            ) -> Tuple[Array]:
        """Computes the mixing weights.

        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param normal: _description_
        :type normal: Array
        :return: _description_
        :rtype: Tuple[Array]
        """
        
        dim = self.domain_information.dim
        active_axes_indices = self.domain_information.active_axes_indices
        nh_geometry = self.domain_information.nh_geometry

        mixing_targets = self.mixing_setup.mixing_targets

        normal = normal[self.slices_nh_]
        normal_sign = jnp.sign(normal)

        # NOTE we compute mixing flux at source position, therefore moving target
        # arrays to source. We need offset=1, since source mixing flux
        # is subsequently moved to target, which requires 1 halo cell

        # MIXING WEIGHTS II
        mixing_weight_sum = 0.0
        mixing_weight_ii_list = []
        for i in active_axes_indices:
            mixing_weight_ii = jnp.square(normal[i])
            volume_fraction_target = move_target_to_source_ii(volume_fraction, normal_sign, i,
                                                              nh_geometry, active_axes_indices,
                                                              offset=1)
            mask = jnp.where(volume_fraction[self.slices_nh_] > volume_fraction_target, 0, 1)
            mixing_weight_ii *= mask
            mixing_weight_ii *= volume_fraction_target
            mixing_weight_sum += mixing_weight_ii
            mixing_weight_ii_list.append( mixing_weight_ii )
        mixing_weight_ii = jnp.stack(mixing_weight_ii_list, axis=0)

        # MIXING WEIGHTS IJ
        if dim > 1 and mixing_targets > 1:
            mixing_weight_ij_list = []
            for (i, j) in self.index_pairs_mixing:
                mixing_weight_ij = jnp.abs(normal[i]*normal[j])
                volume_fraction_target = move_target_to_source_ij(volume_fraction, normal_sign, i, j,
                                                                  nh_geometry, active_axes_indices,
                                                                  offset=1)
                mask = jnp.where(volume_fraction[self.slices_nh_] > volume_fraction_target, 0, 1)
                mixing_weight_ij *= mask
                mixing_weight_ij *= volume_fraction_target
                mixing_weight_sum += mixing_weight_ij
                mixing_weight_ij_list.append( mixing_weight_ij )
            mixing_weight_ij = jnp.stack(mixing_weight_ij_list, axis=0)

        # MIXING WEIGHTS IJK
        if dim == 3 and mixing_targets == 3:
            mixing_weight_ijk = jnp.abs(normal[0]*normal[1]*normal[2])**(2/3)
            volume_fraction_target = move_target_to_source_ijk(volume_fraction, normal_sign,
                                                               nh_geometry, offset=1)
            mask = jnp.where(volume_fraction[self.slices_nh_] > volume_fraction_target, 0, 1)
            mixing_weight_ijk *= mask
            mixing_weight_ijk *= volume_fraction_target
            mixing_weight_sum += mixing_weight_ijk

        # NORMALIZATION
        mixing_weight_ii = mixing_weight_ii/(mixing_weight_sum + self.eps)
        if dim > 1 and mixing_targets > 1:
            mixing_weight_ij = mixing_weight_ij/(mixing_weight_sum + self.eps)
        else:
            mixing_weight_ij = None
        if dim == 3 and mixing_targets == 3:
            mixing_weight_ijk = mixing_weight_ijk/(mixing_weight_sum + self.eps)
        else:
            mixing_weight_ijk = None

        return mixing_weight_ii, mixing_weight_ij, mixing_weight_ijk

    def compute_mixing_fluxes(
            self,
            conserved_quantity: Array,
            volume_fraction: Array,
            normal: Array,
            source_cells: Array,
            mixing_weight_ii: Array,
            mixing_weight_ij: Array,
            mixing_weight_ijk: Array,
            ) -> Tuple[Array]:
        """Computes the mixing fluxes.

        :param conserved_quantity: _description_
        :type conserved_quantity: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param normal: _description_
        :type normal: Array
        :param source_cells: _description_
        :type source_cells: Array
        :param mixing_weight_ii: _description_
        :type mixing_weight_ii: Array
        :param mixing_weight_ij: _description_
        :type mixing_weight_ij: Array
        :param mixing_weight_ijk: _description_
        :type mixing_weight_ijk: Array
        :return: _description_
        :rtype: List[Array]
        """

        dim = self.domain_information.dim
        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        active_axes_indices = self.domain_information.active_axes_indices
        nh_geometry = self.domain_information.nh_geometry
        nh_conservatives = self.domain_information.nh_conservatives

        mixing_targets = self.mixing_setup.mixing_targets

        normal = normal[self.slices_nh_]
        normal_sign = jnp.sign(normal)

        Mixing_fluxes_source = 0.0
        Mixing_fluxes_target = 0.0

        # NOTE we first compute mixing flux at source cells,
        # then move this buffer to target

        # MIXING CONTRIBUTIONS II
        for i, axis in enumerate(active_axes_indices):

            volume_fraction_target = move_target_to_source_ii(
                volume_fraction, normal_sign, axis, nh_geometry,
                active_axes_indices, offset=1)
            conserved_quantity_target = move_target_to_source_ii(
                conserved_quantity, normal_sign, axis, nh_conservatives,
                active_axes_indices, offset=1)

            denominator = volume_fraction[self.slices_nh_] * mixing_weight_ii[i] + volume_fraction_target
            factor = mixing_weight_ii[i]/(denominator + self.eps)
            M_ii_source = conserved_quantity_target * volume_fraction[self.slices_nh_] - \
                 conserved_quantity[self.slices_nh] * volume_fraction_target
            M_ii_source = factor * M_ii_source * source_cells[self.slices_nh_]
            M_ii_target = move_source_to_target_ii(-M_ii_source, normal_sign, axis,
                                                   1, active_axes_indices, 0)
            M_ii_source = M_ii_source[self.s1]

            Mixing_fluxes_source += M_ii_source
            Mixing_fluxes_target += M_ii_target

        # MIXING CONTRIBUTIONS IJ
        if dim > 1 and mixing_targets > 1:
            for k, (axis_i, axis_j) in enumerate(self.index_pairs_mixing):
                volume_fraction_target = move_target_to_source_ij(
                    volume_fraction, normal_sign, axis_i, axis_j,
                    nh_geometry, active_axes_indices)
                conserved_quantity_target = move_target_to_source_ij(
                    conserved_quantity, normal_sign, axis_i, axis_j,
                    nh_conservatives, active_axes_indices)

                denominator = volume_fraction[self.slices_nh_] * mixing_weight_ij[k] + volume_fraction_target
                factor = mixing_weight_ij[k]/(denominator + self.eps)
                M_ij_source = conserved_quantity_target * volume_fraction[self.slices_nh_] - \
                    conserved_quantity[self.slices_nh] * volume_fraction_target
                M_ij_source = factor * M_ij_source * source_cells[self.slices_nh_]
                M_ij_target = move_source_to_target_ij(-M_ij_source, normal_sign, axis_i, axis_j,
                                                       1, active_axes_indices, 0) # TODO aaron memory bottleneck in 3D
                M_ij_source = M_ij_source[self.s1]
                Mixing_fluxes_source += M_ij_source
                Mixing_fluxes_target += M_ij_target

        # MIXING CONTRIBUTIONS IJK
        if dim == 3 and mixing_targets == 3:
            volume_fraction_target = move_target_to_source_ijk(
                volume_fraction, normal_sign, nh_geometry)
            conserved_quantity_target = move_target_to_source_ijk(
                conserved_quantity, normal_sign, nh_conservatives)

            denominator = volume_fraction[self.slices_nh_] * mixing_weight_ijk + volume_fraction_target
            factor = mixing_weight_ijk/(denominator + self.eps)
            M_ijk_source = conserved_quantity_target * volume_fraction[self.slices_nh_] - \
                conserved_quantity[self.slices_nh] * volume_fraction_target
            M_ijk_source = factor * M_ijk_source * source_cells[self.slices_nh_]
            M_ijk_target = move_source_to_target_ijk(-M_ijk_source, normal_sign, 1, 0) # TODO aaron memory bottleneck in 3D
            M_ijk_source = M_ijk_source[self.s1]

            Mixing_fluxes_source += M_ijk_source
            Mixing_fluxes_target += M_ijk_target

        return Mixing_fluxes_source, Mixing_fluxes_target


    def compute_mixing_weights_cell_based(
            self,
            volume_fraction: Array, 
            normal: Array,
            source_indices: Array,
            target_ii_indices: Tuple[Array],
            target_ij_indices: Tuple[Array],
            target_ijk_indices: Array,
            ) -> Tuple[Array]:

        dim = self.domain_information.dim
        active_axes_indices = self.domain_information.active_axes_indices
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        mixing_targets = self.mixing_setup.mixing_targets

        normal = normal[self.slices_nh_]
        volume_fraction = volume_fraction[self.slices_nh_]

        src_ = (...,) + source_indices
        volume_fraction_src = volume_fraction[src_]
        normal_src = normal[src_]
        
        # MIXING WEIGHTS II
        mixing_weight_sum = 0.0
        mixing_weight_ii_tuple = []
        for axis_index in range(3):
            if axis_index in active_axes_indices:
                trg_ = (...,) + target_ii_indices[axis_index]
                volume_fraction_trg = volume_fraction[trg_]
                mask = jnp.where(volume_fraction_src > volume_fraction_trg, 0, 1)
                mixing_weight_ii = jnp.square(normal_src[axis_index])
                mixing_weight_ii *= mask
                mixing_weight_ii *= volume_fraction_trg
                mixing_weight_sum += mixing_weight_ii
            else:
                mixing_weight_ii = None
            mixing_weight_ii_tuple.append(mixing_weight_ii)

        # MIXING WEIGHTS IJ
        if dim > 1 and mixing_targets > 1:
            mixing_weight_ij_tuple = []
            for k, (axis_index_i, axis_index_j) in enumerate(((0,1),(0,2),(1,2))):
                if axis_index_i in active_axes_indices and axis_index_j in active_axes_indices:
                    trg_ = (...,) + target_ij_indices[k]
                    volume_fraction_trg = volume_fraction[trg_]
                    mask = jnp.where(volume_fraction_src > volume_fraction_trg, 0, 1)
                    mixing_weight_ij = jnp.abs(normal_src[axis_index_i]*normal_src[axis_index_j])
                    mixing_weight_ij *= mask
                    mixing_weight_ij *= volume_fraction_trg
                    mixing_weight_sum += mixing_weight_ij

                else:
                    mixing_weight_ij = None
                mixing_weight_ij_tuple.append(mixing_weight_ij)
        else:
            mixing_weight_ij_tuple = (None,None,None)

        # MIXING WEIGHTS IJK
        if dim == 3 and mixing_targets == 3:
            trg_ = (...,) + target_ijk_indices
            volume_fraction_trg = volume_fraction[trg_]
            mask = jnp.where(volume_fraction_src > volume_fraction_trg, 0, 1)
            mixing_weight_ijk = jnp.abs(normal_src[0]*normal_src[1]*normal_src[2])**(2/3)
            mixing_weight_ijk *= mask
            mixing_weight_ijk *= volume_fraction_trg
            mixing_weight_sum += mixing_weight_ijk
        else:
            mixing_weight_ijk = None

        # NORMALIZATION
        mixing_weight_ii_tuple = tuple([
            mixing_weight_ii/(mixing_weight_sum + self.eps)
            if mixing_weight_ii is not None else None
            for mixing_weight_ii in mixing_weight_ii_tuple
        ])

        if dim > 1 and mixing_targets > 1:
            mixing_weight_ij_tuple = tuple([
                mixing_weight_ij/(mixing_weight_sum + self.eps)
                if mixing_weight_ij is not None else None
                for mixing_weight_ij in mixing_weight_ij_tuple
            ])

        if dim == 3 and mixing_targets == 3:
            mixing_weight_ijk = mixing_weight_ijk/(mixing_weight_sum + self.eps)

        mixing_weights = mixing_weight_ii_tuple + mixing_weight_ij_tuple + (mixing_weight_ijk,)

        return mixing_weights


    def compute_mixing_fluxes_cell_based(
            self,
            conserved_quantity: Array,
            volume_fraction: Array,
            mixing_weights: Tuple[Array],
            source_indices: Array,
            target_indices: Tuple[Array],
            ) -> Tuple[Array, Tuple[Array]]:
        """Computes the mixing fluxes.

        :param conserved_quantity: _description_
        :type conserved_quantity: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param normal: _description_
        :type normal: Array
        :param source_cells: _description_
        :type source_cells: Array
        :param mixing_weight_ii: _description_
        :type mixing_weight_ii: Array
        :param mixing_weight_ij: _description_
        :type mixing_weight_ij: Array
        :param mixing_weight_ijk: _description_
        :type mixing_weight_ijk: Array
        :return: _description_
        :rtype: List[Array]
        """

        dim = self.domain_information.dim
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives

        volume_fraction = volume_fraction[self.slices_nh_]
        conserved_quantity = conserved_quantity[self.slices_nh]

        src_ = (...,) + source_indices
        volume_fraction_src = volume_fraction[src_]
        conserved_quantity_src = conserved_quantity[src_]

        Mixing_fluxes_source = 0.0

        # NOTE we first compute mixing flux at source cells,
        # then move this buffer to target

        Mixing_fluxes_target_tuple = []

        for mixing_weight_i, trg_ in zip(mixing_weights, target_indices):

            if mixing_weight_i is not None:

                trg_ = (...,) + trg_
                volume_fraction_trg = volume_fraction[trg_]
                conserved_quantity_trg = conserved_quantity[trg_]

                denominator = volume_fraction_src * mixing_weight_i + volume_fraction_trg
                factor = mixing_weight_i/(denominator + self.eps)

                M_ii_source = conserved_quantity_trg * volume_fraction_src - \
                    conserved_quantity_src * volume_fraction_trg
                M_ii_source = factor * M_ii_source

                Mixing_fluxes_source += M_ii_source
                Mixing_fluxes_target_tuple.append(-M_ii_source)

            else:
                Mixing_fluxes_target_tuple.append(None)

        return Mixing_fluxes_source, Mixing_fluxes_target_tuple



    @abstractmethod
    def perform_mixing(
            self,
            conserved_quantity: Array,
            *args
            ) -> Tuple[Array, Array]:
        """Performs a mixing procedure on the
        integrated conserved_quantity buffer treating
        vanished and newly created cells.
        Returns the volume-averaged conserved_quantity 
        which can be used to compute the integrated
        primitives. Implementation in child
        class.

        :param conserved_quantity: _description_
        :type conserved_quantity: Array
        :return: _description_
        :rtype: Tuple
        """



