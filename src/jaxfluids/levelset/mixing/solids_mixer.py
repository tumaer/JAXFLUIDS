
from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxfluids.levelset.helper_functions import transform_to_volume_average
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.mixing.base_mixer import Mixer
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.data_types.buffers import LevelsetSolidCellIndices
from jaxfluids.domain.helper_functions import add_halo_offset

if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetMixingFieldSetup

Array = jax.Array

class SolidsMixer(Mixer):
    """_summary_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            mixing_setup: LevelsetMixingFieldSetup
            ) -> None:

        super().__init__(domain_information,
                         mixing_setup)
    
        active_axes_indices = domain_information.active_axes_indices
        index_pairs = [(0,1), (0,2), (1,2)]
        self.index_pairs_mixing = [] 
        for pair in index_pairs:
            if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
                self.index_pairs_mixing.append(pair)

    def perform_mixing(
            self,
            solid_energy: Array,
            levelset: Array,
            normal: Array,
            volume_fraction_new: Array,
            volume_fraction_old: Array,
            solid_cell_indices: LevelsetSolidCellIndices = None
            ) -> Tuple[Array, Array, Array,
                       LevelsetPositivityInformation]:
        """ Implements the mixing procedure
        as described in Lauer et. al (2012).

        :param solid_energy: _description_
        :type solid_energy: Array
        :param levelset: _description_
        :type levelset: Array
        :param normal: _description_
        :type normal: Array
        :param volume_fraction_new: _description_
        :type volume_fraction_new: Array
        :param volume_fraction_old: _description_
        :type volume_fraction_old: Array
        :return: _description_
        :rtype: Tuple[Array, Array, LevelsetPositivityInformation]
        """

        # DOMAIN INFORMATION
        nh = self.domain_information.nh_conservatives
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        is_parallel = self.domain_information.is_parallel
        active_axes_indices = self.domain_information.active_axes_indices

        is_cell_based_computation = self.mixing_setup.is_cell_based_computation
        is_cell_based_computation = is_cell_based_computation and solid_cell_indices is not None

        if not is_cell_based_computation:
                    
            threshold_cells, vanished_cells, \
            new_cells = self.compute_source_cells(
                volume_fraction_new, volume_fraction_old,
                levelset)
            source_cells = threshold_cells | vanished_cells | new_cells

            mixing_weight_ii, mixing_weight_ij, \
            mixing_weight_ijk = self.compute_mixing_weights(
                volume_fraction_new, normal)

            Mixing_fluxes_source, \
            Mixing_fluxes_target = self.compute_mixing_fluxes(
                solid_energy, volume_fraction_new, normal,
                source_cells, mixing_weight_ii, mixing_weight_ij,
                mixing_weight_ijk)

            Mixing_fluxes = Mixing_fluxes_source + Mixing_fluxes_target
            solid_energy = solid_energy.at[...,nhx,nhy,nhz].add(Mixing_fluxes)

        else:
            
            source_indices = solid_cell_indices.mixing_source_solid.indices
            target_ii_indices = (
                solid_cell_indices.mixing_target_ii_0_solid.indices,
                solid_cell_indices.mixing_target_ii_1_solid.indices,
                solid_cell_indices.mixing_target_ii_2_solid.indices
                )
            target_ij_indices = (
                solid_cell_indices.mixing_target_ij_01_solid.indices,
                solid_cell_indices.mixing_target_ij_02_solid.indices,
                solid_cell_indices.mixing_target_ij_12_solid.indices,
                )
            
            target_ijk_indices = solid_cell_indices.mixing_target_ijk_solid.indices

            mixing_weights = self.compute_mixing_weights_cell_based(
                volume_fraction_new, normal, source_indices,
                target_ii_indices, target_ij_indices, target_ijk_indices)

            target_indices = target_ii_indices + target_ij_indices + (target_ijk_indices,)

            Mixing_fluxes_source, Mixing_fluxes_target = self.compute_mixing_fluxes_cell_based(
                solid_energy, volume_fraction_new, mixing_weights,
                source_indices, target_indices)

            source_mask = solid_cell_indices.mixing_source_solid.mask
            target_mask = (
                solid_cell_indices.mixing_target_ii_0_solid.mask,
                solid_cell_indices.mixing_target_ii_1_solid.mask,
                solid_cell_indices.mixing_target_ii_2_solid.mask,
                solid_cell_indices.mixing_target_ij_01_solid.mask,
                solid_cell_indices.mixing_target_ij_02_solid.mask,
                solid_cell_indices.mixing_target_ij_12_solid.mask,
                solid_cell_indices.mixing_target_ijk_solid.mask
                )

            src_ = add_halo_offset(source_indices, nh-1, active_axes_indices) # NOTE nh-1 because these indices have offset of 1
            if is_parallel:
                solid_energy = solid_energy.at[src_].add(Mixing_fluxes_source*source_mask)
                for flux_i, trg_, mask_ in zip(Mixing_fluxes_target, target_indices, target_mask):
                    if flux_i is not None:
                        trg_ = add_halo_offset(trg_, nh-1, active_axes_indices)
                        solid_energy = solid_energy.at[trg_].add(flux_i*mask_)
            else:
                solid_energy = solid_energy.at[src_].add(Mixing_fluxes_source)
                for flux_i, trg_ in zip(Mixing_fluxes_target, target_indices):
                    if flux_i is not None:
                        trg_ = add_halo_offset(trg_, nh-1, active_axes_indices)
                        solid_energy = solid_energy.at[trg_].add(flux_i)

        solid_energy = transform_to_volume_average(
            solid_energy, volume_fraction_new,
            self.domain_information)

        mask_real = volume_fraction_new > 0.0
        mask_real = mask_real[...,nhx_,nhy_,nhz_]
        solid_energy, invalid_cells, invalid_cell_count = self.tag_invalid_cells(
            solid_energy, mask_real)

        return solid_energy, invalid_cells, invalid_cell_count



    def tag_invalid_cells(
            self,
            solid_energy: Array,
            mask_real: Array,
            ) -> Tuple[Array]:
        """Tags invalid cells, i.e., real cells
        with zero or negative density and/or
        pressure after mixing. Sets eps in those
        cells.

        :param solid_energy: _description_
        :type solid_energy: Array
        :param mask_real: _description_
        :type mask_real: Array
        :return: _description_
        :rtype: Tuple[Array]
        """
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        is_parallel = self.domain_information.is_parallel

        invalid_cells = solid_energy[...,nhx,nhy,nhz] <= 0.0
        invalid_cells *= mask_real

        mask_ghost = 1 - mask_real
        mask = jnp.maximum(mask_ghost, invalid_cells)

        solid_energy = solid_energy.at[...,nhx,nhy,nhz].mul(1 - mask)
        solid_energy = solid_energy.at[...,nhx,nhy,nhz].add(mask * self.eps)
        invalid_cell_count = jnp.sum(invalid_cells, axis=(-3,-2,-1))
        if is_parallel:
            invalid_cell_count = jax.lax.psum(invalid_cell_count, axis_name="i")

        return solid_energy, invalid_cells, invalid_cell_count

