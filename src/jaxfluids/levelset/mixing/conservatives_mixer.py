
from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks
from jaxfluids.levelset.helper_functions import transform_to_volume_average
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.mixing.base_mixer import Mixer
from jaxfluids.data_types.buffers import LevelsetSolidCellIndices, LevelsetSolidCellIndicesField
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager
from jaxfluids.data_types.numerical_setup import LevelsetSetup
from jaxfluids.math.filter.linear_averaging import linear_averaging
from jaxfluids.levelset.geometry.mask_functions import compute_cut_cell_mask_sign_change_based
from jaxfluids.domain.helper_functions import add_halo_offset

Array = jax.Array

class ConservativeMixer(Mixer):
    """_summary_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            levelset_setup: LevelsetSetup,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            halo_manager: HaloManager
            ) -> None:


        super().__init__(domain_information,
                         levelset_setup.mixing.conservatives)
        
        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager
        self.levelset_model = levelset_setup.model

        active_axes_indices = domain_information.active_axes_indices
        index_pairs = [(0,1), (0,2), (1,2)]
        self.index_pairs_mixing = [] 
        for pair in index_pairs:
            if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
                self.index_pairs_mixing.append(pair)

    def perform_mixing(
            self,
            conservatives: Array,
            levelset: Array,
            normal: Array,
            volume_fraction_new: Array,
            volume_fraction_old: Array,
            solid_cell_indices: LevelsetSolidCellIndices = None
            ) -> Tuple[Array, Array, Array,
                       LevelsetPositivityInformation]:
        """ Implements the mixing procedure
        as described in Lauer et. al (2012).

        :param conservatives: _description_
        :type conservatives: Array
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

        mask_real = compute_fluid_masks(volume_fraction_new, self.levelset_model)
        mask_real = mask_real[...,nhx_,nhy_,nhz_]

        if self.levelset_model == "FLUID-FLUID":
            normal = jnp.stack([normal, -normal], axis=1)
            volume_fraction_new = jnp.stack([volume_fraction_new, 1.0 - volume_fraction_new], axis=0)
            volume_fraction_old = jnp.stack([volume_fraction_old, 1.0 - volume_fraction_old], axis=0)

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
                conservatives, volume_fraction_new, normal,
                source_cells, mixing_weight_ii, mixing_weight_ij,
                mixing_weight_ijk)

            Mixing_fluxes = Mixing_fluxes_source + Mixing_fluxes_target
            conservatives = conservatives.at[...,nhx,nhy,nhz].add(Mixing_fluxes)

            mask = (Mixing_fluxes_source != 0.0) | (Mixing_fluxes_target != 0.0)

        else:

            source_indices = solid_cell_indices.mixing_source_fluid.indices

            target_ii_indices = (
                solid_cell_indices.mixing_target_ii_0_fluid.indices,
                solid_cell_indices.mixing_target_ii_1_fluid.indices,
                solid_cell_indices.mixing_target_ii_2_fluid.indices
                )
            target_ij_indices = (
                solid_cell_indices.mixing_target_ij_01_fluid.indices,
                solid_cell_indices.mixing_target_ij_02_fluid.indices,
                solid_cell_indices.mixing_target_ij_12_fluid.indices,
                )
            
            target_ijk_indices = solid_cell_indices.mixing_target_ijk_fluid.indices

            mixing_weights = self.compute_mixing_weights_cell_based(
                volume_fraction_new, normal, source_indices,
                target_ii_indices, target_ij_indices, target_ijk_indices)

            target_indices = target_ii_indices + target_ij_indices + (target_ijk_indices,)

            Mixing_fluxes_source, Mixing_fluxes_target = self.compute_mixing_fluxes_cell_based(
                conservatives, volume_fraction_new, mixing_weights,
                source_indices, target_indices)

            source_mask = solid_cell_indices.mixing_source_fluid.mask
            target_mask = (
                solid_cell_indices.mixing_target_ii_0_fluid.mask,
                solid_cell_indices.mixing_target_ii_1_fluid.mask,
                solid_cell_indices.mixing_target_ii_2_fluid.mask,
                solid_cell_indices.mixing_target_ij_01_fluid.mask,
                solid_cell_indices.mixing_target_ij_02_fluid.mask,
                solid_cell_indices.mixing_target_ij_12_fluid.mask,
                solid_cell_indices.mixing_target_ijk_fluid.mask
                )

            src_ = add_halo_offset(source_indices, nh-1, active_axes_indices) # NOTE nh-1 because these indices have offset of 1
            if is_parallel:
                conservatives = conservatives.at[src_].add(Mixing_fluxes_source*source_mask)
                for flux_i, trg_, mask_ in zip(Mixing_fluxes_target, target_indices, target_mask):
                    if flux_i is not None:
                        trg_ = add_halo_offset(trg_, nh-1, active_axes_indices)
                        conservatives = conservatives.at[trg_].add(flux_i*mask_)
            else:
                conservatives = conservatives.at[src_].add(Mixing_fluxes_source)
                for flux_i, trg_ in zip(Mixing_fluxes_target, target_indices):
                    if flux_i is not None:
                        trg_ = add_halo_offset(trg_, nh-1, active_axes_indices)
                        conservatives = conservatives.at[trg_].add(flux_i)
            
        # NOTE transform to volume averages
        conservatives = transform_to_volume_average(
            conservatives, volume_fraction_new,
            self.domain_information)

        # NOTE invalid cells
        conservatives, invalid_cells, invalid_cell_count = self.tag_invalid_cells(
            conservatives, mask_real, volume_fraction_new, levelset)
        
        return conservatives, invalid_cells, invalid_cell_count

    def tag_invalid_cells(
            self,
            conservatives: Array,
            mask_real: Array,
            volume_fraction_stacked: Array,
            levelset: Array
            ) -> Tuple[Array, Array, int]:
        """Tags invalid cells, i.e., real cells
        with zero or negative density and/or
        pressure after mixing. Sets eps in those
        cells.

        :param conservatives: _description_
        :type conservatives: Array
        :param mask_real: _description_
        :type mask_real: Array
        :return: _description_
        :rtype: Tuple[Array]
        """

        is_parallel = self.domain_information.is_parallel
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nh = self.domain_information.nh_conservatives

        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nh_ = self.domain_information.nh_geometry

        ids_mass_and_energy = self.equation_information.ids_mass_and_energy
        ids_mass = self.equation_information.ids_mass
        ids_energy = self.equation_information.ids_energy

        is_interpolate_invalid_cells = self.mixing_setup.is_interpolate_invalid_cells

        cut_cells = compute_cut_cell_mask_sign_change_based(levelset, nh)

        # NOTE tag invalid cells (density, energy) in real cut cells
        mask_invalid = mask_real & cut_cells
        invalid_cells_cons = (conservatives[ids_mass_and_energy,...,nhx,nhy,nhz] <= 0.0).any(axis=0)
        invalid_cells_cons *= mask_invalid

        # NOTE set epsilon in ghost cells and invalid cells
        mask_ghost = 1 - mask_real
        mask_cutoff = mask_ghost | invalid_cells_cons
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_cutoff)
        conservatives = conservatives.at[ids_mass_and_energy,...,nhx,nhy,nhz].add(mask_cutoff * self.eps)
        primitives = self.equation_manager.get_primitives_from_conservatives(
            conservatives[...,nhx,nhy,nhz])
        pressure = primitives[ids_energy]
        p_b = self.material_manager.get_background_pressure()

        # NOTE tag invalid cells (pressure) in real cut cells
        invalid_cells_primes = (pressure + p_b <= 0.0)
        invalid_cells_primes *= mask_invalid

        # NOTE invalid cells (density, energy, pressure)
        invalid_cells = invalid_cells_cons | invalid_cells_primes
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - invalid_cells)
        conservatives = conservatives.at[ids_mass_and_energy,...,nhx,nhy,nhz].add(invalid_cells * self.eps)
        
        if is_interpolate_invalid_cells:
            conservatives = self.halo_manager.perform_halo_update_conservatives_mixing(conservatives)
            filtered_conservatives = linear_averaging(conservatives, nh, False, nh_, volume_fraction_stacked)
            conservatives = conservatives.at[...,nhx,nhy,nhz].add(invalid_cells * filtered_conservatives)
            primitives = self.equation_manager.get_primitives_from_conservatives(
                conservatives[...,nhx,nhy,nhz])

        invalid_cell_count = jnp.sum(invalid_cells, axis=(-3,-2,-1))
        if is_parallel:
            invalid_cell_count = jax.lax.psum(invalid_cell_count, axis_name="i")

        return conservatives, invalid_cells, invalid_cell_count
