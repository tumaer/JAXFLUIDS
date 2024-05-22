
from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

import jax
from jax import Array
import jax.numpy as jnp

from jaxfluids.levelset.geometry_calculator import compute_fluid_masks, compute_cut_cell_mask
from jaxfluids.levelset.helper_functions import linear_filtering
from jaxfluids.levelset.mixing.helper_functions import move_source_to_target_ii, \
    move_source_to_target_ij, move_source_to_target_ijk, move_target_to_source_ii, \
    move_target_to_source_ij, move_target_to_source_ijk
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.levelset.mixing.conservative_mixer import ConservativeMixer
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup import LevelsetSetup


class LauerMixer(ConservativeMixer):
    """_summary_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            equation_information: EquationInformation,
            levelset_setup: LevelsetSetup,
            halo_manager: HaloManager,
            material_manager: MaterialManager,
            equation_manager: EquationManager
            ) -> None:

        super(LauerMixer, self).__init__(
            domain_information, equation_information,
            levelset_setup, halo_manager, material_manager,
            equation_manager)

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
            volume_fraction_old: Array
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
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        is_parallel = self.domain_information.is_parallel

        levelset_model = self.levelset_setup.model
        mixing_setup = self.levelset_setup.mixing

        # GEOMETRICAL QUANTITIES
        mask_real = compute_fluid_masks(volume_fraction_new, levelset_model)
        mask_real = mask_real[...,nhx_,nhy_,nhz_]
        if levelset_model == "FLUID-FLUID":
            normal = jnp.stack([normal, -normal], axis=1)
            volume_fraction_new = jnp.stack([volume_fraction_new, 1.0 - volume_fraction_new], axis=0)
            volume_fraction_old = jnp.stack([volume_fraction_old, 1.0 - volume_fraction_old], axis=0)

        # COMPUTE CELLS THAT REQUIRE MIXING
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

        # ADD MIXING FLUXES
        Mixing_fluxes_source = Mixing_fluxes_source[...,nhx_,nhy_,nhz_]
        Mixing_fluxes_target = Mixing_fluxes_target[...,nhx_,nhy_,nhz_]
        Mixing_fluxes = Mixing_fluxes_source + Mixing_fluxes_target
        conservatives = conservatives.at[...,nhx,nhy,nhz].add(Mixing_fluxes)

        # TRANSFORM TO VOLUME AVERAGES
        mask = jnp.where(volume_fraction_new[...,nhx_,nhy_,nhz_] == 0.0, 1, 0)
        denominator = volume_fraction_new[...,nhx_,nhy_,nhz_] + mask * self.eps
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1.0/denominator)

        # TREAT INVALID CELLS
        conservatives, invalid_cells = self.tag_invalid_cells(
            conservatives, mask_real)

        # INVALID CELL COUNT
        invalid_cell_count = jnp.sum(invalid_cells, axis=(-3,-2,-1))
        if is_parallel:
            invalid_cell_count = jax.lax.psum(invalid_cell_count, axis_name="i")

        return conservatives, invalid_cells, invalid_cell_count

    def compute_source_cells(
            self,
            volume_fraction_new: Array,
            volume_fraction_old: Array,
            levelset: Array,
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
        mixing_setup = self.levelset_setup.mixing
        volume_fraction_threshold = mixing_setup.volume_fraction_threshold

        threshold_cells = volume_fraction_new < volume_fraction_threshold

        mask_cut_cells = compute_cut_cell_mask(levelset, nh_offset)
        threshold_cells *= mask_cut_cells

        new_cells =(volume_fraction_new > 0.0) & (volume_fraction_old == 0.0)
        vanished_cells = (volume_fraction_new == 0.0) & (volume_fraction_old > 0.0)

        return threshold_cells, vanished_cells, new_cells    

    def compute_mixing_fluxes(
            self,
            conservatives: Array,
            volume_fraction: Array,
            normal: Array,
            source_cells: Array,
            mixing_weight_ii: Array,
            mixing_weight_ij: Array,
            mixing_weight_ijk: Array,
            ) -> List[Array]:
        """Computes the mixing fluxes.

        :param conservatives: _description_
        :type conservatives: Array
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

        normal_sign = jnp.sign(normal)

        dim = self.domain_information.dim
        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        active_axes_indices = self.domain_information.active_axes_indices

        mixing_setup = self.levelset_setup.mixing
        mixing_targets = mixing_setup.mixing_targets

        Mixing_fluxes_source_list = []
        Mixing_fluxes_target_list = []

        # MIXING CONTRIBUTIONS II
        for i, axis in enumerate(active_axes_indices):

            volume_fraction_target = move_target_to_source_ii(
                volume_fraction, normal_sign, axis)
            conservatives_target = move_target_to_source_ii(
                conservatives[...,nhx__,nhy__,nhz__], normal_sign, axis)

            denominator = volume_fraction * mixing_weight_ii[i] + volume_fraction_target
            factor = mixing_weight_ii[i]/(denominator + self.eps)
            M_ii_source = conservatives_target * volume_fraction - \
                 conservatives[...,nhx__,nhy__,nhz__] * volume_fraction_target
            M_ii_source = factor * M_ii_source * source_cells
            M_ii_target = move_source_to_target_ii(-M_ii_source, normal_sign, axis)

            Mixing_fluxes_source_list.append(M_ii_source)
            Mixing_fluxes_target_list.append(M_ii_target)

        # MIXING CONTRIBUTIONS IJ
        if dim > 1 and mixing_targets > 1:
            for k, (axis_i, axis_j) in enumerate(self.index_pairs_mixing):
                volume_fraction_target = move_target_to_source_ij(
                    volume_fraction, normal_sign, axis_i, axis_j)
                conservatives_target = move_target_to_source_ij(
                    conservatives[...,nhx__,nhy__,nhz__], normal_sign, axis_i, axis_j)

                denominator = volume_fraction * mixing_weight_ij[k] + volume_fraction_target
                factor = mixing_weight_ij[k]/(denominator + self.eps)
                M_ij_source = conservatives_target * volume_fraction - \
                    conservatives[...,nhx__,nhy__,nhz__] * volume_fraction_target
                M_ij_source = factor * M_ij_source * source_cells
                M_ij_target = move_source_to_target_ij(-M_ij_source, normal_sign, axis_i, axis_j)

                Mixing_fluxes_source_list.append(M_ij_source)
                Mixing_fluxes_target_list.append(M_ij_target)

        # MIXING CONTRIBUTIONS IJK
        if dim == 3 and mixing_targets == 3:
            volume_fraction_target = move_target_to_source_ijk(
                volume_fraction, normal_sign)
            conservatives_target = move_target_to_source_ijk(
                conservatives[...,nhx__,nhy__,nhz__], normal_sign)

            denominator = volume_fraction * mixing_weight_ijk + volume_fraction_target
            factor = mixing_weight_ijk/(denominator + self.eps)
            M_ijk_source = conservatives_target * volume_fraction - \
                conservatives[...,nhx__,nhy__,nhz__] * volume_fraction_target
            M_ijk_source = factor * M_ijk_source * source_cells
            M_ijk_target = move_source_to_target_ijk(-M_ijk_source, normal_sign)

            Mixing_fluxes_source_list.append(M_ijk_source)
            Mixing_fluxes_target_list.append(M_ijk_target)

        Mixing_fluxes_source = sum(Mixing_fluxes_source_list)
        Mixing_fluxes_target = sum(Mixing_fluxes_target_list)

        return Mixing_fluxes_source, Mixing_fluxes_target

    def compute_mixing_weights(
            self,
            volume_fraction: Array, 
            normal: Array
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

        mixing_setup = self.levelset_setup.mixing
        mixing_targets = mixing_setup.mixing_targets

        normal_sign = jnp.sign(normal)

        # MIXING WEIGHTS II
        mixing_weight_sum = 0.0
        mixing_weight_ii_list = []
        for i in active_axes_indices:
            mixing_weight_ii = jnp.square(normal[i])
            volume_fraction_target = move_target_to_source_ii(volume_fraction, normal_sign, i)
            mask = jnp.where(volume_fraction > volume_fraction_target, 0, 1)
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
                volume_fraction_target = move_target_to_source_ij(volume_fraction, normal_sign, i, j)
                mask = jnp.where(volume_fraction > volume_fraction_target, 0, 1)
                mixing_weight_ij *= mask
                mixing_weight_ij *= volume_fraction_target
                mixing_weight_sum += mixing_weight_ij
                mixing_weight_ij_list.append( mixing_weight_ij )
            mixing_weight_ij = jnp.stack(mixing_weight_ij_list, axis=0)

        # MIXING WEIGHTS IJK
        if dim == 3 and mixing_targets == 3:
            mixing_weight_ijk = jnp.abs(normal[0]*normal[1]*normal[2])**(2/3)
            volume_fraction_target = move_target_to_source_ijk(volume_fraction, normal_sign)
            mask = jnp.where(volume_fraction > volume_fraction_target, 0, 1)
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

        return mixing_weight_ii, mixing_weight_ij, \
            mixing_weight_ijk

    def tag_invalid_cells(
            self,
            conservatives: Array,
            mask_real: Array,
            ) -> Tuple[Array]:
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
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        ids = self.equation_information.mass_and_energy_ids
        energy_ids = self.equation_information.energy_ids

        invalid_cells_cons = (conservatives[ids,...,nhx,nhy,nhz] <= 0.0).any(axis=0)
        invalid_cells_cons *= mask_real

        mask_ghost = 1 - mask_real
        mask = jnp.maximum(mask_ghost, invalid_cells_cons)
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask)
        conservatives = conservatives.at[ids,...,nhx,nhy,nhz].add(mask * self.eps)
        primitives = self.equation_manager.get_primitives_from_conservatives(
            conservatives[...,nhx,nhy,nhz])
        pressure = primitives[energy_ids]
        p_b = self.material_manager.get_background_pressure()

        invalid_cells_primes = (pressure + p_b <= 0.0)
        invalid_cells_primes *= mask_real
        invalid_cells = invalid_cells_cons | invalid_cells_primes
        
        return conservatives, invalid_cells
