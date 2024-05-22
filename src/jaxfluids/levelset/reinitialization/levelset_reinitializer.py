from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.halos.outer.levelset import BoundaryConditionLevelset
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.geometry_calculator import compute_cut_cell_mask
from jaxfluids.levelset.helper_functions import linear_filtering
from jaxfluids.config import precision
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

class LevelsetReinitializer(ABC):
    """Abstract class for levelset reinitialization.
    """
    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_condition_levelset: BoundaryConditionLevelset,
            reinitialization_setup: LevelsetReinitializationSetup,
            narrowband_setup: NarrowBandSetup,
            ) -> None:
        
        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.boundary_condition_levelset = boundary_condition_levelset
        self.narrowband_setup = narrowband_setup
        self.reinitialization_setup = reinitialization_setup

    @abstractmethod
    def perform_reinitialization(
            self,
            levelset: Array,
            **kwargs
            ) -> Array:
        """Reinitializes the levelset buffer.
        This is an abstract method. See child class 
        for implementation and key word arguments.
        

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Array
        """
        pass

    def compute_reinitialization_mask(
            self,
            levelset: Array
            ) -> Tuple[Array, Array]:
        """Computes a mask indicating where to reintialize.
        User can specify domain, halos, cut cells and 
        and inactive reinitialization bandwidth.

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        # DOMAIN INFORMATION
        slice_cons_to_geometry = self.domain_information.domain_slices_conservatives_to_geometry
        slice_geometry = self.domain_information.domain_slices_geometry
        slice_conservatives = self.domain_information.domain_slices_conservatives
        nh_conservatives = self.domain_information.nh_conservatives
        smallest_cell_size = self.domain_information.smallest_cell_size

        is_domain = self.reinitialization_setup.is_domain
        is_halos = self.reinitialization_setup.is_halos
        is_cut_cell = self.reinitialization_setup.is_cut_cell
        inactive_bandwidth = self.narrowband_setup.inactive_reinitialization_bandwidth

        if not is_cut_cell:
            mask_cut_cells = compute_cut_cell_mask(levelset, nh_conservatives)
            inverse_mask_cut_cells = 1 - mask_cut_cells

        if inactive_bandwidth > 0:
            inverse_inactive_bandwidth_mask = jnp.where(jnp.abs(levelset[slice_conservatives]/smallest_cell_size) <= inactive_bandwidth, 0, 1)

        # REINITIALIZATION MASK
        if is_halos:
            levelset_mask = levelset[slice_cons_to_geometry]
            mask_halos = self.boundary_condition_levelset.get_halo_mask(levelset_mask)
            mask_domain = jnp.zeros_like(levelset_mask, dtype=jnp.uint32)
            mask_domain = mask_domain.at[slice_geometry].set(1) if is_domain else mask_domain
            mask_reinitialize = jnp.maximum(mask_domain, mask_halos)
            if not is_cut_cell:
                mask_reinitialize = mask_reinitialize.at[slice_geometry].mul(inverse_mask_cut_cells)
            if inactive_bandwidth > 0:
                mask_reinitialize = mask_reinitialize.at[slice_geometry].mul(inverse_inactive_bandwidth_mask)

        else:
            levelset_mask = levelset[slice_conservatives]
            mask_reinitialize = jnp.ones_like(levelset_mask, dtype=jnp.uint32) if is_domain else jnp.zeros_like(levelset_mask, dtype=jnp.uint32)
            if not is_cut_cell:
                mask_reinitialize *= inverse_mask_cut_cells
            if inactive_bandwidth > 0:
                mask_reinitialize *= inverse_inactive_bandwidth_mask

        return mask_reinitialize
    

    def remove_underresolved_structures(
            self,
            levelset: Array,
            volume_fraction: Array
            ) -> Array:
        """Interpolates the levelset linearly
        with neighboring cells for underresolved
        structures. Underresolved structures
        are present in sign change based cut
        cells, whose neighbors are solely
        full cells, i.e., a volume fraction
        of 1 or 0, respectively.

        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :return: _description_
        :rtype: Array
        """
    
        dim = self.domain_information.dim
        nh = self.domain_information.nh_conservatives
        nh_ = self.domain_information.nh_geometry
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nh_offset = self.domain_information.nh_offset
        active_axes_indices = self.domain_information.active_axes_indices

        if active_axes_indices == (0,):
            slice_list = [
                jnp.s_[...,nhx_        ,nhy_,nhz_],
                jnp.s_[...,nh_-1:-nh_-1,nhy_,nhz_],
                jnp.s_[...,nh_+1:-nh_+1,nhy_,nhz_]
            ]
        if active_axes_indices == (1,):
            slice_list = [
                jnp.s_[...,nhx_,nhy_        ,nhz_],
                jnp.s_[...,nhx_,nh_-1:-nh_-1,nhz_],
                jnp.s_[...,nhx_,nh_+1:-nh_+1,nhz_]
            ]
        if active_axes_indices == (2,):
            slice_list = [
                jnp.s_[...,nhx_     ,nhy_   ,nhz_],
                jnp.s_[...,nhx_,nhy_,nh_-1:-nh_-1],
                jnp.s_[...,nhx_,nhy_,nh_+1:-nh_+1]
                ]
            
        if active_axes_indices == (0,1):
            slice_list = [
                jnp.s_[...,nhx_        ,nhy_        ,nhz_],
                jnp.s_[...,nh_-1:-nh_-1,nh_-1:-nh_-1,nhz_],
                jnp.s_[...,nh_-1:-nh_-1,nh_  :-nh_  ,nhz_],
                jnp.s_[...,nh_-1:-nh_-1,nh_+1:-nh_+1,nhz_],
                jnp.s_[...,nh_  :-nh_  ,nh_+1:-nh_+1,nhz_],
                jnp.s_[...,nh_+1:-nh_+1,nh_+1:-nh_+1,nhz_],
                jnp.s_[...,nh_+1:-nh_+1,nh_  :-nh_  ,nhz_],
                jnp.s_[...,nh_+1:-nh_+1,nh_-1:-nh_-1,nhz_],
                jnp.s_[...,nh_  :-nh_,  nh_-1:-nh_-1,nhz_],
                ]
        if active_axes_indices == (0,2):
            slice_list = [
                jnp.s_[...,nhx_        ,nhy_        ,nhz_],
                jnp.s_[...,nh_-1:-nh_-1,nhy_,nh_-1:-nh_-1],
                jnp.s_[...,nh_-1:-nh_-1,nhy_,nh_  :-nh_  ],
                jnp.s_[...,nh_-1:-nh_-1,nhy_,nh_+1:-nh_+1],
                jnp.s_[...,nh_  :-nh_  ,nhy_,nh_+1:-nh_+1],
                jnp.s_[...,nh_+1:-nh_+1,nhy_,nh_+1:-nh_+1],
                jnp.s_[...,nh_+1:-nh_+1,nhy_,nh_  :-nh_  ],
                jnp.s_[...,nh_+1:-nh_+1,nhy_,nh_-1:-nh_-1],
                jnp.s_[...,nh_  :-nh_,  nhy_,nh_-1:-nh_-1],
                ]
        if active_axes_indices == (1,2):
            slice_list = [
                jnp.s_[...,nhx_,       nhy_ ,        nhz_],
                jnp.s_[...,nhx_,nh_-1:-nh_-1,nh_-1:-nh_-1],
                jnp.s_[...,nhx_,nh_-1:-nh_-1,nh_  :-nh_  ],
                jnp.s_[...,nhx_,nh_-1:-nh_-1,nh_+1:-nh_+1],
                jnp.s_[...,nhx_,nh_  :-nh_  ,nh_+1:-nh_+1],
                jnp.s_[...,nhx_,nh_+1:-nh_+1,nh_+1:-nh_+1],
                jnp.s_[...,nhx_,nh_+1:-nh_+1,nh_  :-nh_  ],
                jnp.s_[...,nhx_,nh_+1:-nh_+1,nh_-1:-nh_-1],
                jnp.s_[...,nhx_,nh_  :-nh_,  nh_-1:-nh_-1],
                ]
        if active_axes_indices == (0,1,2):
            slice_list = [
                jnp.s_[...,nh_  :-nh_  ,nh_  :-nh_  ,nh_  :-nh_  ],
                jnp.s_[...,nh_-1:-nh_-1,nh_-1:-nh_-1,nh_  :-nh_  ],
                jnp.s_[...,nh_-1:-nh_-1,nh_  :-nh_  ,nh_  :-nh_  ],
                jnp.s_[...,nh_-1:-nh_-1,nh_+1:-nh_+1,nh_  :-nh_  ],
                jnp.s_[...,nh_  :-nh_  ,nh_+1:-nh_+1,nh_  :-nh_  ],
                jnp.s_[...,nh_+1:-nh_+1,nh_+1:-nh_+1,nh_  :-nh_  ],
                jnp.s_[...,nh_+1:-nh_+1,nh_  :-nh_  ,nh_  :-nh_  ],
                jnp.s_[...,nh_+1:-nh_+1,nh_-1:-nh_-1,nh_  :-nh_  ],
                jnp.s_[...,nh_  :-nh_,  nh_-1:-nh_-1,nh_  :-nh_  ],

                jnp.s_[...,nh_  :-nh_  ,nh_  :-nh_  ,nh_-1:-nh_-1],
                jnp.s_[...,nh_-1:-nh_-1,nh_-1:-nh_-1,nh_-1:-nh_-1],
                jnp.s_[...,nh_-1:-nh_-1,nh_  :-nh_  ,nh_-1:-nh_-1],
                jnp.s_[...,nh_-1:-nh_-1,nh_+1:-nh_+1,nh_-1:-nh_-1],
                jnp.s_[...,nh_  :-nh_  ,nh_+1:-nh_+1,nh_-1:-nh_-1],
                jnp.s_[...,nh_+1:-nh_+1,nh_+1:-nh_+1,nh_-1:-nh_-1],
                jnp.s_[...,nh_+1:-nh_+1,nh_  :-nh_  ,nh_-1:-nh_-1],
                jnp.s_[...,nh_+1:-nh_+1,nh_-1:-nh_-1,nh_-1:-nh_-1],
                jnp.s_[...,nh_  :-nh_,  nh_-1:-nh_-1,nh_-1:-nh_-1],

                jnp.s_[...,nh_  :-nh_  ,nh_  :-nh_  ,nh_+1:-nh_+1],
                jnp.s_[...,nh_-1:-nh_-1,nh_-1:-nh_-1,nh_+1:-nh_+1],
                jnp.s_[...,nh_-1:-nh_-1,nh_  :-nh_  ,nh_+1:-nh_+1],
                jnp.s_[...,nh_-1:-nh_-1,nh_+1:-nh_+1,nh_+1:-nh_+1],
                jnp.s_[...,nh_  :-nh_  ,nh_+1:-nh_+1,nh_+1:-nh_+1],
                jnp.s_[...,nh_+1:-nh_+1,nh_+1:-nh_+1,nh_+1:-nh_+1],
                jnp.s_[...,nh_+1:-nh_+1,nh_  :-nh_  ,nh_+1:-nh_+1],
                jnp.s_[...,nh_+1:-nh_+1,nh_-1:-nh_-1,nh_+1:-nh_+1],
                jnp.s_[...,nh_  :-nh_,  nh_-1:-nh_-1,nh_+1:-nh_+1],
                ]

        include_center_value = False
        if not include_center_value:
            slice_list = slice_list[1:]

        mask = compute_cut_cell_mask(levelset, nh_offset)
        mask = mask[nhx_,nhy_,nhz_]
        mask = jnp.stack([mask, mask])
        for s_ in slice_list:
            condition_positive = (volume_fraction[s_] == 1.0)
            condition_negative = (volume_fraction[s_] == 0.0)
            conditions = jnp.stack([condition_positive, condition_negative])
            mask *= conditions
        mask = mask.any(axis=0)
        filtered_levelset = linear_filtering(levelset, nh, include_center_value)
        levelset = levelset.at[nhx,nhy,nhz].mul(1 - mask)
        levelset = levelset.at[nhx,nhy,nhz].add(filtered_levelset*mask)
        return levelset



    def set_levelset_cutoff(
            self,
            levelset: Array
            ) -> Array:
        """Sets cut off values for levelset values that
        lie outside of the narrow band.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Levelset buffer with cutoff values
        :rtype: Array
        """

        narrowband_cutoff = self.narrowband_setup.cutoff_width

        # DOMAIN INFORMATION
        s_ = self.domain_information.domain_slices_conservatives
        smallest_cell_size = self.domain_information.smallest_cell_size

        # CUT OFF MASKS
        mask_cut_off_positive = jnp.where(levelset[s_]/smallest_cell_size > narrowband_cutoff, 1, 0)
        mask_cut_off_negative = jnp.where(levelset[s_]/smallest_cell_size < -narrowband_cutoff, 1, 0)

        # SET CUTOFF VALUES
        levelset = levelset.at[s_].mul(1 - mask_cut_off_positive)
        levelset = levelset.at[s_].mul(1 - mask_cut_off_negative)
        levelset = levelset.at[s_].add(mask_cut_off_positive * narrowband_cutoff * smallest_cell_size)
        levelset = levelset.at[s_].add(-mask_cut_off_negative * narrowband_cutoff * smallest_cell_size)

        return levelset
