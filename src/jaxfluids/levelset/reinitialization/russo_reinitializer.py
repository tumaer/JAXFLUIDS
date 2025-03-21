from __future__ import annotations
from typing import Tuple, TYPE_CHECKING, Dict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.time_integration.euler import Euler
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.levelset.geometry.mask_functions import compute_cut_cell_mask_sign_change_based, compute_narrowband_mask
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.reinitialization.helper_functions import compute_godunov_hamiltonian
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

Array = jax.Array

class RussoReinitializer(PDEBasedReinitializer):
    """First order reinitializer with subcell fix
    according to \cite Russo2000
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            reinitialization_setup: LevelsetReinitializationSetup,
            narrowband_setup: NarrowBandSetup,
            ) -> None:

        super(RussoReinitializer, self).__init__(
            domain_information, halo_manager,
            reinitialization_setup, narrowband_setup)

        self.time_integrator = Euler(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes)

        self.derivative_stencil = FirstDerivativeFirstOrderCenter(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes)

        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives 
        nh = self.domain_information.nh_conservatives 

        self.s_1 = [[  
                jnp.s_[nh-1*j:-nh-1*j,nhy,nhz],
                jnp.s_[nhx,nh-1*j:-nh-1*j,nhz],
                jnp.s_[nhx,nhy,nh-1*j:-nh-1*j]
            ] for j in [1,-1]]


    def get_kwargs(
            self,
            levelset
            ) -> Dict[str, Array]:
        nh = self.domain_information.nh_conservatives
        mask_cut_cells = compute_cut_cell_mask_sign_change_based(levelset, nh)
        distance = self.compute_distance_approximation(levelset)
        kwargs = {"distance": distance, "mask_cut_cells": mask_cut_cells}
        return kwargs

    def compute_rhs(
            self,
            levelset: Array,
            levelset_0: Array,
            mask: Array,
            distance: Array,
            mask_cut_cells: Array
            ) -> Tuple[Array, Array]:
        """Computes the right-hand-side of the
        levelset reinitialization equation.

        :param levelset: _description_
        :type levelset: Array
        :param levelset_0: _description_
        :type levelset_0: Array
        :param mask: _description_
        :type mask: Array
        :param distance: _description_, defaults to None
        :type distance: Array, optional
        :param mask_cut_cells: _description_, defaults to None
        :type mask_cut_cells: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices
        smallest_cell_size = self.domain_information.smallest_cell_size
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives

        # DERIVATIVES
        derivatives_L = []
        derivatives_R = []
        for axis in active_axes_indices:
            derivatives_L.append( self.derivative_stencil.derivative_xi(levelset, cell_size, axis, 0) )
            derivatives_R.append( self.derivative_stencil.derivative_xi(levelset, cell_size, axis, 1) )

        sign = jnp.sign(levelset_0[nhx,nhy,nhz])
        godunov_hamiltonian = compute_godunov_hamiltonian(derivatives_L, derivatives_R, sign)
        rhs_godunov = - sign * (godunov_hamiltonian - 1)

        rhs_subcell = - 1.0/smallest_cell_size * (sign * jnp.abs(levelset[nhx,nhy,nhz]) - distance)
    
        rhs = rhs_subcell * mask_cut_cells + (1 - mask_cut_cells) * rhs_godunov
        rhs *= mask

        return rhs


    def compute_distance_approximation(
            self,
            levelset: Array,
            ) -> Array:
        """Distance approximation from
        the levelset field.

        :param levelset: _description_
        :type levelset: Array
        :param nh_offset: _description_
        :type nh_offset: int
        :param smallest_cell_size: _description_
        :type smallest_cell_size: float
        :return: _description_
        :rtype: Array
        """

        smallest_cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives

        denominator_1 = 0.0
        denominator_2 = 0.0
        denominator_3 = 0.0
        for axis in active_axes_indices:
            s_L = self.s_1[0][axis]
            s_R = self.s_1[1][axis]
            denominator_1 += jnp.square(levelset[s_R] - levelset[s_L])
            denominator_2 += jnp.square(levelset[s_R] - levelset[nhx,nhy,nhz])
            denominator_3 += jnp.square(levelset[nhx,nhy,nhz] - levelset[s_L])
        denominator_1 = jnp.sqrt(denominator_1 + self.eps)/2.0
        denominator_2 = jnp.sqrt(denominator_2 + self.eps)
        denominator_3 = jnp.sqrt(denominator_3 + self.eps)
        denominator = jnp.maximum(denominator_1, denominator_2)
        denominator = jnp.maximum(denominator, denominator_3)
        denominator = jnp.maximum(denominator, self.eps)

        distance = smallest_cell_size * levelset[nhx,nhy,nhz]/denominator

        return distance

