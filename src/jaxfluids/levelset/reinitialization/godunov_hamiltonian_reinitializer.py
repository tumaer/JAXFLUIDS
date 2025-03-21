from __future__ import annotations
from typing import Tuple, TYPE_CHECKING, Dict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.geometry.mask_functions import compute_narrowband_mask
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.levelset.reinitialization.helper_functions import compute_godunov_hamiltonian
from jaxfluids.levelset.geometry.mask_functions import compute_cut_cell_mask_sign_change_based, compute_narrowband_mask

if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.levelset import LevelsetReinitializationSetup, NarrowBandSetup

Array = jax.Array

class GodunovHamiltonianReinitializer(PDEBasedReinitializer):
    """Solves the reinitialization equation using the 
    monotone Godunov Hamiltonian \cite Bardi1991 according to 
    \cite Sussman1994. Temporal and spatial
    discretization is user specified.

    :param LevelsetReinitializer: _description_
    :type LevelsetReinitializer: _type_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            reinitialization_setup: LevelsetReinitializationSetup,
            narrowband_setup: NarrowBandSetup,
            ) -> None:

        super(GodunovHamiltonianReinitializer, self).__init__(
            domain_information, halo_manager,
            reinitialization_setup, narrowband_setup)

        time_integrator = reinitialization_setup.time_integrator
        derivative_stencil = reinitialization_setup.spatial_stencil
        self.time_integrator: TimeIntegrator = time_integrator(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes)
        self.derivative_stencil: SpatialDerivative = derivative_stencil(
            nh=self.domain_information.nh_conservatives,
            inactive_axes=self.domain_information.inactive_axes)
    
    def get_kwargs(self, levelset: Array):
        return {}

    def compute_rhs(
            self,
            levelset: Array,
            levelset_0: Array,
            mask: Array
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


        active_axes_indices = self.domain_information.active_axes_indices
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives

        # DERIVATIVES
        derivatives_L = []
        derivatives_R = []
        for axis in active_axes_indices:
            derivatives_L.append( self.derivative_stencil.derivative_xi(levelset, self.cell_size, axis, 0) )
            derivatives_R.append( self.derivative_stencil.derivative_xi(levelset, self.cell_size, axis, 1) )

        levelset_0 = levelset_0[nhx,nhy,nhz]

        sign = jnp.sign(levelset_0)
        smooth_sign = levelset_0/jnp.sqrt(levelset_0**2 + self.cell_size**2)
        godunov_hamiltonian = compute_godunov_hamiltonian(derivatives_L, derivatives_R, sign)

        rhs = -smooth_sign * (godunov_hamiltonian - 1.0)
        rhs *= mask

        return rhs