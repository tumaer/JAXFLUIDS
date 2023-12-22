#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from typing import Tuple

import jax.numpy as jnp
import numpy as np
import time

from jaxfluids.boundary_condition import BoundaryCondition
from jaxfluids.domain_information import DomainInformation
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class LevelsetReinitializer:
    """The LevelsetReinitializer class implements functionality to reinitialize the levelset buffer.
    """
    
    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, domain_information: DomainInformation, boundary_condition: BoundaryCondition,
        time_integrator: TimeIntegrator, derivative_stencil: SpatialDerivative) -> None:

        # DOMAIN
        self.nh                         = domain_information.nh_conservatives
        self.dim                        = domain_information.dim
        self.nhx, self.nhy, self.nhz    = domain_information.domain_slices_conservatives
        self.cell_sizes                 = domain_information.cell_sizes
        self.active_axis_indices        = domain_information.active_axis_indices
        self.smallest_cell_size         = jnp.min(jnp.array([self.cell_sizes[i] for i in self.active_axis_indices]))

        # NUMERICAL SETUP
        self.time_integrator            = time_integrator
        self.derivative_stencil         = derivative_stencil
        self.boundary_condition         = boundary_condition

    def reinitialize(self, levelset: jnp.ndarray, mask_reinitialize: jnp.ndarray,
            mask_residual: jnp.ndarray, CFL: float, steps: int) -> Tuple[jnp.ndarray, float]:
        """Reinitializes the levelset buffer.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param mask_reinitialize: Mask indicating which cells to reinitialize
        :type mask_reinitialize: jnp.ndarray
        :param mask_residual: Mask indicating where to compute the maximum residual of the reinitialization equation 
        :type mask_residual: jnp.ndarray
        :param CFL: CFL number
        :type CFL: float
        :param steps: Number of integration steps
        :type steps: int
        :return: Reinitialized levelset buffer and maximum residual of the reinitialization equation 
        :rtype: Tuple[jnp.ndarray, float]
        """
        levelset_0      = jnp.array(levelset, copy=True)
        timestep_size   = CFL * self.smallest_cell_size
        max_residual    = -1.0
        for i in range(steps):
            levelset, residual  = self.do_integration_step(levelset, levelset_0, mask_reinitialize, timestep_size)
            max_residual        = jnp.max(jnp.abs(mask_residual*residual))
        return levelset, max_residual

    def do_integration_step(self, levelset: jnp.ndarray, levelset_0: jnp.ndarray,
            mask: jnp.ndarray, timestep_size: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs an integration step of the levelset reinitialization equation.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param levelset_0: Levelset buffer at fictitious time = 0.0
        :type levelset_0: jnp.ndarray
        :param mask: Mask for right-hand-side
        :type mask: jnp.ndarray
        :param timestep_size: Timestep size
        :type timestep_size: float
        :return: Tuple containing integrated levelset buffer and signed distance residual
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        # FILL INIT
        if self.time_integrator.no_stages > 1:
            init = jnp.array(levelset, copy=True)
        for stage in range( self.time_integrator.no_stages ):
            # RHS
            rhs, residual = self.compute_rhs(levelset, levelset_0, mask)
            # PREPARE BUFFER FOR INTEGRATION
            if stage > 0:
                levelset = self.time_integrator.prepare_buffer_for_integration(levelset, init, stage)
            # INTEGRATE
            levelset = self.time_integrator.integrate(levelset, rhs, timestep_size, stage)
            # FILL BOUNDARIES
            levelset = self.boundary_condition.fill_boundary_levelset(levelset)

        return levelset, residual

    def compute_rhs(self, levelset: jnp.ndarray, levelset_0: jnp.ndarray,
            mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        """Computes the right-hand-side of the levelset reinitialization equation.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param levelset_0: Levelset buffer at fictitious time = 0.0
        :type levelset_0: jnp.ndarray
        :param mask: Mask for right-hand-side
        :type mask: jnp.ndarray
        :return: Tuple of right-hand-side and signed distance residual
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        # TODO STENCILS IN __INIT__ FOR MASK CUT CELL AND DISTANCE
        if self.dim == 1:
            if self.active_axis_indices == [0]:
                distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.abs(
                                levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] - levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz] + self.eps)
                mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] < 0), 1, 0    )
            elif self.active_axis_indices == [1]:
                distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.abs(
                                levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] - levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz] + self.eps)
                mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] < 0), 1, 0    )
            elif self.active_axis_indices == [2]:
                distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.abs(
                                levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] - levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1] + self.eps)
                mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] < 0), 1, 0    )

        elif self.dim == 2:
            if self.active_axis_indices == [0,1]:
                distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.sqrt(
                                    (levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] - levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz])**2 + \
                                    (levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] - levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz])**2 + self.eps)
                mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] < 0), 1, 0    )
            elif self.active_axis_indices == [0,2]:
                distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.sqrt(
                                    (levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] - levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz])**2 + \
                                    (levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] - levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1])**2 + self.eps)
                mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] < 0), 1, 0    )
            elif self.active_axis_indices == [1,2]:
                distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.sqrt(
                                    (levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] - levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz])**2 + \
                                    (levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] - levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1])**2 + self.eps)
                mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1] < 0) |
                                                (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] < 0), 1, 0    )
        else:
            distance        = 2 * self.smallest_cell_size * levelset_0[self.nhx,self.nhy,self.nhz] / jnp.sqrt(
                                (levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] - levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz])**2 + \
                                (levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] - levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz])**2 + \
                                (levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] - levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1])**2 + self.eps)
            mask_cut_cells  = jnp.where(    (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh-1:-self.nh-1,self.nhy,self.nhz] < 0) |
                                            (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nh+1:-self.nh+1,self.nhy,self.nhz] < 0) |
                                            (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh-1:-self.nh-1,self.nhz] < 0) |
                                            (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nh+1:-self.nh+1,self.nhz] < 0) |
                                            (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh-1:-self.nh-1] < 0) |
                                            (levelset_0[self.nhx,self.nhy,self.nhz]*levelset_0[self.nhx,self.nhy,self.nh+1:-self.nh+1] < 0), 1, 0    )

        # DERIVATIVES
        derivatives_L    = []
        derivatives_R    = []
        for axis in self.active_axis_indices:
            derivatives_L.append( self.derivative_stencil.derivative_xi(levelset, self.cell_sizes[axis], axis, 0, levelset_0, distance) )
            derivatives_R.append( self.derivative_stencil.derivative_xi(levelset, self.cell_sizes[axis], axis, 1, levelset_0, distance) )

        # SMOOTH SIGN
        sign        = jnp.sign(levelset_0[self.nhx,self.nhy,self.nhz])
        smooth_sign = mask_cut_cells * distance / self.smallest_cell_size + (1 - mask_cut_cells) * sign                                  
        
        # GODUNOV HAMILTONIAN
        godunov_hamiltonian = 0.0
        for der_L, der_R in zip(derivatives_L, derivatives_R):
            godunov_hamiltonian += jnp.maximum( jnp.maximum(0.0, sign * der_L)**2, jnp.minimum(0.0, sign * der_R)**2 )
        godunov_hamiltonian = jnp.sqrt(godunov_hamiltonian + self.eps)

        # RHS
        signed_distance_residual = (godunov_hamiltonian - 1)
        rhs = -smooth_sign*signed_distance_residual
        rhs *= mask

        return rhs, signed_distance_residual