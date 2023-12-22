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

class QuantityExtender:
    """The QuantiyExtender performs a zero-gradient extension in interface normal direction of an arbitrary quantity.
    """

    def __init__(self, domain_information: DomainInformation, boundary_condition: BoundaryCondition,
        time_integrator: TimeIntegrator, spatial_stencil: SpatialDerivative, is_interface: bool) -> None:

        self.nhx, self.nhy, self.nhz    = domain_information.domain_slices_conservatives
        self.cell_sizes                 = domain_information.cell_sizes
        self.active_axis_indices        = domain_information.active_axis_indices
        self.smallest_cell_size         = jnp.min(jnp.array([self.cell_sizes[i] for i in self.active_axis_indices]))
 
        self.time_integrator    = time_integrator
        self.spatial_stencil    = spatial_stencil
        self.boundary_condition = boundary_condition
        self.is_interface       = is_interface

    def extend(self, quantity: jnp.ndarray, normal: jnp.ndarray,
            mask: jnp.ndarray, CFL: float, steps: int) -> Tuple[jnp.ndarray, float]:
        """Extends the quantity in normal direction. 

        :param quantity: Quantity buffer
        :type quantity: jnp.ndarray
        :param normal: Normal buffer
        :type normal: jnp.ndarray
        :param mask: Mask indicating where to extend
        :type mask: jnp.ndarray
        :param CFL: CFL number
        :type CFL: float
        :param steps: Number of integration steps
        :type steps: int
        :return: Extended quantity buffer and corresponding residual
        :rtype: Tuple[jnp.ndarray, float]
        """
        timestep_size = self.smallest_cell_size * CFL
        for i in range(steps):
            quantity, rhs   = self.do_integration_step(quantity, normal, mask, timestep_size)
            max_residual    = jnp.max(jnp.abs(rhs))
        return quantity, max_residual

    def do_integration_step(self, quantity: jnp.ndarray, normal: jnp.ndarray,
            mask: jnp.ndarray, timestep_size: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs an integration step of the extension equation.

        :param quantity: Quantity buffer
        :type quantity: jnp.ndarray
        :param normal: Normal buffer
        :type normal: jnp.ndarray
        :param mask: Mask indicating where to extend
        :type mask: jnp.ndarray
        :param timestep_size: Fictitious time step size
        :type timestep_size: float
        :return: Integrated quantity buffer and corresponding right-hand-side buffer
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        # FILL INIT
        if self.time_integrator.no_stages > 1:
            init = jnp.array(quantity, copy=True)
        for stage in range( self.time_integrator.no_stages ):
            # RHS
            rhs = self.compute_rhs(quantity, normal, mask)
            # PREPARE BUFFER FOR INTEGRATION
            if stage > 0:
                quantity = self.time_integrator.prepare_buffer_for_integration(quantity, init, stage)
            # INTEGRATE
            quantity = self.time_integrator.integrate(quantity, rhs, timestep_size, stage)
            # FILL BOUNDARIES
            if self.is_interface:
                quantity = self.boundary_condition.fill_boundary_levelset(quantity)
            else:
                _, quantity = self.boundary_condition.fill_boundary_primes(quantity, quantity, 0.0)

        return quantity, rhs

    def compute_rhs(self, quantity: jnp.ndarray, normal: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Computes the right-hand-side of the exension equation.

        :param quantity: Quantity buffer
        :type quantity: jnp.ndarray
        :param normal: Normal buffer
        :type normal: jnp.ndarray
        :param mask: Mask indiciating where to extend
        :type mask: jnp.ndarray
        :return: Right-hand-side of the extension equation
        :rtype: jnp.ndarray
        """
        rhs = 0.0
        for axis in self.active_axis_indices:
            cell_state_L = self.spatial_stencil.derivative_xi(quantity, 1.0, axis, 0)
            cell_state_R = self.spatial_stencil.derivative_xi(quantity, 1.0, axis, 1)
            # UPWINDING
            mask_L = jnp.where(normal[axis] >= 0.0, 1.0, 0.0)
            mask_R = 1.0 - mask_L
            rhs -= normal[axis] * (mask_L * cell_state_L + mask_R * cell_state_R) / self.cell_sizes[axis]
        rhs *= mask
        return rhs
