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

from functools import partial

import jax
import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class SourceTermSolver:
    """ The SourceTermSolver class manages source term contributions
    on the right-hand-side of NSE. Contributions have to be activated
    in the active_physics section in the numerical setup file.
    
    This includes amongst others:
    1) Gravitational force 
    2) Heat flux
    3) Viscous flux

    """
    
    def __init__(self, material_manager: MaterialManager, gravity: jnp.ndarray, domain_information: DomainInformation, derivative_stencil_center: SpatialDerivative,
            derivative_stencil_face: SpatialDerivative, reconstruct_stencil_duidxi: SpatialReconstruction, reconstruct_stencil_ui: SpatialReconstruction,
            levelset_type: str) -> None:
        
        self.gravity = gravity
        self.material_manager = material_manager

        self.derivative_stencil_center  = derivative_stencil_center
        self.derivative_stencil_face    = derivative_stencil_face
        self.reconstruct_stencil_duidxi = reconstruct_stencil_duidxi
        self.reconstruct_stencil_ui     = reconstruct_stencil_ui
        
        self.active_axis_indices    = [{"x": 0, "y": 1, "z": 2}[axis] for axis in domain_information.active_axis]

        nx, ny, nz          = domain_information.number_of_cells
        
        if levelset_type == "FLUID-FLUID":
            self.shape_fluxes   = [ (3, 2, nx+1, ny, nz), (3, 2, nx, ny+1, nz), (3, 2, nx, ny, nz+1) ]
        else:
            self.shape_fluxes   = [ (3, nx+1, ny, nz), (3, nx, ny+1, nz), (3, nx, ny, nz+1) ] 


        self.cell_sizes                 = domain_information.cell_sizes
        self.nhx, self.nhy, self.nhz    = domain_information.domain_slices_conservatives

    def compute_gravity_forces(self, cons: jnp.ndarray) -> jnp.ndarray:
        """Computes flux due to gravitational force.

        :param cons: Buffer of conservative variables.
        :type cons: jnp.ndarray
        :return: Buffer with gravitational forces.
        :rtype: jnp.ndarray
        """

        density  = cons[0:1, ..., self.nhx, self.nhy, self.nhz]
        momentum = cons[1:4, ..., self.nhx, self.nhy, self.nhz]

        gravity_momentum = jnp.einsum("ij..., jk...->ik...", self.gravity.reshape(3,1), density)
        gravity_energy   = jnp.einsum("ij..., jk...->ik...", self.gravity.reshape(1,3), momentum)

        gravity_forces = jnp.vstack([jnp.zeros(gravity_energy.shape), gravity_momentum, gravity_energy])

        return gravity_forces

    def compute_heat_flux_xi(self, temperature: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes the heat flux in axis direction.

        :param temperature: Buffer with temperature.
        :type temperature: jnp.ndarray
        :param axis: Spatial direction along which the heat flux is calculated.
        :type axis: int
        :return: Heat flux in axis direction.
        :rtype: jnp.ndarray
        """
        temperature_at_xj       = self.reconstruct_stencil_ui.reconstruct_xi(temperature, axis)
        thermal_conductivity    = self.material_manager.get_thermal_conductivity(temperature_at_xj)
        temperature_grad        = self.derivative_stencil_face.derivative_xi(temperature, self.cell_sizes[axis], axis)
        flux_xi                 = -thermal_conductivity*temperature_grad
        heat_flux_xi            = jnp.vstack([jnp.zeros(flux_xi.shape), jnp.zeros(flux_xi.shape), jnp.zeros(flux_xi.shape), jnp.zeros(flux_xi.shape), flux_xi])
        return heat_flux_xi
    
    def compute_viscous_flux_xi(self, vels: jnp.ndarray, temperature: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes viscous flux in one spatial direction

        vel_grad = [
            du/dx du/dy du/dz
            dv/dx dv/dy dv/dz
            dw/dx dw/dy dw/dz
        ]

        :param vels: Buffer of velocities.
        :type vels: jnp.ndarray
        :param temperature: Buffer of temperature.
        :type temperature: jnp.ndarray
        :param axis: Axis along which the viscous flux is computed.
        :type axis: int
        :return: Viscous flux along axis direction.
        :rtype: jnp.ndarray
        """
        temperature_at_cf   = self.reconstruct_stencil_ui.reconstruct_xi(temperature, axis)
        dynamic_viscosity   = self.material_manager.get_dynamic_viscosity(temperature_at_cf[0])
        bulk_viscosity      = self.material_manager.bulk_viscosity
        vel_grad            = jnp.stack([self.compute_xi_derivatives_at_xj(vels, i, axis) if i in self.active_axis_indices else jnp.zeros(self.shape_fluxes[axis]) for i in range(3)], axis=1)        
        tau_j               = self.compute_tau(vel_grad, dynamic_viscosity, bulk_viscosity, axis)

        vel_at_xj = self.reconstruct_stencil_ui.reconstruct_xi(vels, axis)

        vel_tau_at_xj = 0
        for k in self.active_axis_indices:
            vel_tau_at_xj += tau_j[k] * vel_at_xj[k]

        fluxes_xj = jnp.stack([jnp.zeros(tau_j[0].shape), tau_j[0], tau_j[1], tau_j[2], vel_tau_at_xj])

        return fluxes_xj

    def compute_tau(self, vel_grad: jnp.ndarray, dynamic_viscosity: jnp.ndarray, 
        bulk_viscosity: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes the stress tensor at a cell face in axis direction.
        tau_axis = [tau_axis0, tau_axis1, tau_axis2]

        :param vel_grad: Buffer of velocity gradient. Shape is 3 x 3 (x 2) x Nx x Ny x Nz 
        :type vel_grad: jnp.ndarray
        :param dynamic_viscosity: Buffer of dynamic viscosity.
        :type dynamic_viscosity: jnp.ndarray
        :param bulk_viscosity: Buffer of bulk viscosity.
        :type bulk_viscosity: jnp.ndarray
        :param axis: Cell face direction at which viscous stresses are calculated.
        :type axis: int
        :return: Buffer of viscous stresses.
        :rtype: jnp.ndarray
        """
        mu_1 = dynamic_viscosity
        mu_2 = bulk_viscosity - 2.0 / 3.0 * dynamic_viscosity
        tau_list = [
            mu_1 * (vel_grad[axis,0] + vel_grad[0,axis]) if axis in self.active_axis_indices and 0 in self.active_axis_indices else jnp.zeros(vel_grad.shape[2:]), 
            mu_1 * (vel_grad[axis,1] + vel_grad[1,axis]) if axis in self.active_axis_indices and 1 in self.active_axis_indices else jnp.zeros(vel_grad.shape[2:]),
            mu_1 * (vel_grad[axis,2] + vel_grad[2,axis]) if axis in self.active_axis_indices and 2 in self.active_axis_indices else jnp.zeros(vel_grad.shape[2:]) 
        ]
        tau_list[axis] += mu_2 * sum([vel_grad[k,k] for k in self.active_axis_indices])
        return jnp.stack(tau_list)

    def compute_xi_derivatives_at_xj(self, prime: jnp.ndarray, axis_i: int, axis_j: int) -> jnp.ndarray:
        """Computes the spatial derivative in axis_i direction at the cell face in axis_j direction.

        :param prime: Buffer of primitive variables.
        :type prime: jnp.ndarray
        :param axis_i: Spatiald direction wrt which the derivative is taken.
        :type axis_i: int
        :param axis_j: Spatial direction along which derivative is evaluated.
        :type axis_j: int
        :return: Derivative wrt axis_i direction at cell face in axis_j direction.
        :rtype: jnp.ndarray
        """
        
        if axis_i == axis_j:
            # Direct derivative evaluation at cell face
            deriv_xi_at_xj = self.derivative_stencil_face.derivative_xi(prime, self.cell_sizes[axis_i], axis_i)
        else:
            # 1) Derivative at cell center 2) Interpolate to cell face
            deriv_xi_at_c   = self.derivative_stencil_center.derivative_xi(prime, self.cell_sizes[axis_i], axis_i)
            deriv_xi_at_xj  = self.reconstruct_stencil_duidxi.reconstruct_xi(deriv_xi_at_c, axis_j)

        return deriv_xi_at_xj
