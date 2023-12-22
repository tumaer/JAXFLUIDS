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

from typing import Dict

import jax.numpy as jnp
import numpy as np

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils import DICT_FIRST_DERIVATIVE_CENTER


class InterfaceFluxComputer:
    """The InterfaceFluxComputer computes the two-phase interface fluxes depending on the present interface interaction type
    and active physics. The Interface interaction types are
    1) FLUID-SOLID-STATIC
    2) FLUID-SOLID-DYNAMIC
    3) FLUID-FLUID
    """
    def __init__(self, domain_information: DomainInformation, material_manager: MaterialManager, numerical_setup: Dict) -> None:
        
        self.material_manager       = material_manager

        self.nhx__, self.nhy__, self.nhz__  = domain_information.domain_slices_conservatives_to_geometry
        self.nhx_, self.nhy_, self.nhz_     = domain_information.domain_slices_geometry
        self.nhx, self.nhy, self.nhz        = domain_information.domain_slices_conservatives
        self.cell_centers                   = domain_information.cell_centers
        self.cell_sizes                     = domain_information.cell_sizes
        self.active_axis_indices            = domain_information.active_axis_indices

        self.derivative_stencil : SpatialDerivative = DICT_FIRST_DERIVATIVE_CENTER[numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_center"]](nh=domain_information.nh_geometry, inactive_axis=domain_information.inactive_axis) if numerical_setup["active_physics"]["is_viscous_flux"] else None

        self.is_convective_flux   = numerical_setup["active_physics"]["is_convective_flux"]
        self.is_viscous_flux    = numerical_setup["active_physics"]["is_viscous_flux"]
        self.levelset_type      = numerical_setup["levelset"]["interface_interaction"]

        self.aperture_slices = [ 
            [np.s_[...,1:,:,:], np.s_[...,:-1,:,:]],
            [np.s_[...,:,1:,:], np.s_[...,:,:-1,:]],
            [np.s_[...,:,:,1:], np.s_[...,:,:,:-1]],
        ]

    def compute_interface_flux_xi(self, primes: jnp.ndarray, interface_velocity: jnp.ndarray, interface_pressure: jnp.ndarray,
            volume_fraction: jnp.ndarray, apertures: jnp.ndarray, normal: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes the interface flux in axis direction.

        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param interface_velocity: Interface velocity buffer
        :type interface_velocity: jnp.ndarray
        :param interface_pressure: Interface pressure buffer
        :type interface_pressure: jnp.ndarray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :param apertures: Aperture buffers
        :type apertures: jnp.ndarray
        :param normal: Normal buffer
        :type normal: jnp.ndarray
        :param axis: axis direction
        :type axis: int
        :return: Interface flux in axis direction
        :rtype: jnp.ndarray
        """
        # BUFFER
        interface_flux_xi = jnp.zeros(primes[..., self.nhx, self.nhy, self.nhz].shape)

        if self.levelset_type in ["FLUID-SOLID-STATIC", "FLUID-SOLID-DYNAMIC"]:
            
            # CONVECTIVE FLUX
            if self.is_convective_flux:

                # MOMENTUM CONTRIBUTION
                convective_flux_xi_momentum   = primes[4, self.nhx, self.nhy, self.nhz] * (apertures[self.aperture_slices[axis][0]][...,self.nhx_,self.nhy_,self.nhz_] - apertures[self.aperture_slices[axis][1]][...,self.nhx_,self.nhy_,self.nhz_])
                interface_flux_xi             = interface_flux_xi.at[axis+1].add(convective_flux_xi_momentum)

                # ENERGY CONTRIBUTION
                if self.levelset_type == "FLUID-SOLID-DYNAMIC":
                    convective_flux_xi_energy = convective_flux_xi_momentum * interface_velocity[axis] 
                    interface_flux_xi         = interface_flux_xi.at[4].add(convective_flux_xi_energy)
                        
            # VISCOUS FLUX
            if self.is_viscous_flux:
                
                # COMPUTE VELOCITY GRADIENT
                velocity            = primes[1:4, self.nhx__, self.nhy__, self.nhz__]
                velocity_gradient   = jnp.stack([self.derivative_stencil.derivative_xi(velocity, self.cell_sizes[k], k) if k in self.active_axis_indices else jnp.zeros(velocity[:,self.nhx_,self.nhy_,self.nhz_].shape) for k in range(3)], axis=1)
                
                mu_1 = self.material_manager.get_dynamic_viscosity(self.material_manager.get_temperature(primes[4,self.nhx,self.nhy,self.nhz], primes[0,self.nhx,self.nhy,self.nhz]))
                mu_2 = self.material_manager.bulk_viscosity - 2.0 / 3.0 * mu_1

                # COMPUTE TAU
                tau_i = [
                    mu_1 * (velocity_gradient[axis,0] + velocity_gradient[0,axis]) if axis in self.active_axis_indices and 0 in self.active_axis_indices else jnp.zeros(velocity_gradient.shape[2:]), 
                    mu_1 * (velocity_gradient[axis,1] + velocity_gradient[1,axis]) if axis in self.active_axis_indices and 1 in self.active_axis_indices else jnp.zeros(velocity_gradient.shape[2:]),
                    mu_1 * (velocity_gradient[axis,2] + velocity_gradient[2,axis]) if axis in self.active_axis_indices and 2 in self.active_axis_indices else jnp.zeros(velocity_gradient.shape[2:]) 
                ]
                tau_i[axis] += mu_2 * sum([velocity_gradient[k,k] for k in self.active_axis_indices])
                tau_i = - jnp.stack(tau_i) * (apertures[self.aperture_slices[axis][0]][...,self.nhx_,self.nhy_,self.nhz_] - apertures[self.aperture_slices[axis][1]][...,self.nhx_,self.nhy_,self.nhz_])
                
                # MOMENTUM CONTRIBUTION
                interface_flux_xi = interface_flux_xi.at[1:4].add(tau_i)

                # ENERGY CONTRIBUTION
                if self.levelset_type == "FLUID-SOLID-DYNAMIC":
                    viscid_flux_xi_energy = 0
                    for k in self.active_axis_indices:
                        viscid_flux_xi_energy += tau_i[k] * interface_velocity[k]
                    interface_flux_xi = interface_flux_xi.at[4].add(viscid_flux_xi_energy)

        elif self.levelset_type == "FLUID-FLUID":

            # GEOMETRICAL QUANTITIES
            apertures       = jnp.stack([apertures, 1.0 - apertures], axis=0) 
            volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0) 

            # CONVECTIVE FLUX
            if self.is_convective_flux:
                convective_flux_xi_momentum = interface_pressure * (apertures[self.aperture_slices[axis][0]][...,self.nhx_,self.nhy_,self.nhz_] - apertures[self.aperture_slices[axis][1]][...,self.nhx_,self.nhy_,self.nhz_])
                convective_flux_xi_energy   = convective_flux_xi_momentum * interface_velocity * normal[axis][...,self.nhx_,self.nhy_,self.nhz_]
                interface_flux_xi           = interface_flux_xi.at[axis+1].add(convective_flux_xi_momentum)
                interface_flux_xi           = interface_flux_xi.at[4].add(convective_flux_xi_energy)

            # VISCID FLUX
            if self.is_viscous_flux:
                
                # COMPUTE VELOCITY GRADIENT
                velocity_0, velocity_1  = primes[1:4,0,self.nhx__,self.nhy__,self.nhz__], primes[1:4,1,self.nhx__,self.nhy__,self.nhz__]
                real_velocity           = velocity_0 * volume_fraction[0] + velocity_1 * volume_fraction[1]
                velocity_gradient       = jnp.stack([self.derivative_stencil.derivative_xi(real_velocity, self.cell_sizes[k], k) if k in self.active_axis_indices else jnp.zeros(real_velocity[:,self.nhx_,self.nhy_,self.nhz_].shape) for k in range(3)], axis=1)
                
                # COMPUTE INTERFACE VISCOSITY
                mu_1            = self.material_manager.get_dynamic_viscosity(self.material_manager.get_temperature(primes[4,...,self.nhx,self.nhy,self.nhz], primes[0,...,self.nhx,self.nhy,self.nhz]))
                mu_2            = self.material_manager.bulk_viscosity - 2.0 / 3.0 * mu_1
                mu_1_interface  = mu_1[0]*mu_1[1]/(volume_fraction[0,self.nhx_,self.nhy_,self.nhz_]*mu_1[1] + volume_fraction[1,self.nhx_,self.nhy_,self.nhz_]*mu_1[0])
                mu_2_interface  = mu_2[0]*mu_2[1]/(volume_fraction[0,self.nhx_,self.nhy_,self.nhz_]*mu_2[1] + volume_fraction[1,self.nhx_,self.nhy_,self.nhz_]*mu_2[0])

                # COMPUTE TAU
                tau_i = [
                    mu_1_interface * (velocity_gradient[axis,0] + velocity_gradient[0,axis]) if axis in self.active_axis_indices and 0 in self.active_axis_indices else jnp.zeros(velocity_gradient.shape[2:]), 
                    mu_1_interface * (velocity_gradient[axis,1] + velocity_gradient[1,axis]) if axis in self.active_axis_indices and 1 in self.active_axis_indices else jnp.zeros(velocity_gradient.shape[2:]),
                    mu_1_interface * (velocity_gradient[axis,2] + velocity_gradient[2,axis]) if axis in self.active_axis_indices and 2 in self.active_axis_indices else jnp.zeros(velocity_gradient.shape[2:]) 
                ]
                tau_i[axis] += mu_2_interface * sum([velocity_gradient[k,k] for k in self.active_axis_indices])
                tau_i       = - jnp.expand_dims(jnp.stack(tau_i), axis=1) * (apertures[self.aperture_slices[axis][0]][...,self.nhx_,self.nhy_,self.nhz_] - apertures[self.aperture_slices[axis][1]][...,self.nhx_,self.nhy_,self.nhz_])
                
                # MOMENTUM CONTRIBUTION
                interface_flux_xi = interface_flux_xi.at[1:4].add(tau_i)

                # ENERGY CONTRIBUTION
                viscid_flux_xi_energy = 0
                for k in self.active_axis_indices:
                    viscid_flux_xi_energy += tau_i[k] * interface_velocity * normal[k,self.nhx_,self.nhy_,self.nhz_]
                interface_flux_xi = interface_flux_xi.at[4].add(viscid_flux_xi_energy)

        # TODO HEAT CONDUCTION

        return interface_flux_xi
