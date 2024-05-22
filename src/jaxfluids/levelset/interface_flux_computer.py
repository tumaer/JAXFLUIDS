from typing import Dict, Union

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.levelset.geometry_calculator import compute_cut_cell_mask
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered
from jaxfluids.config import precision

class InterfaceFluxComputer:
    """The InterfaceFluxComputer computes the two-phase interface
    fluxes depending on the present interface interaction type
    and active physics. The Interface interaction types are
    1) FLUID-SOLID-STATIC
    2) FLUID-SOLID-DYNAMIC
    3) FLUID-FLUID
    """


    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            numerical_setup: NumericalSetup,
            solid_temperature_model: str
            ) -> None:

        self.eps = precision.get_eps()

        self.material_manager = material_manager
        self.equation_information = material_manager.equation_information
        self.domain_information = domain_information
        self.numerical_setup = numerical_setup

        self.solid_temperature_model = solid_temperature_model

        is_viscous_flux = self.numerical_setup.active_physics.is_viscous_flux
        is_heat_flux = self.numerical_setup.active_physics.is_heat_flux

        if is_viscous_flux or is_heat_flux:
            derivative_stencil = numerical_setup.levelset.interface_flux.derivative_stencil
            self.derivative_stencil : SpatialDerivative = derivative_stencil(
                        nh = domain_information.nh_geometry,
                        inactive_axes = domain_information.inactive_axes)

        self.aperture_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

    def compute_interface_flux_xi(
            self,
            primitives: Array,
            interface_velocity: Array,
            interface_pressure: Array,
            volume_fraction: Array,
            apertures: Array,
            normal: Array,
            axis_index: int,
            temperature: Array
            ) -> Array:
        """Computes the interface flux in axis_index direction.

        :param primitives: _description_
        :type primitives: Array
        :param interface_velocity: _description_
        :type interface_velocity: Array
        :param interface_pressure: _description_
        :type interface_pressure: Array
        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param apertures: _description_
        :type apertures: Array
        :param normal: _description_
        :type normal: Array
        :param axis_index: _description_
        :type axis_index: int
        :param temperature: _description_
        :type temperature: Union[Array, None]
        :param solid_temperature: _description_
        :type solid_temperature: Union[Array, None]
        :return: _description_
        :rtype: Array
        """

        levelset_model = self.numerical_setup.levelset.model
        if levelset_model == "FLUID-FLUID":
            interface_flux_xi = self.fluid_fluid_flux_xi(
                primitives, interface_velocity,
                interface_pressure, volume_fraction,
                apertures, normal, axis_index, temperature)
        else:
            interface_flux_xi = self.fluid_solid_flux_xi(
                primitives, interface_velocity, apertures,
                axis_index, temperature)

        return interface_flux_xi

    def fluid_solid_flux_xi(
            self,
            primitives: Array,
            interface_velocity: Array,
            apertures: Array,
            axis_index: int,
            temperature: Array
            ) -> Array:

        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices

        is_convective_flux = self.numerical_setup.active_physics.is_convective_flux
        is_viscous_flux = self.numerical_setup.active_physics.is_viscous_flux
        is_heat_flux = self.numerical_setup.active_physics.is_heat_flux
        levelset_model = self.numerical_setup.levelset.model
        viscous_flux_method = self.numerical_setup.levelset.interface_flux.viscous_flux_method

        is_moving_levelset = self.equation_information.is_moving_levelset
        diffuse_interface_model = self.equation_information.diffuse_interface_model

        energy_slices = self.equation_information.energy_slices
        velocity_slices = self.equation_information.velocity_slices
        momentum_xi_slices = self.equation_information.momentum_xi_slices

        if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            interface_velocity = interface_velocity[...,nhx,nhy,nhz]

        interface_flux_xi = jnp.zeros(primitives[...,nhx,nhy,nhz].shape)

        # INTERFACE SEGMENT LENGTH - AXIS PROJECTION
        delta_aperture = apertures[self.aperture_slices[axis_index][0]][...,nhx_,nhy_,nhz_] - \
                            apertures[self.aperture_slices[axis_index][1]][...,nhx_,nhy_,nhz_]
        
        # CONVECTIVE FLUX
        if is_convective_flux:
            
            convective_flux_xi_momentum = primitives[energy_slices,nhx,nhy,nhz] * delta_aperture
            interface_flux_xi = interface_flux_xi.at[momentum_xi_slices[axis_index]].add(convective_flux_xi_momentum)
            if is_moving_levelset:
                convective_flux_xi_energy = convective_flux_xi_momentum * interface_velocity[axis_index] 
                interface_flux_xi = interface_flux_xi.at[energy_slices].add(convective_flux_xi_energy)

        # VISCOUS FLUX
        if is_viscous_flux and viscous_flux_method == "JAXFLUIDS":
            
            velocity = primitives[velocity_slices,nhx__,nhy__,nhz__]
            velocity_gradient = self.compute_velocity_gradient(velocity)
            
            mu_1 = self.material_manager.get_dynamic_viscosity(
                temperature[nhx,nhy,nhz],
                primitives[...,nhx,nhy,nhz])
            mu_2 = self.material_manager.get_bulk_viscosity(
                temperature[nhx,nhy,nhz],
                primitives[...,nhx,nhy,nhz]) - 2.0 / 3.0 * mu_1
            
            tau_i = self.compute_tau(velocity_gradient, axis_index, mu_1, mu_2)
            tau_i *= delta_aperture
            interface_flux_xi = interface_flux_xi.at[velocity_slices].add(tau_i)

            if is_moving_levelset:
                viscid_flux_xi_energy = 0
                for k in active_axes_indices:
                    viscid_flux_xi_energy += tau_i[k] * interface_velocity[k]
                interface_flux_xi = interface_flux_xi.at[energy_slices].add(viscid_flux_xi_energy)

        # HEAT FLUX
        if is_heat_flux and self.solid_temperature_model != "ADIABATIC":
            temperature_grad = self.derivative_stencil.derivative_xi(
                temperature[nhx__,nhy__,nhz__], cell_size, axis_index)
            thermal_conductivity = self.material_manager.get_thermal_conductivity(
                temperature[nhx,nhy,nhz], primitives[...,nhx,nhy,nhz])
            heat_xi = -thermal_conductivity * temperature_grad * delta_aperture
            interface_flux_xi = interface_flux_xi.at[energy_slices].add(heat_xi)

        return interface_flux_xi

    def fluid_fluid_flux_xi(
            self,
            primitives: Array,
            interface_velocity: Array,
            interface_pressure: Array,
            volume_fraction: Array,
            apertures: Array,
            normal: Array,
            axis_index: int,
            temperature: Array
            ) -> Array:

        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices

        is_convective_flux = self.numerical_setup.active_physics.is_convective_flux
        is_viscous_flux = self.numerical_setup.active_physics.is_viscous_flux
        is_heat_flux = self.numerical_setup.active_physics.is_heat_flux

        interface_velocity = interface_velocity[...,nhx_,nhy_,nhz_]
        interface_pressure = interface_pressure[...,nhx_,nhy_,nhz_]

        interface_flux_xi = jnp.zeros(primitives[...,nhx,nhy,nhz].shape)

        apertures = jnp.stack([apertures, 1.0 - apertures], axis=0) 
        volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0) 
        delta_aperture = apertures[self.aperture_slices[axis_index][0]][...,nhx_,nhy_,nhz_] - \
                            apertures[self.aperture_slices[axis_index][1]][...,nhx_,nhy_,nhz_]

        # CONVECTIVE FLUX
        if is_convective_flux:

            convective_flux_xi_momentum = interface_pressure * delta_aperture
            interface_flux_xi = interface_flux_xi.at[axis_index+1].add(convective_flux_xi_momentum)

            convective_flux_xi_energy = convective_flux_xi_momentum * interface_velocity * normal[axis_index,nhx_,nhy_,nhz_]
            interface_flux_xi = interface_flux_xi.at[4].add(convective_flux_xi_energy)

        # VISCOUS FLUX
        if is_viscous_flux:
            
            # COMPUTE VELOCITY GRADIENT
            velocity_0, velocity_1 = primitives[1:4,0,nhx__,nhy__,nhz__], primitives[1:4,1,nhx__,nhy__,nhz__]
            real_velocity = velocity_0 * volume_fraction[0] + velocity_1 * volume_fraction[1]
            velocity_gradient = self.compute_velocity_gradient(real_velocity)

            # COMPUTE INTERFACE VISCOSITY
            mu_1 = self.material_manager.get_dynamic_viscosity(temperature[...,nhx,nhy,nhz], None)
            mu_2 = self.material_manager.get_bulk_viscosity(temperature[...,nhx,nhy,nhz], None) - 2.0 / 3.0 * mu_1

            mu_1_interface = mu_1[0]*mu_1[1]/(volume_fraction[0,nhx_,nhy_,nhz_]*mu_1[1] + volume_fraction[1,nhx_,nhy_,nhz_]*mu_1[0] + 1e-20)
            mu_2_interface = mu_2[0]*mu_2[1]/(volume_fraction[0,nhx_,nhy_,nhz_]*mu_2[1] + volume_fraction[1,nhx_,nhy_,nhz_]*mu_2[0] + 1e-20)

            tau_i = self.compute_tau(velocity_gradient, axis_index, mu_1_interface, mu_2_interface)
            tau_i = jnp.expand_dims(tau_i, axis=1) * delta_aperture
            interface_flux_xi = interface_flux_xi.at[1:4].add(tau_i)

            viscid_flux_xi_energy = 0
            for k in active_axes_indices:
                viscid_flux_xi_energy += tau_i[k] * interface_velocity * normal[k,nhx_,nhy_,nhz_]
            interface_flux_xi = interface_flux_xi.at[4].add(viscid_flux_xi_energy)

        # HEAT FLUX
        if is_heat_flux:
            real_temperature = temperature[0,nhx__,nhy__,nhz__] * volume_fraction[0] + temperature[1,nhx__,nhy__,nhz__] * volume_fraction[1]
            temperature_grad = self.derivative_stencil.derivative_xi(real_temperature, cell_size, axis_index)

            thermal_conductivity = self.material_manager.get_thermal_conductivity(temperature[...,nhx,nhy,nhz], None)
            volume_fraction_0 = volume_fraction[0,nhx_,nhy_,nhz_]
            volume_fraction_1 = volume_fraction[1,nhx_,nhy_,nhz_]

            denominator_0 = thermal_conductivity[0]*volume_fraction_1
            denominator_1 = thermal_conductivity[1]*volume_fraction_0
            denominator = denominator_0 + denominator_1
            
            lambda_interface = thermal_conductivity[0]*thermal_conductivity[1]/(denominator + 1e-20)

            heat_xi = -lambda_interface * temperature_grad * delta_aperture
            interface_flux_xi = interface_flux_xi.at[4].add(heat_xi)

        return interface_flux_xi

    def compute_velocity_gradient(
            self,
            velocity: Array
            ) -> Array:
        active_axes_indices = self.domain_information.active_axes_indices
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        cell_size = self.domain_information.smallest_cell_size
        shape = velocity[...,nhx_,nhy_,nhz_].shape
        velocity_gradient = []
        for i in range(3):
            if i in active_axes_indices:
                gradient_xi = self.derivative_stencil.derivative_xi(
                    velocity, cell_size, i)
            else:
                gradient_xi = jnp.zeros(shape)
            velocity_gradient.append(gradient_xi)
        velocity_gradient = jnp.stack(velocity_gradient, axis=1)
        return velocity_gradient

    def compute_tau(
            self,
            velocity_gradient: Array,
            i: int,
            mu_1: Array,
            mu_2: Array
            ) -> Array:
        active_axes_indices = self.domain_information.active_axes_indices
        tau_i_list = []
        shape = velocity_gradient.shape[2:]
        for j in range(3):
            if j in active_axes_indices and i in active_axes_indices:
                tau_ij = velocity_gradient[i,j] + velocity_gradient[j,i]
                tau_ij *= mu_1
            else:
                tau_ij = jnp.zeros(shape)
            tau_i_list.append(tau_ij)
        tau_i_list[i] += mu_2 * sum([velocity_gradient[k,k] for k in active_axes_indices])
        tau_i = - jnp.stack(tau_i_list)
        return tau_i


    def compute_viscous_solid_force(
            self,
            primitives: Array,
            interface_velocity: Array,
            levelset: Array,
            normal: Array,
            interface_length: Array,
            ) -> Array:
        """Computes the viscous interface
        flux according to Meyer 2010
        """
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nh_conservatives = self.domain_information.nh_conservatives
        nx,ny,nz = self.domain_information.device_number_of_cells
        dim = self.domain_information.dim
        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices
        cell_centers_halos = self.domain_information.get_device_cell_centers_halos()

        no_primes = self.equation_information.no_primes
        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset

        mass_ids = self.equation_information.mass_ids
        energy_ids = self.equation_information.energy_ids
        velocity_slices = self.equation_information.velocity_slices

        normal = normal[...,nhx_,nhy_,nhz_]

        if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            interface_velocity = interface_velocity[...,nhx,nhy,nhz]

        dh = 0.0
        for axis_index in active_axes_indices:
            temp = normal[axis_index] * cell_size
            dh += temp * temp
        dh = 0.5 * jnp.sqrt(dh)

        mesh_grid = self.domain_information.compute_device_mesh_grid()
        mesh_grid = jnp.array(mesh_grid)    

        mask = compute_cut_cell_mask(levelset, nh_conservatives)
        s_ = jnp.s_[active_axes_indices,...]
        interpolation_point = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] + dh) * mask
        interpolation_point = interpolation_point.reshape(dim,-1)
        interpolation_point = jnp.swapaxes(interpolation_point, -1, 0)

        interpolated_primes = linear_interpolation_scattered(interpolation_point, primitives, cell_centers_halos)
        interpolated_primes = interpolated_primes.reshape(no_primes,nx,ny,nz)

        interpolated_density = interpolated_primes[mass_ids]
        interpolated_pressure = interpolated_primes[energy_ids]
        interpolated_velocity = interpolated_primes[velocity_slices]

        interpolated_velocity_normal = normal * jnp.sum(normal * interpolated_velocity, axis=0)
        interpolated_velocity_tangetial = interpolated_velocity - interpolated_velocity_normal

        if is_moving_levelset:
            interface_velocity_normal = normal * jnp.sum(normal * interface_velocity, axis=0)
            interface_velocity_tangential = interface_velocity - interface_velocity_normal
        else:
            interface_velocity_normal = 0.0
            interface_velocity_tangential = 0.0

        # COMPUTE INTERFACE VISCOSITY
        interpolated_temperature = self.material_manager.get_temperature(interpolated_primes)
        mu = self.material_manager.get_dynamic_viscosity(
            interpolated_temperature,
            interpolated_primes)
        momentum_contribution = -mu * (4/3*(interpolated_velocity_normal - interface_velocity_normal)/(dh + 1e-20)
            + (interpolated_velocity_tangetial - interface_velocity_tangential)/(dh + 1e-20)) * interface_length[nhx_,nhy_,nhz_]
        momentum_contribution *= mask

        return momentum_contribution
