from typing import Tuple

import jax.numpy as jnp
import jax

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.numerical_setup.levelset import InterfaceFluxSetup
from jaxfluids.levelset.geometry.mask_functions import compute_cut_cell_mask_sign_change_based
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered

Array = jax.Array


def convective_interface_flux_xi(
        interface_pressure: Array,
        interface_velocity: Array,
        delta_aperture: Array,
        normal: Array,
        axis_index: int,
        domain_information: DomainInformation
        ) -> Array:
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    momentum_flux_xi = interface_pressure * delta_aperture
    energy_flux_xi = momentum_flux_xi * interface_velocity * normal[axis_index,nhx_,nhy_,nhz_] # TODO would it be more consistent to use aperture based normal here ?
    return momentum_flux_xi, energy_flux_xi

def viscous_interface_flux_xi(
        primitives: Array,
        volume_fraction: Array,
        temperature: Array,
        delta_aperture: Array,
        interface_velocity: Array,
        normal: Array,
        axis_index: int,
        material_manager: MaterialManager,
        derivative_stencil: SpatialDerivative,
        domain_information: DomainInformation,
        ) -> Array:

    active_physics = material_manager.equation_information.active_physics
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry
    active_axes_indices = domain_information.active_axes_indices
    cell_size = domain_information.smallest_cell_size  # NOTE levelset must be on finest, uniform grid !

    # NOTE real velocity is volume averaged
    velocity_0, velocity_1 = primitives[1:4,0,nhx__,nhy__,nhz__], primitives[1:4,1,nhx__,nhy__,nhz__]
    velocity = velocity_0 * volume_fraction[0] + velocity_1 * volume_fraction[1]

    # NOTE compute velocity gradient
    shape = velocity[...,nhx_,nhy_,nhz_].shape
    velocity_gradient = []
    for i in range(3):
        if i in active_axes_indices:
            gradient_xi = derivative_stencil.derivative_xi(
                velocity, cell_size, i)
        else:
            gradient_xi = jnp.zeros(shape)
        velocity_gradient.append(gradient_xi)
    velocity_gradient = jnp.stack(velocity_gradient, axis=1)

    # NOTE interface viscosity is volume averaged
    mu_1 = material_manager.get_dynamic_viscosity(temperature[...,nhx,nhy,nhz], None)
    mu_2 = material_manager.get_bulk_viscosity(temperature[...,nhx,nhy,nhz], None) - 2.0 / 3.0 * mu_1

    mu_1_interface = mu_1[0]*mu_1[1]/(volume_fraction[0,nhx_,nhy_,nhz_]*mu_1[1] + volume_fraction[1,nhx_,nhy_,nhz_]*mu_1[0] + 1e-20)
    mu_2_interface = mu_2[0]*mu_2[1]/(volume_fraction[0,nhx_,nhy_,nhz_]*mu_2[1] + volume_fraction[1,nhx_,nhy_,nhz_]*mu_2[0] + 1e-20)

    # NOTE compute stress tensor
    tau_i_list = []
    shape = velocity_gradient.shape[2:]
    for j in range(3):
        if j in active_axes_indices and axis_index in active_axes_indices:
            tau_ij = velocity_gradient[axis_index,j] + velocity_gradient[j,axis_index]
            tau_ij *= mu_1_interface
        else:
            tau_ij = jnp.zeros(shape)
        tau_i_list.append(tau_ij)
    tau_i_list[axis_index] += mu_2_interface * sum([velocity_gradient[k,k] for k in active_axes_indices])
    tau_i = - jnp.stack(tau_i_list)
    tau_i = jnp.expand_dims(tau_i, axis=1) * delta_aperture
    momentum_flux_xi = tau_i

    energy_flux_xi = 0
    if active_physics.is_viscous_heat_production:
        for k in active_axes_indices:
            energy_flux_xi += tau_i[k] * interface_velocity * normal[k,nhx_,nhy_,nhz_]
    
    return momentum_flux_xi, energy_flux_xi


def heat_interface_flux_xi(
        temperature: Array,
        volume_fraction: Array,
        delta_aperture: Array,
        axis_index: int,
        material_manager: MaterialManager,
        derivative_stencil: SpatialDerivative,
        domain_information: DomainInformation,
        ):

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry

    cell_size = domain_information.smallest_cell_size # NOTE levelset must be on finest, uniform grid !
        
    real_temperature = temperature[0,nhx__,nhy__,nhz__] * volume_fraction[0] + temperature[1,nhx__,nhy__,nhz__] * volume_fraction[1]
    temperature_grad = derivative_stencil.derivative_xi(real_temperature, cell_size, axis_index)

    thermal_conductivity = material_manager.get_thermal_conductivity(temperature[...,nhx,nhy,nhz], None)
    volume_fraction_0 = volume_fraction[0,nhx_,nhy_,nhz_]
    volume_fraction_1 = volume_fraction[1,nhx_,nhy_,nhz_]

    denominator_0 = thermal_conductivity[0]*volume_fraction_1
    denominator_1 = thermal_conductivity[1]*volume_fraction_0
    denominator = denominator_0 + denominator_1
    lambda_interface = thermal_conductivity[0]*thermal_conductivity[1]/(denominator + 1e-20)
    energy_flux_xi = -lambda_interface * temperature_grad * delta_aperture
    
    return energy_flux_xi


def viscous_interface_flux(
        primitives: Array,
        levelset: Array,
        interface_velocity: Array,
        normal: Array,
        interface_length: Array,
        interpolation_dh: float,
        material_properties_averaging: str,
        material_manager: MaterialManager,
        domain_information: DomainInformation,
        ):

    active_physics = material_manager.equation_information.active_physics
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry
    nx,ny,nz = domain_information.device_number_of_cells
    
    cell_size = domain_information.smallest_cell_size
    active_axes_indices = domain_information.active_axes_indices
    nh_conservatives = domain_information.nh_conservatives
    dim = domain_information.dim
    cell_centers_halos = domain_information.get_device_cell_centers_halos()

    normal = normal[...,nhx_,nhy_,nhz_]
    interface_velocity = interface_velocity[...,nhx_,nhy_,nhz_]

    dh = interpolation_dh * cell_size

    mesh_grid = domain_information.compute_device_mesh_grid()
    mesh_grid = jnp.array(mesh_grid)    

    s_ = jnp.s_[active_axes_indices,...]

    ip_pos = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] + dh)
    ip_pos = ip_pos.reshape(dim,-1)
    ip_pos = jnp.swapaxes(ip_pos, -1, 0)

    ip_neg = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] - dh)
    ip_neg = ip_neg.reshape(dim,-1)
    ip_neg = jnp.swapaxes(ip_neg, -1, 0)

    ip = jnp.stack([ip_pos, ip_neg], axis=0)

    interpolated_primitives = jax.vmap(linear_interpolation_scattered, in_axes=(0,1,None), out_axes=(1))(
        ip, primitives, cell_centers_halos)
    interpolated_primitives = interpolated_primitives.reshape(5,2,nx,ny,nz)

    normal = jnp.expand_dims(normal, axis=1)
    interpolated_velocity = interpolated_primitives[1:4]
    interpolated_velocity_normal = normal * jnp.sum(normal * interpolated_velocity, axis=0)
    interpolated_velocity_tangetial = interpolated_velocity - interpolated_velocity_normal

    interpolated_temperature = material_manager.get_temperature(interpolated_primitives)
    mu = material_manager.get_dynamic_viscosity(interpolated_temperature, None)
    
    mu_pos = mu[0]
    mu_neg = mu[1]

    if material_properties_averaging == "HARMONIC":
        mu = 2*mu_pos*mu_neg/(mu_pos + mu_neg + 1e-20)
    elif material_properties_averaging == "GEOMETRIC":
        mu = jnp.sqrt(mu_pos*mu_neg + 1e-20)
    else:
        raise NotImplementedError

    interpolated_velocity_tangetial_pos = interpolated_velocity_tangetial[:,0]
    interpolated_velocity_tangetial_neg = interpolated_velocity_tangetial[:,1]
    interpolated_velocity_normal_pos = interpolated_velocity_normal[:,0]
    interpolated_velocity_normal_neg = interpolated_velocity_normal[:,1]

    momentum_flux = - mu * (4/3*(interpolated_velocity_normal_pos - interpolated_velocity_normal_neg)/(2*dh) \
        + (interpolated_velocity_tangetial_pos - interpolated_velocity_tangetial_neg)/(2*dh)) * interface_length[nhx_,nhy_,nhz_]
    momentum_flux = jnp.stack([momentum_flux, -momentum_flux], axis=1)

    if active_physics.is_viscous_heat_production:
        energy_flux = jnp.sum(interface_velocity * normal * momentum_flux, axis=0)
    else:
        energy_flux = 0.0

    return momentum_flux, energy_flux

def heat_interface_flux(
        temperature: Array,
        levelset: Array,
        normal: Array,
        interface_length: Array,
        interpolation_dh: float,
        material_properties_averaging: str,
        material_manager: MaterialManager,
        domain_information: DomainInformation,
        ) -> Array:

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry
    nx,ny,nz = domain_information.device_number_of_cells
    
    cell_size = domain_information.smallest_cell_size
    active_axes_indices = domain_information.active_axes_indices
    nh_conservatives = domain_information.nh_conservatives
    dim = domain_information.dim
    cell_centers_halos = domain_information.get_device_cell_centers_halos()

    normal = normal[...,nhx_,nhy_,nhz_]

    dh = interpolation_dh * cell_size

    mesh_grid = domain_information.compute_device_mesh_grid()
    mesh_grid = jnp.array(mesh_grid)    

    s_ = jnp.s_[active_axes_indices,...]

    ip_pos = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] + dh)
    ip_pos = ip_pos.reshape(dim,-1)
    ip_pos = jnp.swapaxes(ip_pos, -1, 0)

    ip_neg = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] - dh)
    ip_neg = ip_neg.reshape(dim,-1)
    ip_neg = jnp.swapaxes(ip_neg, -1, 0)

    ip = jnp.stack([ip_pos, ip_neg], axis=0)

    temperature = jax.vmap(linear_interpolation_scattered, in_axes=(0,0,None), out_axes=(0))(
        ip, temperature, cell_centers_halos)

    temperature = temperature.reshape(2,nx,ny,nz)

    thermal_conductivity = material_manager.get_thermal_conductivity(
        temperature, None)

    temperature_pos = temperature[0]
    temperature_neg = temperature[1]
    lambda_pos = thermal_conductivity[0]
    lambda_neg = thermal_conductivity[1]

    if material_properties_averaging == "HARMONIC":
        thermal_conductivity = 2*lambda_pos*lambda_neg/(lambda_pos + lambda_neg + 1e-20)
    elif material_properties_averaging == "GEOMETRIC":
        thermal_conductivity = jnp.sqrt(lambda_pos*lambda_neg + 1e-20)
    else:
        raise NotImplementedError

    temperature_gradient = (temperature_pos - temperature_neg)/(2*dh)
    heat_flux = - thermal_conductivity * temperature_gradient * interface_length[nhx_,nhy_,nhz_]
    heat_flux = jnp.stack([heat_flux, -heat_flux], axis=0)

    return heat_flux
    

