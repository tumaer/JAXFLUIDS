from typing import Tuple, Dict

import jax.numpy as jnp
import jax

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.numerical_setup.levelset import InterfaceFluxSetup
from jaxfluids.levelset.geometry.mask_functions import compute_cut_cell_mask_sign_change_based
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered
from jaxfluids.initialization.helper_functions import create_field_buffer
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

Array = jax.Array

def convective_interface_flux(
        primitives: Array,
        levelset: Array,
        normal: Array,
        interface_length: Array,
        solid_velocity: Array,
        mesh_grid: Array,
        dh: float,
        is_cell_based_computation: bool,
        is_interpolate_pressure: bool,
        material_manager: MaterialManager,
        domain_information: DomainInformation,
        ) -> Tuple[Array, Array]:


    equation_information = material_manager.equation_information
    is_moving_levelset = equation_information.is_moving_levelset
    s_energy = equation_information.s_energy
    
    pressure = primitives[s_energy]

    cell_centers_halos = domain_information.get_device_cell_centers_halos()
    active_axes_indices = domain_information.active_axes_indices

    dim = domain_information.dim
    nx,ny,nz = domain_information.device_number_of_cells

    if is_interpolate_pressure:

        s_ = jnp.s_[active_axes_indices,...]
        ip_fluid = mesh_grid + normal[s_] * (-levelset + dh)
        if not is_cell_based_computation:
            ip_fluid = ip_fluid.reshape(dim,-1)
        pressure_ip = linear_interpolation_scattered(
            jnp.swapaxes(ip_fluid, -1, 0), primitives[s_energy],
            cell_centers_halos)
        if not is_cell_based_computation:
            pressure_ip = pressure_ip.reshape(1,nx,ny,nz)

        y2 = pressure_ip
        y1 = pressure
        n2 = dh
        n1 = levelset

        # NOTE second order poly
        # pressure_interface = y1 - (y2 - y1)/(n2**2 - n1**2)*n1**2
        # pressure_interface = pressure_ip

        # NOTE linear poly
        pressure_interface = y2 - (y2-y1)/(n2-n1) * n2

    else:

        pressure_interface = pressure

    momentum_flux = pressure_interface * normal * interface_length

    if is_moving_levelset:
        energy_flux = jnp.sum(momentum_flux * solid_velocity, axis=0)
    else:
        energy_flux = 0.0

    return momentum_flux, energy_flux


def viscous_interface_flux(
        primitives: Array,
        normal: Array,
        interface_length: Array,
        solid_velocity: Array,
        dh: float,
        material_manager: MaterialManager,
        domain_information: DomainInformation,
        ) -> Tuple[Array, Array]:

    equation_information = material_manager.equation_information
    is_moving_levelset = equation_information.is_moving_levelset
    s_velocity = equation_information.s_velocity

    velocity = primitives[s_velocity]
    velocity_normal = normal * jnp.sum(normal * velocity, axis=0)
    velocity_tangential = velocity - velocity_normal

    if is_moving_levelset:
        solid_velocity_normal = normal * jnp.sum(normal * solid_velocity, axis=0)
        solid_velocity_tangential = solid_velocity - solid_velocity_normal
    else:
        solid_velocity_normal = 0.0
        solid_velocity_tangential = 0.0

    temperature = material_manager.get_temperature(primitives)
    mu = material_manager.get_dynamic_viscosity(
        temperature,
        primitives)
    momentum_flux = - mu * (4/3*(velocity_normal - solid_velocity_normal)/dh \
        + (velocity_tangential - solid_velocity_tangential)/dh) * interface_length

    if is_moving_levelset:
        energy_flux = jnp.sum(solid_velocity * momentum_flux, axis=0)
    else:
        energy_flux = 0.0

    return momentum_flux, energy_flux

def heat_interface_flux(
        primitives: Array,
        solid_temperature: Array,
        ip_solid: Array,
        interface_length: Array,
        dh: float,
        material_properties_averaging: str,
        ml_setup: MachineLearningSetup,
        material_manager: MaterialManager,
        solid_properties_manager: SolidPropertiesManager,
        domain_information: DomainInformation,
    ) -> Array:

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nx,ny,nz = domain_information.device_number_of_cells
    cell_centers_halos = domain_information.get_device_cell_centers_halos()

    temperature = material_manager.get_temperature(primitives)
    thermal_conductivity = material_manager.get_thermal_conductivity(
        temperature, primitives)
    
    solid_coupling = material_manager.equation_information.solid_coupling
    
    if solid_coupling.thermal == "TWO-WAY":
        raise NotImplementedError

    temperature_gradient = (temperature - solid_temperature)/dh
    heat_flux = - thermal_conductivity * temperature_gradient * interface_length

    return heat_flux
    

