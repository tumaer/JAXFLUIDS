from typing import Tuple

import jax.numpy as jnp
import jax

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered
from jaxfluids.data_types.numerical_setup.levelset import InterfaceFluxSetup

Array = jax.Array

# free function as this is needed in sim manager and init manager

# TODO we currently perform the interpolation procedure at the interface twice,
# TODO 1) for the interface flux for the rhs, 2) for the extension procedure
# TODO should we only do it once and store the interpolated interface state ?
def compute_thermal_interface_state(
        primitives: Array,
        solid_temperature: Array,
        levelset: Array,
        normal: Array,
        interface_flux_setup: InterfaceFluxSetup,
        material_manager: MaterialManager,
        solid_properties_manager: SolidPropertiesManager,
        domain_information: DomainInformation
        ) -> Tuple[Array, Array]:
    """Computes interface temperatures and interface heat flux.1

    :param temperature: _description_
    :type temperature: Array
    :param solid_temperature: _description_
    :type solid_temperature: Array
    :param levelset: _description_
    :type levelset: Array
    :param normal: _description_
    :type normal: Array
    :param interpolation_dh: _description_
    :type interpolation_dh: float
    :param domain_information: _description_
    :type domain_information: DomainInformation
    :return: _description_
    :rtype: Tuple[Array, Array]
    """

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nx,ny,nz = domain_information.device_number_of_cells
    dim = domain_information.dim
    cell_size = domain_information.smallest_cell_size
    active_axes_indices = domain_information.active_axes_indices
    no_primes = material_manager.equation_information.no_primes

    dh = interface_flux_setup.interpolation_dh * cell_size

    cell_centers_halos = domain_information.get_device_cell_centers_halos()
    mesh_grid = domain_information.compute_device_mesh_grid()
    mesh_grid = jnp.array(mesh_grid)    

    s_ = jnp.s_[active_axes_indices,nhx_,nhy_,nhz_]

    fluid_ip = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] + dh)
    fluid_ip = fluid_ip.reshape(dim,-1)

    primitives_ip = linear_interpolation_scattered(
        jnp.swapaxes(fluid_ip, -1, 0), primitives,
        cell_centers_halos)
    primitives_ip = primitives_ip.reshape(no_primes,nx,ny,nz)
    temperature_ip = material_manager.get_temperature(primitives_ip)

    solid_ip_grid = mesh_grid + normal[s_] * (-levelset[nhx,nhy,nhz] - dh)
    solid_ip = solid_ip_grid.reshape(dim,-1)
    solid_temperature_ip = linear_interpolation_scattered(
        jnp.swapaxes(solid_ip, -1, 0), solid_temperature,
        cell_centers_halos)
    solid_temperature_ip = solid_temperature_ip.reshape(nx,ny,nz)

    lambda_fluid_ip = material_manager.get_thermal_conductivity(temperature_ip, primitives_ip)
    lambda_solid_ip = solid_properties_manager.compute_thermal_conductivity(solid_ip_grid, solid_temperature_ip)

    interface_temperature = (temperature_ip*lambda_fluid_ip + solid_temperature_ip*lambda_solid_ip)/(lambda_fluid_ip+lambda_solid_ip)

    if interface_flux_setup.material_properties_averaging == "HARMONIC":
        lambda_interface = 2*lambda_fluid_ip*lambda_solid_ip/(lambda_fluid_ip+lambda_solid_ip)
    elif interface_flux_setup.material_properties_averaging == "GEOMETRIC":
        lambda_interface = jnp.sqrt(lambda_fluid_ip*lambda_solid_ip)
    temperature_gradient = (temperature_ip - solid_temperature_ip)/dh/2
    interface_heat_flux = - lambda_interface * temperature_gradient

    return interface_heat_flux, interface_temperature