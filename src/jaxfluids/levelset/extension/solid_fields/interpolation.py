from typing import Dict, Tuple

import jax.numpy as jnp
import jax
from jax import Array

from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.numerical_setup.levelset import InterpolationExtensionSetup
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField
from jaxfluids.domain.helper_functions import add_halo_offset
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

def interpolation_extension(
        solid_temperature: Array,
        solid_energy: Array,
        levelset: Array,
        interface_heat_flux: Array,
        interface_temperature: Array,
        mask_extend: Array,
        normal: Array,
        cell_indices: LevelsetSolidCellIndicesField,
        interpolation_setup: InterpolationExtensionSetup,
        domain_information: DomainInformation,
        solid_properties_manager: SolidPropertiesManager,
        ml_setup: MachineLearningSetup
    ) -> Array:

    is_parallel = domain_information.is_parallel
    active_axes_indices = domain_information.active_axes_indices
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    dim = domain_information.dim
    nx,ny,nz = domain_information.device_number_of_cells

    is_cell_based_computation = interpolation_setup.is_cell_based_computation
    is_cell_based_computation = is_cell_based_computation and cell_indices is not None

    mesh_grid = domain_information.compute_device_mesh_grid()
    mesh_grid = jnp.array(mesh_grid)
    cell_centers_halos = domain_information.get_device_cell_centers_halos()

    # for static solids, we only compute extension in narrowband, i.e., the
    # buffers with _cc have shape (...,N_gh), where N_gh is the number of ghost cells
    # in the extension band
    if is_cell_based_computation:
        s_gh = (...,) + cell_indices.indices
        normal_cc = normal[...,nhx_,nhy_,nhz_][s_gh]
        levelset_cc = levelset[nhx,nhy,nhz][s_gh]
        mesh_grid_cc = mesh_grid[s_gh]
        interface_heat_flux_cc = interface_heat_flux[s_gh]
        interface_temperature_cc = interface_temperature[s_gh]
    else:
        normal_cc = normal[...,nhx_,nhy_,nhz_]
        levelset_cc = levelset[nhx,nhy,nhz]
        mesh_grid_cc = mesh_grid
        interface_heat_flux_cc = interface_heat_flux
        interface_temperature_cc = interface_temperature

    s_ = jnp.s_[active_axes_indices,...]
    ip_meshgrid = mesh_grid_cc + normal_cc[s_] * (-2*levelset_cc)
    if not is_cell_based_computation:
        ip_meshgrid = ip_meshgrid.reshape(dim,-1)
    solid_temperature_ip = linear_interpolation_scattered(
        jnp.swapaxes(ip_meshgrid, -1, 0), solid_temperature,
        cell_centers_halos)
    if not is_cell_based_computation:
        solid_temperature_ip = solid_temperature_ip.reshape(nx,ny,nz)

    # neumann
    lambda_solid_ip = solid_properties_manager.compute_thermal_conductivity(
        ip_meshgrid, solid_temperature_ip)
    solid_temperature_gp = solid_temperature_ip - 2*jnp.abs(levelset_cc)/lambda_solid_ip * interface_heat_flux_cc
    solid_temperature_gp = jnp.clip(solid_temperature_ip, 1e-10, None) # TODO 

    # dirichlet
    # solid_temperature_gp = 2*interface_temperature_cc - solid_temperature_ip

    cht_cases_fix = False
    dim = domain_information.dim
    if cht_cases_fix:
        # # hard coded fix for conjugate heat conduction (2D,3D) case
        # mask1 = (jnp.sqrt(jnp.sum(mesh_grid**2,axis=0)) > 0.9)
        # mask2 = (jnp.sqrt(jnp.sum(mesh_grid**2,axis=0)) < 0.2)
        # solid_temperature_gp *= 1 - mask1
        # solid_temperature_gp *= 1 - mask2
        # solid_temperature_gp += mask1*(2.0*2 - solid_temperature_ip)
        # solid_temperature_gp += mask2*(1.0*2 - solid_temperature_ip)

        # hard coded fix for conjugate heat cylinder incompressible case
        mask1 = (jnp.sqrt(jnp.sum(mesh_grid**2,axis=0)) < 0.3)
        solid_temperature_gp *= 1 - mask1
        solid_temperature_gp += mask1*(1.1*2 - solid_temperature_ip)

        #  hard coded fix for conjugate heat cylinder incompressible case composite
        # mask1 = (jnp.sqrt(jnp.sum(mesh_grid**2,axis=0)) < 0.15)
        # solid_temperature_gp *= 1 - mask1
        # solid_temperature_gp += mask1*(1.1*2 - solid_temperature_ip)

    if is_cell_based_computation:
        nh = domain_information.nh_conservatives
        s_ = add_halo_offset(cell_indices.indices, nh, active_axes_indices)
        if is_parallel:
            solid_temperature = solid_temperature.at[s_].mul(1 - cell_indices.mask)
            solid_temperature = solid_temperature.at[s_].add(cell_indices.mask*solid_temperature_gp)
        else:
            solid_temperature = solid_temperature.at[s_].set(solid_temperature_gp)
    else:
        solid_temperature = solid_temperature.at[..., nhx,nhy,nhz].mul(1 - mask_extend)
        solid_temperature = solid_temperature.at[..., nhx,nhy,nhz].add(solid_temperature_gp*mask_extend)
    
    solid_energy_in_extend = solid_properties_manager.compute_internal_energy(solid_temperature[nhx,nhy,nhz])
    solid_energy = solid_energy.at[nhx,nhy,nhz].mul(1 - mask_extend)
    solid_energy = solid_energy.at[nhx,nhy,nhz].add(solid_energy_in_extend*mask_extend)

    return solid_temperature, solid_energy