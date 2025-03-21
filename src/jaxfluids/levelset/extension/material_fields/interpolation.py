
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField
from jaxfluids.data_types.numerical_setup.levelset import InterpolationExtensionSetup
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered
from jaxfluids.levelset.fluid_solid.helper_functions import get_solid_velocity_and_temperature
from jaxfluids.domain.helper_functions import add_halo_offset
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

def interpolation_extension(
        primitives: Array,
        conservatives: Array,
        levelset: Array,
        normal: Array,
        solid_velocity: Array,
        solid_temperature: Array,
        interface_heat_flux: Array,
        interface_temperature: Array,
        mask_extend: Array,
        physical_simulation_time: float,
        cell_indices: LevelsetSolidCellIndicesField,
        interpolation_setup: InterpolationExtensionSetup,
        domain_information: DomainInformation,
        equation_manager: EquationManager,
        solid_properties_manager: SolidPropertiesManager,
        ml_setup: MachineLearningSetup
    ) -> Tuple[Array, Array]:

    equation_information = equation_manager.equation_information
    material_manager = equation_manager.material_manager

    is_viscous_flux = equation_information.active_physics.is_viscous_flux
    is_heat_flux = equation_information.active_physics.is_heat_flux
    no_primes = equation_information.no_primes

    vel_slices = equation_information.s_velocity
    s_energy = equation_information.s_energy
    s_mass = equation_information.s_mass

    dim = domain_information.dim
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nx,ny,nz = domain_information.device_number_of_cells
    active_axes_indices = domain_information.active_axes_indices
    is_parallel = domain_information.is_parallel

    solid_coupling = equation_information.solid_coupling

    if solid_coupling.dynamic == "ONE-WAY":
        solid_velocity = solid_properties_manager.compute_imposed_solid_velocity(
            physical_simulation_time,
            ml_setup
        )
    elif solid_coupling.dynamic == "TWO-WAY":
        raise NotImplementedError
    else:
        pass

    if solid_coupling.thermal == "ONE-WAY":
        solid_temperature = solid_properties_manager.compute_imposed_solid_temperature(physical_simulation_time)
    elif solid_coupling.thermal == "TWO-WAY":
        raise NotImplementedError
    else:
        pass

    cell_centers_halos = domain_information.get_device_cell_centers_halos()
    mesh_grid = domain_information.compute_device_mesh_grid()
    mesh_grid = jnp.stack(mesh_grid, axis=0)

    is_cell_based_computation = interpolation_setup.is_cell_based_computation
    is_cell_based_computation = is_cell_based_computation and cell_indices is not None

    # for static solids, we only compute extension in narrowband, i.e., the
    # buffers with _cc have shape (...,N_gh), where N_gh is the number of ghost cells
    # in the extension band
    if is_cell_based_computation:

        s_gh = (...,) + cell_indices.indices
        normal_cc = normal[...,nhx_,nhy_,nhz_][s_gh]
        levelset_cc = levelset[nhx,nhy,nhz][s_gh]
        mesh_grid_cc = mesh_grid[s_gh]
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        if solid_velocity is not None:
            solid_velocity = solid_velocity[s_gh]
        if solid_temperature is not None:
            solid_temperature = solid_temperature[s_gh]
    else:
        normal_cc = normal[...,nhx_,nhy_,nhz_]
        levelset_cc = levelset[nhx,nhy,nhz]
        mesh_grid_cc = mesh_grid
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
    
    s_ = jnp.s_[active_axes_indices,...]
    ip = mesh_grid_cc + normal_cc[s_] * (-2*levelset_cc)
    if not is_cell_based_computation:
        ip = ip.reshape(dim,-1)
    primitives_ip = linear_interpolation_scattered(
        jnp.swapaxes(ip, -1, 0), primitives,
        cell_centers_halos)
    if not is_cell_based_computation:
        primitives_ip = primitives_ip.reshape(no_primes,nx,ny,nz)

    velocity_ip = primitives_ip[vel_slices]
    pressure_ip = primitives_ip[s_energy]
    density_ip = primitives_ip[s_mass]

    # symmetry for pressure (zerogradient)
    pressure_gp = pressure_ip 

    # no slip for viscous, symmetry for inviscid
    if is_viscous_flux:
        velocity_gp = 2*solid_velocity - velocity_ip if solid_velocity is not None else -velocity_ip
    else:
        velocity_gp = velocity_ip - 2 * jnp.sum(velocity_ip * normal_cc, axis=0) * normal_cc

    # symmetry for inactive heat flux
    # for active heat flux, we first compute temperature and
    # then compute density from pressure and temperature
    if solid_coupling.thermal == "ONE-WAY" and is_heat_flux:
        # negative temperatures in GP can occur due to significantly
        # higher fluid temperatures adjacent to solid
        temperature_ip = material_manager.get_temperature(primitives_ip)
        temperature_gp = 2*solid_temperature - temperature_ip # TODO aaron if prescribed solid_temperature is not uniform, it should be interpolated on interface
        temperature_gp = jnp.clip(temperature_gp, 1e-10, None) # TODO 

        if equation_information.equation_type in "SINGLE-PHASE":
            density_gp = material_manager.get_density_from_pressure_and_temperature(
                pressure_gp, temperature_gp)
        else:
            raise NotImplementedError
        
    elif solid_coupling.thermal == "TWO-WAY":
        raise NotImplementedError

    else:
        density_gp = density_ip

    if is_cell_based_computation:
    
        nh = domain_information.nh_conservatives
        s_tuple = add_halo_offset(cell_indices.indices, nh, active_axes_indices)
        
        s_vel = (jnp.s_[vel_slices],) + s_tuple
        s_energy = (jnp.s_[s_energy],) + s_tuple
        s_mass = (jnp.s_[s_mass],) + s_tuple

        if is_parallel:
            primitives = primitives.at[s_vel].mul(1 - cell_indices.mask)
            primitives = primitives.at[s_energy].mul(1 - cell_indices.mask)
            primitives = primitives.at[s_mass].mul(1 - cell_indices.mask)
            primitives = primitives.at[s_vel].add(cell_indices.mask*velocity_gp)
            primitives = primitives.at[s_energy].add(cell_indices.mask*pressure_gp)
            primitives = primitives.at[s_mass].add(cell_indices.mask*density_gp)
        else:
            primitives = primitives.at[s_vel].set(velocity_gp)
            primitives = primitives.at[s_energy].set(pressure_gp)
            primitives = primitives.at[s_mass].set(density_gp)

    else:
        primitives = primitives.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
        primitives = primitives.at[vel_slices,nhx,nhy,nhz].add(mask_extend * velocity_gp)
        primitives = primitives.at[s_energy,nhx,nhy,nhz].add(mask_extend * pressure_gp)
        primitives = primitives.at[s_mass,nhx,nhy,nhz].add(mask_extend * density_gp)

    cons_in_extend = equation_manager.get_conservatives_from_primitives(primitives[...,nhx,nhy,nhz]) * mask_extend

    conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
    conservatives = conservatives.at[...,nhx,nhy,nhz].add(cons_in_extend)

    return primitives, conservatives

