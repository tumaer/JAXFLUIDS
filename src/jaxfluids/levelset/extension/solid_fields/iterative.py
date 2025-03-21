from typing import Dict, Tuple

import jax.numpy as jnp
import jax
from jax import Array

from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.numerical_setup.levelset import IterativeExtensionSetup
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.math.filter.linear_averaging import linear_averaging
from jaxfluids.config import precision
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

def iterative_extension(
        solid_temperature: Array,
        solid_energy: Array,
        normal: Array,
        mask_extend: Array,
        mixing_invalid_cells: Array,
        physical_simulation_time: Array,
        iterative_setup: IterativeExtensionSetup,
        extender: IterativeExtender,
        solid_properties_manager: SolidPropertiesManager,
        is_initialization: bool = False,
        ml_setup: MachineLearningSetup = None
        ) -> Tuple[Array, int, LevelsetProcedureInformation]:

    domain_information = extender.domain_information
    halo_manager = extender.halo_manager

    if is_initialization:
        CFL = 0.5
        steps = 50
    else:
        CFL = iterative_setup.CFL
        steps = iterative_setup.steps
    
    is_interpolate_invalid_cells = iterative_setup.is_interpolate_invalid_cells
    is_extend_into_invalid_mixing_cells = iterative_setup.is_extend_into_invalid_mixing_cells

    if mixing_invalid_cells is not None and is_extend_into_invalid_mixing_cells:
        mask_extend = jnp.maximum(mask_extend, mixing_invalid_cells) # NOTE we extend in invalid mixing cells

    solid_temperature, info_solid_extension = extender.extend(
        solid_temperature, normal, mask_extend, physical_simulation_time,
        CFL, steps, ml_setup=ml_setup)
    
    if is_interpolate_invalid_cells:
        nh = domain_information.nh_conservatives
        nhx,nhy,nhz = domain_information.domain_slices_conservatives
        is_parallel = domain_information.is_parallel

        invalid_cells = solid_temperature[...,nhx,nhy,nhz] <= 0.0

        solid_temperature = solid_temperature.at[...,nhx,nhy,nhz].mul(1 - invalid_cells)
        solid_temperature = solid_temperature.at[...,nhx,nhy,nhz].add(invalid_cells * precision.get_eps())

        solid_temperature = halo_manager.perform_halo_update_solids(
            solid_temperature, physical_simulation_time, True, True, None, False) # NOTE face halos already updated from extension
        avergaged_solid_temperature = linear_averaging(solid_temperature, nh)
        solid_temperature = solid_temperature.at[...,nhx,nhy,nhz].add(invalid_cells * avergaged_solid_temperature)

        invalid_cell_count = jnp.sum(invalid_cells, axis=(-3,-2,-1))
        if is_parallel:
            invalid_cell_count = jax.lax.psum(invalid_cell_count, axis_name="i")
    else:
        invalid_cell_count = None

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    solid_energy_extend = solid_properties_manager.compute_internal_energy(solid_temperature[...,nhx,nhy,nhz]) * mask_extend.astype(float)
    solid_energy = solid_energy.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
    solid_energy = solid_energy.at[...,nhx,nhy,nhz].add(solid_energy_extend)

    return solid_temperature, solid_energy, invalid_cell_count, info_solid_extension, mask_extend
