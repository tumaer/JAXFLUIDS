
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.math.filter.linear_averaging import linear_averaging
from jaxfluids.equation_manager import EquationManager
from jaxfluids.config import precision
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.data_types.numerical_setup.levelset import IterativeExtensionSetup


def iterative_extension(
        primitives: Array,
        conservatives: Array,
        normal: Array,
        mask_extend: Array,
        mixing_invalid_cells: Array,
        physical_simulation_time: float,
        iterative_setup: IterativeExtensionSetup,
        extender: IterativeExtender,
        equation_manager: EquationManager,
        is_initialization: bool = False,
        ml_setup: MachineLearningSetup = None
        ) -> Tuple[Array, Array, int, LevelsetProcedureInformation]:

    equation_information = equation_manager.equation_information
    domain_information = extender.domain_information

    levelset_model = equation_information.levelset_model

    is_parallel = domain_information.is_parallel
    nh = domain_information.nh_conservatives
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    is_parallel = domain_information.is_parallel
    ids_mass_and_energy = equation_information.ids_mass_and_energy
    ids_mass = equation_information.ids_mass
    ids_energy = equation_information.ids_energy

    halo_manager = extender.halo_manager
    material_manager = equation_manager.material_manager

    is_interpolate_invalid_cells = iterative_setup.is_interpolate_invalid_cells
    is_extend_into_invalid_mixing_cells = iterative_setup.is_extend_into_invalid_mixing_cells
    if mixing_invalid_cells is not None and is_extend_into_invalid_mixing_cells:
        mask_extend = jnp.maximum(mask_extend, mixing_invalid_cells) # for iterative procedure, we also extend into invalid cells after mixing
        
    if levelset_model == "FLUID-FLUID":
        normal_extend = jnp.stack([-normal, normal], axis=1)
    else:
        normal_extend = -normal

    # hard coded 50 steps for extension during initialization
    if is_initialization:
        CFL = 0.5
        steps = 50
    else:
        CFL = iterative_setup.CFL
        steps = iterative_setup.steps

    primitives, info_prime_extension = extender.extend(
        primitives, normal_extend, mask_extend, physical_simulation_time,
        CFL, steps, ml_setup=ml_setup)

    if is_interpolate_invalid_cells:
        density = primitives[ids_mass,...,nhx,nhy,nhz]
        pressure = primitives[ids_energy,...,nhx,nhy,nhz]
        p_b = material_manager.get_background_pressure()
        invalid_cells = ((density <= 0.0) | (pressure + p_b <= 0.0)) & mask_extend # only interpolate extension cells

        primitives = primitives.at[...,nhx,nhy,nhz].mul(1 - invalid_cells)
        primitives = primitives.at[ids_mass_and_energy,...,nhx,nhy,nhz].add(invalid_cells * precision.get_eps())

        primitives = halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, True, True, None, False,
            ml_setup=ml_setup) # face halos already updated from extension
        filtered_primitives = linear_averaging(primitives, nh)
        primitives = primitives.at[...,nhx,nhy,nhz].add(invalid_cells * filtered_primitives)

        invalid_cell_count = jnp.sum(invalid_cells, axis=(-3,-2,-1))
        if is_parallel:
            invalid_cell_count = jax.lax.psum(invalid_cell_count, axis_name="i")
    else:
        invalid_cell_count = None

    cons_in_extend = equation_manager.get_conservatives_from_primitives(primitives[...,nhx,nhy,nhz]) * mask_extend.astype(float)
    conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
    conservatives = conservatives.at[...,nhx,nhy,nhz].add(cons_in_extend)

    return primitives, conservatives, invalid_cell_count, info_prime_extension, mask_extend

    
