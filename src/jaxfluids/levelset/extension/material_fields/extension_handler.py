
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks, compute_narrowband_mask
from jaxfluids.data_types.numerical_setup.levelset import LevelsetExtensionFieldSetup, NarrowBandSetup
from jaxfluids.equation_manager import EquationManager
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.config import precision
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

from jaxfluids.levelset.extension.material_fields.interpolation import interpolation_extension
from jaxfluids.levelset.extension.material_fields.iterative import iterative_extension


# free functions as we need this in init manager and sim manager
def ghost_cell_extension_material_fields(
        conservatives: Array,
        primitives: Array,
        levelset: Array,
        volume_fraction: Array,
        normal: Array,
        solid_temperature: Array,
        solid_velocity: Array,
        interface_heat_flux: Array,
        interface_temperature: Array,
        mixing_invalid_cells: Array,
        physical_simulation_time: float,
        extension_setup: LevelsetExtensionFieldSetup,
        narrowband_setup: NarrowBandSetup,
        cell_indices: LevelsetSolidCellIndicesField,
        extender: IterativeExtender,
        equation_manager: EquationManager,
        solid_properties_manager: SolidPropertiesManager,
        is_initialization: bool = False,
        ml_setup: MachineLearningSetup = None,
        ) -> Tuple[Array, Array, int,
                   LevelsetProcedureInformation]:
    """Performs the ghost cell extension
    the material fields.

    :param conservatives: _description_
    :type conservatives: Array
    :param primitives: _description_
    :type primitives: Array
    :param levelset: _description_
    :type levelset: Array
    :param volume_fraction: _description_
    :type volume_fraction: Array
    :param physical_simulation_time: _description_
    :type physical_simulation_time: float
    :param normal: _description_
    :type normal: Array
    :param mixing_invalid_cells: _description_
    :type mixing_invalid_cells: Array
    :param domain_information: _description_
    :type domain_information: DomainInformation
    :param extender: _description_
    :type extender: IterativeExtender
    :param equation_manager: _description_
    :type equation_manager: EquationManager
    :param steps: _description_, defaults to None
    :type steps: int, optional
    :param CFL: _description_, defaults to None
    :type CFL: int, optional
    :return: _description_
    :rtype: Tuple[Array, Array, int, LevelsetProcedureInformation]
    """
    

    equation_information = equation_manager.equation_information
    domain_information = extender.domain_information
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    smallest_cell_size = domain_information.smallest_cell_size

    narrowband_computation = narrowband_setup.computation_width

    method = extension_setup.method

    interpolation_setup = extension_setup.interpolation
    iterative_setup = extension_setup.iterative

    levelset_model = equation_information.levelset_model

    mask_real = compute_fluid_masks(volume_fraction, levelset_model)
    mask_real = mask_real[...,nhx_,nhy_,nhz_]
    mask_ghost = 1 - mask_real
    mask_narrowband = compute_narrowband_mask(levelset, smallest_cell_size, narrowband_computation)
    mask_narrowband = mask_narrowband[nhx,nhy,nhz]
    mask_extend = mask_ghost * mask_narrowband # we extend into narrowband ghost cells


    if method == "INTERPOLATION":
        
        primitives_extend, conservatives_extend = interpolation_extension(
            primitives, conservatives, levelset, normal,
            solid_velocity, solid_temperature,
            interface_heat_flux, interface_temperature,
            mask_extend, physical_simulation_time,
            cell_indices, interpolation_setup,
            domain_information, equation_manager,
            solid_properties_manager,
            ml_setup
        )

        info_prime_extension = None
        invalid_cell_count = None

    elif method == "ITERATIVE":
        
        (
            primitives_extend,
            conservatives_extend,
            invalid_cell_count,
            info_prime_extension,
            mask_extend
        ) = iterative_extension(
            primitives, conservatives, normal, mask_extend,
            mixing_invalid_cells, physical_simulation_time,
            iterative_setup, extender, equation_manager,
            is_initialization, ml_setup=ml_setup
        )

    is_stop_gradient = extension_setup.is_stopgradient
    if is_stop_gradient:
        primitives_extend = jax.lax.stop_gradient(primitives_extend)
        conservatives_extend = jax.lax.stop_gradient(conservatives_extend)
        primitives = primitives.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
        primitives = primitives.at[...,nhx,nhy,nhz].add(primitives_extend[...,nhx,nhy,nhz] * mask_extend)
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
        conservatives = conservatives.at[...,nhx,nhy,nhz].add(conservatives_extend[...,nhx,nhy,nhz] * mask_extend)
    else:
        primitives = primitives_extend
        conservatives = conservatives_extend


    # set eps in ghost cells outside narrowband
    eps = precision.get_eps()
    ids = equation_information.ids_mass_and_energy
    mask_cut_off = (1 - mask_narrowband) * mask_ghost
    primitives = primitives.at[...,nhx,nhy,nhz].mul(1 - mask_cut_off)
    primitives = primitives.at[ids,...,nhx,nhy,nhz].add(mask_cut_off * eps)
    conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_cut_off)
    conservatives = conservatives.at[ids,...,nhx,nhy,nhz].add(mask_cut_off * eps)

    return conservatives, primitives, invalid_cell_count, info_prime_extension