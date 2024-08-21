
from typing import Tuple

import jax
from jax import Array
import jax.numpy as jnp

from jaxfluids.levelset.quantity_extender import QuantityExtender
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks
from jaxfluids.levelset.helper_functions import linear_filtering
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.equation_manager import EquationManager
from jaxfluids.config import precision

class GhostCellHandler():
    """_summary_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            extender_primes: QuantityExtender,
            equation_manager: EquationManager,
            levelset_setup: LevelsetSetup,
            ) -> None:

        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager
        self.extender_primes = extender_primes
        self.material_manager = equation_manager.material_manager
        self.levelset_setup = levelset_setup

    def perform_ghost_cell_treatment(
            self,
            conservatives: Array,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            physical_simulation_time: float,
            normal: Array,
            solid_velocity: Array = None,
            mixing_invalid_cells: Array = None,
            steps: int = None,
            CFL: int = None
            ) -> Tuple[Array, Array, float, float]:
        """Performs the ghost cell treatment on
        the material fields.
        1) Extend primitives into narrow band ghost cells
        2) Linear interpolation in invalid extension cells
        3) Adjust solid velocity in ghost cells (only for
            FLUID-SOLID simulations)
        4) Compute conservatives from primitives in
            narrow band ghost cells
        5) Set cut off values for material fields in
            ghost cells outside the narrow band 

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
        :param solid_velocity: _description_, defaults to None
        :type solid_velocity: Array, optional
        :param mask_small_cells: _description_, defaults to None
        :type mask_small_cells: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array, float, float]
        """
        
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        smallest_cell_size = self.domain_information.smallest_cell_size

        levelset_model = self.levelset_setup.model
        extension_setup = self.levelset_setup.extension
        reset_cells = extension_setup.reset_cells
        narrowband_setup = self.levelset_setup.narrowband
        narrowband_computation = narrowband_setup.computation_width

        # GEOMETRICAL QUANTITIES
        if levelset_model == "FLUID-FLUID":
            normal_extend = jnp.stack([-normal, normal], axis=1)
        else:
            normal_extend = -normal
        mask_real = compute_fluid_masks(volume_fraction, levelset_model)
        mask_real = mask_real[...,nhx_,nhy_,nhz_]
        mask_ghost = 1 - mask_real
        normalized_levelset = jnp.abs(levelset[nhx,nhy,nhz])/smallest_cell_size
        mask_narrowband = jnp.where(normalized_levelset <= narrowband_computation, 1, 0)
        mask_extend = mask_ghost * mask_narrowband
        if mixing_invalid_cells != None:
            mask_extend = jnp.maximum(mask_extend, mixing_invalid_cells)

        if reset_cells:
            ids = self.equation_information.mass_and_energy_ids
            primitives = primitives.at[...,nhx,nhy,nhz].mul(mask_real)
            primitives = primitives.at[ids,...,nhx,nhy,nhz].add((1.0 - mask_real)*self.eps)

        # EXTEND PRIMITIVES
        force_steps = True if steps != None else False
        CFL = extension_setup.CFL_primes if CFL == None else CFL
        steps = extension_setup.steps_primes if steps == None else steps
        primitives, step_count = self.extender_primes.extend(
            primitives, normal_extend, mask_extend, physical_simulation_time,
            CFL, steps, force_steps=force_steps)

        primitives, invalid_cell_count = \
        self.treat_invalid_cells(primitives,
                                 physical_simulation_time)

        # ADJUST SOLID VELOCITY
        is_solid_levelset = self.equation_information.is_solid_levelset
        is_viscous_flux = self.equation_information.active_physics.is_viscous_flux
        viscous_flux_method = self.levelset_setup.interface_flux.viscous_flux_method
        if is_solid_levelset and is_viscous_flux and viscous_flux_method == "JAXFLUIDS":
            primitives = self.treat_solid_ghost_cells(
                primitives, volume_fraction, solid_velocity)

        # COMPUTE CONSERVATIVES IN EXTENSION BAND
        cons_in_extend = self.equation_manager.get_conservatives_from_primitives(primitives[...,nhx,nhy,nhz]) * mask_extend
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_extend)
        conservatives = conservatives.at[...,nhx,nhy,nhz].add(cons_in_extend)

        # SET CUT OFF MATERIAL BUFFERS
        ids = self.equation_information.mass_and_energy_ids
        mask_cut_off = (1 - mask_narrowband) * mask_ghost
        primitives = primitives.at[...,nhx,nhy,nhz].mul(1 - mask_cut_off)
        primitives = primitives.at[ids,...,nhx,nhy,nhz].add(mask_cut_off * self.eps)
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1 - mask_cut_off)
        conservatives = conservatives.at[ids,...,nhx,nhy,nhz].add(mask_cut_off * self.eps)

        return conservatives, primitives, invalid_cell_count, step_count
    
    def treat_invalid_cells(
            self,
            primitives: Array,
            physical_simulation_time: float
            ) -> Tuple[Array, int]:
        """Tags invalid cells, i.e., cells
        with negative/zero density or pressure
        after the extension procedure.
        Subsequently interpolates the primitive
        state in these cells.

        :param primitives: _description_
        :type primitives: Array
        :return: _description_
        :rtype: Tuple[Array, int]
        """
    
        nh = self.domain_information.nh_conservatives
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        is_parallel = self.domain_information.is_parallel
        ids = self.equation_information.mass_and_energy_ids
        mass_ids = self.equation_information.mass_ids
        energy_ids = self.equation_information.energy_ids

        density = primitives[mass_ids,...,nhx,nhy,nhz]
        pressure = primitives[energy_ids,...,nhx,nhy,nhz]
        p_b = self.material_manager.get_background_pressure()
        invalid_cells = (density <= 0.0) | (pressure + p_b <= 0.0)

        primitives = primitives.at[...,nhx,nhy,nhz].mul(1 - invalid_cells)
        primitives = primitives.at[ids,...,nhx,nhy,nhz].add(invalid_cells * self.eps)

        primitives = self.halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, True, True)
        filtered_primitives = linear_filtering(primitives, nh)
        primitives = primitives.at[...,nhx,nhy,nhz].add(invalid_cells * filtered_primitives)

        invalid_cell_count = jnp.sum(invalid_cells, axis=(-3,-2,-1))
        if is_parallel:
            invalid_cell_count = jax.lax.psum(invalid_cell_count, axis_name="i")

        return primitives, invalid_cell_count

    
    def treat_solid_ghost_cells(
            self,
            primitives: Array,
            volume_fraction: Array,
            solid_velocity: Array
            ) -> Array:
        """Adjusts the velocity within the solid
        ghost cells for viscous effects.

        :param normal: _description_
        :type normal: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param solid_velocity: _description_
        :type solid_velocity: Array
        :return: _description_
        :rtype: Array
        """
        
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        is_moving_levelset = self.equation_information.is_moving_levelset
        velocity_slices = self.equation_information.velocity_slices
        levelset_model = self.levelset_setup.model

        mask_real = compute_fluid_masks(volume_fraction, levelset_model)
        mask_ghost = 1 - mask_real[...,nhx_,nhy_,nhz_]
        mask = jnp.where(mask_ghost == 1.0, -1.0, 1.0)
        primitives = primitives.at[velocity_slices,...,nhx,nhy,nhz].mul(mask)
        
        if is_moving_levelset:
            primitives = primitives.at[velocity_slices,...,nhx,nhy,nhz].add(mask_ghost * 2 * solid_velocity)

        return primitives
