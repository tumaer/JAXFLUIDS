from functools import partial
from typing import List, Tuple, Union, Dict
import types

import jax
import jax.numpy as jnp

from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.helper_functions import transform_to_volume_average
from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks, compute_cut_cell_mask_sign_change_based, compute_narrowband_mask
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.levelset.fluid_fluid.fluid_fluid_handler import FluidFluidLevelsetHandler
from jaxfluids.levelset.fluid_solid.fluid_solid_handler import FluidSolidLevelsetHandler
from jaxfluids.levelset.extension.material_fields.extension_handler import ghost_cell_extension_material_fields
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.equation_manager import EquationManager
from jaxfluids.levelset.mixing import ConservativeMixer, SolidsMixer
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.information import LevelsetPositivityInformation, LevelsetResidualInformation
from jaxfluids.levelset.residual_computer import ResidualComputer
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField, LevelsetSolidCellIndices
from jaxfluids.config import precision

Array = jax.Array

class LevelsetHandler():
    
    def __init__(
            self,
            domain_information: DomainInformation,
            numerical_setup: NumericalSetup,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            halo_manager: HaloManager,
            solid_properties_manager: SolidPropertiesManager
            ) -> None:

        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.halo_manager = halo_manager
        self.domain_information = domain_information
        self.equation_information = equation_manager.equation_information
        self.levelset_setup = numerical_setup.levelset
        
        model = self.levelset_setup.model

        levelset_setup = numerical_setup.levelset
        self.geometry_calculator = GeometryCalculator(
            domain_information = domain_information,
            geometry_setup = numerical_setup.levelset.geometry,
            halo_cells_geometry = levelset_setup.halo_cells,
            narrowband_computation = levelset_setup.narrowband.computation_width
            )
        
        levelset_setup = numerical_setup.levelset

        iterative_extension_setup = levelset_setup.extension.primitives.iterative
        self.extender_primes = IterativeExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            is_jaxwhileloop = iterative_extension_setup.is_jaxwhileloop,
            residual_threshold = iterative_extension_setup.residual_threshold,
            extension_quantity = "primitives"
            )
    
        narrowband_setup = levelset_setup.narrowband
        reinitialization_setup = levelset_setup.reinitialization_runtime
        reinitializer = reinitialization_setup.type
        self.reinitializer: PDEBasedReinitializer = reinitializer(
            domain_information = domain_information,
            halo_manager = halo_manager,
            reinitialization_setup = reinitialization_setup,
            narrowband_setup = narrowband_setup
            )
    
        self.conservatives_mixer = ConservativeMixer(
            domain_information = domain_information,
            levelset_setup = levelset_setup,
            material_manager = material_manager,
            equation_manager = equation_manager,
            halo_manager = halo_manager
            )

        levelset_advection_stencil = levelset_setup.levelset_advection_stencil
        self.levelset_advection_stencil: SpatialDerivative = levelset_advection_stencil(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes
            )

        if model == "FLUID-FLUID":
            iterative_extension_setup = levelset_setup.extension.interface.iterative
            extender_interface = IterativeExtender(
                domain_information = domain_information,
                halo_manager = halo_manager,
                is_jaxwhileloop = iterative_extension_setup.is_jaxwhileloop,
                residual_threshold = iterative_extension_setup.residual_threshold,
                extension_quantity = "interface"
                )
            self.fluid_fluid_handler = FluidFluidLevelsetHandler(
                domain_information = domain_information,
                material_manager = material_manager,
                geometry_calculator = self.geometry_calculator,
                extender_interface = extender_interface,
                levelset_setup = levelset_setup
                )

            self.fluid_solid_handler = None
            extender_solids = None

        elif "FLUID-SOLID" in model:
            if levelset_setup.solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

            else:
                extender_solids = None

            self.fluid_solid_handler = FluidSolidLevelsetHandler(
                domain_information = domain_information,
                material_manager = material_manager,
                halo_manager = halo_manager,
                geometry_calculator = self.geometry_calculator,
                levelset_setup = levelset_setup,
                solid_properties_manager = solid_properties_manager,
                extender = extender_solids,
                )
            
            self.fluid_fluid_handler = None
            extender_interface = None

        else:
            raise NotImplementedError

        active_axes_indices = domain_information.active_axes_indices
        index_pairs = [(0,1), (0,2), (1,2)]
        self.index_pairs_mixing = [] 
        for pair in index_pairs:
            if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
                self.index_pairs_mixing.append(pair)


    def compute_levelset_advection_rhs_xi(
            self,
            levelset: Array,
            interface_velocity: Array,
            solid_velocity: Array,
            axis: int,
            physical_simulation_time: float,
            ml_setup: MachineLearningSetup
        ) -> Array:
        """Computes the right-hand-side of the
        levelset advection equation.

        FLUID-FLUID
            solid_velocity is None
        FLUID-SOLID
            interface_velocity is None
            ONE-WAY: solid_velocity is None, 
            
        :param levelset: Levelset buffer
        :type levelset: Array
        :param interface_velocity: Interface velocity buffer
        :type interface_velocity: Array
        :param axis: Current axis
        :type axis: int
        :return: right-hand-side contribution for current axis
        :rtype: Array
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        smallest_cell_size = self.domain_information.smallest_cell_size
        cell_sizes = self.domain_information.get_device_cell_sizes()

        levelset_model = self.levelset_setup.model
        narrowband_computation = self.levelset_setup.narrowband.computation_width
        solid_coupling = self.levelset_setup.solid_coupling
        
        if levelset_model == "FLUID-FLUID":
            levelset_velocity = interface_velocity[...,nhx_,nhy_,nhz_]
        elif levelset_model == "FLUID-SOLID":
            if solid_coupling.dynamic == "ONE-WAY":
                solid_properties_manager = self.fluid_solid_handler.solid_properties_manager
                levelset_velocity = solid_properties_manager.compute_imposed_solid_velocity(
                    physical_simulation_time,
                    ml_setup
                )
            elif solid_coupling.dynamic == "TWO-WAY":
                raise NotImplementedError

            else:
                levelset_velocity = 0.0
        else:
            RuntimeError

        # GEOMETRICAL QUANTITIES
        normal = self.geometry_calculator.compute_normal(levelset)
        if levelset_model == "FLUID-FLUID":
            mask_narrowband = compute_narrowband_mask(levelset, smallest_cell_size, narrowband_computation)
            mask_narrowband = mask_narrowband[nhx,nhy,nhz]
        else:
            mask_narrowband = jnp.ones_like(levelset[nhx,nhy,nhz])

        # DERIVATIVE
        derivative_L = self.levelset_advection_stencil.derivative_xi(levelset, cell_sizes[axis], axis, 0)
        derivative_R = self.levelset_advection_stencil.derivative_xi(levelset, cell_sizes[axis], axis, 1)

        # UPWINDING
        if levelset_model == "FLUID-FLUID":
            velocity = levelset_velocity * normal[axis,nhx_,nhy_,nhz_]
        else:
            velocity = levelset_velocity[axis]
        derivative = jnp.where(velocity >= 0.0, derivative_L, derivative_R)
    
        # SUM RHS
        rhs_contribution = - velocity * derivative
        rhs_contribution *= mask_narrowband

        return rhs_contribution

    def get_reinitialization_flag(
            self,
            simulation_step: int
            ) -> bool:
        """Decides whether to perform 
        reinitialization for the current
        simulation step based on the 
        the reinitialization interval.

        :param simulation_step: _description_
        :type simulation_step: jnp.int32
        :return: _description_
        :rtype: bool
        """
        reinitialization_setup = self.levelset_setup.reinitialization_runtime
        interval_reinitialization = reinitialization_setup.interval
        if simulation_step % interval_reinitialization == 0:
            perform_reinitialization = True
        else:
            perform_reinitialization = False
        return perform_reinitialization

    def treat_integrated_levelset(
            self,
            levelset: Array,
            perform_reinitialization: bool,
            is_last_stage: bool
            ) -> Tuple[Array, Array, Array,
                       LevelsetProcedureInformation]:
        """Treats the integrated levelset field.
        1) Reinitialization (only for last stage)
        2) Halo update and interface recontruction
        3) Remove small structures (only for last stage)

        :param levelset: _description_
        :type levelset: Array
        :param perform_reinitialization: _description_
        :type perform_reinitialization: bool
        :param is_last_stage: _description_
        :type is_last_stage: bool
        :return: _description_
        :rtype: Tuple[Array]
        """
        
        reinitialization_setup = self.levelset_setup.reinitialization_runtime
        remove_underresolved = reinitialization_setup.remove_underresolved
        perform_cutoff = self.levelset_setup.narrowband.perform_cutoff
        
        CFL = reinitialization_setup.CFL
        steps = reinitialization_setup.steps

        if perform_reinitialization and steps > 0 and is_last_stage:
            levelset = self.halo_manager.perform_halo_update_levelset(
                levelset, False, False)
            mask_reinitialize = self.reinitializer.compute_reinitialization_mask(levelset)
            levelset, info = self.reinitializer.perform_reinitialization(
                levelset, CFL, steps, mask_reinitialize)
            if perform_cutoff:
                levelset = self.reinitializer.set_levelset_cutoff(levelset)
        else:
            info = None

        levelset = self.halo_manager.perform_halo_update_levelset(
            levelset, True, True)
            
        volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(
            levelset)

        if remove_underresolved and is_last_stage:
            levelset = self.reinitializer.remove_underresolved_structures(
                levelset, volume_fraction)
            levelset = self.halo_manager.perform_halo_update_levelset(
                levelset, True, True)
            volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(
                levelset)
            
        return levelset, volume_fraction, apertures, info

    def mixing_material_fields(
            self,
            conservatives: Array,
            primitives: Array,
            levelset: Array,
            volume_fraction_new: Array,
            volume_fraction_old: Array,
            physical_simulation_time: Array,
            solid_cell_indices: LevelsetSolidCellIndices = None,
            ml_setup: MachineLearningSetup = None
        ) -> Tuple[Array, Array, Array, int, Array]:
        """ 

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction_new: _description_
        :type volume_fraction_new: Array
        :param volume_fraction_old: _description_
        :type volume_fraction_old: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: Array
        :param solid_cell_indices: _description_, defaults to None
        :type solid_cell_indices: LevelsetSolidCellIndices, optional
        :return: _description_
        :rtype: Tuple[Array, Array, Array, int]
        """

        # NOTE halo update for mixing
        conservatives = self.halo_manager.perform_halo_update_conservatives_mixing(
            conservatives)
        normal = self.geometry_calculator.compute_normal(levelset)
        conservatives, invalid_cells, mixing_invalid_cell_count = self.conservatives_mixer.perform_mixing(
            conservatives, levelset, normal, volume_fraction_new,
            volume_fraction_old, solid_cell_indices=solid_cell_indices)

        # NOTE primitives in real fluid
        primitives = self.compute_primitives_from_conservatives_in_real_fluid(
            conservatives, primitives, volume_fraction_new)
        
        # NOTE halo update for extension
        # interpolation based extension requires all halo regions to be updated
        extension_setup = self.levelset_setup.extension.primitives
        is_use_interpolation = extension_setup.method == "INTERPOLATION"
        primitives = self.halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, is_use_interpolation,
            is_use_interpolation, ml_setup=ml_setup)

        return (
            primitives,
            conservatives,
            invalid_cells,
            mixing_invalid_cell_count,
        )

    def extension_material_fields(
            self,
            conservatives: Array,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            physical_simulation_time: Array,
            invalid_cells: Array,
            solid_temperature: Array,
            solid_velocity: Array,
            interface_heat_flux: Array,
            interface_temperature: Array,
            cell_indices: LevelsetSolidCellIndicesField,
            ml_setup: MachineLearningSetup = None
        ) -> Tuple[Array, Array, LevelsetPositivityInformation,
                   LevelsetProcedureInformation]:

        if self.fluid_solid_handler is not None:
            solid_properties_manager = self.fluid_solid_handler.solid_properties_manager
        else:
            solid_properties_manager = None

        extension_setup = self.levelset_setup.extension.primitives
        narrowband_setup = self.levelset_setup.narrowband

        normal = self.geometry_calculator.compute_normal(levelset)
        (
            conservatives,
            primitives,
            extension_invalid_cell_count,
            info_prime_extension
        ) = ghost_cell_extension_material_fields(
            conservatives, primitives, levelset, volume_fraction,
            normal, solid_temperature, solid_velocity,
            interface_heat_flux, interface_temperature,
            invalid_cells, physical_simulation_time,
            extension_setup, narrowband_setup, cell_indices,
            self.extender_primes, self.equation_manager, solid_properties_manager,
            ml_setup=ml_setup
        )
        
        return conservatives, primitives, extension_invalid_cell_count, info_prime_extension

    def compute_primitives_from_conservatives_in_real_fluid(
            self,
            conservatives: Array,
            primitives: Array,
            volume_fraction: Array,
            ) -> Tuple[Array]:
        """Computes the primitive variables from the
        integrated mixed conservatives within the real fluid.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param invalid_cells: _description_
        :type invalid_cells: Array
        :return: _description_
        :rtype: Array
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry

        levelset_model = self.levelset_setup.model

        mask_real = compute_fluid_masks(volume_fraction, levelset_model)
        mask_real = mask_real[...,nhx_,nhy_,nhz_]
        mask_ghost = 1 - mask_real

        primes_in_real = self.equation_manager.get_primitives_from_conservatives(
            conservatives[...,nhx,nhy,nhz],
            fluid_mask=mask_real)
        primes_in_real *= mask_real 

        primitives = primitives.at[...,nhx,nhy,nhz].mul(mask_ghost)
        primitives = primitives.at[...,nhx,nhy,nhz].add(primes_in_real)

        return primitives