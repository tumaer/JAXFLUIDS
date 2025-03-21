import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup 
from jaxfluids.data_types.information import LevelsetResidualInformation, LevelsetProcedureInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.levelset.geometry.mask_functions import (compute_fluid_masks, compute_cut_cell_mask_sign_change_based,
                                                    compute_cut_cell_mask_value_based,
                                                    compute_narrowband_mask, compute_cut_cell_mask_vf_based)
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.config import precision

Array = jax.Array

class ResidualComputer:

    def __init__(
            self,
            domain_information: DomainInformation,
            geometry_calculator: GeometryCalculator,
            levelset_reinitializer: PDEBasedReinitializer,
            levelset_setup: LevelsetSetup,
            extender_interface: IterativeExtender = None,
            extender_primes: IterativeExtender = None,
            extender_solids: IterativeExtender = None,
            ) -> None:

        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.levelset_reinitializer = levelset_reinitializer
        self.extender_interface = extender_interface
        self.extender_primes = extender_primes
        self.extender_solids = extender_solids
        self.levelset_setup = levelset_setup
        self.geometry_calculator = geometry_calculator

    def compute_residuals(
            self,
            levelset: Array,
            primitives: Array = None,
            volume_fraction: Array = None,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            solid_temperature: Array = None
            ) -> LevelsetResidualInformation:
        """Computes the residuals 
        of the reinitialization and 
        extension procedure.

        :param levelset: _description_
        :type levelset: Array
        :param primitives: _description_
        :type primitives: Array
        :param conservatives: _description_
        :type conservatives: Array
        :return: _description_
        :rtype: Tuple[float]
        """

        nh_conservatives = self.domain_information.nh_conservatives
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nhx__,nhy__,nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        smallest_cell_size = self.domain_information.smallest_cell_size
        narrowband_computation = self.levelset_setup.narrowband.computation_width
        levelset_model = self.levelset_setup.model

        mask_narrowband = compute_narrowband_mask(levelset, smallest_cell_size, narrowband_computation)
        mask_narrowband = mask_narrowband[nhx,nhy,nhz]
        normal = self.geometry_calculator.compute_normal(levelset)

        # LEVELSET REINITIALIZATION
        reinit_max = self.levelset_reinitializer.compute_residual(levelset)
        reinitialization_info = LevelsetProcedureInformation(
            0, reinit_max, None)

        # PRIMITIVE EXTENSION
        if primitives is not None:
            if levelset_model == "FLUID-FLUID":
                normal_extend = jnp.stack([-normal, normal], axis=1)
            else:
                normal_extend = -normal
            mask_real = compute_fluid_masks(volume_fraction, levelset_model)
            mask_real = mask_real[...,nhx_,nhy_,nhz_]
            mask_ghost = 1 - mask_real
            mask_extend = mask_ghost * mask_narrowband
            prime_extension_mean, prime_extension_max, _ = \
                self.extender_primes.compute_residual(primitives, normal_extend, mask_extend)
            primitives_extension_info = LevelsetProcedureInformation(
                0, prime_extension_max, prime_extension_mean)
        else:
            primitives_extension_info = None

        # INTERFACE EXTENSION
        if interface_pressure is not None and interface_velocity is not None:
            interface_quantities = jnp.concatenate([jnp.expand_dims(interface_velocity, axis=0),
                                                    interface_pressure], axis=0)
            cut_cell_mask = compute_cut_cell_mask_value_based(levelset, smallest_cell_size)
            cut_cell_mask = cut_cell_mask[nhx_,nhy_,nhz_]
            inverse_cut_cell_mask = 1 - cut_cell_mask
            mask_extend = inverse_cut_cell_mask * mask_narrowband
            normal_extend = normal * jnp.sign(levelset[nhx__,nhy__,nhz__])
            interface_extension_mean, interface_extension_max, _ = \
                self.extender_interface.compute_residual(interface_quantities, normal_extend, mask_extend)
            interface_extension_info = LevelsetProcedureInformation(
                0, interface_extension_max, interface_extension_mean)
        else:
            interface_extension_info = None

        # SOLID TEMPERATURE EXTENSION
        if solid_temperature is not None:
            normal_extend = normal
            volume_fraction_solid = 1.0 - volume_fraction
            mask_real_solid = volume_fraction_solid > 0.0
            mask_ghost_solid = 1 - mask_real_solid
            mask_ghost_solid = mask_ghost_solid[nhx_,nhy_,nhz_]
            mask_extend = mask_ghost_solid * mask_narrowband
            solid_temperature_extension_mean, solid_temperature_extension_max, _ = \
                self.extender_solids.compute_residual(solid_temperature, normal_extend, mask_extend)
            solids_extension_info = LevelsetProcedureInformation(
                0, solid_temperature_extension_max, solid_temperature_extension_mean)
        else:
            solids_extension_info = None

        levelset_residuals_info = LevelsetResidualInformation(
            reinitialization_info,
            primitives_extension_info,
            interface_extension_info,
            solids_extension_info
            )

        return levelset_residuals_info