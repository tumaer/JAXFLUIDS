import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup 
from jaxfluids.data_types.information import LevelsetResidualInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks, compute_cut_cell_mask
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer
from jaxfluids.levelset.quantity_extender import QuantityExtender
from jaxfluids.config import precision

class ResidualComputer:

    def __init__(
            self,
            domain_information: DomainInformation,
            levelset_reinitializer: PDEBasedReinitializer,
            extender_interface: QuantityExtender,
            extender_primes: QuantityExtender,
            levelset_setup: LevelsetSetup
            ) -> None:

        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.levelset_reinitializer = levelset_reinitializer
        self.extender_interface = extender_interface
        self.extender_primes = extender_primes
        self.levelset_setup = levelset_setup

    def compute_residuals(
            self,
            primitives: Array,
            volume_fraction: Array,
            levelset: Array,
            normal: Array,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            reinitialization_step_count: int = None,
            prime_extension_step_count: int = None,
            interface_extension_step_count: int = None,
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

        normalized_levelset = jnp.abs(levelset[nhx,nhy,nhz])/smallest_cell_size
        mask_narrowband = jnp.where(normalized_levelset <= narrowband_computation, 1, 0)

        # LEVELSET REINITIALIZATION
        mask = self.levelset_reinitializer.compute_reinitialization_mask(levelset)
        mask *= mask_narrowband
        reinit_mean, reinit_max, _ = self.levelset_reinitializer.compute_residual(levelset, mask)

        # PRIMITIVE EXTENSION
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
        prime_extension_mean, prime_extension_max, _ = \
            self.extender_primes.compute_residual(primitives, normal_extend, mask_extend)

        # INTERFACE EXTENSION
        if levelset_model == "FLUID-FLUID":
            interface_quantities = jnp.concatenate([jnp.expand_dims(interface_velocity, axis=0),
                                                    interface_pressure], axis=0)
            mask_narrowband = jnp.where(normalized_levelset <= narrowband_computation, 1, 0)
            # cut_cell_mask = compute_cut_cell_mask(levelset, nh_conservatives)
            cut_cell_mask = (volume_fraction[nhx_,nhy_,nhz_] > 0.0) & (volume_fraction[nhx_,nhy_,nhz_] < 1.0)
            inverse_cut_cell_mask = 1 - cut_cell_mask
            mask_extend = inverse_cut_cell_mask * mask_narrowband
            normal_extend = normal * jnp.sign(levelset[nhx__,nhy__,nhz__])
            interface_extension_mean, interface_extension_max, _ = \
                self.extender_interface.compute_residual(interface_quantities, normal_extend, mask_extend)
        else:
            interface_extension_mean = None
            interface_extension_max = None

        levelset_residuals_info = LevelsetResidualInformation(
            reinit_mean, reinit_max, reinitialization_step_count,
            prime_extension_mean, prime_extension_max, prime_extension_step_count,
            interface_extension_mean, interface_extension_max,
            interface_extension_step_count)

        return levelset_residuals_info