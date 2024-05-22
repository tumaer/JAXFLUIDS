from typing import Dict, Union, Tuple, List

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.information import PositivityStateInformation, LevelsetPositivityInformation
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.data_types.numerical_setup import NumericalSetup

from jaxfluids.solvers.positivity.limiter_flux import PositivityLimiterFlux
from jaxfluids.solvers.positivity.limiter_interpolation import PositivityLimiterInterpolation
from jaxfluids.solvers.positivity.limiter_state import PositivityLimiterState

class PositivityHandler:
    """The PositivityHandler implements functionality to
    keep the state of the conservative/primitive variables
    in a physically admissible set.
    
    Submodules are:
    1) PositivityLimiterInterpolation
    2) PositivityLimiterFlux
    3) PositivityLimiterState
    
    :return: _description_
    :rtype: _type_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            halo_manager: HaloManager,
            numerical_setup: NumericalSetup,
            levelset_handler: LevelsetHandler = None,
            diffuse_interface_handler: DiffuseInterfaceHandler = None
            ) -> None:
        
        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager

        self.is_parallel = domain_information.is_parallel
        self.domain_slices_conservatives = domain_information.domain_slices_conservatives
        self.domain_slices_geometry = domain_information.domain_slices_geometry

        is_logging = numerical_setup.conservatives.positivity.is_logging

        self.mass_ids = self.equation_information.mass_ids
        self.mass_slices = self.equation_information.mass_slices
        self.vel_ids = self.equation_information.velocity_ids
        self.vel_slices = self.equation_information.velocity_slices
        self.energy_ids = self.equation_information.energy_ids
        self.energy_slices = self.equation_information.energy_slices
        self.vf_ids = self.equation_information.vf_ids
        self.vf_slices = self.equation_information.vf_slices
            
        self.positivity_limiter_flux = PositivityLimiterFlux(
            domain_information=domain_information,
            material_manager=material_manager,
            equation_manager=equation_manager,
            numerical_setup=numerical_setup,
            diffuse_interface_handler=diffuse_interface_handler,)
        self.positivity_limiter_interpolation = PositivityLimiterInterpolation(
            domain_information=domain_information,
            material_manager=material_manager,
            equation_manager=equation_manager,
            is_logging=is_logging)
        self.positivity_limiter_state = PositivityLimiterState(
            domain_information=domain_information,
            material_manager=material_manager,
            equation_manager=equation_manager,)

        self.is_logging = numerical_setup.conservatives.positivity.is_logging
        
    def compute_positivity_preserving_interpolation(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: List,
            axis: int,
            apertures: Tuple[Array] = None
            ) -> Tuple[Array, Array, int]:
        return self.positivity_limiter_interpolation.compute_positivity_preserving_interpolation_xi(
            primitives,
            primitives_xi_j,
            j,
            cell_sizes,
            axis,
            apertures=apertures)
    
    def compute_positivity_preserving_thinc_interpolation(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: List,
            axis: int,
            ) -> Tuple[Array, Array, int]:
        return self.positivity_limiter_interpolation.compute_positivity_preserving_thinc_interpolation_xi(
            primitives,
            primitives_xi_j,
            j,
            cell_sizes,
            axis)
    
    def compute_positivity_preserving_flux(
            self,
            flux_xi_convective: Array, 
            u_hat_xi: Array,
            alpha_hat_xi: Array,
            primitives: Array,
            conservatives: Array, 
            levelset: Array,
            volume_fraction: Array, 
            apertures: Array, 
            curvature: Array, 
            physical_timestep_size: float, 
            axis: int, 
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None
            ) -> Tuple[Array, Array, Array, int]:
        return self.positivity_limiter_flux.compute_positivity_preserving_flux(
            flux_xi_convective,
            u_hat_xi,
            alpha_hat_xi,
            primitives,
            conservatives,
            levelset,
            volume_fraction,
            apertures,
            curvature,
            physical_timestep_size,
            axis,
            ml_parameters_dict,
            ml_networks_dict)
    
    def compute_positivity_preserving_sharpening_flux(
            self,
            flux_xi_convective: Array,
            u_hat_xi: Array,
            interface_regularization_flux_xi: Array,
            primitives: Array,
            conservatives: Array,
            physical_timestep_size: float,
            axis: int
            ) -> Tuple[Array, int]:
        return self.positivity_limiter_flux.compute_positivity_preserving_sharpening_flux(
            flux_xi_convective,
            u_hat_xi,
            interface_regularization_flux_xi,
            primitives,
            conservatives,
            physical_timestep_size,
            axis)
    
    def correct_volume_fraction(self, conservatives: Array) -> Array:
        return self.positivity_limiter_state.correct_volume_fraction(conservatives)
   
    def get_positvity_state_info(
            self,
            primitives: Array,
            positivity_count_flux: jnp.int32,
            positivity_count_interpolation: jnp.int32 = None,
            vf_correction_count: jnp.int32 = None,
            positivity_count_thinc: jnp.int32 = None,
            positivity_count_acdi: jnp.int32 = None,
            count_acdi: jnp.int32 = None,
            volume_fraction: Array = None,
            levelset_positivity_info: LevelsetPositivityInformation = None
            ) -> PositivityStateInformation:
        """Computes minimum and maximum values
        for critical state variables.
        I.e., pressure, density, and volume-fraction.

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_, defaults to None
        :type levelset: Array, optional
        :param volume_fraction: _description_, defaults to None
        :type volume_fraction: Array, optional
        :return: _description_
        :rtype: PrimitiveStateInformation
        """

        primitives = primitives[(...,) + self.domain_slices_conservatives]
        min_alpharho, min_alpha, max_alpha = None, None, None

        if self.equation_information.levelset_model == "FLUID-FLUID":
            volume_fraction = volume_fraction[(...,) + self.domain_slices_geometry]
            primes_real = primitives[:,0] * volume_fraction + \
                primitives[:,1] * (1.0 - volume_fraction)
            min_density  = jnp.min(primes_real[self.mass_ids])
            min_pressure = jnp.min(primes_real[self.energy_ids])
            
        elif self.equation_information.levelset_model in \
            ["FLUID-SOLID-STATIC", "FLUID-SOLID-DYNAMIC", "FLUID-SOLID-DYNAMIC-COUPLED"]:
            mask_real = compute_fluid_masks(volume_fraction, self.equation_information.levelset_model)
            mask_real = mask_real[(...,) + self.domain_slices_geometry]
            primitives = primitives * mask_real + (1 - mask_real) * 1e+10   # TODO what is this???
            min_density = jnp.min(primitives[self.mass_ids])
            min_pressure = jnp.min(primitives[self.energy_ids])

        elif self.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM":
            min_alpharho = jnp.min(primitives[self.mass_slices])
            min_alpha = jnp.min(primitives[self.vf_slices]) 
            max_alpha = jnp.max(primitives[self.vf_slices])
            rho = self.material_manager.get_density(primitives)
            pb = self.material_manager.get_background_pressure(primitives[self.vf_slices])
            min_pressure = jnp.min(primitives[self.energy_ids] + pb)
            min_density = jnp.min(rho)
        
        elif self.equation_information.equation_type == "DIFFUSE-INTERFACE-4EQM":
            min_alpharho = jnp.min(primitives[self.mass_slices])
            rho = self.material_manager.get_density(primitives)
            pb = self.material_manager.get_background_pressure(primitives[self.vf_slices])
            min_pressure = jnp.min(primitives[self.energy_ids] + pb)
            min_density = jnp.min(rho)

        elif self.equation_information.equation_type == "SINGLE-PHASE":
            min_density = jnp.min(primitives[self.mass_ids])
            min_pressure = jnp.min(primitives[self.energy_ids])
        
        elif self.equation_information.equation_type == "MULTI-COMPONENT":
            rho = self.material_manager.get_density(primitives)
            Y_i = primitives[self.mass_slices] / rho
            min_alpharho = jnp.min(primitives[self.mass_slices])
            min_alpha = jnp.min(Y_i)
            max_alpha = jnp.max(Y_i)
            min_density = jnp.min(rho)
            min_pressure = jnp.min(primitives[self.energy_ids])            
        
        else:
            raise NotImplementedError
        
        if self.is_parallel:
            min_pressure = jax.lax.pmin(min_pressure, axis_name="i")
            min_density = jax.lax.pmin(min_density, axis_name="i")
            if self.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM":
                min_alpharho = jax.lax.pmin(min_alpharho, axis_name="i")
                min_alpha = jax.lax.pmin(min_alpha, axis_name="i")
                max_alpha = jax.lax.pmax(max_alpha, axis_name="i")

        positivity_state = PositivityStateInformation(
            min_pressure, min_density,
            min_alpharho, min_alpha,
            max_alpha, positivity_count_flux,
            positivity_count_interpolation,
            positivity_count_thinc,
            positivity_count_acdi,
            vf_correction_count,
            count_acdi, None,
            levelset_positivity_info)
                
        return positivity_state
