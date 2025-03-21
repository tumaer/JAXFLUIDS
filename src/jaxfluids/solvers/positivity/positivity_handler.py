from typing import Dict, Union, Tuple, List

import jax
import jax.numpy as jnp

from jaxfluids.data_types.information import PositivityStateInformation, LevelsetPositivityInformation, PositivityCounter, DiscretizationCounter
from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_information import EquationInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.equation_manager import EquationManager
from jaxfluids.data_types.numerical_setup import NumericalSetup

from jaxfluids.solvers.positivity.limiter_flux import PositivityLimiterFlux
from jaxfluids.solvers.positivity.limiter_interpolation import PositivityLimiterInterpolation
from jaxfluids.solvers.positivity.limiter_state import PositivityLimiterState

Array = jax.Array

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
            diffuse_interface_handler: DiffuseInterfaceHandler = None,
            ) -> None:
        
        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager

        self.is_parallel = domain_information.is_parallel
        self.domain_slices_conservatives = domain_information.domain_slices_conservatives
        self.domain_slices_geometry = domain_information.domain_slices_geometry

        self.ids_mass = self.equation_information.ids_mass
        self.s_mass = self.equation_information.s_mass
        self.vel_ids = self.equation_information.ids_velocity
        self.vel_slices = self.equation_information.s_velocity
        self.ids_energy = self.equation_information.ids_energy
        self.s_energy = self.equation_information.s_energy
        self.ids_volume_fraction = self.equation_information.ids_volume_fraction
        self.s_volume_fraction = self.equation_information.s_volume_fraction
            
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
            numerical_setup=numerical_setup)
        self.positivity_limiter_state = PositivityLimiterState(
            domain_information=domain_information,
            material_manager=material_manager,
            equation_manager=equation_manager,)
        
    def limit_interpolation_xi(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: Tuple[Array],
            axis: int,
            apertures: Tuple[Array] = None
            ) -> Tuple[Array, Array, int]:
        return self.positivity_limiter_interpolation.limit_interpolation_xi(
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
            cell_sizes: Tuple[Array],
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
            temperature: Array,
            levelset: Array,
            volume_fraction: Array, 
            apertures: Array, 
            curvature: Array, 
            physical_timestep_size: float, 
            axis: int, 
            ml_setup: MachineLearningSetup = None
            ) -> Tuple[Array, Array, Array, int]:
        return self.positivity_limiter_flux.compute_positivity_preserving_flux(
            flux_xi_convective,
            u_hat_xi,
            alpha_hat_xi,
            primitives,
            conservatives,
            temperature,
            levelset,
            volume_fraction,
            apertures,
            curvature,
            physical_timestep_size,
            axis,
            ml_setup,
        )
    
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
        primitives: Array,
        temperature: Array,
        positivity_counter: PositivityCounter,
        discretization_counter: DiscretizationCounter,
        volume_fraction: Array,
        levelset_positivity_fluid_info: LevelsetPositivityInformation,
        levelset_positivity_solid_info: LevelsetPositivityInformation,
        material_manager: MaterialManager,
        equation_information: EquationInformation,
        domain_information: DomainInformation,
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

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry

    levelset_model = equation_information.levelset_model
    ids_mass = equation_information.ids_mass
    ids_energy = equation_information.ids_energy
    s_mass = equation_information.s_mass
    s_volume_fraction = equation_information.s_volume_fraction

    primitives = primitives[...,nhx,nhy,nhz]
    if temperature is not None:
        temperature = temperature[...,nhx,nhy,nhz]

    min_temperature = None
    min_alpharho, min_alpha, max_alpha = None, None, None

    equation_type = equation_information.equation_type

    if equation_information.is_solid_levelset:
        mask_real = compute_fluid_masks(volume_fraction, levelset_model)
        mask_real = mask_real[...,nhx_,nhy_,nhz_]
        initial_min = 1e+10
        initial_max = -1.0
    else:
        initial_min = initial_max = None
        mask_real = None

    if equation_type == "TWO-PHASE-LS":
        volume_fraction = volume_fraction[...,nhx_,nhy_,nhz_]
        primes_real = primitives[:,0] * volume_fraction + \
            primitives[:,1] * (1.0 - volume_fraction)
        min_density  = jnp.min(primes_real[ids_mass])
        min_pressure = jnp.min(primes_real[ids_energy])

    elif equation_type == "DIFFUSE-INTERFACE-5EQM":
        min_alpharho = jnp.min(primitives[s_mass], initial=initial_min, where=mask_real)
        min_alpha = jnp.min(primitives[s_volume_fraction], initial=initial_min, where=mask_real) 
        max_alpha = jnp.max(primitives[s_volume_fraction], initial=initial_max, where=mask_real)
        rho = material_manager.get_density(primitives)
        pb = material_manager.get_background_pressure(primitives[s_volume_fraction])
        min_pressure = jnp.min(primitives[ids_energy] + pb, initial=initial_min, where=mask_real)
        min_density = jnp.min(rho, initial=initial_min, where=mask_real)
    
    elif equation_type == "DIFFUSE-INTERFACE-4EQM":
        min_alpharho = jnp.min(primitives[s_mass], initial=initial_min, where=mask_real)
        rho = material_manager.get_density(primitives)
        pb = material_manager.get_background_pressure(primitives[s_volume_fraction])
        min_pressure = jnp.min(primitives[ids_energy] + pb, initial=initial_min, where=mask_real)
        min_density = jnp.min(rho, initial=initial_min, where=mask_real)

    elif equation_type == "SINGLE-PHASE":
        min_density = jnp.min(primitives[ids_mass], initial=initial_min, where=mask_real)
        min_pressure = jnp.min(primitives[ids_energy], initial=initial_min, where=mask_real)
    
    else:
        raise NotImplementedError
    
    if domain_information.is_parallel:
        min_pressure = jax.lax.pmin(min_pressure, axis_name="i")
        min_density = jax.lax.pmin(min_density, axis_name="i")
        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            min_alpharho = jax.lax.pmin(min_alpharho, axis_name="i")
            min_alpha = jax.lax.pmin(min_alpha, axis_name="i")
            max_alpha = jax.lax.pmax(max_alpha, axis_name="i")
        elif equation_type == "DIFFUSE-INTERFACE-4EQM":
            min_alpharho = jax.lax.pmin(min_alpharho, axis_name="i")

    positivity_state = PositivityStateInformation(
        min_pressure, min_density, min_temperature,
        min_alpharho, min_alpha,
        max_alpha, positivity_counter,
        discretization_counter,
        levelset_positivity_fluid_info,
        levelset_positivity_solid_info,
        )
            
    return positivity_state
