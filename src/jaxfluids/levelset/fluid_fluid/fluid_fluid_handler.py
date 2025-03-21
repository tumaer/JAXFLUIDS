from typing import Dict, Union, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.data_types.numerical_setup import ActivePhysicsSetup, LevelsetSetup
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.levelset.fluid_fluid.interface_flux_contributions import (
    convective_interface_flux_xi, heat_interface_flux_xi, viscous_interface_flux_xi,
    heat_interface_flux, viscous_interface_flux)
from jaxfluids.levelset.fluid_fluid.interface_quantities import compute_interface_quantities
from jaxfluids.config import precision
from jaxfluids.data_types.information import LevelsetProcedureInformation

Array = jax.Array

class FluidFluidLevelsetHandler:

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            geometry_calculator: GeometryCalculator,
            extender_interface: IterativeExtender, 
            levelset_setup: LevelsetSetup,
            ) -> None:

        self.eps = precision.get_eps()

        self.extender_interface = extender_interface
        self.material_manager = material_manager
        self.domain_information = domain_information
        self.equation_information = material_manager.equation_information

        self.levelset_setup = levelset_setup
        self.geometry_calculator = geometry_calculator

        active_physics_setup = self.equation_information.active_physics
        is_viscous_flux = active_physics_setup.is_viscous_flux
        is_heat_flux = active_physics_setup.is_heat_flux
        interface_flux_method = levelset_setup.interface_flux.method
        if is_viscous_flux or is_heat_flux and interface_flux_method == "CELLCENTER":
            derivative_stencil = levelset_setup.interface_flux.derivative_stencil
            self.derivative_stencil : SpatialDerivative = derivative_stencil(
                        nh = domain_information.nh_geometry,
                        inactive_axes = domain_information.inactive_axes)

        self.aperture_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

    # these are fluxes that are computed in axis-by-axis manner
    def compute_interface_flux_xi(
            self,
            primitives: Array,
            levelset: Array,
            interface_velocity: Array,
            interface_pressure: Array,
            volume_fraction: Array,
            apertures: Tuple[Array],
            axis_index: int,
            temperature: Array
            ) -> Array:

        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives

        active_physics_setup = self.equation_information.active_physics
        is_convective_flux = active_physics_setup.is_convective_flux
        is_viscous_flux = active_physics_setup.is_viscous_flux
        is_heat_flux = active_physics_setup.is_heat_flux

        interface_flux_setup = self.levelset_setup.interface_flux
        method = interface_flux_setup.method

        interface_velocity = interface_velocity[...,nhx_,nhy_,nhz_]
        interface_pressure = interface_pressure[...,nhx_,nhy_,nhz_]

        apertures = jnp.stack([apertures, 1.0 - apertures], axis=0) 
        volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
        s1 = self.aperture_slices[axis_index][0]
        s2 = self.aperture_slices[axis_index][1]
        delta_aperture = apertures[s1][...,nhx_,nhy_,nhz_] - \
                            apertures[s2][...,nhx_,nhy_,nhz_]
        normal = self.geometry_calculator.compute_normal(levelset)
        # normal_aperture_based = self.geometry_calculator.compute_normal_apertures_based(apertures)
        interface_flux_xi = jnp.zeros(primitives[...,nhx,nhy,nhz].shape)

        # NOTE convective flux
        if is_convective_flux:
            momentum_flux_xi, energy_flux_xi = convective_interface_flux_xi(
                interface_pressure, interface_velocity, delta_aperture,
                normal, axis_index, self.domain_information)
            interface_flux_xi = interface_flux_xi.at[axis_index+1].add(momentum_flux_xi)
            interface_flux_xi = interface_flux_xi.at[4].add(energy_flux_xi)

        # NOTE viscous flux
        if is_viscous_flux and method == "CELLCENTER":
            momentum_flux_xi, energy_flux_xi = viscous_interface_flux_xi(
                primitives, volume_fraction, temperature, delta_aperture,
                interface_velocity, normal, axis_index, self.material_manager,
                self.derivative_stencil, self.domain_information)
            interface_flux_xi = interface_flux_xi.at[1:4].add(momentum_flux_xi)
            interface_flux_xi = interface_flux_xi.at[4].add(energy_flux_xi)

        # NOTE heat flux
        if is_heat_flux and method == "CELLCENTER":
            energy_flux_xi = heat_interface_flux_xi(
                temperature, volume_fraction, delta_aperture,
                axis_index, self.material_manager, self.derivative_stencil,
                self.domain_information)
            interface_flux_xi = interface_flux_xi.at[4].add(energy_flux_xi)

        return interface_flux_xi

    # these are fluxes that are computed at once for all axis directions  
    def compute_interface_flux(
            self,
            primitives: Array,
            levelset: Array,
            interface_velocity: Array,
            apertures: Tuple[Array],
            temperature: Array
            ) -> Array:
        
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        
        active_physics_setup = self.equation_information.active_physics
        is_convective_flux = active_physics_setup.is_convective_flux
        is_viscous_flux = active_physics_setup.is_viscous_flux
        is_heat_flux = active_physics_setup.is_heat_flux
        interpolation_dh = self.levelset_setup.interface_flux.interpolation_dh

        interface_flux_setup = self.levelset_setup.interface_flux
        method = interface_flux_setup.method
        material_properties_averaging = interface_flux_setup.material_properties_averaging

        interface_flux = jnp.zeros(primitives[...,nhx,nhy,nhz].shape)

        if method == "INTERPOLATION":
            normal = self.geometry_calculator.compute_normal(levelset)
            interface_length = self.geometry_calculator.compute_interface_length(apertures)

        # NOTE viscous flux
        if is_viscous_flux and method == "INTERPOLATION":
            momentum_flux_xi, energy_flux_xi = viscous_interface_flux(
                primitives, levelset, interface_velocity, normal,
                interface_length, interpolation_dh,
                material_properties_averaging,
                self.material_manager,
                self.domain_information)
            interface_flux = interface_flux.at[1:4].add(momentum_flux_xi)
            interface_flux = interface_flux.at[4].add(energy_flux_xi)

        # NOTE heat flux
        if is_heat_flux and method == "INTERPOLATION":
            energy_flux_xi = heat_interface_flux(
                temperature, levelset, normal,
                interface_length, interpolation_dh,
                material_properties_averaging,
                self.material_manager,
                self.domain_information)
            interface_flux = interface_flux.at[4].add(energy_flux_xi)

        return interface_flux


    def compute_interface_quantities(
            self,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            interface_velocity_old: Array,
            interface_pressure_old: Array,
            ml_setup: MachineLearningSetup = None
            ) -> Tuple[Array, Array,
                       LevelsetProcedureInformation]:
        """Computes interface velocity and pressure for
        FLUID-FLUID interface interaction and
        extends the values into the narrowband_computation.

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param normal: _description_
        :type normal: Array
        :param curvature: _description_
        :type curvature: Array
        :return: _description_
        :rtype: Tuple[Array, Array]
        """
        
        active_physics_setup = self.equation_information.active_physics
        is_convective_flux = active_physics_setup.is_convective_flux
        is_viscous_flux = active_physics_setup.is_viscous_flux
        is_surface_tension = active_physics_setup.is_surface_tension

        iterative_setup = self.levelset_setup.extension.interface.iterative
        narrowband_setup = self.levelset_setup.narrowband

        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        
        if is_convective_flux:

            normal = self.geometry_calculator.compute_normal(levelset)
            if is_surface_tension:
                curvature = self.geometry_calculator.compute_curvature(levelset)
            else:
                curvature = None

            interface_velocity, interface_pressure, info = compute_interface_quantities(
                primitives, levelset, volume_fraction, normal, curvature, 
                self.material_manager, self.extender_interface,
                iterative_setup, narrowband_setup,
                interface_velocity_old, interface_pressure_old,
                ml_setup=ml_setup)

        else:

            info = None
            interface_velocity = jnp.zeros_like(levelset[nhx__,nhy__,nhz__])
            interface_pressure = jnp.ones((2,) + levelset[nhx__,nhy__,nhz__].shape)

        return interface_velocity, interface_pressure, info
