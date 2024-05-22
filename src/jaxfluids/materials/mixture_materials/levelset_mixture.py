from abc import ABC, abstractmethod
from typing import Dict, List, Union
import types

import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.materials import DICT_MATERIAL
from jaxfluids.materials.mixture_materials.mixture import Mixture
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.materials.single_materials.material import Material
from jaxfluids.data_types.case_setup.material_properties import LevelsetMixtureSetup, MaterialPropertiesSetup

class LevelsetMixture(Mixture):

    def __init__(
            self,
            unit_handler: UnitHandler,
            levelset_mixture_setup: LevelsetMixtureSetup
        ) -> None:

        super().__init__(unit_handler, levelset_mixture_setup)

        for fluid in ["positive", "negative"]:
            material_setup: MaterialPropertiesSetup = getattr(levelset_mixture_setup, fluid + "_fluid")
            material_type = material_setup.eos.model
            self.materials[fluid] = DICT_MATERIAL[material_type](unit_handler, material_setup)

        self.gamma = jnp.stack([self.materials["positive"].gamma, self.materials["negative"].gamma]).reshape(2,1,1,1)
        self.R = jnp.stack([self.materials["positive"].R, self.materials["negative"].R]).reshape(2,1,1,1)
        self.pb = jnp.stack([self.materials["positive"].pb, self.materials["negative"].pb]).reshape(2,1,1,1)

        pairing_properties = levelset_mixture_setup.pairing_properties
        self.sigma = pairing_properties.surface_tension_coefficient

    def get_thermal_conductivity(self, T: Array) -> Array:
        thermal_conductivity_1 = self.materials["positive"].get_thermal_conductivity(T[0])
        thermal_conductivity_2 = self.materials["negative"].get_thermal_conductivity(T[1])
        thermal_conductivity = jnp.stack(
            [thermal_conductivity_1 * jnp.ones_like(thermal_conductivity_2),
            thermal_conductivity_2 * jnp.ones_like(thermal_conductivity_1)],
            axis=0)
        if thermal_conductivity.shape == (2,):
            thermal_conductivity = thermal_conductivity.reshape(2,1,1,1)
        return thermal_conductivity

    def get_dynamic_viscosity(self, T: Array) -> Array:
        dynamic_viscosity_1 = self.materials["positive"].get_dynamic_viscosity(T[0])
        dynamic_viscosity_2 = self.materials["negative"].get_dynamic_viscosity(T[1])
        dynamic_viscosity = jnp.stack(
            [dynamic_viscosity_1 * jnp.ones_like(dynamic_viscosity_2),
            dynamic_viscosity_2 * jnp.ones_like(dynamic_viscosity_1)],
            axis=0)
        if dynamic_viscosity.shape == (2,):
            dynamic_viscosity = dynamic_viscosity.reshape(2,1,1,1)
        return dynamic_viscosity

    def get_bulk_viscosity(self, temperature: Array) -> Array:
        bulk_viscosity = jnp.stack([   
            self.materials["positive"].get_bulk_viscosity(temperature),
            self.materials["negative"].get_bulk_viscosity(temperature)  ], axis=0).reshape(2,1,1,1)
        return bulk_viscosity
    
    def get_specific_heat_capacity(self, temperature: Array) -> Array:
        specific_heat_capacity_1 = self.materials["positive"].get_dynamic_viscosity(temperature[0])
        specific_heat_capacity_2 = self.materials["negative"].get_dynamic_viscosity(temperature[1])
        specific_heat_capacity = jnp.stack(
            [specific_heat_capacity_1 * jnp.ones_like(specific_heat_capacity_2),
            specific_heat_capacity_2 * jnp.ones_like(specific_heat_capacity_1)],
            axis=0)
        if specific_heat_capacity.shape == (2,):
            specific_heat_capacity = specific_heat_capacity.reshape(2,1,1,1)
        return specific_heat_capacity

    def get_speed_of_sound(self,
        p: Array,
        rho: Array
        ) -> Array:
        speed_of_sound = []
        for i, fluid in enumerate(self.materials):
            speed_of_sound.append( self.materials[fluid].get_speed_of_sound(p[i], rho[i]) )
        speed_of_sound = jnp.stack(speed_of_sound, axis=0)
        return speed_of_sound

    def get_pressure(self, e: Array, rho: Array) -> Array:
        pressure = []
        for i, fluid in enumerate(self.materials):
            pressure.append( self.materials[fluid].get_pressure(e[i], rho[i]) )
        pressure = jnp.stack(pressure, axis=0)
        return pressure

    def get_temperature(self,
        p: Array,
        rho: Array) -> Array:
        temperature = []
        for i, fluid in enumerate(self.materials):
            temperature.append( self.materials[fluid].get_temperature(p[i], rho[i]) )
        temperature = jnp.stack(temperature, axis=0)
        return temperature
    
    def get_specific_energy(self,
        p: Array,
        rho: Array
        ) -> Array:
        # Specific internal energy
        energy = []
        for i, fluid in enumerate(self.materials):
            energy.append( self.materials[fluid].get_specific_energy(p[i], rho[i]) )
        energy = jnp.stack(energy, axis=0)
        return energy

    def get_total_energy(self,
        p: Array, u: Array,
        v: Array, w: Array,
        rho: Array
        ) -> Array:
        # Total energy per unit volume
        total_energy = []
        for i, fluid in enumerate(self.materials):
            total_energy.append( self.materials[fluid].get_total_energy(p[i], rho[i], u[i], v[i], w[i]) )
        total_energy = jnp.stack(total_energy, axis=0)
        return total_energy

    def get_total_enthalpy(self,
        p:Array, u:Array,
        v:Array, w:Array,
        rho: Array
        ) -> Array:
        # Total specific enthalpy
        total_enthalpy = []
        for i, fluid in enumerate(self.materials):
            total_enthalpy.append( self.materials[fluid].get_total_enthalpy(p[i], rho[i], u[i], v[i], w[i]) )
        total_enthalpy = jnp.stack(total_enthalpy, axis=0)
        return total_enthalpy

    def get_psi(self,
        p: Array,
        rho: Array
        ) -> Array:
        psi = []
        for i, fluid in enumerate(self.materials):
            psi.append( self.materials[fluid].get_psi(p[i], rho[i]) )
        psi = jnp.stack(psi, axis=0)
        return psi

    def get_grueneisen(self,
        rho: Array
        ) -> Array:
        grueneisen = []
        for i, fluid in enumerate(self.materials):
            grueneisen.append( self.materials[fluid].get_grueneisen(rho[i]) )
        grueneisen = jnp.stack(grueneisen, axis=0)
        if grueneisen.shape == (2,):
            grueneisen = grueneisen.reshape(2,1,1,1)
        return grueneisen

    def get_gamma(self) -> Array:
        return self.gamma

    def get_sigma(self) -> Array:
        return self.sigma

    def get_R(self) -> Array:
        return self.R

    def get_background_pressure(self) -> Array:
        return self.pb
