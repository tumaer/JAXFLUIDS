from typing import Dict, Union

import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.case_setup.material_properties import MaterialManagerSetup, \
    MaterialPropertiesSetup, DiffuseMixtureSetup, LevelsetMixtureSetup
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.mixture_materials.levelset_mixture import LevelsetMixture
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.materials.single_materials.material import Material
from jaxfluids.materials.mixture_materials.diffuse_mixture import DiffuseMixture
from jaxfluids.materials.mixture_materials.diffuse_mixture_five_equation import DiffuseFiveEquationMixture
from jaxfluids.materials import DICT_MATERIAL, DICT_MIXTURE

class MaterialManager:
    """The MaterialManager class is a wrapper class
    that holds the materials. The main purpose of this
    class is to enable the computation of material
    parameters for two-phase flows, i.e., the presence
    of two (different) materials.
    """

    # TODO: check if else conditions and check explicityly for SINGLE-PHASE

    def __init__(
            self,
            equation_information: EquationInformation,
            unit_handler: UnitHandler,
            material_manager_setup: MaterialManagerSetup
            ) -> None:

        self.equation_information = equation_information
        self.unit_handler = unit_handler
        
        self.equation_type = equation_information.equation_type
        self.levelset_model = equation_information.levelset_model
        self.equation_type = equation_information.equation_type
        self.diffuse_interface_model = equation_information.diffuse_interface_model

        self.mass_ids = equation_information.mass_ids 
        self.vel_ids = equation_information.velocity_ids 
        self.energy_ids = equation_information.energy_ids 
        self.vf_ids = equation_information.vf_ids
        self.species_ids = equation_information.species_ids
        
        self.mass_slices = equation_information.mass_slices
        self.vel_slices = equation_information.velocity_slices
        self.energy_slices = equation_information.energy_slices
        self.vf_slices = equation_information.vf_slices
        self.species_slices = equation_information.species_slices

        self.levelset_mixture = None
        self.diffuse_5eqm_mixture = None
        self.material = None

        if self.equation_type == "TWO-PHASE-LS":
            levelset_mixture_setup: LevelsetMixtureSetup = material_manager_setup.levelset_mixture
            self.levelset_mixture = LevelsetMixture(
                unit_handler, levelset_mixture_setup)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            diffuse_mixture_setup: DiffuseMixtureSetup = material_manager_setup.diffuse_mixture
            self.diffuse_5eqm_mixture = DiffuseFiveEquationMixture(
                unit_handler, diffuse_mixture_setup)

        elif self.equation_type in ("SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS",):
            material_setup: MaterialPropertiesSetup = material_manager_setup.single_material
            material_type = material_setup.eos.model
            self.material : Material = DICT_MATERIAL[material_type](unit_handler, material_setup)
        
        else:
            raise NotImplementedError

    def get_thermal_conductivity(
            self,
            temperature: Array,
            primitives: Array,
            density: Array = None,
            partial_densities: Array = None,
            volume_fractions: Array = None,
        ) -> Array:

        if self.equation_type in ("SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS"):
                thermal_conductivity = self.material.get_thermal_conductivity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            thermal_conductivity = self.levelset_mixture.get_thermal_conductivity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if volume_fractions is None:
                volume_fractions = primitives[self.vf_slices]
            thermal_conductivity = self.diffuse_5eqm_mixture.get_thermal_conductivity(
                temperature, volume_fractions)
        
        else:
            raise NotImplementedError

        return thermal_conductivity

    def get_dynamic_viscosity(
            self,
            temperature: Array,
            primitives: Array,
            density: Array = None,
            volume_fractions: Array = None,
        ) -> Array:

        if self.equation_type in ("SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS"):
                dynamic_viscosity = self.material.get_dynamic_viscosity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            dynamic_viscosity = self.levelset_mixture.get_dynamic_viscosity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if volume_fractions is None:
                volume_fractions = primitives[self.vf_slices]
            dynamic_viscosity = self.diffuse_5eqm_mixture.get_dynamic_viscosity(
                temperature, volume_fractions)
        
        else:
            raise NotImplementedError

        return dynamic_viscosity

    def get_bulk_viscosity(
            self,
            temperature: Array,
            primitives: Array,
            density: Array = None,
            volume_fractions: Array = None,
        ) -> Array:

        if self.equation_type in ("SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS"):
                bulk_viscosity = self.material.get_bulk_viscosity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            bulk_viscosity = self.levelset_mixture.get_bulk_viscosity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if volume_fractions is None:
                volume_fractions = primitives[self.vf_slices]
            bulk_viscosity = self.diffuse_5eqm_mixture.get_bulk_viscosity(
                temperature, volume_fractions)
        
        else:
            raise NotImplementedError
            
        return bulk_viscosity

    def get_speed_of_sound(
            self,
            primitives: Array = None,
            pressure: Array = None,
            density: Array = None,
            partial_densities: Array = None,
            volume_fractions: Array = None,
            ) -> Array:

        if self.equation_type in ("SINGLE-PHASE", "SINGLE-PHASE-SOLID-LS"):
            if pressure is None:
                pressure = primitives[self.energy_ids]
            if density is None:
                density = primitives[self.mass_ids]
            speed_of_sound = self.material.get_speed_of_sound(
                pressure, density)

        elif self.equation_type == "TWO-PHASE-LS":
            if pressure is None:
                pressure = primitives[self.energy_ids]
            if density is None:
                density = primitives[self.mass_ids]
            speed_of_sound = self.levelset_mixture.get_speed_of_sound(
                pressure, density)
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if pressure is None:
                pressure = primitives[self.energy_ids]
            if partial_densities is None and primitives is not None:
                partial_densities = primitives[self.mass_slices]
            if density is None:
                density = self.diffuse_5eqm_mixture.get_density(partial_densities)
            if volume_fractions is None:
                volume_fractions = primitives[self.vf_slices]
            speed_of_sound = self.diffuse_5eqm_mixture.get_speed_of_sound(
                pressure, density, volume_fractions)

        else:
            raise NotImplementedError

        return speed_of_sound

    def get_pressure(
            self,
            e: Array,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
        ) -> Array:
        if self.levelset_model == "FLUID-FLUID":
            pressure = self.levelset_mixture.get_pressure(e, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            pressure = self.diffuse_5eqm_mixture.get_pressure(e, rho, alpha_i)
        else:
            pressure = self.material.get_pressure(e, rho)

        return pressure

    def get_temperature(
            self,
            primitives: Array = None,
            pressure: Array = None,
            density: Array = None,
            volume_fractions: Array = None
        ) -> Array:

        if self.equation_type in ("SINGLE-PHASE",
                                  "SINGLE-PHASE-SOLID-LS"):
            if pressure is None:
                pressure = primitives[self.energy_ids]
            if density is None:
                density = primitives[self.mass_ids]
            temperature = self.material.get_temperature(pressure, density)

        elif self.equation_type == "TWO-PHASE-LS":
            if pressure is None:
                pressure = primitives[self.energy_ids]
            if density is None:
                density = primitives[self.mass_ids]
            temperature = self.levelset_mixture.get_temperature(
                pressure, density)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if pressure is None:
                pressure = primitives[self.energy_ids]
            if density is None:
                density = self.get_density(primitives)            
            temperature = self.diffuse_5eqm_mixture.get_temperature(
                pressure,
                density,
                primitives[self.vf_slices] if volume_fractions is None else volume_fractions)

        else:
            raise NotImplementedError            

        return temperature
    
    def get_specific_energy(
            self,
            p: Array,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
        ) -> Array:

        # Specific internal energy
        if self.levelset_model == "FLUID-FLUID":
            energy = self.levelset_mixture.get_specific_energy(p, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            energy = self.diffuse_5eqm_mixture.get_specific_energy(p, rho, alpha_i)
        else:
            energy = self.material.get_specific_energy(p, rho)

        return energy

    def get_total_energy(
            self,
            p: Array,
            u: Array,
            v: Array, 
            w: Array,
            rho:Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
        ) -> Array:

        # Total energy per unit volume
        if self.levelset_model == "FLUID-FLUID":
            total_energy = self.levelset_mixture.get_total_energy(p, u, v,  w, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            total_energy = self.diffuse_5eqm_mixture.get_total_energy(p, rho, u, v, w, alpha_i)
        else:
            total_energy = self.material.get_total_energy(p, rho, u, v, w)

        return total_energy

    def get_total_enthalpy(
            self,
            p: Array,
            u: Array,
            v: Array,
            w: Array,
            rho:Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            ) -> Array:

        # Total specific enthalpy
        if self.levelset_model == "FLUID-FLUID":
            total_enthalpy = self.levelset_mixture.get_total_enthalpy(p, u, v, w, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            total_enthalpy = self.diffuse_5eqm_mixture.get_total_enthalpy(p, rho, u, v, w, alpha_i)
        else:
            total_enthalpy = self.material.get_total_enthalpy(p, rho, u, v, w)

        return total_enthalpy

    def get_psi(
            self,
            p: Array,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
        ) -> Array:

        if self.levelset_model == "FLUID-FLUID":
            psi = self.levelset_mixture.get_psi(p, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            psi = self.diffuse_5eqm_mixture.get_psi(p, rho, alpha_i)
        else:
            psi = self.material.get_psi(p, rho)
        return psi

    def get_grueneisen(
            self,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            T: Array = None
        ) -> Array:

        if self.levelset_model == "FLUID-FLUID":
            grueneisen = self.levelset_mixture.get_grueneisen(rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            grueneisen = self.diffuse_5eqm_mixture.get_grueneisen(rho, alpha_i)
        else:
            grueneisen = self.material.get_grueneisen(rho, T)
        return grueneisen

    def get_density(self, primitives: Array) -> Array:
        """get_density [summary]

        :param primitives: [description]
        :type primitives: Array
        :return: [description]
        :rtype: Array
        """

        if self.equation_type in ("SINGLE-PHASE",
                                  "SINGLE-PHASE-SOLID-LS",
                                  "TWO-PHASE-LS"):
            density = primitives[self.mass_ids]

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            density = self.diffuse_5eqm_mixture.get_density(primitives[self.mass_slices])

        else:
            raise NotImplementedError

        return density

    def get_gamma(
            self,
            alpha_i: Array = None,
        ) -> Union[float, Array]:
        """Computes the specific heat capacity ratio gamma.
        For diffuse interface model

        :param alpha_i: _description_, defaults to None
        :type alpha_i: Array, optional
        :return: _description_
        :rtype: Union[float, Array]
        """

        if self.levelset_model == "FLUID-FLUID":
            gamma = self.levelset_mixture.get_gamma()
        elif self.diffuse_interface_model == "5EQM":
            gamma, _ = self.diffuse_5eqm_mixture.compute_mixture_EOS_params(alpha_i)
        else:
            gamma = self.material.gamma
        return gamma

    def get_sigma(self, volume_fractions: Array = None) -> Union[float, Array]:
        """Wrapper function which returns the surface tension coefficient.

        :param volume_fractions: _description_, defaults to None
        :type volume_fractions: Array, optional
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Union[float, Array]
        """

        if self.equation_type in ("SINGLE-PHASE",
                                  "SINGLE-PHASE-SOLID-LS"):
            raise NotImplementedError

        elif self.equation_type == "TWO-PHASE-LS":
            sigma = self.levelset_mixture.get_sigma()
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            sigma = self.diffuse_5eqm_mixture.get_sigma()

        else:
            raise NotImplementedError

        return sigma

    def get_specific_gas_constant(
            self,
            alpha_i: Array = None,
        ) -> Union[float, Array]:
        if self.levelset_model == "FLUID-FLUID":
            R = self.levelset_mixture.get_R()
        elif self.diffuse_interface_model == "5EQM":
            # TODO
            R = 1
        else:
            R = self.material.R
        return R

    def get_background_pressure(self, alpha_i: Array = None) -> Array:
        if self.levelset_model == "FLUID-FLUID":
            pb = self.levelset_mixture.get_background_pressure()
        elif self.diffuse_interface_model == "5EQM":
            _, pb = self.diffuse_5eqm_mixture.compute_mixture_EOS_params(alpha_i)
        else:
            pb = self.material.pb
        return pb
    
    def get_phase_background_pressure(self) -> Array:
        if self.levelset_model == "FLUID-FLUID":
            raise NotImplementedError
        elif self.diffuse_interface_model == "5EQM":
            pb_phase = self.diffuse_5eqm_mixture.get_phase_background_pressure()
        else:
            raise NotImplementedError
        return pb_phase

    def get_phasic_density(
            self,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            p: Array = None,
            T: Array = None
        ) -> Array:
        
        if self.diffuse_interface_model == "5EQM":
            rho_i = self.diffuse_5eqm_mixture.get_phasic_density(alpha_rho_i, alpha_i)
        else:
            raise NotImplementedError
        return rho_i

    def get_phasic_specific_energy(
            self,
            p: Array,
            T: Array = None,
        ) -> Array:

        if self.diffuse_interface_model == "5EQM":
            energy = self.diffuse_5eqm_mixture.get_phasic_energy(p)
        else:
            raise NotImplementedError
        return energy
    
    def get_phasic_volume_specific_enthalpy(
            self,
            p: Array,
            rho_k: Array = None
        ) -> Array:
        """Computes the phasic volume specific enthalpies.
        
        rho_k h_k = rho_k e_k + p_k

        :param p: _description_
        :type p: Array
        :param rho_k: _description_, defaults to None
        :type rho_k: Array, optional
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        
        if self.diffuse_interface_model == "5EQM":
            rhoh = self.diffuse_5eqm_mixture.get_phasic_volume_specific_enthalpy(p)
        else:
            raise NotImplementedError 
        return rhoh


    def get_specific_heat_capacity(self, temperature: Array, primitives: Array) -> Array:

        if self.equation_type in ("SINGLE-PHASE",
                                  "SINGLE-PHASE-SOLID-LS"):
            specific_heat_capacity = self.material.get_specific_heat_capacity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            specific_heat_capacity = self.levelset_mixture.get_specific_heat_capacity(temperature)
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            raise NotImplementedError # TODO deniz

        return specific_heat_capacity
    

    def get_stagnation_temperature(
            self,
            primitives:Array,
        ) -> Array:
        """Computes the stagnation temperature

        :param rho: Density buffer
        :type rho: Array
        :return: Grueneisen
        :rtype: Array
        """
        raise NotImplementedError
