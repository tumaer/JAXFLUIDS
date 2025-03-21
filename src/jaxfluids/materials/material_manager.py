from typing import Dict, Union, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.data_types.case_setup.material_properties import MaterialManagerSetup, \
    MaterialPropertiesSetup, DiffuseMixtureSetup, LevelsetMixtureSetup
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.mixture_materials.levelset_mixture import LevelsetMixture
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.materials.single_materials.material import Material
from jaxfluids.materials.mixture_materials.diffuse_mixture import DiffuseMixture
from jaxfluids.materials.mixture_materials.diffuse_mixture_four_equation import DiffuseFourEquationMixture
from jaxfluids.materials.mixture_materials.diffuse_mixture_five_equation import DiffuseFiveEquationMixture
from jaxfluids.materials import DICT_MATERIAL, DICT_MIXTURE

Array = jax.Array

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
            material_manager_setup: MaterialManagerSetup,
            ) -> None:

        self.equation_information = equation_information
        self.unit_handler = unit_handler
        
        self.equation_type = equation_information.equation_type
        self.levelset_model = equation_information.levelset_model
        self.equation_type = equation_information.equation_type
        self.diffuse_interface_model = equation_information.diffuse_interface_model
        self.cavitation_model = equation_information.cavitation_model

        self.ids_mass = equation_information.ids_mass 
        self.vel_ids = equation_information.ids_velocity 
        self.ids_energy = equation_information.ids_energy 
        self.ids_volume_fraction = equation_information.ids_volume_fraction
        self.ids_species = equation_information.ids_species
        
        self.s_mass = equation_information.s_mass
        self.vel_slices = equation_information.s_velocity
        self.s_energy = equation_information.s_energy
        self.s_volume_fraction = equation_information.s_volume_fraction
        self.s_species = equation_information.s_species

        self.levelset_mixture = None
        self.diffuse_4eqm_mixture = None
        self.diffuse_5eqm_mixture = None
        self.material = None

        if self.equation_type == "TWO-PHASE-LS":
            levelset_mixture_setup: LevelsetMixtureSetup = material_manager_setup.levelset_mixture
            self.levelset_mixture = LevelsetMixture(
                unit_handler, levelset_mixture_setup)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            diffuse_mixture_setup: DiffuseMixtureSetup = material_manager_setup.diffuse_mixture
            self.diffuse_4eqm_mixture = DiffuseFourEquationMixture(
                unit_handler, diffuse_mixture_setup)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            diffuse_mixture_setup: DiffuseMixtureSetup = material_manager_setup.diffuse_mixture
            self.diffuse_5eqm_mixture = DiffuseFiveEquationMixture(
                unit_handler, diffuse_mixture_setup)

        elif self.equation_type == "SINGLE-PHASE":
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
            mass_fractions: Array = None,
        ) -> Array:

        if self.equation_type == "SINGLE-PHASE":
            if isinstance(self.material, DICT_MATERIAL["BarotropicCavitationFluid"]):
                if density is None:
                    density = primitives[self.ids_mass]
                thermal_conductivity = self.material.get_thermal_conductivity(
                    density,
                    temperature)
            else:
                thermal_conductivity = self.material.get_thermal_conductivity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            thermal_conductivity = self.levelset_mixture.get_thermal_conductivity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            raise NotImplementedError
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if volume_fractions is None:
                volume_fractions = primitives[self.s_volume_fraction]
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
            mass_fractions: Array = None,
        ) -> Array:

        if self.equation_type == "SINGLE-PHASE":
            if isinstance(self.material, DICT_MATERIAL["BarotropicCavitationFluid"]):
                if density is None:
                    density = primitives[self.ids_mass]
                dynamic_viscosity = self.material.get_dynamic_viscosity(
                    density, temperature)
            else:
                dynamic_viscosity = self.material.get_dynamic_viscosity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            dynamic_viscosity = self.levelset_mixture.get_dynamic_viscosity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            raise NotImplementedError
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if volume_fractions is None:
                volume_fractions = primitives[self.s_volume_fraction]
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
            mass_fractions: Array = None,
        ) -> Array:

        if self.equation_type == "SINGLE-PHASE":
            if isinstance(self.material, DICT_MATERIAL["BarotropicCavitationFluid"]):
                if density is None:
                    density = primitives[self.ids_mass]
                bulk_viscosity = self.material.get_bulk_viscosity(
                    density, temperature)
            else:
                bulk_viscosity = self.material.get_bulk_viscosity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            bulk_viscosity = self.levelset_mixture.get_bulk_viscosity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            raise NotImplementedError
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if volume_fractions is None:
                volume_fractions = primitives[self.s_volume_fraction]
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
            mass_fractions: Array = None
        ) -> Array:

        if self.equation_type == "SINGLE-PHASE":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if density is None:
                density = primitives[self.ids_mass]
            speed_of_sound = self.material.get_speed_of_sound(
                pressure, density)

        elif self.equation_type == "TWO-PHASE-LS":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if density is None:
                density = primitives[self.ids_mass]
            speed_of_sound = self.levelset_mixture.get_speed_of_sound(
                pressure, density)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if partial_densities is None:
                partial_densities = primitives[self.s_mass]
            speed_of_sound = self.diffuse_4eqm_mixture.get_speed_of_sound(
                pressure, partial_densities)
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if partial_densities is None and primitives is not None:
                partial_densities = primitives[self.s_mass]
            if density is None:
                density = self.diffuse_5eqm_mixture.get_density(partial_densities)
            if volume_fractions is None:
                volume_fractions = primitives[self.s_volume_fraction]
            speed_of_sound = self.diffuse_5eqm_mixture.get_speed_of_sound(
                pressure, density, volume_fractions)
        
        else:
            raise NotImplementedError

        return speed_of_sound

    def get_speed_of_sound_liquid(
            self,
            pressure: Array,
            density: Array
        ) -> Array:
        if isinstance(self.material, DICT_MATERIAL["BarotropicCavitationFluid"]):
            speed_of_sound_liquid = self.material.get_speed_of_sound_liquid(pressure, density)
        else:
            raise NotImplementedError

        return speed_of_sound_liquid

    def get_pressure(
            self,
            e: Array,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            Y_i: Array = None,
            T_guess: Array = None,
            fluid_mask: Array = None
        ) -> Array:

        if self.levelset_model == "FLUID-FLUID":
            pressure = self.levelset_mixture.get_pressure(e, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            pressure = self.diffuse_5eqm_mixture.get_pressure(e, rho, alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            pressure = self.diffuse_4eqm_mixture.get_pressure(e, alpha_rho_i)
        else:
            pressure = self.material.get_pressure(e, rho)

        return pressure

    def get_temperature(
            self,
            primitives: Array = None,
            pressure: Array = None,
            density: Array = None,
            volume_fractions: Array = None,
            mass_fractions: Array = None,
        ) -> Array:

        if self.equation_type == "SINGLE-PHASE":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if density is None:
                density = primitives[self.ids_mass]
            temperature = self.material.get_temperature(pressure, density)

        elif self.equation_type == "TWO-PHASE-LS":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if density is None:
                density = primitives[self.ids_mass]
            temperature = self.levelset_mixture.get_temperature(
                pressure, density)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if density is None:
                density = self.diffuse_4eqm_mixture.get_density(
                    primitives[self.s_mass])
            if mass_fractions is None:
                mass_fractions = primitives[self.s_mass] / density
            temperature = self.diffuse_4eqm_mixture.get_temperature_from_density_and_pressure(
                density,
                pressure,
                mass_fractions)
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            if pressure is None:
                pressure = primitives[self.ids_energy]
            if density is None:
                density = self.get_density(primitives)            
            temperature = self.diffuse_5eqm_mixture.get_temperature(
                pressure,
                density,
                primitives[self.s_volume_fraction] if volume_fractions is None else volume_fractions)

        else:
            raise NotImplementedError            

        return temperature
    
    def get_specific_energy(
            self,
            p: Array,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            Y_i: Array = None,
        ) -> Array:

        # Specific internal energy
        if self.levelset_model == "FLUID-FLUID":
            energy = self.levelset_mixture.get_specific_energy(p, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            energy = self.diffuse_5eqm_mixture.get_specific_energy(p, rho, alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            energy = self.diffuse_4eqm_mixture.get_specific_energy(p, alpha_rho_i)
        else:
            energy = self.material.get_specific_energy(p, rho)

        return energy

    def get_total_energy(
            self,
            p: Array,
            velocity_vec: Array,
            rho:Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            Y_i: Array = None,
        ) -> Array:

        # Total energy per unit volume
        if self.levelset_model == "FLUID-FLUID":
            total_energy = self.levelset_mixture.get_total_energy(p, velocity_vec, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            total_energy = self.diffuse_5eqm_mixture.get_total_energy(p, rho, velocity_vec, alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            raise NotImplementedError
        else:
            total_energy = self.material.get_total_energy(p, rho, velocity_vec)

        return total_energy

    def get_total_enthalpy(
            self,
            p: Array,
            velocity_vec: Array,
            rho:Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            Y_i: Array = None,
        ) -> Array:

        # Total specific enthalpy
        if self.levelset_model == "FLUID-FLUID":
            total_enthalpy = self.levelset_mixture.get_total_enthalpy(p, velocity_vec, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            total_enthalpy = self.diffuse_5eqm_mixture.get_total_enthalpy(p, rho, velocity_vec, alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            raise NotImplementedError
        else:
            total_enthalpy = self.material.get_total_enthalpy(p, rho, velocity_vec)

        return total_enthalpy

    def get_psi(
            self,
            p: Array,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            Y_i: Array = None,
        ) -> Array:

        if self.levelset_model == "FLUID-FLUID":
            psi = self.levelset_mixture.get_psi(p, rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            psi = self.diffuse_5eqm_mixture.get_psi(p, rho, alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            raise NotImplementedError
        else:
            psi = self.material.get_psi(p, rho)
        return psi

    def get_grueneisen(
            self,
            rho: Array = None,
            alpha_rho_i: Array = None,
            alpha_i: Array = None,
            Y_i: Array = None,
            T: Array = None
        ) -> Array:

        if self.levelset_model == "FLUID-FLUID":
            grueneisen = self.levelset_mixture.get_grueneisen(rho)
        elif self.diffuse_interface_model == "5EQM":
            rho = self.diffuse_5eqm_mixture.get_density(alpha_rho_i) if rho is None else rho
            grueneisen = self.diffuse_5eqm_mixture.get_grueneisen(rho, alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            raise NotImplementedError
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

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            density = primitives[self.ids_mass]

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            density = self.diffuse_4eqm_mixture.get_density(primitives[self.s_mass])
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            density = self.diffuse_5eqm_mixture.get_density(primitives[self.s_mass])

       
        else:
            raise NotImplementedError

        return density



    def get_density_from_pressure_and_temperature(
            self,
            p: Array,
            T: Array,
            Y_k: Array = None
            ) -> Array:
        
        # NOTE this is needed to fill density in isothermal wall halo cells and ghost cells
        if self.equation_type == "SINGLE-PHASE":
            density = self.material.get_density_from_pressure_and_temperature(p, T)

        elif self.equation_type == "TWO-PHASE-LS":
            raise NotImplementedError
        
        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            raise NotImplementedError
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            raise NotImplementedError

        else:
            raise NotImplementedError

        return density         


    def get_gamma(
            self,
            alpha_i: Array = None,
            Y_i: Array = None,
            T: Array = None
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
        elif self.diffuse_interface_model == "4EQM":
            raise NotImplementedError
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

        if self.equation_type == "SINGLE-PHASE":
            raise NotImplementedError

        elif self.equation_type == "TWO-PHASE-LS":
            sigma = self.levelset_mixture.get_sigma()

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            sigma = self.diffuse_4eqm_mixture.get_sigma()
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            sigma = self.diffuse_5eqm_mixture.get_sigma()

        else:
            raise NotImplementedError

        return sigma

    def get_specific_gas_constant(
            self,
            alpha_i: Array = None,
            Y_i: Array = None,
        ) -> Union[float, Array]:
        # TODO DOCSTRING
        # TODO RENAME
        if self.levelset_model == "FLUID-FLUID":
            R = self.levelset_mixture.get_R()
        elif self.diffuse_interface_model == "5EQM":
            # TODO
            R = 1
        elif self.diffuse_interface_model == "4EQM":
            raise NotImplementedError
        else:
            R = self.material.R
        return R

    def get_background_pressure(self, alpha_i: Array = None) -> Array:
        if self.levelset_model == "FLUID-FLUID":
            pb = self.levelset_mixture.get_background_pressure()
        elif self.diffuse_interface_model == "5EQM":
            _, pb = self.diffuse_5eqm_mixture.compute_mixture_EOS_params(alpha_i)
        elif self.diffuse_interface_model == "4EQM":
            # TODO 4EQM
            return 0.0
        else:
            pb = self.material.pb
        return pb
    
    def get_phase_background_pressure(self) -> Array:
        if self.levelset_model == "FLUID-FLUID":
            raise NotImplementedError
        elif self.diffuse_interface_model == "5EQM":
            pb_phase = self.diffuse_5eqm_mixture.get_phase_background_pressure()
        elif self.diffuse_interface_model == "4EQM":
            pb_phase = self.diffuse_4eqm_mixture.get_phase_background_pressure()
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
        elif self.diffuse_interface_model == "4EQM":
            rho_i = self.diffuse_4eqm_mixture.get_phasic_density_from_pressure_temperature(
                p, T)
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
        elif self.diffuse_interface_model == "4EQM":
            energy = self.diffuse_4eqm_mixture.get_phasic_energy(p, T)
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
        elif self.diffuse_interface_model == "4EQM":
            rhoh = self.diffuse_4eqm_mixture.get_phasic_volume_specific_enthalpy(p, rho_k)
        else:
            raise NotImplementedError 
        return rhoh

    def get_mass_fraction(self, partial_densities: Array = None) -> Array:
        """Computes the mass fraction from partial densities.
        :param rhoY_k: _description_
        :type rhoY_k: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            mass_fractions = None

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            mass_fractions = self.diffuse_4eqm_mixture.get_mass_fractions(partial_densities)
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            mass_fractions = self.diffuse_5eqm_mixture.get_mass_fractions(partial_densities)

        else:
            raise NotImplementedError

        return mass_fractions
    
    def get_specific_heat_capacity(self, temperature: Array, primitives: Array) -> Array:

        if self.equation_type == "SINGLE-PHASE":
            specific_heat_capacity = self.material.get_specific_heat_capacity(temperature)

        elif self.equation_type == "TWO-PHASE-LS":
            specific_heat_capacity = self.levelset_mixture.get_specific_heat_capacity(temperature)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            raise NotImplementedError # TODO deniz
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            raise NotImplementedError # TODO deniz

        else:
            raise NotImplementedError

        return specific_heat_capacity
    

    def get_stagnation_temperature(
            self,
            primitives: Array,
        ) -> Array:
        """Computes the stagnation temperature

        :param rho: Density buffer
        :type rho: Array
        :return: Grueneisen
        :rtype: Array
        """
        raise NotImplementedError

    def get_dynamic_pressure(
        self,
        primitives: Array = None,
        density: Array = None,
        velocity_vec: Array = None,
    ) -> Array:
        
        if self.equation_type == "SINGLE-PHASE":
            raise NotImplementedError
        
        elif self.equation_type == "TWO-PHASE-LS":
            raise NotImplementedError

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            raise NotImplementedError
        
        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            raise NotImplementedError

        else:
            raise NotImplementedError

        return dynamic_pressure