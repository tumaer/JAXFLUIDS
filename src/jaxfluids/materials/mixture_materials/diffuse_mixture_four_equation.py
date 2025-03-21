from typing import Dict, List, Union, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.materials import DICT_MATERIAL
from jaxfluids.materials.mixture_materials.mixture import Mixture
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.materials.single_materials.material import Material
from jaxfluids.data_types.case_setup.material_properties import DiffuseMixtureSetup, MaterialPropertiesSetup
from jaxfluids.math.sum_consistent import sum3_consistent

Array = jax.Array

class DiffuseFourEquationMixture(Mixture):
    """DiffuseFourEquationMixture

    :param Mixture: _description_
    :type Mixture: _type_
    """

    def __init__(
            self,
            unit_handler: UnitHandler,
            diffuse_mixture_setup: DiffuseMixtureSetup
            ) -> None:

        super().__init__(unit_handler, diffuse_mixture_setup)

        # INSTANTIATE FLUID PHASES
        self.fluid_names = diffuse_mixture_setup.fluids._fields
        self.number_fluids = len(self.fluid_names)
        for fluid in self.fluid_names:
            material_setup: MaterialPropertiesSetup = getattr(diffuse_mixture_setup.fluids, fluid)
            material_type = material_setup.eos.model
            self.materials[fluid] = DICT_MATERIAL[material_type](unit_handler, material_setup)

        gamma_vec, Gamma_vec, pi_vec, Pi_vec, q_vec, b_vec = [], [], [], [], [], []
        cp_vec, cv_vec = [], []
        bulk_viscosity = []
        for material_key, material in self.materials.items():
            
            gamma_vec.append(material.gamma)
            Gamma_vec.append(1.0 / (material.gamma - 1.0))
            pi_vec.append(material.pb)
            Pi_vec.append(material.gamma * material.pb / (material.gamma - 1.0))
            cp_vec.append(material.cp)
            cv_vec.append(material.cv)
            bulk_viscosity.append(material.bulk_viscosity)

        if self.number_fluids == 2:
            Delta_Gamma = Gamma_vec[0] - Gamma_vec[1]
            Delta_Pb = Pi_vec[0] - Pi_vec[1]
            self.is_volume_fraction_admissible = Delta_Gamma != 0.0 and Delta_Pb != 0.0

        self.gamma_vec = jnp.array(gamma_vec).reshape(-1,1,1,1)
        self.Gamma_vec = jnp.array(Gamma_vec).reshape(-1,1,1,1)
        self.pi_vec = jnp.array(pi_vec).reshape(-1,1,1,1)
        self.Pi_vec = jnp.array(Pi_vec).reshape(-1,1,1,1)
        self.cp_vec = jnp.array(cp_vec).reshape(-1,1,1,1)
        self.cv_vec = jnp.array(cv_vec).reshape(-1,1,1,1)

        self.bulk_viscosity = jnp.array(bulk_viscosity)
        
        pairing_properties = diffuse_mixture_setup.pairing_properties
        self.surface_tension_coefficient = pairing_properties.surface_tension_coefficient

    def compute_mixture_EOS_params(self, alpha_i: Array) -> Tuple[Array, Array]:
        """Calculates the parameters of the stiffened EOS in the mixture region.
        Returns gamma_mixture and pb_mixture.

        :param alpha_i: [description]
        :type alpha_i: Array
        :return: [description]
        :rtype: Array
        """
        # TODO can we improve on sum implementation??
        alpha_i = jnp.stack([*alpha_i, 1.0 - jnp.sum(alpha_i, axis=0)], axis=0) 
        gamma_mixture = 1.0 + 1.0 / sum(alpha_i[ii] * self.Gamma_vec[ii] for ii in range(len(self.Gamma_vec)))
        pb_mixture = (gamma_mixture - 1) / gamma_mixture * sum(alpha_i[ii] * self.Pi_vec[ii] for ii in range(len(self.Pi_vec)))
        return gamma_mixture, pb_mixture 

    def get_density(self, alpha_rho_i: Array) -> Array:
        return jnp.sum(alpha_rho_i, axis=0)

    def get_mass_fractions(self, alpha_rho_i: Array) -> Array:
        return alpha_rho_i / jnp.sum(alpha_rho_i, axis=0, keepdims=True)

    def get_volume_fractions_from_pressure_temperature(
            self,
            alpha_rho_i: Array,
            p: Array,
            T: Array
            ) -> Array:
        rho_i = self.get_phasic_density_from_pressure_temperature(p, T)
        return alpha_rho_i / rho_i

    def get_phasic_density(
            self,
            alpha_rho_i: Array,
            alpha_i: Array,
            ) -> Array:

        volume_fraction_full = jnp.concatenate([
            alpha_i,
            1.0 - jnp.sum(alpha_i, axis=0, keepdims=True)
        ], axis=0)
        return alpha_rho_i / (volume_fraction_full + 1e-100)

    def get_phasic_density_from_pressure_temperature(
            self,
            p: Array,
            T: Array
            ) -> Array:
        b_vec = 0.0
        specific_volume = ((self.gamma_vec - 1.0) * self.cv_vec * T) / (p + self.pi_vec) + b_vec
        return 1.0 / specific_volume

    def get_thermal_conductivity(self, T: Array, alpha_i: Array) -> Array:
        alpha_i = jnp.stack([*alpha_i, 1.0 - jnp.sum(alpha_i, axis=0)], axis=0)
        # TODO consider substituting with for loop instead of stacking
        thermal_conductivity = jnp.stack([material.get_thermal_conductivity(T) for material in self.materials.values()], axis=0)
        thermal_conductivity = sum(alpha_i[ii] * thermal_conductivity[ii] for ii in range(alpha_i.shape[0]))
        return thermal_conductivity

    def get_dynamic_viscosity(self, T: Array, alpha_i: Array) -> Array:
        alpha_i = jnp.stack([*alpha_i, 1.0 - jnp.sum(alpha_i, axis=0)], axis=0)
        # TODO consider substituting with for loop instead of stacking
        dynamic_viscosity = jnp.stack([material.get_dynamic_viscosity(T) for material in self.materials.values()], axis=0)
        dynamic_viscosity = sum(alpha_i[ii] * dynamic_viscosity[ii] for ii in range(alpha_i.shape[0]))
        return dynamic_viscosity

    def get_bulk_viscosity(self, alpha_i: Array) -> Array:
        bulk_viscosity = sum(alpha_i[ii] * self.bulk_viscosity[ii] for ii in range(alpha_i.shape[0]))
        return bulk_viscosity

    def get_speed_of_sound(self, p: Array, alpha_rho_i: Array) -> Array:
        """Computes speed of sound from pressure and density.
        c = c(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Speed of sound buffer
        :rtype: Array
        """
        density = jnp.sum(alpha_rho_i, axis=0)
        mass_fractions = alpha_rho_i / density
        T = self.get_temperature_from_density_and_pressure(
            density, p, mass_fractions)
        rho_k = (p + self.pi_vec) / ((self.gamma_vec - 1.0) * self.cv_vec * T)
        volume_fractions = alpha_rho_i / rho_k
        c_k = jnp.sqrt(self.gamma_vec * (p + self.pi_vec) / (rho_k * (1.0 - rho_k * 0.0)))
        tmp = jnp.sum(volume_fractions / (rho_k * c_k * c_k), axis=0)
        return jnp.sqrt(1.0 / (density * tmp))

    def get_pressure(self, e: Array, alpha_rho_i: Array) -> Array:
        """Computes pressure from internal energy and density.
        p = p(e, rho)

        :param e: Specific internal energy buffer
        :type e: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Pressue buffer
        :rtype: Array
        """
        Q = self.get_density(alpha_rho_i) * e
        a2 = alpha_rho_i[0] * self.cv_vec[0] + alpha_rho_i[1] * self.cv_vec[1]
        a1 = alpha_rho_i[0] * self.cv_vec[0] * (self.pi_vec[1] + self.gamma_vec[0] * self.pi_vec[0] - (self.gamma_vec[0] - 1.0) * Q) \
            + alpha_rho_i[1] * self.cv_vec[1] * (self.pi_vec[0] + self.gamma_vec[1] * self.pi_vec[1] - (self.gamma_vec[1] - 1.0) * Q)
        a0 = -Q * (
            (self.gamma_vec[0] - 1.0) * alpha_rho_i[0] * self.cv_vec[0] * self.pi_vec[1] \
            + (self.gamma_vec[1] - 1.0) * alpha_rho_i[1] * self.cv_vec[1] * self.pi_vec[0]) \
            + self.pi_vec[0] * self.pi_vec[1] * (
                self.gamma_vec[0] * alpha_rho_i[0] * self.cv_vec[0] \
                + self.gamma_vec[1] * alpha_rho_i[1] * self.cv_vec[1])
        return (-a1 + jnp.sqrt(a1 * a1 - 4.0 * a0 * a2)) / (2 * a2)

    def get_temperature(self, p: Array, rho: Array, alpha_i: Array) -> Array:
        """Computes temperature from pressure and density.
        T = T(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Temperature buffer
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        R = 1 # TODO
        return ( p + pb ) / ( rho * R )

    def get_temperature_from_density_and_pressure(
            self,
            rho: Array,
            p: Array,
            Y_k: Array
            ) -> Array:
        b = 0.0
        tmp = jnp.sum((self.gamma_vec - 1.0) * Y_k * self.cv_vec / (p + self.pi_vec), axis=0)
        return (1.0 / rho - b) / tmp
    
    def get_temperature_from_energy_and_pressure(
            self,
            e: Array,
            p: Array,
            Y_k: Array
            ) -> Array:
        q = 0.0
        tmp = jnp.sum(Y_k * self.cv_vec * (p + self.gamma_vec * self.pi_vec) / (p + self.pi_vec), axis=0)
        return (e - q) / tmp

    def get_specific_energy(self, p: Array, alpha_rho_i: Array) -> Array:
        """Computes specific internal energy
        e = e(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Specific internal energy buffer
        :rtype: Array
        """
        density = jnp.sum(alpha_rho_i, axis=0)
        mass_fractions = alpha_rho_i / density
        T = self.get_temperature_from_density_and_pressure(density, p, mass_fractions)
        e_k = self.get_phasic_energy(p, T)
        return jnp.sum(e_k * mass_fractions, axis=0)

    def get_phasic_energy(self, p: Array, T: Array) -> Array:
        """Computes (volume-specific) internal energy
        for each phase in a diffuse mixture.

        rho_i e_i = (p + \gamma_i pb_i) / (\gamma_i - 1)
        """
        qk = 0.0
        return (p + self.gamma_vec * self.pi_vec) * self.cv_vec * T / (p + self.pi_vec) + qk

    def get_phasic_volume_specific_enthalpy(
            self,
            p: Array,
            rho_k: Array
            ) -> Array:
        """Computes the (volume-specific) enthalpy
        for each phase in a diffuse mixture.

        :param p: _description_
        :type p: Array
        :return: _description_
        :rtype: Array
        """
        bk = 0.0
        qk = 0.0
        return (p + self.gamma_vec * self.pi_vec) * (1.0 - rho_k * bk) / (self.gamma_vec - 1.0) \
            + qk * rho_k + p

    def get_total_energy(self, p: Array, rho: Array, velocity_vec: Array, alpha_i: Array) -> Array:
        """Computes total energy per unit volume from pressure, density, and velocities.
        E = E(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :param velocity_vec: Velocity vector, shape = (N_vel,Nx,Ny,Nz)
        :type velocity_vec: Array
        :return: Total energy per unit volume
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        return ( p + gamma * pb ) / (gamma - 1) + 0.5 * rho * sum3_consistent(*jnp.square(velocity_vec))

    def get_total_enthalpy(self, p: Array, rho: Array, velocity_vec: Array, alpha_i: Array) -> Array:
        """Computes total specific enthalpy from pressure, density, and velocities.
        H = H(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :param velocity_vec: Velocity vector, shape = (N_vel,Nx,Ny,Nz)
        :type velocity_vec: Array
        :return: Total specific enthalpy buffer
        :rtype: Array
        """
        return ( self.get_total_energy(p, rho, velocity_vec, alpha_i) + p ) / rho

    def get_psi(self, p: Array, rho: Array, alpha_i: Array) -> Array:
        """Computes psi from pressure and density.
        psi = p_rho; p_rho is partial derivative of pressure wrt density.

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Psi
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        return ( p + gamma * pb ) / rho

    def get_grueneisen(self, rho: Array, alpha_i: Array) -> Array:
        """Computes the Grueneisen coefficient from density.
        Gamma = p_e / rho; p_e is partial derivative of pressure wrt internal specific energy.

        :param rho: Density buffer
        :type rho: Array
        :return: Grueneisen
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        return gamma - 1
    
    def get_sigma(self) -> Array:
        return self.surface_tension_coefficient

    def get_phase_background_pressure(self) -> Array:
        return self.pi_vec
