from typing import Dict, List, Union, Tuple

import jax.numpy as jnp
from jax import Array

from jaxfluids.materials import DICT_MATERIAL
from jaxfluids.materials.mixture_materials.mixture import Mixture
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.materials.single_materials.material import Material
from jaxfluids.data_types.case_setup.material_properties import DiffuseMixtureSetup, MaterialPropertiesSetup

class DiffuseMixture(Mixture):
    """DiffuseMixture

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

        self.gamma      = jnp.array([material.gamma for material in self.materials.values()])
        self.one_gamma_ = jnp.array([1.0 / (material.gamma - 1.0) for material in self.materials.values()])
        self.pb         = jnp.array([material.pb for material in self.materials.values()])
        self.gamma_pb_  = jnp.array([material.gamma * material.pb / (material.gamma - 1.0) for material in self.materials.values()])

        if self.number_fluids == 2:
            Delta_Gamma = self.one_gamma_[0] - self.one_gamma_[1]
            Delta_Pb = self.gamma_pb_[0] - self.gamma_pb_[1]
            self.is_volume_fraction_admissible = Delta_Gamma != 0.0 and Delta_Pb != 0.0

        self.gamma_vec = self.gamma.reshape(-1,1,1,1)
        self.one_gamma_vec_ = self.one_gamma_.reshape(-1,1,1,1)
        self.pb_vec_ = self.pb.reshape(-1,1,1,1)
        self.gamma_pb_vec_ = self.gamma_pb_.reshape(-1,1,1,1)

        self.bulk_viscosity = jnp.array([material.bulk_viscosity for material in self.materials.values()])
        
        pairing_properties = diffuse_mixture_setup.pairing_properties
        self.sigma = pairing_properties.surface_tension_coefficient

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
        gamma   = 1.0 + 1.0 / sum(alpha_i[ii] * self.one_gamma_[ii] for ii in range(len(self.one_gamma_)))
        pb      = (gamma - 1) / gamma * sum(alpha_i[ii] * self.gamma_pb_[ii] for ii in range(len(self.gamma_pb_)))
        return gamma, pb 

    def get_density(self, alpha_rho_i: Array) -> Array:
        return jnp.sum(alpha_rho_i, axis=0)
    
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

    def get_speed_of_sound(self, p: Array, rho: Array, alpha_i: Array) -> Array:
        """Computes speed of sound from pressure and density.
        c = c(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Speed of sound buffer
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        return jnp.sqrt( gamma * ( p + pb ) / rho )

    def get_pressure(self, e: Array, rho: Array, alpha_i: Array) -> Array:
        """Computes pressure from internal energy and density.
        p = p(e, rho)

        :param e: Specific internal energy buffer
        :type e: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Pressue buffer
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)

        return ( gamma - 1 ) * e * rho - gamma * pb

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
    
    def get_specific_energy(self, p: Array, rho: Array, alpha_i: Array) -> Array:
        """Computes specific internal energy
        e = e(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Specific internal energy buffer
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        return ( p + gamma * pb ) / ( rho * (gamma - 1) )

    def get_phasic_energy(self, p: Array):
        """Computes (volume-specific) internal energy
        for each phase in a diffuse mixture.

        rho_i e_i = (p + \gamma_i pb_i) / (\gamma_i - 1)
        """
        return self.one_gamma_vec_ * p + self.gamma_pb_vec_

    def get_phasic_volume_specific_enthalpy(self, p: Array) -> Array:
        """Computes the (volume-specific) enthalpy
        for each phase in a diffuse mixture.

        :param p: _description_
        :type p: Array
        :return: _description_
        :rtype: Array
        """
        return self.gamma_vec * self.one_gamma_vec_ * p + self.gamma_pb_vec_

    def get_total_energy(self, p: Array, rho: Array, u: Array, v:Array,
        w:Array, alpha_i: Array) -> Array:
        """Computes total energy per unit volume from pressure, density, and velocities.
        E = E(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :param u: Velocity in x direction
        :type u: Array
        :param v: Velocity in y direction
        :type v: Array
        :param w: Velocity in z direction
        :type w: Array
        :return: Total energy per unit volume
        :rtype: Array
        """
        gamma, pb = self.compute_mixture_EOS_params(alpha_i)
        return ( p + gamma * pb ) / (gamma - 1) + 0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(self, p: Array, rho: Array, u: Array, v: Array, 
        w: Array, alpha_i: Array) -> Array:
        """Computes total specific enthalpy from pressure, density, and velocities.
        H = H(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :param u: Velocity in x direction
        :type u: Array
        :param v: Velocity in y direction
        :type v: Array
        :param w: Velocity in z direction
        :type w: Array
        :return: Total specific enthalpy buffer
        :rtype: Array
        """
        return ( self.get_total_energy(p, rho, u, v, w, alpha_i) + p ) / rho

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
        return self.sigma
