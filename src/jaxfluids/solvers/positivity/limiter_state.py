import jax.numpy as jnp
from jax import Array

from jaxfluids.config import precision
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

class PositivityLimiterState:

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            ) -> None:
        self.eps = precision.get_interpolation_limiter_eps()

        self.material_manager = material_manager
        self.equation_information = equation_manager.equation_information
        
        self.domain_slices_conservatives = domain_information.domain_slices_conservatives
        
        self.mass_ids = self.equation_information.mass_ids
        self.mass_slices = self.equation_information.mass_slices
        self.vel_ids = self.equation_information.velocity_ids
        self.vel_slices = self.equation_information.velocity_slices
        self.energy_ids = self.equation_information.energy_ids
        self.energy_slices = self.equation_information.energy_slices
        self.vf_ids = self.equation_information.vf_ids
        self.vf_slices = self.equation_information.vf_slices
    
    def correct_volume_fraction(self, conservatives: Array) -> Array:
        """Corrects the volume fraction in cells which produce a negative
        effective pressure p + pb_mix - only for two-phase flows. Only the
        volume fraction is updated which influences pb_mix but not the rest
        of the conservative variables.

        :param conservatives: _description_
        :type conservatives: Array
        :return: _description_
        :rtype: Array
        """

        counter = None
        if self.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM" \
            and self.equation_information.no_fluids == 2:

            # Ill-conditioned if Gamma_0 = Gamma_1 and Pb_0 = Pb_1
            # TODO this check is not very nice
            if self.material_manager.diffuse_5eqm_mixture.is_volume_fraction_admissible:

                alpha_rho_vec = conservatives[(self.mass_slices,) + self.domain_slices_conservatives]
                alpha_vec = conservatives[(self.vf_slices,) + self.domain_slices_conservatives]
                momentum_vec = conservatives[(self.vel_slices,) + self.domain_slices_conservatives]
                total_energy = conservatives[(self.energy_ids,) + self.domain_slices_conservatives]

                rho = self.material_manager.get_density(conservatives[(...,) + self.domain_slices_conservatives])
                gamma_mix, pb_mix = self.material_manager.diffuse_5eqm_mixture.compute_mixture_EOS_params(alpha_vec)
                rhoe = total_energy - 0.5 * jnp.sum(momentum_vec * momentum_vec, axis=0) / rho

                # primitives = self.equation_manager.get_primitives_from_conservatives(conservatives[...,self.nhx,self.nhy,self.nhz])
                # min_alpharho0 = jnp.min(alpha_rho_vec[0])
                # min_alpharho1 = jnp.min(alpha_rho_vec[1])
                # min_alpha, max_alpha = jnp.min(alpha_vec[-1]), jnp.max(alpha_vec[-1])
                
                # min_pressure_1 = jnp.min(_primitives[-2] + pb_mix)
                # min_pressure_2 = jnp.min((gamma_mix - 1) * (rhoe - pb_mix))
                # min_density  = jnp.min(rho)
                # mask = jnp.where(_primitives[-2] + pb_mix < 0, 1, 0)
                # print(min_alpharho0, min_alpharho1)
                # print(min_alpha, max_alpha)
                # print(min_pressure_1, jnp.sum(mask))
                # print(min_pressure_2)
                # print(min_density)

                # mask_neg_pressure = jnp.where(_primitives[-2] + pb_mix < 0)
                # mask_neg_pressure = jnp.where(rhoe - pb_mix < 0)
                # print(mask_neg_pressure)
                # print(primitives[-1][mask_neg_pressure])
                # print((primitives[-2] + pb_mix)[mask_neg_pressure])
                
                Gamma_vec = self.material_manager.diffuse_5eqm_mixture.one_gamma_vec_
                Pb_vec = self.material_manager.diffuse_5eqm_mixture.gamma_pb_vec_
                Gamma_1 = Gamma_vec[1]
                delta_Gamma = Gamma_vec[0] - Gamma_vec[1]
                Pb_1 = Pb_vec[1]
                delta_Pb = Pb_vec[0] - Pb_vec[1]

                alpha_cor = (Pb_1 + (1.0 + Gamma_1) * (self.eps.pressure - rhoe)) / (delta_Gamma * (rhoe - self.eps.pressure) - delta_Pb)
                mask = jnp.where(rhoe - pb_mix < self.eps.pressure, 1, 0)
                alpha_vec = alpha_cor * mask + (1 - mask) * alpha_vec

                # print(alpha_cor[mask_neg_pressure])
                # gamma_mix, pb_mix = self.material_manager.diffuse_5eqm_mixture.compute_mixture_EOS_params(alpha_vec)
                # mask_neg_pressure = jnp.where(rhoe - pb_mix < 0) # TODO
                # print("DEBUG VF LIMITER, MASK NEG P =", mask_neg_pressure)

                # C1 = - 1.0 / (delta_Gamma**2 * eps) * (delta_Gamma * rhoe - delta_Pb - delta_Gamma * (1 + Gamma_1) * eps - delta_Gamma * Gamma_1 * eps)
                # C2 = - 1.0 / (delta_Gamma**2 * eps) * (rhoe * (1 + Gamma_1) - Pb_1 - Gamma_1 * (1 + Gamma_1) * eps)

                # alpha_cor_1 = - 0.5 * C1 + jnp.sqrt(0.25 * C1**2 - C2)
                # alpha_cor_2 = - 0.5 * C1 - jnp.sqrt(0.25 * C1**2 - C2)
                # print(alpha_cor_1[mask_neg_pressure])
                # print(alpha_cor_2[mask_neg_pressure])
                # input()

                conservatives = conservatives.at[(self.vf_slices,) + self.domain_slices_conservatives].set(alpha_vec)
                counter = jnp.sum(mask)
                
            else:
                counter = 0

        else:
            raise NotImplementedError

        return conservatives, counter
    