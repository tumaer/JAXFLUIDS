from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.solvers.riemann_solvers.signal_speeds import compute_sstar
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

class HLLCLM(RiemannSolver):
    """HLLCLM implements functionality for the HLLC-LM Riemann Solver
    according to Fleischmann et al. 2020.
    """

    def __init__(
            self,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            signal_speed: Callable,
            **kwargs
            ) -> None:
        super().__init__(material_manager, equation_manager, signal_speed)

        self.s_star = compute_sstar
        self.Ma_limit = 0.1

    def _wave_speed_limiter(
            self,
            velocity_normal_L: Array,
            velocity_normal_R: Array,
            speed_of_sound_L: Array,
            speed_of_sound_R: Array
        ) -> Array:
        """_summary_

        Fleischmann et al. - 2020 - Eq (23 - 25)

        :param velocity_normal_L: _description_
        :type velocity_normal_L: Array
        :param velocity_normal_R: _description_
        :type velocity_normal_R: Array
        :param speed_of_sound_L: _description_
        :type speed_of_sound_L: Array
        :param speed_of_sound_R: _description_
        :type speed_of_sound_R: Array
        :return: _description_
        :rtype: Array
        """
        Ma_local = jnp.maximum(
            jnp.abs(velocity_normal_L / speed_of_sound_L), 
            jnp.abs(velocity_normal_R / speed_of_sound_R)
        )
        phi = jnp.sin(jnp.minimum(1.0, Ma_local / self.Ma_limit) * jnp.pi * 0.5)
        return phi

    def _solve_riemann_problem_xi_single_phase(
            self, 
            primitives_L: Array,
            primitives_R: Array, 
            conservatives_L: Array,
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Array, Array]:
        
        speed_of_sound_L = self.material_manager.get_speed_of_sound(primitives_L)
        speed_of_sound_R = self.material_manager.get_speed_of_sound(primitives_R)

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            primitives_L[self.velocity_ids[axis]], 
            primitives_R[self.velocity_ids[axis]], 
            speed_of_sound_L, 
            speed_of_sound_R, 
            rho_L=primitives_L[self.mass_ids], 
            rho_R=primitives_R[self.mass_ids], 
            p_L=primitives_L[self.energy_ids], 
            p_R=primitives_R[self.energy_ids], 
            gamma=self.material_manager.get_gamma()
        )
        wave_speed_contact = self.s_star(
            primitives_L[self.velocity_ids[axis]], 
            primitives_R[self.velocity_ids[axis]], 
            primitives_L[self.energy_ids], 
            primitives_R[self.energy_ids], 
            primitives_L[self.mass_ids], 
            primitives_R[self.mass_ids],
            wave_speed_simple_L, 
            wave_speed_simple_R
        )

        # Toro 10.73
        pre_factor_L = (wave_speed_simple_L - primitives_L[self.velocity_ids[axis]]) / (wave_speed_simple_L - wave_speed_contact) * primitives_L[self.mass_ids]
        pre_factor_R = (wave_speed_simple_R - primitives_R[self.velocity_ids[axis]]) / (wave_speed_simple_R - wave_speed_contact) * primitives_R[self.mass_ids]

        u_star_L = [pre_factor_L, pre_factor_L, pre_factor_L, pre_factor_L, pre_factor_L * (conservatives_L[self.energy_ids] / conservatives_L[self.mass_ids] + (wave_speed_contact - primitives_L[self.velocity_ids[axis]]) * (wave_speed_contact + primitives_L[self.energy_ids] / primitives_L[self.mass_ids] / (wave_speed_simple_L - primitives_L[self.velocity_ids[axis]]) )) ]
        u_star_L[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_L[self.velocity_minor[axis][0]] *= primitives_L[self.velocity_minor[axis][0]]
        u_star_L[self.velocity_minor[axis][1]] *= primitives_L[self.velocity_minor[axis][1]]
        u_star_L = jnp.stack(u_star_L)

        u_star_R = [pre_factor_R, pre_factor_R, pre_factor_R, pre_factor_R, pre_factor_R * (conservatives_R[self.energy_ids] / conservatives_R[self.mass_ids] + (wave_speed_contact - primitives_R[self.velocity_ids[axis]]) * (wave_speed_contact + primitives_R[self.energy_ids] / primitives_R[self.mass_ids] / (wave_speed_simple_R - primitives_R[self.velocity_ids[axis]]) )) ]
        u_star_R[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_R[self.velocity_minor[axis][0]] *= primitives_R[self.velocity_minor[axis][0]]
        u_star_R[self.velocity_minor[axis][1]] *= primitives_R[self.velocity_minor[axis][1]]
        u_star_R = jnp.stack(u_star_R)

        # Fleischmann et al. - 2020 - Eq (23 - 25)
        phi = self._wave_speed_limiter(
            primitives_L[self.velocity_ids[axis]],
            primitives_R[self.velocity_ids[axis]],
            speed_of_sound_L,
            speed_of_sound_R)
        wave_speed_L = phi * wave_speed_simple_L
        wave_speed_R = phi * wave_speed_simple_R

        # Physical fluxes
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        # Fleischmann et al. - 2020 - Eq. (19)
        flux_star = 0.5 * (fluxes_L + fluxes_R) \
                    + 0.5 * (
                        wave_speed_L * (u_star_L - conservatives_L) \
                        + jnp.abs(wave_speed_contact) * (u_star_L - u_star_R) \
                        + wave_speed_R * (u_star_R - conservatives_R)
                        )

        # Fleischmann et al. - 2020 - Eq. (18)
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_simple_L)) * fluxes_L \
                    + 0.5 * (1 - jnp.sign(wave_speed_simple_R)) * fluxes_R \
                    + 0.25 * (1 - jnp.sign(wave_speed_simple_L)) * (1 + jnp.sign(wave_speed_simple_R)) * flux_star

        return fluxes_xi, None, None

    def _solve_riemann_problem_xi_diffuse_five_equation(
            self, 
            primitives_L: Array, 
            primitives_R: Array, 
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int, 
            curvature_L: Array,
            curvature_R: Array,
            **kwargs
        ) -> Tuple[Array, Array, Array]:
        """
        HLLC-LM approximate solution of the Riemann problem for the five-equation
        diffuse-interface model. Following Coralic & Colonius.
        """

        rho_L = self.material_manager.get_density(primitives_L)
        rho_R = self.material_manager.get_density(primitives_R)
        u_L = primitives_L[self.velocity_ids[axis]]
        u_R = primitives_R[self.velocity_ids[axis]]
        p_L = primitives_L[self.energy_ids]
        p_R = primitives_R[self.energy_ids]

        if self.is_surface_tension:
            kappa = 0.5 * (curvature_L + curvature_R)
            sigma = self.material_manager.get_sigma()
            alpha_L = primitives_L[self.vf_ids]
            alpha_R = primitives_R[self.vf_ids]
            p_L_prime = p_L - sigma * kappa * alpha_L
            p_R_prime = p_R - sigma * kappa * alpha_R
        else:
            p_L_prime = p_L
            p_R_prime = p_R

        speed_of_sound_L = self.material_manager.get_speed_of_sound(
            pressure=p_L, density=rho_L, volume_fractions=primitives_L[self.vf_slices])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(
            pressure=p_R, density=rho_R, volume_fractions=primitives_R[self.vf_slices])

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            u_L, u_R,
            speed_of_sound_L, speed_of_sound_R,
            rho_L=rho_L, rho_R=rho_R,
            p_L=p_L, p_R=p_R,
            gamma=None)
        
        wave_speed_contact = self.s_star(
            u_L, u_R,
            p_L_prime, p_R_prime,
            rho_L, rho_R,
            wave_speed_simple_L, wave_speed_simple_R,)

        # Toro 10.73
        pre_factor_L = (wave_speed_simple_L - u_L) / (wave_speed_simple_L - wave_speed_contact) 
        pre_factor_R = (wave_speed_simple_R - u_R) / (wave_speed_simple_R - wave_speed_contact)

        U_star_L = [
            *primitives_L[self.mass_slices],
            rho_L,
            rho_L,
            rho_L,
            conservatives_L[self.energy_ids] + (wave_speed_contact - u_L) * (rho_L * wave_speed_contact + p_L_prime / (wave_speed_simple_L - u_L) ),
            *primitives_L[self.vf_slices],
        ]
        U_star_L[self.velocity_ids[axis]] *= wave_speed_contact
        U_star_L[self.velocity_minor[axis][0]] *= primitives_L[self.velocity_minor[axis][0]]
        U_star_L[self.velocity_minor[axis][1]] *= primitives_L[self.velocity_minor[axis][1]]
        U_star_L = pre_factor_L * jnp.stack(U_star_L)

        U_star_R = [
            *primitives_R[self.mass_slices],
            rho_R,
            rho_R,
            rho_R,
            conservatives_R[self.energy_ids] + (wave_speed_contact - u_R) * (rho_R * wave_speed_contact + p_R_prime / (wave_speed_simple_R - u_R) ),
            *primitives_R[self.vf_slices],
        ]
        U_star_R[self.velocity_ids[axis]] *= wave_speed_contact
        U_star_R[self.velocity_minor[axis][0]] *= primitives_R[self.velocity_minor[axis][0]]
        U_star_R[self.velocity_minor[axis][1]] *= primitives_R[self.velocity_minor[axis][1]]
        U_star_R = pre_factor_R * jnp.stack(U_star_R)

        # Limit wave speeds
        phi = self._wave_speed_limiter(
            u_L,
            u_R,
            speed_of_sound_L,
            speed_of_sound_R)
        wave_speed_L = phi * wave_speed_simple_L
        wave_speed_R = phi * wave_speed_simple_R

        # Fleischmann et al. - 2020 - Eq (23 - 25)

        # Physical fluxes
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        flux_star = 0.5 * (fluxes_L + fluxes_R) \
            + 0.5 * (
                wave_speed_L * (U_star_L - conservatives_L) \
                + jnp.abs(wave_speed_contact) * (U_star_L - U_star_R) \
                + wave_speed_R * (U_star_R - conservatives_R)
                )

        # Fleischmann et al. - 2020 - Eq. (18)
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_simple_L)) * fluxes_L \
                    + 0.5 * (1 - jnp.sign(wave_speed_simple_R)) * fluxes_R \
                    + 0.25 * (1 - jnp.sign(wave_speed_simple_L)) * (1 + jnp.sign(wave_speed_simple_R)) * flux_star

        # u_star = 0.5 * (u_L + u_R) \
        #     + 0.5 * (
        #         wave_speed_L * (pre_factor_L - 1.0) \
        #         + jnp.abs(wave_speed_contact) * (pre_factor_L - pre_factor_R) \
        #         + wave_speed_R * (pre_factor_R - 1.0)
        #     )
        # u_hat = 0.5 * (1 + jnp.sign(wave_speed_simple_L)) * u_L \
        #     + 0.5 * (1 - jnp.sign(wave_speed_simple_R)) * u_R \
        #     + 0.25 * (1 - jnp.sign(wave_speed_simple_L)) * (1 + jnp.sign(wave_speed_simple_R)) * u_star
        u_hat = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * (u_L + wave_speed_L * (pre_factor_L - 1.0)) \
              + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * (u_R + wave_speed_R * (pre_factor_R - 1.0))

        if self.is_surface_tension:
            alpha_hat = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * alpha_L \
                + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * alpha_R
        else:
            alpha_hat = None

        return fluxes_xi, u_hat, alpha_hat