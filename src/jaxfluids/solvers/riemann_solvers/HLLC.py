from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.solvers.riemann_solvers.signal_speeds import compute_sstar
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

class HLLC(RiemannSolver):
    """HLLC Riemann Solver
    Toro et al. 1994

    Supports:
    1) Single-phase / Two-phase level-set
    3) 5-Equation Diffuse-interface model

    For single-phase or two-phase level-set equations, the standard Riemann
    solver proposed by Toro is used. For diffuse-interface method, the HLLC
    modification as proposed by Coralic & Colonius with the surface-tension
    extension by Garrick is used.
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
            wave_speed_simple_R)

        wave_speed_L = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_R = jnp.maximum(wave_speed_simple_R, 0.0)

        # Toro 10.73
        pre_factor_L = (wave_speed_simple_L - primitives_L[self.velocity_ids[axis]]) / (wave_speed_simple_L - wave_speed_contact) * primitives_L[self.mass_ids]
        pre_factor_R = (wave_speed_simple_R - primitives_R[self.velocity_ids[axis]]) / (wave_speed_simple_R - wave_speed_contact) * primitives_R[self.mass_ids]

        # TODO check out performance with u_star_L = jnp.expand_dims(prefactor_L) / jnp.ones_like() 
        # to avoid list + jnp.stack
        u_star_L = [
            pre_factor_L,
            pre_factor_L,
            pre_factor_L,
            pre_factor_L,
            pre_factor_L * (conservatives_L[self.energy_ids] / conservatives_L[self.mass_ids] + (wave_speed_contact - primitives_L[self.velocity_ids[axis]]) * (wave_speed_contact + primitives_L[self.energy_ids] / primitives_L[self.mass_ids] / (wave_speed_simple_L - primitives_L[self.velocity_ids[axis]]) )) ]
        u_star_L[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_L[self.velocity_minor[axis][0]] *= primitives_L[self.velocity_minor[axis][0]]
        u_star_L[self.velocity_minor[axis][1]] *= primitives_L[self.velocity_minor[axis][1]]
        u_star_L = jnp.stack(u_star_L)

        u_star_R = [
            pre_factor_R,
            pre_factor_R,
            pre_factor_R,
            pre_factor_R,
            pre_factor_R * (conservatives_R[self.energy_ids] / conservatives_R[self.mass_ids] + (wave_speed_contact - primitives_R[self.velocity_ids[axis]]) * (wave_speed_contact + primitives_R[self.energy_ids] / primitives_R[self.mass_ids] / (wave_speed_simple_R - primitives_R[self.velocity_ids[axis]]) )) ]
        u_star_R[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_R[self.velocity_minor[axis][0]] *= primitives_R[self.velocity_minor[axis][0]]
        u_star_R[self.velocity_minor[axis][1]] *= primitives_R[self.velocity_minor[axis][1]]
        u_star_R = jnp.stack(u_star_R)

        # Phyiscal fluxes
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        # Toro 10.72
        flux_star_L = fluxes_L + wave_speed_L * (u_star_L - conservatives_L)
        flux_star_R = fluxes_R + wave_speed_R * (u_star_R - conservatives_R)

        # Kind of Toro 10.71
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1 - jnp.sign(wave_speed_contact)) * flux_star_R
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
        HLLC approximate solution of the Riemann problem for the five-equation
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

        wave_speed_L = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_R = jnp.maximum(wave_speed_simple_R, 0.0)

        # Toro 10.73
        pre_factor_L = (wave_speed_simple_L - u_L) / (wave_speed_simple_L - wave_speed_contact) 
        pre_factor_R = (wave_speed_simple_R - u_R) / (wave_speed_simple_R - wave_speed_contact)

        u_star_L = [
            *primitives_L[self.mass_slices],
            rho_L,
            rho_L,
            rho_L,
            conservatives_L[self.energy_ids] + (wave_speed_contact - u_L) * (rho_L * wave_speed_contact + p_L_prime / (wave_speed_simple_L - u_L) ),
            *primitives_L[self.vf_slices],
        ]
        u_star_L[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_L[self.velocity_minor[axis][0]] *= primitives_L[self.velocity_minor[axis][0]]
        u_star_L[self.velocity_minor[axis][1]] *= primitives_L[self.velocity_minor[axis][1]]
        u_star_L = pre_factor_L * jnp.stack(u_star_L)

        u_star_R = [
            *primitives_R[self.mass_slices],
            rho_R,
            rho_R,
            rho_R,
            conservatives_R[self.energy_ids] + (wave_speed_contact - u_R) * (rho_R * wave_speed_contact + p_R_prime / (wave_speed_simple_R - u_R) ),
            *primitives_R[self.vf_slices],
        ]
        u_star_R[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_R[self.velocity_minor[axis][0]] *= primitives_R[self.velocity_minor[axis][0]]
        u_star_R[self.velocity_minor[axis][1]] *= primitives_R[self.velocity_minor[axis][1]]
        u_star_R = pre_factor_R * jnp.stack(u_star_R)

        # Physical fluxes
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        # Toro 10.72
        flux_star_L = fluxes_L + wave_speed_L * (u_star_L - conservatives_L)
        flux_star_R = fluxes_R + wave_speed_R * (u_star_R - conservatives_R)

        # Kind of Toro 10.71
        fluxes_xi = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * flux_star_R

        u_hat = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * (u_L + wave_speed_L * (pre_factor_L - 1.0)) \
              + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * (u_R + wave_speed_R * (pre_factor_R - 1.0))
        
        if self.is_surface_tension:
            alpha_hat = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * alpha_L \
                      + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * alpha_R
        else:
            alpha_hat = None
        
        # NUMERICAL DISSIPATION
        # numerical_dissipation = primitives_L[-1] * (jnp.abs(wave_speed_simple_L) * (wave_speed_contact - u_L) \
        #     - jnp.abs(wave_speed_contact) * (wave_speed_simple_L - u_L)) / (wave_speed_simple_L - wave_speed_contact) \
        #     + primitives_R[-1] * (jnp.abs(wave_speed_contact) * (wave_speed_simple_R - u_R) \
        #     + jnp.abs(wave_speed_contact) * (u_R - wave_speed_contact)) / (wave_speed_simple_R - wave_speed_contact)
        # numerical_dissipation = jnp.abs(0.5 * numerical_dissipation)

        return fluxes_xi, u_hat, alpha_hat