from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.solvers.riemann_solvers.signal_speeds import compute_sstar
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

Array = jax.Array

class HLLC(RiemannSolver):
    """HLLC Riemann Solver
    Toro et al. 1994

    Supports:
    1) Single-phase / Two-phase level-set
    2) 4-Equation Diffuse-interface model
    3) 5-Equation Diffuse-interface model
    4) Multi-component Navier-Stokes

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


    def _compute_flux_star_K_single_phase(
            self,
            primitives_K: Array,
            conservatives_K: Array,
            wave_speed_simple_K: Array,
            wave_speed_contact: Array,
            axis: int,
            left_right_id: str
        ) -> Array:

        # Toro 10.73
        pre_factor_K = (wave_speed_simple_K - primitives_K[self.ids_velocity[axis]]) / (wave_speed_simple_K - wave_speed_contact) * primitives_K[self.ids_mass]

        u_star_K = [
            pre_factor_K,
            pre_factor_K,
            pre_factor_K,
            pre_factor_K,
            pre_factor_K * (conservatives_K[self.ids_energy] / conservatives_K[self.ids_mass] + (wave_speed_contact - primitives_K[self.ids_velocity[axis]]) * (wave_speed_contact + primitives_K[self.ids_energy] / primitives_K[self.ids_mass] / (wave_speed_simple_K - primitives_K[self.ids_velocity[axis]]) )) 
        ]
        u_star_K[self.ids_velocity[axis]] *= wave_speed_contact
        u_star_K[self.velocity_minor[axis][0]] *= primitives_K[self.velocity_minor[axis][0]]
        u_star_K[self.velocity_minor[axis][1]] *= primitives_K[self.velocity_minor[axis][1]]
        u_star_K = jnp.stack(u_star_K)

        fluxes_K = self.equation_manager.get_fluxes_xi(primitives_K, conservatives_K, axis)

        # Toro 10.72
        if left_right_id == "L":
            wave_speed_K = jnp.minimum(wave_speed_simple_K, 0.0)
        elif left_right_id == "R":
            wave_speed_K = jnp.maximum(wave_speed_simple_K, 0.0)
        else:
            raise NotImplementedError

        flux_star_K = fluxes_K + wave_speed_K * (u_star_K - conservatives_K)

        return flux_star_K

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
            primitives_L[self.ids_velocity[axis]],
            primitives_R[self.ids_velocity[axis]],
            speed_of_sound_L,
            speed_of_sound_R,
            rho_L=primitives_L[self.ids_mass],
            rho_R=primitives_R[self.ids_mass],
            p_L=primitives_L[self.ids_energy],
            p_R=primitives_R[self.ids_energy],
            gamma=self.material_manager.get_gamma())

        wave_speed_contact = self.s_star(
            primitives_L[self.ids_velocity[axis]],
            primitives_R[self.ids_velocity[axis]],
            primitives_L[self.ids_energy],
            primitives_R[self.ids_energy],
            primitives_L[self.ids_mass],
            primitives_R[self.ids_mass],
            wave_speed_simple_L, 
            wave_speed_simple_R)

        flux_star_L = self._compute_flux_star_K_single_phase(
            primitives_L, conservatives_L, wave_speed_simple_L,
            wave_speed_contact, axis, "L")

        flux_star_R = self._compute_flux_star_K_single_phase(
            primitives_R, conservatives_R, wave_speed_simple_R,
            wave_speed_contact, axis, "R")

        # Kind of Toro 10.71
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1 - jnp.sign(wave_speed_contact)) * flux_star_R

        return fluxes_xi, None, None


    def _compute_flux_star_K_5eqm(
            self,
            primitives_K: Array,
            conservatives_K: Array,
            rho_K: Array,
            p_K_prime: Array,
            wave_speed_simple_K: Array,
            wave_speed_contact: Array,
            axis: int,
            left_right_id: str
        ) -> Array:
        velocity_major_K = primitives_K[self.ids_velocity[axis]]

        pre_factor_K = (wave_speed_simple_K - velocity_major_K) / (wave_speed_simple_K - wave_speed_contact)

        u_star_K = [
            *primitives_K[self.s_mass],
            rho_K,
            rho_K,
            rho_K,
            conservatives_K[self.ids_energy] + (wave_speed_contact - velocity_major_K) * (rho_K * wave_speed_contact + p_K_prime / (wave_speed_simple_K - velocity_major_K)),
            *primitives_K[self.s_volume_fraction],
        ]
        u_star_K[self.ids_velocity[axis]] *= wave_speed_contact
        u_star_K[self.velocity_minor[axis][0]] *= primitives_K[self.velocity_minor[axis][0]]
        u_star_K[self.velocity_minor[axis][1]] *= primitives_K[self.velocity_minor[axis][1]]
        u_star_K = pre_factor_K * jnp.stack(u_star_K)

        # Toro 10.72
        if left_right_id == "L":
            wave_speed_K = jnp.minimum(wave_speed_simple_K, 0.0)
        elif left_right_id == "R":
            wave_speed_K = jnp.maximum(wave_speed_simple_K, 0.0)
        else:
            raise NotImplementedError

        fluxes_K = self.equation_manager.get_fluxes_xi(primitives_K, conservatives_K, axis)

        flux_star_K = fluxes_K + wave_speed_K * (u_star_K - conservatives_K)
        velocity_star_K = velocity_major_K + wave_speed_K * (pre_factor_K - 1.0)

        return flux_star_K, velocity_star_K

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
        u_L = primitives_L[self.ids_velocity[axis]]
        u_R = primitives_R[self.ids_velocity[axis]]
        p_L = primitives_L[self.ids_energy]
        p_R = primitives_R[self.ids_energy]
        
        if self.is_surface_tension:
            kappa = 0.5 * (curvature_L + curvature_R)
            sigma_kappa = self.material_manager.get_sigma() * kappa
            alpha_L = primitives_L[self.ids_volume_fraction]
            alpha_R = primitives_R[self.ids_volume_fraction]
            p_L_prime = p_L - sigma_kappa * alpha_L
            p_R_prime = p_R - sigma_kappa * alpha_R
        else:
            p_L_prime = p_L
            p_R_prime = p_R

        speed_of_sound_L = self.material_manager.get_speed_of_sound(
            pressure=p_L, density=rho_L, volume_fractions=primitives_L[self.s_volume_fraction])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(
            pressure=p_R, density=rho_R, volume_fractions=primitives_R[self.s_volume_fraction])
    
        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            u_L, u_R, speed_of_sound_L, speed_of_sound_R,
            rho_L=rho_L, rho_R=rho_R, p_L=p_L, p_R=p_R,
            gamma=None)
        
        wave_speed_contact = self.s_star(
            u_L, u_R, p_L_prime, p_R_prime, rho_L, rho_R, 
            wave_speed_simple_L, wave_speed_simple_R)

        flux_star_L, velocity_star_L = self._compute_flux_star_K_5eqm(
            primitives_L, conservatives_L, rho_L, p_L_prime,
            wave_speed_simple_L, wave_speed_contact, axis, "L")
        flux_star_R, velocity_star_R = self._compute_flux_star_K_5eqm(
            primitives_R, conservatives_R, rho_R, p_R_prime,
            wave_speed_simple_R, wave_speed_contact, axis, "R")
        
        # Kind of Toro 10.71
        fluxes_xi = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * flux_star_R

        u_hat = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * velocity_star_L \
              + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * velocity_star_R
        
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


    def _compute_flux_star_K_4eqm(
            self,
            primitives_K: Array,
            conservatives_K: Array,
            rho_K: Array,
            p_K_prime: Array,
            wave_speed_simple_K: Array,
            wave_speed_contact: Array,
            axis: int,
            left_right_id: str
        ) -> Tuple[Array, Array]:
        velocity_major_K = primitives_K[self.ids_velocity[axis]]

        pre_factor_K = (wave_speed_simple_K - velocity_major_K) / (wave_speed_simple_K - wave_speed_contact) 

        u_star_K = [
            *primitives_K[self.s_mass],
            rho_K,
            rho_K,
            rho_K,
            conservatives_K[self.ids_energy] + (wave_speed_contact - velocity_major_K) * (rho_K * wave_speed_contact + p_K_prime / (wave_speed_simple_K - velocity_major_K) ),
        ]
        u_star_K[self.ids_velocity[axis]] *= wave_speed_contact
        u_star_K[self.velocity_minor[axis][0]] *= primitives_K[self.velocity_minor[axis][0]]
        u_star_K[self.velocity_minor[axis][1]] *= primitives_K[self.velocity_minor[axis][1]]
        u_star_K = pre_factor_K * jnp.stack(u_star_K)

        if left_right_id == "L":
            wave_speed_K = jnp.minimum(wave_speed_simple_K, 0.0)
        elif left_right_id == "R":
            wave_speed_K = jnp.maximum(wave_speed_simple_K, 0.0)
        else:
            raise NotImplementedError

        fluxes_K = self.equation_manager.get_fluxes_xi(primitives_K, conservatives_K, axis)

        flux_star_K = fluxes_K + wave_speed_K * (u_star_K - conservatives_K)
        velocity_star_K = velocity_major_K + wave_speed_K * (pre_factor_K - 1.0)

        return flux_star_K, velocity_star_K

    def _solve_riemann_problem_xi_diffuse_four_equation(
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
        HLLC approximate solution of the Riemann problem for the diffuse-interface model.
        Following Coralic & Colonius.
        """
        rho_L = self.material_manager.get_density(primitives_L)
        rho_R = self.material_manager.get_density(primitives_R)
        u_L = primitives_L[self.ids_velocity[axis]]
        u_R = primitives_R[self.ids_velocity[axis]]
        p_L = primitives_L[self.ids_energy]
        p_R = primitives_R[self.ids_energy]
        
        if self.is_surface_tension:
            #TODO 4EQM
            raise NotImplementedError
            kappa = 0.5 * (curvature_L + curvature_R)
            sigma = self.material_manager.get_sigma()
            alpha_L = primitives_L[self.ids_volume_fraction]
            alpha_R = primitives_R[self.ids_volume_fraction]
            p_L_prime = p_L - sigma * kappa * alpha_L
            p_R_prime = p_R - sigma * kappa * alpha_R
        else:
            p_L_prime = p_L
            p_R_prime = p_R

        speed_of_sound_L = self.material_manager.get_speed_of_sound(
            pressure=p_L, partial_densities=primitives_L[self.s_mass])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(
            pressure=p_R, partial_densities=primitives_R[self.s_mass])
    
        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            u_L, u_R, speed_of_sound_L, speed_of_sound_R,
            rho_L=rho_L, rho_R=rho_R, p_L=p_L, p_R=p_R,
            gamma=None)
        
        wave_speed_contact = self.s_star(
            u_L, u_R, p_L_prime, p_R_prime, rho_L, rho_R, 
            wave_speed_simple_L, wave_speed_simple_R,)

        flux_star_L, velocity_star_L = self._compute_flux_star_K_5eqm(
            primitives_L, conservatives_L, rho_L, p_L_prime,
            wave_speed_simple_L, wave_speed_contact, axis, "L")
        flux_star_R, velocity_star_R = self._compute_flux_star_K_5eqm(
            primitives_R, conservatives_R, rho_R, p_R_prime,
            wave_speed_simple_R, wave_speed_contact, axis, "R")

        # Kind of Toro 10.71
        fluxes_xi = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * flux_star_R

        u_hat = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * velocity_star_L \
              + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * velocity_star_R

        return fluxes_xi, u_hat, None
