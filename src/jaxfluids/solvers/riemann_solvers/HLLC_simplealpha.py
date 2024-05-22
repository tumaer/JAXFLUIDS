from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.solvers.riemann_solvers.signal_speeds import compute_sstar
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

class HLLC_SIMPLEALPHA(RiemannSolver):
    """HLLC Riemann Solver
    Toro et al. 1994
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
        )  -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        raise NotImplementedError

    def _solve_riemann_problem_xi_diffuse_five_equation(
            self,
            primitives_L: Array,
            primitives_R: Array, 
            conservatives_L: Array,
            conservatives_R: Array,
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """
        HLLC approximate solution of the Riemann problem for the diffuse-interface model. 
        NOTE: treatment of volume fraction equation is according to Wong et al. 2021.
        alpha flux is a simple upwinding switch, transport velocity is s_star. 
        Remainder following Coralic & Colonius.
        """

        rho_L = self.material_manager.get_density(primitives_L)
        rho_R = self.material_manager.get_density(primitives_R)

        u_L   = primitives_L[self.velocity_ids[axis]]
        u_R   = primitives_R[self.velocity_ids[axis]]

        p_L   = primitives_L[self.energy_ids]
        p_R   = primitives_R[self.energy_ids]

        fluxes_L  = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        speed_of_sound_L = self.material_manager.get_speed_of_sound(p = p_L, rho = rho_L, alpha_i = primitives_L[self.vf_slices])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(p = p_R, rho = rho_R, alpha_i = primitives_R[self.vf_slices])

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            u_L, 
            u_R, 
            speed_of_sound_L, 
            speed_of_sound_R, 
            rho_L = rho_L, 
            rho_R = rho_R, 
            p_L = p_L, 
            p_R = p_R, 
            gamma = None
        )
        wave_speed_contact = self.s_star(
            u_L, 
            u_R, 
            p_L, 
            p_R, 
            rho_L, 
            rho_R,
            wave_speed_simple_L, 
            wave_speed_simple_R
        )

        wave_speed_L  = jnp.minimum( wave_speed_simple_L, 0.0 )
        wave_speed_R = jnp.maximum( wave_speed_simple_R, 0.0 )

        ''' Toro 10.73 '''
        pre_factor_L = (wave_speed_simple_L - u_L) / (wave_speed_simple_L - wave_speed_contact) 
        pre_factor_R = (wave_speed_simple_R - u_R) / (wave_speed_simple_R - wave_speed_contact)

        u_star_L = [
            *primitives_L[self.mass_slices], 
            rho_L, 
            rho_L, 
            rho_L, 
            conservatives_L[self.energy_ids] + (wave_speed_contact - u_L) * (rho_L * wave_speed_contact + p_L / (wave_speed_simple_L - u_L) ),
            *primitives_L[self.vf_slices], 
        ]
        u_star_L[self.velocity_ids[axis]]      *= wave_speed_contact
        u_star_L[self.velocity_minor[axis][0]] *= primitives_L[self.velocity_minor[axis][0]]
        u_star_L[self.velocity_minor[axis][1]] *= primitives_L[self.velocity_minor[axis][1]]
        u_star_L = pre_factor_L * jnp.stack(u_star_L)

        u_star_R = [
            *primitives_R[self.mass_slices], 
            rho_R, 
            rho_R, 
            rho_R, 
            conservatives_R[self.energy_ids] + (wave_speed_contact - u_R) * (rho_R * wave_speed_contact + p_R / (wave_speed_simple_R - u_R) ),
            *primitives_R[self.vf_slices], 
        ]
        u_star_R[self.velocity_ids[axis]]      *= wave_speed_contact
        u_star_R[self.velocity_minor[axis][0]] *= primitives_R[self.velocity_minor[axis][0]]
        u_star_R[self.velocity_minor[axis][1]] *= primitives_R[self.velocity_minor[axis][1]]
        u_star_R = pre_factor_R * jnp.stack(u_star_R)

        ''' Toro 10.72 '''
        flux_star_L = fluxes_L + wave_speed_L * (u_star_L - conservatives_L)
        flux_star_R = fluxes_R + wave_speed_R * (u_star_R - conservatives_R)

        flux_star_L = flux_star_L.at[self.vf_slices].set(wave_speed_contact * conservatives_L[self.vf_slices])
        flux_star_R = flux_star_R.at[self.vf_slices].set(wave_speed_contact * conservatives_R[self.vf_slices])

        ''' Kind of Toro 10.71 '''
        fluxes_xi = 0.5 * (1.0 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1.0 - jnp.sign(wave_speed_contact)) * flux_star_R

        u_hat = wave_speed_contact
        
        return fluxes_xi, u_hat, None