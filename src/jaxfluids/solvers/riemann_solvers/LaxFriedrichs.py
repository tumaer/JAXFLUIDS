from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.equation_manager import EquationManager

class LaxFriedrichs(RiemannSolver):

    def __init__(
            self,
            material_manager: MaterialManager, 
            equation_manager: EquationManager,
            signal_speed: Callable,
            **kwargs
            ) -> None:
        super().__init__(material_manager, equation_manager, signal_speed)

    def _solve_riemann_problem_xi_single_phase(
            self, 
            primitives_L: Array,
            primitives_R: Array, 
            conservatives_L: Array,
            conservatives_R: Array, 
            axis: int,
            **kwargs
            ) -> Tuple[Array, Array, Array]:

        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        speed_of_sound_L = self.material_manager.get_speed_of_sound(primitives_L)
        speed_of_sound_R = self.material_manager.get_speed_of_sound(primitives_R)
        
        alpha = jnp.maximum(
            jnp.max(jnp.abs(primitives_L[self.velocity_ids[axis]]) + speed_of_sound_L), 
            jnp.max(jnp.abs(primitives_R[self.velocity_ids[axis]]) + speed_of_sound_R))

        fluxes_xi = 0.5 * (fluxes_L + fluxes_R) - 0.5 * alpha * (conservatives_R - conservatives_L)
            
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
        rho_L = self.material_manager.get_density(primitives_L)
        rho_R = self.material_manager.get_density(primitives_R)
        u_L = primitives_L[self.velocity_ids[axis]]
        u_R = primitives_R[self.velocity_ids[axis]]
        p_L = primitives_L[self.energy_ids]
        p_R = primitives_R[self.energy_ids]

        speed_of_sound_L = self.material_manager.get_speed_of_sound(
            pressure=p_L, density=rho_L, volume_fractions=primitives_L[self.vf_slices])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(
            pressure=p_R, density=rho_R, volume_fractions=primitives_R[self.vf_slices])

        alpha = jnp.maximum(
            jnp.max(jnp.abs(u_L) + speed_of_sound_L), 
            jnp.max(jnp.abs(u_R) + speed_of_sound_R))

        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        fluxes_xi = 0.5 * (fluxes_L + fluxes_R) - 0.5 * alpha * (conservatives_R - conservatives_L)

        u_hat = 0.5 * (u_L + u_R)

        if self.is_surface_tension:
            raise NotImplementedError
        else:
            alpha_hat = None

        return fluxes_xi, u_hat, alpha_hat