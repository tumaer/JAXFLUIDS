from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.equation_manager import EquationManager
# from jaxfluids.data_types.numerical_setup.conservatives import CATUMSetup

class CATUM(RiemannSolver):
    """CATUM Flux Function
    
    There are five different transport velocity formulations to choose from:
    
    1) EGERER
        Egerer et al. (2016) - Efficient implicit LES method
        for the simulation of turbulent cavitating flows
    
    2) SCHMIDT
        Schmidt (2006) - A low Mach number consistent
        compressible approach for simulation of cavitating flows
    
    3) SEZAL
        Sezal (2009) - Compressible Dynamics of Cavitating
        3-D Multi-Phase Flows
    
    4) MIHATSCH
        Mihatsch (2017) - Numerical Prediction of Erosion
        and Degassing Effects in CavitatingFlows
    
    5) KYRIAZIS
        Kyriazis (2017) - Numerical investigation of
        bubbledynamics using tabulated data

    :param RiemannSolver: _description_
    :type RiemannSolver: _type_
    """

    def __init__(
            self, 
            material_manager: MaterialManager, 
            equation_manager: EquationManager, 
            signal_speed: Callable,
            catum_setup,
            **kwargs
            ) -> None:
        super().__init__(material_manager, equation_manager, signal_speed)

        self.transport_velocity = catum_setup.transport_velocity
        self.speed_of_sound_min = catum_setup.minimum_speed_of_sound

    def _solve_riemann_problem_xi_single_phase(
            self,
            primitives_L: Array,
            primitives_R: Array,
            conservatives_L: Array,
            conservatives_R: Array,
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        rho_L = primitives_L[self.mass_ids]
        rho_R = primitives_R[self.mass_ids]
        p_L = primitives_L[self.energy_ids]
        p_R = primitives_R[self.energy_ids]
        u_xi_L = primitives_L[self.velocity_ids[axis]]
        u_xi_R = primitives_R[self.velocity_ids[axis]]

        H_L = rho_L * self.material_manager.get_total_enthalpy(
            p=p_L, 
            u=primitives_L[self.velocity_ids[0]], 
            v=primitives_L[self.velocity_ids[1]], 
            w=primitives_L[self.velocity_ids[2]], 
            rho=rho_L)
        H_R = rho_R * self.material_manager.get_total_enthalpy(
            p=p_R, 
            u=primitives_R[self.velocity_ids[0]], 
            v=primitives_R[self.velocity_ids[1]], 
            w=primitives_R[self.velocity_ids[2]], 
            rho=rho_R)

        q_h_L = conservatives_L.at[self.energy_ids].set(H_L)
        q_h_R = conservatives_R.at[self.energy_ids].set(H_R)

        speed_of_sound_L = self.material_manager.get_speed_of_sound(pressure=p_L, density=rho_L)
        speed_of_sound_R = self.material_manager.get_speed_of_sound(pressure=p_R, density=rho_R)
        speed_of_sound_liquid_L = self.material_manager.get_speed_of_sound_liquid(pressure=p_L, density=rho_L)
        speed_of_sound_liquid_R = self.material_manager.get_speed_of_sound_liquid(pressure=p_R, density=rho_R)

        p_star = 0.5 * (p_L + p_R)  # Schmidt Eq. (2.43)

        if self.transport_velocity == "EGERER":
            speed_of_sound_liquid_max = jnp.maximum(speed_of_sound_liquid_L, speed_of_sound_liquid_R)
            I_L = 0.25 * (3.0 * rho_L + rho_R) * speed_of_sound_liquid_max
            I_R = 0.25 * (rho_L + 3.0 * rho_R) * speed_of_sound_liquid_max
            u_star = (I_L * u_xi_L + I_R * u_xi_R + p_L - p_R) / (I_L + I_R)
            
        elif self.transport_velocity == "SCHMIDT":
            c_max = jnp.maximum(self.speed_of_sound_min, jnp.maximum(speed_of_sound_L, speed_of_sound_R))   # Eq. (2.37)
            rho_c_star = jnp.maximum(rho_L, rho_R) *  c_max     # Eq. (2.37)
            u_star = ((3.0 * rho_L + rho_R) * u_xi_L + (rho_L + 3.0 * rho_R) * u_xi_R) / (4.0 * (rho_L + rho_R)) \
                - (p_R - p_L) / (2*rho_c_star)  # Eq. (2.41)

            # v_vel = 0.5 * (conservatives_L[self.velocity_ids[axis]]/rho_L + conservatives_R[self.velocity_ids[axis]]/rho_R)   #2.38
            # v_mom = (conservatives_L[self.velocity_ids[axis]] + conservatives_R[self.velocity_ids[axis]]) / (rho_L + rho_R)   #2.39
            # xi = 0.5
            # u_star = xi * v_vel + (1.0 - xi) * v_mom - (p_R -p_L) / (2 *rho_c_star)   #2.40
        
        elif self.transport_velocity == "SEZAL":
            rho_c_L = rho_L * speed_of_sound_L
            rho_c_R = rho_R * speed_of_sound_R
            u_star = (rho_c_L * u_xi_L + rho_c_R * u_xi_R + p_L - p_R) / (rho_c_L + rho_c_R)    # Eq. (3.35)

        elif self.transport_velocity == "MIHATSCH":
            c_max = jnp.maximum(self.speed_of_sound_min, jnp.maximum(speed_of_sound_L, speed_of_sound_R))
            rho_R_n = 0.25 * (3.0 * rho_R + rho_L)
            rho_L_n = 0.25 * (3.0 * rho_L + rho_R)
            u_star = (rho_R_n * u_xi_R + rho_L_n * u_xi_L) / (rho_L_n + rho_R_n) - (p_R - p_L) / ((rho_L_n + rho_R_n) * c_max)

        elif self.transport_velocity == "KYRIAZIS":
            c_star = jnp.maximum(speed_of_sound_L, speed_of_sound_R)
            u_star = (rho_L * u_xi_L + rho_R * u_xi_R + (p_L - p_R) / c_star) / (rho_L + rho_R)

        else:
            raise NotImplementedError

        #advected_q_H = 0.5 * u_star * (q_h_L+q_h_R) - 0.5 * jnp.abs(u_star) * (q_h_R-q_h_L)    # Schmidt Eq. (2.42)
        advected_q_H = jnp.where(u_star > 0.0, q_h_L, q_h_R) * u_star                           # Schmidt Eq. (2.42)

        fluxes_xi = advected_q_H.at[self.velocity_ids[axis]].add(p_star)    # Schmidt Eq. (2.44)
        #fluxes_xi = 0.5 * u_star * (q_h_L + q_h_R) - 0.5 * jnp.abs(u_star) * (q_h_R-q_h_L) \
        #   + p_star * jnp.stack([0,unit_vector,0]) # Schmidt Eq. (2.44)
            
        return fluxes_xi, None, None

    def _solve_riemann_problem_xi_diffuse_five_equation(
            self,
            primitives_L: Array,
            primitives_R: Array,
            conservatives_L: Array,
            conservatives_R: Array,
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        raise NotImplementedError