from __future__ import annotations
from typing import Tuple, Dict, Union, TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.shock_sensor.ducros import Ducros
from jaxfluids.iles.ALDM_WENO1 import ALDM_WENO1
from jaxfluids.iles.ALDM_WENO3 import ALDM_WENO3
from jaxfluids.iles.ALDM_WENO5 import ALDM_WENO5
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup

class ALDM(ConvectiveFluxSolver):
    """ Adaptive Local Deconvolution Method - ALDM - Hickel et al. 2014

    ALDM is a numerical scheme for computation of convective fluxes. It consits
    of a combined reconstruction and flux-function. ALDM is optimized to model
    subgrid-scale terms in underresolved LES. 

    ALDM consists of a  
    1) cell face reconstruction based on a convex sum of 
    adapted WENO1, WENO3, and WENO5 
    
    2) flux-function with adjusted dissipation of SGS modeling 
    and low Mach number consistency.
    """

    def __init__(
            self,
            convective_fluxes_setup: ConvectiveFluxesSetup,
            material_manager: MaterialManager,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            **kwargs
            ) -> None:

        super(ALDM, self).__init__(
            convective_fluxes_setup, material_manager, domain_information, equation_manager)

        self._sigma_rho = 0.615
        self._sigma_rhou = 0.125
        self._sigma_rhoe = 0.615

        # TODO update stencils for adaptive meshes
        nh_conservatives = domain_information.nh_conservatives
        inactive_axes = domain_information.inactive_axes
        is_mesh_stretching = domain_information.is_mesh_stretching
        cell_sizes_halos = domain_information.get_global_cell_sizes_halos()

        # ILES SETUP
        iles_setup = convective_fluxes_setup.iles_setup
        smoothness_measure = iles_setup.aldm_smoothness_measure
        wall_damping = iles_setup.wall_damping
        shock_sensor = iles_setup.shock_sensor

        # STENCILS
        self.ALDM_WENO1 = ALDM_WENO1(
            nh=nh_conservatives,
            inactive_axes=inactive_axes,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_halos,
            smoothness_measure=smoothness_measure)
        self.ALDM_WENO3 = ALDM_WENO3(
            nh=nh_conservatives,
            inactive_axes=inactive_axes,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_halos,
            smoothness_measure=smoothness_measure)
        self.ALDM_WENO5 = ALDM_WENO5(
            nh=nh_conservatives,
            inactive_axes=inactive_axes,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_halos,
            smoothness_measure=smoothness_measure)

        # WALL DAMPING
        self.wall_damping = wall_damping
        self.vd_constant_d = 3.0
        self.vd_constant_s = 1.0 / self.vd_constant_d
        self.vd_constant_a_plus = 50.0

        # SHOCK SENSOR
        if shock_sensor == "DUCROS":
            self.shock_sensor = Ducros(domain_information)
            self.is_shock_sensor_active = True
        else:
            self.is_shock_sensor_active = False

        self.is_central_formulation = False

    def compute_flux_xi(
            self, 
            primitives: Array,
            conservatives: Array,
            axis: int,
            **kwargs
            ) -> Tuple[Array, None, None, None]:
        """Computes the numerical flux in the axis direction.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param conservatives: Buffer of conservative variables.
        :type conservatives: Array
        :param axis: Spatial direction in which the flux is computed.
        :type axis: int
        :return: Numerical flux in specified direction.
        :rtype: Array
        """
        # SHOCK SENSOR
        if self.is_shock_sensor_active:
            fs = self.shock_sensor.compute_sensor_function(primitives[1:4], axis)
        else:
            fs = 0.0

        # Solution adaptive alpha parameters
        alpha_1 = (1.0 - fs) / 3.0
        alpha_2 = (1.0 - fs) / 3.0
        alpha_3 = 1.0 - alpha_1 - alpha_2

        # CELL FACE RECONSTRUCTION
        phi = self.compute_phi(primitives, conservatives)
        phi_L, p3_L = self.reconstruct_xi(phi, alpha_1, alpha_2, alpha_3, fs, axis, 0)
        phi_R, p3_R = self.reconstruct_xi(phi, alpha_1, alpha_2, alpha_3, fs, axis, 1)

        # Speed of sound
        speed_of_sound = self.material_manager.get_speed_of_sound(primitives)
        speed_of_sound_L = self.ALDM_WENO1.reconstruct_xi(speed_of_sound, axis, 0)
        speed_of_sound_R = self.ALDM_WENO1.reconstruct_xi(speed_of_sound, axis, 1)

        # WALL CORRECTION
        # TODO

        # NUMERICAL FLUX EVALUATION
        if self.is_central_formulation:
            return self.solve_riemann_problem_xi_central_formulation(
                phi_L, phi_R, p3_L, p3_R, alpha_3, fs, axis), None, None, None

        fluxes_xi = self.solve_riemann_problem_xi(
            phi_L, phi_R, p3_L, p3_R, alpha_3,
            speed_of_sound_L, speed_of_sound_R, fs, axis)
        return fluxes_xi, None, None, None, None

    def solve_riemann_problem_xi(
            self,
            phi_L: Array,
            phi_R: Array,
            p3_L: Array,
            p3_R: Array,
            alpha_3: Array,
            speed_of_sound_L: Array,
            speed_of_sound_R: Array,
            fs: Array,
            axis: int
        ) -> Array:
        """Solves the Riemann problem, i.e., calculates the numerical flux,
        in the direction specified by axis.

        phi = [rho, u1, u2, u3, p, rho_e]
        p3_K is third-order pressure reconstruction

        :param phi_L: Phi vector of left neighboring state
        :type phi_L: Array
        :param phi_R: Phi vector of right neighboring state
        :type phi_R: Array
        :param p3_L: Third-order pressure reconstruction of left neighboring state
        :type p3_L: Array
        :param p3_R: Third-order pressure reconstruction of right neighboring state
        :type p3_R: Array
        :param alpha_3: Third-order reconstruction weight
        :type alpha_3: Array
        :param fs: Shock sensor.
        :type fs: Array
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis drection.
        :rtype: Array
        """
        
        phi_delta = phi_R - phi_L

        # Interface pressure and transport velocity
        # Eq. (34a)
        p_star = 0.5 * (phi_L[4] + phi_R[4])

        # speed_of_sound_L = self.material_manager.get_speed_of_sound(
        #     pressure=phi_L[4], density=phi_L[0])
        # speed_of_sound_R = self.material_manager.get_speed_of_sound(
        #     pressure=phi_R[4], density=phi_R[0])
        speed_of_sound = jnp.maximum(speed_of_sound_L, speed_of_sound_R)
        # Eq. (34b)
        u_star = 0.5 * (phi_L[axis+1] + phi_R[axis+1]) - alpha_3 * (p3_R - p3_L) / (speed_of_sound * (phi_L[0] + phi_R[0])) 

        # Dissipation matrix
        R_diss = jnp.stack([
            self._sigma_rho  * jnp.abs(phi_delta[axis+1]) + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhou * jnp.abs(phi_delta[1])      + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhou * jnp.abs(phi_delta[2])      + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhou * jnp.abs(phi_delta[3])      + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhoe * jnp.abs(phi_delta[axis+1]) + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
        ])

        # Flux computation
        flux_rho  = u_star * 0.5 * (phi_R[0] + phi_L[0]) - R_diss[0] * (phi_R[0] - phi_L[0])
        flux_ui   = [
            flux_rho * 0.5 * (phi_R[1] + phi_L[1]) - R_diss[1] * 0.5 * (phi_R[0] + phi_L[0]) * (phi_R[1] - phi_L[1]),
            flux_rho * 0.5 * (phi_R[2] + phi_L[2]) - R_diss[2] * 0.5 * (phi_R[0] + phi_L[0]) * (phi_R[2] - phi_L[2]),
            flux_rho * 0.5 * (phi_R[3] + phi_L[3]) - R_diss[3] * 0.5 * (phi_R[0] + phi_L[0]) * (phi_R[3] - phi_L[3]),
        ]
        flux_rhoe = u_star * 0.5 * (phi_R[5] + phi_L[5]) \
            + 0.5 * (phi_R[1] + phi_L[1]) * (flux_ui[0] - 0.25 * (phi_R[1] + phi_L[1]) * flux_rho) \
            + 0.5 * (phi_R[2] + phi_L[2]) * (flux_ui[1] - 0.25 * (phi_R[2] + phi_L[2]) * flux_rho) \
            + 0.5 * (phi_R[3] + phi_L[3]) * (flux_ui[2] - 0.25 * (phi_R[3] + phi_L[3]) * flux_rho) \
            - R_diss[4] * (phi_R[5] - phi_L[5])

        fluxes_xi = [flux_rho, flux_ui[0], flux_ui[1], flux_ui[2], flux_rhoe]
        
        # Add pressure flux
        fluxes_xi[axis+1] += p_star
        fluxes_xi[4] += u_star * p_star

        return jnp.stack(fluxes_xi)

    def solve_riemann_problem_xi_central_formulation(
            self,
            phi_L: Array,
            phi_R: Array,
            p3_L: Array,
            p3_R: Array,
            alpha_3: Array,
            fs: Array,
            axis: int
        ) -> Array:
        """Solves the Riemann problem, i.e., calculates the numerical flux,
        in the direction specified by axis.

        phi = [rho, u1, u2, u3, p, rho_e]
        p3_K is third-order pressure reconstruction

        :param phi_L: Phi vector of left neighboring state
        :type phi_L: Array
        :param phi_R: Phi vector of right neighboring state
        :type phi_R: Array
        :param p3_L: Third-order pressure reconstruction of left neighboring state
        :type p3_L: Array
        :param p3_R: Third-order pressure reconstruction of right neighboring state
        :type p3_R: Array
        :param alpha_3: Third-order reconstruction weight
        :type alpha_3: Array
        :param fs: Shock sensor.
        :type fs: Array
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis drection.
        :rtype: Array
        """
        phi_mean = 0.5 * (phi_L + phi_R)
        phi_delta = phi_R - phi_L
        u_mean = phi_mean[axis+1]
        p_mean = phi_mean[4]
        tke_mean = 0.5 * (
            phi_mean[1]*phi_mean[1] \
            + phi_mean[2]*phi_mean[2] \
            + phi_mean[3]*phi_mean[3])

        speed_of_sound_L = self.material_manager.get_speed_of_sound(
            pressure=phi_L[4], density=phi_L[0])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(
            pressure=phi_R[4], density=phi_R[0])
        c = jnp.maximum(speed_of_sound_L, speed_of_sound_R)

        # Central Flux
        F_rho = u_mean * phi_mean[0]
        F_rhou1 = u_mean * phi_mean[0] * phi_mean[1]
        F_rhou2 = u_mean * phi_mean[0] * phi_mean[2]
        F_rhou3 = u_mean * phi_mean[0] * phi_mean[3]
        F_rhoe = u_mean * (phi_mean[5] + tke_mean * phi_mean[0])
        fluxes_central = [F_rho, F_rhou1, F_rhou2, F_rhou3, F_rhoe]
        fluxes_central[axis+1] += p_mean
        fluxes_central[4] += u_mean * p_mean
        fluxes_central = jnp.array(fluxes_central)

        # Convective dissipation
        R_diss_rho = self._sigma_rho * jnp.abs(phi_delta[axis+1]) * phi_delta[0]
        R_diss_rhou1 = self._sigma_rho * jnp.abs(phi_delta[axis+1]) * phi_mean[1] * phi_delta[0] \
            + self._sigma_rhou * jnp.abs(phi_delta[1]) * phi_mean[0] * phi_delta[1]
        R_diss_rhou2 = self._sigma_rho * jnp.abs(phi_delta[axis+1]) * phi_mean[2] * phi_delta[0] \
            + self._sigma_rhou * jnp.abs(phi_delta[2]) * phi_mean[0] * phi_delta[2]
        R_diss_rhou3 = self._sigma_rho * jnp.abs(phi_delta[axis+1]) * phi_mean[3] * phi_delta[0] \
            + self._sigma_rhou * jnp.abs(phi_delta[3]) * phi_mean[0] * phi_delta[3]
        R_diss_rhoe = self._sigma_rho * jnp.abs(phi_delta[axis+1]) * tke_mean * phi_delta[0] \
            + self._sigma_rhou * phi_mean[0] * ( jnp.abs(phi_delta[1]) * phi_mean[1] * phi_delta[1] \
                + jnp.abs(phi_delta[2]) * phi_mean[2] * phi_delta[2] \
                + jnp.abs(phi_delta[3]) * phi_mean[3] * phi_delta[3] ) \
            + self._sigma_rhoe * jnp.abs(phi_delta[axis+1]) * phi_delta[5]

        R_diss = jnp.stack([R_diss_rho, R_diss_rhou1, R_diss_rhou2, R_diss_rhou3, R_diss_rhoe])

        # Pressure dissipation
        R_diss_p_rho = phi_mean[0]
        R_diss_p_rhou1 = phi_mean[0] * phi_mean[1]
        R_diss_p_rhou2 = phi_mean[0] * phi_mean[2]
        R_diss_p_rhou3 = phi_mean[0] * phi_mean[3]
        R_diss_p_rhoe = phi_mean[5] + phi_mean[0] * tke_mean + phi_mean[4]
        
        R_diss_p = jnp.stack([R_diss_p_rho, R_diss_p_rhou1, R_diss_p_rhou2, R_diss_p_rhou3, R_diss_p_rhoe])
        R_diss_p *= alpha_3 * (p3_R - p3_L) / (2 * phi_mean[0] * c) 

        fluxes_xi = fluxes_central - R_diss - R_diss_p

        return fluxes_xi

    def reconstruct_xi(
            self,
            phi: Array,
            alpha_1: Array,
            alpha_2: Array,
            alpha_3: Array,
            fs: Array,
            axis: int,
            j: int,
            dx: float = None
        ) -> Tuple[Array, Array]:
        """Reconstructs the phi vector along the axis direction. Reconstruction is done
        via a convex combination of modified WENO1, WENO3 and WENO5.

        :param phi: Buffer of phi vector.
        :type phi: Array
        :param alpha_1: First-order reconstruction weight.
        :type alpha_1: Array
        :param alpha_2: Second-order reconstruction weight.
        :type alpha_2: Array
        :param alpha_3: Third-order reconstruction weight.
        :type alpha_3: Array
        :param fs: Shock sensor.
        :type fs: Array
        :param axis: Spatial direction along which reconstruction is done.
        :type axis: int
        :param j: Bit indicating whether reconstruction is left (j=0) or right (j=1)
            of the cell face.
        :type j: int
        :param dx: Vector of cell sizes in axis direction, defaults to None
        :type dx: float, optional
        :return: Reconstructed phi vector and reconstructed third-oder pressure value.
        :rtype: Tuple[Array, Array]
        """
        cell_state_1 = self.ALDM_WENO1.reconstruct_xi(phi, axis, j)             # WENO 1
        cell_state_2 = self.ALDM_WENO3.reconstruct_xi(phi, axis, j)             # WENO 3
        cell_state_3 = self.ALDM_WENO5.reconstruct_xi(phi, axis, j, fs=fs)      # WENO 5
        
        cell_state_xi_j = alpha_1 * cell_state_1 \
            + alpha_2 * cell_state_2 \
            + alpha_3 * cell_state_3

        return cell_state_xi_j, cell_state_3[4]

    def compute_phi(self, primitives: Array, conservatives: Array) -> Array:
        """Computes the phi vector which is the quantity that is reconstructed
        in the ALDM scheme.

        phi vector notation different from paper,
            \bar{phi} = {\bar{rho}, \bar{u1}, \bar{u2}, \bar{u3},
            \bar{p}, \bar{rho_e}}

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param conservatives: Buffer of conservative variables.
        :type conservatives: Array
        :return: Buffer of the phi vector.
        :rtype: Array
        """
        rho_e = conservatives[4] - 0.5 * primitives[0] * (primitives[1] * primitives[1] + primitives[2] * primitives[2] + primitives[3] * primitives[3])
        phi = jnp.stack([primitives[0], primitives[1], primitives[2], primitives[3], primitives[4], rho_e], axis=0)
        return phi

    def compute_numerical_dissipation(
        self,
        primitives: Array,
    ) -> Array:
        """Corrects the numerical dissipation coefficient for the momentum equation
        in wall vicinity. Three different models are available.

        1) Standard ALDM coefficients (no correction/damping)
        2) Van-Driest damping
        3) Coherent structure damping

        :param primitives: _description_
        :type primitives: Array
        :return: _description_
        :rtype: Array
        """
        if self.wall_damping is None:
            sigma_rhou = self._sigma_rhou

        if self.wall_damping == "VANDRIEST":
            l_w = 0.0
            u_tau = 0.0
            nu = 0.0
            f_VD = (1.0 - jnp.exp(-(l_w * u_tau / (self.vd_constant_a_plus * nu))**self.vd_constant_d))**self.vd_constant_s
            sigma_rhou = self._sigma_rhou * f_VD

        if self.wall_damping == "COHERENTSTRUCTURE":
            W_ij_mean = 0.0
            S_ij_mean = 0.0
            W_ij_W_ij_mean = W_ij_mean * W_ij_mean
            S_ij_S_ij_mean = S_ij_mean * S_ij_mean
            Q_mean = 0.5 * (W_ij_W_ij_mean - S_ij_S_ij_mean)
            E_mean = 0.5 * (W_ij_W_ij_mean + S_ij_S_ij_mean)
            F_CS = Q_mean / E_mean
            F_omega = 0.9 * (1.0 - F_CS)
            f_CS = jnp.pi * jnp.power(F_CS, 1.5) * F_omega
            sigma_rhou = self._sigma_rhou * f_CS

        return sigma_rhou
