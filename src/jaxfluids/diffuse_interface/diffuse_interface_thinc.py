from typing import List, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.config import precision
from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.math.power_functions import squared

class DiffuseInterfaceTHINC(SpatialReconstruction):
    """The DiffuseInterfaceTHINC class implements functionality for THINC
    reconstruction in diffuse interface methods.

    :param SpatialReconstruction: _description_
    :type SpatialReconstruction: _type_
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            material_manager: MaterialManager,
            diffuse_interface_setup: DiffuseInterfaceSetup,
            halo_manager: HaloManager
            ) -> None:

        self.eps = precision.get_eps()

        nh = domain_information.nh_conservatives
        inactive_axes = domain_information.inactive_axes
        super(DiffuseInterfaceTHINC, self).__init__(nh=nh, inactive_axes=inactive_axes)

        self.material_manager = material_manager
        self.equation_manager = equation_manager
        
        equation_information = equation_manager.equation_information
        self.equation_type = equation_information.equation_type
        self.mass_ids = equation_information.mass_ids
        self.mass_slices = equation_information.mass_slices
        self.vel_ids = equation_information.velocity_ids 
        self.vel_slices = equation_information.velocity_slices
        self.energy_ids = equation_information.energy_ids
        self.energy_slices = equation_information.energy_slices
        self.vf_ids = equation_information.vf_ids
        self.vf_slices = equation_information.vf_slices

        self.diffuse_interface_model = diffuse_interface_setup.model

        thinc_setup = diffuse_interface_setup.thinc
        self.thinc_type = thinc_setup.thinc_type
        self.interface_treatment = thinc_setup.interface_treatment
        self.beta_calculation = thinc_setup.interface_projection
        self.beta = thinc_setup.interface_parameter
        self.one_beta = 1.0 / self.beta

        self.volume_fraction_threshold = thinc_setup.volume_fraction_threshold
        self.eps_interface_location = 1e-20
        self.eps_monotonicity = 1e-8

        self._stencil_size = 4
        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])

        nhx, nhy, nhz = domain_information.domain_slices_conservatives

        # Slices to give i-1, i, i+1
        self.thinc_slices = [
            [jnp.s_[..., nh-2:-nh, nhy, nhz], jnp.s_[..., nh-1:-nh+1, nhy, nhz], jnp.s_[..., nh:-nh+2, nhy, nhz]],
            [jnp.s_[..., nhx, nh-2:-nh, nhz], jnp.s_[..., nhx, nh-1:-nh+1, nhz], jnp.s_[..., nhx, nh:-nh+2, nhz]],
            [jnp.s_[..., nhx, nhy, nh-2:-nh], jnp.s_[..., nhx, nhy, nh-1:-nh+1], jnp.s_[..., nhx, nhy, nh:-nh+2]]
        ]

        self.slices_LR = [
            [jnp.s_[..., :-1, :, :], jnp.s_[..., 1:, :, :]],
            [jnp.s_[..., :, :-1, :], jnp.s_[..., :, 1:, :]],
            [jnp.s_[..., :, :, :-1], jnp.s_[..., :, :, 1:]],
        ]

        nhx_ = jnp.s_[1:-1] if 0 in domain_information.active_axes_indices else jnp.s_[:]
        nhy_ = jnp.s_[1:-1] if 1 in domain_information.active_axes_indices else jnp.s_[:]
        nhz_ = jnp.s_[1:-1] if 2 in domain_information.active_axes_indices else jnp.s_[:]

        self.slice_ = [
            jnp.s_[..., :, nhy_, nhz_], 
            jnp.s_[..., nhx_, :, nhz_], 
            jnp.s_[..., nhx_, nhy_, :]
        ]

    def compute_interface_location(
            self,
            volume_fraction: Array,
            sigma: Array,
            beta_xi: Array
            ) -> Array:
        """Computes the interface location based on a 
        tanh fit to the volume fraction field.

        Garrick et al. - Eq. (40)

        NOTE x_tilde seems to make trouble in AD, 
        that's why we have implemented custom derivatives here.
        """

        @partial(jax.custom_vjp)
        def _compute_interface_location(
                volume_fraction: Array,
                sigma: Array,
                beta_xi: Array
            ):
            factor = 2.0 * sigma * beta_xi
            A = jnp.exp(factor)
            B = jnp.exp(factor * volume_fraction)
            # x_tilde = 0.5 / beta_xi * jnp.log((B - 1.0) / (A - B))
            x_tilde = 0.5 / (beta_xi + self.eps_interface_location) * jnp.log(((B - 1.0) + self.eps_interface_location) / ((A - B) + self.eps_interface_location))
            return x_tilde

        def f_fwd(volume_fraction, sigma, beta_xi):
            # Returns primal output and residuals to be used in backward pass by f_bwd.
            return _compute_interface_location(volume_fraction, sigma, beta_xi), (volume_fraction, sigma, beta_xi)

        def f_bwd(res, g):
            volume_fraction, sigma, beta = res # Gets residuals computed in f_fwd
            A = jnp.exp(2.0 * sigma * beta)
            B = jnp.exp(2.0 * sigma * beta * volume_fraction)
            return (
                sigma * (A - 1.0) * B / ((A - B) * (B - 1) + self.eps_interface_location) * g, 
                (sigma * beta * (volume_fraction * B * (A - B) + (volume_fraction * B - A) * (B - 1.0)) - 0.5 * (A - B) * (B - 1.0) * jnp.log(((B - 1.0) + self.eps_interface_location) / ((A - B) + self.eps_interface_location))) / (beta * beta * (A - B) * (B - 1.0) + self.eps_interface_location) * g,
                (volume_fraction * B * (A - B) + (volume_fraction * B - A) * (B - 1.0)) / ((A - B) * (B - 1.0) + self.eps_interface_location) * g)

        _compute_interface_location.defvjp(f_fwd, f_bwd)

        return _compute_interface_location(volume_fraction, sigma, beta_xi)

    def reconstruct_xi(
            self,
            conservatives_L: Array, 
            conservatives_R: Array,
            primitives_L: Array, 
            primitives_R: Array,
            conservatives: Array,
            primitives: Array,
            normal: Array,
            curvature: Array,
            axis: int,
            ) -> Array:
        """Performs THINC reconstruction on the volume fraction field
        and adjusts other reconstructed fields (e.g., partial densites
        or pressure) accordingly.
        
        
        Options interface equilibirum
        1) SHYUE
            Homogenous-equilibrium by adjusting partial densites,
            momenta, and total energy
        2) RHOTHINC
            Adjusts partial densities
        3) PRIMITIVE
            Adjusts partial densities and pressure (pressure, only
            under the presence of surface tension)

        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param normal: _description_
        :type normal: Array
        :param curvature: _description_
        :type curvature: Array
        :param axis: _description_
        :type axis: int
        :return: _description_
        :rtype: Array
        """
        slice_ = self.slice_[axis]

        # GETS RID OF OFFSET IN NON-AXIS DIMENSIONS
        normal = normal[slice_]
        if curvature is not None:
            curvature = curvature[slice_]

        if self.diffuse_interface_model == "5EQM":
            volume_fraction = conservatives[self.vf_ids]
            volume_fraction_L = conservatives_L[self.vf_ids]
            volume_fraction_R = conservatives_R[self.vf_ids]

        else:
            raise NotImplementedError

        interface_cell_mask = self.compute_interface_mask(volume_fraction, axis)
        sigma = self.compute_thinc_sign(volume_fraction, axis)
        beta_xi = self._compute_directional_interface_thickness(normal, axis)

        # RECONSTRUCT VOLUME FRACTION FIELD ACCORDING TO THINC
        volume_fraction_L, volume_fraction_R = self._reconstruct_volume_fraction(
            volume_fraction, volume_fraction_L, volume_fraction_R,
            sigma, beta_xi, interface_cell_mask, axis)

        # COMPUTE INTERFACE EQUILIBIRUM AND SET THINC RECONSTRUCTED VALUES
        conservatives_L, conservatives_R, primitives_L, primitives_R \
        = self._compute_interface_equilibirum(
            conservatives, primitives, 
            conservatives_L, conservatives_R,
            primitives_L, primitives_R,
            volume_fraction_L, volume_fraction_R,
            interface_cell_mask, curvature, 
            sigma, beta_xi, axis)

        return conservatives_L, conservatives_R, primitives_L, primitives_R

    def compute_interface_mask(
            self, 
            volume_fraction: Array,
            axis: int,
            ) -> Array:
        """Computes mask for inteface cells, i.e., cells for which
        1) vf_{i} > eps_vf
        2) vf_{i} < 1 - eps_vf
        3) (vf_{i} - vf_{i-1}) * (vf_{i+1} - vf_{i}) > eps_monoton

        :param volume_fraction_im1: _description_
        :type volume_fraction_im1: Array
        :param volume_fraction_i: _description_
        :type volume_fraction_i: Array
        :param volume_fraction_ip1: _description_
        :type volume_fraction_ip1: Array
        :return: _description_
        :rtype: Array
        """
        s0, s1, s2 = self.thinc_slices[axis]
        interface_cell_marker = jnp.where(
            (volume_fraction[s1] > self.volume_fraction_threshold) & 
            (volume_fraction[s1] < 1.0 - self.volume_fraction_threshold) &
            ((volume_fraction[s2] - volume_fraction[s1]) * (volume_fraction[s1] - volume_fraction[s0]) > self.eps_monotonicity),
            1, 0)
        return interface_cell_marker

    def compute_thinc_sign(
            self,
            volume_fraction: Array,
            axis: int,
        ) -> Array:
        s0, s1, s2 = self.thinc_slices[axis]
        return jnp.sign(volume_fraction[s2] - volume_fraction[s0])

    def _reconstruct_volume_fraction(
            self,
            volume_fraction: Array,
            volume_fraction_L: Array, 
            volume_fraction_R: Array,
            sigma: Array,
            beta_xi: Array,
            interface_cell_mask: Array,
            axis: int
            ) -> Tuple[Array, Array]:
        s0, s1, s2 = self.thinc_slices[axis]
        slice_L, slice_R = self.slices_LR[axis]
        
        if self.thinc_type == "SHYUE":
            # Shyue et al. - 2014 - An Eulerian interface sharpening algorithm 
            # for compressible two-phase flow: The algebraic THINC approach
            # Eqs. 14 a) + b)
            x_tilde = 1.0 / (2.0 * beta_xi) * jnp.log(
                jnp.exp(beta_xi * (1.0 + sigma - 2.0 * volume_fraction[s1]) / sigma) \
                / (1.0 - jnp.exp(beta_xi * (1.0 - sigma - 2.0 * volume_fraction[s1]) / sigma)))
            volume_fraction_L_thinc = 0.5 * (1 + sigma * jnp.tanh(beta_xi * (1.0 - x_tilde)))
            volume_fraction_R_thinc = 0.5 * (1 + sigma * jnp.tanh(-beta_xi * x_tilde))
        
        elif self.thinc_type == "RHOTHINC":
            # Garrick et al. - 2017 - An interface capturing scheme for 
            # modeling atomization in compressible flows
            x_tilde = self.compute_interface_location(volume_fraction[s1], sigma, beta_xi)
            volume_fraction_L_thinc = 0.5 * (1.0 + jnp.tanh(beta_xi * (sigma + x_tilde)))
            volume_fraction_R_thinc = 0.5 * (1.0 + jnp.tanh(beta_xi * x_tilde))

        elif self.thinc_type == "DENG":
            # Deng et al. - 2018 - High fidelity discontinuity-resolving 
            # reconstruction for compressible multiphase flows with moving 
            # interfaces
            q_min = jnp.minimum(volume_fraction[s0], volume_fraction[s2])
            q_max = jnp.maximum(volume_fraction[s0], volume_fraction[s2]) - q_min
            C = (volume_fraction[s1] - q_min + 1e-20) / (q_max + 1e-20)
            B = jnp.exp(sigma * beta_xi * (2.0 * C - 1.0))
            A = (B / (jnp.cosh(beta_xi) - 1.0)) / jnp.tanh(beta_xi)
            volume_fraction_L_thinc = q_min \
                + 0.5 * q_max * (1.0 + sigma * (jnp.tanh(beta_xi) + A) / (1 + A * jnp.tanh(beta_xi)))
            volume_fraction_R_thinc = q_min \
                + 0.5 * q_max * (1.0 + sigma * A)

        elif self.thinc_type == "PRIMITIVE":
            # Symmetric THINC reconstruction assuming
            # phi(x) = 0.5 * ( 1 + tanh(beta_i * (X + c_i)) )
            # with X = (x - x_{i-1/2}) / Delta x if sigma_i >= 0
            # and X = (x_{i+1/2} - x) / Delta x if sigma_i < 0
            tmp = 2.0 * beta_xi
            A = jnp.exp(tmp)
            B = jnp.exp(tmp * volume_fraction[s1])
            xc = 1.0 / tmp * jnp.log((B - 1.0) / (A - B))
            phi_1 = 0.5 * (1.0 + jnp.tanh(beta_xi * xc))
            phi_2 = 0.5 * (1.0 + jnp.tanh(beta_xi * (1.0 + xc)))
            mask = sigma >= 0
            volume_fraction_L_thinc = jnp.where(mask, phi_2, phi_1)
            volume_fraction_R_thinc = jnp.where(mask, phi_1, phi_2)

        else:
            raise NotImplementedError

        volume_fraction_L = jnp.where(interface_cell_mask[slice_L], volume_fraction_L_thinc[slice_L], volume_fraction_L)
        volume_fraction_R = jnp.where(interface_cell_mask[slice_R], volume_fraction_R_thinc[slice_R], volume_fraction_R)

        return volume_fraction_L, volume_fraction_R

    def _compute_interface_equilibirum(
            self,
            conservatives: Array,
            primitives: Array,
            conservatives_L: Array,
            conservatives_R: Array,
            primitives_L: Array,
            primitives_R: Array,
            volume_fraction_L: Array,
            volume_fraction_R: Array,
            interface_mask: Array,
            curvature: Array,
            sigma: Array,
            beta_xi: Array,
            axis: int,
            volume_fraction: Array = None
        ) -> Tuple[Array, Array]:
        """_summary_

        1) NONE or DENG
        2) SHYUE
        3) GARRICK
        4) PRIMITIVE

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param volume_fraction_L: _description_
        :type volume_fraction_L: Array
        :param volume_fraction_R: _description_
        :type volume_fraction_R: Array
        :param interface_cell_marker: _description_
        :type interface_cell_marker: Array
        :param curvature: _description_
        :type curvature: Array
        :param sigma: _description_
        :type sigma: Array
        :param beta_xi: _description_
        :type beta_xi: Array
        :param axis: _description_
        :type axis: int
        :param volume_fraction: _description_, defaults to None
        :type volume_fraction: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """
        
        slice_L, slice_R = self.slices_LR[axis]
        s1 = self.thinc_slices[axis][1]

        # Slices the primitives/conservatives buffer from (Nx+2Nh,Ny+2Nh,Nz+2Nh)
        # such that for e.g., axis = 0, (Nx + 4, Ny, Nz) is returned
        conservatives = conservatives[s1]
        primitives = primitives[s1]

        if self.diffuse_interface_model == "5EQM":
            volume_fraction = conservatives[self.vf_ids]
        else:
            raise NotImplementedError

        if self.interface_treatment in ("NONE", "DENG"):
            conservatives_L = conservatives_L.at[self.vf_ids].set(volume_fraction_L)
            conservatives_R = conservatives_R.at[self.vf_ids].set(volume_fraction_R)
            primitives_L = primitives_L.at[self.vf_ids].set(volume_fraction_L)
            primitives_R = primitives_R.at[self.vf_ids].set(volume_fraction_R)

        elif self.interface_treatment == "SHYUE":
            rho_alpha = conservatives[self.mass_slices]
            pressure = primitives[self.energy_ids]
            rho_u = conservatives[self.vel_slices]
            total_energy = conservatives[self.energy_ids]
            velocity = primitives[self.vel_slices]

            rho = self.material_manager.get_density(conservatives)
            phasic_internal_energy = self.material_manager.get_phasic_specific_energy(pressure)

            volume_fraction_full = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
            phasic_densities = rho_alpha / volume_fraction_full
            kinetic_energy = 0.5 * jnp.sum(velocity * velocity, axis=0)

            def _interface_treatment_shyue(volume_fraction_K, conservatives_K, slice_K) -> Array:
                volume_fraction_K_full = jnp.stack([volume_fraction_K, 1.0 - volume_fraction_K], axis=0)
                rho_alpha_K = rho_alpha[slice_K] + phasic_densities[slice_K] * (volume_fraction_K_full - volume_fraction_full[slice_K])
                rho_K = jnp.sum(rho_alpha_K, axis=0)
                rhou_K = rho_u[slice_K] + velocity[slice_K] * (rho_K - rho[slice_K])
                E_K = total_energy[slice_K] + kinetic_energy[slice_K] * (rho_K - rho[slice_K]) \
                    + jnp.sum(phasic_internal_energy[slice_K] * (volume_fraction_K_full - volume_fraction_full[slice_K]), axis=0)
                rho_alpha_K = jnp.where(interface_mask[slice_K], rho_alpha_K, conservatives_K[self.mass_slices])
                rhou_K = jnp.where(interface_mask[slice_K], rhou_K, conservatives_K[self.vel_slices])
                E_K = jnp.where(interface_mask[slice_K], E_K, conservatives_K[self.energy_ids])
                
                conservatives_K = conservatives_K.at[self.mass_slices].set(rho_alpha_K)
                conservatives_K = conservatives_K.at[self.vel_slices].set(rhou_K)
                conservatives_K = conservatives_K.at[self.energy_ids].set(E_K)
                if self.diffuse_interface_model == "5EQM":
                    conservatives_K = conservatives_K.at[self.vf_ids].set(volume_fraction_K)
                return conservatives_K

            conservatives_L = _interface_treatment_shyue(volume_fraction_L, conservatives_L, slice_L)
            conservatives_R = _interface_treatment_shyue(volume_fraction_R, conservatives_R, slice_R)
            primitives_L = self.equation_manager.get_primitives_from_conservatives(conservatives_L)
            primitives_R = self.equation_manager.get_primitives_from_conservatives(conservatives_R)

        elif self.interface_treatment == "RHOTHINC":
            rho_alpha = conservatives[self.mass_slices]
            tmp = 2.0 * sigma * beta_xi
            A = jnp.exp(tmp)
            B = jnp.exp(tmp * volume_fraction)
            xc = 1.0 / (2.0 * beta_xi) * jnp.log((B - 1.0) / (A - B))
            D = jnp.exp(2 * beta_xi * xc)
            E = jnp.log( squared(A * D + 1.0) / (A * squared(D + 1.0) + self.eps) )
            rho_1 = 2.0 * tmp * rho_alpha[0] / (E + tmp)
            rho_2 = -2.0 * tmp * rho_alpha[1] / (E - tmp)

            def _interface_treatment_rhothinc(\
                    volume_fraction_K: Array, 
                    primitives_K: Array, 
                    slice_K: Array
                ) -> Array:
                rho_alpha_1_K = rho_1[slice_K] * volume_fraction_K
                rho_alpha_2_K = rho_2[slice_K] * (1.0 - volume_fraction_K)
                rho_alpha_K_thinc = jnp.stack([rho_alpha_1_K, rho_alpha_2_K], axis=0)
                rho_alpha_K = jnp.where(interface_mask[slice_K], rho_alpha_K_thinc, primitives_K[self.mass_slices])
                primitives_K = primitives_K.at[self.mass_slices].set(rho_alpha_K)
                if self.diffuse_interface_model == "5EQM":
                    primitives_K = primitives_K.at[self.vf_ids].set(volume_fraction_K)
                return primitives_K

            primitives_L = _interface_treatment_rhothinc(volume_fraction_L, primitives_L, slice_L)
            primitives_R = _interface_treatment_rhothinc(volume_fraction_R, primitives_R, slice_R)
            conservatives_L = self.equation_manager.get_conservatives_from_primitives(primitives_L)
            conservatives_R = self.equation_manager.get_conservatives_from_primitives(primitives_R)

        elif self.interface_treatment == "PRIMITIVE":
            # At cell center
            rho_alpha = conservatives[self.mass_slices]
            pressure = primitives[self.energy_ids]
            volume_fraction_full = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
            phasic_densities = rho_alpha / volume_fraction_full

            def _interface_treatment_primitive(
                    volume_fraction_K: Array, 
                    primitives_K: Array, 
                    slice_K: Array
                ) -> Array:
                volume_fraction_K_full = jnp.stack([volume_fraction_K, 1.0 - volume_fraction_K], axis=0)
                rho_alpha_K_thinc = rho_alpha[slice_K] + phasic_densities[slice_K] * (volume_fraction_K_full - volume_fraction_full[slice_K])
                rho_alpha_K = jnp.where(interface_mask[slice_K], rho_alpha_K_thinc, primitives_K[self.mass_slices])
                if curvature is not None:
                    # Adjust pressure according to Laplace law
                    # based on volume fraction according to THINC
                    sigma_kappa_K = self.material_manager.get_sigma() * curvature[slice_K]
                    p_K_thinc = pressure[slice_K] + sigma_kappa_K * (volume_fraction_K - volume_fraction[slice_K])
                    p_K = jnp.where(interface_mask[slice_K], p_K_thinc, primitives_K[self.energy_ids])
                    primitives_K = primitives_K.at[self.energy_ids].set(p_K)

                primitives_K = primitives_K.at[self.mass_slices].set(rho_alpha_K)
                if self.diffuse_interface_model == "5EQM":
                    primitives_K = primitives_K.at[self.vf_ids].set(volume_fraction_K)
                return primitives_K

            primitives_L = _interface_treatment_primitive(volume_fraction_L, primitives_L, slice_L)
            primitives_R = _interface_treatment_primitive(volume_fraction_R, primitives_R, slice_R)
            conservatives_L = self.equation_manager.get_conservatives_from_primitives(primitives_L)
            conservatives_R = self.equation_manager.get_conservatives_from_primitives(primitives_R)

        return conservatives_L, conservatives_R, primitives_L, primitives_R
    
    def _compute_directional_interface_thickness(
            self,
            normal: Array,
            axis: int
            ) -> Array:
        """Computes the interface thickness in axis-direction,
        based on the projection of the nominal interface thickness
        in the axis-component of the normal.

        1) CONST    beta_xi = beta
        2) NORM_1   beta_xi = beta * abs(n_xi) + 0.01
        3) NORM_2   beta_xi = beta * abs(n_xi)

        :param normal: Normal vector
        :type normal: Array
        :param axis: Axis direction
        :type axis: int
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        if self.beta_calculation == "CONST":
            beta_xi = self.beta
        elif self.beta_calculation == "NORM_1":
            beta_xi = self.beta * jnp.abs(normal[axis]) + 0.01
        elif self.beta_calculation == "NORM_2":
            beta_xi = self.beta * jnp.abs(normal[axis])
        else:
            raise NotImplementedError
        return beta_xi
