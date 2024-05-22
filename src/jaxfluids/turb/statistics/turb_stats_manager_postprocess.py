from functools import partial
from typing import Callable, Dict, Tuple, List, Union
import json

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.config import precision
from jaxfluids.math.parallel_fft import parallel_fft
from jaxfluids.turb.statistics.utilities import (
    real_wavenumber_grid, wavenumber_grid_parallel,
    _calculate_sheartensor_spectral, _calculate_sheartensor_spectral_parallel,
    calculate_vorticity_spectral, calculate_vorticity_spectral_parallel,
    calculate_dilatation_spectral, calculate_dilatation_spectral_parallel,
    _helmholtz_projection,
    reynolds_average, favre_average, van_driest_transform,
    energy_spectrum_spectral, energy_spectrum_physical,
    vmap_energy_spectrum_spectral, vmap_energy_spectrum_physical,
    vmap_energy_spectrum_spectral_parallel, vmap_energy_spectrum_physical_parallel,
    energy_spectrum_1D_spectral)

class TurbulentStatisticsManager:
    """ Provides functionality to calculate statistics of turbulent flows.
    The TurbStatsManager provides turbulent statistics of the initial flow
    field as well as cumulative statistics over the course of a simulation.

    TODO think about namespaces for statistics outputs?
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager
            ) -> None:

        self.domain_information = domain_information
        self.material_manager = material_manager
        self.eps = precision.get_eps()

    def hit_statistics_timeseries(self, primitives: Array) -> Dict:
        return jax.vmap(self.hit_statistics)(primitives)

    def hit_statistics(self, primitives: Array) -> Dict:
        """Calculates statistics for homogeneous isotropic turbulence.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :return: Dictionary with information on the HIT statistics.
        :rtype: Dict
        """
        # TODO rewrite for (N_ensemble,5,Nx,Ny,Nz)

        number_of_cells = self.domain_information.global_number_of_cells
        is_parallel = self.domain_information.is_parallel
        dx, _, _ = self.domain_information.get_local_cell_sizes()
        if is_parallel:
            split_factors = self.domain_information.split_factors
            split_axis_in = np.argmax(np.array(split_factors))
            split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
            split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])

        assert_str = ("To evaluate HIT statistics, provide a single snapshot (5,Nx,Ny,Nz)"
            " or ensemble data (N_ensemble,5,Nx,Ny,Nz).")
        assert primitives.ndim in [4,5], assert_str
        if primitives.ndim == 4:
            primitives = jnp.expand_dims(primitives, axis=0)

        if is_parallel:
            k_field = wavenumber_grid_parallel(number_of_cells, split_factors_out)
            k_vec = jnp.arange(number_of_cells[0])
        else:
            k_field, k_vec, _ = real_wavenumber_grid(number_of_cells[0])

        # Sum over ensemble axis (axis=0) and three spatial dimension (-3,-2,-1)
        s_axis = (0,-1,-2,-3)

        # TODO: rename to density, velocityX, velocityY, velocityZ
        rho = primitives[...,0,:,:,:]
        velocity = primitives[...,1:4,:,:,:]
        pressure = primitives[...,4,:,:,:]
        velX = velocity[...,0,:,:,:]
        velY = velocity[...,1,:,:,:]
        velZ = velocity[...,2,:,:,:]
        temperature = self.material_manager.get_temperature(pressure=pressure, density=rho)
        if is_parallel:
            velocity_hat = parallel_fft(velocity, split_factors, split_axis_out)
        else:
            velocity_hat = jnp.fft.rfftn(velocity, axes=(-1,-2,-3))

        # THERMODYNAMIC STATE AND VISCOSITY
        rho_mean = jnp.mean(rho, axis=s_axis)
        p_mean = jnp.mean(pressure, axis=s_axis)
        T_mean = jnp.mean(temperature, axis=s_axis)
        if is_parallel:
            rho_mean = jax.lax.pmean(rho_mean, axis_name="i")
            p_mean = jax.lax.pmean(p_mean, axis_name="i")
            T_mean = jax.lax.pmean(T_mean, axis_name="i")
        rho_rms = jnp.mean(jnp.square(rho-rho_mean), axis=s_axis)
        p_rms = jnp.mean(jnp.square(pressure-p_mean), axis=s_axis)
        T_rms = jnp.mean(jnp.square(temperature-T_mean), axis=s_axis)
        if is_parallel:
            rho_rms = jax.lax.pmean(rho_rms, axis_name="i")
            p_rms = jax.lax.pmean(p_rms, axis_name="i")
            T_rms = jax.lax.pmean(T_rms, axis_name="i")
        rho_rms = jnp.sqrt(rho_rms)
        p_rms = jnp.sqrt(p_rms)
        T_rms = jnp.sqrt(T_rms)

        mu = self.material_manager.get_dynamic_viscosity(temperature, None)
        mu_mean = jnp.mean(mu, axis=s_axis)
        if is_parallel:
            mu_mean = jax.lax.pmean(mu_mean, axis_name="i")
        nu_mean = mu_mean / rho_mean

        # KINETIC ENERGY AND U_RMS
        velocity_squared = velX * velX + velY * velY + velZ * velZ
        u_rms = jnp.mean(velocity_squared, axis=s_axis)
        if is_parallel:
            u_rms = jax.lax.pmean(u_rms, axis_name="i")
        u_rms = u_prime = jnp.sqrt(1/3 * u_rms)
        q_rms = jnp.sqrt(3) * u_rms
        tke = 3 / 2 * u_rms**2
        TKE = 0.5 * jnp.mean(rho * velocity_squared, axis=s_axis)
        if is_parallel:
            TKE = jax.lax.pmean(TKE, axis_name="i")

        # TURBULENT MACH NUMBER
        speed_of_sound = self.material_manager.get_speed_of_sound(pressure=pressure, density=rho)
        speed_of_sound_mean_1 = self.material_manager.get_speed_of_sound(pressure=p_mean, density=rho_mean)
        speed_of_sound_mean_2 = jnp.mean(speed_of_sound, axis=s_axis)
        if is_parallel:
            speed_of_sound_mean_2 = jax.lax.pmean(speed_of_sound_mean_2, axis_name="i")
        Ma_t_urms = u_rms / speed_of_sound_mean_1
        Ma_t_qrms = q_rms / speed_of_sound_mean_1
        Ma_rms = jnp.mean(velocity_squared / speed_of_sound, axis=s_axis)
        if is_parallel:
            Ma_rms = jax.lax.pmean(Ma_rms, axis_name="i")
        Ma_rms = jnp.sqrt(Ma_rms)

        if is_parallel:
            duidj = _calculate_sheartensor_spectral_parallel(velocity_hat, k_field,
                                                             split_factors_out, split_axis_in)
        else:
            duidj = _calculate_sheartensor_spectral(velocity_hat, k_field)
        S_ij = 0.5 * (duidj + jnp.swapaxes(duidj, -4, -5))
        SijSij_bar = jnp.mean(jnp.sum(S_ij*S_ij, axis=(-4, -5)), axis=s_axis)
        if is_parallel:
            SijSij_bar = jax.lax.pmean(SijSij_bar, axis_name="i")
        eps = 2 * nu_mean * SijSij_bar 

        # VORTICITY
        if is_parallel:
            vorticity = calculate_vorticity_spectral_parallel(velocity_hat, k_field,
                                                              split_factors_out, split_axis_in)
        else:
            vorticity = calculate_vorticity_spectral(velocity_hat, k_field)
        vorticity_squared = jnp.sum(vorticity * vorticity, axis=(-4))   # omega_i omega_i
        enstrophy_mean = jnp.mean(vorticity_squared, axis=s_axis)
        if is_parallel:
            enstrophy_mean = jax.lax.pmean(enstrophy_mean, axis_name="i")
        vorticity_rms = jnp.sqrt(1/3 * enstrophy_mean)

        # DILATATION
        if is_parallel:
            dilatation = calculate_dilatation_spectral_parallel(velocity_hat, k_field, 
                                                       split_factors_out, split_axis_in)
        else:
            dilatation = calculate_dilatation_spectral(velocity_hat, k_field)
        dilatation_mean = jnp.mean(dilatation, axis=s_axis)
        if is_parallel:
            dilatation_mean = jax.lax.pmean(dilatation_mean, axis_name="i")
        dilatation_std = jnp.mean(jnp.square(dilatation - dilatation_mean), axis=s_axis)
        if is_parallel:
            dilatation_std = jax.lax.pmean(dilatation_std, axis_name="i")
        dilatation_std = jnp.sqrt(dilatation_std)
        divergence_rms_spec = jnp.mean(jnp.square(dilatation), axis=s_axis)
        if is_parallel:
            divergence_rms_spec = jax.lax.pmean(divergence_rms_spec, axis_name="i")
        divergence_rms_spec = jnp.sqrt(divergence_rms_spec)
        divergence_rms_fd = jnp.mean(jnp.square(
            duidj[...,0,0,:,:,:] + + duidj[...,1,1,:,:,:] + duidj[...,2,2,:,:,:]
        ), axis=s_axis)
        if is_parallel:
            divergence_rms_fd = jax.lax.pmean(divergence_rms_fd, axis_name="i")
        divergence_rms_fd = jnp.sqrt(divergence_rms_fd)

        # SPECTRA
        if is_parallel:
            ek_spec = jnp.mean(vmap_energy_spectrum_spectral_parallel(velocity_hat, split_factors_out, multiplicative_factor=0.5), axis=0)
            omega_spec = jnp.mean(vmap_energy_spectrum_physical_parallel(vorticity, split_factors), axis=0)
            rho_spec = jnp.mean(vmap_energy_spectrum_physical_parallel(rho, split_factors, is_scalar_field=True), axis=0)
            p_spec = jnp.mean(vmap_energy_spectrum_physical_parallel(pressure, split_factors, is_scalar_field=True), axis=0)
            T_spec = jnp.mean(vmap_energy_spectrum_physical_parallel(temperature, split_factors, is_scalar_field=True), axis=0)
            d_spec = jnp.mean(vmap_energy_spectrum_physical_parallel(dilatation, split_factors, is_scalar_field=True), axis=0)
        else:
            ek_spec = jnp.mean(vmap_energy_spectrum_spectral(velocity_hat, number_of_cells,
                multiplicative_factor=0.5), axis=0)
            omega_spec = jnp.mean(vmap_energy_spectrum_physical(vorticity), axis=0)
            # velocity_sol_hat, velocity_comp_hat = _helmholtz_projection(velocity_hat, k_field)
            # ek_sol_spec = jnp.mean(self.energy_spectrum_spectral(velocity_sol_hat), axis=0)
            # ek_comp_spec = jnp.mean(self.energy_spectrum_spectral(velocity_comp_hat), axis=0)
            rho_spec = jnp.mean(vmap_energy_spectrum_physical(rho, is_scalar_field=True), axis=0)
            p_spec = jnp.mean(vmap_energy_spectrum_physical(pressure, is_scalar_field=True), axis=0)
            T_spec = jnp.mean(vmap_energy_spectrum_physical(temperature, is_scalar_field=True), axis=0)
            d_spec = jnp.mean(vmap_energy_spectrum_physical(dilatation, is_scalar_field=True), axis=0)

        # AUTO CORRELATION
        # Eq. 6.45
        f_corr = []
        g_corr = []
        u1u1_bar = jnp.mean(velX*velX, axis=s_axis)
        u2u2_bar = jnp.mean(velY*velY, axis=s_axis)
        if is_parallel:
            u1u1_bar = jax.lax.pmean(u1u1_bar, axis_name="i")
            u2u2_bar = jax.lax.pmean(u2u2_bar, axis_name="i")
            if split_axis_in == 0:
                for ii in range(number_of_cells[1] // 2):
                    f_corr_ii = jnp.mean(velY * jnp.roll(velY, shift=ii, axis=-2), axis=s_axis) / u2u2_bar # longitudinal
                    g_corr_ii = jnp.mean(velX * jnp.roll(velX, shift=ii, axis=-2), axis=s_axis) / u1u1_bar # transversal
                    f_corr_ii = jax.lax.pmean(f_corr_ii, axis_name="i")
                    g_corr_ii = jax.lax.pmean(g_corr_ii, axis_name="i")
                    f_corr.append(f_corr_ii)
                    g_corr.append(g_corr_ii)
            else:
                for ii in range(number_of_cells[0] // 2):
                    f_corr_ii = jnp.mean(velX * jnp.roll(velX, shift=ii, axis=-3), axis=s_axis) / u1u1_bar # longitudinal
                    g_corr_ii = jnp.mean(velY * jnp.roll(velY, shift=ii, axis=-3), axis=s_axis) / u2u2_bar # transversal
                    f_corr_ii = jax.lax.pmean(f_corr_ii, axis_name="i")
                    g_corr_ii = jax.lax.pmean(g_corr_ii, axis_name="i")
                    f_corr.append(f_corr_ii)
                    g_corr.append(g_corr_ii)
        else:
            for ii in range(number_of_cells[0] // 2):
                f_corr_ii = jnp.mean(velX * jnp.roll(velX, shift=ii, axis=-3), axis=s_axis) / u1u1_bar # longitudinal
                g_corr_ii = jnp.mean(velY * jnp.roll(velY, shift=ii, axis=-3), axis=s_axis) / u2u2_bar # transversal
                f_corr.append(f_corr_ii)
                g_corr.append(g_corr_ii)
        f_corr = jnp.array(f_corr)
        g_corr = jnp.array(g_corr)

        # LENGTH SCALES 
        # Integral length scale
        L_11 = jnp.sum(f_corr * dx, axis=-1)     # Eq. 6.47
        L_22 = jnp.sum(g_corr * dx, axis=-1)     # Eq. 6.48

        int_Ek = jnp.trapz(ek_spec)
        int_Ek_over_k = jnp.trapz(ek_spec / (k_vec + 1e-10))
        L_I = 0.75 * jnp.pi * int_Ek_over_k / int_Ek

        # Longitudinal (lambda_f) and lateral (lambda_g) Taylor length scale
        lambda_f = jnp.mean(velX * velX, axis=s_axis) / jnp.mean(duidj[...,0,0,:,:,:]*duidj[...,0,0,:,:,:], axis=s_axis)
        lambda_g = jnp.mean(velX * velX, axis=s_axis) / jnp.mean(duidj[...,0,1,:,:,:]*duidj[...,0,1,:,:,:], axis=s_axis)
        if is_parallel:
            lambda_f = jax.lax.pmean(lambda_f, axis_name="i")
            lambda_g = jax.lax.pmean(lambda_g, axis_name="i")
        lambda_f = jnp.sqrt(lambda_f)
        lambda_g = jnp.sqrt(lambda_g)

        # Kolmogorov
        eta = (nu_mean**3 / eps)**0.25
        kmax = jnp.sqrt(2) * number_of_cells[0] / 3 # better Nx/2 ?
        eta_kmax = eta * kmax

        # TIME SCALES
        tau_LI = L_I / u_rms            # Large-eddy-turnover time
        tau_l0 = tke / eps
        tau_lambda = lambda_f / u_rms   # Eddy turnover time based on longitudinal Taylor scale
        tau_lambda_g = lambda_g / u_rms # Eddy turnover time based on transversal Taylor scale

        # REYNOLDS NUMBER
        Re_turb = q_rms**4 / eps / nu_mean     # Turbulent Reynolds number, Spyropoulos et al. 1996
        Re_lambda = lambda_f * u_rms / nu_mean
        Re_lambda_g = lambda_g * u_rms / nu_mean
        Re_int = u_rms * L_I / nu_mean
        Re_L = tke**2 / eps / nu_mean
        Re_T = u_prime * L_11 / nu_mean

        turb_stats_dict = {
            "THERMODYNAMICS": {
                "P_MEAN": p_mean, "P_RMS": p_rms,
                "RHO_MEAN": rho_mean, "RHO_RMS": rho_rms,
                "T_MEAN": T_mean, "T_RMS": T_rms,
                "C_MEAN": speed_of_sound_mean_1,
                "MU_MEAN": mu_mean
            },
            "VELOCITY_SCALES": {
                "MA_T_URMS": Ma_t_urms,
                "MA_T_QRMS": Ma_t_qrms,
                "MA_RMS": Ma_rms,
                "U_RMS": u_rms,
                "Q_RMS": q_rms,
                "TKE": TKE, "tke": tke,
                "ENSTROPHY_MEAN": enstrophy_mean,
                "DILATATION_STD": dilatation_std,
                "DIVERGENCE_RMS_SPEC": divergence_rms_spec,
                "DIVERGENCE_RMS_FD": divergence_rms_fd,
                "DISSIPATION": eps
            },
            "LENGTH_SCALES": {
                "INTEGRAL": L_I,
                "LAMBDA": lambda_f,
                "LAMBDA_G": lambda_g,
                "KOLMOGOROV": eta,
                "KOLMOGOROV_RESOLUTION": eta_kmax,
                "L_11": L_11, "L_22": L_22
            },
            "TIME_SCALES": {
                "TAU_LI": tau_LI,
                "TAU_L0": tau_l0,
                "TAU_LAMBDA": tau_lambda,
                "TAU_LAMBDA_G": tau_lambda_g
            },
            "SPECTRA": {
                "ENERGY_SPECTRUM": ek_spec,
                # "ENERGY_SPECTRUM_SOL": ek_sol_spec,
                # "ENERGY_SPECTRUM_COMP": ek_comp_spec,
                "DENSITY_SPECTRUM": rho_spec,
                "PRESSURE_SPECTRUM": p_spec,
                "TEMPERATURE_SPECTRUM": T_spec,
                "VORTICITY_SPECTRUM": omega_spec,
                "DILATATION_SPECTRUM": d_spec,
                "WAVE_NUMBER_VECTOR": k_vec
            },
            "CORRELATIONS": {
                "LONGITUDINAL": f_corr,
                "TRANSVERSAL": g_corr,
            },
            "REYNOLDS_NUMBERS": {
                "RE_LAMBDA": Re_lambda,
                "RE_LAMBDA_G": Re_lambda_g,
                "RE_INTEGRAL": Re_int,
                "RE_TURBULENT": Re_turb,
                "RE_L": Re_L,
                "RE_T": Re_T,
            },
        }
        return turb_stats_dict

    def channel_statistics(
            self, 
            primitives: Array,
            is_energy_spectra: bool = False,
            is_two_point_correlation: bool = False,
            is_favre_averages: bool = False,
            is_spatial_correlation: bool = False,
        ) -> Dict:
        """Calculates statistics of a compressible channel flow and returns 
        them as a dictionary.

        -

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param is_energy_spectra: when true energy
        spectrum is calculated, defaults to False
        :type is_energy_spectra: bool, optional
        :param is_two_point_correlation: when true 
        two-point correlation is calculated, defaults to False
        :type is_two_point_correlation: bool, optional
        :return: [description]
        :rtype: Dict
        """

        assert_str = ("To evaluate channel statistics, provide a single snapshot (5,Nx,Ny,Nz) "
            "or a timeseries data (Nt,5,Nx,Ny,Nz).")
        assert primitives.ndim in [4,5], assert_str
        if primitives.ndim == 4:
            primitives = jnp.expand_dims(primitives, axis=0)
            
        if self.domain_information.is_parallel:
            raise NotImplementedError

        Nsamples = primitives.shape[0]
        Nt, _, Nx, Ny, Nz = primitives.shape

        # TODO: rename to density, velocityX, velocityY, velocityZ
        rho = primitives[...,0,:,:,:]
        velocity = primitives[...,1:4,:,:,:]
        U = primitives[...,1,:,:,:]
        V = primitives[...,2,:,:,:]
        W = primitives[...,3,:,:,:]
        pressure = primitives[...,4,:,:,:]
        temperature = self.material_manager.get_temperature(pressure=pressure, density=rho)

        # CHANNEL SIZES AND CELL SIZES
        domain_size = self.domain_information.get_global_domain_size()
        Lx = domain_size[0][1] - domain_size[0][0]
        Ly = domain_size[1][1] - domain_size[1][0]
        Lz = domain_size[2][1] - domain_size[2][0]
        delta = 0.5 * Ly   # Half channel height
        cell_centers_y = self.domain_information.get_local_cell_centers()[1]
        cell_sizes = self.domain_information.get_local_cell_sizes()
        cell_sizes_x = cell_sizes[0]
        cell_sizes_y = jnp.squeeze(cell_sizes[1])
        cell_sizes_z = cell_sizes[2]

        if cell_sizes_y.ndim == 0:
            cell_sizes_y = cell_sizes_y.reshape(1,)
          
        # THERMODYNAMICS
        rho_mean_y = reynolds_average(rho)
        rhop = rho - jnp.expand_dims(rho_mean_y, axis=(0,2))    # Fluctuating density # TODO
        rho_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * rho_mean_y)
        rho_rms_y = jnp.sqrt(reynolds_average(rhop * rhop)) # rho_rms = <(rho - <rho>)**2>**0.5
        rho_rms = jnp.std(rho)  # TODO does this quantity make sense???
        rho_wall = rho_mean_y[0]
        
        p_mean_y = reynolds_average(pressure)
        pp = pressure - jnp.expand_dims(p_mean_y, axis=(0,2))
        p_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * p_mean_y)
        p_rms_y = jnp.sqrt(reynolds_average(pp * pp)) # rho_rms = <(p - <p>)**2>**0.5
        p_rms = jnp.std(pressure) # TODO does this quantity make sense???
        p_wall = p_mean_y[0]

        T_mean_y = reynolds_average(temperature)
        Tp = temperature - jnp.expand_dims(T_mean_y, axis=(0,2))  # TODO
        T_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * T_mean_y)
        T_rms_y = jnp.sqrt(reynolds_average(Tp * Tp)) # rho_rms = <(T - <T>)**2>**0.5
        T_rms = jnp.std(temperature)  # TODO does this quantity make sense???
        T_wall = T_mean_y[0]

        # VISCOSITY
        mu = self.material_manager.get_dynamic_viscosity(temperature, None)
        mu_mean_y = reynolds_average(mu)
        mu_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * mu_mean_y)
        mu_wall = mu_mean_y[0]
        # nu = mu / rho
        # nu_mean_y = reynolds_average(nu)
        nu_mean_y = mu_mean_y / rho_mean_y  # This quantity is more often used in literature

        # BULK VELOCITY AND CENTER LINE VELOCITY
        U_mean = jnp.mean(U, axis=(0,1,3), keepdims=True)
        U_mean_y = jnp.squeeze(U_mean)
        momentum_x_mean = jnp.mean(rho * U, axis=(0,1,3), keepdims=True)
        U_bar = 1/(2 * delta) * jnp.sum(cell_sizes[1] * U_mean)
        U_bar_rho = 1/(2 * delta) * jnp.sum(cell_sizes[1] * momentum_x_mean) / rho_mean
        U_0 = jnp.max(U_mean_y)
        U_mean_VD_y = van_driest_transform(U_mean_y, rho_mean_y)

        # TURBULENT SCALES
        # Wall shear stress Eq. (7.24)
        dudy_wall = U_mean_y[0] / (0.5 * cell_sizes_y[0])
        tau_wall = mu_wall * dudy_wall
        # Friction velocity Eq. (7.25)
        u_tau = jnp.sqrt(tau_wall / rho_wall)

        U_plus_max = U_0 / u_tau
        U_plus_y = U_mean_y / u_tau
        U_mean_plus_VD_y = van_driest_transform(U_plus_y, rho_mean_y)

        # MACH NUMBER
        speed_of_sound_mean = self.material_manager.get_speed_of_sound(pressure=p_mean, density=rho_mean) # TODO sensible???
        # speed_of_sound_mean_y = self.material_manager.get_speed_of_sound(pressure=p_mean_y, density=rho_mean_y)
        speed_of_sound = self.material_manager.get_speed_of_sound(pressure=pressure, density=rho)
        speed_of_sound_mean_y = reynolds_average(speed_of_sound)
        Ma_0    = U_0 / speed_of_sound_mean
        Ma_bulk = U_bar / speed_of_sound_mean 
        Ma_rms  = jnp.sqrt( jnp.mean( cell_sizes[1] * ((U - U_mean)*(U - U_mean) + V*V + W*W) ) ) / speed_of_sound_mean
        Mach_mean_y = U_mean_y / speed_of_sound_mean_y
        m = jnp.sqrt(U*U + V*V + W*W) / speed_of_sound
        Mach_rms_y = jnp.sqrt(reynolds_average((m - reynolds_average(m, keepdims=True))**2))

        # KINETIC ENERGY
        tke = 0.5 * jnp.mean(cell_sizes[1] * (U*U + V*V + W*W))
        TKE = 0.5 * jnp.mean(cell_sizes[1] * rho * (U*U + V*V + W*W))

        # LENGTH SCALES
        delta_nu = nu_mean_y[0] / u_tau # Viscous lengthscale Eq. (7.26)
        delta_x_plus = cell_sizes_x / delta_nu                   
        delta_y_plus_min = jnp.min(cell_sizes_y) / delta_nu     # Pope 7.28
        delta_y_plus_max = jnp.max(cell_sizes_y) / delta_nu
        delta_z_plus = cell_sizes_z / delta_nu                   

        # WALL DISTANCE
        y = jnp.minimum(
            jnp.abs(cell_centers_y - domain_size[1][0]), 
            jnp.abs(cell_centers_y - domain_size[1][1]))
        y_plus = y / delta_nu

        # TIME SCALES
        flow_through_time_outer = Lx / U_bar
        flow_through_time_inner = Lx / u_tau

        # REYNOLDS NUMBER
        Re_tau  = delta * u_tau * rho_wall / mu_wall        # Pope 7.27
        Re_bulk = 2 * delta * U_bar * rho_wall / mu_wall    # Pope 7.1
        Re_0    = delta * U_0 * rho_wall / mu_wall          # Pope 7.2
        Re_tau_coleman = delta * u_tau * rho_wall / mu_wall
        Re_coleman = rho_mean * U_bar_rho * delta / mu_wall

        # REYNOLDS STRESSES
        up = U - U_mean
        vp = V     # V_mean = 0.0
        wp = W     # W_mean = 0.0

        # Reynolds stresses
        up_up = reynolds_average(up * up)    # <u'u'>
        vp_vp = reynolds_average(vp * vp)    # <v'v'>
        wp_wp = reynolds_average(wp * wp)    # <w'w'>
        up_vp = reynolds_average(up * vp)    # <u'v'>
        up_wp = reynolds_average(up * wp)    # <u'w'>
        vp_wp = reynolds_average(vp * wp)    # <v'w'>
        Tp_Tp = reynolds_average(Tp * Tp)    # <T'T'>
        vp_Tp = reynolds_average(vp * Tp)    # <v'T'>

        q_squared = up_up + vp_vp + wp_wp
        Mach_turbulent_y = jnp.sqrt(q_squared) / speed_of_sound_mean_y

        # 
        rhop_up_up = reynolds_average(rhop * up * up)
        rhop_vp_vp = reynolds_average(rhop * vp * vp)
        rhop_wp_wp = reynolds_average(rhop * wp * wp)
        rhop_up_vp = reynolds_average(rhop * up * vp)
        rhop_up_wp = reynolds_average(rhop * up * wp)
        rhop_vp_wp = reynolds_average(rhop * vp * wp)
        rhop_Tp_Tp = reynolds_average(rhop * Tp * Tp)
        rhop_vp_Tp = reynolds_average(rhop * vp * Tp)

        if is_favre_averages:
            T_tilde = favre_average(temperature, rho)
            u_tilde = favre_average(U, rho)
            v_tilde = favre_average(V, rho)
            w_tilde = favre_average(W, rho)
            
            Tpp = temperature - jnp.expand_dims(T_tilde, axis=(0,1,3))
            upp = U - jnp.expand_dims(u_tilde, axis=(0,1,3))
            vpp = V - jnp.expand_dims(v_tilde, axis=(0,1,3))
            wpp = W - jnp.expand_dims(w_tilde, axis=(0,1,3))
            
            upp_upp = reynolds_average(upp * upp)     # <u"u">
            vpp_vpp = reynolds_average(vpp * vpp)     # <v"v">
            wpp_wpp = reynolds_average(wpp * wpp)     # <w"w">
            upp_vpp = reynolds_average(upp * vpp)     # <u"v">
            upp_wpp = reynolds_average(upp * wpp)     # <u"w">
            vpp_wpp = reynolds_average(vpp * wpp)     # <v"w">
            Tpp_Tpp = reynolds_average(Tpp * Tpp)     # <T"T">
            vpp_Tpp = reynolds_average(vpp * Tpp)     # <v"T">
            
            upp_upp_tilde = favre_average(upp * upp, rho)     # <rho u"u"> / <rho>
            vpp_vpp_tilde = favre_average(vpp * vpp, rho)     # <rho v"v"> / <rho>
            wpp_wpp_tilde = favre_average(wpp * wpp, rho)     # <rho w"w"> / <rho>
            upp_vpp_tilde = favre_average(upp * vpp, rho)     # <rho u"v"> / <rho>
            upp_wpp_tilde = favre_average(upp * wpp, rho)     # <rho u"w"> / <rho>
            vpp_wpp_tilde = favre_average(vpp * wpp, rho)     # <rho v"w"> / <rho>
            Tpp_Tpp_tilde = favre_average(Tpp * Tpp, rho)     # <rho T"T"> / <rho>
            vpp_Tpp_tilde = favre_average(vpp * Tpp, rho)     # <rho v"T"> / <rho>
        else:
            T_tilde = u_tilde = v_tilde = w_tilde = 0.0
            Tpp = upp = vpp = wpp = 0.0
            upp_upp = vpp_vpp = wpp_wpp = upp_vpp = upp_wpp \
            = vpp_wpp = Tpp_Tpp = vpp_Tpp = 0.0
            upp_upp_tilde = vpp_vpp_tilde = wpp_wpp_tilde \
            = upp_vpp_tilde = upp_wpp_tilde = vpp_wpp_tilde \
            = Tpp_Tpp_tilde = vpp_Tpp_tilde = 0.0

        if is_energy_spectra:
            # 1-D Energy spectra
            y_ids = [jnp.argmin(jnp.abs(y_plus - 5.0)), jnp.argmin(jnp.abs(y_plus - 150.0))]
            E_uiui = {
                "x": jnp.zeros((len(y_ids), 3, Nx)), 
                "z": jnp.zeros((len(y_ids), 3, Nz)),
            }
            for ii, y_id in enumerate(y_ids):
                # Streamwise
                velocity_hat_x = jnp.fft.fft(velocity[:,:,:,y_id,:], axis=2)
                velocity_hat_x = (jnp.moveaxis(velocity_hat_x, 3, 1)).reshape(Nsamples * Nz, 3, Nx)
                for jj in range(Nsamples * Nz):
                    for kk in range(3):
                        ek = energy_spectrum_1D_spectral(velocity_hat_x[jj,kk:kk+1,:])
                        E_uiui["x"] = E_uiui["x"].at[ii,kk].add(1.0 / (Nsamples * Nz) * ek)

                # Spanwise
                velocity_hat_z = jnp.fft.fft(velocity[:,:,:,y_id,:], axis=3)
                velocity_hat_z = (jnp.moveaxis(velocity_hat_z, 2, 1)).reshape(Nsamples * Nx, 3, Nz)
                for jj in range(Nsamples * Nx):
                    for kk in range(3):
                        ek = energy_spectrum_1D_spectral(velocity_hat_z[jj,kk:kk+1,:])
                        E_uiui["z"] = E_uiui["z"].at[ii,kk].add(1.0 / (Nsamples * Nx) * ek)
        else:
            E_uiui = 0

        if is_spatial_correlation:
            raise NotImplementedError

        turb_stats_dict = {
            "THERMODYNAMICS": {
                "P_MEAN": p_mean, "P_RMS": p_rms,
                "RHO_MEAN": rho_mean, "RHO_RMS": rho_rms,
                "T_MEAN": T_mean, "T_RMS": T_rms,
                "MU_MEAN": mu_mean,
            },
            "THERMODYNAMICS_Y": {
                "P_MEAN": p_mean_y, "P_RMS": p_rms_y,
                "RHO_MEAN": rho_mean_y, "RHO_RMS": rho_rms_y,
                "T_MEAN": T_mean_y, "T_RMS": T_rms_y,
                "MU_MEAN": mu_mean_y, 
            },
            "VELOCITY_SCALES": {
                "U_BAR": U_bar,
                "U_BAR_RHO": U_bar_rho,
                "U_0": U_0,
                "U_TAU": u_tau,
                "TAU_WALL": tau_wall,
                "C_MEAN": speed_of_sound_mean,
                "MA_BULK": Ma_bulk,
                "MA_0": Ma_0,
                "MA_RMS": Ma_rms,
                "TKE": TKE
            },
            "VELOCITY_SCALES_Y": {
                "U_MEAN": U_mean_y,
                "U+": U_plus_y,
                "TKE": TKE,
                "MA_MEAN": Mach_mean_y,
                "MA_RMS": Mach_rms_y,
                "MA_T": Mach_turbulent_y,
                "U_VD": U_mean_VD_y,
                "U_VD+": U_mean_plus_VD_y
            },
            "LENGTH_SCALES": {
                "DELTA": delta,
                "DELTA_NU": delta_nu,
                "DELTA_X+": delta_x_plus,
                "DELTA_Y+_MIN": delta_y_plus_min,
                "DELTA_Y+_MAX": delta_y_plus_max,
                "DELTA_Z+": delta_z_plus,
            },
            "LENGTH_SCALES_Y": {
                "Y_PLUS": y_plus,                
            },
            "TIME_SCALES": {
                "FLOW_THROUGH_TIME_OUTER": flow_through_time_outer,
                "FLOW_THROUGH_TIME_INNER": flow_through_time_inner,
            },
            "SPECTRA": {
                "ENERGY_SPECTRUM": E_uiui,
            },
            "CORRELATIONS": {

            },
            "REYNOLDS_NUMBERS": {
                "RE_TAU": Re_tau,
                "RE_BULK": Re_bulk,
                "RE_0": Re_0,
                "RE_TAU_COLEMAN": Re_tau_coleman,
                "RE_COLEMAN": Re_coleman
            },
            "REYNOLDS_STRESSES": {
                "up_up": up_up, "vp_vp": vp_vp, "wp_wp": wp_wp,
                "up_vp": up_vp, "up_wp": up_wp, "vp_wp": vp_wp,
                "Tp_Tp": Tp_Tp, "vp_Tp": vp_Tp,
                "rhop_up_up": rhop_up_up, "rhop_vp_vp": rhop_vp_vp,
                "rhop_wp_wp": rhop_wp_wp, "rhop_up_vp": rhop_up_vp,
                "rhop_up_wp": rhop_up_wp, "rhop_vp_wp": rhop_vp_wp,
                "rhop_Tp_Tp": rhop_Tp_Tp, "rhop_vp_Tp": rhop_vp_Tp,
                "upp_upp": upp_upp, "vpp_vpp": vpp_vpp, "wpp_wpp": wpp_wpp,
                "upp_vpp": upp_vpp, "upp_wpp": upp_wpp, "vpp_wpp": vpp_wpp,
                "Tpp_Tpp": Tpp_Tpp, "vpp_Tpp": vpp_Tpp,
                "upp_upp_tilde": upp_upp_tilde, "vpp_vpp_tilde": vpp_vpp_tilde,
                "wpp_wpp_tilde": wpp_wpp_tilde, "upp_vpp_tilde": upp_vpp_tilde,
                "upp_wpp_tilde": upp_wpp_tilde, "vpp_wpp_tilde": vpp_wpp_tilde,
                "Tpp_Tpp_tilde": Tpp_Tpp_tilde, "vpp_Tpp_tilde": vpp_Tpp_tilde
            },
        }

        return turb_stats_dict

    def boundarylayer_statistics(
            self,
            primitives: Array,
            profile_measure_position: float,
            developed_region_bounds: Tuple[float],
            freestream_state: Dict[str, float],
            y_plus_positions: Tuple[float] = (2.16, 27.9, 157.9),
            is_energy_spectra: bool = False,
            is_correlation: bool = False
            ) -> Dict[str, Array]:

        if self.domain_information.is_parallel:
            raise NotImplementedError
        
        assert_str = ("To evaluate channel statistics, provide a single snapshot (5,Nx,Ny,Nz) "
            "or a timeseries data (Nt,5,Nx,Ny,Nz).")
        assert primitives.ndim in [4,5], assert_str
        if primitives.ndim == 4:
            primitives = jnp.expand_dims(primitives, axis=0)

        keys = ("U_e", "rho_e", "T_e", "mu_e")
        for key in keys:
            if key not in freestream_state:
                RuntimeError
        
        U_e = freestream_state["U_e"]
        rho_e = freestream_state["rho_e"]
        T_e = freestream_state["T_e"]
        mu_e = freestream_state["mu_e"]

        Nsamples = primitives.shape[0]
        Nt, _, Nx, Ny, Nz = primitives.shape

        density = primitives[...,0,:,:,:]
        velocity = primitives[...,1:4,:,:,:]
        velocityX = primitives[...,1,:,:,:]
        velocityY = primitives[...,2,:,:,:]
        velocityZ = primitives[...,3,:,:,:]
        pressure = primitives[...,4,:,:,:]
        temperature = self.material_manager.get_temperature(pressure=pressure, density=density)

        s_axes = (0,3)

        x, y, z = self.domain_information.get_local_cell_centers()
        dx, dy, dz = self.domain_information.get_local_cell_sizes()
        dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
        dx_min, dy_min, dz_min = jnp.min(dx), jnp.min(dy), jnp.min(dz)

        # MEASUREMENT LOCATIONS
        x_id = jnp.argmin(jnp.abs(x-profile_measure_position))
        x1_id = jnp.argmin(jnp.abs(x-developed_region_bounds[0]))
        x2_id = jnp.argmin(jnp.abs(x-developed_region_bounds[1]))

        # MEAN
        rho_mean = reynolds_average(density, s_axes, keepdims=True)
        u_mean = reynolds_average(velocityX, s_axes, keepdims=True)
        v_mean = reynolds_average(velocityY, s_axes, keepdims=True)
        w_mean = reynolds_average(velocityZ, s_axes, keepdims=True)
        p_mean = reynolds_average(pressure, s_axes, keepdims=True)
        T_mean = reynolds_average(temperature, s_axes, keepdims=True)

        # MEAN FLUCTUATIONS
        rhop = density - rho_mean
        up = velocityX - u_mean
        vp = velocityY - v_mean
        wp = velocityZ - w_mean
        pp = pressure - p_mean
        Tp = temperature - T_mean

        rhop_rhop_mean = reynolds_average(rhop*rhop, s_axes)
        up_up_mean = reynolds_average(up*up, s_axes)
        vp_vp_mean = reynolds_average(vp*vp, s_axes)
        wp_wp_mean = reynolds_average(wp*wp, s_axes)
        up_vp_mean = reynolds_average(up*vp, s_axes)
        up_wp_mean = reynolds_average(up*wp, s_axes)
        vp_wp_mean = reynolds_average(vp*wp, s_axes)
        pp_pp_mean = reynolds_average(pp*pp, s_axes)
        Tp_Tp_mean = reynolds_average(Tp*Tp, s_axes)

        # SQUEEZE
        rho_mean = jnp.squeeze(rho_mean)
        u_mean = jnp.squeeze(u_mean)
        v_mean = jnp.squeeze(v_mean)
        w_mean = jnp.squeeze(w_mean)
        p_mean = jnp.squeeze(p_mean)
        T_mean = jnp.squeeze(T_mean)

        rhop_rhop_mean = jnp.squeeze(rhop_rhop_mean)
        up_up_mean = jnp.squeeze(up_up_mean)
        vp_vp_mean = jnp.squeeze(vp_vp_mean)
        wp_wp_mean = jnp.squeeze(wp_wp_mean)
        up_vp_mean = jnp.squeeze(up_vp_mean)
        up_wp_mean = jnp.squeeze(up_wp_mean)
        vp_wp_mean = jnp.squeeze(vp_wp_mean)
        pp_pp_mean = jnp.squeeze(pp_pp_mean)
        Tp_Tp_mean = jnp.squeeze(Tp_Tp_mean)

        # MEAN VISCOSITY
        mu_mean = self.material_manager.get_dynamic_viscosity(T_mean, primitives)
        nu_mean = mu_mean/rho_mean

        # THERMODYNAMIC RMS
        rhop_rhop_rms = jnp.sqrt(rhop_rhop_mean)
        Tp_Tp_rms = jnp.sqrt(Tp_Tp_mean)
        pp_pp_rms = jnp.sqrt(pp_pp_mean)

        # WALL UNITS
        mu_wall = mu_mean[:,0]
        nu_wall = nu_mean[:,0]
        rho_wall = rho_mean[:,0]
        dudy_wall = u_mean[:,0]/y[0]
        tau_wall = mu_wall * dudy_wall
        u_tau = jnp.sqrt(tau_wall/rho_wall)
        u_vd = jax.vmap(van_driest_transform, in_axes=(0,0))(u_mean, rho_mean)
        Cf = 2*tau_wall/rho_e/U_e**2

        # LENGTH SCALES
        l_plus = nu_wall/u_tau
        delta_y_min_plus = dy_min/l_plus[x_id]
        delta_x_min_plus = dx_min/l_plus[x_id]
        delta_z_min_plus = dz_min/l_plus[x_id]
        y_plus = y/l_plus.reshape(-1,1)

        # BOUNDARY/DISPLACEMENT/MOMENTUM THICKNESS
        y_index_delta_0 = jnp.argmax(u_mean/U_e > 0.99, axis=1)
        y_index_delta_c = jnp.argmax(u_vd/u_vd[:,-2:-1] > 0.99, axis=1)
        delta_0 = y[y_index_delta_0]
        delta_c = y[y_index_delta_c]
        delta_1 = jnp.trapz((1-rho_mean/rho_e*u_mean/U_e),y, axis=1)
        delta_2 = jnp.trapz(rho_mean/rho_e*u_mean/U_e*(1-u_mean/U_e), y, axis=1)

        # REYNOLDS NUMBERS
        Re_tau = delta_0/l_plus
        Re_0 = U_e*delta_0*rho_e/mu_e
        Re_1 = U_e*delta_1*rho_e/mu_e
        Re_2 = U_e*delta_2*rho_e/mu_e

        # SPANWISE CORRELATION
        if is_correlation:
            primep = jnp.stack([rhop, up, vp, wp, pp], axis=1)
            y_ids = [jnp.argmin(jnp.abs(y_plus[x1_id:x2_id] - value), axis=1) for value in y_plus_positions]
            primep = primep[...,x1_id:x2_id,:,:]
            correlations_list = []
            mean_axes = (0,2,3)
            for y_id in y_ids:
                primep_corr = primep[...,jnp.arange(len(y_id)),y_id,:]
                correlation_y_list = []
                primep_primep_corr_0 = jnp.mean(primep_corr*primep_corr, axis=mean_axes)
                for ii in range(Nz // 2):
                    R_ii = jnp.mean(primep_corr * jnp.roll(primep_corr, shift=ii, axis=-1), axis=mean_axes)
                    correlation_y_list.append(R_ii)
                correlation_y = jnp.array(correlation_y_list)
                correlation_y /= primep_primep_corr_0
                correlations_list.append(correlation_y)
            correlations = jnp.array(correlations_list)
        else:
            correlations = 0.0

        if is_energy_spectra:
            Nx = x2_id - x1_id
            y_ids = [jnp.argmin(jnp.abs(y_plus[x1_id:x2_id] - value), axis=1) for value in y_plus_positions]
            primep = jnp.stack([rhop, up, vp, wp, pp], axis=1)
            primep = primep[...,x1_id:x2_id,:,:]
            Ek = jnp.zeros((len(y_ids), 5, Nz))
            for ii, y_id in enumerate(y_ids):
                primep_hat = primep[...,jnp.arange(len(y_id)),y_id,:]
                primep_hat = jnp.fft.fft(primep_hat, axis=-1)
                primep_hat = jnp.moveaxis(primep_hat, 2, 1).reshape(Nsamples * Nx, 5, Nz)
                for kk in range(5):
                    ek = jax.vmap(energy_spectrum_1D_spectral)(primep_hat[:,kk:kk+1,:])
                    Ek = Ek.at[ii,kk].add(jnp.mean(ek, axis=0))
        else:
            Ek = 0.0
        
        turb_stats_dict = {
            "MEAN_VALUES": {
                "RHO_MEAN": rho_mean[x_id],
                "U_MEAN": u_mean[x_id],
                "V_MEAN": v_mean[x_id],
                "W_MEAN": w_mean[x_id],
                "P_MEAN": p_mean[x_id],
                "T_MEAN": T_mean[x_id],
                "MU_MEAN": mu_mean[x_id],
            },
            "THERMODYNAMIC_RMS":{
                "RHO_RMS": rhop_rhop_rms[x_id],
                "T_RMS": Tp_Tp_rms[x_id],
                "P_RMS": pp_pp_rms[x_id]
            },
            "VELOCITY_SCALES": {
                "U_TAU": u_tau[x_id],
                "TAU_WALL": tau_wall[x_id],
                "U_VD": u_vd[x_id],
                "C_F": Cf,
            },
            "LENGTH_SCALES": {
                "L_PLUS": l_plus[x_id],
                "Y_PLUS": y_plus[x_id],
                "DELTA_Y+_MIN": delta_y_min_plus,
                "DELTA_X+_MIN": delta_x_min_plus,
                "DELTA_Z+_MIN": delta_z_min_plus,
            },
            "SPECTRA_AND_CORRELATIONS": {
                "ENERGY_SPECTRUM": Ek,
                "CORRELATIONS": correlations
            },
            "LAYER_THICKNESS": {
                "DELTA_0": delta_0,
                "DELTA_1": delta_1,
                "DELTA_2": delta_2,
                "DELTA_C": delta_c
            },
            "REYNOLDS_NUMBERS": {
                "RE_TAU": Re_tau,
                "RE_0": Re_0,
                "RE_1": Re_1,
                "RE_2": Re_2,
            },
            "REYNOLDS_STRESSES": {
                "up_up": up_up_mean[x_id], "vp_vp": vp_vp_mean[x_id], "wp_wp": wp_wp_mean[x_id],
                "up_vp": up_vp_mean[x_id], "up_wp": up_wp_mean[x_id], "vp_wp": vp_wp_mean[x_id]
            }
        }

        return turb_stats_dict

def turbulent_statistics_for_logging(
        turbulent_statistics_manager: TurbulentStatisticsManager,
        primitives: Array,
        turbulent_case: str
    ) -> Dict:
    """Computes the turbulent statistics for the given primitive buffer.
    Subsequently, statistics are formatted for print output.

    :param primitives: Buffer of primitive variables.
    :type primitives: Array
    :return: Dictionary with turbulent statistics.
    :rtype: Dict
    """
    if turbulent_case == "HIT":
        turbulent_stats = turbulent_statistics_manager.hit_statistics(primitives)
        turbulent_stats = {key: turbulent_stats[key] for key in [
            "THERMODYNAMICS", "VELOCITY_SCALES", "LENGTH_SCALES",
            "TIME_SCALES", "REYNOLDS_NUMBERS"
        ]}
    elif turbulent_case == "CHANNEL":
        turbulent_stats = turbulent_statistics_manager.channel_statistics(primitives)
        turbulent_stats = {key: turbulent_stats[key] for key in [
            "THERMODYNAMICS", "VELOCITY_SCALES", "LENGTH_SCALES",
            "TIME_SCALES", "REYNOLDS_NUMBERS"
        ]}
    
    elif turbulent_case == "BOUNDARYLAYER":
        turbulent_stats = turbulent_statistics_manager.boundarylayer_statistics(primitives)
        turbulent_stats = {key: turbulent_stats[key] for key in [
            "THERMODYNAMICS", "VELOCITY_SCALES", "LENGTH_SCALES",
            "TIME_SCALES", "REYNOLDS_NUMBERS"
        ]}

    elif turbulent_case == "DUCT":
        raise NotImplementedError

    elif turbulent_case == "TGV":
        # TODO DENIZ
        turbulent_stats = {}

    else:
        raise NotImplementedError

    return turbulent_stats
