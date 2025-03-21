from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.turbulence.statistics.online.turbulence_statistics_computer import TurbulenceStatisticsComputer
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager

from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.data_types.statistics import (
    ChannelStatisticsLogging, TurbulenceStatisticsInformation,
    StatisticsLogging, Metrics, StatisticsCumulative)

from jaxfluids.turbulence.statistics.utilities import van_driest_transform

Array = jax.Array


class ChannelStatisticsComputer(TurbulenceStatisticsComputer):


    def __init__(
            self,
            turbulence_statistics_setup: TurbulenceStatisticsSetup, 
            domain_information: DomainInformation,
            material_manager: MaterialManager
        ) -> None:
        super().__init__(turbulence_statistics_setup, domain_information, material_manager)

        self.s_axes = (-1,-3)

        self.reynolds_means_keys = [
            "density", "velX", "velY", "velZ", "pressure",
            "kineticEnergy", "temperature", "speedOfSound",
            "machNumber", "mu", "energy"
        ]

        self.favre_means_keys = [
            "velX", "velY", "velZ", "kineticEnergy",
            "temperature", "energy"
        ]

        self.reynolds_cov_keys = [
            "velX_velX", "velY_velY", "velZ_velZ",
            "velX_velY", "velX_velZ", "velY_velZ",
            "temperature_temperature", "velX_temperature",
            "velY_temperature", "velZ_temperature",
            "density_density", "pressure_pressure",
            "machNumber_machNumber"
        ]

        self.favre_cov_keys = [
            "velX_velX", "velY_velY", "velZ_velZ",
            "velX_velY", "velX_velZ", "velY_velZ",
            "temperature_temperature", "velX_temperature",
            "velY_temperature", "velZ_temperature",
        ]

    def initialize_statistics(self) -> TurbulenceStatisticsInformation:
        assert_string = (
            "Turbulence statistics for CHANNEL cases with "
            "multiple devices in wall-normal direction (y-axis) "
            "are currently not supported. Please use a splitting "
            "in x- or z-axis.")
        assert self.split_factors[1] == 1, assert_string

        statistics_logging = None
        if self.is_logging:
            channel_statistics = ChannelStatisticsLogging()
            statistics_logging = StatisticsLogging(
                channel=channel_statistics)
            
        statistics_cumulative = None
        if self.is_cumulative:

            reynolds_means = {k: jnp.zeros((1,self.Ny,1)) for k in self.reynolds_means_keys}
            favre_means = {k: jnp.zeros((1,self.Ny,1)) for k in self.favre_means_keys}
            reynolds_covs = {k: jnp.zeros((1,self.Ny,1)) for k in self.reynolds_cov_keys}
            favre_covs = {k: jnp.zeros((1,self.Ny,1)) for k in self.favre_cov_keys}

            metrics = Metrics(
                sampling_dt=self.sampling_dt,
                next_sampling_time=self.start_sampling,
                sample_steps=0, total_sample_points=0,
                total_sample_weights=0.0, 
                reynolds_means=reynolds_means,
                favre_means=favre_means,
                reynolds_covs=reynolds_covs,
                favre_covs=favre_covs)
            
            statistics_cumulative = StatisticsCumulative(metrics)

        turbulent_statistics = TurbulenceStatisticsInformation(
            statistics_logging, statistics_cumulative)

        return turbulent_statistics


    def compute_cumulative_statistics(
            self, 
            primitives: Array, 
            cumulative_statistics: StatisticsCumulative
        ) -> StatisticsCumulative:
           
        density, velX, velY, velZ, pressure = primitives
        kinetic_energy = jnp.sum(jnp.square(primitives[self.s_velocity]), axis=0)
        temperature = self.material_manager.get_temperature(
            pressure=pressure, density=density)
        speed_of_sound = self.material_manager.get_speed_of_sound(
            pressure=pressure, density=density)
        mach_number = jnp.sqrt(kinetic_energy) / speed_of_sound
        mu = self.material_manager.get_dynamic_viscosity(temperature, None)
        energy = self.material_manager.get_specific_energy(pressure, density)

        quantities = {
            "density": density, "velX": velX, "velY": velY, "velZ": velZ, 
            "pressure": pressure, "kineticEnergy": kinetic_energy, 
            "temperature": temperature, "speedOfSound": speed_of_sound,
            "machNumber": mach_number, "mu": mu, "energy": energy
        }

        metrics = self.update_metrics(quantities, cumulative_statistics.metrics)
        channel_statistics = self.compute_channel_statistics(metrics)

        statistics_cumulative = StatisticsCumulative(metrics, channel=channel_statistics)

        return statistics_cumulative

    def compute_logging_statistics(self, primitives: Array) -> StatisticsLogging:
        
        # CHANNEL SIZES AND CELL SIZES
        global_domain_size_y = self.domain_information.get_global_domain_size()[1]
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            global_domain_size_y = global_domain_size_y[device_id]
        cell_sizes = self.domain_information.get_device_cell_sizes()
        cell_sizes_x, cell_sizes_y, cell_sizes_z = [jnp.squeeze(dxi) for dxi in cell_sizes]

        channel_height = global_domain_size_y[1] - global_domain_size_y[0]
        half_channel_height = 0.5 * channel_height   # Half channel height

        density = primitives[self.ids_mass]
        velX = primitives[self.ids_velocity[0]]
        pressure = primitives[self.ids_energy]
        temperature = self.material_manager.get_temperature(pressure=pressure, density=density)
        speed_of_sound = self.material_manager.get_speed_of_sound(pressure=pressure, density=density)
        mu = self.material_manager.get_dynamic_viscosity(temperature, primitives)
        
        # AVERAGING
        quantities = {
            "density": density, "velX": velX, "pressure": pressure,
            "temperature": temperature, "speed_of_sound": speed_of_sound,
            "mu": mu
        }
        reynolds_means = {
            k: jnp.mean(quantities[k], axis=self.s_axes, keepdims=False)
            for k in quantities
        }
        if self.is_parallel:
            reynolds_means = jax.lax.pmean(reynolds_means, axis_name="i")

        bulk_means = {
            k: jnp.sum(cell_sizes_y * reynolds_means[k]) / channel_height
            for k in reynolds_means
        }


        # THERMODYNAMICS
        density_mean = bulk_means["density"]
        density_mean_y = reynolds_means["density"]
        density_wall = 0.5 * (density_mean_y[0] + density_mean_y[-1])

        pressure_mean = bulk_means["pressure"]
        temperature_mean = bulk_means["temperature"]

        # VISCOSITY
        mu_mean_y = reynolds_means["mu"]
        mu_wall = 0.5 * (mu_mean_y[0] + mu_mean_y[-1])
        nu_wall = mu_wall / density_wall

        # BULK VELOCITY AND CENTER LINE VELOCITY
        velX_mean = bulk_means["velX"]
        velX_mean_y = reynolds_means["velX"]
        velX_wall = 0.5 * (velX_mean_y[0] + velX_mean_y[-1])
        velX_0 = jnp.max(velX_mean_y)

        # TURBULENT SCALES
        # WALL SHEAR STRESS EQ. (7.24)
        dudy_wall = velX_wall / (0.5 * cell_sizes_y[0])
        tau_wall = mu_wall * dudy_wall
        # FRICTION VELOCITY EQ. (7.25)
        u_tau = jnp.sqrt(tau_wall / density_wall)

        # MACH NUMBER
        speed_of_sound_mean = bulk_means["speed_of_sound"]
        speed_of_sound_mean_y = reynolds_means["speed_of_sound"]
        Ma_bulk = velX_mean / speed_of_sound_mean
        Mach_mean_y = velX_mean_y / speed_of_sound_mean_y
        Ma_0 = jnp.max(Mach_mean_y)

        # Viscous lengthscale Eq. (7.26)
        delta_nu = nu_wall / u_tau # Viscous lengthscale Eq. (7.26)
        one_delta_nu = 1.0 / delta_nu
        delta_x_plus = cell_sizes_x * one_delta_nu     # Pope 7.28
        delta_y_plus_min = jnp.min(cell_sizes_y) * one_delta_nu
        delta_y_plus_max = jnp.max(cell_sizes_y) * one_delta_nu
        delta_z_plus = cell_sizes_z * one_delta_nu

        # REYNOLDS NUMBER
        one_nu_wall = 1.0 / nu_wall
        Re_tau = half_channel_height * u_tau * one_nu_wall  # Pope 7.27
        Re_bulk = channel_height * velX_mean * one_nu_wall     # Pope 7.1
        Re_0 = half_channel_height * velX_0 * one_nu_wall   # Pope 7.2

        channel_statistics = ChannelStatisticsLogging(
            rho_bulk=density_mean, pressure_bulk=pressure_mean, temperature_bulk=temperature_mean,
            u_bulk=velX_mean, mach_bulk=Ma_bulk, mach_0=Ma_0, reynolds_tau=Re_tau,
            reynolds_bulk=Re_bulk, reynolds_0=Re_0, delta_x_plus=delta_x_plus,
            delta_y_plus_min=delta_y_plus_min, delta_y_plus_max=delta_y_plus_max,
            delta_z_plus=delta_z_plus)

        statistics_logging = StatisticsLogging(
            channel=channel_statistics)

        return statistics_logging

    def compute_channel_statistics(self, metrics: Metrics) -> Dict:

        reynolds_means = metrics.reynolds_means
        favre_means = metrics.favre_means
        reynolds_covs = metrics.reynolds_covs
        favre_covs = metrics.favre_covs

        reynolds_means = {k: jnp.squeeze(reynolds_means[k]) for k in reynolds_means}
        favre_means = {k: jnp.squeeze(favre_means[k]) for k in favre_means}
        reynolds_covs = {k: jnp.squeeze(reynolds_covs[k]) for k in reynolds_covs}
        favre_covs = {k: jnp.squeeze(favre_covs[k]) for k in favre_covs}

        cell_centers_y = self.domain_information.get_device_cell_centers()[1]
        cell_centers_y = jnp.squeeze(cell_centers_y)
        cell_sizes = self.domain_information.get_device_cell_sizes()
        cell_sizes_x, cell_sizes_y, cell_sizes_z = [jnp.squeeze(dxi) for dxi in cell_sizes]

        domain_size = self.domain_information.get_global_domain_size()
        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            domain_size = [domain_size_xi[device_id] for domain_size_xi in domain_size]


        channel_length = domain_size[0][1] - domain_size[0][0]
        channel_height = domain_size[1][1] - domain_size[1][0]
        half_channel_height = 0.5 * channel_height
        wall_normal_distance = jnp.minimum(
            jnp.abs(cell_centers_y - domain_size[1][0]), 
            jnp.abs(cell_centers_y - domain_size[1][1])
        )

        density_mean = reynolds_means["density"]
        velX_mean = reynolds_means["velX"]
        velY_mean = reynolds_means["velY"]
        velZ_mean = reynolds_means["velZ"]
        pressure_mean = reynolds_means["pressure"]
        temperature_mean = reynolds_means["temperature"]
        mu_mean = reynolds_means["mu"]
        speed_of_sound_mean = reynolds_means["speedOfSound"]
        kinetic_energy = reynolds_means["kineticEnergy"]

        density_rms = reynolds_covs["density_density"]
        pressure_rms = reynolds_covs["pressure_pressure"]
        temperature_rms = reynolds_covs["temperature_temperature"]

        density_bulk = jnp.sum(density_mean * cell_sizes_y)
        velX_bar = jnp.sum(velX_mean * cell_sizes_y) / channel_height
        velX_bulk = jnp.sum(velX_mean * density_mean * cell_sizes_y) / (channel_height * density_bulk)
        velX_0 = jnp.max(velX_mean)

        rho_wall = 0.5 * (density_mean[0] + density_mean[-1])
        u_wall = 0.5 * (velX_mean[0] + velX_mean[-1])
        speed_of_sound_wall = 0.5 * (speed_of_sound_mean[0] + speed_of_sound_mean[-1])
        mu_wall = 0.5 * (mu_mean[0] + mu_mean[-1])
        nu_wall = mu_wall / rho_wall

        # NOTE Velocity scales
        dudy_wall = u_wall / (0.5 * cell_sizes_y[0])
        tau_wall = mu_wall * dudy_wall
        u_tau = jnp.sqrt(jnp.abs(tau_wall) / rho_wall)
        turbulent_kinetic_energy = sum([
            reynolds_covs[f"vel{xi}_vel{xi}"] for xi in ("X","Y","Z")
        ])

        velX_plus = velX_mean / u_tau
        velX_VD = van_driest_transform(velX_mean, density_mean)
        velX_plus_VD = velX_VD / u_tau


        # NOTE Length scales
        delta_nu = nu_wall / u_tau
        one_delta_nu = 1.0 / delta_nu
        delta_x_plus = cell_sizes_x * one_delta_nu
        delta_y_plus_min = jnp.min(cell_sizes_y) * one_delta_nu
        delta_y_plus_max = jnp.max(cell_sizes_y) * one_delta_nu
        delta_z_plus = cell_sizes_z * one_delta_nu
        y_plus = one_delta_nu * wall_normal_distance

        # NOTE Reynolds numbers
        Re_tau = half_channel_height * u_tau / nu_wall              # Pope 7.27
        Re_bulk = channel_height * velX_bar / nu_wall               # Pope 7.1
        Re_0 = half_channel_height * velX_0 / nu_wall               # Pope 7.2
        Re_coleman = half_channel_height * velX_bulk * density_mean / mu_wall

        # NOTE Mach numbers
        one_speed_of_sound_mean = 1.0 / speed_of_sound_mean
        Ma_bulk = velX_bar * one_speed_of_sound_mean
        Ma_tau = u_tau / speed_of_sound_wall
        Mach_mean_y = velX_mean * one_speed_of_sound_mean
        Ma_0 = jnp.max(Mach_mean_y)
        Mach_turbulent_y = jnp.sqrt(turbulent_kinetic_energy) * one_speed_of_sound_mean
        Mach_rms_y = jnp.sqrt(reynolds_covs["machNumber_machNumber"])

        # NOTE Time scales
        ftt_outer = channel_length / velX_bar
        ftt_inner = channel_length / u_tau
        # See Lechner et al. - 2001 - Turbulent supersonic channel flow
        char_inner = half_channel_height / u_tau


        statistics = {
            # "THERMODYNAMICS": {
            #     "P_MEAN": p_mean,
            #     "P_RMS": p_rms,
            #     "RHO_MEAN": rho_mean,
            #     "RHO_RMS": rho_rms,
            #     "T_MEAN": T_mean,
            #     "T_RMS": T_rms,
            #     "MU_MEAN": mu_mean,
            # },
            "THERMODYNAMICS_Y": {
                "P_MEAN": pressure_mean,
                "P_RMS": pressure_rms,
                "RHO_MEAN": density_mean,
                "RHO_RMS": density_rms,
                "T_MEAN": temperature_mean,
                "T_RMS": temperature_rms,
                "MU_MEAN": mu_mean,
            },
            "VELOCITY_SCALES": {
                "velX_BAR": velX_bar,
                "velX_BULK": velX_bulk,
                "velX_0": velX_0,
                "U_TAU": u_tau,
                "TAU_WALL": tau_wall,
                "C_WALL": speed_of_sound_wall,
                "MA_BULK": Ma_bulk,
                "MA_0": Ma_0,
                "MA_TAU": Ma_tau
            },
            "VELOCITY_SCALES_Y": {
                "velX_MEAN": velX_mean,
                "velY_MEAN": velY_mean,
                "velZ_MEAN": velZ_mean,
                "U+": velX_plus,
                "U_VD": velX_VD,
                "U_VD+": velX_plus_VD,
                "MA_MEAN": Mach_mean_y,
                "MA_RMS": Mach_rms_y,
                "MA_T": Mach_turbulent_y,
            },
            "LENGTH_SCALES": {
                "DELTA": half_channel_height,
                "DELTA_NU": delta_nu,
                "DELTA_X+": delta_x_plus,
                "DELTA_Y+_MIN": delta_y_plus_min,
                "DELTA_Y+_MAX": delta_y_plus_max,
                "DELTA_Z+": delta_z_plus,
                "Y_PLUS": y_plus
            },
            "TIME_SCALES": {
                "FTT_OUTER": ftt_outer,
                "FTT_INNER": ftt_inner,
                "CHAR_INNER": char_inner
            },
            "REYNOLDS_NUMBERS": {
                "RE_TAU": Re_tau,
                "RE_BULK": Re_bulk,
                "RE_0": Re_0,
                "RE_COLEMAN": Re_coleman
            },
            "REYNOLDS_MEANS": reynolds_means,
            "FAVRE_MEANS": favre_means,
            "REYNOLDS_COVARIANCES": reynolds_covs,
            "FAVRE_COVARIANCES": favre_covs
            # "rhop_up_up": rhop_up_up, "rhop_vp_vp": rhop_vp_vp,
            # "rhop_wp_wp": rhop_wp_wp, "rhop_up_vp": rhop_up_vp,
            # "rhop_up_wp": rhop_up_wp, "rhop_vp_wp": rhop_vp_wp,
            # "rhop_Tp_Tp": rhop_Tp_Tp, "rhop_vp_Tp": rhop_vp_Tp,
            # "upp_upp": upp_upp, "vpp_vpp": vpp_vpp, "wpp_wpp": wpp_wpp,
            # "upp_vpp": upp_vpp, "upp_wpp": upp_wpp, "vpp_wpp": vpp_wpp,
            # "Tpp_Tpp": Tpp_Tpp, "vpp_Tpp": vpp_Tpp,
        }

        return statistics

