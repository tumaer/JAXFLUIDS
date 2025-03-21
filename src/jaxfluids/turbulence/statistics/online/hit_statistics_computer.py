from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.turbulence.statistics.online.helper_functions import update_sum_square, update_sum_square_cov
from jaxfluids.turbulence.statistics.online.turbulence_statistics_computer import TurbulenceStatisticsComputer
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager

from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.data_types.statistics import (
    HITStatisticsLogging, TurbulenceStatisticsInformation,
    StatisticsLogging, Metrics, StatisticsCumulative)

Array = jax.Array


class HITStatisticsComputer(TurbulenceStatisticsComputer):

    def __init__(
            self,
            turbulence_statistics_setup: TurbulenceStatisticsSetup,
            domain_information: DomainInformation,
            material_manager: MaterialManager
        ) -> None:
        super().__init__(turbulence_statistics_setup, domain_information, material_manager)

        self.s_axes = (-1,-2,-3)

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

        statistics_logging = None
        if self.is_logging:
            hit_stats_logging = HITStatisticsLogging()
            statistics_logging = StatisticsLogging(
                hit=hit_stats_logging)

        statistics_cumulative = None
        if self.is_cumulative:
            # TODO maybe shape not needed??
            reynolds_means = {k: jnp.zeros((1,1,1)) for k in self.reynolds_means_keys}
            favre_means = {k: jnp.zeros((1,1,1)) for k in self.favre_means_keys}
            reynolds_covs = {k: jnp.zeros((1,1,1)) for k in self.reynolds_cov_keys}
            favre_covs = {k: jnp.zeros((1,1,1)) for k in self.favre_cov_keys}

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
        hit_statistics = self.compute_hit_statistics(cumulative_statistics)

        statistics_cumulative = StatisticsCumulative(metrics, hit=hit_statistics)

        return cumulative_statistics


    def compute_logging_statistics(
            self, primitives: Array, 
            ) -> StatisticsLogging:
        
        density = primitives[self.equation_information.ids_mass]
        velocity = primitives[self.equation_information.s_velocity]
        pressure = primitives[self.equation_information.ids_energy]
        temperature = self.material_manager.get_temperature(pressure=pressure, density=density)

        density_mean = jnp.mean(density)
        velocity_mean = jnp.mean(velocity, axis=(-1,-2,-3))
        pressure_mean = jnp.mean(pressure)
        temperature_mean = jnp.mean(temperature)

        if self.is_parallel:
            density_mean = jax.lax.pmean(density_mean, axis_name="i")
            velocity_mean = jax.lax.pmean(velocity_mean, axis_name="i")
            pressure_mean = jax.lax.pmean(pressure_mean, axis_name="i")
            temperature_mean = jax.lax.pmean(temperature_mean, axis_name="i")

        speed_of_sound_mean = self.material_manager.get_speed_of_sound(pressure=pressure_mean, density=density_mean)

        density_prime = density - density_mean
        velocity_prime = velocity - velocity_mean[:,None,None,None]
        pressure_prime = pressure - pressure_mean
        temperature_prime = temperature - temperature_mean
        density_prime = density - density_mean

        density_rms = jnp.mean(jnp.square(density_prime))
        velocity_rms = jnp.mean(jnp.square(velocity_prime), axis=(-1,-2,-3))
        pressure_rms = jnp.mean(jnp.square(pressure_prime))
        temperature_rms = jnp.mean(jnp.square(temperature_prime))

        if self.is_parallel:
            density_rms = jax.lax.pmean(density_rms, axis_name="i")
            velocity_rms = jax.lax.pmean(velocity_rms, axis_name="i")
            pressure_rms = jax.lax.pmean(pressure_rms, axis_name="i")
            temperature_rms = jax.lax.pmean(temperature_rms, axis_name="i")

        density_rms = jnp.sqrt(density_rms)
        pressure_rms = jnp.sqrt(pressure_rms)
        temperature_rms = jnp.sqrt(temperature_rms)
        q_rms = jnp.sqrt(jnp.sum(velocity_rms))
        u_rms = q_rms / jnp.sqrt(3.0)
        mach_rms = q_rms / speed_of_sound_mean

        hit_statistics = HITStatisticsLogging(
            rho_bulk=density_mean, pressure_bulk=pressure_mean,
            temperature_bulk=temperature_mean,
            rho_rms=density_rms, pressure_rms=pressure_rms,
            temperature_rms=temperature_rms,
            u_rms=u_rms, mach_rms=mach_rms)

        statistics_logging = StatisticsLogging(
            hit_statistics=hit_statistics)

        return statistics_logging

    def compute_hit_statistics(self, metrics: Metrics) -> Dict:
        raise NotImplementedError 