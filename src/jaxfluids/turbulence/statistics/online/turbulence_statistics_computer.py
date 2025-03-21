from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np


from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.domain.helper_functions import reassemble_buffer
from jaxfluids.config import precision
from jaxfluids.data_types.buffers import TimeControlVariables

from jaxfluids.data_types.statistics import (
    TurbulenceStatisticsInformation, 
    StatisticsLogging,
    StatisticsCumulative,
    Metrics)

from jaxfluids.turbulence.statistics.online.helper_functions import update_mean, update_sum_square, update_sum_square_cov, \
    parallel_mean, parallel_sum

Array = jax.Array


class TurbulenceStatisticsComputer(ABC):
    """ Provides functionality to calculate statistics of turbulent flows.
    The TurbulenceStatisticsComputer provides turbulence statistics of the initial flow
    field as well as cumulative statistics over the course of a simulation.

    TODO think about namespaces for statistics outputs?
    """
    
    def __init__(
            self,
            turbulence_statistics_setup: TurbulenceStatisticsSetup,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            ) -> None:

        self.domain_information = domain_information
        self.equation_information = material_manager.equation_information
        self.material_manager = material_manager

        self.turbulence_case = turbulence_statistics_setup.case
        self.is_logging = turbulence_statistics_setup.is_logging
        self.is_cumulative = turbulence_statistics_setup.is_cumulative
        self.start_sampling = turbulence_statistics_setup.start_sampling
        self.sampling_dt = turbulence_statistics_setup.sampling_dt
        self.streamwise_measure_position = turbulence_statistics_setup.streamwise_measure_position

        self.Nx, self.Ny, self.Nz = domain_information.global_number_of_cells
        self.cells_per_device = domain_information.cells_per_device
        self.device_number_of_cells = domain_information.device_number_of_cells
        self.is_parallel = domain_information.is_parallel
        self.split_factors = domain_information.split_factors

        self.ids_mass = self.equation_information.ids_mass
        self.ids_velocity = self.equation_information.ids_velocity
        self.s_velocity = self.equation_information.s_velocity
        self.ids_energy = self.equation_information.ids_energy

        self.eps = precision.get_eps()

        domain_size = self.domain_information.get_local_domain_size()
        is_parallel = self.domain_information.is_parallel
        no_subdomains = self.domain_information.no_subdomains
        
        x = self.streamwise_measure_position
        if is_parallel:
            mask = []
            for i in range(no_subdomains):
                domain_size_x = domain_size[0][i]
                if x > domain_size_x[0] and x <= domain_size_x[1]:
                    mask.append(1)
                else:
                    mask.append(0)
            self.device_mask_measure_position = jnp.array(mask)
        else:
            self.device_mask_measure_position = None

        self.s_axes: Tuple[int] = None
        self.reynolds_means_keys: List[str] = None
        self.favre_means_keys: List[str] = None
        self.reynolds_cov_keys: List[str] = None
        self.favre_cov_keys: List[str] = None


    @abstractmethod
    def initialize_statistics(self) -> TurbulenceStatisticsInformation:
        """Abstract method which initializes statistics container 
        corresponding to the present turbulent case, i.e., self.turbulence_case.

        :return: _description_
        :rtype: TurbulenceStatisticsInformation
        """


    @abstractmethod
    def compute_logging_statistics(self, primitives: Array) -> StatisticsLogging:
        pass

    
    @abstractmethod
    def compute_cumulative_statistics(
        self, primitives: Array, 
        cumulative_statistics: StatisticsCumulative
        ) -> StatisticsCumulative:
        pass


    def compute_turbulent_statistics(
            self,
            primitives: Array,
            turbulent_statistics: TurbulenceStatisticsInformation,
            is_cumulative: bool,
            is_logging: bool,
        ) -> TurbulenceStatisticsInformation:
        """Computes the turbulence statistics (logging and/or cumulative) for
        the given primitive buffer.

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulenceStatisticsInformation
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: TurbulenceStatisticsInformation
        """

        if is_logging:
            logging_statistics = self.compute_logging_statistics(primitives)
        else:
            logging_statistics = turbulent_statistics.logging

        if is_cumulative:
            cumulative_statistics = self.compute_cumulative_statistics(
                primitives, turbulent_statistics.cumulative)
        else:
            cumulative_statistics = turbulent_statistics.cumulative

        turbulent_statistics = TurbulenceStatisticsInformation(
            logging_statistics, cumulative_statistics)

        return turbulent_statistics


    def get_statistics_flags(
            self,
            turbulence_statistics: TurbulenceStatisticsInformation,
            logging_frequency: int,
            time_control_variables: TimeControlVariables,
            ) -> Tuple[bool]:

        physical_simulation_time = time_control_variables.physical_simulation_time
        simulation_step = time_control_variables.simulation_step

        if self.is_cumulative:
            next_sampling_time = turbulence_statistics.cumulative.metrics.next_sampling_time
            if physical_simulation_time >= next_sampling_time:
                is_cumulative_statistics = True
            else:
                is_cumulative_statistics = False
        else:
            is_cumulative_statistics = False

        if self.is_logging and simulation_step % logging_frequency == 0:
            is_logging_statistics = True
        else:
            is_logging_statistics = False

        return is_cumulative_statistics, is_logging_statistics


    def update_metrics(
            self,
            quantities: Dict[str, Array], 
            metrics: Metrics
            ) -> Metrics:

        sampling_dt = metrics.sampling_dt
        next_sampling_time = metrics.next_sampling_time
        sample_steps = metrics.sample_steps
        N_agg = metrics.total_sample_points
        Omega_agg = metrics.total_sample_weights
        reynolds_means = metrics.reynolds_means
        favre_means = metrics.favre_means
        reynolds_covs = metrics.reynolds_covs
        favre_covs = metrics.favre_covs

        nx,ny,nz = self.domain_information.global_number_of_cells
        N_new = nx * nz
        Omega_new = parallel_sum(quantities["density"], self.is_parallel, self.s_axes, True)

        next_sampling_time = next_sampling_time + sampling_dt
        sample_steps = sample_steps + 1
        total_sample_points = N_agg + N_new
        total_sample_weights = Omega_agg + Omega_new

        reynolds_means_new, favre_means_new = self.compute_means(quantities)

        reynolds_fluctuations_new = {
            k: quantities[k] - reynolds_means_new[k] for k in reynolds_means_new
        }
        favre_fluctuations_new = {
            k: quantities[k] - favre_means_new[k] for k in favre_means_new
        }

        reynolds_covs_new, favre_cov_new \
        = self.compute_covariances(
            reynolds_fluctuations_new, favre_fluctuations_new, quantities["density"])

        # UPDATE AGGREGATE COV
        for k in reynolds_covs:
            k1,k2 = k.split("_")
            reynolds_covs[k] = update_sum_square_cov(
                reynolds_covs[k] * N_agg, reynolds_means[k1], reynolds_means[k2], N_agg,
                reynolds_covs_new[k], reynolds_means_new[k1], reynolds_means[k2], N_new
            ) / total_sample_points
        for k in favre_covs:
            k1,k2 = k.split("_")
            favre_covs[k] = update_sum_square_cov(
                favre_covs[k] * Omega_agg, favre_means[k1], favre_means[k2], Omega_agg,
                favre_cov_new[k], favre_means_new[k1], favre_means_new[k2], Omega_new
            ) / total_sample_weights

        # UPDATE AGGREGATE MEANS
        for k in reynolds_means:
            reynolds_means[k] = update_mean(
                reynolds_means[k], N_agg,
                reynolds_means_new[k], N_new
            )
        for k in favre_means:
            favre_means[k] = update_mean(
                favre_means[k], N_agg,
                favre_means_new[k], N_new
            )

        metrics = Metrics(
            sampling_dt=sampling_dt,
            next_sampling_time=next_sampling_time,
            sample_steps=sample_steps,
            total_sample_points=total_sample_points,
            total_sample_weights=total_sample_weights,
            reynolds_means=reynolds_means,
            favre_means=favre_means,
            reynolds_covs=reynolds_covs,
            favre_covs=favre_covs)

        return metrics
    

    def compute_means(
        self, 
        quantities: Dict[str, Array]
        ) -> Tuple[Dict[str, Array], Dict[str, Array]]:

        # reynolds_means = {
        #     k: parallel_mean(quantities[k], self.is_parallel, self.s_axes, True)
        #     for k in self.reynolds_means_keys
        # }
        reynolds_means = {
            k: jnp.mean(quantities[k], axis=self.s_axes, keepdims=True)
            for k in self.reynolds_means_keys
        }

        one_density_mean = 1.0 / reynolds_means["density"]  # 1.0 / density_mean
        density = quantities["density"]
        # favre_means = {
        #     k: parallel_mean(density * quantities[k], self.is_parallel, self.s_axes, True) * one_density_mean 
        #     for k in self.favre_means_keys 
        # }
        favre_means = {
            k: jnp.mean(density * quantities[k], axis=self.s_axes, keepdims=True) * one_density_mean 
            for k in self.favre_means_keys 
        }

        if self.is_parallel:
            reynolds_means = jax.lax.pmean(reynolds_means, axis_name="i")
            favre_means = jax.lax.pmean(favre_means, axis_name="i")

        return reynolds_means, favre_means

    def compute_covariances(
        self, 
        reynolds_fluctuations: Dict[str, Array],
        favre_fluctuations: Dict[str, Array],
        density: Array
        ) -> Tuple[Dict[str, Array], Dict[str, Array]]:


        # CALCULATE NEW SUM OF SQUARES
        reynolds_sum_products = {}
        # for k in self.reynolds_cov_keys:
        #     k1, k2 = k.split("_")
        #     reynolds_sum_products[k] = parallel_sum(
        #         reynolds_fluctuations[k1] * reynolds_fluctuations[k2],
        #         self.is_parallel, self.s_axes, True)
        for k in self.reynolds_cov_keys:
            k1, k2 = k.split("_")
            reynolds_sum_products[k] = jnp.sum(
                reynolds_fluctuations[k1] * reynolds_fluctuations[k2],
                axis=self.s_axes, keepdims=True)

        favre_sum_products = {}
        for k in self.favre_cov_keys:
            k1, k2 = k.split("_")
            # favre_sum_products[k] = parallel_sum(
            #     density * favre_fluctuations[k1] * favre_fluctuations[k2],\
            #     self.is_parallel, self.s_axes, True)
            favre_sum_products[k] = jnp.sum(
                density * favre_fluctuations[k1] * favre_fluctuations[k2],\
                axis=self.s_axes, keepdims=True)
        
        if self.is_parallel:
            reynolds_sum_products = jax.lax.pmean(reynolds_sum_products, axis_name="i")
            favre_sum_products = jax.lax.pmean(favre_sum_products, axis_name="i")
        
        return reynolds_sum_products, favre_sum_products