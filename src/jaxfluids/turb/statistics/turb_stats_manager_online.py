from functools import partial
from typing import Callable, Dict, Tuple, List

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from jaxfluids.data_types.information import TurbulentStatisticsInformation, \
    HITStatisticsLogging, ChannelStatisticsLogging, StatisticsLogging, \
    HITStatisticsCumulative, ChannelStatisticsCumulative, \
    StatisticsCumulative, BoundaryLayerStatisticsLogging, BoundaryLayerStatisticsCumulative
from jaxfluids.data_types.numerical_setup.turbulence_statistics import TurbulenceStatisticsSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.domain.helper_functions import reassemble_buffer
from jaxfluids.turb.statistics.utilities import (
    rfft3D, real_wavenumber_grid,
    _calculate_sheartensor_spectral,
    calculate_vorticity_spectral, 
    calculate_dilatation_spectral,
    _helmholtz_projection,
    reynolds_average, favre_average, van_driest_transform)
from jaxfluids.config import precision

def update_sum_square(
        S: Array,
        S_new: Array,
        T: Array,
        T_new: Array,
        N: int,
        N_new: int
        ) -> Array:
    # TODO references
    factor_1 = N / (N_new * (N + N_new) + 1e-20)
    factor_2 = N_new * N/ (N**2 + 1e-20)
    S_star = S + S_new + factor_1 * (factor_2 * T - T_new) * (factor_2 * T - T_new)
    return S_star

def update_sum_square_cov(
        S: Array,
        S_new: Array,
        T_x: Array,
        T_x_new: Array,
        T_y: Array,
        T_y_new: Array,
        N: int, 
        N_new: int
        ) -> Array:
    # TODO references
    factor_1 = N / (N_new * (N + N_new) + 1e-20)
    factor_2 = N_new * N/ (N**2 + 1e-20)
    S_star = S + S_new + factor_1 * (factor_2 * T_x - T_x_new) * (factor_2 * T_y - T_y_new)
    return S_star

class TurbulentOnlineStatisticsManager:
    """ Provides functionality to calculate statistics of turbulent flows.
    The TurbStatsManager provides turbulent statistics of the initial flow
    field as well as cumulative statistics over the course of a simulation.

    TODO think about namespaces for statistics outputs?
    """
    
    def __init__(
            self,
            turbulence_statistics_setup: TurbulenceStatisticsSetup,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            ) -> None:

        self.turbulence_case = turbulence_statistics_setup.turbulence_case
        self.streamwise_measure_position = turbulence_statistics_setup.streamwise_measure_position

        self.domain_information = domain_information
        self.Nx, self.Ny, self.Nz = domain_information.global_number_of_cells
        self.cells_per_device = domain_information.cells_per_device
        self.device_number_of_cells = domain_information.device_number_of_cells
        self.cell_sizes = domain_information.get_local_cell_sizes()
        self.is_parallel = domain_information.is_parallel
        self.split_factors = domain_information.split_factors

        self.temperature_fun = material_manager.get_temperature
        self.speed_of_sound_fun = material_manager.get_speed_of_sound
        self.dynamic_viscosity_fun = material_manager.get_dynamic_viscosity

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

    def initialize_statistics(self) -> TurbulentStatisticsInformation:
        """Initialize statistics container corresponding to the
        present turbulent case, i.e., self.turbulence_case.

        :return: Container with logging and cumulative
            turbulent statistics
        :rtype: TurbulentStatisticsInformation
        """

        hit_statistics_logging = None
        channel_statistics_logging = None
        boundarylayer_statistics_logging = None
        hit_statistics_cumulative = None
        channel_statistics_cumulative = None
        boundarylayer_statistics_cumulative = None

        if self.turbulence_case == "HIT":
            hit_statistics_logging = HITStatisticsLogging(
                rho_bulk=0.0, pressure_bulk=0.0, temperature_bulk=0.0,
                rho_rms=0.0, pressure_rms=0.0, temperature_rms=0.0,
                u_rms=0.0, mach_rms=0.0,)

            hit_statistics_cumulative = HITStatisticsCumulative(
                number_sample_steps=0.0, number_sample_points=0.0,
                density_T=0.0, pressure_T=0.0, temperature_T=0.0,
                c_T=0.0, rhop_rhop_S=0.0, pp_pp_S=0.0,
                Tp_Tp_S=0.0, machp_machp_S=0.0,
                up_up_S=0.0, vp_vp_S=0.0, wp_wp_S=0.0,)

        if self.turbulence_case == "CHANNEL":
            assert_string = ("Turbulent statistics for CHANNEL cases with "
                             "multiple devices in wall-normal direction (y-axis) "
                             "are currently not supported. Please use a splitting "
                             "in x- or z-axis.")
            assert self.split_factors[1] == 1, assert_string

            channel_statistics_logging = ChannelStatisticsLogging(
                rho_bulk=0.0, pressure_bulk=0.0, temperature_bulk=0.0,
                u_bulk=0.0, mach_bulk=0.0, reynolds_tau=0.0,
                reynolds_bulk=0.0, delta_x_plus=0.0, delta_y_plus_min=0.0,
                delta_y_plus_max=0.0, delta_z_plus=0.0)

            channel_statistics_cumulative = ChannelStatisticsCumulative(
                number_sample_steps=0,
                number_sample_points=0,
                U_T=jnp.zeros((1,self.Ny,1)),
                V_T=jnp.zeros((1,self.Ny,1)),
                W_T=jnp.zeros((1,self.Ny,1)),
                density_T=jnp.zeros((1,self.Ny,1)),
                pressure_T=jnp.zeros((1,self.Ny,1)),
                T_T=jnp.zeros((1,self.Ny,1)),
                c_T=jnp.zeros((1,self.Ny,1)),
                mach_T=jnp.zeros((1,self.Ny,1)),
                pp_pp_S=jnp.zeros((1,self.Ny,1)),
                rhop_rhop_S=jnp.zeros((1,self.Ny,1)),
                machp_machp_S=jnp.zeros((1,self.Ny,1)),
                up_up_S=jnp.zeros((1,self.Ny,1)),
                vp_vp_S=jnp.zeros((1,self.Ny,1)),
                wp_wp_S=jnp.zeros((1,self.Ny,1)),
                up_vp_S=jnp.zeros((1,self.Ny,1)),
                up_wp_S=jnp.zeros((1,self.Ny,1)),
                vp_wp_S=jnp.zeros((1,self.Ny,1)),
                Tp_Tp_S=jnp.zeros((1,self.Ny,1)),
                vp_Tp_S=jnp.zeros((1,self.Ny,1)),)

        elif self.turbulence_case == "BOUNDARYLAYER":
            assert_string = ("Turbulent statistics for CHANNEL cases with "
                             "multiple devices in wall-normal direction (y-axis) "
                             "are currently not supported. Please use a splitting "
                             "in x- or z-axis.")
            assert self.split_factors[1] == 1, assert_string

            boundarylayer_statistics_logging = BoundaryLayerStatisticsLogging(
                l_plus=0.0, delta_0=0.0, delta_1=0.0, delta_2=0.0,
                reynolds_tau=0.0, delta_x_plus=0.0, delta_y_plus_edge=0.0,
                delta_y_plus_min=0.0, delta_z_plus=0.0)

            boundarylayer_statistics_cumulative = BoundaryLayerStatisticsCumulative(
                number_sample_steps=0,
                number_sample_points=0,
                U_T=jnp.zeros((1,self.Ny,1)),
                V_T=jnp.zeros((1,self.Ny,1)),
                W_T=jnp.zeros((1,self.Ny,1)),
                density_T=jnp.zeros((1,self.Ny,1)),
                pressure_T=jnp.zeros((1,self.Ny,1)),
                T_T=jnp.zeros((1,self.Ny,1)),
                c_T=jnp.zeros((1,self.Ny,1)),
                mach_T=jnp.zeros((1,self.Ny,1)),
                pp_pp_S=jnp.zeros((1,self.Ny,1)),
                rhop_rhop_S=jnp.zeros((1,self.Ny,1)),
                machp_machp_S=jnp.zeros((1,self.Ny,1)),
                up_up_S=jnp.zeros((1,self.Ny,1)),
                vp_vp_S=jnp.zeros((1,self.Ny,1)),
                wp_wp_S=jnp.zeros((1,self.Ny,1)),
                up_vp_S=jnp.zeros((1,self.Ny,1)),
                up_wp_S=jnp.zeros((1,self.Ny,1)),
                vp_wp_S=jnp.zeros((1,self.Ny,1)),
                Tp_Tp_S=jnp.zeros((1,self.Ny,1)),
                vp_Tp_S=jnp.zeros((1,self.Ny,1)),)
            
        elif self.turbulence_case == "DUCT":
            raise NotImplementedError

        elif self.turbulence_case == "TGV":
            raise NotImplementedError

        else:
            raise NotImplementedError

        # CREATE CONTAINER
        statistics_logging = StatisticsLogging(
            hit_statistics=hit_statistics_logging,
            channel_statistics=channel_statistics_logging,
            boundarylayer_statistics=boundarylayer_statistics_logging)

        statistics_cumulative = StatisticsCumulative(
            hit_statistics=hit_statistics_cumulative,
            channel_statistics=channel_statistics_cumulative,
            boundarylayer_statistics=boundarylayer_statistics_cumulative)

        turbulent_statistics = TurbulentStatisticsInformation(
            logging=statistics_logging,
            cumulative=statistics_cumulative)

        return turbulent_statistics

    def hit_statistics_online(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool
            ) -> TurbulentStatisticsInformation:
        """Computes HIT statistics for logging and
        cumulative statistics for h5 output.

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :param is_running_statistics: _description_
        :type is_running_statistics: bool
        :return: _description_
        :rtype: TurbulentStatisticsInformation
        """

        statistics_logging = turbulent_statistics.logging.hit_statistics
        statistics_cumulative = turbulent_statistics.cumulative.hit_statistics

        number_of_cells = self.domain_information.global_number_of_cells
        number_sample_points_new = number_of_cells[0]*number_of_cells[1]*number_of_cells[2]
        one_sample_points_new = 1.0 / number_sample_points_new

        rho = primitives[0]
        velocity = primitives[1:4]
        pressure = primitives[4]
        temperature = self.temperature_fun(pressure=pressure, density=rho)
        speed_of_sound = self.speed_of_sound_fun(pressure=pressure, density=rho)

        # SUMED QUANTITIES
        density_T_new = jnp.sum(rho)
        pressure_T_new = jnp.sum(pressure)
        temperature_T_new = jnp.sum(temperature)
        c_T_new = jnp.sum(speed_of_sound)
        if self.is_parallel:
            density_T_new = jax.lax.psum(density_T_new, axis_name="i")
            pressure_T_new = jax.lax.psum(pressure_T_new, axis_name="i")
            temperature_T_new = jax.lax.psum(temperature_T_new, axis_name="i")
            c_T_new = jax.lax.psum(c_T_new, axis_name="i")

        # MEAN QUANTITIES
        density_mean = one_sample_points_new * density_T_new
        pressure_mean = one_sample_points_new * pressure_T_new
        temperature_mean = one_sample_points_new * temperature_T_new
        speed_of_sound_mean = self.speed_of_sound_fun(pressure=pressure_mean, density=density_mean)

        # PRIMED QUANTITIES
        rhop = rho - density_mean
        pp = pressure - pressure_mean
        Tp = temperature - temperature_mean

        # SUM OF SQUARES
        up_up_S_new = jnp.sum(velocity[0] * velocity[0])
        vp_vp_S_new = jnp.sum(velocity[1] * velocity[1])
        wp_wp_S_new = jnp.sum(velocity[2] * velocity[2])
        Tp_Tp_S_new = jnp.sum(Tp * Tp)
        pp_pp_S_new = jnp.sum(pp * pp)
        rhop_rhop_S_new = jnp.sum(rhop * rhop)
        machp_machp_S_new = jnp.sum(velocity * velocity, axis=0) / (speed_of_sound * speed_of_sound)
        if self.is_parallel:
            up_up_S_new = jax.lax.psum(up_up_S_new, axis_name="i")
            vp_vp_S_new = jax.lax.psum(vp_vp_S_new, axis_name="i")
            wp_wp_S_new = jax.lax.psum(wp_wp_S_new, axis_name="i")
            Tp_Tp_S_new = jax.lax.psum(Tp_Tp_S_new, axis_name="i")
            pp_pp_S_new = jax.lax.psum(pp_pp_S_new, axis_name="i")
            rhop_rhop_S_new = jax.lax.psum(rhop_rhop_S_new, axis_name="i")

        # RMS VALUES
        rho_rms = one_sample_points_new * rhop_rhop_S_new
        p_rms = one_sample_points_new * pp_pp_S_new
        T_rms = one_sample_points_new * Tp_Tp_S_new
        q_rms = one_sample_points_new * (up_up_S_new + vp_vp_S_new + wp_wp_S_new)

        rho_rms = jnp.sqrt(rho_rms)
        p_rms = jnp.sqrt(p_rms)
        T_rms = jnp.sqrt(T_rms)
        q_rms = jnp.sqrt(q_rms)
        u_rms = q_rms / jnp.sqrt(3.0)
        mach_rms = q_rms / speed_of_sound_mean

        statistics_logging = HITStatisticsLogging(
            rho_bulk=density_mean, pressure_bulk=pressure_mean,
            temperature_bulk=temperature_mean,
            rho_rms=rho_rms, pressure_rms=p_rms, temperature_rms=T_rms,
            u_rms=u_rms, mach_rms=mach_rms)

        # RUNNING STATISTICS
        if is_running_statistics:

            # UPDATE SAMPLE POINT SIZE
            number_sample_steps = statistics_cumulative.number_sample_steps + 1
            number_sample_points = statistics_cumulative.number_sample_points + number_sample_points_new

            # UPDATE SUMS
            density_T = statistics_cumulative.density_T + density_T_new
            pressure_T = statistics_cumulative.pressure_T + pressure_T_new
            temperature_T = statistics_cumulative.temperature_T + temperature_T_new
            c_T = statistics_cumulative.c_T + c_T_new

            # UPDATE SUM OF SQUARES
            # <p'p'>
            pp_pp_S = update_sum_square(statistics_cumulative.pp_pp_S,
                pp_pp_S_new, statistics_cumulative.pressure_T, pressure_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <rho'rho'>
            rhop_rhop_S = update_sum_square(statistics_cumulative.rhop_rhop_S,
                rhop_rhop_S_new, statistics_cumulative.density_T, density_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <T'T'>
            Tp_Tp_S = update_sum_square(statistics_cumulative.Tp_Tp_S,
                Tp_Tp_S_new, statistics_cumulative.temperature_T, temperature_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'u'>
            up_up_S = update_sum_square(statistics_cumulative.up_up_S,
                up_up_S_new, 0.0, 0.0,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'v'>
            vp_vp_S = update_sum_square(statistics_cumulative.vp_vp_S,
                vp_vp_S_new, 0.0, 0.0,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <w'w'>
            wp_wp_S = update_sum_square(statistics_cumulative.wp_wp_S,
                wp_wp_S_new, 0.0, 0.0,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <M'M'>
            machp_machp_S = update_sum_square(statistics_cumulative.machp_machp_S,
                machp_machp_S_new, 0.0, 0.0,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)

            statistics_cumulative = HITStatisticsCumulative(
                number_sample_steps=number_sample_steps,
                number_sample_points=number_sample_points,
                density_T=density_T, pressure_T=pressure_T,
                temperature_T=temperature_T, c_T=c_T,
                rhop_rhop_S=rhop_rhop_S, pp_pp_S=pp_pp_S,
                Tp_Tp_S=Tp_Tp_S, machp_machp_S=machp_machp_S,
                up_up_S=up_up_S, vp_vp_S=vp_vp_S, wp_wp_S=wp_wp_S)

        statistics_logging = StatisticsLogging(hit_statistics=statistics_logging)
        statistics_cumulative = StatisticsCumulative(hit_statistics=statistics_cumulative)

        turbulent_statistics = TurbulentStatisticsInformation(
            logging=statistics_logging,
            cumulative=statistics_cumulative)

        return turbulent_statistics

    def channel_statistics_online(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool
            ) -> TurbulentStatisticsInformation:
        """Computes turbulent channel statistics for logging and
        cumulative statistics for h5 output.

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :param is_running_statistics: _description_
        :type is_running_statistics: bool
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: TurbulentStatisticsInformation
        """

        statistics_logging = turbulent_statistics.logging.channel_statistics
        statistics_cumulative = turbulent_statistics.cumulative.channel_statistics

        # CHANNEL SIZES AND CELL SIZES
        global_domain_size = self.domain_information.get_global_domain_size()
        Lx = global_domain_size[0][1] - global_domain_size[0][0]
        Ly = global_domain_size[1][1] - global_domain_size[1][0]
        Lz = global_domain_size[2][1] - global_domain_size[2][0]
        delta = 0.5 * Ly   # Half channel height
        cell_sizes_x, cell_sizes_y, cell_sizes_z = self.cell_sizes
        cell_sizes_y = jnp.squeeze(cell_sizes_y)
        if cell_sizes_y.ndim == 0:
            cell_sizes_y = cell_sizes_y.reshape(1,)

        rho = primitives[0]
        velocity = primitives[1:4]
        U, V, W = velocity
        pressure = primitives[4]
        temperature = self.temperature_fun(pressure=pressure, density=rho)
        speed_of_sound = self.speed_of_sound_fun(pressure=pressure, density=rho)

        # THERMODYNAMICS
        rho_mean_y = reynolds_average(rho, axis=(0,2))
        if self.is_parallel:
            rho_mean_y = jax.lax.pmean(rho_mean_y, axis_name="i")
        rho_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * rho_mean_y)
        rho_wall = rho_mean_y[0]

        p_mean_y = reynolds_average(pressure, axis=(0,2))
        if self.is_parallel:
            p_mean_y = jax.lax.pmean(p_mean_y, axis_name="i")
        p_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * p_mean_y)
        p_wall = p_mean_y[0]

        T_mean_y = reynolds_average(temperature, axis=(0,2))
        if self.is_parallel:
            T_mean_y = jax.lax.pmean(T_mean_y, axis_name="i")
        T_mean = 1/(2 * delta) * jnp.sum(cell_sizes_y * T_mean_y)
        T_wall = T_mean_y[0]

        # VISCOSITY
        mu = self.dynamic_viscosity_fun(
            temperature, primitives)
        mu_mean_y = reynolds_average(mu, axis=(0,2))
        if self.is_parallel:
            mu_mean_y = jax.lax.pmean(mu_mean_y, axis_name="i")
        mu_wall = mu_mean_y[0]
        nu_wall = mu_wall / rho_wall

        # BULK VELOCITY AND CENTER LINE VELOCITY
        U_mean = jnp.mean(U, axis=(0,2), keepdims=True)
        if self.is_parallel:
            U_mean = jax.lax.pmean(U_mean, axis_name="i")
        U_mean_y = jnp.squeeze(U_mean)
        U_bulk = 1/(2 * delta) * jnp.sum(cell_sizes_y * U_mean)
        U_0 = jnp.max(U_mean_y)

        # TURBULENT SCALES
        # WALL SHEAR STRESS EQ. (7.24)
        dudy_wall = U_mean_y[0] / (0.5 * cell_sizes_y[0])
        tau_wall = mu_wall * dudy_wall
        # FRICTION VELOCITY EQ. (7.25)
        u_tau = jnp.sqrt(tau_wall / rho_wall)
        u_plus_max = U_0 / u_tau

        # MACH NUMBER
        speed_of_sound_mean = self.speed_of_sound_fun(pressure=p_mean, density=rho_mean)
        speed_of_sound_mean_y = reynolds_average(speed_of_sound, axis=(0,2))
        if self.is_parallel:
            speed_of_sound_mean_y = jax.lax.pmean(speed_of_sound_mean_y, axis_name="i")
        Ma_bulk = U_bulk / speed_of_sound_mean
        Mach_mean_y = U_mean_y / speed_of_sound_mean_y
        Ma_0 = jnp.max(Mach_mean_y)

        # Viscous lengthscale Eq. (7.26)
        delta_nu = nu_wall / u_tau # Viscous lengthscale Eq. (7.26)
        delta_x_plus = cell_sizes_x / delta_nu     # Pope 7.28
        delta_y_plus_min = jnp.min(cell_sizes_y) / delta_nu
        delta_y_plus_max = jnp.max(cell_sizes_y) / delta_nu
        delta_z_plus = cell_sizes_z / delta_nu

        # REYNOLDS NUMBER
        Re_tau = delta * u_tau * rho_wall / mu_wall         # Pope 7.27
        Re_bulk = 2 * delta * U_bulk * rho_wall / mu_wall   # Pope 7.1
        Re_0 = delta * U_0 * rho_wall / mu_wall             # Pope 7.2

        statistics_logging = ChannelStatisticsLogging(
            rho_bulk=rho_mean, pressure_bulk=p_mean, temperature_bulk=T_mean,
            u_bulk=U_bulk, mach_bulk=Ma_bulk, reynolds_tau=Re_tau,
            reynolds_bulk=Re_bulk, delta_x_plus=delta_x_plus,
            delta_y_plus_min=delta_y_plus_min, delta_y_plus_max=delta_y_plus_max,
            delta_z_plus=delta_z_plus)

        # RUNNING STATISTICS
        if is_running_statistics:
            # 1.3a & 1.3b from Chan et al.
            # _T denotes sum and _S ednotes sum of squares
            number_sample_steps = statistics_cumulative.number_sample_steps + 1
            number_sample_points_new = self.device_number_of_cells[0] * self.device_number_of_cells[2]
            if self.is_parallel:
                number_sample_points_new = jax.lax.psum(number_sample_points_new, axis_name="i")
            one_sample_points_new = 1.0 / number_sample_points_new
            number_sample_points = statistics_cumulative.number_sample_points + number_sample_points_new

            s_axes = (-1,-3)
            # NEW SUMS
            U_T_new = jnp.sum(velocity[0], axis=s_axes, keepdims=True)
            V_T_new = jnp.sum(velocity[1], axis=s_axes, keepdims=True)
            W_T_new = jnp.sum(velocity[2], axis=s_axes, keepdims=True)
            density_T_new = jnp.sum(rho, axis=s_axes, keepdims=True)
            pressure_T_new = jnp.sum(pressure, axis=s_axes, keepdims=True)
            T_T_new = jnp.sum(temperature, axis=s_axes, keepdims=True)
            c_T_new = jnp.sum(speed_of_sound, axis=s_axes, keepdims=True)
            Ma_x = velocity[0] / speed_of_sound
            mach_T_new = jnp.sum(Ma_x, axis=s_axes, keepdims=True)

            if self.is_parallel:
                U_T_new = jax.lax.psum(U_T_new, axis_name="i")
                V_T_new = jax.lax.psum(V_T_new, axis_name="i")
                W_T_new = jax.lax.psum(W_T_new, axis_name="i")
                density_T_new = jax.lax.psum(density_T_new, axis_name="i")
                pressure_T_new = jax.lax.psum(pressure_T_new, axis_name="i")
                T_T_new = jax.lax.psum(T_T_new, axis_name="i")
                c_T_new = jax.lax.psum(c_T_new, axis_name="i")
                mach_T_new = jax.lax.psum(mach_T_new, axis_name="i")

            # NEW PRIMED QUANTITIES
            up = velocity[0] - one_sample_points_new * U_T_new
            vp = velocity[1] - one_sample_points_new * V_T_new
            wp = velocity[2] - one_sample_points_new * W_T_new
            rhop = rho - one_sample_points_new * density_T_new
            pp = pressure - one_sample_points_new * pressure_T_new
            Tp = temperature - one_sample_points_new * T_T_new
            machp = Ma_x - one_sample_points_new * mach_T_new

            up_up_S_new = jnp.sum(up * up, axis=s_axes, keepdims=True)
            vp_vp_S_new = jnp.sum(vp * vp, axis=s_axes, keepdims=True)
            wp_wp_S_new = jnp.sum(wp * wp, axis=s_axes, keepdims=True)
            up_vp_S_new = jnp.sum(up * vp, axis=s_axes, keepdims=True)
            up_wp_S_new = jnp.sum(up * wp, axis=s_axes, keepdims=True)
            vp_wp_S_new = jnp.sum(vp * wp, axis=s_axes, keepdims=True)
            Tp_Tp_S_new = jnp.sum(Tp * Tp, axis=s_axes, keepdims=True)
            vp_Tp_S_new = jnp.sum(vp * Tp, axis=s_axes, keepdims=True)
            pp_pp_S_new = jnp.sum(pp * pp, axis=s_axes, keepdims=True)
            rhop_rhop_S_new = jnp.sum(rhop * rhop, axis=s_axes, keepdims=True)
            machp_machp_S_new = jnp.sum(machp * machp, axis=s_axes, keepdims=True)

            if self.is_parallel:
                up_up_S_new = jax.lax.psum(up_up_S_new, axis_name="i")
                vp_vp_S_new = jax.lax.psum(vp_vp_S_new, axis_name="i")
                wp_wp_S_new = jax.lax.psum(wp_wp_S_new, axis_name="i")
                up_vp_S_new = jax.lax.psum(up_vp_S_new, axis_name="i")
                up_wp_S_new = jax.lax.psum(up_wp_S_new, axis_name="i")
                vp_wp_S_new = jax.lax.psum(vp_wp_S_new, axis_name="i")
                Tp_Tp_S_new = jax.lax.psum(Tp_Tp_S_new, axis_name="i")
                vp_Tp_S_new = jax.lax.psum(vp_Tp_S_new, axis_name="i")
                pp_pp_S_new = jax.lax.psum(pp_pp_S_new, axis_name="i")
                rhop_rhop_S_new = jax.lax.psum(rhop_rhop_S_new, axis_name="i")
                machp_machp_S_new = jax.lax.psum(machp_machp_S_new, axis_name="i")

            # UPDATE SUMS
            U_T = statistics_cumulative.U_T + U_T_new
            V_T = statistics_cumulative.V_T + V_T_new
            W_T = statistics_cumulative.W_T + W_T_new
            density_T = statistics_cumulative.density_T + density_T_new
            pressure_T = statistics_cumulative.pressure_T + pressure_T_new
            T_T = statistics_cumulative.T_T + T_T_new
            c_T = statistics_cumulative.c_T + c_T_new
            mach_T = statistics_cumulative.mach_T + mach_T_new

            # UPDATE SUM OF SQUARES
            # <p'p'>
            pp_pp_S = update_sum_square(statistics_cumulative.pp_pp_S,
                pp_pp_S_new, statistics_cumulative.pressure_T, pressure_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <rho'rho'>
            rhop_rhop_S = update_sum_square(statistics_cumulative.rhop_rhop_S,
                rhop_rhop_S_new, statistics_cumulative.density_T, density_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <Ma'Ma'>
            machp_machp_S = update_sum_square(statistics_cumulative.machp_machp_S,
                machp_machp_S_new, statistics_cumulative.mach_T, mach_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'u'>
            up_up_S = update_sum_square(statistics_cumulative.up_up_S,
                up_up_S_new, statistics_cumulative.U_T, U_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'v'>
            vp_vp_S = update_sum_square(statistics_cumulative.vp_vp_S,
                vp_vp_S_new, statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <w'w'>
            wp_wp_S = update_sum_square(statistics_cumulative.wp_wp_S,
                wp_wp_S_new, statistics_cumulative.W_T, W_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'v'>
            up_vp_S = update_sum_square_cov(statistics_cumulative.up_vp_S,
                up_vp_S_new, statistics_cumulative.U_T, U_T_new,
                statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'w'>
            up_wp_S = update_sum_square_cov(statistics_cumulative.up_wp_S,
                up_wp_S_new, statistics_cumulative.U_T, U_T_new,
                statistics_cumulative.W_T, W_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'w'>
            vp_wp_S = update_sum_square_cov(statistics_cumulative.vp_wp_S,
                vp_wp_S_new, statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.W_T, W_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <T'T'>
            Tp_Tp_S = update_sum_square(statistics_cumulative.Tp_Tp_S,
                Tp_Tp_S_new, statistics_cumulative.T_T, T_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'T'>
            vp_Tp_S = update_sum_square_cov(statistics_cumulative.vp_Tp_S,
                vp_Tp_S_new, statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.T_T, T_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)

            statistics_cumulative = ChannelStatisticsCumulative(
                number_sample_steps=number_sample_steps,
                number_sample_points=number_sample_points,
                U_T=U_T, V_T=V_T, W_T=W_T,
                density_T=density_T, pressure_T=pressure_T,
                T_T=T_T, c_T=c_T, mach_T=mach_T,
                pp_pp_S=pp_pp_S, rhop_rhop_S=rhop_rhop_S,
                machp_machp_S=machp_machp_S,
                up_up_S=up_up_S, vp_vp_S=vp_vp_S, wp_wp_S=wp_wp_S,
                up_vp_S=up_vp_S, up_wp_S=up_wp_S, vp_wp_S=vp_wp_S,
                Tp_Tp_S=Tp_Tp_S, vp_Tp_S=vp_Tp_S)

        statistics_logging = StatisticsLogging(channel_statistics=statistics_logging)
        statistics_cumulative = StatisticsCumulative(channel_statistics=statistics_cumulative)

        turbulent_statistics = TurbulentStatisticsInformation(
            logging=statistics_logging,
            cumulative=statistics_cumulative)

        return turbulent_statistics

    def boundarylayer_statistics_online(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool
            ) -> TurbulentStatisticsInformation:
        
        statistics_logging = turbulent_statistics.logging.boundarylayer_statistics
        statistics_cumulative = turbulent_statistics.cumulative.boundarylayer_statistics

        x, y, z = self.domain_information.get_device_cell_centers()
        dx, dy, dz = self.domain_information.get_device_cell_sizes()
        dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
        dx_min, dz_min = jnp.min(dx), jnp.min(dz)

        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            mask = self.device_mask_measure_position[device_id]
            sum_devices = jnp.sum(self.device_mask_measure_position)

        x_id = jnp.argmin(jnp.abs(x - self.streamwise_measure_position))

        s_axes = (-1)

        density = primitives[0,x_id]
        velocityX = primitives[1,x_id]
        velocity = primitives[1:4,x_id]
        pressure = primitives[4,x_id]
        temperature = self.temperature_fun(primitives)[x_id]

        # MEAN
        rho_mean = reynolds_average(density, s_axes)
        u_mean = reynolds_average(velocityX, s_axes)
        T_mean = reynolds_average(temperature, s_axes)
        if self.is_parallel:
            rho_mean = rho_mean * mask
            u_mean = u_mean * mask
            T_mean = T_mean * mask
            rho_mean = jax.lax.psum(rho_mean, axis_name="i")/sum_devices
            u_mean = jax.lax.psum(u_mean, axis_name="i")/sum_devices
            T_mean = jax.lax.psum(T_mean, axis_name="i")/sum_devices

        rho_e = rho_mean[-1]
        U_e = u_mean[-1]

        # MEAN VISCOSITY
        mu_mean = self.dynamic_viscosity_fun(T_mean, primitives)
        nu_mean = mu_mean/rho_mean

        # WALL UNITS
        mu_wall = mu_mean[0]
        nu_wall = nu_mean[0]
        rho_wall = rho_mean[0]
        dudy_wall = u_mean[0]/y[0]
        tau_wall = mu_wall * dudy_wall
        u_tau = jnp.sqrt(tau_wall/rho_wall)

        # LENGTH SCALES
        l_plus = nu_wall/u_tau
        dy_plus = dy/l_plus
        delta_y_min_plus = jnp.min(dy_plus)
        delta_x_min_plus = dx_min/l_plus
        delta_z_min_plus = dz_min/l_plus

        # BOUNDARY/DISPLACEMENT/MOMENTUM THICKNESS
        y_index_delta_0 = jnp.argmax(u_mean/U_e > 0.99, axis=0)
        delta_0 = y[y_index_delta_0]
        delta_1 = jnp.trapz((1-rho_mean/rho_e*u_mean/U_e),y, axis=0)
        delta_2 = jnp.trapz(rho_mean/rho_e*u_mean/U_e*(1-u_mean/U_e), y, axis=0)

        delta_y_plus_edge = dy_plus[y_index_delta_0]

        # REYNOLDS NUMBERS
        Re_tau = delta_0/l_plus

        statistics_logging = BoundaryLayerStatisticsLogging(
            l_plus=l_plus, delta_0=delta_0, delta_1=delta_1, delta_2=delta_2,
            reynolds_tau=Re_tau, delta_x_plus=delta_x_min_plus, delta_y_plus_edge=delta_y_plus_edge,
            delta_y_plus_min=delta_y_min_plus, delta_z_plus=delta_z_min_plus)

        # RUNNING STATISTICS
        if is_running_statistics:
            # 1.3a & 1.3b from Chan et al.
            # _T denotes sum and _S ednotes sum of squares
            number_sample_steps = statistics_cumulative.number_sample_steps + 1
            number_sample_points_new = self.domain_information.global_number_of_cells[2]
            one_sample_points_new = 1.0 / number_sample_points_new
            number_sample_points = statistics_cumulative.number_sample_points + number_sample_points_new

            # MEAN
            rho = jnp.expand_dims(primitives[0,x_id], 0)
            velocity = jnp.expand_dims(primitives[1:4,x_id], 1)
            pressure = jnp.expand_dims(primitives[4,x_id], 0)
            temperature = self.temperature_fun(pressure=pressure, density=rho)
            speed_of_sound = self.speed_of_sound_fun(pressure=pressure, density=rho)

            # NEW SUMS
            U_T_new = jnp.sum(velocity[0], axis=s_axes, keepdims=True)
            V_T_new = jnp.sum(velocity[1], axis=s_axes, keepdims=True)
            W_T_new = jnp.sum(velocity[2], axis=s_axes, keepdims=True)
            density_T_new = jnp.sum(rho, axis=s_axes, keepdims=True)
            pressure_T_new = jnp.sum(pressure, axis=s_axes, keepdims=True)
            T_T_new = jnp.sum(temperature, axis=s_axes, keepdims=True)
            c_T_new = jnp.sum(speed_of_sound, axis=s_axes, keepdims=True)
            Ma_x = velocity[0] / speed_of_sound
            mach_T_new = jnp.sum(Ma_x, axis=s_axes, keepdims=True)

            if self.is_parallel:
                U_T_new *= mask
                V_T_new *= mask
                W_T_new *= mask
                density_T_new *= mask
                pressure_T_new *= mask
                T_T_new *= mask
                c_T_new *= mask
                mach_T_new *= mask
                U_T_new = jax.lax.psum(U_T_new, axis_name="i")
                V_T_new = jax.lax.psum(V_T_new, axis_name="i")
                W_T_new = jax.lax.psum(W_T_new, axis_name="i")
                density_T_new = jax.lax.psum(density_T_new, axis_name="i")
                pressure_T_new = jax.lax.psum(pressure_T_new, axis_name="i")
                T_T_new = jax.lax.psum(T_T_new, axis_name="i")
                c_T_new = jax.lax.psum(c_T_new, axis_name="i")
                mach_T_new = jax.lax.psum(mach_T_new, axis_name="i")

            # NEW PRIMED QUANTITIES
            up = velocity[0] - one_sample_points_new * U_T_new
            vp = velocity[1] - one_sample_points_new * V_T_new
            wp = velocity[2] - one_sample_points_new * W_T_new
            rhop = rho - one_sample_points_new * density_T_new
            pp = pressure - one_sample_points_new * pressure_T_new
            Tp = temperature - one_sample_points_new * T_T_new
            machp = Ma_x - one_sample_points_new * mach_T_new

            up_up_S_new = jnp.sum(up * up, axis=s_axes, keepdims=True)
            vp_vp_S_new = jnp.sum(vp * vp, axis=s_axes, keepdims=True)
            wp_wp_S_new = jnp.sum(wp * wp, axis=s_axes, keepdims=True)
            up_vp_S_new = jnp.sum(up * vp, axis=s_axes, keepdims=True)
            up_wp_S_new = jnp.sum(up * wp, axis=s_axes, keepdims=True)
            vp_wp_S_new = jnp.sum(vp * wp, axis=s_axes, keepdims=True)
            Tp_Tp_S_new = jnp.sum(Tp * Tp, axis=s_axes, keepdims=True)
            vp_Tp_S_new = jnp.sum(vp * Tp, axis=s_axes, keepdims=True)
            pp_pp_S_new = jnp.sum(pp * pp, axis=s_axes, keepdims=True)
            rhop_rhop_S_new = jnp.sum(rhop * rhop, axis=s_axes, keepdims=True)
            machp_machp_S_new = jnp.sum(machp * machp, axis=s_axes, keepdims=True)

            if self.is_parallel:
                up_up_S_new *= mask
                vp_vp_S_new *= mask
                wp_wp_S_new *= mask
                up_vp_S_new *= mask
                up_wp_S_new *= mask
                vp_wp_S_new *= mask
                Tp_Tp_S_new *= mask
                vp_Tp_S_new *= mask
                pp_pp_S_new *= mask
                rhop_rhop_S_new *= mask
                machp_machp_S_new *= mask
                up_up_S_new = jax.lax.psum(up_up_S_new, axis_name="i")
                vp_vp_S_new = jax.lax.psum(vp_vp_S_new, axis_name="i")
                wp_wp_S_new = jax.lax.psum(wp_wp_S_new, axis_name="i")
                up_vp_S_new = jax.lax.psum(up_vp_S_new, axis_name="i")
                up_wp_S_new = jax.lax.psum(up_wp_S_new, axis_name="i")
                vp_wp_S_new = jax.lax.psum(vp_wp_S_new, axis_name="i")
                Tp_Tp_S_new = jax.lax.psum(Tp_Tp_S_new, axis_name="i")
                vp_Tp_S_new = jax.lax.psum(vp_Tp_S_new, axis_name="i")
                pp_pp_S_new = jax.lax.psum(pp_pp_S_new, axis_name="i")
                rhop_rhop_S_new = jax.lax.psum(rhop_rhop_S_new, axis_name="i")
                machp_machp_S_new = jax.lax.psum(machp_machp_S_new, axis_name="i")

            # UPDATE SUMS
            U_T = statistics_cumulative.U_T + U_T_new
            V_T = statistics_cumulative.V_T + V_T_new
            W_T = statistics_cumulative.W_T + W_T_new
            density_T = statistics_cumulative.density_T + density_T_new
            pressure_T = statistics_cumulative.pressure_T + pressure_T_new
            T_T = statistics_cumulative.T_T + T_T_new
            c_T = statistics_cumulative.c_T + c_T_new
            mach_T = statistics_cumulative.mach_T + mach_T_new

            # UPDATE SUM OF SQUARES
            # <p'p'>
            pp_pp_S = update_sum_square(statistics_cumulative.pp_pp_S,
                pp_pp_S_new, statistics_cumulative.pressure_T, pressure_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <rho'rho'>
            rhop_rhop_S = update_sum_square(statistics_cumulative.rhop_rhop_S,
                rhop_rhop_S_new, statistics_cumulative.density_T, density_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <Ma'Ma'>
            machp_machp_S = update_sum_square(statistics_cumulative.machp_machp_S,
                machp_machp_S_new, statistics_cumulative.mach_T, mach_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'u'>
            up_up_S = update_sum_square(statistics_cumulative.up_up_S,
                up_up_S_new, statistics_cumulative.U_T, U_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'v'>
            vp_vp_S = update_sum_square(statistics_cumulative.vp_vp_S,
                vp_vp_S_new, statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <w'w'>
            wp_wp_S = update_sum_square(statistics_cumulative.wp_wp_S,
                wp_wp_S_new, statistics_cumulative.W_T, W_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'v'>
            up_vp_S = update_sum_square_cov(statistics_cumulative.up_vp_S,
                up_vp_S_new, statistics_cumulative.U_T, U_T_new,
                statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <u'w'>
            up_wp_S = update_sum_square_cov(statistics_cumulative.up_wp_S,
                up_wp_S_new, statistics_cumulative.U_T, U_T_new,
                statistics_cumulative.W_T, W_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'w'>
            vp_wp_S = update_sum_square_cov(statistics_cumulative.vp_wp_S,
                vp_wp_S_new, statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.W_T, W_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <T'T'>
            Tp_Tp_S = update_sum_square(statistics_cumulative.Tp_Tp_S,
                Tp_Tp_S_new, statistics_cumulative.T_T, T_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)
            # <v'T'>
            vp_Tp_S = update_sum_square_cov(statistics_cumulative.vp_Tp_S,
                vp_Tp_S_new, statistics_cumulative.V_T, V_T_new,
                statistics_cumulative.T_T, T_T_new,
                statistics_cumulative.number_sample_points,
                number_sample_points_new)

            statistics_cumulative = BoundaryLayerStatisticsCumulative(
                number_sample_steps=number_sample_steps,
                number_sample_points=number_sample_points,
                U_T=U_T, V_T=V_T, W_T=W_T,
                density_T=density_T, pressure_T=pressure_T,
                T_T=T_T, c_T=c_T, mach_T=mach_T,
                pp_pp_S=pp_pp_S, rhop_rhop_S=rhop_rhop_S,
                machp_machp_S=machp_machp_S,
                up_up_S=up_up_S, vp_vp_S=vp_vp_S, wp_wp_S=wp_wp_S,
                up_vp_S=up_vp_S, up_wp_S=up_wp_S, vp_wp_S=vp_wp_S,
                Tp_Tp_S=Tp_Tp_S, vp_Tp_S=vp_Tp_S)

        statistics_logging = StatisticsLogging(boundarylayer_statistics=statistics_logging)
        statistics_cumulative = StatisticsCumulative(boundarylayer_statistics=statistics_cumulative)

        turbulent_statistics = TurbulentStatisticsInformation(
            logging=statistics_logging,
            cumulative=statistics_cumulative)
        
        return turbulent_statistics


    def _compute_turbulent_statistics(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool
            ) -> TurbulentStatisticsInformation:
        """Computes the turbulent statistics (logging and cumulative) for
        the given primitive buffer.

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: TurbulentStatisticsInformation
        """
        if self.turbulence_case == "HIT":
            turbulent_statistics = self.hit_statistics_online(
                primitives, turbulent_statistics, is_running_statistics)

        if self.turbulence_case == "CHANNEL":
            turbulent_statistics = self.channel_statistics_online(
                primitives, turbulent_statistics, is_running_statistics)

        if self.turbulence_case == "BOUNDARYLAYER":
            turbulent_statistics = self.boundarylayer_statistics_online(
                primitives, turbulent_statistics, is_running_statistics)

        if self.turbulence_case == "DUCT":
            turbulent_statistics = None

        if self.turbulence_case == "TGV":
            turbulent_statistics = None

        return turbulent_statistics


    @partial(jax.pmap, static_broadcasted_argnums=(0,3), axis_name="i",
             in_axes=(None,0,None,None), out_axes=(None))
    def _compute_turbulent_statistics_pmap(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool):
        """Pmap wrapper for self._compute_turbulent_statistics()

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :param is_running_statistics: _description_
        :type is_running_statistics: bool
        :return: _description_
        :rtype: _type_
        """
        return self._compute_turbulent_statistics(
            primitives,
            turbulent_statistics,
            is_running_statistics,)

    @partial(jax.jit, static_argnums=(0,3))
    def _compute_turbulent_statistics_jit(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool):
        """Jit wrapper for self._compute_turbulent_statistics()

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :param is_running_statistics: _description_
        :type is_running_statistics: bool
        :return: _description_
        :rtype: _type_
        """
        return self._compute_turbulent_statistics(
            primitives,
            turbulent_statistics,
            is_running_statistics,)

    def compute_turbulent_statistics(
            self,
            primitives: Array,
            turbulent_statistics: TurbulentStatisticsInformation,
            is_running_statistics: bool
            ) -> TurbulentStatisticsInformation:
        """Wrapper function for jit and pmap.

        :param primitives: _description_
        :type primitives: Array
        :param turbulent_statistics: _description_
        :type turbulent_statistics: TurbulentStatisticsInformation
        :param is_running_statistics: _description_
        :type is_running_statistics: bool
        :return: _description_
        :rtype: _type_
        """

        if self.is_parallel:
            return self._compute_turbulent_statistics_pmap(
                primitives,
                turbulent_statistics,
                is_running_statistics)

        else:
            return self._compute_turbulent_statistics_jit(
                primitives,
                turbulent_statistics,
                is_running_statistics)
