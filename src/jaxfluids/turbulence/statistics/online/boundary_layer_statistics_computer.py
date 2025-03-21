from typing import Tuple

import jax
import jax.numpy as jnp

from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.turbulence.statistics.online.helper_functions import update_sum_square, update_sum_square_cov
from jaxfluids.turbulence.statistics.online.turbulence_statistics_computer import TurbulenceStatisticsComputer
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager

from jaxfluids.data_types.case_setup.statistics import TurbulenceStatisticsSetup
from jaxfluids.data_types.statistics import (
    BoundaryLayerStatisticsLogging, TurbulenceStatisticsInformation,
    StatisticsLogging, Metrics, StatisticsCumulative)


from jaxfluids.turbulence.statistics.utilities import reynolds_average

Array = jax.Array


class BoundaryLayerStatisticsComputer(TurbulenceStatisticsComputer):

    def __init__(
            self,
            turbulence_statistics_setup: TurbulenceStatisticsSetup,
            domain_information: DomainInformation,
            material_manager: MaterialManager
        ) -> None:
        super().__init__(turbulence_statistics_setup, domain_information, material_manager)

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
            boundarylayer_statistics = BoundaryLayerStatisticsLogging()
            statistics_logging = StatisticsLogging(
                boundarylayer=boundarylayer_statistics)

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

        raise NotImplementedError

        # 1.3a & 1.3b from Chan et al.
        # _T denotes sum and _S ednotes sum of squares
        
        bl_statistics = cumulative_statistics.boundarylayer_statistics
        
        x,y,z = self.domain_information.get_device_cell_centers()
        x_id = jnp.argmin(jnp.abs(x - self.streamwise_measure_position))
        s_axes = (-1)

        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            mask = self.device_mask_measure_position[device_id]

        # UPDATE SAMPLE POINT SIZE
        number_sample_steps = bl_statistics.number_sample_steps + 1
        number_sample_points_new = self.device_number_of_cells[2]
        if self.is_parallel:
            number_sample_points_new = jax.lax.psum(number_sample_points_new, axis_name="i")
        one_sample_points_new = 1.0 / number_sample_points_new
        number_sample_points = bl_statistics.number_sample_points + number_sample_points_new

        # MEAN
        rho = jnp.expand_dims(primitives[0,x_id], 0)
        velocity = jnp.expand_dims(primitives[1:4,x_id], 1)
        pressure = jnp.expand_dims(primitives[4,x_id], 0)
        temperature = self.material_manager.get_temperature(pressure=pressure, density=rho)
        speed_of_sound = self.material_manager.get_speed_of_sound(pressure=pressure, density=rho)

        # NEW SUMS
        density_T_new = jnp.sum(rho, axis=s_axes, keepdims=True)
        U_T_new = jnp.sum(velocity[0], axis=s_axes, keepdims=True)
        V_T_new = jnp.sum(velocity[1], axis=s_axes, keepdims=True)
        W_T_new = jnp.sum(velocity[2], axis=s_axes, keepdims=True)
        pressure_T_new = jnp.sum(pressure, axis=s_axes, keepdims=True)
        T_T_new = jnp.sum(temperature, axis=s_axes, keepdims=True)
        c_T_new = jnp.sum(speed_of_sound, axis=s_axes, keepdims=True)
        Ma_x = velocity[0] / speed_of_sound
        mach_T_new = jnp.sum(Ma_x, axis=s_axes, keepdims=True)

        if self.is_parallel:
            U_T_new = jax.lax.psum(mask * U_T_new, axis_name="i")
            V_T_new = jax.lax.psum(mask * V_T_new, axis_name="i")
            W_T_new = jax.lax.psum(mask * W_T_new, axis_name="i")
            density_T_new = jax.lax.psum(mask * density_T_new, axis_name="i")
            pressure_T_new = jax.lax.psum(mask * pressure_T_new, axis_name="i")
            T_T_new = jax.lax.psum(mask * T_T_new, axis_name="i")
            c_T_new = jax.lax.psum(mask * c_T_new, axis_name="i")
            mach_T_new = jax.lax.psum(mask * mach_T_new, axis_name="i")

        # NEW PRIMED QUANTITIES
        rhop = rho - one_sample_points_new * density_T_new
        up = velocity[0] - one_sample_points_new * U_T_new
        vp = velocity[1] - one_sample_points_new * V_T_new
        wp = velocity[2] - one_sample_points_new * W_T_new
        pp = pressure - one_sample_points_new * pressure_T_new
        Tp = temperature - one_sample_points_new * T_T_new
        machp = Ma_x - one_sample_points_new * mach_T_new

        # NEW SUM OF SQUARES
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
            up_up_S_new = jax.lax.psum(mask * up_up_S_new, axis_name="i")
            vp_vp_S_new = jax.lax.psum(mask * vp_vp_S_new, axis_name="i")
            wp_wp_S_new = jax.lax.psum(mask * wp_wp_S_new, axis_name="i")
            up_vp_S_new = jax.lax.psum(mask * up_vp_S_new, axis_name="i")
            up_wp_S_new = jax.lax.psum(mask * up_wp_S_new, axis_name="i")
            vp_wp_S_new = jax.lax.psum(mask * vp_wp_S_new, axis_name="i")
            Tp_Tp_S_new = jax.lax.psum(mask * Tp_Tp_S_new, axis_name="i")
            vp_Tp_S_new = jax.lax.psum(mask * vp_Tp_S_new, axis_name="i")
            pp_pp_S_new = jax.lax.psum(mask * pp_pp_S_new, axis_name="i")
            rhop_rhop_S_new = jax.lax.psum(mask * rhop_rhop_S_new, axis_name="i")
            machp_machp_S_new = jax.lax.psum(mask * machp_machp_S_new, axis_name="i")

        # UPDATE SUMS
        U_T = bl_statistics.U_T + U_T_new
        V_T = bl_statistics.V_T + V_T_new
        W_T = bl_statistics.W_T + W_T_new
        density_T = bl_statistics.density_T + density_T_new
        pressure_T = bl_statistics.pressure_T + pressure_T_new
        T_T = bl_statistics.T_T + T_T_new
        c_T = bl_statistics.c_T + c_T_new
        mach_T = bl_statistics.mach_T + mach_T_new

        # UPDATE SUM OF SQUARES
        # <p'p'>
        pp_pp_S = update_sum_square(bl_statistics.pp_pp_S,
            pp_pp_S_new, bl_statistics.pressure_T, pressure_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <rho'rho'>
        rhop_rhop_S = update_sum_square(bl_statistics.rhop_rhop_S,
            rhop_rhop_S_new, bl_statistics.density_T, density_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <Ma'Ma'>
        machp_machp_S = update_sum_square(bl_statistics.machp_machp_S,
            machp_machp_S_new, bl_statistics.mach_T, mach_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <u'u'>
        up_up_S = update_sum_square(bl_statistics.up_up_S,
            up_up_S_new, bl_statistics.U_T, U_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <v'v'>
        vp_vp_S = update_sum_square(bl_statistics.vp_vp_S,
            vp_vp_S_new, bl_statistics.V_T, V_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <w'w'>
        wp_wp_S = update_sum_square(bl_statistics.wp_wp_S,
            wp_wp_S_new, bl_statistics.W_T, W_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <u'v'>
        up_vp_S = update_sum_square_cov(bl_statistics.up_vp_S,
            up_vp_S_new, bl_statistics.U_T, U_T_new,
            bl_statistics.V_T, V_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <u'w'>
        up_wp_S = update_sum_square_cov(bl_statistics.up_wp_S,
            up_wp_S_new, bl_statistics.U_T, U_T_new,
            bl_statistics.W_T, W_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <v'w'>
        vp_wp_S = update_sum_square_cov(bl_statistics.vp_wp_S,
            vp_wp_S_new, bl_statistics.V_T, V_T_new,
            bl_statistics.W_T, W_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <T'T'>
        Tp_Tp_S = update_sum_square(bl_statistics.Tp_Tp_S,
            Tp_Tp_S_new, bl_statistics.T_T, T_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)
        # <v'T'>
        vp_Tp_S = update_sum_square_cov(bl_statistics.vp_Tp_S,
            vp_Tp_S_new, bl_statistics.V_T, V_T_new,
            bl_statistics.T_T, T_T_new,
            bl_statistics.number_sample_points,
            number_sample_points_new)

        bl_statistics = BoundaryLayerStatisticsCumulative(
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

        cumulative_statistics = StatisticsCumulative(
            sampling_dt=cumulative_statistics.sampling_dt,
            next_sampling_time=cumulative_statistics.next_sampling_time + cumulative_statistics.sampling_dt,
            boundarylayer_statistics=bl_statistics
        )

        return cumulative_statistics


    def compute_logging_statistics(self, primitives: Array) -> StatisticsLogging:

        x,y,z = self.domain_information.get_device_cell_centers()
        dx,dy,dz = self.domain_information.get_device_cell_sizes()
        dx,dy,dz = dx.flatten(), dy.flatten(), dz.flatten()
        dx_min, dz_min = jnp.min(dx), jnp.min(dz)

        if self.is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            mask = self.device_mask_measure_position[device_id]
            sum_devices = jnp.sum(self.device_mask_measure_position)

        x_id = jnp.argmin(jnp.abs(x - self.streamwise_measure_position))
        s_axes = (-1)

        density = primitives[self.ids_mass,x_id]
        velocityX = primitives[self.ids_velocity[0],x_id]
        velocity = primitives[self.s_velocity,x_id]
        pressure = primitives[self.ids_energy,x_id]
        temperature = self.material_manager.get_temperature(primitives)[x_id]

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
        mu_mean = self.material_manager.get_dynamic_viscosity(T_mean, primitives)
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
        delta_1 = jnp.trapezoid((1-rho_mean/rho_e*u_mean/U_e),y, axis=0)
        delta_2 = jnp.trapezoid(rho_mean/rho_e*u_mean/U_e*(1-u_mean/U_e), y, axis=0)

        delta_y_plus_edge = dy_plus[y_index_delta_0]

        # REYNOLDS NUMBERS
        Re_tau = delta_0/l_plus

        boundarylayer_statistics = BoundaryLayerStatisticsLogging(
            l_plus=l_plus, delta_0=delta_0, delta_1=delta_1, delta_2=delta_2,
            reynolds_tau=Re_tau, delta_x_plus=delta_x_min_plus, delta_y_plus_edge=delta_y_plus_edge,
            delta_y_plus_min=delta_y_min_plus, delta_z_plus=delta_z_min_plus)

        statistics_logging = StatisticsLogging(
            boundarylayer_statistics=boundarylayer_statistics)

        return statistics_logging