from typing import List

from jaxfluids.data_types.statistics import StatisticsLogging


def prepate_turbulent_statistics_for_logging(
    turbulent_statistics_logging: StatisticsLogging, 
    ) -> List[str]:

    if turbulent_statistics_logging.hit is not None:
        hit_statistics = turbulent_statistics_logging.hit
        density_mean = hit_statistics.rho_bulk
        pressure_mean = hit_statistics.pressure_bulk
        temperature_mean = hit_statistics.temperature_bulk
        density_rms = hit_statistics.rho_rms
        pressure_rms = hit_statistics.pressure_rms
        temperature_rms = hit_statistics.temperature_rms
        mach_turbulent = hit_statistics.mach_rms
        velocity_rms = hit_statistics.u_rms

        log_list = [
            "TURBULENCE STATISTICS - HIT",
            f"MEAN DENSITY               = {density_mean:4.4e}",
            f"MEAN PRESSURE              = {pressure_mean:4.4e}",
            f"MEAN TEMPERATURE           = {temperature_mean:4.4e}",                        
            f"TURBULENT MACH RMS         = {mach_turbulent:4.4e}",
            f"VELOCITY RMS               = {velocity_rms:4.4e}",
            f"DENSITY RMS                = {density_rms:4.4e}",
            f"PRESSURE RMS               = {pressure_rms:4.4e}",
            f"TEMPERATURE RMS            = {temperature_rms:4.4e}",
        ]
    
    elif turbulent_statistics_logging.channel is not None:
        channel_statistics = turbulent_statistics_logging.channel
        rho_bulk = channel_statistics.rho_bulk
        temperature_bulk = channel_statistics.temperature_bulk
        u_bulk = channel_statistics.u_bulk
        mach_bulk = channel_statistics.mach_bulk
        reynolds_tau = channel_statistics.reynolds_tau
        reynolds_bulk = channel_statistics.reynolds_bulk
        delta_x_plus = channel_statistics.delta_x_plus
        delta_y_plus_min = channel_statistics.delta_y_plus_min
        delta_y_plus_max = channel_statistics.delta_y_plus_max
        delta_z_plus = channel_statistics.delta_z_plus

        log_list = [
            "TURBULENCE STATISTICS - CHANNEL",
            f"DENSITY BULK               = {rho_bulk:4.4e}",
            f"TEMPERATURE BULK           = {temperature_bulk:4.4e}",
            f"VELOCITY BULK              = {u_bulk:4.4e}",
            f"MACH NUBMER BULK           = {mach_bulk:4.4e}",
            f"REYNOLDS NUMBER TAU        = {reynolds_tau:4.4e}",
            f"REYNOLDS NUMBER BULK       = {reynolds_bulk:4.4e}",
            f"DELTA X+                   = {delta_x_plus:4.4e}",
            f"DELTA Y+ MIN               = {delta_y_plus_min:4.4e}",
            f"DELTA Y+ MAX               = {delta_y_plus_max:4.4e}",
            f"DELTA Z+                   = {delta_z_plus:4.4e}",
        ]
    
    else:
        raise NotImplementedError

    return log_list