from typing import Dict, List

import jax.numpy as jnp
from jax import Array

from jaxfluids.turb.statistics.utilities.averaging import (
    favre_average,reynolds_average, van_driest_transform
    )
from jaxfluids.turb.statistics.utilities.dilatation import (
    calculate_dilatation_physical, calculate_dilatation_spectral,
    calculate_dilatation_spectral_parallel, _calculate_dilatation_spectral
    )
from jaxfluids.turb.statistics.utilities.energy_spectrum import (
    energy_spectrum_1D_spectral, energy_spectrum_physical, energy_spectrum_physical_parallel,
    energy_spectrum_spectral, energy_spectrum_spectral_parallel, vmap_energy_spectrum_physical,
    vmap_energy_spectrum_physical_parallel, vmap_energy_spectrum_spectral, vmap_energy_spectrum_spectral_parallel
    ) 
from jaxfluids.turb.statistics.utilities.helmholtz_projection import _helmholtz_projection
from jaxfluids.turb.statistics.utilities.profiles import (
    channel_laminar_profile, channel_log_law, channel_turbulent_profile
    )
from jaxfluids.turb.statistics.utilities.rfft import (
    irfft3D, rfft3D, irfft3D_np, rfft3D_np
    )
from jaxfluids.turb.statistics.utilities.sheartensor import (
    calculate_sheartensor, calculate_sheartensor_spectral, calculate_sheartensor_spectral_parallel,
    calculate_strain, _calculate_sheartensor_spectral, _calculate_sheartensor_spectral_parallel
)
from jaxfluids.turb.statistics.utilities.vorticity import (
    calculate_vorticity, calculate_vorticity_spectral, calculate_vorticity_spectral_parallel,
    vmap_calculate_vorticity, _calculate_vorticity_spectral
)
from jaxfluids.turb.statistics.utilities.wavenumber import (
    real_wavenumber_grid, wavenumber_grid_parallel
)

def timeseries_statistics(stats_list: List, mode: str = "append") -> Dict:
    """Combines a list of turbulent statistics 
    - in a single statistic with the time series data (mode = append)
    or 
    - in a single statistic with time-averaged data (mode = mean)

    :param stats_list: _description_
    :type stats_list: List
    :param mode: _description_, defaults to "append"
    :type mode: str, optional
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Dict
    """
    assert mode in ["append", "mean"]
    timeseries_stats = {}
    
    stat_0 = stats_list[0]
    for key, value in stat_0.items():
        timeseries_stats[key] = {}
        for subkey in value.keys():
            timeseries_stats[key][subkey] = []
            for stat in stats_list:
                value = stat[key][subkey]
                timeseries_stats[key][subkey].append(value)
                
            if mode == "append":
                timeseries_stats[key][subkey] = jnp.array(
                    timeseries_stats[key][subkey])
            elif mode == "mean":
                timeseries_stats[key][subkey] = jnp.mean(
                    jnp.array(timeseries_stats[key][subkey]), axis=0
                )
            else:
                raise NotImplementedError

    return timeseries_stats