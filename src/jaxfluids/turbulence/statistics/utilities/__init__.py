from typing import Dict, List

import jax.numpy as jnp

from jaxfluids.turbulence.statistics.utilities.averaging import (
    favre_average,reynolds_average, van_driest_transform
    )
from jaxfluids.turbulence.statistics.utilities.dilatation import (
    calculate_dilatation, 
    calculate_dilatation_spectral,
    calculate_dilatation_spectral_parallel,
    )
from jaxfluids.turbulence.statistics.utilities.energy_spectrum import (
    energy_spectrum_1D_spectral,
    energy_spectrum_physical,
    energy_spectrum_physical_parallel,
    energy_spectrum_physical_real_parallel,
    energy_spectrum_spectral, 
    energy_spectrum_spectral_parallel,
    energy_spectrum_spectral_real_parallel,
    vmap_energy_spectrum_physical,
    vmap_energy_spectrum_physical_parallel, 
    vmap_energy_spectrum_spectral, 
    vmap_energy_spectrum_spectral_parallel
    ) 
from jaxfluids.turbulence.statistics.utilities.helmholtz_projection import _helmholtz_projection
from jaxfluids.turbulence.statistics.utilities.profiles import (
    channel_laminar_profile, channel_log_law, channel_turbulent_profile
    )
from jaxfluids.turbulence.statistics.utilities.sheartensor import (
    calculate_sheartensor, calculate_sheartensor_spectral, calculate_sheartensor_spectral_parallel,
    calculate_strain, _calculate_sheartensor_spectral, _calculate_sheartensor_spectral_parallel
)
from jaxfluids.turbulence.statistics.utilities.vorticity import (
    calculate_vorticity,
    calculate_vorticity_parallel,
    calculate_vorticity_spectral, 
    calculate_vorticity_spectral_parallel,
)
