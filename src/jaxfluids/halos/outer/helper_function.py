from typing import Tuple

from jaxfluids.domain.domain_information import DomainInformation

from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.derivative.deriv_upwind_center_1 import DerivativeFirstOrderUpwindCenter
from jaxfluids.stencils.derivative.deriv_downwind_center_1 import DerivativeFirstOrderDownwindCenter
from jaxfluids.stencils.derivative.deriv_downwind_center_1_adap import DerivativeFirstOrderDownwindAdapCenter
from jaxfluids.stencils.derivative.deriv_upwind_center_1_adap import DerivativeFirstOrdeUpwindAdapCenter

def get_derivative_stencils_linear_extrapolation(
        domain_information: DomainInformation
        ) -> Tuple[SpatialDerivative, SpatialDerivative]:

    is_mesh_stretching = domain_information.is_mesh_stretching
    if any(is_mesh_stretching):
        
        derivative_upwind = DerivativeFirstOrdeUpwindAdapCenter(
            domain_information.nh_conservatives,
            domain_information.inactive_axes,
            is_mesh_stretching=domain_information.is_mesh_stretching,
            cell_sizes=domain_information.get_global_cell_sizes_halos()
        )

        derivative_downwind = DerivativeFirstOrderDownwindAdapCenter(
            domain_information.nh_conservatives,
            domain_information.inactive_axes,
            is_mesh_stretching=domain_information.is_mesh_stretching,
            cell_sizes=domain_information.get_global_cell_sizes_halos()
        )
    else:
        derivative_upwind = DerivativeFirstOrderUpwindCenter(
            domain_information.nh_conservatives,
            domain_information.inactive_axes,
        )

        derivative_downwind = DerivativeFirstOrderDownwindCenter(
            domain_information.nh_conservatives,
            domain_information.inactive_axes,
        )

    return derivative_upwind, derivative_downwind