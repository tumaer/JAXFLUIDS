from typing import NamedTuple

from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

import jax
import jax.numpy as jnp

Array = jax.Array

class GeometryCalculationSetup(NamedTuple):
    steps_curvature: int
    volume_fraction_mapping: str
    interface_smoothing: jnp.float32
    surface_tension_kernel: str
    derivative_stencil_curvature: SpatialDerivative
    derivative_stencil_center: SpatialDerivative
    reconstruction_stencil: SpatialReconstruction
    derivative_stencil_face: SpatialReconstruction

class InterfaceCompressionSetup(NamedTuple):
    is_interface_compression: bool
    time_integrator: TimeIntegrator
    CFL: float
    interval: int
    steps: int
    heaviside_parameter: float
    interface_thickness_parameter: float

class THINCSetup(NamedTuple):
    is_thinc_reconstruction: bool
    thinc_type: str
    interface_treatment: str
    interface_projection: str
    interface_parameter: float
    volume_fraction_threshold: float

class DiffusionSharpeningSetup(NamedTuple):
    is_diffusion_sharpening: bool
    model: str
    density_model: str
    incompressible_density: Array
    interface_thickness_parameter: float
    interface_velocity_parameter: float
    mobility_model: str
    acdi_threshold: float
    volume_fraction_threshold: float
    
class DiffuseInterfaceSetup(NamedTuple):
    model: str
    halo_cells: jnp.int32
    is_consistent_reconstruction: bool
    geometry_calculation: GeometryCalculationSetup
    interface_compression: InterfaceCompressionSetup
    thinc: THINCSetup
    diffusion_sharpening: DiffusionSharpeningSetup
