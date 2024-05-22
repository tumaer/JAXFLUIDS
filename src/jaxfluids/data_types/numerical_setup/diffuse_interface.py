from typing import NamedTuple

from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

import jax.numpy as jnp
from jax import Array

class GeometryCalculationSetup(NamedTuple):
    """GeometryCalculationSetup describes the numerical setup
    of the calculation of geometric quantities in diffuse interface
    simulations. This includes:
    - steps_curvature: how many steps are done in curvature correction
    - volume_fraction_mapping: mapping applied to the volume fraction field
        before it is used for evaluating geometric quantities
    - interface_smoothing:
    - surface_tension_kernel: kernel for smearing the surface tension
    - derivative_stencil_curvature: central derivative stencil for calculating
        curvature from the normal vector
    - derivative_stencil_center: central derivative stencil for calculating
        the normal vector at cell centers
    - reconstruction_stencil: central reconstruction stencil for reconstructing
        the normal from cell centers to corresponding cell faces
    - derivative_stencil_face: central derivative stencil for calculating
        the normal vector at cell faces

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    steps_curvature: int
    volume_fraction_mapping: str
    interface_smoothing: jnp.float32
    surface_tension_kernel: str
    derivative_stencil_curvature: SpatialDerivative
    derivative_stencil_center: SpatialDerivative
    reconstruction_stencil: SpatialReconstruction
    derivative_stencil_face: SpatialReconstruction

class InterfaceCompressionSetup(NamedTuple):
    """InterfaceCompressionSetup describes the numerical setup for performing
    interface compression.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    is_interface_compression: bool
    time_integrator: TimeIntegrator
    CFL: float
    interval: int
    steps: int
    heaviside_parameter: float
    interface_thickness_parameter: float

class THINCSetup(NamedTuple):
    """THINCSetup describes the numerical setup for interface sharpening
    via THINC reconstruction.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    is_thinc_reconstruction: bool
    thinc_type: str
    interface_treatment: str
    interface_projection: str
    interface_parameter: float
    volume_fraction_threshold: float

class DiffusionSharpeningSetup(NamedTuple):
    """DiffusionSharpeningSetup describes the numerical setup for interace
    sharpening via PDE based diffusion-sharpening.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
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
    """DiffuseInterfaceSetup is the high-level numerical setup
    with specifies computations related to diffuse interface simulations.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    model: str
    halo_cells: jnp.int32
    is_consistent_reconstruction: bool
    geometry_calculation: GeometryCalculationSetup
    interface_compression: InterfaceCompressionSetup
    thinc: THINCSetup
    diffusion_sharpening: DiffusionSharpeningSetup
