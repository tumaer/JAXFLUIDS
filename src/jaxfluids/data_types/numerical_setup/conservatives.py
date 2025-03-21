from typing import Callable, Dict, NamedTuple, Union
import jax.numpy as jnp

from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class TimeIntegrationSetup(NamedTuple):
    integrator: TimeIntegrator
    CFL: float
    fixed_timestep: float

class ILESSetup(NamedTuple):
    aldm_smoothness_measure: str = "TV"
    wall_damping: str = None
    shock_sensor: str = "DUCROS"

class CATUMSetup(NamedTuple):
    transport_velocity: str = "EGERER"
    minimum_speed_of_sound: float = 1e-3

class SplitReconstructionSetup(NamedTuple):
    density: SpatialReconstruction
    velocity: SpatialReconstruction
    pressure: SpatialReconstruction
    volume_fraction: SpatialReconstruction = None

class CentralSchemeSetup(NamedTuple):
    split_form: str
    reconstruction_variable: str
    reconstruction_stencil: SpatialReconstruction

class HighOrderGodunovSetup(NamedTuple):
    riemann_solver: RiemannSolver
    signal_speed: Callable
    reconstruction_stencil: Union[str, SpatialReconstruction]
    split_reconstruction: SplitReconstructionSetup
    reconstruction_variable: str
    frozen_state: str
    catum_setup: CATUMSetup = None

class FluxSplittingSetup(NamedTuple):
    flux_splitting: str
    reconstruction_stencil: Union[str, SpatialReconstruction]
    split_reconstruction: SplitReconstructionSetup
    frozen_state: str

class ALDMSetup(NamedTuple):
    smoothness_measure: str = "TV"
    wall_damping: str = None
    shock_sensor: str = "DUCROS"

class CATUMSetup_(NamedTuple):
    # TODO
    transport_velocity: str
    reconstruction_stencil: Union[str, SpatialReconstruction]
    split_reconstruction: SplitReconstructionSetup
    minimum_speed_of_sound: float = 1e-3

class ConvectiveFluxesSetup(NamedTuple):
    convective_solver: ConvectiveFluxSolver
    godunov: HighOrderGodunovSetup = None
    flux_splitting: FluxSplittingSetup = None
    aldm: ALDMSetup = None
    central: CentralSchemeSetup = None

class DissipativeFluxesSetup(NamedTuple):
    reconstruction_stencil: SpatialReconstruction
    derivative_stencil_center: SpatialDerivative
    derivative_stencil_face: SpatialDerivative
    is_laplacian: bool
    second_derivative_stencil_center: SpatialDerivative

class PositivitySetup(NamedTuple):
    flux_limiter: Union[str, bool]
    flux_partition: str
    is_interpolation_limiter: bool
    limit_velocity: bool
    is_thinc_interpolation_limiter: bool
    is_volume_fraction_limiter: bool
    is_acdi_flux_limiter: bool

class ConservativesSetup(NamedTuple):
    halo_cells: int
    time_integration: TimeIntegrationSetup
    convective_fluxes: ConvectiveFluxesSetup
    dissipative_fluxes: DissipativeFluxesSetup
    positivity: PositivitySetup
