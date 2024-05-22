from typing import NamedTuple

from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.levelset.mixing.conservative_mixer import ConservativeMixer

class InterfaceFluxSetup(NamedTuple):
    viscous_flux_method: str
    derivative_stencil: SpatialDerivative

class LevelsetExtensionSetup(NamedTuple):
    time_integrator: TimeIntegrator
    spatial_stencil: SpatialDerivative
    steps_primes: int
    CFL_primes: float
    steps_interface: int
    CFL_interface: float
    reset_cells: bool
    is_jaxforloop: bool
    is_jaxhileloop: bool
    residual_threshold: float

class LevelsetReinitializationSetup(NamedTuple):
    type: LevelsetReinitializer
    time_integrator: TimeIntegrator
    spatial_stencil: SpatialDerivative
    CFL: float
    interval: int
    steps: int
    is_cut_cell: int
    is_domain: bool
    is_halos: bool
    remove_underresolved: bool
    is_jaxforloop: bool
    is_jaxwhileloop: bool
    residual_threshold: float

class LevelsetMixingSetup(NamedTuple):
    type: ConservativeMixer
    mixing_targets: int
    volume_fraction_threshold: float

class LevelsetGeometryComputationSetup(NamedTuple):
    derivative_stencil_normal: SpatialDerivative
    derivative_stencil_curvature: SpatialDerivative
    subcell_reconstruction: bool

class NarrowBandSetup(NamedTuple):
    cutoff_width: int
    computation_width: int
    inactive_reinitialization_bandwidth: int
    perform_cutoff: bool

class LevelsetSetup(NamedTuple):
    model: str
    halo_cells: int
    levelset_advection_stencil: SpatialDerivative
    narrowband: NarrowBandSetup
    geometry: LevelsetGeometryComputationSetup
    extension: LevelsetExtensionSetup
    mixing: LevelsetMixingSetup
    reinitialization_runtime: LevelsetReinitializationSetup
    reinitialization_startup: LevelsetReinitializationSetup
    interface_flux: InterfaceFluxSetup