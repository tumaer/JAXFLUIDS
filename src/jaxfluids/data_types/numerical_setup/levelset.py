from typing import NamedTuple

from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.levelset.reinitialization.pde_based_reinitializer import PDEBasedReinitializer

class InterfaceFluxSetup(NamedTuple):
    method: str
    derivative_stencil: SpatialDerivative
    material_properties_averaging: str
    interpolation_dh: float
    is_interpolate_pressure: bool
    is_cell_based_computation: bool

class IterativeExtensionSetup(NamedTuple):
    steps: int
    CFL: float
    is_jaxwhileloop: bool
    residual_threshold: float
    is_interpolate_invalid_cells: bool
    is_extend_into_invalid_mixing_cells: bool

class InterpolationExtensionSetup(NamedTuple):
    is_cell_based_computation: bool

class LevelsetExtensionFieldSetup(NamedTuple):
    method: str
    iterative: IterativeExtensionSetup
    interpolation: InterpolationExtensionSetup
    is_stopgradient: bool
    
class LevelsetExtensionSetup(NamedTuple):
    primitives: LevelsetExtensionFieldSetup
    interface: LevelsetExtensionFieldSetup
    solids: LevelsetExtensionFieldSetup

class LevelsetReinitializationSetup(NamedTuple):
    type: PDEBasedReinitializer
    time_integrator: TimeIntegrator
    spatial_stencil: SpatialDerivative
    CFL: float
    interval: int
    steps: int
    is_cut_cell: int
    remove_underresolved: bool
    is_jaxwhileloop: bool
    residual_threshold: float

class LevelsetMixingFieldSetup(NamedTuple):
    mixing_targets: int
    volume_fraction_threshold: float
    is_interpolate_invalid_cells: bool
    normal_computation_method: str
    is_cell_based_computation: bool

class LevelsetMixingSetup(NamedTuple):
    conservatives: LevelsetMixingFieldSetup
    solids: LevelsetMixingFieldSetup

class LevelsetGeometryComputationSetup(NamedTuple):
    derivative_stencil_normal: SpatialDerivative
    derivative_stencil_curvature: SpatialDerivative
    interface_reconstruction_method: str
    path_nn: str
    symmetries_nn: int
    subcell_reconstruction: bool

class NarrowBandSetup(NamedTuple):
    cutoff_width: int
    computation_width: int
    inactive_reinitialization_bandwidth: int
    perform_cutoff: bool

class SolidCouplingSetup(NamedTuple):
    thermal: str
    dynamic: str

class SolidHeatFluxSetup(NamedTuple):
    derivative_stencil: SpatialDerivative
    reconstruction_stencil: SpatialReconstruction

class LevelsetSetup(NamedTuple):
    halo_cells: int
    model: str
    solid_coupling: SolidCouplingSetup
    levelset_advection_stencil: SpatialDerivative
    narrowband: NarrowBandSetup
    geometry: LevelsetGeometryComputationSetup
    extension: LevelsetExtensionSetup
    mixing: LevelsetMixingSetup
    reinitialization_runtime: LevelsetReinitializationSetup
    reinitialization_startup: LevelsetReinitializationSetup
    interface_flux: InterfaceFluxSetup
    solid_heat_flux: SolidHeatFluxSetup