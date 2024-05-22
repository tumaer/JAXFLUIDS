from typing import NamedTuple, Callable, Union
import jax.numpy as jnp
from jax import Array

from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class TimeIntegrationSetup(NamedTuple):
    """TimeIntegrationSetup describes the numerical setup 
    for temporal integration of the conservative quantities. 
    This includes:
    - integrator: specific type of time integration scheme
    - CFL: CFL number for calculating the adaptive time step
    - fixed_timestep: optional fixed time step

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    integrator: TimeIntegrator
    CFL: float
    fixed_timestep: float

class ILESSetup(NamedTuple):
    """ILESSetup describes the numerical setup of the ALDM
    ILES scheme. This includes:
    - aldm_smoothness_measure: specific type of smoothness measure
    - wall_damping: type of wall damping
    - shock_sensor: type of shock sensor

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    aldm_smoothness_measure: str = "TV"
    wall_damping: str = None
    shock_sensor: str = "DUCROS"

class CATUMSetup(NamedTuple):
    """CATUMSetup describes the numerical setup of the CATUM
    scheme for simulating cavitating flows. This includes:
    - transport_velocity: specifies computation of the
        transport velocity in the CATUM flux function
    - minimum_speed_of_sound: specifies the minimum speed of
        sound used for uwpinding in the CATUM flux function

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    transport_velocity: str = "EGERER"
    minimum_speed_of_sound: float = 1e-3

class SplitReconstructionSetup(NamedTuple):
    density: SpatialReconstruction
    velocity: SpatialReconstruction
    pressure: SpatialReconstruction
    volume_fraction: SpatialReconstruction = None

class ConvectiveFluxesSetup(NamedTuple):
    """ConvectiveFluxesSetup describes the setup
    for the calculation of convective fluxes. This includes:
    - convective_solver: Flux solver
    - riemann_solver: Riemann solver
    - signal_speed: signal speed estimate in Riemann solver
    - reconstruction_stencil: cell face reconstruction stencil
    - split_reconstruction: split recosnstruction setup
    - reconstruction_variable: 
    - frozen_state: averaging routine to calculate frozen state
        in the computation of local characteristics
    - iles_setup: Setup for implicit LES simulations
    - catum_setup: Setup for cavitation simulations
    
    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    convective_solver: ConvectiveFluxSolver
    riemann_solver: RiemannSolver
    signal_speed: Callable
    reconstruction_stencil: Union[str, SpatialReconstruction]
    split_reconstruction: SplitReconstructionSetup
    flux_splitting: str
    reconstruction_variable: str
    frozen_state: str
    iles_setup: ILESSetup = None
    catum_setup: CATUMSetup = None

class DissipativeFluxesSetup(NamedTuple):
    """DissipativeFluxesSetup describes the spatial stencils 
    used for evaluating dissipative fluxes. This includes:
    - reconstruction_stencil: central stencil for cell face reconstruction
    - derivative_stencil_center: central stencil for evaluating 
        first derivatives at cell centers
    - derivative_stencil_face: central stencil for evaluating
        first derivatives at cell faces

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    reconstruction_stencil: SpatialReconstruction
    derivative_stencil_center: SpatialDerivative
    derivative_stencil_face: SpatialDerivative

class PositivitySetup(NamedTuple):
    """PositivitySetup describes which positivity-preserving
    techniques are active.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    flux_limiter: Union[str, bool]
    is_interpolation_limiter: bool
    is_thinc_interpolation_limiter: bool
    is_volume_fraction_limiter: bool
    is_acdi_flux_limiter: bool
    is_logging: bool

class ConservativesSetup(NamedTuple):
    """ConservativesSetup describes the numerical setup
    for the calculation of the conservative quantities.
    - halo_cells: number of halo cells padded to both sides 
    of each active axis
    - time_integration: setup for the time integration
    - convective_fluxes: setup for the calculation of the convective fluxes
    - dissipative_fluxes: setup for the calculation of the dissipative fluxes
    - positivity: setup for positivity-preserving techniques

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    halo_cells: jnp.int32
    time_integration: TimeIntegrationSetup
    convective_fluxes: ConvectiveFluxesSetup
    dissipative_fluxes: DissipativeFluxesSetup
    positivity: PositivitySetup
