from typing import Any, Dict, Tuple

from jaxfluids.unit_handler import UnitHandler
from jaxfluids.stencils import DICT_SPATIAL_RECONSTRUCTION, \
    DICT_DERIVATIVE_FACE, DICT_FIRST_DERIVATIVE_CENTER, CENTRAL_RECONSTRUCTION_DICT, \
    DICT_SECOND_DERIVATIVE_CENTER
from jaxfluids.solvers.convective_fluxes import DICT_CONVECTIVE_SOLVER, TUPLE_CONVECTIVE_SOLVER
from jaxfluids.solvers.riemann_solvers import DICT_RIEMANN_SOLVER, DICT_SIGNAL_SPEEDS, \
    TUPLE_CATUM_TRANSPORT_VELOCITIES
from jaxfluids.time_integration import DICT_TIME_INTEGRATION
from jaxfluids.solvers import TUPLE_FROZEN_STATE, TUPLE_FLUX_SPLITTING, \
    TUPLE_RECONSTRUCTION_VARIABLES
from jaxfluids.iles import TUPLE_SMOOTHNESS_MEASURE, TUPLE_WALL_DAMPING, TUPLE_SHOCK_SENSOR
from jaxfluids.data_types.numerical_setup import ActivePhysicsSetup
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup
from jaxfluids.data_types.numerical_setup.conservatives import *
from jaxfluids.input.numerical_setup import get_setup_value, get_path_to_key
from jaxfluids.input.numerical_setup.read_positivity import read_positivity_setup

def read_conservatives_setup(
        numerical_setup_dict: Dict,
        active_physics_setup: ActivePhysicsSetup,
        unit_handler: UnitHandler
        ) -> ConservativesSetup:

    basepath = "conservatives"
    conservatives_dict = get_setup_value(
        numerical_setup_dict, "conservatives", basepath, dict,
        is_optional=False)

    path = get_path_to_key(basepath, "halo_cells")
    halo_cells = get_setup_value(
        conservatives_dict, "halo_cells", path, int, is_optional=False,
        numerical_value_condition=(">=", 0))        
    
    time_integration_setup = read_time_integration(conservatives_dict, unit_handler)
    convective_fluxes_setup = read_convective_fluxes(
        conservatives_dict, active_physics_setup, 
        halo_cells, unit_handler)
    dissipative_fluxes_setup = read_dissipative_fluxes(
        conservatives_dict, active_physics_setup, 
        halo_cells)
    positivity_setup = read_positivity_setup(conservatives_dict)

    conservatives_setup = ConservativesSetup(
        halo_cells,
        time_integration_setup,
        convective_fluxes_setup,
        dissipative_fluxes_setup,
        positivity_setup)

    return conservatives_setup


def read_time_integration(
        conservatives_dict: Dict,
        unit_handler: UnitHandler
        ) -> TimeIntegrationSetup:

    basepath = "conservatives"

    path_time_integration = get_path_to_key(basepath, "time_integration")
    time_integration_dict = get_setup_value(
        conservatives_dict, "time_integration", path_time_integration, dict,
        is_optional=False)

    path = get_path_to_key(path_time_integration, "integrator")
    time_integrator_str = get_setup_value(
        time_integration_dict, "integrator", path, str, is_optional=False,
        possible_string_values=tuple(DICT_TIME_INTEGRATION.keys()))
    time_integrator = DICT_TIME_INTEGRATION[time_integrator_str]

    path = get_path_to_key(path_time_integration, "CFL")
    CFL_number = get_setup_value(
        time_integration_dict, "CFL", path, float,
        is_optional=True, default_value=0.5,
        numerical_value_condition=(">", 0.0))

    path = get_path_to_key(path_time_integration, "fixed_timestep")
    fixed_timestep = get_setup_value(
        time_integration_dict, "fixed_timestep", path, float,
        is_optional=True, default_value=False, numerical_value_condition=(">", 0.0))
    if fixed_timestep:
        fixed_timestep = unit_handler.non_dimensionalize(fixed_timestep, "time")

    time_integration_setup = TimeIntegrationSetup(
        time_integrator, CFL_number, fixed_timestep)
    
    return time_integration_setup


def read_spatial_reconstruction(
        base_path: str, input_dict: Dict
        ) -> Tuple[SpatialReconstruction, SplitReconstructionSetup, int]:

    path = get_path_to_key(base_path, "reconstruction_stencil")
    reconstruction_stencil_str = get_setup_value(
        input_dict, "reconstruction_stencil", path, str,
        is_optional=False, default_value="WENO5-Z",
        possible_string_values=tuple(DICT_SPATIAL_RECONSTRUCTION.keys())+("SPLIT-RECONSTRUCTION",))

    if reconstruction_stencil_str == "SPLIT-RECONSTRUCTION":
        path_split = get_path_to_key(base_path, "split_reconstruction")
        split_reconstruction_dict: Dict = get_setup_value(
            input_dict, "split_reconstruction", path, dict,
            is_optional=False)
        required_halos = 0
        split_reconstruction_setup = {}
        for field_name, reconstruction_stencil_field_str in split_reconstruction_dict.items():
            path = get_path_to_key(path_split, field_name)
            reconstruction_stencil_field_str = get_setup_value(
                split_reconstruction_dict, field_name, path, str, is_optional=False,
                possible_string_values=tuple(DICT_SPATIAL_RECONSTRUCTION.keys()))
            reconstruction_stencil_field = DICT_SPATIAL_RECONSTRUCTION[reconstruction_stencil_field_str]
            required_halos = max(required_halos, reconstruction_stencil_field.required_halos)
            split_reconstruction_setup[field_name] = reconstruction_stencil_field
        split_reconstruction_setup = SplitReconstructionSetup(**split_reconstruction_setup)
        spatial_reconstruction = None
    else:
        spatial_reconstruction: SpatialReconstruction = DICT_SPATIAL_RECONSTRUCTION[reconstruction_stencil_str]
        required_halos = spatial_reconstruction.required_halos
        split_reconstruction_setup = None

    return spatial_reconstruction, split_reconstruction_setup, required_halos

def read_high_order_godunov(
        convective_fluxes_dict: Dict,
        unit_handler: UnitHandler
        ) -> Tuple[HighOrderGodunovSetup, int]:

    base_path = get_path_to_key("conservatives", "convective_fluxes")

    path_godunov = get_path_to_key(base_path, "godunov")
    godunov_dict = get_setup_value(
        convective_fluxes_dict, "godunov", base_path, 
        dict, is_optional=False, default_value={})

    path = get_path_to_key(path_godunov, "riemann_solver")
    riemann_solver_str = get_setup_value(
        godunov_dict, "riemann_solver", path, str,
        is_optional=False, default_value="HLLC",
        possible_string_values=tuple(DICT_RIEMANN_SOLVER.keys()))
    riemann_solver = DICT_RIEMANN_SOLVER[riemann_solver_str]
    
    path = get_path_to_key(path_godunov, "signal_speed")
    is_optional = riemann_solver in [DICT_RIEMANN_SOLVER["CATUM"],]
    signal_speed_str = get_setup_value(
        godunov_dict, "signal_speed", path, str,
        is_optional=is_optional, default_value="EINFELDT",
        possible_string_values=tuple(DICT_SIGNAL_SPEEDS.keys()))
    signal_speed = DICT_SIGNAL_SPEEDS[signal_speed_str]

    path = get_path_to_key(path_godunov, "reconstruction_variable")
    reconstruction_variable = get_setup_value(
        godunov_dict, "reconstruction_variable", path, str,
        is_optional=False, default_value="PRIMITIVE",
        possible_string_values=TUPLE_RECONSTRUCTION_VARIABLES)
    
    reconstruction_stencil, split_reconstruction_setup, required_halos =\
    read_spatial_reconstruction(path_godunov, godunov_dict)

    path = get_path_to_key(path_godunov, "frozen_state")
    frozen_state = get_setup_value(
        godunov_dict, "frozen_state", path, str,
        is_optional=True, default_value="ARITHMETIC",
        possible_string_values=TUPLE_FROZEN_STATE)

    # CATUM SETUP
    is_optional = riemann_solver_str != "CATUM"
    path_catum = get_path_to_key(path_godunov, "catum_setup")
    catum_setup_dict = get_setup_value(
        godunov_dict, "catum_setup", path_catum, dict,
        is_optional=is_optional, default_value={})

    path = get_path_to_key(path_catum, "transport_velocity")
    transport_velocity = get_setup_value(
        catum_setup_dict, "transport_velocity", path, str,
        is_optional=is_optional, default_value="EGERER",
        possible_string_values=TUPLE_CATUM_TRANSPORT_VELOCITIES)

    path = get_path_to_key(path_catum, "minimum_speed_of_sound")
    minimum_speed_of_sound = get_setup_value(
        catum_setup_dict, "minimum_speed_of_sound", path, float,
        is_optional=is_optional, default_value=1e-3,
        numerical_value_condition=(">", 0.0))
    minimum_speed_of_sound = unit_handler.non_dimensionalize(
        minimum_speed_of_sound, "velocity")

    catum_setup = CATUMSetup(
        transport_velocity=transport_velocity,
        minimum_speed_of_sound=minimum_speed_of_sound)

    godunov_setup = HighOrderGodunovSetup(
        riemann_solver,
        signal_speed,
        reconstruction_stencil,
        split_reconstruction_setup,
        reconstruction_variable,
        frozen_state,
        catum_setup
    )

    return godunov_setup, required_halos

def read_flux_splitting(convective_fluxes_dict: Dict) -> FluxSplittingSetup:

    base_path = get_path_to_key("conservatives", "convective_fluxes")

    path_flux_splitting = get_path_to_key(
        base_path, "flux_splitting")
    flux_splitting_dict = get_setup_value(
        convective_fluxes_dict, "flux_splitting", base_path, 
        dict, is_optional=False, default_value={})

    path = get_path_to_key(path_flux_splitting, "flux_splitting")
    flux_splitting = get_setup_value(
        flux_splitting_dict, "flux_splitting", path, str,
        is_optional=False, default_value="ROE",
        possible_string_values=TUPLE_FLUX_SPLITTING)

    # TODO: what is this???
    path = get_path_to_key(path_flux_splitting, "reconstruction_stencil")
    reconstruction_stencil_str = get_setup_value(
        flux_splitting_dict, "reconstruction_stencil", path, str,
        is_optional=False, default_value="WENO5-Z",
        possible_string_values=tuple(DICT_SPATIAL_RECONSTRUCTION.keys())+("SPLIT-RECONSTRUCTION",))

    spatial_reconstruction, split_reconstruction_setup, required_halos =\
    read_spatial_reconstruction(path_flux_splitting, flux_splitting_dict)

    path = get_path_to_key(path_flux_splitting, "frozen_state")
    frozen_state = get_setup_value(
        flux_splitting_dict, "frozen_state", path, str,
        is_optional=True, default_value="ARITHMETIC",
        possible_string_values=TUPLE_FROZEN_STATE)

    flux_splitting_setup = FluxSplittingSetup(
        flux_splitting,
        spatial_reconstruction,
        split_reconstruction_setup,
        frozen_state
    )

    return flux_splitting_setup, required_halos

def read_aldm(convective_fluxes_dict: Dict) -> ALDMSetup:

    base_path = get_path_to_key("conservatives", "convective_fluxes")

    path_aldm = get_path_to_key(base_path, "aldm")
    aldm_dict = get_setup_value(
        convective_fluxes_dict, "aldm", base_path, 
        dict, is_optional=False, default_value={})

    path = get_path_to_key(path_aldm, "smoothness_measure")
    smoothness_measure = get_setup_value(
        aldm_dict, "smoothness_measure", path, str,
        is_optional=True, default_value="TV",
        possible_string_values=TUPLE_SMOOTHNESS_MEASURE)

    path = get_path_to_key(path_aldm, "wall_damping")
    wall_damping = get_setup_value(
        aldm_dict, "wall_damping", path, str,
        is_optional=True, default_value="VANDRIEST",
        possible_string_values=TUPLE_WALL_DAMPING)
    
    path = get_path_to_key(path_aldm, "shock_sensor")
    shock_sensor = get_setup_value(
        aldm_dict, "shock_sensor", path, str,
        is_optional=True, default_value="DUCROS",
        possible_string_values=TUPLE_SHOCK_SENSOR)
    
    aldm_setup = ALDMSetup(
        smoothness_measure,
        wall_damping,
        shock_sensor)
    
    required_halos = 3 

    return aldm_setup, required_halos


def read_catum():
    pass


def read_central(convective_fluxes_dict: Dict):
    base_path = get_path_to_key("conservatives", "convective_fluxes")

    path_central = get_path_to_key(base_path, "central")
    central_dict = get_setup_value(
        convective_fluxes_dict, "central", path_central, 
        dict, is_optional=False)

    path = get_path_to_key(path_central, "split_form")
    split_form = get_setup_value(
        central_dict, "split_form", path, 
        str, is_optional=True, default_value=None)
    
    path = get_path_to_key(path_central, "reconstruction_variable")
    reconstruction_variable = get_setup_value(
        central_dict, "reconstruction_variable", path, 
        str, is_optional=True, default_value="FLUX")
    
    path = get_path_to_key(path_central, "reconstruction_stencil")
    reconstruction_stencil_str = get_setup_value(
        central_dict, "reconstruction_stencil", path, 
        str, is_optional=False, possible_string_values=CENTRAL_RECONSTRUCTION_DICT.keys())
    
    reconstruction_stencil = CENTRAL_RECONSTRUCTION_DICT[reconstruction_stencil_str]

    central_scheme_setup = CentralSchemeSetup(
        split_form, reconstruction_variable,
        reconstruction_stencil)

    return central_scheme_setup, reconstruction_stencil.required_halos


def read_convective_fluxes(
        conservatives_dict: Dict,
        active_physics_setup: ActivePhysicsSetup,
        halo_cells: int,
        unit_handler: UnitHandler
        ) -> ConvectiveFluxesSetup:

    basepath = "conservatives"

    is_convective_flux = active_physics_setup.is_convective_flux
    is_optional = not is_convective_flux

    path_convective_fluxes = get_path_to_key(basepath, "convective_fluxes")
    convective_fluxes_dict = get_setup_value(
        conservatives_dict, "convective_fluxes", path_convective_fluxes, dict,
        is_optional=is_optional, default_value={})

    path = get_path_to_key(path_convective_fluxes, "convective_solver")
    convective_solver_str = get_setup_value(
        convective_fluxes_dict, "convective_solver", path, str,
        is_optional=is_optional, default_value="GODUNOV",
        possible_string_values=TUPLE_CONVECTIVE_SOLVER)
    convective_solver = DICT_CONVECTIVE_SOLVER[convective_solver_str]

    godunov_setup = flux_splitting_setup = aldm_setup = None
    central_scheme_setup = None
    if convective_solver_str == "GODUNOV":
        godunov_setup, required_halos = read_high_order_godunov(
            convective_fluxes_dict, unit_handler)
    
    elif convective_solver_str == "FLUX-SPLITTING":
        flux_splitting_setup, required_halos = read_flux_splitting(convective_fluxes_dict)
    
    elif convective_solver_str == "ALDM":
        aldm_setup, required_halos = read_aldm(convective_fluxes_dict)

    elif convective_solver_str == "CENTRAL":
        central_scheme_setup, required_halos = read_central(convective_fluxes_dict)

    else:
        raise NotImplementedError

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but spatial reconstruction"
        f"stencil for the convective fluxes requires at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string

    convective_fluxes_setup = ConvectiveFluxesSetup(
        convective_solver, godunov_setup,
        flux_splitting_setup, aldm_setup,
        central_scheme_setup)
    
    return convective_fluxes_setup

def read_dissipative_fluxes(
        conservatives_dict: Dict,
        active_physics_setup: ActivePhysicsSetup,
        halo_cells: int
        ) -> DissipativeFluxesSetup:

    basepath = "conservatives"

    # DISSIPATIVE FLUXES
    is_viscous_flux = active_physics_setup.is_viscous_flux
    is_heat_flux = active_physics_setup.is_heat_flux

    is_optional = False if is_heat_flux or is_viscous_flux else True
    path_dissipative_fluxes = get_path_to_key(basepath, "dissipative_fluxes")
    dissipative_fluxes_dict = get_setup_value(
        conservatives_dict, "dissipative_fluxes", path_dissipative_fluxes, dict,
        is_optional=is_optional, default_value={})

    is_optional = False if is_heat_flux or is_viscous_flux else True
    path = get_path_to_key(path_dissipative_fluxes, "derivative_stencil_face")
    derivative_stencil_face_str = get_setup_value(
        dissipative_fluxes_dict, "derivative_stencil_face", path, str,
        is_optional=is_optional, default_value="CENTRAL4",
        possible_string_values=tuple(DICT_DERIVATIVE_FACE.keys()))

    is_optional = False if is_viscous_flux else True
    path = get_path_to_key(path_dissipative_fluxes, "reconstruction_stencil")
    reconstruction_stencil_str = get_setup_value(
        dissipative_fluxes_dict, "reconstruction_stencil", path, str,
        is_optional=is_optional, default_value="CENTRAL4",
        possible_string_values=tuple(CENTRAL_RECONSTRUCTION_DICT.keys()))
    
    is_optional = False if is_viscous_flux else True
    path = get_path_to_key(path_dissipative_fluxes, "derivative_stencil_center")
    derivative_stencil_center_str = get_setup_value(
        dissipative_fluxes_dict, "derivative_stencil_center", path, str,
        is_optional=is_optional, default_value="CENTRAL4",
        possible_string_values=tuple(DICT_FIRST_DERIVATIVE_CENTER.keys()))
    
    path = get_path_to_key(path_dissipative_fluxes, "is_laplacian")
    is_laplacian = get_setup_value(
        dissipative_fluxes_dict, "is_laplacian", path, bool,
        is_optional=True, default_value=False)
    
    is_optional = False if is_laplacian else True
    path = get_path_to_key(path_dissipative_fluxes, "second_derivative_stencil_center")
    second_derivative_stencil_center_str = get_setup_value(
        dissipative_fluxes_dict, "second_derivative_stencil_center", path, str,
        is_optional=is_optional, default_value="CENTRAL4",
        possible_string_values=tuple(DICT_SECOND_DERIVATIVE_CENTER.keys()))

    reconstruction_stencil = CENTRAL_RECONSTRUCTION_DICT[reconstruction_stencil_str]
    derivative_stencil_center = DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_center_str]
    derivative_stencil_face = DICT_DERIVATIVE_FACE[derivative_stencil_face_str]
    second_derivative_stencil_center = DICT_SECOND_DERIVATIVE_CENTER[second_derivative_stencil_center_str]

    required_halos = reconstruction_stencil.required_halos
    required_halos = max(required_halos, derivative_stencil_center.required_halos)
    required_halos = max(required_halos, derivative_stencil_face.required_halos)

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but spatial reconstruction"
        f"stencil for the dissipative fluxes requires at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string

    dissipative_fluxes_setup = DissipativeFluxesSetup(
        reconstruction_stencil, derivative_stencil_center,
        derivative_stencil_face, is_laplacian,
        second_derivative_stencil_center)

    return dissipative_fluxes_setup