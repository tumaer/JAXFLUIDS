from typing import Dict, Any

from jaxfluids.data_types.numerical_setup.conservatives import ConservativesSetup
from jaxfluids.data_types.numerical_setup.levelset import *
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.solvers.convective_fluxes import ALDM, HighOrderGodunov, FluxSplittingScheme
from jaxfluids.levelset.mixing import DICT_LEVELSET_MIXER
from jaxfluids.levelset.reinitialization import DICT_LEVELSET_REINITIALIZER
from jaxfluids.stencils import DICT_DERIVATIVE_REINITIALIZATION, \
    DICT_DERIVATIVE_LEVELSET_ADVECTION, \
    DICT_DERIVATIVE_QUANTITY_EXTENDER, DICT_FIRST_DERIVATIVE_CENTER
from jaxfluids.time_integration import DICT_TIME_INTEGRATION
from jaxfluids.levelset.reinitialization import DICT_LEVELSET_REINITIALIZER
from jaxfluids.levelset.mixing import DICT_LEVELSET_MIXER
from jaxfluids.levelset import TUPLE_LEVELSET_MODELS, TUPLE_MIXER_TYPE, TUPLE_VISCOUS_FLUX_METHOD
from jaxfluids.input.numerical_setup import get_setup_value, loop_fields, get_path_to_key


def read_levelset_setup(
        numerical_setup_dict: Dict,
        unit_handler: UnitHandler,
        conservatives_setup: ConservativesSetup
        ) -> LevelsetSetup:

    halos_conservatives = conservatives_setup.halo_cells

    basepath = "levelset"

    levelset_dict = get_setup_value(
        numerical_setup_dict, "levelset", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "model")
    model = get_setup_value(
        levelset_dict, "model", path, str,
        is_optional=True, default_value=False,
        possible_string_values=TUPLE_LEVELSET_MODELS)

    is_optional = False if model else True
    path = get_path_to_key(basepath, "halo_cells")
    halos_geometry = get_setup_value(
        levelset_dict, "halo_cells", path, int,
        is_optional=is_optional, default_value=2,
        numerical_value_condition=(">", 0))

    is_optional = True if model in ["FLUID-SOLID-STATIC", False] else False
    path = get_path_to_key(basepath, "levelset_advection_stencil")
    levelset_advection_stencil_str = get_setup_value(
        levelset_dict, "levelset_advection_stencil", path, str,
        is_optional=is_optional, default_value="HOUC5",
        possible_string_values=tuple(DICT_DERIVATIVE_LEVELSET_ADVECTION.keys()))
    levelset_advection_stencil = DICT_DERIVATIVE_LEVELSET_ADVECTION[levelset_advection_stencil_str]
    required_halos = levelset_advection_stencil.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halos_conservatives:d} but levelset advection stencil"
        f"stencil requires at least {required_halos:d}.")
    assert halos_conservatives >= required_halos, assert_string

    narrowband_setup = read_narrowband(levelset_dict)
    geometry_setup = read_geometry(levelset_dict, halos_conservatives)
    extension_setup = read_extension(levelset_dict,halos_conservatives,
                                     halos_geometry)
    mixing_setup = read_mixing(levelset_dict, unit_handler)
    reinitialization_setup_runtime, reinitialization_setup_startup = \
        read_reinitialization(model, levelset_dict, halos_conservatives)
    interface_flux_setup = read_interface_flux(levelset_dict)

    levelset_setup = LevelsetSetup(
        model, halos_geometry, levelset_advection_stencil,
        narrowband_setup, geometry_setup, extension_setup,
        mixing_setup, reinitialization_setup_runtime,
        reinitialization_setup_startup,
        interface_flux_setup)

    if model:
        sanity_check(levelset_setup, conservatives_setup)

    return levelset_setup

def read_narrowband(
        levelset_dict: Dict
        ) -> NarrowBandSetup:

    basepath = get_path_to_key("levelset", "narrowband")

    narrowband_dict = get_setup_value(
        levelset_dict, "narrowband", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "computation_width")
    computation_width = get_setup_value(
        narrowband_dict, "computation_width", path, int,
        is_optional=True, default_value=5,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "cutoff_width")
    cutoff_width = get_setup_value(
        narrowband_dict, "cutoff_width", path, int,
        is_optional=True, default_value=15,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "inactive_reinitialization_width")
    inactive_reinitialization_width = get_setup_value(
        narrowband_dict, "inactive_reinitialization_width", path, int,
        is_optional=True, default_value=0,
        numerical_value_condition=(">=", 0))

    path = get_path_to_key(basepath, "perform_cutoff")
    perform_cutoff = get_setup_value(
        narrowband_dict, "perform_cutoff", path, bool,
        is_optional=True, default_value=True)

    narrowband_setup = NarrowBandSetup(
        cutoff_width, computation_width,
        inactive_reinitialization_width, perform_cutoff)

    return narrowband_setup

def read_extension(
        levelset_dict: Dict,
        halos_conservatives: int,
        halos_geometry: int
        ) -> LevelsetGeometryComputationSetup:

    basepath = get_path_to_key("levelset", "extension")
    extension_dict = get_setup_value(
        levelset_dict, "extension", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "time_integrator")
    time_integrator_str = get_setup_value(
        extension_dict, "time_integrator", path, str,
        is_optional=True, default_value="EULER",
        possible_string_values=tuple(DICT_TIME_INTEGRATION.keys()))
    time_integrator = DICT_TIME_INTEGRATION[time_integrator_str]

    path = get_path_to_key(basepath, "spatial_stencil")
    spatial_stencil_str = get_setup_value(
        extension_dict, "spatial_stencil", path, str,
        is_optional=True, default_value="FIRSTORDER",
        possible_string_values=tuple(DICT_DERIVATIVE_QUANTITY_EXTENDER.keys()))
    spatial_stencil = DICT_DERIVATIVE_QUANTITY_EXTENDER[spatial_stencil_str]
    required_halos = spatial_stencil.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halos_conservatives:d} but "
        f"prime extension stencil requires at least {required_halos:d}.")
    assert halos_conservatives >= required_halos, assert_string
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halos_conservatives:d} but "
        f"interface extension stencil requires at least {required_halos:d}.")
    assert halos_geometry >= required_halos, assert_string

    path = get_path_to_key(basepath, "steps_primes")
    steps_primes = get_setup_value(
        extension_dict, "steps_primes", path, int,
        is_optional=True, default_value=20,
        numerical_value_condition=(">=", 0))
    
    path = get_path_to_key(basepath, "CFL_primes")
    CFL_primes = get_setup_value(
        extension_dict, "CFL_primes", path, float,
        is_optional=True, default_value=0.25,
        numerical_value_condition=(">", 0.0))
    
    path = get_path_to_key(basepath, "steps_interface")
    steps_interface = get_setup_value(
        extension_dict, "steps_interface", path, int,
        is_optional=True, default_value=20,
        numerical_value_condition=(">=", 0))
    
    path = get_path_to_key(basepath, "CFL_interface")
    CFL_interface = get_setup_value(
        extension_dict, "CFL_interface", path, float,
        is_optional=True, default_value=0.25,
        numerical_value_condition=(">", 0.0))
    
    path = get_path_to_key(basepath, "reset_cells")
    reset_cells = get_setup_value(
        extension_dict, "reset_cells", path, bool,
        is_optional=True, default_value=False)
    
    path = get_path_to_key(basepath, "is_jaxforloop")
    is_jaxforloop = get_setup_value(
        extension_dict, "is_jaxforloop", path, bool,
        is_optional=True, default_value=False)
    
    path = get_path_to_key(basepath, "is_jaxwhileloop")
    is_jaxwhileloop = get_setup_value(
        extension_dict, "is_jaxwhileloop", path, bool,
        is_optional=True, default_value=True)
    
    path = get_path_to_key(basepath, "residual_threshold")
    residual_threshold = get_setup_value(
        extension_dict, "residual_threshold", path, float,
        is_optional=True, default_value=5e-2,
        numerical_value_condition=(">=", 0.0))

    assert_string = ("Consistency error in numerical setup. "
                    "One must be True between is_jaxforloop and "
                    "is_jaxwhileloop.")
    assert sum([is_jaxforloop, is_jaxwhileloop]) == 1, assert_string

    extension_dict = levelset_dict.get("extension", {})
    time_integrator_str = extension_dict.get("time_integrator", "EULER")
    spatial_stencil_str = extension_dict.get("spatial_stencil", "FIRSTORDER")
    extension = LevelsetExtensionSetup(
        time_integrator, spatial_stencil,
        steps_primes, CFL_primes, steps_interface,
        CFL_interface, reset_cells, is_jaxforloop,
        is_jaxwhileloop, residual_threshold)

    return extension

def read_reinitialization(
        model: str,
        levelset_dict: Dict,
        halos_conservatives: int
        ) -> LevelsetReinitializationSetup:

    def _read_setup(
            reinitialization_dict: Dict,
            basepath: str,
            is_runtime: bool
            ) -> LevelsetReinitializationSetup:
        
        default_value = "GODUNOVHAMILTONIAN" if is_runtime else "MIN"
        path = get_path_to_key(basepath, "type")
        reinitializer_type_str = get_setup_value(
            reinitialization_dict, "type", path, str,
            is_optional=True, default_value=default_value,
            possible_string_values=tuple(DICT_LEVELSET_REINITIALIZER.keys()))
        reinitializer = DICT_LEVELSET_REINITIALIZER[reinitializer_type_str]

        path = get_path_to_key(basepath, "time_integrator")
        time_integrator_str = get_setup_value(
            reinitialization_dict, "time_integrator", path, str,
            is_optional=True, default_value="RK2",
            possible_string_values=tuple(DICT_TIME_INTEGRATION.keys()))
        time_integrator = DICT_TIME_INTEGRATION[time_integrator_str]

        path = get_path_to_key(basepath, "spatial_stencil")
        spatial_stencil_str = get_setup_value(
            reinitialization_dict, "spatial_stencil", path, str,
            is_optional=True, default_value="WENO3DERIV",
            possible_string_values=tuple(DICT_DERIVATIVE_REINITIALIZATION.keys()))
        spatial_stencil = DICT_DERIVATIVE_REINITIALIZATION[spatial_stencil_str]
        required_halos = spatial_stencil.required_halos
        assert_string = (
            f"Consistency error in numerical setup file. "
            f"Number of conservative halos is {halos_conservatives:d} but "
            f"reinitialization stencil requires at least {required_halos:d}.")
        assert halos_conservatives >= required_halos, assert_string

        is_optional = True if is_runtime else False
        path = get_path_to_key(basepath, "CFL")
        CFL = get_setup_value(
            reinitialization_dict, "CFL", path, float,
            is_optional=is_optional, default_value=0.25)

        default_value = 40 if model == "FLUID-FLUID" else 0
        is_optional = True if is_runtime else False
        path = get_path_to_key(basepath, "steps")
        steps = get_setup_value(
            reinitialization_dict, "steps", path, int,
            is_optional=is_optional, default_value=default_value,
            numerical_value_condition=(">=", 0))
                
        path = get_path_to_key(basepath, "interval")
        interval = get_setup_value(
            reinitialization_dict, "interval", path, int,
            is_optional=True, default_value=1,
            numerical_value_condition=(">", 0))

        path = get_path_to_key(basepath, "is_cut_cell")
        is_cut_cell = get_setup_value(
            reinitialization_dict, "is_cut_cell", path, bool,
            is_optional=True, default_value=True)
        
        path = get_path_to_key(basepath, "is_domain")
        is_domain = get_setup_value(
            reinitialization_dict, "is_domain", path, bool,
            is_optional=True, default_value=True)
        
        path = get_path_to_key(basepath, "is_halos")
        is_halos = get_setup_value(
            reinitialization_dict, "is_halos", path, bool,
            is_optional=True, default_value=False)

        path = get_path_to_key(basepath, "remove_underresolved")
        remove_underresolved = get_setup_value(
            reinitialization_dict, "remove_underresolved", path, bool,
            is_optional=True, default_value=True)
        
        path = get_path_to_key(basepath, "is_jaxforloop")
        is_jaxforloop = get_setup_value(
            reinitialization_dict, "is_jaxforloop", path, bool,
            is_optional=True, default_value=False)

        path = get_path_to_key(basepath, "is_jaxwhileloop")
        is_jaxwhileloop = get_setup_value(
            reinitialization_dict, "is_jaxwhileloop", path, bool,
            is_optional=True, default_value=True)
        
        path = get_path_to_key(basepath, "residual_threshold")
        residual_threshold = get_setup_value(
            reinitialization_dict, "residual_threshold", path, float,
            is_optional=True, default_value=1e-2,
            numerical_value_condition=(">=", 0.0))

        assert_string = ("Consistency error in numerical setup. "
                        "One must be True between is_jaxforloop and "
                        "is_jaxwhileloop.")
        assert sum([is_jaxforloop, is_jaxwhileloop]) == 1, assert_string

        reinitialization_setup = LevelsetReinitializationSetup(
            reinitializer, time_integrator, spatial_stencil,
            CFL, interval, steps, is_cut_cell, is_domain,
            is_halos, remove_underresolved, is_jaxforloop,
            is_jaxwhileloop, residual_threshold)

        return reinitialization_setup

    basepath = get_path_to_key("levelset", "reinitialization")
    reinitialization_dict = get_setup_value(
        levelset_dict, "reinitialization", basepath, dict,
        is_optional=True, default_value={})
    
    # RUNTIME REINITIALIZER
    if "runtime" in reinitialization_dict:
        path_runtime = get_path_to_key(basepath, "runtime")
        reinitialization_runtime_dict = get_setup_value(
            reinitialization_dict, "runtime", path_runtime, dict,
            is_optional=True, default_value={})
    else:
        path_runtime = basepath
        reinitialization_runtime_dict = reinitialization_dict

    reinitialization_setup_runtime = _read_setup(reinitialization_runtime_dict, path_runtime, True)

    # INITIAL CONDITION REINITIALIZER
    if "startup" in reinitialization_dict:
        path_initialization = get_path_to_key(basepath, "startup")
        reinitialization_init_dict = get_setup_value(
            reinitialization_dict, "startup", path_initialization, dict,
            is_optional=True, default_value={})
        reinitialize_initial_condition = True
    else:
        path_initialization = None
        reinitialize_initial_condition = False

    if reinitialize_initial_condition:
        reinitialization_setup_startup = _read_setup(reinitialization_init_dict, path_initialization, False)
    else:
        reinitialization_setup_startup = LevelsetReinitializationSetup(
            DICT_LEVELSET_REINITIALIZER["GODUNOVHAMILTONIAN"],
            DICT_TIME_INTEGRATION["RK2"],
            DICT_DERIVATIVE_REINITIALIZATION["WENO3DERIV"],
            0.25, 1, 0, True, True, False, False, True, False, 0.0)

    return reinitialization_setup_runtime, reinitialization_setup_startup

def read_mixing(
        levelset_dict: Dict,
        unit_handler: UnitHandler
        ) -> LevelsetMixingSetup:

    basepath = get_path_to_key("levelset", "mixing")
    mixing_dict = get_setup_value(
        levelset_dict, "mixing", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "type")
    mixer_type_str = get_setup_value(
        mixing_dict, "type", path, str,
        is_optional=True, default_value="LAUER",
        possible_string_values=TUPLE_MIXER_TYPE)
    mixer = DICT_LEVELSET_MIXER[mixer_type_str]

    path = get_path_to_key(basepath, "volume_fraction_threshold")
    volume_fraction_threshold = get_setup_value(
        mixing_dict, "volume_fraction_threshold", path, float,
        is_optional=True, default_value=0.6,
        numerical_value_condition=(">=", 0.0))

    path = get_path_to_key(basepath, "mixing_targets")
    mixing_targets = get_setup_value(
        mixing_dict, "mixing_targets", path, int,
        is_optional=True, default_value=1,
        numerical_value_condition=(">", 0))

    mixing = LevelsetMixingSetup(
        mixer, mixing_targets, volume_fraction_threshold)
    
    return mixing

def read_geometry(
        levelset_dict: Dict,
        halos_conservatives: int
        ) -> LevelsetGeometryComputationSetup:

    basepath = get_path_to_key("levelset", "geometry")
    geometry_dict = get_setup_value(
        levelset_dict, "geometry", basepath, dict,
        is_optional=True, default_value={})
    
    path = get_path_to_key(basepath, "derivative_stencil_normal")
    derivative_stencil_normal_str = get_setup_value(
        geometry_dict, "derivative_stencil_normal", path, str,
        is_optional=True, default_value="CENTRAL4",
        possible_string_values=tuple(DICT_FIRST_DERIVATIVE_CENTER.keys()))

    path = get_path_to_key(basepath, "derivative_stencil_curvature")
    derivative_stencil_curvature_str = get_setup_value(
        geometry_dict, "derivative_stencil_curvature", path, str,
        is_optional=True, default_value="CENTRAL2",
        possible_string_values=tuple(DICT_FIRST_DERIVATIVE_CENTER.keys()))

    path = get_path_to_key(basepath, "subcell_reconstruction")
    subcell_reconstruction = get_setup_value(
        geometry_dict, "subcell_reconstruction", path, bool,
        is_optional=True, default_value=False)

    geometry = LevelsetGeometryComputationSetup(
        DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_normal_str],
        DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_curvature_str],
        subcell_reconstruction)
    
    required_halos = geometry.derivative_stencil_normal.required_halos
    required_halos = max(required_halos, geometry.derivative_stencil_curvature.required_halos)

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halos_conservatives:d} "
        f"but levelset geometry stencils require {required_halos:d}"
    )
    assert halos_conservatives >= required_halos, assert_string
   
    return geometry

def read_interface_flux(levelset_dict: Dict) -> InterfaceFluxSetup:

    basepath = get_path_to_key("levelset", "interface_flux")
    interface_flux_dict = get_setup_value(
        levelset_dict, "interface_flux", basepath, dict,
        is_optional=True, default_value={})

    path = get_path_to_key(basepath, "viscous_flux_method")
    viscous_flux_method = get_setup_value(
        interface_flux_dict, "viscous_flux_method", path, str,
        is_optional=True, default_value="MEYER",
        possible_string_values=TUPLE_VISCOUS_FLUX_METHOD)
    
    path = get_path_to_key(basepath, "derivative_stencil")
    derivative_stencil = get_setup_value(
        interface_flux_dict, "derivative_stencil", path, str,
        is_optional=True, default_value="CENTRAL4",
        possible_string_values=DICT_FIRST_DERIVATIVE_CENTER.keys())
    
    interface_flux_setup = InterfaceFluxSetup(
        viscous_flux_method,
        DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil])
    
    return interface_flux_setup

def sanity_check(
        levelset_setup: LevelsetSetup,
        conservatives_setup: ConservativesSetup
        ) -> None:
    """Performs sanity checks for the halo cells
    and narrowband widths. The narrowband computation
    width must be greater than the required halos of the
    conservative spatial stencils,
    i.e., reconstruction stencil for convective fluxes and
    spatial stencils for the dissipative flxues.
    The difference between narrowband cutoff width and
    computation width must be greater than
    1) the required halos of the levelset 
    advection stencil and 2) the sum of the required
    halos of the residual computation stencil and 
    the geometry stencil.
    The difference between conservative and geometry
    halo cells must be greater than or equal to the
    required halos of the (interface) extension stencil
    and the spatial stencils for the dissipative
    fluxes.

    :param levelset_setup: _description_
    :type levelset_setup: LevelsetSetup
    :param conservatives_setup: _description_
    :type conservatives_setup: ConservativesSetup
    """
    
    # REQUIRED NARROWBAND GHOST CELLS FOR CONVECTIVE/DISSIPATIVE FLUXES STENCILS
    convective_fluxes_setup = conservatives_setup.convective_fluxes
    dissipative_fluxes_setup = conservatives_setup.dissipative_fluxes


    convective_solver = convective_fluxes_setup.convective_solver

    if convective_solver in (HighOrderGodunov, FluxSplittingScheme):

        reconstruction_stencil = convective_fluxes_setup.reconstruction_stencil
        split_reconstruction = convective_fluxes_setup.split_reconstruction

        if reconstruction_stencil is not None \
            and split_reconstruction is None:
            required_halos = reconstruction_stencil.required_halos
    
        elif reconstruction_stencil is None \
            and split_reconstruction is not None:
            required_halos_list = []
            for field in split_reconstruction._fields:
                field_reconstructor = getattr(split_reconstruction, field)
                if field_reconstructor is not None:
                    required_halos_field = field_reconstructor.required_halos
                required_halos_list.append(required_halos_field)
            required_halos = max(required_halos_list)
    
    elif convective_solver == ALDM:
        required_halos = 3

    else:
        raise NotImplementedError

    derivative_stencil_center = dissipative_fluxes_setup.derivative_stencil_center
    derivative_stencil_face = dissipative_fluxes_setup.derivative_stencil_face
    reconstruction_stencil = dissipative_fluxes_setup.reconstruction_stencil

    required_halos = max(required_halos, derivative_stencil_center.required_halos)
    required_halos = max(required_halos, derivative_stencil_face.required_halos)
    required_halos = max(required_halos, reconstruction_stencil.required_halos)

    narrowband_setup = levelset_setup.narrowband
    narrowband_cutoff_width = narrowband_setup.cutoff_width
    narrowband_computation_width = narrowband_setup.computation_width
    narrowband_offset = narrowband_cutoff_width - narrowband_computation_width

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Narrowband computation width is {narrowband_computation_width:d} "
        f"but provided stencil setup requires more than {required_halos:d}.")
    assert narrowband_computation_width > required_halos, assert_string

    # REQUIRED NARROWBAND OFFSET FOR LEVELSET ADVECTION
    geometry_setup = levelset_setup.geometry
    if narrowband_setup.perform_cutoff:
        required_halos = levelset_setup.levelset_advection_stencil.required_halos
        assert_string = (
            f"Consistency error in numerical setup file. "
            f"Difference of narrowband computation and cutoff is {narrowband_offset:d} "
            f"but provided stencil setup requires more than {required_halos:d}.")
        assert narrowband_offset > required_halos, assert_string
    
    nh_conservatives = conservatives_setup.halo_cells
    nh_geometry = levelset_setup.halo_cells
    nh_offset = nh_conservatives - nh_geometry

    # REQUIRED GEOMETRY HALOS FOR REAL FLUID GRADIENTS AND EXTENSION
    extension_setup = levelset_setup.extension
    interface_flux = levelset_setup.interface_flux
    nh_extension = extension_setup.spatial_stencil.required_halos
    nh_interface_flux = interface_flux.derivative_stencil.required_halos
    required_halos = max(nh_interface_flux, nh_extension)

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Geometry halos is {nh_geometry:d} but provided "
        f"stencil setup requires at least {required_halos:d}.")
    assert nh_geometry >= required_halos, assert_string

    # REQUIRED OFFSET BETWEEN DOMAIN AND GEOMETRY HALOS FOR NORMAL/CURVATURE
    required_halos = geometry_setup.derivative_stencil_normal.required_halos
    required_halos = max(required_halos, 2*geometry_setup.derivative_stencil_curvature.required_halos)

    # TODO this checks are active even if no level-set is present???
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Difference of conservative and geometry halos is {nh_offset:d} "
        f"but provided stencil setup requires at least {required_halos:d}.")
    assert nh_offset >= required_halos, assert_string