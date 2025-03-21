from typing import Dict, Any

from jaxfluids.data_types.numerical_setup.conservatives import ConservativesSetup
from jaxfluids.data_types.numerical_setup.levelset import *
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.solvers.convective_fluxes import ALDM, HighOrderGodunov, FluxSplittingScheme, CentralScheme
from jaxfluids.levelset.reinitialization import DICT_LEVELSET_REINITIALIZER
from jaxfluids.levelset.extension import EXTENSION_METHODS_TUPLE
from jaxfluids.stencils import (DICT_DERIVATIVE_REINITIALIZATION, 
    DICT_DERIVATIVE_LEVELSET_ADVECTION, DICT_DERIVATIVE_FACE, CENTRAL_RECONSTRUCTION_DICT,
    DICT_FIRST_DERIVATIVE_CENTER)
from jaxfluids.time_integration import DICT_TIME_INTEGRATION
from jaxfluids.levelset.reinitialization import DICT_LEVELSET_REINITIALIZER
from jaxfluids.levelset import (TUPLE_LEVELSET_MODELS, TUPLE_NORMAL_COMPUTATION_METHOD, TUPLE_SOLID_COUPLINGS,
                                TUPLE_VISCOUS_FLUX_METHOD, TUPLE_INTERFACE_RECONSTRUCTION_METHOD)
from jaxfluids.levelset.fluid_fluid import TUPLE_INTERFACE_FLUX_METHOD, TUPLE_MATERIAL_PROPERTIES_AVERAGING
from jaxfluids.input.numerical_setup import get_setup_value, loop_fields, get_path_to_key
from jaxfluids.input.setup_reader import assert_numerical


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

    if model == "FLUID-SOLID":
        path_solid_coupling = get_path_to_key(basepath, "solid_coupling")
        solid_coupling_dict = get_setup_value(
            levelset_dict, "solid_coupling", path_solid_coupling,
            dict, is_optional=True, default_value={})
        path = get_path_to_key(path_solid_coupling, "thermal")
        thermal = get_setup_value(
            solid_coupling_dict, "thermal", path, str,
            is_optional=True, default_value=False,
            possible_string_values=TUPLE_SOLID_COUPLINGS)
        path = get_path_to_key(path_solid_coupling, "dynamic")
        dynamic = get_setup_value(
            solid_coupling_dict, "dynamic", path, str,
            is_optional=True, default_value=False,
            possible_string_values=TUPLE_SOLID_COUPLINGS)
        solid_coupling_setup = SolidCouplingSetup(
            thermal, dynamic)
    else:
        solid_coupling_setup = SolidCouplingSetup(
            False, False)

    is_optional = False if model else True
    path = get_path_to_key(basepath, "halo_cells")
    halos_geometry = get_setup_value(
        levelset_dict, "halo_cells", path, int,
        is_optional=is_optional, default_value=2,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "levelset_advection_stencil")
    levelset_advection_stencil_str = get_setup_value(
        levelset_dict, "levelset_advection_stencil", path, str,
        is_optional=True, default_value="HOUC5",
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
    extension_setup = read_extension(levelset_dict, model, solid_coupling_setup)
    mixing_setup = read_mixing(levelset_dict, unit_handler, model, solid_coupling_setup)
    reinitialization_setup_runtime, reinitialization_setup_startup = read_reinitialization(model, levelset_dict, halos_conservatives)
    interface_flux_setup = read_interface_flux(levelset_dict, model, solid_coupling_setup)
    solid_heat_flux_setup = read_solid_heat_flux(levelset_dict)

    levelset_setup = LevelsetSetup(
        halos_geometry, model, solid_coupling_setup,
        levelset_advection_stencil,
        narrowband_setup, geometry_setup, extension_setup,
        mixing_setup, reinitialization_setup_runtime,
        reinitialization_setup_startup,
        interface_flux_setup, solid_heat_flux_setup)

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
        is_optional=True, default_value=4,
        numerical_value_condition=(">", 0))

    path = get_path_to_key(basepath, "cutoff_width")
    cutoff_width = get_setup_value(
        narrowband_dict, "cutoff_width", path, int,
        is_optional=True, default_value=10,
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
        levelset_model: str,
        solid_coupling: SolidCouplingSetup
        ) -> LevelsetGeometryComputationSetup:

    basepath = get_path_to_key("levelset", "extension")
    extension_dict = get_setup_value(
        levelset_dict, "extension", basepath, dict,
        is_optional=True, default_value={})

    def _read_setup(
            field: str,
            field_dict: Dict,
            field_path: str,
            ) -> LevelsetExtensionFieldSetup:
    
        path = get_path_to_key(field_path, "method")
        method = get_setup_value(
            field_dict, "method", path, str,
            is_optional=True, default_value="ITERATIVE",
            possible_string_values=EXTENSION_METHODS_TUPLE)

        if levelset_model == "FLUID-FLUID":
            assert method == "ITERATIVE", "FLUID-FLUID levelset requires ITERATIVE extension."
        
        path = get_path_to_key(field_path, "is_stopgradient")
        is_stopgradient = get_setup_value(
            field_dict, "is_stopgradient", path, bool,
            is_optional=True, default_value=False)

        # iterative
        path = get_path_to_key(field_path, "iterative")
        iterative_dict = get_setup_value(
            field_dict, "iterative", path, dict,
            is_optional=True, default_value={})
        
        path = get_path_to_key(field_path, "steps")
        steps = get_setup_value(
            iterative_dict, "steps", path, int,
            is_optional=True, default_value=40,
            numerical_value_condition=(">=", 0))

        path = get_path_to_key(field_path, "CFL")
        CFL = get_setup_value(
            iterative_dict, "CFL", path, float,
            is_optional=True, default_value=0.25,
            numerical_value_condition=(">", 0.0))
        
        path = get_path_to_key(field_path, "is_jaxwhileloop")
        is_jaxwhileloop = get_setup_value(
            iterative_dict, "is_jaxwhileloop", path, bool,
            is_optional=True, default_value=True)
        
        path = get_path_to_key(field_path, "residual_threshold")
        residual_threshold = get_setup_value(
            iterative_dict, "residual_threshold", path, float,
            is_optional=True, default_value=1e-2,
            numerical_value_condition=(">=", 0.0))
        
        path = get_path_to_key(field_path, "is_extend_into_invalid_mixing_cells")
        is_extend_into_invalid_mixing_cells = get_setup_value(
            iterative_dict, "is_extend_into_invalid_mixing_cells", path, bool,
            is_optional=True, default_value=True)
        
        path = get_path_to_key(field_path, "is_interpolate_invalid_cells")
        is_interpolate_invalid_cells = get_setup_value(
            iterative_dict, "is_interpolate_invalid_cells", path, bool,
            is_optional=True, default_value=False)


        iterative_setup = IterativeExtensionSetup(
            steps, CFL, is_jaxwhileloop,
            residual_threshold, is_interpolate_invalid_cells,
            is_extend_into_invalid_mixing_cells
            )

        # interpolation
        path = get_path_to_key(field_path, "interpolation")
        interpolation_dict = get_setup_value(
            field_dict, "interpolation", path, dict,
            is_optional=True, default_value={})
        
        default_value = False if solid_coupling.dynamic or method == "ITERATIVE" else True
        path = get_path_to_key(field_path, "is_cell_based_computation")
        is_cell_based_computation = get_setup_value(
            interpolation_dict, "is_cell_based_computation", path, bool,
            is_optional=True, default_value=default_value)
        
        if solid_coupling.dynamic and method == "INTERPOLATION":
            assert is_cell_based_computation == False, "cell based interpolation extension not implemented for moving level-set."
        
        interpolation_setup = InterpolationExtensionSetup(is_cell_based_computation)

        extension_setup = LevelsetExtensionFieldSetup(
            method, iterative_setup,
            interpolation_setup,
            is_stopgradient
            )

        return extension_setup

    extension_dict = levelset_dict.get("extension", {})
    extension_field_dict = {}
    for field in ("primitives", "interface", "solids"):
        path = get_path_to_key(basepath, field)
        extension_case_setup_dict = get_setup_value(
            extension_dict, field, path, dict,
            is_optional=True, default_value={})
        extension_setup = _read_setup(field, extension_case_setup_dict, path)
        extension_field_dict[field] = extension_setup

    extension_setup = LevelsetExtensionSetup(**extension_field_dict)

    return extension_setup

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
        
        default_value = "GODUNOVHAMILTONIAN" if is_runtime else "RUSSO"
        path = get_path_to_key(basepath, "type")
        reinitializer_type_str = get_setup_value(
            reinitialization_dict, "type", path, str,
            is_optional=True, default_value=default_value,
            possible_string_values=tuple(DICT_LEVELSET_REINITIALIZER.keys()))
        reinitializer = DICT_LEVELSET_REINITIALIZER[reinitializer_type_str]

        path = get_path_to_key(basepath, "time_integrator")
        time_integrator_str = get_setup_value(
            reinitialization_dict, "time_integrator", path, str,
            is_optional=True, default_value="EULER",
            possible_string_values=tuple(DICT_TIME_INTEGRATION.keys()))
        time_integrator = DICT_TIME_INTEGRATION[time_integrator_str]

        path = get_path_to_key(basepath, "spatial_stencil")
        spatial_stencil_str = get_setup_value(
            reinitialization_dict, "spatial_stencil", path, str,
            is_optional=True, default_value="WENO3HJ",
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

        path = get_path_to_key(basepath, "remove_underresolved")
        remove_underresolved = get_setup_value(
            reinitialization_dict, "remove_underresolved", path, bool,
            is_optional=True, default_value=False)
        
        path = get_path_to_key(basepath, "is_jaxwhileloop")
        is_jaxwhileloop = get_setup_value(
            reinitialization_dict, "is_jaxwhileloop", path, bool,
            is_optional=True, default_value=False)
        
        path = get_path_to_key(basepath, "residual_threshold")
        residual_threshold = get_setup_value(
            reinitialization_dict, "residual_threshold", path, float,
            is_optional=True, default_value=1e-2,
            numerical_value_condition=(">=", 0.0))

        reinitialization_setup = LevelsetReinitializationSetup(
            reinitializer, time_integrator, spatial_stencil,
            CFL, interval, steps, is_cut_cell,
            remove_underresolved, is_jaxwhileloop,
            residual_threshold)

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
            DICT_TIME_INTEGRATION["EULER"],
            DICT_DERIVATIVE_REINITIALIZATION["WENO3HJ"],
            0.25, 1, 0, True, False, True, 0.0)

    return reinitialization_setup_runtime, reinitialization_setup_startup

def read_mixing(
        levelset_dict: Dict,
        unit_handler: UnitHandler,
        levelset_model: bool,
        solid_coupling: SolidCouplingSetup
        ) -> LevelsetMixingSetup:

    def _read_setup(
            field_dict: Dict,
            field_path: str,
            ) -> LevelsetMixingSetup:
        
        path = get_path_to_key(field_path, "volume_fraction_threshold")
        volume_fraction_threshold = get_setup_value(
            field_dict, "volume_fraction_threshold", path, float,
            is_optional=True, default_value=0.6,
            numerical_value_condition=(">=", 0.0))

        path = get_path_to_key(field_path, "mixing_targets")
        mixing_targets = get_setup_value(
            field_dict, "mixing_targets", path, int,
            is_optional=True, default_value=1,
            numerical_value_condition=(">", 0))
        
        path = get_path_to_key(field_path, "is_interpolate_invalid_cells")
        is_interpolate_invalid_cells = get_setup_value(
            field_dict, "is_interpolate_invalid_cells", path, bool,
            is_optional=True, default_value=False)

        path = get_path_to_key(field_path, "normal_computation_method")
        normal_computation_method = get_setup_value(
            field_dict, "normal_computation_method", path, str,
            is_optional=True, default_value="FINITEDIFFERENCE",
            possible_string_values=TUPLE_NORMAL_COMPUTATION_METHOD)

        default_value = False if any((levelset_model == "FLUID-FLUID", solid_coupling.dynamic)) else True
        path = get_path_to_key(field_path, "is_cell_based_computation")
        is_cell_based_computation = get_setup_value(
            field_dict, "is_cell_based_computation", path, bool,
            is_optional=True, default_value=default_value)
        
        if solid_coupling.dynamic:
            assert is_cell_based_computation == False, "cell based mixing not implemented for moving level-set."

        mixing_setup = LevelsetMixingFieldSetup(
            mixing_targets, volume_fraction_threshold,
            is_interpolate_invalid_cells,
            normal_computation_method,
            is_cell_based_computation)
        
        return mixing_setup

    basepath = get_path_to_key("levelset", "mixing")
    mixing_dict = get_setup_value(
        levelset_dict, "mixing", basepath, dict,
        is_optional=True, default_value={})

    mixing_field_dict = {}
    for field in ("conservatives", "solids"):
        path = get_path_to_key(basepath, field)
        mixing_case_setup_dict = get_setup_value(
            mixing_dict, field, path, dict,
            is_optional=True, default_value={})
        mixing_field_setup = _read_setup(mixing_case_setup_dict, path)
        mixing_field_dict[field] = mixing_field_setup

    mixing_setup = LevelsetMixingSetup(**mixing_field_dict)

    return mixing_setup

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

    path = get_path_to_key(basepath, "interface_reconstruction_method")
    interface_reconstruction_method = get_setup_value(
        geometry_dict, "interface_reconstruction_method", path, str,
        is_optional=True, default_value="MARCHINGSQUARES",
        possible_string_values=TUPLE_INTERFACE_RECONSTRUCTION_METHOD
        )
    
    is_optional = False if interface_reconstruction_method == "NEURALNETWORK" else True
    path = get_path_to_key(basepath, "path_nn")
    path_nn = get_setup_value(
        geometry_dict, "path_nn", path, str,
        is_optional=is_optional)
    
    path = get_path_to_key(basepath, "symmetries_nn")
    symmetries_nn = get_setup_value(
        geometry_dict, "symmetries_nn", path, int,
        is_optional=True, default_value=1,
        numerical_value_condition=(">",0)
        )
    
    path = get_path_to_key(basepath, "subcell_reconstruction")
    subcell_reconstruction = get_setup_value(
        geometry_dict, "subcell_reconstruction", path, bool,
        is_optional=True, default_value=False
        )

    geometry = LevelsetGeometryComputationSetup(
        DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_normal_str],
        DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_curvature_str],
        interface_reconstruction_method,
        path_nn, symmetries_nn,
        subcell_reconstruction
        )
    
    required_halos = geometry.derivative_stencil_normal.required_halos
    required_halos = max(required_halos, geometry.derivative_stencil_curvature.required_halos)

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halos_conservatives:d} "
        f"but levelset geometry stencils require {required_halos:d}"
    )
    assert halos_conservatives >= required_halos, assert_string
   
    return geometry

def read_interface_flux(
        levelset_dict: Dict,
        model: bool,
        solid_coupling: SolidCouplingSetup
        ) -> InterfaceFluxSetup:

    basepath = get_path_to_key("levelset", "interface_flux")
    interface_flux_dict = get_setup_value(
        levelset_dict, "interface_flux", basepath, dict,
        is_optional=True, default_value={})
    
    default_value = "CELLCENTER" if model == "FLUID-FLUID" else "INTERPOLATION" 
    path = get_path_to_key(basepath, "method")
    method = get_setup_value(
        interface_flux_dict, "method", path, str,
        is_optional=True, default_value=default_value,
        possible_string_values=TUPLE_INTERFACE_FLUX_METHOD)

    path = get_path_to_key(basepath, "derivative_stencil")
    derivative_stencil = get_setup_value(
        interface_flux_dict, "derivative_stencil", path, str,
        is_optional=True, default_value="CENTRAL4",
        possible_string_values=DICT_FIRST_DERIVATIVE_CENTER.keys())

    path = get_path_to_key(basepath, "material_properties_averaging")
    material_properties_averaging = get_setup_value(
        interface_flux_dict, "material_properties_averaging", path, str,
        is_optional=True, default_value="HARMONIC",
        possible_string_values=TUPLE_MATERIAL_PROPERTIES_AVERAGING)

    path = get_path_to_key(basepath, "interpolation_dh")
    interpolation_dh = get_setup_value(
        interface_flux_dict, "interpolation_dh", path, float,
        is_optional=True, default_value=0.5, numerical_value_condition=(">", 0.0))

    path = get_path_to_key(basepath, "is_interpolate_pressure")
    is_interpolate_pressure = get_setup_value(
        interface_flux_dict, "is_interpolate_pressure", path, bool,
        is_optional=True, default_value=False)
    
    default_value = False if any((model == "FLUID-FLUID", solid_coupling.dynamic)) else True

    path = get_path_to_key(basepath, "is_cell_based_computation")
    is_cell_based_computation = get_setup_value(
        interface_flux_dict, "is_cell_based_computation", path, bool,
        is_optional=True, default_value=default_value)
    
    if solid_coupling.dynamic:
        assert is_cell_based_computation == False, "cell based interface flux not implemented for moving level-set."
    
    interface_flux_setup = InterfaceFluxSetup(
        method, DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil],
        material_properties_averaging, interpolation_dh,
        is_interpolate_pressure, is_cell_based_computation)
    
    return interface_flux_setup


def read_solid_heat_flux(levelset_dict: Dict) -> SolidHeatFluxSetup:

    basepath = get_path_to_key("levelset", "solid_heat_flux")
    solid_heat_flux_dict = get_setup_value(
        levelset_dict, "solid_heat_flux", basepath, dict,
        is_optional=True, default_value={})
    
    path = get_path_to_key(basepath, "derivative_stencil")
    derivative_stencil = get_setup_value(
        solid_heat_flux_dict, "derivative_stencil", path, str,
        is_optional=True, default_value="CENTRAL4",
        possible_string_values=DICT_DERIVATIVE_FACE.keys())

    path = get_path_to_key(basepath, "reconstruction_stencil")
    reconstruction_stencil = get_setup_value(
        solid_heat_flux_dict, "reconstruction_stencil", path, str,
        is_optional=True, default_value="CENTRAL4",
        possible_string_values=CENTRAL_RECONSTRUCTION_DICT.keys())
    
    solid_heat_flux_setup = SolidHeatFluxSetup(
        DICT_DERIVATIVE_FACE[derivative_stencil],
        CENTRAL_RECONSTRUCTION_DICT[reconstruction_stencil],
        )
    
    return solid_heat_flux_setup

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

        if convective_solver == HighOrderGodunov:
            reconstruction_stencil = convective_fluxes_setup.godunov.reconstruction_stencil
            split_reconstruction = convective_fluxes_setup.godunov.split_reconstruction
        
        elif convective_solver == FluxSplittingScheme:
            reconstruction_stencil = convective_fluxes_setup.flux_splitting.reconstruction_stencil
            split_reconstruction = convective_fluxes_setup.flux_splitting.split_reconstruction

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

    elif convective_solver == SGSNN:
        required_halos = 3

    elif convective_solver == CentralScheme:
        required_halos = convective_fluxes_setup.central.reconstruction_stencil.required_halos

    else:
        raise NotImplementedError

    derivative_stencil_center = dissipative_fluxes_setup.derivative_stencil_center
    derivative_stencil_face = dissipative_fluxes_setup.derivative_stencil_face
    reconstruction_stencil = dissipative_fluxes_setup.reconstruction_stencil

    required_halos = max(required_halos, derivative_stencil_center.required_halos)
    required_halos = max(required_halos, derivative_stencil_face.required_halos)
    required_halos = max(required_halos, reconstruction_stencil.required_halos)
    if levelset_setup.model == "FLUID-FLUID":
        levelset_advection_stencil = levelset_setup.levelset_advection_stencil
        required_halos = max(required_halos, levelset_advection_stencil.required_halos)

    narrowband_setup = levelset_setup.narrowband
    narrowband_cutoff_width = narrowband_setup.cutoff_width
    narrowband_computation_width = narrowband_setup.computation_width
    narrowband_offset = narrowband_cutoff_width - narrowband_computation_width

    # NOTE narrowband must be sufficiently large for cell face flux stencils
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Narrowband computation width is {narrowband_computation_width:d} "
        f"but provided stencil setup requires more than {required_halos:d}.")
    assert narrowband_computation_width >= required_halos, assert_string

    # NOTE narrowband computation offset to cutoff must be
    # sufficiently large for levelset advection stencil
    geometry_setup = levelset_setup.geometry
    if narrowband_setup.perform_cutoff:
        required_halos = levelset_setup.levelset_advection_stencil.required_halos
        assert_string = (
            f"Consistency error in numerical setup file. "
            f"Difference of narrowband computation and cutoff is {narrowband_offset:d} "
            f"but provided stencil setup requires more than {required_halos:d}.")
        assert narrowband_offset >= required_halos, assert_string
    
    nh_conservatives = conservatives_setup.halo_cells
    nh_geometry = levelset_setup.halo_cells
    nh_offset = nh_conservatives - nh_geometry

    # NOTE diffusive fluid-fluid interface flux needs gradients on real fluid buffer which as nh geometry
    interface_flux = levelset_setup.interface_flux
    nh_interface_flux = interface_flux.derivative_stencil.required_halos
    required_halos = max(1, nh_interface_flux)

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Geometry halos is {nh_geometry:d} but provided "
        f"stencil setup requires at least {required_halos:d}.")
    assert nh_geometry >= required_halos, assert_string

    # NOTE normal and curvature need to have nh geometry
    required_halos = geometry_setup.derivative_stencil_normal.required_halos
    required_halos = max(required_halos, 2*geometry_setup.derivative_stencil_curvature.required_halos)

    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Difference of conservative and geometry halos is {nh_offset:d} "
        f"but provided stencil setup requires at least {required_halos:d}.")
    assert nh_offset >= required_halos, assert_string