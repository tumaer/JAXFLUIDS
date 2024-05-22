from typing import Dict, Any

from jaxfluids.stencils import (DICT_DERIVATIVE_FACE, DICT_FIRST_DERIVATIVE_CENTER,
    DICT_CENTRAL_RECONSTRUCTION)
from jaxfluids.time_integration import DICT_TIME_INTEGRATION
from jaxfluids.data_types.numerical_setup.diffuse_interface import *
from jaxfluids.data_types.numerical_setup import ConservativesSetup
from jaxfluids.diffuse_interface import (TUPLE_DIFFUSE_INTERFACE_MODELS,
    TUPLE_INTERFACE_PROJECTION, TUPLE_INTERFACE_TREATMENT, TUPLE_THINC_TYPES,
    TUPLE_VOLUME_FRACTION_TRANSFORMATION, TUPLE_DIFFUSION_SHARPENING_MODELS,
    TUPLE_DIFFUSION_SHARPENING_DENSITY_MODELS, TUPLE_MOBILITY_MODELS,
    TUPLE_SURFACE_TENSION_KERNEL)
from jaxfluids.input.numerical_setup import get_setup_value, get_path_to_key
from jaxfluids.unit_handler import UnitHandler

def read_diffuse_interface_setup(
        numerical_setup_dict: Dict,
        conservatives_setup: ConservativesSetup,
        unit_handler: UnitHandler
        ) -> DiffuseInterfaceSetup:
    """Reads the Diffuse Interface Setup from a dictionary.
    
    The main components of a diffuse interface setup are:
    1) GeometrySetup
    2) Interface Compression Setup
    3) THINC Setup
    4) Diffusion Sharpening Setup

    :param numerical_setup_dict: _description_
    :type numerical_setup_dict: Dict
    :param conservatives_setup: _description_
    :type conservatives_setup: ConservativesSetup
    :return: _description_
    :rtype: DiffuseInterfaceSetup
    """

    basepath = "diffuse_interface"
    diffuse_interface_dict = get_setup_value(
        numerical_setup_dict, "diffuse_interface", basepath, dict,
        is_optional=True, default_value={})

    # MODEL
    path = get_path_to_key(basepath, "model")
    model = get_setup_value(
        diffuse_interface_dict, "model", path, str,
        is_optional=True, default_value=False,
        possible_string_values=TUPLE_DIFFUSE_INTERFACE_MODELS)

    # HALO CELLS
    is_optional = False if model else True
    path = get_path_to_key(basepath, "halo_cells")
    halos_geometry = get_setup_value(
        diffuse_interface_dict, "halo_cells", path, int,
        is_optional=is_optional, default_value=2,
        numerical_value_condition=(">", 0))

    # CONSISTENT RECONSTRUCTION
    is_optional = True
    path = get_path_to_key(basepath, "is_consistent_reconstruction")
    is_consistent_reconstruction = get_setup_value(
        diffuse_interface_dict, "is_consistent_reconstruction", path,
        bool, is_optional=is_optional, default_value=False,)

    halos_conservatives = conservatives_setup.halo_cells
    geometry_setup = read_geometry(diffuse_interface_dict, halos_conservatives)
    thinc_setup = read_thinc(diffuse_interface_dict,)
    interface_compression_setup = read_interface_compression(diffuse_interface_dict)
    diffusion_sharpening_setup = read_diffusion_sharpening_setup(diffuse_interface_dict,
                                                                 unit_handler)
    
    is_thinc_reconstruction = thinc_setup.is_thinc_reconstruction
    is_interface_compression = interface_compression_setup.is_interface_compression
    is_diffusion_sharpening = diffusion_sharpening_setup.is_diffusion_sharpening
    assert_string = ("Consistency error in numerical setup file. "
                     "Multiple diffuse-interface sharpening procedures "
                     "are active "
                     f"(THINC: {is_thinc_reconstruction}, "
                     f"Compression: {is_interface_compression} "
                     f"CDI/ACDI: {is_diffusion_sharpening}). "
                     "Only one sharpening procedure allowed at a time.")
    assert sum((is_thinc_reconstruction,
                is_interface_compression,
                is_diffusion_sharpening)) <= 1, assert_string

    diffuse_interface_setup = DiffuseInterfaceSetup(
        model, halos_geometry, is_consistent_reconstruction,
        geometry_setup, interface_compression_setup,
        thinc_setup, diffusion_sharpening_setup)

    return diffuse_interface_setup

def read_geometry(
        diffuse_interface_dict: Dict,
        halo_cells: int
        ) -> GeometryCalculationSetup:
    # TODO DENIZ CHECK ASSERTS 
    basepath = get_path_to_key("diffuse_interface", "geometry_calculation")
    geometry_dict = get_setup_value(
        diffuse_interface_dict, "geometry_calculation", basepath, dict,
        is_optional=True, default_value={})
    
    path = get_path_to_key(basepath, "derivative_stencil_curvature")
    derivative_stencil_curvature_str = get_setup_value(
        geometry_dict, "derivative_stencil_curvature", path, str,
        is_optional=True, default_value="CENTRAL2",
        possible_string_values=tuple(DICT_FIRST_DERIVATIVE_CENTER.keys()))
    derivative_stencil_curvature = DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_curvature_str]
    required_halos = derivative_stencil_curvature.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but diffuse "
        f"interface geometry stencil requires at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string
    
    path = get_path_to_key(basepath, "derivative_stencil_center")
    derivative_stencil_center_str = get_setup_value(
        geometry_dict, "derivative_stencil_center", path, str,
        is_optional=True, default_value="CENTRAL2",
        possible_string_values=tuple(DICT_FIRST_DERIVATIVE_CENTER.keys()))
    derivative_stencil_center = DICT_FIRST_DERIVATIVE_CENTER[derivative_stencil_center_str]
    required_halos = derivative_stencil_center.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but diffuse "
        f"interface geometry stencil requires at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string

    path = get_path_to_key(basepath, "reconstruction_stencil")
    reconstruction_stencil_str = get_setup_value(
        geometry_dict, "reconstruction_stencil", path, str,
        is_optional=True, default_value="CENTRAL2",
        possible_string_values=tuple(DICT_CENTRAL_RECONSTRUCTION.keys()))
    reconstruction_stencil = DICT_CENTRAL_RECONSTRUCTION[reconstruction_stencil_str]
    required_halos = reconstruction_stencil.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but diffuse "
        f"interface geometry stencil requires at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string

    path = get_path_to_key(basepath, "derivative_stencil_face")
    derivative_stencil_face_str = get_setup_value(
        geometry_dict, "derivative_stencil_face", path, str,
        is_optional=True, default_value="CENTRAL2",
        possible_string_values=tuple(DICT_DERIVATIVE_FACE.keys()))
    derivative_stencil_face = DICT_DERIVATIVE_FACE[derivative_stencil_face_str]
    required_halos = derivative_stencil_face.required_halos
    assert_string = (
        f"Consistency error in numerical setup file. "
        f"Number of conservative halos is {halo_cells:d} but diffuse "
        f"interface geometry stencil requires at least {required_halos:d}.")
    assert halo_cells >= required_halos, assert_string
    
    path = get_path_to_key(basepath, "steps_curvature")
    steps_curvature = get_setup_value(
        geometry_dict, "steps_curvature", path, int,
        is_optional=True, default_value=0)
    
    path = get_path_to_key(basepath, "volume_fraction_mapping")
    volume_fraction_mapping = get_setup_value(
        geometry_dict, "volume_fraction_mapping", path, str,
        is_optional=True, default_value="SMOOTHING",
        possible_string_values=TUPLE_VOLUME_FRACTION_TRANSFORMATION)

    path = get_path_to_key(basepath, "interface_smoothing")
    interface_smoothing = get_setup_value(
        geometry_dict, "interface_smoothing", path, float,
        is_optional=True, default_value=0.1,
        numerical_value_condition=(">", 0.0))

    path = get_path_to_key(basepath, "surface_tension_kernel")
    surface_tension_kernel = get_setup_value(
        geometry_dict, "surface_tension_kernel", path, str,
        is_optional=True, default_value=False,
        possible_string_values=TUPLE_SURFACE_TENSION_KERNEL,)

    # GEOMETRY CALCULATION SETUP
    geometry_calculation = GeometryCalculationSetup(
        steps_curvature, volume_fraction_mapping,
        interface_smoothing, surface_tension_kernel,
        derivative_stencil_curvature, derivative_stencil_center,
        reconstruction_stencil, derivative_stencil_face)
    return geometry_calculation

def read_thinc(diffuse_interface_dict: Dict) -> THINCSetup:

    basepath = get_path_to_key("diffuse_interface", "thinc")
    thinc_dict = get_setup_value(
        diffuse_interface_dict, "thinc", basepath, dict,
        is_optional=True, default_value={})
    
    path = get_path_to_key(basepath, "is_thinc_reconstruction")
    is_thinc_reconstruction = get_setup_value(
        thinc_dict, "is_thinc_reconstruction", path, bool,
        is_optional=True, default_value=False)
    is_optional = not is_thinc_reconstruction
    
    path = get_path_to_key(basepath, "thinc_type")
    thinc_type = get_setup_value(
        thinc_dict, "thinc_type", path, str,
        is_optional=is_optional, default_value="RHOTHINC",
        possible_string_values=TUPLE_THINC_TYPES)
    
    path = get_path_to_key(basepath, "interface_treatment")
    interface_treatment = get_setup_value(
        thinc_dict, "interface_treatment", path, str,
        is_optional=is_optional, default_value="RHOTHINC",
        possible_string_values=TUPLE_INTERFACE_TREATMENT)
    
    path = get_path_to_key(basepath, "interface_projection")
    interface_projection = get_setup_value(
        thinc_dict, "interface_projection", path, str,
        is_optional=is_optional, default_value="NORM_1",
        possible_string_values=TUPLE_INTERFACE_PROJECTION)

    path = get_path_to_key(basepath, "interface_parameter")
    interface_parameter = get_setup_value(
        thinc_dict, "interface_parameter", path, float,
        is_optional=is_optional, default_value=2.3,
        possible_string_values=TUPLE_INTERFACE_PROJECTION)

    path = get_path_to_key(basepath, "volume_fraction_threshold")
    volume_fraction_threshold = get_setup_value(
        thinc_dict, "volume_fraction_threshold", path, float,
        is_optional=is_optional, default_value=1e-4,
        numerical_value_condition=(">", 0.0))

    thinc_setup = THINCSetup(is_thinc_reconstruction,
                             thinc_type,
                             interface_treatment,
                             interface_projection,
                             interface_parameter,
                             volume_fraction_threshold)

    return thinc_setup


def read_interface_compression(diffuse_interface_dict: Dict) -> InterfaceCompressionSetup:
    # TODO DENIZ CHECK ASSERTS 
    basepath = get_path_to_key("diffuse_interface", "interface_compression")
    interface_compression_dict = get_setup_value(
        diffuse_interface_dict, "interface_compression", basepath, dict,
        is_optional=True, default_value={})
    
    path = get_path_to_key(basepath, "is_interface_compression")
    is_interface_compression = get_setup_value(
        interface_compression_dict, "is_interface_compression", path,
        bool, is_optional=True, default_value=False)
    is_optional = not is_interface_compression

    path = get_path_to_key(basepath, "time_integrator")
    time_integrator_str = get_setup_value(
        interface_compression_dict, "time_integrator", path, str,
        is_optional=is_optional, default_value="EULER",
        possible_string_values=tuple(DICT_TIME_INTEGRATION.keys()))
    time_integrator = DICT_TIME_INTEGRATION[time_integrator_str]
    
    path = get_path_to_key(basepath, "CFL")
    CFL = get_setup_value(
        interface_compression_dict, "CFL", path, float,
        is_optional=is_optional, default_value=0.1,
        numerical_value_condition=(">", 0.0))
    
    path = get_path_to_key(basepath, "interval")
    interval = get_setup_value(
        interface_compression_dict, "interval", path, int,
        is_optional=is_optional, default_value=1,
        numerical_value_condition=(">=", 0))
    
    path = get_path_to_key(basepath, "steps")
    steps = get_setup_value(
        interface_compression_dict, "steps", path, int,
        is_optional=is_optional, default_value=0,
        numerical_value_condition=(">=", 0))

    path = get_path_to_key(basepath, "heaviside_parameter")
    heaviside_parameter = get_setup_value(
        interface_compression_dict, "heaviside_parameter", path, float,
        is_optional=is_optional, default_value=0.01,
        numerical_value_condition=(">", 0.0))
    
    path = get_path_to_key(basepath, "interface_thickness_parameter")
    interface_thickness_parameter = get_setup_value(
        interface_compression_dict, "interface_thickness_parameter", path, float,
        is_optional=is_optional, default_value=0.72,
        numerical_value_condition=(">", 0.0))

    interface_compression_setup = InterfaceCompressionSetup(is_interface_compression,
                                                            time_integrator,
                                                            CFL,
                                                            interval,
                                                            steps,
                                                            heaviside_parameter,
                                                            interface_thickness_parameter)

    return interface_compression_setup

def read_diffusion_sharpening_setup(
        diffuse_interface_dict: Dict,
        unit_handler: UnitHandler
        ) -> DiffusionSharpeningSetup:
    """Reads and sets up the DiffusionSharpeningSetup
    from the diffuse_interface_dict.

    :param diffuse_interface_dict: _description_
    :type diffuse_interface_dict: Dict
    :return: _description_
    :rtype: DiffusionSharpeningSetup
    """
    
    basepath = get_path_to_key("diffuse_interface", "diffusion_sharpening")
    is_optional = not "diffusion_sharpening" in diffuse_interface_dict
    diffusion_sharpening_dict = get_setup_value(
        diffuse_interface_dict, "diffusion_sharpening", basepath, dict,
        is_optional=is_optional, default_value={})
    
    path = get_path_to_key(basepath, "is_diffusion_sharpening")
    is_diffusion_sharpening = get_setup_value(
        diffusion_sharpening_dict, "is_diffusion_sharpening", path,
        bool, is_optional=is_optional, default_value=False)
    is_optional = not is_diffusion_sharpening

    path = get_path_to_key(basepath, "model")
    diffusion_sharpening_model = get_setup_value(
        diffusion_sharpening_dict, "model", path, str,
        is_optional=is_optional, default_value="ACDI",
        possible_string_values=TUPLE_DIFFUSION_SHARPENING_MODELS)
    
    path = get_path_to_key(basepath, "density_model")
    density_model = get_setup_value(
        diffusion_sharpening_dict, "density_model", path, str,
        is_optional=is_optional, default_value="COMPRESSIBLE",
        possible_string_values=TUPLE_DIFFUSION_SHARPENING_DENSITY_MODELS)
    
    path = get_path_to_key(basepath, "incompressible_density")
    incompressible_density = get_setup_value(
        diffusion_sharpening_dict, "incompressible_density", path, list,
        is_optional=not density_model == "INCOMPRESSIBLE",
        default_value=0.0)
    incompressible_density = jnp.array(incompressible_density)
    incompressible_density = unit_handler.non_dimensionalize(incompressible_density, "density")

    path = get_path_to_key(basepath, "interface_thickness_parameter")
    interface_thickness_parameter = get_setup_value(
        diffusion_sharpening_dict, "interface_thickness_parameter", path, float,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">=", 0.0))
    
    path = get_path_to_key(basepath, "interface_velocity_parameter")
    interface_velocity_parameter = get_setup_value(
        diffusion_sharpening_dict, "interface_velocity_parameter", path, float,
        is_optional=is_optional, default_value=0.0,
        numerical_value_condition=(">=", 0.0))
    
    path = get_path_to_key(basepath, "mobility_model")
    mobility_model = get_setup_value(
        diffusion_sharpening_dict, "mobility_model", path, str,
        is_optional=True, default_value=False,
        possible_string_values=TUPLE_MOBILITY_MODELS,)

    path = get_path_to_key(basepath, "acdi_threshold")
    acdi_threshold = get_setup_value(
        diffusion_sharpening_dict, "acdi_threshold", path, float,
        is_optional=True, default_value=0.0,
        numerical_value_condition=(">=", 0.0))

    path = get_path_to_key(basepath, "volume_fraction_threshold")
    volume_fraction_threshold = get_setup_value(
        diffusion_sharpening_dict, "volume_fraction_threshold", path, float,
        is_optional=True, default_value=1e-8,
        numerical_value_condition=(">", 0.0))

    diffusion_sharpening_setup = DiffusionSharpeningSetup(is_diffusion_sharpening,
                                                          diffusion_sharpening_model,
                                                          density_model,
                                                          incompressible_density,
                                                          interface_thickness_parameter,
                                                          interface_velocity_parameter,
                                                          mobility_model,
                                                          acdi_threshold,
                                                          volume_fraction_threshold)

    return diffusion_sharpening_setup
