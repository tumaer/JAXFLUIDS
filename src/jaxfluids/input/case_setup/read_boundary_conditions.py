from typing import Dict
import os
import h5py

from jaxfluids.data_types.case_setup import GetPrimitivesCallable, DomainSetup
from jaxfluids.data_types.case_setup.boundary_conditions import *
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.domain import FACE_LOCATIONS, AXES
from jaxfluids.equation_information import EquationInformation
from jaxfluids.input.setup_reader import get_path_to_key, create_wrapper_for_callable
from jaxfluids.input.case_setup import get_setup_value
from jaxfluids.halos.outer import PRIMITIVES_TYPES, LEVELSET_TYPES
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.domain.domain_information import DomainInformation


def get_active_axes_at_face(domain_setup: DomainSetup, face_location: str) -> Dict:
    active_axes = domain_setup.active_axes
    face_location_to_axis_indices = { 
        "east"  : ("y","z"), "west"  : ("y","z"),
        "north" : ("x","z"), "south" : ("x","z"),
        "top"   : ("x","y"), "bottom": ("x","y")
    }
    axes_tuple = face_location_to_axis_indices[face_location]
    active_axes_at_face = tuple([axis for axis in axes_tuple if axis in active_axes])
    return active_axes_at_face

def read_boundary_condition_setup(
        case_setup_dict: Dict,
        equation_information: EquationInformation,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler,
        domain_setup: DomainSetup
        ) -> BoundaryConditionSetup:
    """Reads the case setup and initializes
    the boundary conditions.
    """
    basepath = "boundary_conditions"
    boundary_conditions_case_setup = get_setup_value(
        case_setup_dict, "boundary_conditions", basepath, dict,
        is_optional=False)

    active_models = ["primitives"]
    if equation_information.levelset_model:
        active_models.append("levelset")

    boundary_conditions_dict = {}
    for field in active_models:
        if equation_information.levelset_model:
            path_field = get_path_to_key(basepath, field)
            boundary_condition_field_case_setup = get_setup_value(
                boundary_conditions_case_setup, field, path_field,
                dict, is_optional=False)
        else:
            path_field = basepath
            boundary_condition_field_case_setup = boundary_conditions_case_setup

        boundary_conditions_field_dict = {}
        for face_location in FACE_LOCATIONS:
            path_face = get_path_to_key(path_field, face_location)
            boundary_conditions_face_list_case_setup = get_setup_value(
                boundary_condition_field_case_setup, face_location,
                path_face, (dict, list), is_optional=False)

            if isinstance(boundary_conditions_face_list_case_setup, list):
                multiple_boundary_types_at_face = True
            else:
                multiple_boundary_types_at_face = False
                boundary_conditions_face_list_case_setup = [boundary_conditions_face_list_case_setup]

            boundary_conditions_face_list = []
            for boundary_conditions_face_case_setup in boundary_conditions_face_list_case_setup:

                boundary_conditions_face = read_boundary_condition_face(
                    boundary_conditions_face_case_setup, multiple_boundary_types_at_face,
                    path_face, field, face_location, equation_information, unit_handler,
                    domain_setup)
                boundary_conditions_face_list.append(boundary_conditions_face)

                if multiple_boundary_types_at_face:
                    boundary_type = boundary_conditions_face.boundary_type
                    assert_string = ("Consistency error in case setup file. "
                        f"Periodic boundary condition at face location {face_location} "
                        "with multiple boundary types not allowed.")
                    assert boundary_type != "PERIODIC", assert_string

            boundary_conditions_field_dict[face_location] = tuple(boundary_conditions_face_list)     

        boundary_conditions_field = BoundaryConditionsField(**boundary_conditions_field_dict)
        boundary_conditions_dict[field] = boundary_conditions_field
    boundary_condition_setup = BoundaryConditionSetup(**boundary_conditions_dict)

    sanity_check(boundary_condition_setup, domain_setup, numerical_setup)

    return boundary_condition_setup

def read_boundary_condition_face(
        boundary_conditions_face_case_setup: Dict,
        multiple_boundary_types_at_face: bool,
        basepath: str,
        field: str,
        face_location: str,
        equation_information: EquationInformation,
        unit_handler: UnitHandler,
        domain_setup: DomainSetup
        ) -> BoundaryConditionsFace:
    """Reads the boundary condition for a single
    boundary type along a face from the case setup .json
    file and creates the corresponding jaxfluids container.

    :param boundary_conditions_face_case_setup: _description_
    :type boundary_conditions_face_case_setup: Dict
    :param multiple_boundary_types_at_face: _description_
    :type multiple_boundary_types_at_face: bool
    :param field: _description_
    :type field: str
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: BoundaryConditionsFace
    """

    path_type = get_path_to_key(basepath, "type")
    boundary_type = get_setup_value(
        boundary_conditions_face_case_setup, "type",
        path_type, str, is_optional=False)

    assert_string = "Consistency error in case setup file. "\
        f"Boundary type {boundary_type} not implemented for {field}"
    if field == "primitives":
        assert boundary_type in PRIMITIVES_TYPES, assert_string
    if field == "levelset":
        assert boundary_type in LEVELSET_TYPES, assert_string

    bounding_domain_callable = None
    primitives_callable = None
    levelset_callable = None
    wall_velocity_callable = None
    wall_temperature_callable = None
    wall_mass_transfer = None
    primitives_table = None

    active_axes_at_face = get_active_axes_at_face(domain_setup, face_location)
    dim = domain_setup.dim
    input_argument_labels = active_axes_at_face + ("t",)
    input_argument_units = tuple(["length"] * (dim-1) + ["time"])

    if boundary_type in ["ZEROGRADIENT", "SYMMETRY", "PERIODIC", "INACTIVE"]:
        pass

    elif boundary_type in ["DIRICHLET", "NEUMANN"]:
        
        is_spatial_derivative = boundary_type == "NEUMANN"

        if field == "primitives":
            h5file_path = get_path_to_key(basepath, "h5file_path")
            h5file_path_str = get_setup_value(
                boundary_conditions_face_case_setup, "h5file_path",
                h5file_path, str, is_optional=True, default_value=None)
            path_callable = get_path_to_key(basepath, "primitives_callable")
            primitives_case_setup = get_setup_value(
                boundary_conditions_face_case_setup, "primitives_callable",
                path_callable, dict, is_optional=True, default_value=None)
            assert_string = ("Consistency error in case setup file. "
                f"Either h5file_path or primitives_callable must be given for {basepath:s}.")
            assert h5file_path_str != None or primitives_case_setup != None, assert_string

            if h5file_path_str != None:
                with h5py.File(h5file_path_str, "r") as h5file:
                    primitives = jnp.array(h5file["primitives"][:])
                    for axis in AXES:
                        if axis in h5file.keys():
                            axis_values = jnp.array(h5file[axis][:])
                            if axis not in active_axes_at_face:
                                raise RuntimeError
                            break
                mass_slices = equation_information.mass_slices
                velocity_slices = equation_information.velocity_slices
                energy_slices = equation_information.energy_slices
                primitives = primitives.at[mass_slices].set(unit_handler.non_dimensionalize(primitives[mass_slices], "density"))
                primitives = primitives.at[velocity_slices].set(unit_handler.non_dimensionalize(primitives[velocity_slices], "velocity"))
                primitives = primitives.at[energy_slices].set(unit_handler.non_dimensionalize(primitives[energy_slices], "pressure"))
                axis_values = unit_handler.non_dimensionalize(axis_values, "length")
                primitives_table = PrimitivesTable(primitives, axis_values, axis)
                no_primes = equation_information.no_primes
                assert_string = ("Consistency error in case setup file. "
                                 f"Primitives provided in h5file_path for {basepath:s} "
                                 "do not have the proper shape.")
                assert primitives.shape[0] == no_primes, assert_string

            if primitives_case_setup != None:
                primitives_callables_dict = {}
                for prime_state in equation_information.primes_tuple:
                    path = get_path_to_key(path_callable, prime_state)
                    prime_state_case_setup = get_setup_value(
                        primitives_case_setup, prime_state, path, (float, str),
                        is_optional=False)
                    prime_wrapper = create_wrapper_for_callable(
                        prime_state_case_setup, input_argument_units,
                        input_argument_labels, prime_state,
                        path, True, is_spatial_derivative=is_spatial_derivative,
                        unit_handler=unit_handler)
                    primitives_callables_dict[prime_state] = prime_wrapper
                primitives_callable = GetPrimitivesCallable(primitives_callables_dict)

        if field == "levelset":
            path_callable = get_path_to_key(basepath, "levelset_callable")
            levelset_case_setup = get_setup_value(
                boundary_conditions_face_case_setup, "levelset_callable",
                path_callable, (float, str), is_optional=False)
            levelset_callable = create_wrapper_for_callable(
                levelset_case_setup, input_argument_units,
                input_argument_labels, "length", path_callable,
                perform_nondim=True, unit_handler=unit_handler)

    elif "WALL" in boundary_type:
        path_callable = get_path_to_key(basepath, "wall_velocity_callable")
        wall_velocity_case_setup = get_setup_value(
            boundary_conditions_face_case_setup, "wall_velocity_callable",
            path_callable, dict, is_optional=False)
        wall_velocity_callables_dict = {}
        for velocity_xi in ["u","v","w"]:
            path = get_path_to_key(path_callable, velocity_xi)
            wall_velocity_xi_case_setup = get_setup_value(
                wall_velocity_case_setup, velocity_xi, path, (float, str),
                is_optional=False)
            velocity_wrapper = create_wrapper_for_callable(
                wall_velocity_xi_case_setup, input_argument_units,
                input_argument_labels, "velocity", path,
                perform_nondim=True, unit_handler=unit_handler)
            wall_velocity_callables_dict[velocity_xi] = velocity_wrapper
        wall_velocity_callable = VelocityCallable(**wall_velocity_callables_dict)

        if "ISOTHERMAL" in boundary_type:
            path_callable = get_path_to_key(basepath, "wall_temperature_callable")
            wall_temperature_case_setup = get_setup_value(
                boundary_conditions_face_case_setup, "wall_temperature_callable",
                path_callable, (float, str), is_optional=False)
            wall_temperature_callable = create_wrapper_for_callable(
                wall_temperature_case_setup, input_argument_units,
                input_argument_labels, "temperature", path_callable,
                perform_nondim=True, unit_handler=unit_handler)

        if "MASSTRANSFER" in boundary_type:
            path_masstransfer = get_path_to_key(basepath, "wall_mass_transfer")
            wall_mass_transfer_case_setup = get_setup_value(
                boundary_conditions_face_case_setup, "wall_mass_transfer", path, dict,
                is_optional=False)

            path_callable = get_path_to_key(path_masstransfer, "primitives_callable")
            primitives_case_setup = get_setup_value(
                wall_mass_transfer_case_setup, "primitives_callable", path, dict,
                is_optional=False)

            primitives_callables_dict = {}
            primes_tuple = [state for state in equation_information.primes_tuple if state != "p"]
            for prime_state in primes_tuple:
                path = get_path_to_key(path_callable, prime_state)
                prime_state_case_setup = get_setup_value(
                    primitives_case_setup, prime_state, path, (float, str),
                    is_optional=False)
                prime_wrapper = create_wrapper_for_callable(
                    prime_state_case_setup, input_argument_units,
                    input_argument_labels, prime_state, path, 
                    perform_nondim=True, unit_handler=unit_handler)
                primitives_callables_dict[prime_state] = prime_wrapper

            path_callable = get_path_to_key(path_masstransfer, "bounding_domain")
            bounding_domain_case_setup = get_setup_value(
                wall_mass_transfer_case_setup, "bounding_domain",
                path_callable, (float, str), is_optional=False)
            bounding_domain_callable = create_wrapper_for_callable(
                    bounding_domain_case_setup, tuple(["length"] * (dim-1)),
                    active_axes_at_face, None, path_callable,
                    perform_nondim=False, unit_handler=unit_handler)

            primitives_callable = GetPrimitivesCallable(
                primitives_callables_dict)
            wall_mass_transfer = WallMassTransferSetup(
                primitives_callable, bounding_domain_callable)

    else:
        raise NotImplementedError

    if multiple_boundary_types_at_face:
        path_callable = get_path_to_key(basepath, "bounding_domain")
        bounding_domain_case_setup = get_setup_value(
            boundary_conditions_face_case_setup, "bounding_domain",
            path_callable, (float, str), is_optional=False)
        bounding_domain_callable = create_wrapper_for_callable(
            bounding_domain_case_setup, tuple(["length"] * (dim-1)),
            active_axes_at_face, None, path_callable,
            perform_nondim=False, unit_handler=unit_handler)
    else:
        bounding_domain_callable = None

    boundary_conditions_face = BoundaryConditionsFace(
        boundary_type, bounding_domain_callable, primitives_callable,
        levelset_callable, wall_velocity_callable,
        wall_temperature_callable, wall_mass_transfer,
        primitives_table)

    return boundary_conditions_face

def interpolate_primitives_table():
    pass # TODO

def sanity_check(
        boundary_condition_setup: BoundaryConditionSetup,
        domain_setup: DomainSetup,
        numerical_setup: NumericalSetup
        ) -> None:

    active_axes = domain_setup.active_axes
    active_axes_indices = domain_setup.active_axes_indices
    inactive_axes = domain_setup.inactive_axes
    inactive_axes_indices = domain_setup.inactive_axes_indices

    axis_to_face_location = DomainInformation.axis_to_face_locations
    face_location_to_axis = DomainInformation.face_location_to_axis

    levelset_model = numerical_setup.levelset.model
    diffuse_interface_model = numerical_setup.diffuse_interface.model

    active_models = ["primitives"]
    if levelset_model:
        active_models.append("levelset")

    for model in active_models:

        # CHECK INACTIVE AXES
        boundary_conditions_field = getattr(boundary_condition_setup, model)
        for axis in inactive_axes:
            face_locations = axis_to_face_location[axis]
            for face_location in face_locations:
                boundary_condition_face: Tuple[BoundaryConditionsFace] = getattr(boundary_conditions_field, face_location)
                assert_string = (
                    "Consistency error in case setup file. "
                    f"Boundary condition at {face_location:s} must be "
                    f"INACTIVE."
                    )
                boundary_type = boundary_condition_face[0].boundary_type
                assert boundary_type == "INACTIVE", assert_string
        
        # CHECK PERIODIC CONSISTENCY
        for face_location in FACE_LOCATIONS:
            boundary_condition_face: Tuple[BoundaryConditionsFace] = getattr(boundary_conditions_field, face_location)
            boundary_type = boundary_condition_face[0].boundary_type
            if boundary_type == "PERIODIC":
                axis = face_location_to_axis[face_location]
                locations = axis_to_face_location[axis]
                flag = True
                for loc in locations:
                    boundary_condition_face: Tuple[BoundaryConditionsFace] = getattr(boundary_conditions_field, loc)
                    boundary_type = boundary_condition_face[0].boundary_type
                    if boundary_type != "PERIODIC":
                        flag = False
                        assert_string = (
                            "Consistency error in case setup file. "
                            f"Boundary condition at {face_location:s} must be "
                            "PERIODIC."
                            )
                        break
                    else:
                        pass
                assert flag, assert_string

    
    
    





