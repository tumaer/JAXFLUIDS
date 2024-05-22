from typing import Dict, List, Tuple

import jax
import numpy as np

from jaxfluids.data_types.case_setup.domain import *
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.domain import AXES, TUPLE_MESH_STRETCHING_TYPES, TUPLE_PIECEWISE_STRETCHING_TYPES
from jaxfluids.input.case_setup import get_setup_value, loop_fields, get_path_to_key
from jaxfluids.unit_handler import UnitHandler

def read_domain_setup(
        case_setup_dict: Dict,
        numerical_setup: NumericalSetup,
        unit_handler: UnitHandler
        ) -> DomainSetup:
    """Reads the case setup and initializes the domain and
    domain-related properties."""

    basepath = "domain"
    domain_case_setup = get_setup_value(
        case_setup_dict, "domain", basepath, dict,
        is_optional=False)

    # AXIS SETUPS
    axes_dict: Dict[str, AxisSetup] = {}
    for axis in AXES:
        path_axes = get_path_to_key(basepath, axis)
        axis_case_setup = get_setup_value(domain_case_setup, axis, path_axes, dict,
            is_optional=False)

        path_range = get_path_to_key(path_axes, "range")
        axis_range = get_setup_value(axis_case_setup, "range", path_range, list,
                                     is_optional=False)
        axis_range_nondim = (
            unit_handler.non_dimensionalize(axis_range[0], "length"),
            unit_handler.non_dimensionalize(axis_range[1], "length"))

        path_cells = get_path_to_key(path_axes, "cells")
        cells = get_setup_value(axis_case_setup, "cells", path_cells, int,
                                is_optional=False)

        stretching_setup = read_mesh_stretching(
            axis_case_setup, unit_handler, axis)

        axis_dict = AxisSetup(
            cells,
            axis_range_nondim,
            stretching_setup)

        axes_dict[axis] = axis_dict

    domain_decomposition = read_decomposition(domain_case_setup)

    active_axes = []
    inactive_axes = []
    active_axes_indices = []
    inactive_axes_indices = []
    for axis_index, (axis, axis_setup) in enumerate(axes_dict.items()):
        if axis_setup.cells > 1:
            active_axes.append( axis )
            active_axes_indices.append( axis_index )
        if axis_setup.cells == 1:
            inactive_axes.append( axis )
            inactive_axes_indices.append( axis_index )
    active_axes = tuple(active_axes)
    inactive_axes = tuple(inactive_axes)
    active_axes_indices = tuple(active_axes_indices)
    inactive_axes_indices = tuple(inactive_axes_indices)
    
    domain_setup = DomainSetup(
        **axes_dict,
        decomposition=domain_decomposition,
        active_axes=active_axes,
        inactive_axes=inactive_axes,
        active_axes_indices=active_axes_indices,
        inactive_axes_indices=inactive_axes_indices,
        dim=len(active_axes))
    
    sanity_check(domain_setup, numerical_setup)
    
    return domain_setup


def read_mesh_stretching(
        axis_case_setup: Dict,
        unit_handler: UnitHandler,
        axis: str
        ) -> MeshStretchingSetup:
    
    path_axes = get_path_to_key("domain", axis)

    path_stretching = get_path_to_key(path_axes, "stretching")
    stretching_case_setup = get_setup_value(
        axis_case_setup, "stretching", path_stretching,
        dict, is_optional=True, default_value={})
    
    path_stretching = get_path_to_key(path_stretching, "type")
    type_str = get_setup_value(
        stretching_case_setup, "type", path_stretching, str,
        is_optional=True, default_value=False,
        possible_string_values=TUPLE_MESH_STRETCHING_TYPES)

    is_optional = True if type_str == False else False
    path_params = get_path_to_key(path_stretching, "parameters")
    parameters_case_setup = get_setup_value(
        stretching_case_setup, "parameters", path_stretching, (list, dict),
        is_optional=is_optional, default_value=[])

    tanh_value = None
    piecewise_parameters = None
    ratio_fine_region = None
    cells_fine = None
    if type_str == "PIECEWISE":
        def read_piecewise_parameters(parameters_case_setup: Dict) -> PiecewiseStretchingParameters:
            """Wrapper reading the parameters for
            the piecewise stretching.

            :param parameters: _description_
            :type parameters: Dict
            :return: _description_
            :rtype: PiecewiseStretchingParameters
            """
            basepath = get_path_to_key(path_stretching, "parameters")
            path = get_path_to_key(basepath, "type")
            type_str = get_setup_value(
                parameters_case_setup, "type", path, str, is_optional=False,
                possible_string_values=TUPLE_PIECEWISE_STRETCHING_TYPES)
            path = get_path_to_key(basepath, "lower_bound")
            lower_bound = get_setup_value(
                parameters_case_setup, "lower_bound", path, float, is_optional=False)
            lower_bound = unit_handler.non_dimensionalize(lower_bound, "length")
            path = get_path_to_key(basepath, "upper_bound")
            upper_bound = get_setup_value(
                parameters_case_setup, "upper_bound", path, float, is_optional=False)
            upper_bound = unit_handler.non_dimensionalize(upper_bound, "length")
            path = get_path_to_key(basepath, "cells")
            cells = get_setup_value(
                parameters_case_setup, "cells", path, int, is_optional=False,
                numerical_value_condition=(">", 0))
            piecewise_parameters = PiecewiseStretchingParameters(
                type_str, cells, upper_bound, lower_bound)
            return piecewise_parameters

        piecewise_parameters_list = []
        for parameters in parameters_case_setup:
            piecewise_parameters = read_piecewise_parameters(parameters)
            piecewise_parameters_list.append(piecewise_parameters)
        piecewise_parameters = tuple(piecewise_parameters_list)

    elif type_str in ["CHANNEL", "BOUNDARY_LAYER"]:
        path_stretching = get_path_to_key(path_params, "tanh_value")
        tanh_value = get_setup_value(
            parameters_case_setup, "tanh_value", path_stretching, float,
            is_optional=False, numerical_value_condition=(">=", 1.0))

    stretching_setup = MeshStretchingSetup(
        type_str, tanh_value, ratio_fine_region,
        cells_fine, piecewise_parameters)
        
    return stretching_setup

def read_decomposition(domain_case_setup: Dict) -> DomainDecompositionSetup:
    
    path = get_path_to_key("domain", "decomposition")
    decomposition_case_setup = get_setup_value(
        domain_case_setup, "decomposition", path, dict,
        is_optional=True, default_value={})
    
    domain_decomposition = loop_fields(
        DomainDecompositionSetup, decomposition_case_setup,
        path)

    return domain_decomposition

def sanity_check(domain_setup: DomainSetup, numerical_setup: NumericalSetup) -> None:

    nx = domain_setup.x.cells
    ny = domain_setup.y.cells
    nz = domain_setup.z.cells

    number_of_cells = (nx,ny,nz)

    decomposition_setup = domain_setup.decomposition
    split_x = decomposition_setup.split_x
    split_y = decomposition_setup.split_y
    split_z = decomposition_setup.split_z

    split_factors = (split_x, split_y, split_z)

    device_number_of_cells = (
        nx//split_x, ny//split_y, nz//split_z
    )

    # CHECK SPLIT FACTORS - ACTIVE AXES
    inactive_axes_indices = domain_setup.inactive_axes_indices
    inactive_axes = domain_setup.inactive_axes
    for axis, axis_index in zip(inactive_axes, inactive_axes_indices):
        split_xi = split_factors[axis_index]
        assert_string = (
            "Consistency error in case setup file. "
            f"Split factor {axis:s} must be 1 since the "
            "axis is INACTIVE.")
        assert split_xi == 1, assert_string
    
    # CHECK STRETCHING - ACTIVE AXES
    for axis, axis_index in zip(inactive_axes, inactive_axes_indices):
        axis_setup: AxisSetup = getattr(domain_setup, axis)
        stretching_type = axis_setup.stretching.type
        assert_string = (
            "Consistency error in case setup file. "
            f"Stretching of {axis:s} must be null since the "
            "axis is INACTIVE.")
        assert stretching_type == False, assert_string

    # CHECK EVEN SPLIT
    active_axes_indices = domain_setup.active_axes_indices
    active_axes = domain_setup.active_axes
    for axis, axis_index in zip(active_axes, active_axes_indices):
        split_xi = split_factors[axis_index]
        nxi = number_of_cells[axis_index]
        assert_string = (
            "Consistency error in case setup file. "
            "Split factor and number of cells in "
            f"{axis:s} direction results in unequal "
            "division.")
        assert nxi%split_xi == 0, assert_string

    # CHECK DOMAIN BOUNDS
    for axis in AXES:
        axis_setup: AxisSetup = getattr(domain_setup, axis)
        bounds = axis_setup.range
        assert_string = (
            "Consistency error in case setup file. "
            "Lower domain bound greater than upper "
            f"domain bound for {axis:s}.")
        assert bounds[1] > bounds[0], assert_string

    # CHECK PIECEWISE STRETCHING, BOUNDS, TYPES AND CELLS
    for axis in AXES:
        axis_setup: AxisSetup = getattr(domain_setup, axis)
        domain_size = axis_setup.range
        cells_axis = axis_setup.cells
        stretching_setup = axis_setup.stretching
        stretching_type = stretching_setup.type

        assert_string_cells = (
            "Consistency error in case setup file. "
            f"Total cells of piecewise stretched axis {axis:s} "
            "does not match cells in domain.")
        
        assert_string_bounds = (
            "Consistency error in case setup file. "
            f"Upper and/or lower bounds of piecewise stretched axis {axis:s} "
            "do not match with domain bounds and/or are not consistent "
            "within the stretched axis.")
        
        assert_string_increasing = (
            "Consistency error in case setup file. "
            f"Type of the piecewise INCREASING stretched axis {axis:s} requires "
            "CONSTANT region to its left.")

        assert_string_decreasing = (
            "Consistency error in case setup file. "
            f"Type of the piecewise DECREASING stretched axis {axis:s} requires "
            "CONSTANT region to its right.")
    
        if stretching_type == "PIECEWISE":
            parameters_tuple = stretching_setup.piecewise_parameters
            cells_count = 0
            for i, parameters in enumerate(parameters_tuple):
                cells_count += parameters.cells
                upper_bound = parameters.upper_bound
                lower_bound = parameters.lower_bound
                type_piecewise = parameters.type
                if i == 0:
                    assert lower_bound == domain_size[0], assert_string_bounds
                    assert upper_bound == parameters_tuple[1].lower_bound, assert_string_bounds
                    if type_piecewise == "DECREASING":
                        assert parameters_tuple[1].type == "CONSTANT", assert_string_decreasing
                    if type_piecewise == "INCREASING":
                        assert False, assert_string_increasing
                elif i == len(parameters_tuple) - 1:
                    assert lower_bound == parameters_tuple[i-1].upper_bound, assert_string_bounds
                    assert upper_bound == domain_size[-1], assert_string_bounds
                    if type_piecewise == "INCREASING":
                        assert parameters_tuple[i-1].type == "CONSTANT", assert_string_increasing
                    if type_piecewise == "DECREASING":
                        assert False, assert_string_decreasing
                else:
                    assert upper_bound == parameters_tuple[i+1].lower_bound, assert_string_bounds
                    assert lower_bound == parameters_tuple[i-1].upper_bound, assert_string_bounds
                    if type_piecewise == "INCREASING":
                        assert parameters_tuple[i-1].type == "CONSTANT", assert_string_increasing
                    if type_piecewise == "DECREASING":
                        assert parameters_tuple[i+1].type == "CONSTANT", assert_string_decreasing

            assert cells_count == cells_axis, assert_string_cells

    # PARALLEL
    no_subdomains = np.prod(np.array(split_factors))
    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    host_count = global_device_count // local_device_count
    assert_string = ("Consistency error in case setup file. "
                     f"Mismatch between number of subdomains {no_subdomains:d} "
                     f"and XLA device count {global_device_count:d}.")
    assert no_subdomains <= global_device_count, assert_string
    assert_string = ("Consistency error in case setup file. "
                     f"Mismatch between number of subdomains {no_subdomains:d} "
                     f"and host/process count {host_count:d}.")
    assert no_subdomains >= host_count, assert_string

    nh_conservatives = numerical_setup.conservatives.halo_cells
    for axis_index in active_axes_indices:
        axis = AXES[axis_index]
        nxi = device_number_of_cells[axis_index]
        assert_string = ("Consistency error in case setup file. "
                        f"There are more halo cells than cells in {axis:s} direction.")
        assert nxi >= nh_conservatives, assert_string
