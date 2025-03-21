AXES = ("x", "y", "z")

AXES_INDICES = (0,1,2)

FACE_LOCATIONS = ("east", "west",
                  "north", "south",
                  "top", "bottom")

EDGE_LOCATIONS = (
    "west_south", "west_north",
    "east_north", "east_south",
    "south_bottom", "north_bottom",
    "south_top", "north_top",
    "east_bottom", "west_bottom",
    "east_top", "west_top")

VERTEX_LOCATIONS = (
    "west_south_bottom", "west_south_top",
    "west_north_bottom", "west_north_top",
    "east_north_top", "east_south_top",
    "east_north_bottom", "east_south_bottom")

TUPLE_MESH_STRETCHING_TYPES = ("CHANNEL", "BOUNDARY_LAYER", "PIECEWISE", "BUBBLE_1", "BUBBLE_2")

TUPLE_PIECEWISE_STRETCHING_TYPES = ("DECREASING", "INCREASING", "CONSTANT")

EDGE_LOCATIONS_TO_RUNNING_AXIS = {
    "west_south"    : "z",
    "west_north"    : "z",
    "east_north"    : "z",
    "east_south"    : "z",
    "south_bottom"  : "x",
    "north_bottom"  : "x",
    "south_top"     : "x",
    "north_top"     : "x",
    "east_bottom"   : "y",
    "west_bottom"   : "y",
    "east_top"      : "y",
    "west_top"      : "y",
}

FACE_LOCATIONS_TO_EDGE_LOCATIONS = {
    "west": {
        "y": {"upwind": "west_north"  , "downwind": "west_south"}, 
        "z": {"upwind": "west_top"    , "downwind": "west_bottom"}
    },
    "east": {
        "y": {"upwind": "east_north"  , "downwind": "east_south"},
        "z": {"upwind": "east_top"    , "downwind": "east_bottom"}
    },
    "south": {
        "x": {"upwind": "east_south"  , "downwind": "west_south"},
        "z": {"upwind": "south_top"   , "downwind": "south_bottom"}
    },
    "north": {
        "x": {"upwind": "east_north"  , "downwind": "west_north"},
        "z": {"upwind": "north_top"   , "downwind": "north_bottom"}
    },
    "bottom": {
        "x": {"upwind": "east_bottom" , "downwind": "west_bottom"},
        "y": {"upwind": "north_bottom", "downwind": "south_bottom"}
    },
    "top": {
        "x": {"upwind": "east_top"    , "downwind": "west_top"},
        "y": {"upwind": "north_top"   , "downwind": "south_top"}
    }
}

from jaxfluids.domain.helper_functions import (reassemble_buffer, reassemble_buffer_np,
                                               reassemble_cell_centers, reassemble_cell_faces,
                                               reassemble_cell_sizes, split_and_shard_buffer,
                                               split_and_shard_buffer_np)