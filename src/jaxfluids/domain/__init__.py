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

TUPLE_MESH_STRETCHING_TYPES = ("CHANNEL", "BOUNDARY_LAYER", "PIECEWISE")

TUPLE_PIECEWISE_STRETCHING_TYPES = ("DECREASING", "INCREASING", "CONSTANT")
