from typing import NamedTuple, Tuple, Callable

class PositionCallable(NamedTuple):
    x: Callable = None
    y: Callable = None
    z: Callable = None

class VelocityCallable(NamedTuple):
    u: Callable = None
    v: Callable = None
    w: Callable = None

class SingleBubbleParameters(NamedTuple):
    bubble_radius: float
    bubble_origin_x: float
    bubble_origin_y: float
    bubble_origin_z: float
    vapor_volume_fraction: float
    driving_pressure: float
    is_one_r: int
    is_barotropic: int

class InitialConditionTurbulent(NamedTuple):
    case: str = None
    random_seed: int = None
    parameters: NamedTuple = None

class HITParameters(NamedTuple):
    T_ref: float
    rho_ref: float
    energy_spectrum: str
    ma_target: float
    ic_type: str
    xi_0: int
    xi_1: int = 0
    is_velocity_spectral: bool = False
    
class ChannelParameters(NamedTuple):
    velocity_profile: str
    U_ref: float
    rho_ref: float
    T_ref: float
    noise_level: float

class DuctParameters(NamedTuple):
    velocity_profile: str
    U_ref: float
    rho_ref: float
    T_ref: float
    noise_level: float

class BoundaryLayerParameters(NamedTuple):
    T_e: float
    rho_e: float
    U_e: float
    mu_e: float
    x_position: float

class TGVParameters(NamedTuple):
    Ma_ref: float
    rho_ref: float
    V_ref: float
    L_ref: float

def GetTurbulentParametersTuple(
        case: str
        ) -> NamedTuple:
    case_dictionary = {
        "HIT": HITParameters,
        "CHANNEL": ChannelParameters,
        "DUCT": DuctParameters,
        "BOUNDARYLAYER": BoundaryLayerParameters,
        "TGV": TGVParameters
    }
    assert_string = "Consistency error in case setup file. " \
        "Initial condition turbulent case %s not implemented." % case
    assert case in case_dictionary.keys(), assert_string
    return case_dictionary[case]

class CircleParameters(NamedTuple):
    radius: float
    x_position: float
    y_position: float
    z_position: float

class SphereParameters(NamedTuple):
    radius: float
    x_position: float
    y_position: float
    z_position: float

class SquareParameters(NamedTuple):
    length: float
    x_position: float
    y_position: float
    z_position: float
    radius: float = 0.0

class RectangleParameters(NamedTuple):
    length: float
    height: float
    x_position: float
    y_position: float
    z_position: float
    radius: float = 0.0

class DiamondParameters(NamedTuple):
    chord_length: float
    maximum_thickness_position: float
    maximum_thickness: float
    angle_of_attack: float
    x_position: float
    y_position: float
    z_position: float

class EllipseParameters(NamedTuple):
    x_position: float
    y_position: float
    R_x: float
    R_y: float
    N_points: int
    deg: float

class EllipsoidParameters(NamedTuple):
    x_position: float
    y_position: float
    z_position: float
    R_x: float
    R_y: float
    R_z: float
    N_points: int
    deg_xz: float
    deg_xy: float
    deg_yz: float

def GetInitialLevelsetBlockParametersTuple(
        case: str
        ) -> NamedTuple:
    shape_dictionary = {
        "circle": CircleParameters,
        "sphere": SphereParameters,
        "square": SquareParameters,
        "rectangle": RectangleParameters,
        "diamond": DiamondParameters,
        "ellipse": EllipseParameters,
        "ellipsoid": EllipsoidParameters,
    }
    assert_string = "Consistency error in case setup file. " \
        "Initial levelset block shape %s not implemented." % case
    assert case in shape_dictionary.keys(), assert_string
    return shape_dictionary[case]

class InitialLevelsetBlock(NamedTuple):
    shape: str
    parameters: NamedTuple
    bounding_domain_callable: Callable

class InitialConditionLevelset(NamedTuple):
    blocks: Tuple[InitialLevelsetBlock]
    levelset_callable: Callable
    h5_file_path: str
    NACA_profile: str
    is_blocks: bool
    is_callable: bool
    is_NACA: bool
    is_h5_file: bool

class InitialConditionPrimitivesLevelset(NamedTuple):
    positive: NamedTuple = None
    negative: NamedTuple = None

class InitialConditionSetup(NamedTuple):
    primitives: NamedTuple
    levelset: InitialConditionLevelset
    solid_velocity: VelocityCallable
    turbulent: InitialConditionTurbulent
    is_turbulent: bool