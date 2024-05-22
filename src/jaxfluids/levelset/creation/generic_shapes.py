from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import math

from jaxfluids.data_types.case_setup.initial_conditions import CircleParameters, \
    SphereParameters, RectangleParameters, SquareParameters, DiamondParameters, \
    EllipsoidParameters, EllipseParameters

def get_circle(
        mesh_grid: Tuple[Array],
        parameters: CircleParameters,
        active_axes_indices: Tuple
        ) -> Array:
    """Creates the levelset field for a circle.

    :param radius: Radius
    :type radius: float
    :param position: Center position
    :type position: List
    :return: Levelset buffer
    :rtype: Array
    """
    radius = parameters.radius
    x_pos = parameters.x_position
    y_pos = parameters.y_position
    z_pos = parameters.z_position
    position = [x_pos, y_pos, z_pos]
    position = [position[i] for i in active_axes_indices]
    levelset = - radius + jnp.sqrt((mesh_grid[0] - position[0])**2 + (mesh_grid[1] - position[1])**2)
    return levelset

def get_square(
        mesh_grid: Tuple[Array],
        parameters: SquareParameters,
        active_axes_indices: Tuple
        ) -> Array:

    radius = parameters.radius
    length = parameters.length
    x_pos = parameters.x_position
    y_pos = parameters.y_position
    z_pos = parameters.z_position
    position = [x_pos, y_pos, z_pos]
    position = [position[i] for i in active_axes_indices]

    edge_tr = jnp.array([position[0] + length/2.0, position[1] + length/2.0])
    edge_tl = jnp.array([position[0] - length/2.0, position[1] + length/2.0])

    edge_bl = jnp.array([position[0] - length/2.0, position[1] - length/2.0])
    edge_br = jnp.array([position[0] + length/2.0, position[1] - length/2.0])

    line_1_slope    = (edge_tr[1] - edge_bl[1])/(edge_tr[0] - edge_bl[0])
    line_1_offset   = edge_tr[1] - line_1_slope * edge_tr[0]

    line_2_slope    = (edge_br[1] - edge_tl[1])/(edge_br[0] - edge_tl[0])
    line_2_offset   = edge_br[1] - line_2_slope * edge_br[0]

    levelset  =   (mesh_grid[1] - (position[1] + length/2.0)) * ((mesh_grid[1] > line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] > line_2_slope * mesh_grid[0] + line_2_offset))
    levelset += - (mesh_grid[1] - (position[1] - length/2.0)) * ((mesh_grid[1] < line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] < line_2_slope * mesh_grid[0] + line_2_offset))
    levelset +=   (mesh_grid[0] - (position[0] + length/2.0)) * ((mesh_grid[1] <= line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] >= line_2_slope * mesh_grid[0] + line_2_offset))
    levelset += - (mesh_grid[0] - (position[0] - length/2.0)) * ((mesh_grid[1] >= line_1_slope * mesh_grid[0] + line_1_offset) & (mesh_grid[1] <= line_2_slope * mesh_grid[0] + line_2_offset))

    if radius:

        levelset *= jnp.invert((mesh_grid[0] > edge_tr[0] - radius) & (mesh_grid[1] > edge_tr[1] - radius))
        levelset *= jnp.invert((mesh_grid[0] < edge_tl[0] + radius) & (mesh_grid[1] > edge_tl[1] - radius))
        levelset *= jnp.invert((mesh_grid[0] < edge_bl[0] + radius) & (mesh_grid[1] < edge_bl[1] + radius))
        levelset *= jnp.invert((mesh_grid[0] > edge_br[0] - radius) & (mesh_grid[1] < edge_br[1] + radius))

        def get_circle_parameters(
                position: jnp.DeviceArray
                ) -> CircleParameters:
            """Wrapper to compute the 
            circle parameters

            :param position: _description_
            :type position: jnp.DeviceArray
            :return: _description_
            :rtype: CircleParameters
            """
            position_list = np.array([0.0, 0.0, 0.0])
            position_list[np.array(active_axes_indices)] = position
            circle_parameters = CircleParameters(
                radius, *position_list)
            return circle_parameters

        parameters = get_circle_parameters(edge_tr + jnp.array([-1.0, -1.0]) * radius)
        levelset += get_circle(mesh_grid, parameters, active_axes_indices) * ((mesh_grid[0] > edge_tr[0] - radius) & (mesh_grid[1] > edge_tr[1] - radius))

        parameters = get_circle_parameters(edge_tl + jnp.array([ 1.0, -1.0]) * radius)
        levelset += get_circle(mesh_grid, parameters, active_axes_indices) * ((mesh_grid[0] < edge_tl[0] + radius) & (mesh_grid[1] > edge_tl[1] - radius))

        parameters = get_circle_parameters(edge_bl + jnp.array([ 1.0,  1.0]) * radius)
        levelset += get_circle(mesh_grid, parameters, active_axes_indices) * ((mesh_grid[0] < edge_bl[0] + radius) & (mesh_grid[1] < edge_bl[1] + radius))

        parameters = get_circle_parameters(edge_br + jnp.array([-1.0,  1.0]) * radius)
        levelset += get_circle(mesh_grid, parameters, active_axes_indices) * ((mesh_grid[0] > edge_br[0] - radius) & (mesh_grid[1] < edge_br[1] + radius))

    return levelset

def get_rectangle(
        mesh_grid: Tuple[Array],
        parameters: RectangleParameters,
        active_axes_indices: Tuple
        ) -> Array:
    """Creates the levelset field for a rectangle.
    If the radius argument is specified, the rectangle
    corners will be rounded using that radius. If the height
    argument is not specified, a square will be created.

    :param length: Length
    :type length: float
    :param position: Center position
    :type position: List
    :param height: Height, defaults to None
    :type height: float, optional
    :param radius: Radius of the corners, defaults to None
    :type radius: float, optional
    :return: Leveset buffer
    :rtype: Array
    """

    radius = parameters.radius
    length = parameters.length
    height = parameters.height
    x_pos = parameters.x_position
    y_pos = parameters.y_position
    z_pos = parameters.z_position
    position = [x_pos, y_pos, z_pos]
    position = [position[i] for i in active_axes_indices]

    X1, X2 = mesh_grid
    if length > height:
        pos1 = jnp.array(position) + jnp.array([length/2. - height/2., 0.0])
        pos2 = jnp.array(position) + jnp.array([-length/2. + height/2., 0.0])
        levelset = get_square(height, pos1, mesh_grid, radius) * (X1 > pos1[0])
        levelset += get_square(height, pos2, mesh_grid, radius) * (X1 < pos2[0])
        levelset += (X2 - (position[1] + height/2.)) * ((X1 >= pos2[0]) & (X1 <= pos1[0])) * (X2 > position[1])
        levelset += - (X2 - (position[1] - height/2.)) * ((X1 >= pos2[0]) & (X1 <= pos1[0])) * (X2 <= position[1])

    else:
        pos1 = jnp.array(position) + jnp.array([0.0, -length/2. + height/2.])
        pos2 = jnp.array(position) + jnp.array([0.0, length/2. - height/2.])
        levelset = get_square(length, pos1, mesh_grid, radius) * (X2 > pos1[1])
        levelset += get_square(length, pos2, mesh_grid, radius) * (X2 < pos2[1])
        levelset += (X1 - (position[0] + length/2.)) * ((X2 >= pos2[1]) & (X2 <= pos1[1])) * (X1 > position[0])
        levelset += - (X1 - (position[0] - length/2.)) * ((X2 >= pos2[1]) & (X2 <= pos1[1])) * (X1 <= position[0])

    return levelset

def get_sphere(
        mesh_grid: Tuple[Array],
        parameters: SphereParameters,
        active_axes_indices: Tuple
        ) -> Array:
    """Creates the levelset field for a sphere.

    :param radius: Radius
    :type radius: float
    :param position: Center position
    :type position: float
    :return: _description_
    :rtype: Array
    """
    radius = parameters.radius
    x_pos = parameters.x_position
    y_pos = parameters.y_position
    z_pos = parameters.z_position
    position = [x_pos, y_pos, z_pos]
    levelset = - radius + jnp.sqrt(sum([(mesh_grid[i] - position[i])**2 for i in range(3)]))
    return levelset

def get_diamond(
        mesh_grid: Tuple[Array],
        parameters: DiamondParameters,
        active_axes_indices: Tuple
        ) -> Array:
    """Creates the level-set field for a diamond-shaped airfoil.

    :param mesh_grid: _description_
    :type mesh_grid: Tuple[Array]
    :param parameters: _description_
    :type parameters: DiamondParameters
    :param active_axes_indices: _description_
    :type active_axes_indices: Tuple
    :return: _description_
    :rtype: Array
    """
    l = parameters.chord_length
    l_max = parameters.maximum_thickness_position
    t = parameters.maximum_thickness
    alpha = parameters.angle_of_attack
    x_pos = parameters.x_position
    y_pos = parameters.y_position
    z_pos = parameters.z_position

    X1, X2 = mesh_grid

    X1 = X1 - x_pos
    X2 = X2 - y_pos

    X = X1 * jnp.cos(alpha) - X2 * jnp.sin(alpha)
    Y = X1 * jnp.sin(alpha) + X2 * jnp.cos(alpha)

    phi = 0.0
    # QUADRANT 1
    x11 = l - l_max
    y11 = 0.0
    x21 = 0.0
    y21 = t / 2
    dx1 = x21 - x11
    dy1 = y21 - y11
    f11 = - dx1/dy1 * (X - x11) + y11
    f21 = - dx1/dy1 * (X - x21) + y21

    # QUADRANT 2
    x12 = -l_max
    y12 = 0.0
    x22 = 0.0
    y22 = t / 2
    dx2 = x22 - x12
    dy2 = y22 - y12
    f12 = - dx2/dy2 * (X - x12) + y12
    f22 = - dx2/dy2 * (X - x22) + y22

    # QUADRANT 3
    x13 = 0.0
    y13 = -t / 2
    x23 = -l_max
    y23 = 0.0
    dx3 = x23 - x13
    dy3 = y23 - y13
    f13 = - dx3/dy3 * (X - x13) + y13
    f23 = - dx3/dy3 * (X - x23) + y23

    # QUADRANT 4
    x14 = 0.0
    y14 = -t / 2
    x24 = l - l_max
    y24 = 0.0
    dx4 = x24 - x14
    dy4 = y24 - y14
    f14 = - dx4/dy4 * (X - x14) + y14
    f24 = - dx4/dy4 * (X - x24) + y24

    def signed_distance_fun(x0, y0, x1, y1, x2, y2,):
        dx = x2 - x1
        dy = y2 - y1
        dist = jnp.abs(dx * (y1 - y0) - (x1 - x0) * (y2 - y1)) / jnp.sqrt(dx**2 + dy**2)
        return jnp.sign((y0 - y1) - dy / dx * (x0 - x1)) * dist

    phi1 = signed_distance_fun(X, Y, x11, y11, x21, y21)
    phi2 = signed_distance_fun(X, Y, x12, y12, x22, y22)
    phi3 = -signed_distance_fun(X, Y, x13, y13, x23, y23)
    phi4 = -signed_distance_fun(X, Y, x14, y14, x24, y24)

    levelset = 0.0
    # QUADRANT 1
    levelset += jnp.where((X > 0) & (Y > 0) & (Y >= f11) & (Y <= f21), jnp.where((Y >= f12) & (Y <= f22), jnp.maximum(phi1, phi2), phi1), 0.0)
    levelset += jnp.where((X > 0) & (Y > 0) & (Y < f11), jnp.sqrt((X - x11)**2 + (Y - y11)**2), 0.0)
    levelset += jnp.where((X > 0) & (Y > 0) & (Y > f21), jnp.sqrt((X - x21)**2 + (Y - y21)**2), 0.0)
    # QUADRANT 2
    # phi += np.where((X < 0) & (Y > 0) & (Y >= f1) & (Y <= f2), signed_distance_fun(X, Y, x1, y1, x2, y2), 0.0)
    levelset += jnp.where((X < 0) & (Y > 0) & (Y >= f12) & (Y <= f22), jnp.where((Y >= f11) & (Y <= f21), jnp.maximum(phi1, phi2), phi2), 0.0)
    levelset += jnp.where((X < 0) & (Y > 0) & (Y < f12), jnp.sqrt((X - x12)**2 + (Y - y12)**2), 0.0)
    levelset += jnp.where((X < 0) & (Y > 0) & (Y > f22), jnp.sqrt((X - x22)**2 + (Y - y22)**2), 0.0)
    # QUADRANT 3
    levelset += jnp.where((X < 0) & (Y < 0) & (Y >= f13) & (Y <= f23), jnp.where((Y >= f14) & (Y <= f24), jnp.maximum(phi3, phi4), phi3), 0.0)
    levelset += jnp.where((X < 0) & (Y < 0) & (Y < f13), jnp.sqrt((X - x13)**2 + (Y - y13)**2), 0.0)
    levelset += jnp.where((X < 0) & (Y < 0) & (Y > f23), jnp.sqrt((X - x23)**2 + (Y - y23)**2), 0.0)
    # QUADRANT 4
    levelset += jnp.where((X > 0) & (Y < 0) & (Y >= f14) & (Y <= f24), jnp.where((Y >= f13) & (Y <= f23), jnp.maximum(phi3, phi4), phi4), 0.0)
    levelset += jnp.where((X > 0) & (Y < 0) & (Y < f14), jnp.sqrt((X - x14)**2 + (Y - y14)**2), 0.0)
    levelset += jnp.where((X > 0) & (Y < 0) & (Y > f24), jnp.sqrt((X - x24)**2 + (Y - y24)**2), 0.0)

    return levelset

def get_ellipse(
        mesh_grid: Tuple[Array],
        parameters: EllipseParameters,
        active_axes_indices: Tuple[int],
        *args,
        ) -> Array:

    X,Y = mesh_grid
    R_vector_grid = jnp.stack([X,Y],axis=-1).reshape(-1,2)

    N_points = parameters.N_points
    x_position = parameters.x_position
    y_position = parameters.y_position
    position = jnp.array([x_position, y_position])

    Rx = parameters.R_x
    Ry = parameters.R_y
    deg = parameters.deg
    batch_size = 100000 # NOTE hardcoded

    # POINTS
    theta = np.random.uniform(0,2*np.pi,N_points)
    x = Rx*np.cos(theta)
    y = Ry*np.sin(theta)

    points = np.stack([x,y],axis=0)

    deg = deg * np.pi/180
    rot_mat = np.array([
        [np.cos(deg), -np.sin(deg)],
        [np.sin(deg), np.cos(deg)],
        ])
    points = np.matmul(rot_mat,points)

    points = np.swapaxes(points, 0, 1)

    def compute_distance(R_vector_grid, points, distance_old):

        points += position
        distance_vector = R_vector_grid.reshape(-1,1,2) - points
        distance = jnp.linalg.norm(distance_vector, ord=2, axis=-1)
        indices = jnp.argmin(distance, axis=-1)
        distance = jnp.min(distance, axis=-1)

        mask = distance < distance_old
        distance = distance * mask + (1-mask) * distance_old

        points = points[indices,:]
        points -= position
        R_points = jnp.linalg.norm(points, axis=-1, ord=2)

        return distance, R_points

    def compute_sign(distance, R_vector_grid, R_points):
        R_vector_grid -= position
        R_grid = jnp.linalg.norm(R_vector_grid, axis=-1)
        sign = jnp.where(R_grid < R_points, -1, 1)
        signed_distance = distance * sign
        return signed_distance

    no_batches_grid = math.ceil(X.size/batch_size)
    signed_distance_buffer = np.zeros(X.size)

    no_batches_points = math.ceil(N_points/batch_size)
    for b1 in range(no_batches_grid):
        # print("BATCH %5d/%5d" % (b1, no_batches_grid))
        R_vector_grid_in = R_vector_grid[b1*batch_size:(b1+1)*batch_size]

        distance_start = np.ones(len(R_vector_grid_in))*1e10
        distance = distance_start
        for b2 in range(no_batches_points):
            points_in = points[b2*batch_size:(b2+1)*batch_size]
            distance, R_points = jax.jit(compute_distance)(R_vector_grid_in, points_in, distance)
        signed_distance = jax.jit(compute_sign)(distance, R_vector_grid_in, R_points)
        signed_distance_buffer[b1*batch_size:(b1+1)*batch_size] = signed_distance
    signed_distance = signed_distance_buffer.reshape(X.shape)

    return signed_distance




def get_ellipsoid(
        mesh_grid: Tuple[Array],
        parameters: EllipsoidParameters,
        *args,
        ) -> Array:

    X,Y,Z = mesh_grid
    R_vector_grid = jnp.stack([X,Y,Z],axis=-1).reshape(-1,3)

    N_points = parameters.N_points
    x_position = parameters.x_position
    y_position = parameters.y_position
    z_position = parameters.z_position
    position = jnp.array([x_position, y_position, z_position])
    Rx = parameters.R_x
    Ry = parameters.R_y
    Rz = parameters.R_z
    deg_xy = parameters.deg_xy
    deg_xz = parameters.deg_xz
    deg_yz = parameters.deg_yz
    batch_size = 100000 # NOTE hardcoded

    # POINTS
    theta = np.random.uniform(0,2*np.pi,N_points)
    phi = np.random.uniform(0,np.pi,N_points)

    x = Rx*np.sin(theta)*np.cos(phi)
    y = Ry*np.sin(theta)*np.sin(phi)
    z = Rz*np.cos(theta)

    points = np.stack([x,y,z],axis=0)

    # ROTATION
    deg_xy = deg_xy * np.pi/180
    deg_xz = deg_xz * np.pi/180
    deg_yz = deg_yz * np.pi/180

    rot_mat = np.array([
        [np.cos(deg_xz), 0.0, -np.sin(deg_xz)],
        [0.0, 1, 0.0],
        [np.sin(deg_xz), 0.0, np.cos(deg_xz)],
        ])
    points = np.matmul(rot_mat,points)
    
    rot_mat = np.array([
        [np.cos(deg_xy), -np.sin(deg_xy), 0.0],
        [np.sin(deg_xy), np.cos(deg_xy), 0.0],
        [0.0, 0.0, 1.0],
        ])
    points = np.matmul(rot_mat,points)

    rot_mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(deg_yz), np.sin(deg_yz)],
        [0.0, -np.sin(deg_yz), np.cos(deg_yz)],
        ])
    points = np.matmul(rot_mat,points)

    points = np.swapaxes(points, 0, 1)

    def compute_distance(R_vector_grid, points, distance_old):

        points += position
        distance_vector = R_vector_grid.reshape(-1,1,3) - points
        distance = jnp.linalg.norm(distance_vector, ord=2, axis=-1)
        indices = jnp.argmin(distance, axis=-1)
        distance = jnp.min(distance, axis=-1)

        mask = distance < distance_old
        distance = distance * mask + (1-mask) * distance_old

        points = points[indices,:]
        points -= position
        R_points = jnp.linalg.norm(points, axis=-1, ord=2)

        return distance, R_points

    def compute_sign(distance, R_vector_grid, R_points):
        R_vector_grid -= position
        R_grid = jnp.linalg.norm(R_vector_grid, axis=-1)
        sign = jnp.where(R_grid < R_points, -1, 1)
        signed_distance = distance * sign
        return signed_distance



    no_batches_grid = math.ceil(X.size/batch_size)
    signed_distance_buffer = np.zeros(X.size)

    no_batches_points = math.ceil(N_points/batch_size)
    for b1 in range(no_batches_grid):
        # print("BATCH %5d/%5d" % (b1, no_batches_grid))
        R_vector_grid_in = R_vector_grid[b1*batch_size:(b1+1)*batch_size]

        distance_start = np.ones(len(R_vector_grid_in))*1e10
        distance = distance_start
        for b2 in range(no_batches_points):
            points_in = points[b2*batch_size:(b2+1)*batch_size]
            distance, R_points = jax.jit(compute_distance)(R_vector_grid_in, points_in, distance)
        signed_distance = jax.jit(compute_sign)(distance, R_vector_grid_in, R_points)
        signed_distance_buffer[b1*batch_size:(b1+1)*batch_size] = np.array(signed_distance)
    signed_distance = signed_distance_buffer.reshape(X.shape)

    return signed_distance
