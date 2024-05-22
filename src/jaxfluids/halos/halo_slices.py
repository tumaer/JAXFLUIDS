from typing import Tuple, List, Dict

import jax.numpy as jnp
from jax import Array

class HaloSlices:
    """The HaloSlices class provides
    index tuples to slice the halo cells
    of a block of the computational mesh.
    We distinguish between
    1) Face halos.
    2) Edge halos.
    3) Vertex halos.
    """
    def __init__(
            self,
            nh_conservatives: int,
            nh_geometry: int,
            active_axes_indices: Tuple
            ) -> None:

        nhx, nhy, nhz = tuple(
            [jnp.s_[nh_conservatives:-nh_conservatives] if
            i in active_axes_indices else
            jnp.s_[:] for i in range(3)]
            )
        self.face_slices_conservatives = get_face_slices(
            nh_conservatives, nhx, nhy, nhz)
        self.edge_slices_conservatives = get_edge_slices(
            nh_conservatives, nhx, nhy, nhz)
        self.vertex_slices_conservatives = get_vertex_slices(
            nh_conservatives)

        if nh_geometry != None:
            nhx, nhy, nhz = tuple(
                [jnp.s_[nh_geometry:-nh_geometry] if
                i in active_axes_indices else
                jnp.s_[:] for i in range(3)]
                )
            self.face_slices_geometry = get_face_slices(
                nh_geometry, nhx, nhy, nhz)
            self.edge_slices_geometry = get_edge_slices(
                nh_geometry, nhx, nhy, nhz)
            self.vertex_slices_geometry = get_vertex_slices(
                nh_geometry)

def get_face_slices(
        nh: int,
        nhx: Tuple,
        nhy: Tuple,
        nhz: Tuple
        ) -> Dict:
    """Generates the slices for the halo cells
    that are located at the edges of a block.

    :param nh: _description_
    :type nh: int
    :param nhx: _description_
    :type nhx: Tuple
    :param nhy: _description_
    :type nhy: Tuple
    :param nhz: _description_
    :type nhz: Tuple
    :return: _description_
    :rtype: Dict
    """

    face_slices = {
        "east"  : jnp.s_[..., -nh:, nhy, nhz],
        "west"  : jnp.s_[..., :nh, nhy, nhz],
        "north" : jnp.s_[..., nhx, -nh:, nhz],
        "south" : jnp.s_[..., nhx, :nh, nhz],
        "top"   : jnp.s_[..., nhx, nhy, -nh:],
        "bottom": jnp.s_[..., nhx, nhy, :nh],
    }
    return face_slices

def get_edge_slices(
        nh: int,
        nhx: Tuple,
        nhy: Tuple,
        nhz: Tuple
        ) -> Dict:
    """Generates slices for the halo cells that 
    are located at the edges of a block.

    :param nh: _description_
    :type nh: int
    :param nhx: _description_
    :type nhx: Tuple
    :param nhy: _description_
    :type nhy: Tuple
    :param nhz: _description_
    :type nhz: Tuple
    :return: _description_
    :rtype: Dict
    """

    edge_slices = {
        
        "west_south"    : jnp.s_[..., :nh, :nh, nhz],
        "west_south_10"  : jnp.s_[..., nh:2*nh, :nh, nhz],
        "west_south_01"  : jnp.s_[..., :nh, nh:2*nh, nhz],

        "west_north"    : jnp.s_[..., :nh, -nh:, nhz],
        "west_north_10"  : jnp.s_[..., nh:2*nh, -nh:, nhz],
        "west_north_01"  : jnp.s_[..., :nh, -2*nh:-nh, nhz],

        "east_south"    : jnp.s_[..., -nh:, :nh, nhz], 
        "east_south_10"  : jnp.s_[..., -2*nh:-nh, :nh, nhz],
        "east_south_01"  : jnp.s_[..., -nh:, nh:2*nh, nhz],

        "east_north"    : jnp.s_[..., -nh:, -nh:, nhz],
        "east_north_10"  : jnp.s_[..., -2*nh:-nh, -nh:, nhz],
        "east_north_01"  : jnp.s_[..., -nh:, -2*nh:-nh, nhz],

        "south_bottom"      : jnp.s_[..., nhx, :nh, :nh],  
        "south_bottom_10"   : jnp.s_[..., nhx, nh:2*nh, :nh],
        "south_bottom_01"   : jnp.s_[..., nhx, :nh, nh:2*nh],

        "north_bottom"      : jnp.s_[..., nhx, -nh:, :nh],
        "north_bottom_10"   : jnp.s_[..., nhx, -2*nh:-nh, :nh],
        "north_bottom_01"   : jnp.s_[..., nhx, -nh:, nh:2*nh],

        "south_top"     : jnp.s_[..., nhx, :nh, -nh:],
        "south_top_10"  : jnp.s_[..., nhx, nh:2*nh, -nh:],
        "south_top_01"  : jnp.s_[..., nhx, :nh, -2*nh:-nh],

        "north_top"     : jnp.s_[..., nhx, -nh:, -nh:],
        "north_top_10"  : jnp.s_[..., nhx, -2*nh:-nh, -nh:],
        "north_top_01"  : jnp.s_[..., nhx, -nh:, -2*nh:-nh],

        "west_bottom"   : jnp.s_[..., :nh, nhy, :nh],
        "west_bottom_10" : jnp.s_[..., nh:2*nh, nhy, :nh],
        "west_bottom_01" : jnp.s_[..., :nh, nhy, nh:2*nh],

        "east_bottom"   : jnp.s_[..., -nh:, nhy, :nh],
        "east_bottom_10" : jnp.s_[..., -2*nh:-nh, nhy, :nh],
        "east_bottom_01" : jnp.s_[..., -nh:, nhy, nh:2*nh],

        "east_top"      : jnp.s_[..., -nh:, nhy, -nh:],
        "east_top_10"    : jnp.s_[...,  -2*nh:-nh, nhy, -nh:],
        "east_top_01"    : jnp.s_[...,  -nh:, nhy, -2*nh:-nh], 

        "west_top"      : jnp.s_[..., :nh, nhy, -nh:],
        "west_top_01"    : jnp.s_[..., :nh, nhy, -2*nh:-nh],
        "west_top_10"    : jnp.s_[..., nh:2*nh, nhy, -nh:],
    }

    return edge_slices



def get_vertex_slices(
        nh: int,
        ) -> Dict:
    """Generates slices for the halo cells that 
    are located at the vertices of a block.

    :param nh: _description_
    :type nh: int
    :param nhx: _description_
    :type nhx: Tuple
    :param nhy: _description_
    :type nhy: Tuple
    :param nhz: _description_
    :type nhz: Tuple
    :return: _description_
    :rtype: Dict
    """

    vertex_slices = {
    
        "west_south_bottom"     : jnp.s_[..., :nh, :nh, :nh],
        "west_south_bottom_100" : jnp.s_[..., nh:2*nh, :nh, :nh],
        "west_south_bottom_010" : jnp.s_[..., :nh, nh:2*nh, :nh],
        "west_south_bottom_001" : jnp.s_[..., :nh, :nh, nh:2*nh],

        "west_south_top"        : jnp.s_[..., :nh, :nh, -nh:],
        "west_south_top_100"    : jnp.s_[..., nh:2*nh, :nh, -nh:],
        "west_south_top_010"    : jnp.s_[..., :nh, nh:2*nh, -nh:],
        "west_south_top_001"    : jnp.s_[..., :nh, :nh, -2*nh:-nh],

        "west_north_bottom"     : jnp.s_[..., :nh, -nh:, :nh],
        "west_north_bottom_100" : jnp.s_[..., nh:2*nh, -nh:, :nh],
        "west_north_bottom_010" : jnp.s_[..., :nh, -2*nh:-nh, :nh],
        "west_north_bottom_001" : jnp.s_[..., :nh, -nh:, nh:2*nh],

        "west_north_top"        : jnp.s_[..., :nh, -nh:, -nh:],
        "west_north_top_100"    : jnp.s_[..., nh:2*nh, -nh:, -nh:],
        "west_north_top_010"    : jnp.s_[..., :nh, -2*nh:-nh, -nh:],
        "west_north_top_001"    : jnp.s_[..., :nh, -nh:, -2*nh:-nh],

        "east_north_top"        : jnp.s_[..., -nh:, -nh:, -nh:],
        "east_north_top_100"    : jnp.s_[..., -2*nh:-nh, -nh:, -nh:],
        "east_north_top_010"    : jnp.s_[..., -nh:, -2*nh:-nh, -nh:],
        "east_north_top_001"    : jnp.s_[..., -nh:, -nh:, -2*nh:-nh],

        "east_south_top"        : jnp.s_[..., -nh:, :nh, -nh:],
        "east_south_top_100"    : jnp.s_[..., -2*nh:-nh, :nh, -nh:],
        "east_south_top_010"    : jnp.s_[..., -nh:, nh:2*nh, -nh:],
        "east_south_top_001"    : jnp.s_[..., -nh:, :nh, -2*nh:-nh],

        "east_north_bottom"     : jnp.s_[..., -nh:, -nh:, :nh],
        "east_north_bottom_100" : jnp.s_[..., -2*nh:-nh, -nh:, :nh],
        "east_north_bottom_010" : jnp.s_[..., -nh:, -2*nh:-nh, :nh], 
        "east_north_bottom_001" : jnp.s_[..., -nh:, -nh:, nh:2*nh],

        "east_south_bottom"    : jnp.s_[..., -nh:, :nh, :nh],
        "east_south_bottom_100" : jnp.s_[..., -2*nh:-nh, :nh, :nh],
        "east_south_bottom_010" : jnp.s_[..., -nh:, nh:2*nh, :nh], 
        "east_south_bottom_001" : jnp.s_[..., -nh:, :nh, nh:2*nh],
    }

    return vertex_slices



