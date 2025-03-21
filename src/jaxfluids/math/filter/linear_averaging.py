from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array

def linear_averaging(
        buffer: Array,
        nh: int,
        include_center_value: bool = True,
        nh_: int = None,
        volume_fraction: Array = None
        ) -> Array:
    """Linearly interpolates the cell center
    values with its neighbors within a 3x3x3 stencil.
    The argument include_center_value specifies if the
    cell center value contributes to the
    interpolation. If volume_fraction is not None,
    only cells with a volume_fraction > 0.0 are included, i.e.,
    only real cells.

    :param buffer: _description_
    :type buffer: Array
    :param nh: _description_
    :type nh: int
    :return: _description_
    :rtype: Array
    """

    if volume_fraction is not None and nh_ is None:
        raise RuntimeError

    shape = buffer.shape[-3:]
    active_axes_indices = tuple([i for i in range(3) if shape[i] > 1])
    dim = len(active_axes_indices)

    nhx,nhy,nhz = tuple(
        [jnp.s_[nh:-nh] if
        i in active_axes_indices else
        jnp.s_[:] for i in range(3)])
    slice_list = get_slices(dim, nh, active_axes_indices, nhx,
                            nhy, nhz, include_center_value)

    if nh_ is not None:
        nhx_,nhy_,nhz_ = tuple(
            [jnp.s_[nh_:-nh_] if
            i in active_axes_indices else
            jnp.s_[:] for i in range(3)])
        slice_list_ = get_slices(dim, nh_, active_axes_indices, nhx_,
                                 nhy_, nhz_, include_center_value)

    filtered_buffer = 0.0
    factor = 0
    for i in range(len(slice_list)):

        s_1 = slice_list[i]
        if volume_fraction is not None:
            s_2 = slice_list_[i]

        if volume_fraction is not None:
            filtered_buffer += buffer[s_1] * (volume_fraction[s_2] > 0.0)
            factor += volume_fraction[s_2] > 0.0

        else:
            filtered_buffer += buffer[s_1]
            factor += 1

    if volume_fraction is not None:
        factor = factor + jnp.where(factor == 0, 1, 0)

    filtered_buffer *= 1.0/factor

    return filtered_buffer


def get_slices(
        dim,
        nh,
        active_axes_indices,
        nhx,
        nhy,
        nhz,
        include_center_value: bool
        ) -> Tuple:



    if dim == 1:
        if active_axes_indices == (0,):
            slice_list = [
                jnp.s_[...,nhx       ,nhy,nhz],
                jnp.s_[...,nh-1:-nh-1,nhy,nhz],
                jnp.s_[...,nh+1:-nh+1,nhy,nhz]
            ]
        if active_axes_indices == (1,):
            slice_list = [
                jnp.s_[...,nhx       ,nhy,nhz],
                jnp.s_[...,nhx,nh-1:-nh-1,nhz],
                jnp.s_[...,nhx,nh+1:-nh+1,nhz]
            ]
        if active_axes_indices == (2,):
            slice_list = [
                jnp.s_[...,nhx       ,nhy,nhz],
                jnp.s_[...,nhx,nhy,nh-1:-nh-1],
                jnp.s_[...,nhx,nhy,nh+1:-nh+1]
            ]

    elif dim == 2:
        if active_axes_indices == (0,1):
            slice_list = [
                jnp.s_[...,nhx       ,nhy       ,nhz],
                jnp.s_[...,nh-1:-nh-1,nh-1:-nh-1,nhz],
                jnp.s_[...,nh-1:-nh-1,nh  :-nh  ,nhz],
                jnp.s_[...,nh-1:-nh-1,nh+1:-nh+1,nhz],
                jnp.s_[...,nh  :-nh  ,nh+1:-nh+1,nhz],
                jnp.s_[...,nh+1:-nh+1,nh+1:-nh+1,nhz],
                jnp.s_[...,nh+1:-nh+1,nh  :-nh  ,nhz],
                jnp.s_[...,nh+1:-nh+1,nh-1:-nh-1,nhz],
                jnp.s_[...,nh  :-nh,  nh-1:-nh-1,nhz],
                ]
        if active_axes_indices == (0,2):
            slice_list = [
                jnp.s_[...,nhx       ,nhy,       nhz],
                jnp.s_[...,nh-1:-nh-1,nhy,nh-1:-nh-1],
                jnp.s_[...,nh-1:-nh-1,nhy,nh  :-nh  ],
                jnp.s_[...,nh-1:-nh-1,nhy,nh+1:-nh+1],
                jnp.s_[...,nh  :-nh  ,nhy,nh+1:-nh+1],
                jnp.s_[...,nh+1:-nh+1,nhy,nh+1:-nh+1],
                jnp.s_[...,nh+1:-nh+1,nhy,nh  :-nh  ],
                jnp.s_[...,nh+1:-nh+1,nhy,nh-1:-nh-1],
                jnp.s_[...,nh  :-nh,  nhy,nh-1:-nh-1],
                ]
        if active_axes_indices == (1,2):
            slice_list = [
                jnp.s_[...,nhx,       nhy,       nhz],
                jnp.s_[...,nhx,nh-1:-nh-1,nh-1:-nh-1],
                jnp.s_[...,nhx,nh-1:-nh-1,nh  :-nh  ],
                jnp.s_[...,nhx,nh-1:-nh-1,nh+1:-nh+1],
                jnp.s_[...,nhx,nh  :-nh  ,nh+1:-nh+1],
                jnp.s_[...,nhx,nh+1:-nh+1,nh+1:-nh+1],
                jnp.s_[...,nhx,nh+1:-nh+1,nh  :-nh  ],
                jnp.s_[...,nhx,nh+1:-nh+1,nh-1:-nh-1],
                jnp.s_[...,nhx,nh  :-nh,  nh-1:-nh-1],
                ]
    else:

        slice_list = [
            jnp.s_[...,nh  :-nh  ,nh  :-nh  ,nh  :-nh  ],
            jnp.s_[...,nh-1:-nh-1,nh-1:-nh-1,nh  :-nh  ],
            jnp.s_[...,nh-1:-nh-1,nh  :-nh  ,nh  :-nh  ],
            jnp.s_[...,nh-1:-nh-1,nh+1:-nh+1,nh  :-nh  ],
            jnp.s_[...,nh  :-nh  ,nh+1:-nh+1,nh  :-nh  ],
            jnp.s_[...,nh+1:-nh+1,nh+1:-nh+1,nh  :-nh  ],
            jnp.s_[...,nh+1:-nh+1,nh  :-nh  ,nh  :-nh  ],
            jnp.s_[...,nh+1:-nh+1,nh-1:-nh-1,nh  :-nh  ],
            jnp.s_[...,nh  :-nh,  nh-1:-nh-1,nh  :-nh  ],

            jnp.s_[...,nh  :-nh  ,nh  :-nh  ,nh-1:-nh-1],
            jnp.s_[...,nh-1:-nh-1,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[...,nh-1:-nh-1,nh  :-nh  ,nh-1:-nh-1],
            jnp.s_[...,nh-1:-nh-1,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[...,nh  :-nh  ,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[...,nh+1:-nh+1,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[...,nh+1:-nh+1,nh  :-nh  ,nh-1:-nh-1],
            jnp.s_[...,nh+1:-nh+1,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[...,nh  :-nh,  nh-1:-nh-1,nh-1:-nh-1],

            jnp.s_[...,nh  :-nh  ,nh  :-nh  ,nh+1:-nh+1],
            jnp.s_[...,nh-1:-nh-1,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[...,nh-1:-nh-1,nh  :-nh  ,nh+1:-nh+1],
            jnp.s_[...,nh-1:-nh-1,nh+1:-nh+1,nh+1:-nh+1],
            jnp.s_[...,nh  :-nh  ,nh+1:-nh+1,nh+1:-nh+1],
            jnp.s_[...,nh+1:-nh+1,nh+1:-nh+1,nh+1:-nh+1],
            jnp.s_[...,nh+1:-nh+1,nh  :-nh  ,nh+1:-nh+1],
            jnp.s_[...,nh+1:-nh+1,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[...,nh  :-nh,  nh-1:-nh-1,nh+1:-nh+1],
            ]
    
    if not include_center_value:
        slice_list = slice_list[1:]

    return slice_list
