import jax.numpy as jnp
from jax import Array

def linear_filtering(
        buffer: Array,
        nh: int,
        include_center_value: bool = True
        ) -> Array:
    """Linearly interpolates the cell center
    values with its neighbors within a 3x3x3 stencil.
    The argument include_center_value specifies if the
    cell center value contributes to the
    interpolation.

    :param buffer: _description_
    :type buffer: Array
    :param nh: _description_
    :type nh: int
    :return: _description_
    :rtype: Array
    """
    shape = buffer.shape[-3:]
    active_axes_indices = tuple([i for i in range(3) if shape[i] > 1])
    nhx,nhy,nhz = tuple(
        [jnp.s_[nh:-nh] if
        i in active_axes_indices else
        jnp.s_[:] for i in range(3)])
    dim = len(active_axes_indices)

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

    filtered_buffer = 0.0
    factor = 0
    for s_ in slice_list:
        filtered_buffer += buffer[s_]
        factor += 1
    filtered_buffer *= 1.0/factor

    return filtered_buffer


