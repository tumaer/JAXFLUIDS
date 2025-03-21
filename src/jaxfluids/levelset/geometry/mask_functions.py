import jax.numpy as jnp
from jax import Array

def compute_fluid_masks(
        volume_fraction: Array,
        levelset_model: bool
        ) -> Array:
    """Computes the real fluid mask, i.e., 
    cells where the volume fraction is > 0.
    If the levelset model is FLUID-FLUID, 
    the corresponding buffer for both phases
    is returned.

    :param volume_fraction: _description_
    :type volume_fraction: Array
    :param levelset_model: _description_
    :type levelset_model: bool
    :return: _description_
    :rtype: Tuple[Array, Array]
    """

    if levelset_model == "FLUID-FLUID":
        mask_positive = jnp.where( volume_fraction > 0.0, 1, 0 )
        mask_negative = jnp.where( 1.0 - volume_fraction > 0.0, 1, 0 )
        mask_real = jnp.stack([mask_positive, mask_negative], axis=0)
    else:
        mask_positive = jnp.where( volume_fraction > 0.0, 1, 0 )
        mask_real = mask_positive

    return mask_real

def compute_narrowband_mask(
        levelset: Array,
        dx: float,
        narrowband_width: int
        ) -> Array:
    normalized_levelset = jnp.abs(levelset)/dx
    mask_narrowband = jnp.where(normalized_levelset <= narrowband_width, 1, 0)
    return mask_narrowband

def compute_cut_cell_mask_vf_based(volume_fraction: Array) -> Array:
    cut_cell_mask = (volume_fraction > 0.0) & (volume_fraction < 1.0)
    return cut_cell_mask

def compute_cut_cell_mask_value_based(levelset: Array, dx: float) -> Array:
    resolution = levelset.shape[-3:]
    dim = sum([cells > 1 for cells in resolution])
    if dim == 1:
        cut_cell_mask = jnp.abs(levelset)/dx <= 1/2
    elif dim == 2:
        cut_cell_mask = jnp.abs(levelset)/dx <= jnp.sqrt(2)/2
    elif dim == 3:
        cut_cell_mask = jnp.abs(levelset)/dx <= jnp.sqrt(3)/2
    return cut_cell_mask

def compute_cut_cell_mask_sign_change_based(
        levelset: Array,
        nh_offset: int
        ) -> Array:
    """Computes the cut cell mask, i.e., cells
    with different levelset signs compared to
    neighboring cells within the 3x3x3 stencil.

    :param levelset: _description_
    :type levelset: Array
    :param nh_offset: _description_
    :type nh_offset: int
    :return: _description_
    :rtype: Array
    """

    shape = levelset.shape
    active_axes_indices = [i for i in range(3) if shape[i] > 1]
    nhx, nhy, nhz = tuple(
        [jnp.s_[nh_offset:-nh_offset] if
        i in active_axes_indices else
        jnp.s_[:] for i in range(3)]
        )
    index_pairs = [(0,1), (0,2), (1,2)]
    active_planes = [] 
    for i, pair in enumerate(index_pairs):
        if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
            active_planes.append(i)
    dim = len(active_axes_indices)
    nh = nh_offset
    
    s_0 = (nhx,nhy,nhz)

    # II
    s_ii_list = [
        [
            jnp.s_[nh-1:-nh-1,nhy,nhz],
            jnp.s_[nh+1:-nh+1,nhy,nhz],
        ],
        [
            jnp.s_[nhx,nh-1:-nh-1,nhz],
            jnp.s_[nhx,nh+1:-nh+1,nhz],
        ],
        [
            jnp.s_[nhx,nhy,nh-1:-nh-1],
            jnp.s_[nhx,nhy,nh+1:-nh+1]
        ]
    ]
    
    # IJ
    s_ij_list = [
        [  
            jnp.s_[nh-1:-nh-1,nh-1:-nh-1,nhz],
            jnp.s_[nh-1:-nh-1,nh+1:-nh+1,nhz],
            jnp.s_[nh+1:-nh+1,nh-1:-nh-1,nhz],
            jnp.s_[nh+1:-nh+1,nh+1:-nh+1,nhz],
        ],
        [
            jnp.s_[nh-1:-nh-1,nhy,nh-1:-nh-1],
            jnp.s_[nh-1:-nh-1,nhy,nh+1:-nh+1],
            jnp.s_[nh+1:-nh+1,nhy,nh-1:-nh-1],
            jnp.s_[nh+1:-nh+1,nhy,nh+1:-nh+1],
        ],
        [
            jnp.s_[nhx,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[nhx,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[nhx,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[nhx,nh+1:-nh+1,nh+1:-nh+1],
        ]
    ]

    # IJK
    s_ijk_list = [  
            jnp.s_[nh-1:-nh-1,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[nh-1:-nh-1,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[nh-1:-nh-1,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[nh-1:-nh-1,nh+1:-nh+1,nh+1:-nh+1],
            jnp.s_[nh+1:-nh+1,nh-1:-nh-1,nh-1:-nh-1],
            jnp.s_[nh+1:-nh+1,nh-1:-nh-1,nh+1:-nh+1],
            jnp.s_[nh+1:-nh+1,nh+1:-nh+1,nh-1:-nh-1],
            jnp.s_[nh+1:-nh+1,nh+1:-nh+1,nh+1:-nh+1],
        ]
    
    
    mask_cut_cells = jnp.zeros_like(levelset[s_0], dtype=jnp.uint32)
    for axis in active_axes_indices:
        for s_ii in s_ii_list[axis]:
            mask_cut_cells_temp = jnp.where(levelset[s_0]*levelset[s_ii] <= 0, 1, 0)
            mask_cut_cells = jnp.maximum(mask_cut_cells, mask_cut_cells_temp)
    
    if dim > 1:
        for i in active_planes:
            for s_ij in s_ij_list[i]:
                mask_cut_cells_temp = jnp.where(levelset[s_0]*levelset[s_ij] <= 0, 1, 0)
                mask_cut_cells = jnp.maximum(mask_cut_cells_temp, mask_cut_cells)

    if dim == 3:
        for s_ijk in s_ijk_list:
            mask_cut_cells_temp = jnp.where(levelset[s_0]*levelset[s_ijk] <= 0, 1, 0)
            mask_cut_cells = jnp.maximum(mask_cut_cells_temp, mask_cut_cells)

    return mask_cut_cells