from typing import Dict

from jax import Array
import jax
import jax.numpy as jnp
import h5py

# NOTE helper functions that implement the use of data-driven
# aperture and volume fraction computation from \cite Buhendwa 2022

def hard_sigmoid(x: Array):
    y = 0.2*x + 0.5
    return jnp.clip(y,0.0,1.0)

def load_nn(path: str):
    params_vf = {}
    params_ap = {}
    with h5py.File(path, "r") as h5file:
        for key in h5file["apertures"]["weights"].keys():
            params_vf[key] = {}
            params_ap[key] = {}
            params_ap[key]["weights"] = jnp.array(h5file["apertures"]["weights"][key][:])
            params_ap[key]["biases"] = jnp.array(h5file["apertures"]["biases"][key][:])
            params_vf[key]["weights"] = jnp.array(h5file["volume_fraction"]["weights"][key][:])
            params_vf[key]["biases"] = jnp.array(h5file["volume_fraction"]["biases"][key][:])
    return params_vf, params_ap

def nn_eval(
        params: Dict[str, Dict[str, Array]],
        h_i: Array
        ) -> Array:
    for key in params:
        w_i, b_i = params[key]["weights"], params[key]["biases"]
        h_i = jnp.matmul(h_i,w_i) + b_i
        if key == "output":
            h_i = hard_sigmoid(h_i)
        else:
            h_i = jax.nn.relu(h_i)
    return h_i

def prepare_levelset(
        levelset: Array,
        nh: int,
        symmetries: int
        ) -> Array:

    slice_list = [
        jnp.s_[...,nh+1:-nh+1 if -nh+1 != 0 else None,nh-1:-nh-1,:],
        jnp.s_[...,nh+1:-nh+1 if -nh+1 != 0 else None,nh  :-nh  ,:],
        jnp.s_[...,nh+1:-nh+1 if -nh+1 != 0 else None,nh+1:-nh+1 if -nh+1 != 0 else None,:],

        jnp.s_[...,nh:-nh,nh-1:-nh-1    ,:],
        jnp.s_[...,nh:-nh,nh:-nh        ,:],
        jnp.s_[...,nh:-nh,nh+1:-nh+1 if -nh+1 != 0 else None    ,:],

        jnp.s_[...,nh-1:-nh-1,nh-1:-nh-1,:],
        jnp.s_[...,nh-1:-nh-1,nh  :-nh  ,:],
        jnp.s_[...,nh-1:-nh-1,nh+1:-nh+1 if -nh+1 != 0 else None,:],

        ]
    
    levelset_in = []
    for s_ in slice_list:
        levelset_in.append(levelset[s_])


    levelset_in = jnp.stack([
        jnp.stack(levelset_in[0:3],axis=-1),
        jnp.stack(levelset_in[3:6],axis=-1),
        jnp.stack(levelset_in[6:9],axis=-1),
    ],axis=-1)

    # aranged such that nn output
    # ap_x_l = apertures_nn[...,0]
    # ap_x_r = apertures_nn[...,1]
    # ap_y_l = apertures_nn[...,2]
    # ap_y_r = apertures_nn[...,3]

    if symmetries > 1:
        levelset_in_1 = levelset_in
        if symmetries >= 2:
            levelset_in_2 = levelset_in_1[...,:,::-1]
            levelset_in = jnp.stack([levelset_in_1, levelset_in_2], axis=0)
        if symmetries >= 4:
            levelset_in_3 = levelset_in_1[...,::-1,:]
            levelset_in_4 = levelset_in_1[...,::-1,::-1]
            levelset_in = jnp.stack([levelset_in_1, levelset_in_2,
                                     levelset_in_3, levelset_in_4], axis=0)
        if symmetries == 8:
            levelset_in_5 = jnp.rot90(levelset_in_1, axes=(-1,-2), k=1)
            levelset_in_6 = jnp.rot90(levelset_in_1, axes=(-1,-2), k=-1)
            levelset_in_7 = levelset_in_5[...,::-1,:]
            levelset_in_8 = levelset_in_6[...,::-1,:]
            levelset_in = jnp.stack([levelset_in_1, levelset_in_2,
                                     levelset_in_3, levelset_in_4,
                                     levelset_in_5, levelset_in_6,
                                     levelset_in_7, levelset_in_8,
                                     ], axis=0)
        levelset_in = levelset_in.reshape(symmetries,-1,9)
    else:
        levelset_in = levelset_in.reshape(-1,9)

    return levelset_in


def compute_mean_apertures(
        apertures_nn: Array,
        no_symm: int
        ) -> Array:

    ap_nn_x_l = apertures_nn[...,0]
    ap_nn_x_r = apertures_nn[...,1]
    ap_nn_y_l = apertures_nn[...,2]
    ap_nn_y_r = apertures_nn[...,3]

    if no_symm == 1:
        ap_nn_x_l_mean = ap_nn_x_l
        ap_nn_x_r_mean = ap_nn_x_r
        ap_nn_y_l_mean = ap_nn_y_l
        ap_nn_y_r_mean = ap_nn_y_r

    elif no_symm == 2:
        ap_nn_x_l_mean = 0.5*(ap_nn_x_l[0] + ap_nn_x_r[1])
        ap_nn_x_r_mean = 0.5*(ap_nn_x_r[0] + ap_nn_x_l[1])
        ap_nn_y_l_mean = 0.5*(ap_nn_y_l[0] + ap_nn_y_l[1])
        ap_nn_y_r_mean = 0.5*(ap_nn_y_r[0] + ap_nn_y_r[1])

    elif no_symm == 4:
        ap_nn_x_l_mean = 0.25*(ap_nn_x_l[0] + ap_nn_x_r[1] + ap_nn_x_l[2] + ap_nn_x_r[3])
        ap_nn_x_r_mean = 0.25*(ap_nn_x_r[0] + ap_nn_x_l[1] + ap_nn_x_r[2] + ap_nn_x_l[3])
        ap_nn_y_l_mean = 0.25*(ap_nn_y_l[0] + ap_nn_y_l[1] + ap_nn_y_r[2] + ap_nn_y_r[3])
        ap_nn_y_r_mean = 0.25*(ap_nn_y_r[0] + ap_nn_y_r[1] + ap_nn_y_l[2] + ap_nn_y_l[3])

    elif no_symm == 8:
        ap_nn_x_l_mean = 0.125*(ap_nn_x_l[0] + ap_nn_x_r[1] + ap_nn_x_l[2] + ap_nn_x_r[3] +
                                ap_nn_y_r[4] + ap_nn_y_l[5] + ap_nn_y_l[6] + ap_nn_y_r[7])
        
        ap_nn_x_r_mean = 0.125*(ap_nn_x_r[0] + ap_nn_x_l[1] + ap_nn_x_r[2] + ap_nn_x_l[3] +
                                ap_nn_y_l[4] + ap_nn_y_r[5] + ap_nn_y_r[6] + ap_nn_y_l[7])
        
        ap_nn_y_l_mean = 0.125*(ap_nn_y_l[0] + ap_nn_y_l[1] + ap_nn_y_r[2] + ap_nn_y_r[3] +
                                ap_nn_x_l[4] + ap_nn_x_r[5] + ap_nn_x_l[6] + ap_nn_x_r[7])
        
        ap_nn_y_r_mean = 0.125*(ap_nn_y_r[0] + ap_nn_y_r[1] + ap_nn_y_l[2] + ap_nn_y_l[3] +
                                ap_nn_x_r[4] + ap_nn_x_l[5] + ap_nn_x_r[6] + ap_nn_x_l[7])


    ap_nn_y = 0.5*(ap_nn_y_r_mean[1:-1,:-1] + ap_nn_y_l_mean[1:-1,1:])
    ap_nn_x = 0.5*(ap_nn_x_r_mean[:-1,1:-1] + ap_nn_x_l_mean[1:,1:-1])

    return ap_nn_x, ap_nn_y