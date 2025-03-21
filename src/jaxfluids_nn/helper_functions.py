import os
import pickle
from typing import Any, Dict, Sequence, Tuple

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from jaxfluids_nn.data_types.optimizer import OptimizerSetup

Array = jax.Array

def initialize_optimizer(
        optimizer_setup: OptimizerSetup,
        params, 
        opt_state: optax.OptState = None
    ) -> Tuple[optax.GradientTransformation, optax.OptState, optax.Schedule]:
    """Initializes user-specified optizimer together with 
    a learning rate scheduler.

    :param optimizer_name: [description]
    :type optimizer_name: str
    :param scheduler_name: [description]
    :type scheduler_name: str
    :param scheduler_params: [description]
    :type scheduler_params: [type]
    :param params: [description]
    :type params: [type]
    :param opt_state: [description], defaults to None
    :type opt_state: [type], optional
    :raises NotImplementedError: [description]
    :return: [description]
    :rtype: [type]
    """
        
    # Initialize optimizer and scheduler
    scheduler = optimizer_setup.scheduler
    if scheduler == "Constant":
        schedule_fn = optax.constant_schedule(value=optimizer_setup.init_value)

    elif scheduler == "Polynomial":    
        schedule_fn = optax.polynomial_schedule(
            init_value=optimizer_setup.init_value, 
            end_value=optimizer_setup.end_value, 
            power=optimizer_setup.power,
            transition_steps=optimizer_setup.transition_steps, 
            transition_begin=optimizer_setup.transition_begin
        )

    elif scheduler == "Exponential":
        schedule_fn = optax.exponential_decay(
            init_value=optimizer_setup.init_value, 
            transition_steps=optimizer_setup.transition_steps,
            decay_rate=optimizer_setup.decay_rate, 
            transition_begin=optimizer_setup.transition_begin,
            end_value=optimizer_setup.end_value
        )

    elif scheduler == "Piecewise_constant":
        schedule_fn = optax.piecewise_constant_schedule(
            init_value=optimizer_setup.init_value, 
            boundaries_and_scales=optimizer_setup.boundaries_and_scales
        )
    else:
        raise NotImplementedError

    optimizer = optimizer_setup.optimizer
    if optimizer == "Adam":
        optimizer = optax.adam
    
    else:
        raise NotImplementedError

    opt = optimizer(schedule_fn)

    if not opt_state:
        opt_state = opt.init(params)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # steps = np.arange(60)
    # plt.plot(steps, schedule_fn(steps))
    # plt.yscale("log")
    # plt.savefig("test.png")
    # exit()

    return opt, opt_state, schedule_fn

def load_chkp(checkpoint_file: str) -> Dict:
    with open(checkpoint_file, "rb") as file:
        checkpoint_dict = pickle.load(file)
    return checkpoint_dict

def get_number_samples(data_loader) -> int:
    """Calculates the number of samples in a data loader.

    :param data_loader: [description]
    :type data_loader: [type]
    :return: [description]
    :rtype: int
    """
    sample_count = 0
    batch_size = None
    for ((x,y,t0,dt), sample_idx) in data_loader:
        if batch_size is None:
            batch_size = x.numpy().shape[0]
        sample_count  += x.numpy().shape[0] 
    return sample_count, batch_size

def plot_loss_history(history: Dict, save_path: str) -> None:

    fig, ax1 = plt.subplots()
    ax1.plot(history["loss_train"], label="Train")
    ax1.plot(history["loss_valid"], label="Val")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    ax2 = ax1.twinx()
    ax2.plot(history["learning_rate"], color="red")
    ax2.set_ylabel("lr")
    ax2.set_yscale("log")

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "history.png"),
        bbox_inches="tight")
    plt.close()

    nrows = len(history["loss_components_train"])
    fig, ax = plt.subplots(nrows=nrows+1, ncols=2, sharex=True, sharey="row",
        figsize=(15,10))
    ax[0,0].set_title("loss")
    ax[0,0].plot(history["loss_train"], label="loss")
    ax[0,1].plot(history["loss_valid"], label="loss")
    for ii, key in enumerate(history["loss_components_train"].keys()):
        ax[0,0].plot(history["loss_components_train"][key]["loss"], label=key)
        ax[0,1].plot(history["loss_components_valid"][key]["loss"], label=key)
        ax[ii+1,0].set_title(key)
        for subkey in history["loss_components_train"][key].keys():
            loss_train = history["loss_components_train"][key][subkey]
            loss_valid = history["loss_components_valid"][key][subkey]
            ax[ii+1,0].plot(loss_train, label=subkey)
            ax[ii+1,1].plot(loss_valid, label=subkey)
        ax[ii+1,1].legend(fontsize=10)
    ax[0,1].legend(fontsize=10)
    for axi in ax.flatten():
        axi.set_yscale("log")
        axi.set_xlabel("epoch")
        axi.set_ylabel("loss")
        axi.set_box_aspect(1)

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "history_detailed.png"),
        bbox_inches="tight", dpi=200)
    plt.close()
    plt.close()

def load_array_from_h5(load_path: str, dtype: Any = jnp.float64) -> Dict:
    """Recursively loads parameters as jax.Arrays from the 
    h5 file specified by load_path. 

    :param load_path: Path to an h5 file
    :type load_path: str
    :param dtype: Data type, defaults to jnp.float64
    :type dtype: Any, optional
    :return: Dict with keys and parameters in the form of jax.Arrays
    :rtype: Dict
    """
    my_dict = {}
    with h5py.File(load_path, "r") as h5file:
        load_array_from_grp(h5file, my_dict, dtype)
    return my_dict

def load_array_from_grp(grp: h5py.Dataset, my_dict: Dict, 
    dtype: Any = jnp.float64) -> None:
    """Loads parameters as jax.Arrays from the 
    h5 data set group. 

    :param grp: h5 data set group
    :type grp: h5py.Dataset
    :param my_dict: Dict into which parameters will be loaded
    :type my_dict: Dict
    :param dtype: Data type, defaults to jnp.float64
    :type dtype: Any, optional
    :raises NotImplementedError: _description_
    """
    for key in grp.keys():
        if isinstance(grp[key], h5py.Dataset):
            if grp[key].shape:
                my_dict[key] = jnp.array(grp[key][()], dtype=dtype)
            else:
                val = grp[key][()]
                if isinstance(val, bytes):
                    my_dict[key] = val.decode("utf-8")
                elif isinstance(val, str):
                    my_dict[key] = val
                else:
                    raise NotImplementedError
        else:
            my_dict[key] = {}
            load_array_from_grp(grp[key], my_dict[key])

def save_array_to_h5(my_dict: Dict, save_path: str, dtype: Any = None) -> None:
    """Recursively saves the jax.Arrays givein in the 
    dictionary my_dict to an h5 file specified with save_path.
    The dictionary structure is mirrored in the h5file 
    via groups and datasets.

    :param my_dict: Dictionary with jax.Arrays 
    :type my_dict: Dict
    :param save_path: Path to h5 file
    :type save_path: str
    """
    if not save_path.endswith(".h5"):
        save_path = save_path + ".h5"
    with h5py.File(save_path, "w") as h5file:
        save_array_to_grp(h5file, my_dict)

def save_array_to_grp(grp: h5py.Dataset, my_dict: Dict, dtype: Any = None) -> None:
    for key in my_dict.keys():
        if isinstance(my_dict[key], dict):
            grp1 = grp.create_group(key)
            save_array_to_grp(grp1, my_dict[key])
        elif isinstance(my_dict[key], (Array, np.ndarray)):
            dtype_ = data=my_dict[key].dtype if dtype is None else dtype
            grp.create_dataset(name=key, data=my_dict[key], dtype=dtype_)


def combine_arrays_from_dict(
    dict_sequence: Sequence[Dict],
    combined_dict: Dict = {}, 
    mode: str = "mean") -> None:

    for key in dict_sequence[0].keys():
        combined_dict[key] = {}
        if isinstance(dict_sequence[0][key], dict):
            new_dict_sequence = tuple(d[key] for d in dict_sequence)
            combine_arrays_from_dict(new_dict_sequence, combined_dict[key], mode)
        else:
            val = sum(d[key] for d in dict_sequence)
            if mode == "mean":
                val /= len(dict_sequence)
            combined_dict[key] = val


def create_empty_nested_dict(d1: Dict, d2: Dict) -> None:
    for key in d2.keys():
        if isinstance(d2[key], dict):
            d1[key] = {}
            create_empty_nested_dict(d1[key], d2[key])
        else:
            d1[key] = 0.0
