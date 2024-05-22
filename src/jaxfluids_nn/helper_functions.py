import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import optax

def initialize_optimizer(
    optimizer_name: str, 
    scheduler_name: str, 
    scheduler_params,
    params, 
    opt_state = None):
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
    if scheduler_name == "Constant":
        schedule_fn = optax.constant_schedule(value=-scheduler_params["init_value"])

    elif scheduler_name == "Polynomial":    
        schedule_fn = optax.polynomial_schedule(
            init_value=-scheduler_params["init_value"], 
            end_value=-scheduler_params["end_value"], 
            power=scheduler_params["power"],
            transition_steps=scheduler_params["transition_steps"], 
            transition_begin=scheduler_params["transition_begin"]
        )

    elif scheduler_name == "Exponential":
        schedule_fn = optax.exponential_decay(
            init_value=-scheduler_params["init_value"], 
            transition_steps=scheduler_params["transition_steps"],
            decay_rate=scheduler_params["decay_rate"], 
            transition_begin=scheduler_params["transition_begin"],
            end_value=scheduler_params["end_value"]
        )

    elif scheduler_name == "Piecewise_constant":
        schedule_fn = optax.piecewise_constant_schedule(
            init_value=-scheduler_params["init_value"], 
            boundaries_and_scales=scheduler_params["boundaries_and_scales"]
        )
    else:
        raise NotImplementedError

    if optimizer_name == "Adam":
        scale_by_opt = optax.scale_by_adam

    # CHAIN LEARNING RATE SCHEDULE AND OPTIMIZER
    opt = optax.chain(
        scale_by_opt(),
        optax.scale_by_schedule(schedule_fn)
    )

    if not opt_state:
        opt_state = opt.init(params)

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
    batch_size    = None
    for ((x,y,dt), sample_idx) in data_loader:
        if batch_size is None:
            batch_size = x.numpy().shape[0]
        sample_count  += x.numpy().shape[0] 
    return sample_count, batch_size

def plot_loss_history(
    history: Dict, 
    save_path: str) -> None:

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
        bbox_inches="tight")
    plt.close()
    plt.close()
