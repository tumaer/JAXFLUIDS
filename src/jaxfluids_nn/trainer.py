import os
from datetime import datetime
from functools import partial
import json
import pickle
import time
from typing import Callable, List, Dict, Tuple

import GPUtil
import jax 
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
import numpy as np
import optax

from jaxfluids import SimulationManager
from jaxfluids.io_utils.logger import Logger
from jaxfluids_nn.helper_functions import initialize_optimizer, get_number_samples, \
    plot_loss_history

class Trainer:
    """ Trainer class which is used to train data-driven models 
    in combination with the JAX-Fluids solver.
    """

    def __init__(
            self,
            sim_manager: SimulationManager,
            checkpoint_dir: str, 
            checkpoint_freq: int = 10, 
            log_freq: int = 1, 
            callbacks: List = None,
            train_params: Dict = None, 
            opt_params: Dict = None, 
            network_params: Dict = None
        ) -> None:

        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_freq = checkpoint_freq
        self._log_freq = log_freq

        self._eval_freq = 1

        self.logger: Logger = Logger(sim_manager.numerical_setup)
        self.callbacks = callbacks

        self.train_params = train_params
        self.opt_params = opt_params
        self.network_params = network_params        

    def _initialize(
            self, 
            model_name: str, 
            optimizer_name: str, 
            scheduler_name: str, 
            scheduler_params, 
            params, 
            opt_state
        ):
        """Initializes the trainer and the optax optimizer.

        :param model_name: _description_
        :type model_name: str
        :param optimizer_name: _description_
        :type optimizer_name: str
        :param scheduler_name: _description_
        :type scheduler_name: str
        :param scheduler_params: _description_
        :type scheduler_params: _type_
        :param params: _description_
        :type params: _type_
        :param opt_state: _description_
        :type opt_state: _type_
        :return: _description_
        :rtype: _type_
        """

        self._initialize_trainer(model_name)
        opt, opt_state, schedule_fn = initialize_optimizer(
            optimizer_name, 
            scheduler_name, 
            scheduler_params, 
            params, 
            opt_state)
        
        return opt, opt_state, schedule_fn

    def _initialize_trainer(self, model_name: str) -> None:
        """Initializes the trainer object. 
        
        Creates an output folder for the model and initializes the callbacks.

        :param model_name: [description]
        :type model_name: str
        """

        # CREATE MODEL PATH
        # TODO put in helper functions?
        if not os.path.exists(self._checkpoint_dir):
            os.mkdir(self._checkpoint_dir)

        initial_length = len(model_name)
        create_directory = True
        i = 1
        while create_directory:
            if os.path.exists(os.path.join(self._checkpoint_dir, model_name)):
                if len(model_name) == initial_length:
                    model_name = model_name + f"-{i:d}"
                else:
                    model_name = model_name[:initial_length] + f"-{i:d}"
                i += 1
            else:
                self.savepath_model = os.path.join(self._checkpoint_dir, model_name)
                create_directory    = False
        os.mkdir(self.savepath_model)
        self.savepath_chkps = os.path.join(self.savepath_model, "checkpoints")
        os.mkdir(self.savepath_chkps)

        # SAVE HYPERPARAMETERS
        with open(os.path.join(self.savepath_model, "train_params.json"), "w") as f:
            json.dump(self.train_params, f, indent=4)
        with open(os.path.join(self.savepath_model, "opt_params.json"), "w") as f:
            json.dump(self.opt_params, f, indent=4)
        with open(os.path.join(self.savepath_model, "network_params.json"), "w") as f:
            json.dump(self.network_params, f, indent=4)

        resolution = self.train_params["training_resolution"]
        cg_factor = self.train_params["cg_factor"]
        traj_length = self.train_params["traj_length"]
        min_timestep_size = np.min(self.train_params["timestep_size"])
        max_timestep_size = np.max(self.train_params["timestep_size"])

        self.logger.configure_logger(log_path=self.savepath_model)
        self.logger.hline()
        self.logger.log("JAX-FLUIDS NN-TRAINER")
        self.logger.log(f"MODEL PATH: {self.savepath_model}")
        self.logger.log(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.log("\n")
        self.logger.log("TRAINING RESOLUTION = ({})".format(",".join(str(nx) for nx in resolution)))
        self.logger.log(f"COARSE GRAINING   = {cg_factor}")
        self.logger.log(f"TRAJECTORY LENGTH = {traj_length}")
        self.logger.log("TIME STEP SIZE: MIN = {:4.3e}, MAX = {:4.3e}".format(
            min_timestep_size, max_timestep_size))

        # SET STEP AND HISTORY
        self.step = 0
        self.history = {
            "loss_train": [], 
            "loss_valid": [], 
            "batch_losses": [], 
            "learning_rate": [],
            "loss_components_train": {},
            "loss_components_valid": {},
        }

        # CALLBACK INIT
        if self.callbacks is not None:
            for cb in self.callbacks:
                cb.init_callback(self)

    def init(self, rng, data_x):
        pass

    def train(
            self, 
            model_name: str, 
            dl_train, 
            dl_val, 
            epochs: int, 
            loss_fn: Callable, 
            params_dict: Dict, 
            net_dict: Dict, 
            optimizer_name: str, 
            scheduler_name: str, 
            scheduler_params, 
            opt_state = None
        ):
        """ Main training loop

        1) Initialize optimizer and scheduler
        2) Loop over epochs
            2.1) Loop over training batches
                Train_step method is called for each batch,
                which updates the NN parameters and optimizer
                state.
            2.2) Loop over validation batches
            2.3) Metrics are added to history, plots and logs
                are done.
            2.4) Callback on epoch end

        :param model_name: [description]
        :type model_name: str
        :param dl_train: [description]
        :type dl_train: [type]
        :param dl_val: [description]
        :type dl_val: [type]
        :param epochs: [description]
        :type epochs: int
        :param loss_fn: [description]
        :type loss_fn: [type]
        :param params_dict: [description]
        :type params_dict: [type]
        :param net_dict: [description]
        :type net_dict: [type]
        :param optimizer_name: [description]
        :type optimizer_name: [type]
        :param scheduler_name: [description]
        :type scheduler_name: [type]
        :param scheduler_params: [description]
        :type scheduler_params: [type]
        :param opt_state: [description], defaults to None
        :type opt_state: [type], optional
        :return: [description]
        :rtype: [type]
        """

        opt, opt_state, schedule_fn = self._initialize(
            model_name, 
            optimizer_name, 
            scheduler_name, 
            scheduler_params, 
            params_dict, 
            opt_state
        )

        total_train_samples, train_batch_size = get_number_samples(dl_train)
        total_val_samples, val_batch_size = get_number_samples(dl_val)
        number_total_batches = len(dl_train)
        number_total_steps = len(dl_train) * epochs

        self.logger.log("TRAIN SAMPLES: {}, VALIDATION SAMPLES: {}".format(total_train_samples, total_val_samples))
        self.logger.log("TRAIN BATCHES: {}, VALIDATION BATCHES: {}".format(len(dl_train), len(dl_val)))
        self.logger.log("TRAIN BATCH SIZE: {}, VALIDATION BATCH SIZE: {}".format(train_batch_size, val_batch_size))
        self.logger.log("NUMBER OF TOTAL STEPS: {}, STEPS PER EPOCH: {}".format(number_total_steps, len(dl_train)))
        self.logger.log("\n")

        # print(loss_fn)
        # print(params_dict)
        # print(net_dict)
        # exit()

        # LOOP ON EPOCHS
        self.save_checkpoint(0, params_dict, opt_state, force_save=False)
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            epoch_loss = valid_loss = 0.0
            epoch_loss_components_train = {}
            epoch_loss_components_valid = {}
            epoch_nan_samples = nan_batches = 0  

            # CALLBACK ON EPOCH START
            if self.callbacks is not None:
                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch, params_dict, net_dict)

            # LOOP ON BATCHES
            for ii, ((x, y, dt), sim_idx) in enumerate(dl_train):
                self.step += 1
                batch = (x.numpy(), y.numpy(), dt.numpy())

                # FOR DEBUGGING
                # loss_fn(params_dict, batch)

                # CALLBACK ON BATCH START
                if self.callbacks is not None:
                    for cb in self.callbacks:
                        cb.on_batch_start(self, epoch, ii, batch, params_dict, net_dict)

                new_params_dict, new_opt_state, loss, loss_info = self.train_step(
                    params_dict, opt_state, opt, batch, loss_fn)
                
                sim_idx = np.array(sim_idx)
                number_nan_samples = loss_info["number_nan_samples"]
                if number_nan_samples > 0:
                    print(f"NUMBER OF NAN SAMPLES = {number_nan_samples}")
                    # number_nan_samples = np.sum(nan_idx)
                    epoch_nan_samples += number_nan_samples
                    nan_batches += 1
                    # print("BATCH", ii, "IS NAN", sim_idx[np.argwhere(nan_idx)], "RATIO", np.mean(nan_idx))
                    pass
                else:
                    params_dict, opt_state = new_params_dict, new_opt_state

                # self.history["batch_losses"].append(loss)
                epoch_loss += loss * batch[0].shape[0] / total_train_samples
                if ii == 0:
                    for key in loss_info:
                        if key.startswith("loss_"):
                            epoch_loss_components_train[key] = {}
                            epoch_loss_components_valid[key] = {}
                            for subkey in loss_info[key]:
                                epoch_loss_components_train[key][subkey] = 0.0
                                epoch_loss_components_valid[key][subkey] = 0.0

                for key in loss_info:
                    if key.startswith("loss_"):
                        for subkey in loss_info[key]:
                            epoch_loss_components_train[key][subkey] += \
                                loss_info[key][subkey]*batch[0].shape[0]/total_train_samples

                # CALLBACK ON BATCH END
                if self.callbacks is not None:
                    for cb in self.callbacks:
                        cb.on_batch_end(self, epoch, ii, batch, params_dict, net_dict)

            print("NUM OF NAN SAMPLES: {:d}, RATIO: {:3.2f}, NUMBER OF BATCHES: {:d}, RATIO: {:3.2f}".format(
                epoch_nan_samples, epoch_nan_samples/total_train_samples * 100,
                nan_batches, nan_batches/number_total_batches * 100))

            # EVALUATE MODEL FOR VALIDATION LOSS
            for ii, ((x, y, dt), sim_idx) in enumerate(dl_val):
                batch_valid = (x.numpy(), y.numpy(), dt.numpy())
                batch_samples = batch_valid[0].shape[0]
                batch_valid_loss, loss_info = self.evaluate(params_dict, batch_valid, loss_fn)
                valid_loss += batch_valid_loss * batch_samples / total_val_samples
                for key in loss_info:
                    if key.startswith("loss_"):
                        for subkey in loss_info[key]:
                            epoch_loss_components_valid[key][subkey] += \
                                loss_info[key][subkey] * batch_valid[0].shape[0]/total_val_samples

            epoch_time = time.time() - epoch_start_time

            self.history["loss_train"].append(epoch_loss)
            self.history["loss_valid"].append(valid_loss)
            self.history["learning_rate"].append(-schedule_fn(self.step))

            if epoch == 1:
                for key in epoch_loss_components_train:
                    self.history["loss_components_train"][key] = {}
                    self.history["loss_components_valid"][key] = {}
                    for subkey in epoch_loss_components_train[key]:
                        self.history["loss_components_train"][key][subkey] = []
                        self.history["loss_components_valid"][key][subkey] = []

            for key in epoch_loss_components_train:
                for subkey in epoch_loss_components_train[key]:
                    self.history["loss_components_train"][key][subkey].append(
                        epoch_loss_components_train[key][subkey])
                    self.history["loss_components_valid"][key][subkey].append(
                        epoch_loss_components_valid[key][subkey])

            if epoch == 1:
                GPUS = GPUtil.getGPUs()
                gpu_mem = GPUS[self.train_params["gpu_id"]].memoryUsed
            else:
                gpu_mem = None

            self.printout(
                epoch, epochs, epoch_loss, valid_loss,
                epoch_time, number_total_batches, gpu_mem,
                loss_components_train=epoch_loss_components_train,
                loss_components_valid=epoch_loss_components_valid)
            self.save_checkpoint(epoch, params_dict, opt_state, force_save=False)
            plot_loss_history(self.history, self.savepath_model)

            # CALLBACK ON EPOCH END
            if self.callbacks is not None:
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, params_dict, net_dict)

        self.save_checkpoint(epoch, params_dict, opt_state, force_save=True)

        return params_dict, opt_state, loss

    @partial(jax.jit, static_argnums=(0,3,5))
    def train_step(
            self, 
            params_dict: Dict, 
            opt_state, 
            optimizer, 
            batch, 
            loss_fn: Callable
        ):
        """Training step
        """
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params_dict, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_dict, updates)
        return new_params, opt_state, loss, loss_info

    def save_checkpoint(
            self, 
            epoch: int, 
            params_dict: Dict,
            opt_state,
            force_save: bool = False
        ) -> None:
        if epoch % self._checkpoint_freq == 0 or force_save:
            path = os.path.join(self.savepath_chkps, f"checkpoint_{epoch:d}.pkl")
            if force_save:
                path = os.path.join(self.savepath_model, "checkpoint.pkl")
            with open(path, 'wb') as file:
                pickle.dump(
                    {'epoch': epoch,
                     'params_dict': params_dict, 
                     'opt_state': opt_state, 
                     'history': self.history}, file)

    def printout(
            self, 
            epoch: int,
            epochs: int,
            epoch_loss: float,
            valid_loss: float,
            time: float,
            no_batches: int,
            gpu_mem: float = None,
            loss_components_train: Dict = None,
            loss_components_valid: Dict = None
        ) -> None:

        if epoch == 1:
            self.logger.log("\n")
            self.logger.log(f"COMPILE TIME: {time:4.3e}s")
            if gpu_mem:
                self.logger.log(f"GPU MEMORY PRESSURE: {gpu_mem}MB")
            self.logger.log("\n")
        if epoch % self._log_freq == 0:
            time_per_epoch = time / self._log_freq
            time_per_batch = time_per_epoch / no_batches
            self.logger.log((
                f"Epoch {epoch:4d}/{epochs:4d} - Loss={epoch_loss:4.3e} - Val Loss={valid_loss:4.3e}"
                f" - Time/Epoch={time_per_epoch:4.3e}s - Time/Batch={time_per_batch:4.3e}s"
            ))

            if loss_components_train is not None:
                log_str = "Loss components train"
                self.logger.log(log_str)
                for key in loss_components_train.keys():
                    log_str = f"  {key}"
                    for subkey, value in loss_components_train[key].items():
                        log_str += f" - {subkey}={value:4.3e}"
                    self.logger.log(log_str)

            if loss_components_valid is not None:
                log_str = "Loss components valid"
                self.logger.log(log_str)
                for key in loss_components_valid.keys():
                    log_str = f"  {key}"
                    for subkey, value in loss_components_valid[key].items():
                        log_str += f" - {subkey}={value:4.3e}"
                    self.logger.log(log_str)

    @partial(jax.jit, static_argnums=(0,3))
    def evaluate(self, params_dict, batch, loss_fn) -> float:
        loss, loss_info = loss_fn(params_dict, batch)
        return loss, loss_info
