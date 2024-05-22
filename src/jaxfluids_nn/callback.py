from typing import Dict, Tuple

from jaxfluids_nn import Trainer

class Callback:
    """
    Base Callback class for the JAX-Fluids NN Trainer.
    Users can inherit their custom callback class from this 
    base class.
    """
    def __init__(self):
        pass

    def init_callback(self, trainer: Trainer):
        pass

    def on_epoch_start(self, 
        trainer: Trainer,
        epoch: int,
        params: Dict,
        net: Dict):
        pass

    def on_epoch_end(self, 
        trainer: Trainer,
        epoch: int,
        params: Dict,
        net: Dict):
        pass

    def on_batch_start(self, 
        trainer: Trainer,
        epoch: int,
        batch_no: int,
        batch: Tuple,
        params: Dict,
        net: Dict):
        pass

    def on_batch_end(self, 
        trainer: Trainer, 
        epoch: int,
        batch_no: int,
        batch: Tuple,
        params: Dict,
        net: Dict):
        pass