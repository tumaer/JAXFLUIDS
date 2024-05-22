from abc import ABC, abstractmethod
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from jax import Array

class TimeIntegrator(ABC):
    """Abstract base class for explicit time integration schemes.
    All time intergration schemes are derived from TimeIntegrator.
    """

    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0
            ) -> None:
        
        self.no_stages = None
        self.n = nh - offset
        self.nhx = jnp.s_[:] if "x" in inactive_axes else jnp.s_[self.n:-self.n]    
        self.nhy = jnp.s_[:] if "y" in inactive_axes else jnp.s_[self.n:-self.n]    
        self.nhz = jnp.s_[:] if "z" in inactive_axes else jnp.s_[self.n:-self.n]

        self.timestep_multiplier = ()
        self.timestep_increment_factor = ()

    def integrate_buffer(
            self,
            buffer: Array,
            rhs: Array,
            timestep: float
            ) -> Array:
        """Integrates the conservative variables.

        :param buffer: _description_
        :type buffer: Array
        :param rhs: _description_
        :type rhs: Array
        :param timestep: _description_
        :type timestep: float
        :return: _description_
        :rtype: Array
        """
        if isinstance(rhs, float):
            buffer += timestep * rhs
        else:
            if buffer.shape[-3:] == rhs.shape[-3:]:
                buffer += timestep * rhs
            else: 
                buffer = buffer.at[...,self.nhx,self.nhy,self.nhz].add(timestep * rhs)
        return buffer

    @abstractmethod
    def integrate(
            self,
            buffer: Array,
            rhs: Array,
            timestep: float,
            stage: int,
            ) -> Array:
        """Wrapper function around integrate_buffer. Adjusts the timestep
        according to current RK stage and calls integrate_buffer.
        Implementation in child class.

        :param buffer: _description_
        :type buffer: Array
        :param rhs: _description_
        :type rhs: Array
        :param timestep: _description_
        :type timestep: float
        :param stage: _description_
        :type stage: int
        :return: _description_
        :rtype: Array
        """
        pass

    @abstractmethod
    def prepare_buffer_for_integration(
            self,
            buffer: Array,
            init_buffer: Array,
            stage: int
            ) -> Array:
        """In multi-stage Runge-Kutta methods,
        prepares the buffer for integration.
        Implementation in child class.

        :param buffer: _description_
        :type buffer: Array
        :param init_bufferr: _description_
        :type init_bufferr: Array
        :param stage: _description_
        :type stage: int
        :return: _description_
        :rtype: Array
        """
        pass