from abc import ABC, abstractmethod
from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.data_types.buffers import IntegrationBuffers, EulerIntegrationBuffers
from jaxfluids.equation_information import EquationInformation

Array = jax.Array

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


    def perform_stage_integration(
            self,
            integration_buffers: IntegrationBuffers,
            rhs_buffers: IntegrationBuffers,
            initial_stage_buffers: IntegrationBuffers,
            physical_timestep_size: float,
            stage: int,
            equation_information: EquationInformation
        ) -> IntegrationBuffers:
        """Performs a stage integration step.
        1) Transform volume-averaged conservatives
            to actual conservatives (only for levelset
            simulations)
        2) Compute stage buffer
        3) Integrate

        :param integration_buffers: _description_
        :type integration_buffers: IntegrationBuffers
        :param initial_stage_buffers: _description_
        :type initial_stage_buffers: IntegrationBuffers
        :param stage: _description_
        :type stage: int
        :return: _description_
        :rtype: IntegrationBuffers
        """

        is_moving_levelset = equation_information.is_moving_levelset
        solid_coupling = equation_information.solid_coupling

        integration_euler_buffers = integration_buffers.euler_buffers

        if stage > 0:
            # NOTE EULER FIELDS
            initial_stage_euler_buffers = initial_stage_buffers.euler_buffers
            conservatives = self.prepare_buffer_for_integration(
                integration_euler_buffers.conservatives,
                initial_stage_euler_buffers.conservatives,
                stage
            )
            if is_moving_levelset:
                levelset = self.prepare_buffer_for_integration(
                    integration_euler_buffers.levelset,
                    initial_stage_euler_buffers.levelset,
                    stage
                )
            else:
                levelset = None

            if solid_coupling.dynamic == "TWO-WAY":
                solid_velocity = self.prepare_buffer_for_integration(
                    integration_euler_buffers.solid_velocity,
                    initial_stage_euler_buffers.solid_velocity,
                    stage
                )
            else:
                solid_velocity = None

            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

            else:
                solid_energy = None
            
            integration_euler_buffers = EulerIntegrationBuffers(
                conservatives,
                levelset,
                solid_velocity,
                solid_energy
            )
                    

        # PERFORM INTEGRATION
        # NOTE EULER FIELDS
        rhs_euler_buffers = rhs_buffers.euler_buffers

        conservatives = self.integrate(
            integration_euler_buffers.conservatives,
            rhs_euler_buffers.conservatives,
            physical_timestep_size,
            stage
        )

        if is_moving_levelset:
            levelset = self.integrate(
                integration_euler_buffers.levelset,
                rhs_euler_buffers.levelset,
                physical_timestep_size,
                stage
            )
        else:
            levelset = None

        if solid_coupling.dynamic == "TWO-WAY":
            solid_velocity = self.integrate(
                integration_euler_buffers.solid_velocity,
                rhs_euler_buffers.solid_velocity,
                physical_timestep_size,
                stage
            )
        else:
            solid_velocity = None

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

        else:
            solid_energy = None
        
        integration_euler_buffers = EulerIntegrationBuffers(
            conservatives,
            levelset,
            solid_velocity,
            solid_energy
        )

        # CREATE CONTAINER
        integration_buffers = IntegrationBuffers(
            integration_euler_buffers,
        )

        return integration_buffers