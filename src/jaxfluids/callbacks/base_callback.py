from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.buffers import (
    SimulationBuffers, TimeControlVariables, ForcingParameters, IntegrationBuffers,
    MaterialFieldBuffers, LevelsetFieldBuffers, SolidFieldBuffers, ForcingBuffers
)
from jaxfluids.data_types.information import StepInformation
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

if TYPE_CHECKING:
    from jaxfluids import SimulationManager

Array = jax.Array

class Callback(ABC):
    """Abstract base class for callbacks. Callbacks are passed to the 
    Simulation Manager upon construction. Compute intensive callbacks
    should generally be jitted. Callbacks which are used inside jitted
    functions cannot have state and should be jit-compilable as well. 
    """

    def init_callback(self, sim_manager: SimulationManager) -> None:
        """Initializes the callback.
        """

        self.sim_manager = sim_manager
        self.domain_information = sim_manager.domain_information
        self.equation_information = sim_manager.equation_information

    def on_simulation_start(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: Dict,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Called on simulation start

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return jxf_buffers, callback_dict

    def on_simulation_end(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: Dict,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Called on simulation end

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return jxf_buffers, callback_dict

    def on_step_start(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: Dict,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Called on integration step start

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return jxf_buffers, callback_dict

    def on_step_end(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: Dict,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Called on integration step end

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return jxf_buffers, callback_dict
    
    def before_step_start(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: Dict,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Called before integration step start. This callback can
        have state as it is not called inside do_integration_step.

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return jxf_buffers, callback_dict

    def after_step_end(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: Dict,
            **kwargs
        ) -> Tuple[JaxFluidsBuffers, Dict]:
        """Called after integration step end. This callback can
        have state as it is not called inside do_integration_step.

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return jxf_buffers, callback_dict

    def on_stage_start(
            self, 
            conservatives: Array, 
            primitives: Array, 
            physical_timestep_size: float, 
            physical_simulation_time: float, 
            levelset: Array = None,
            volume_fraction: Array = None, 
            apertures: Union[List, None] = None,
            forcing_buffers: ForcingBuffers = None,
            ml_setup: MachineLearningSetup = None,
            **kwargs
        ) -> Tuple[Array, Array]:
        """Called on integrator stage start before
        the call to the right-hand side calculation. 

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param physical_timestep_size: Current physical time step size
        :type physical_timestep_size: float
        :param physical_simulation_time: Current physical simulation time
        :type physical_simulation_time: float
        :param levelset: Levelset buffer, defaults to None
        :type levelset: Array, optional
        :param volume_fraction: Volume fraction buffer, defaults to None
        :type volume_fraction: Array, optional
        :param apertures: Aperture buffers, defaults to None
        :type apertures: Union[List, None], optional
        :param forcings: Dictionary containing forcing buffers, defaults to None
        :type forcings: Union[Dict, None], optional
        :param ml_setup: Dictionary containing NN weights, defaults to None
        :type ml_setup: Union[Dict, None], optional
        :return: Tuple of conservative and primitive variable buffers
        :rtype: Tuple[Array, Array]
        """

        return conservatives, primitives

    def on_stage_end(
            self, 
            conservatives: Array, 
            primitives: Array, 
            physical_timestep_size: float, 
            physical_simulation_time: float, 
            levelset: Array = None,
            volume_fraction: Array = None, 
            apertures: Union[List, None] = None,
            forcing_buffers: ForcingBuffers = None,
            ml_setup: MachineLearningSetup = None,
            **kwargs
        ) -> Tuple[Array, Array]:
        """Called on integrator stage end before integration
        buffers are created for the next integration stage.

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param physical_timestep_size: Current physical time step size
        :type physical_timestep_size: float
        :param physical_simulation_time: Current physical simulation time
        :type physical_simulation_time: float
        :param levelset: Levelset buffer, defaults to None
        :type levelset: Array, optional
        :param volume_fraction: Volume fraction buffer, defaults to None
        :type volume_fraction: Array, optional
        :param apertures: Aperture buffers, defaults to None
        :type apertures: Union[List, None], optional
        :param forcings: Dictionary containing forcing buffers, defaults to None
        :type forcings: Union[Dict, None], optional
        :param ml_setup: Dictionary containing NN weights, defaults to None
        :type ml_setup: Union[Dict, None], optional
        :return: Tuple of conservative and primitive variable buffers
        :rtype: Tuple[Array, Array]
        """
        return conservatives, primitives

    def on_rhs_axis(self) -> None:
        """Called on cell face reconstruction start"""
        # TODO has to be passed deeper into the solver

    def after_compute_rhs(
            self,
            rhs_buffers: IntegrationBuffers,
            material_fields: MaterialFieldBuffers,
            levelset_fields: LevelsetFieldBuffers,
            solid_fields: SolidFieldBuffers,
            forcing_buffers: ForcingBuffers,
            ml_setup: MachineLearningSetup,
            **kwargs
        ) -> IntegrationBuffers:
        """Called inside the do_runge_kutta_stages after 
        the right-hand side (RHS) terms are computed. 
        This callback is designed to add user-defined terms to
        or modify the RHS.

        :param rhs_buffers: Right-hand side buffer computed by space solver
        :type rhs_buffers: IntegrationBuffers
        :param material_fields: Material fields
        :type material_fields: MaterialFieldBuffers
        :param levelset_fields: Level-set related fields
        :type levelset_fields: LevelsetFieldBuffers
        :param solid_fields: Solid fields
        :type solid_fields: SolidFieldBuffers
        :param forcing_buffers: Forcing buffers
        :type forcing_buffers: ForcingBuffers
        :param ml_setup: ML-setup with callables and parameters
        :type ml_setup: MachineLearningSetup
        :return: Right-hand side buffer
        :rtype: IntegrationBuffers
        """

        return rhs_buffers