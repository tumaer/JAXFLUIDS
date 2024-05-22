from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
from jax import Array

from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.io_utils.logger import Logger
from jaxfluids.io_utils.output_writer import OutputWriter
from jaxfluids.materials.material_manager import MaterialManager

class Callback(ABC):
    """Abstract base class for callbacks. Callbacks are passed to the 
    Simulation Manager upon construction. Compute intensive callbacks
    should generally be jitted. Callbacks which are used inside jitted
    functions cannot have state and should be jit-compilable as well. 
    """

    def init_callback(self,
        domain_information: DomainInformation,
        material_manager: MaterialManager,
        halo_manager: HaloManager,
        logger: Logger,
        output_writer: OutputWriter
        ) -> None:
        """Initializes the callback. In particular,
        this method sets the domain information, material manager, boundary condition,
        logger, and output writer.

        :param domain_information: Domain information
        :type domain_information: DomainInformation
        :param material_manager: Material manager
        :type material_manager: MaterialManager
        :param boundary_conditions: Boundary condition
        :type boundary_conditions: BoundaryCondition
        :param logger: JAX-FLUIDS logger
        :type logger: Logger
        :param output_writer: JAX-FLUIDS output writer
        :type output_writer: OutputWriter
        """

        self.domain_information = domain_information
        self.material_manager   = material_manager
        self.halo_manager = halo_manager

        self.logger         = logger
        self.output_writer  = output_writer

    def on_simulation_start(self, buffer_dictionary: Dict) -> Dict:
        """Called on simulation start

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return buffer_dictionary

    def on_simulation_end(self, buffer_dictionary: Dict) -> Dict:
        """Called on simulation end

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return buffer_dictionary

    def on_step_start(self, buffer_dictionary: Dict) -> Dict:
        """Called on integration step start

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return buffer_dictionary

    def on_step_end(self, buffer_dictionary: Dict) -> Dict:
        """Called on integration step end

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        :return: Updated buffer_dictionary
        :rtype: Dict
        """
        return buffer_dictionary

    def on_stage_start(self, conservatives: Array, primitives: Array, 
        physical_timestep_size: float, physical_simulation_time: float, levelset: Array = None,
        volume_fraction: Array = None, apertures: Union[List, None] = None,
        forcings: Union[Dict, None] = None, 
        ml_parameters_dict: Union[Dict, None] = None, ml_networks_dict: Union[Dict, None] = None) -> Tuple[Array, Array]:
        """Called on integrator stage start

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
        :param ml_parameters_dict: Dictionary containing NN weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary containing NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Tuple of conservative and primitive variable buffers
        :rtype: Tuple[Array, Array]
        """

        return conservatives, primitives

    def on_stage_end(self, conservatives: Array, primitives: Array, 
        physical_timestep_size: float, physical_simulation_time: float, levelset: Array = None,
        volume_fraction: Array = None, apertures: Union[List, None] = None,
        forcings: Union[Dict, None] = None, 
        ml_parameters_dict: Union[Dict, None] = None, ml_networks_dict: Union[Dict, None] = None) -> Tuple[Array, Array]:
        """Called on integrator stage end

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
        :param ml_parameters_dict: Dictionary containing NN weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary containing NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Tuple of conservative and primitive variable buffers
        :rtype: Tuple[Array, Array]
        """
        return conservatives, primitives

    def on_rhs_axis(self) -> None:
        """Called on cell face reconstruction start"""
        # TODO has to be passed deeper into the solver