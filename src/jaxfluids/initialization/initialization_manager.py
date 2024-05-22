import types
from typing import Union, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.input.input_manager import InputManager
from jaxfluids.initialization.material_fields_initializer import MaterialFieldsInitializer
from jaxfluids.initialization.levelset_initializer import LevelsetInitializer
from jaxfluids.initialization.forcings_initializer import ForcingsInitializer
from jaxfluids.data_types.buffers import SimulationBuffers, \
    MaterialFieldBuffers, LevelsetFieldBuffers, ForcingParameters, TimeControlVariables
from jaxfluids.domain.helper_functions import reassemble_buffer, reassemble_buffer_np

class InitializationManager:
    """The InitializationManager class implements functionality to create a dictionary of initial buffers that is 
    passed to the simulate() method of the SimulationManager class. The initialization() method returns this
    dictionary. The initial buffers are created in one of the following ways: 
    1) From a restart file that is specified in the case setup.
    2) From turbulent initial condition parameters that are specified in the case setup
    3) From the initial primitive buffer that is passed to the initialization() method
    4) From the initial conditions for primitive variables specified in case setup
    Note that if multiple of the above are provided, the priority is 1) - 4).
    """

    def __init__(
            self,
            input_manager: InputManager
            ) -> None:

        self.numerical_setup = input_manager.numerical_setup
        self.case_setup = input_manager.case_setup

        unit_handler = input_manager.unit_handler
        material_manager = input_manager.material_manager
        equation_manager = input_manager.equation_manager
        domain_information = input_manager.domain_information
        halo_manager = input_manager.halo_manager
        self.equation_information = input_manager.equation_information

        self.material_fields_initializer = MaterialFieldsInitializer(
            domain_information=domain_information,
            unit_handler=unit_handler,
            equation_manager=equation_manager,
            material_manager=material_manager,
            halo_manager=halo_manager,
            initial_condition=input_manager.case_setup.initial_condition_setup,
            restart_setup=input_manager.case_setup.restart_setup,
            is_double_precision=self.numerical_setup.precision.is_double_precision_compute)

        if input_manager.equation_information.levelset_model:
            self.levelset_initializer = LevelsetInitializer(
                numerical_setup = self.numerical_setup,
                domain_information = domain_information,
                unit_handler = unit_handler,
                equation_manager = equation_manager,
                material_manager = material_manager,
                halo_manager = halo_manager,
                initial_condition_levelset = input_manager.case_setup.initial_condition_setup.levelset,
                initial_condition_solid_velocity = input_manager.case_setup.initial_condition_setup.solid_velocity,
                solid_properties = input_manager.case_setup.solid_properties_setup,
                restart_setup = input_manager.case_setup.restart_setup)

        if input_manager.numerical_setup.active_forcings:
            self.forcings_initializer = ForcingsInitializer(
                numerical_setup = self.numerical_setup,
                domain_information = domain_information,
                material_manager = material_manager,
                restart_setup = input_manager.case_setup.restart_setup)


    def initialization(
            self,
            user_prime_init: Union[np.ndarray, Array] = None,
            user_time_init: float = None,
            user_levelset_init: Union[np.ndarray, Array] = None,
            user_solid_interface_velocity_init: Union[np.ndarray, Array] = None,
            ) -> Tuple[SimulationBuffers, TimeControlVariables,
                       ForcingParameters]:
        """Creates a buffer dictionary containing the initial buffers
        for the material fields, time control variables,
        levelset related fields, and forcings.

        :param user_prime_init: _description_, defaults to None
        :type user_prime_init: Union[np.ndarray, Array], optional
        :param user_levelset_init: _description_, defaults to None
        :type user_levelset_init: Union[np.ndarray, Array], optional
        :return: _description_
        :rtype: Tuple[SimulationBuffers, TimeControlVariables, ForcingParameters]
        """

        # MATERIAL FIELDS
        material_fields, time_control_variables = \
        self.material_fields_initializer.initialize(user_prime_init, user_time_init)

        # LEVELSET
        if self.equation_information.levelset_model:
            levelset_fields, material_fields = \
            self.levelset_initializer.initialize(
                material_fields, time_control_variables, user_levelset_init,
                user_solid_interface_velocity_init)
        else:
            levelset_fields = LevelsetFieldBuffers()

        # FORCINGS
        active_forcings = self.numerical_setup.active_forcings
        if any(active_forcings._asdict().values()):
            forcing_parameters = self.forcings_initializer.initialize(
                material_fields)
        else:
            forcing_parameters = ForcingParameters()

        simulation_buffers = SimulationBuffers(
            material_fields, levelset_fields)

        return simulation_buffers, time_control_variables, forcing_parameters
        
