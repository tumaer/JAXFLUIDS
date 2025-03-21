from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionTurbulent
from jaxfluids.turbulence.initialization.hit import initialize_hit
from jaxfluids.turbulence.initialization.channel import initialize_turbulent_channel
from jaxfluids.turbulence.initialization.duct import initialize_turbulent_duct
from jaxfluids.turbulence.initialization.tgv import initialize_tgv

Array = jax.Array

class TurbulenceInitializationManager:
    """ The TurbulenceInitializationManager implements functionality for the initialization
    of turbulent flow fields. The main function of the TurbulenceInitializationManager is
    the get_turbulent_initial_condition method which returns a randomly
    initialized turbulent flow field according to the user-specified initial
    conditions. Currently there are four different options available:

    1) HIT flow field according to Ristorcelli
    2) Taylor-Green vortex
    3) Turbulent channel flow
    4) Turbulent duct flow

    """

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            initial_condition_turbulent: InitialConditionTurbulent,
            ) -> None:

        self.domain_information = domain_information
        self.material_manager = material_manager
        self.equation_information = material_manager.equation_information
        self.initial_condition_turbulent = initial_condition_turbulent

        self.N = domain_information.global_number_of_cells[0]
        random_seed = initial_condition_turbulent.random_seed
        np.random.seed(random_seed)

    def get_turbulent_initial_condition(
            self,
            mesh_grid: List,
            ) -> Array:
        """Calculates turbulent primitive variables.

        Initialization is based on the turbulent case
        specified in the self.turb_init_params dictionary.

        :return: Primitive variables: density, velocity vector, pressure
        :rtype: np.ndarray
        """

        domain_size = self.domain_information.get_device_domain_size()
        split_factors = self.domain_information.split_factors

        turbulent_case = self.initial_condition_turbulent.case
        parameters = self.initial_condition_turbulent.parameters

        # HOMOGENOUS ISOTROPIC TURBULENCE
        if turbulent_case == "HIT":
            # TODO pass material manager
            primitives_init = initialize_hit(
                mesh_grid=mesh_grid,
                split_factors=split_factors,
                gamma=self.material_manager.get_gamma(),
                R=self.material_manager.get_specific_gas_constant(),
                parameters=parameters)

        # TAYLOR GREEN VORTEX
        elif turbulent_case == "TGV":
            # TODO pass material manager
            primitives_init = initialize_tgv(
                mesh_grid=mesh_grid, 
                gamma=self.material_manager.get_gamma(),
                **parameters._asdict())

        # TURBULENT CHANNEL
        elif turbulent_case == "CHANNEL":
            # TODO pass material manager
            primitives_init = initialize_turbulent_channel(
                mesh_grid=mesh_grid,
                domain_size_y=domain_size[1],
                gamma=self.material_manager.get_gamma(),
                **parameters._asdict(),
                R=self.material_manager.get_specific_gas_constant())

        # TURBULENT DUCT
        elif turbulent_case == "DUCT":
            # TODO pass material manager
            primitives_init = initialize_turbulent_duct(
                mesh_grid=mesh_grid,
                domain_size_y=domain_size[1],
                domain_size_z=domain_size[2],
                gamma=self.material_manager.get_gamma(),
                **parameters._asdict(),
                R=self.material_manager.get_specific_gas_constant())

        else:
            raise NotImplementedError

        return primitives_init
