from typing import Dict, Tuple
from jax import Array
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.numerical_setup.levelset import SolidCouplingSetup
from jaxfluids.data_types.case_setup.solid_properties import SolidPropertiesSetup
from jaxfluids.equation_information import EquationInformation


def get_solid_velocity_and_temperature(
        solid_velocity: Array,
        solid_temperature: Array,
        physical_simulation_time: float,
        solid_properties_manager: SolidPropertiesManager,
        equation_information: EquationInformation,
        domain_slices: Tuple,
        ) -> Tuple[Array, Array]:
    """Implements logic to retrieve solid velocity and temperature
    depending on present numerical setup

    :param solid_velocity: _description_
    :type solid_velocity: Array
    :param solid_temperature: _description_
    :type solid_temperature: Array
    :param physical_simulation_time: _description_
    :type physical_simulation_time: float
    :param solid_properties_manager: _description_
    :type solid_properties_manager: SolidPropertiesManager
    :param equation_information: _description_
    :type equation_information: EquationInformation
    :param domain_slices: _description_
    :type domain_slices: Tuple
    :return: _description_
    :rtype: Tuple[Array, Array]
    """

    nhx,nhy,nhz = domain_slices
    levelset_model = equation_information.levelset_model

    solid_temperature_setup = solid_properties_manager.solid_properties_setup.temperature
    levelset_model = equation_information.levelset_model
    is_moving_levelset = equation_information.is_moving_levelset
    solid_coupling = equation_information.solid_coupling

    if solid_coupling.dynamic == "ONE-WAY":
        solid_velocity = solid_properties_manager.compute_imposed_solid_velocity(physical_simulation_time)
    else:
        pass

    if solid_coupling.thermal == "ONE-WAY":
        solid_temperature = solid_properties_manager.compute_imposed_solid_temperature(physical_simulation_time)
    else:
        pass
    
    return solid_temperature, solid_velocity