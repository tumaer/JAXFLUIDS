from typing import Dict, Union, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.case_setup.solid_properties import SolidPropertiesSetup
from jaxfluids.data_types.ml_buffers import MachineLearningSetup

Array = jax.Array

class SolidPropertiesManager:

    def __init__(
            self,
            domain_information: DomainInformation,
            solid_properties_setup: SolidPropertiesSetup,
            ) -> None:
        
        self.domain_information = domain_information
        self.solid_properties_setup = solid_properties_setup

    def get_solid_density(self):
        return self.solid_properties_setup.density
    
    def get_solid_specific_heat_capacity(self):
        return self.solid_properties_setup.specific_heat_capacity

    def compute_internal_energy(
            self,
            solid_temperature: Array
            ) -> Array:

        density = self.solid_properties_setup.density
        specific_heat_capacity = self.solid_properties_setup.specific_heat_capacity
        energy = density * specific_heat_capacity * solid_temperature
        return energy

    def compute_temperature(
            self,
            solid_energy: Array
            ) -> Array:
        density = self.solid_properties_setup.density
        specific_heat_capacity = self.solid_properties_setup.specific_heat_capacity
        solid_temperature = solid_energy/density/specific_heat_capacity
        return solid_temperature

    def compute_thermal_conductivity(
            self,
            mesh_grid: Array,
            solid_temperature: Array
            ) -> Array:
        thermal_conductivity_callable = self.solid_properties_setup.thermal_conductivity
        thermal_conductivity = thermal_conductivity_callable(*mesh_grid, solid_temperature)
        return thermal_conductivity

    def compute_imposed_solid_velocity(
            self,
            physical_simulation_time: float,
            ml_setup: MachineLearningSetup
            ) -> Array:
        """Computes the solid velocity for
        the FLUID-SOLID level-set model, i.e.,
        user prescribed solid velocity.

        :param physical_simulation_time: Current physical simulation time  
        :type physical_simulation_time: float
        :return: Solid interface velocity
        :rtype: Array
        """
        # TODO add user-specified level-set advection velocity
        # from ml_callables/ml_parameters 
        mesh_grid = self.domain_information.compute_device_mesh_grid()

        solid_velocity_setup = self.solid_properties_setup.velocity

        is_callable = solid_velocity_setup.is_callable
        is_blocks = solid_velocity_setup.is_blocks

        if is_blocks:
            solid_velocity = 0.0
            for block in solid_velocity_setup.blocks:
                velocity_callable = block.velocity_callable
                bounding_domain_callable = block.bounding_domain_callable
                
                solid_velocity_block = []
                for field in velocity_callable._fields:
                    velocity_xi_callable = getattr(velocity_callable, field)
                    velocity_xi = velocity_xi_callable(*mesh_grid, physical_simulation_time)
                    solid_velocity_block.append(velocity_xi)
                solid_velocity_block = jnp.stack(solid_velocity_block)

                mask = bounding_domain_callable(*mesh_grid, physical_simulation_time)
                solid_velocity += solid_velocity_block * mask

        elif is_callable:
            if ml_setup is not None:

                ml_callables = ml_setup.callables
                ml_velocity_callable = (
                    ml_callables.levelset.fluid_solid["velocity"]
                    if ml_callables.levelset and ml_callables.levelset.fluid_solid
                    and "velocity" in ml_callables.levelset.fluid_solid
                    else None
                )

                ml_params = ml_setup.parameters
                ml_velocity_params = (
                    ml_params.levelset.fluid_solid["velocity"]
                    if ml_params.levelset and ml_params.levelset.fluid_solid
                    and "velocity" in ml_params.levelset.fluid_solid
                    else None
                )

                use_ml_setup = ml_velocity_callable is not None and ml_velocity_params is not None
            
            else:
                use_ml_setup = None

            if use_ml_setup:
                solid_velocity = []
                for field in ml_velocity_callable._fields:
                    velocity_xi_callable = getattr(ml_velocity_callable, field)
                    velocity_xi_params = getattr(ml_velocity_params, field)
                    velocity_xi = velocity_xi_callable(*mesh_grid, physical_simulation_time, velocity_xi_params)
                    solid_velocity.append(velocity_xi)
                solid_velocity = jnp.stack(solid_velocity)

            else:
                solid_velocity = []
                velocity_callable = solid_velocity_setup.velocity_callable
                for field in velocity_callable._fields:
                    velocity_xi_callable = getattr(velocity_callable, field)
                    velocity_xi = velocity_xi_callable(*mesh_grid, physical_simulation_time)
                    solid_velocity.append(velocity_xi)
                solid_velocity = jnp.stack(solid_velocity)

        else:
            raise NotImplementedError

        return solid_velocity

    def compute_imposed_solid_temperature(
            self,
            physical_simulation_time: float
            ) -> Array:
        """Computes the solid temperature for
        the FLUID-SOLID level-set model, i.e.,
        user prescribed solid temperature.

        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """
        solid_temperature_fn = self.solid_properties_setup.temperature
        mesh_grid = self.domain_information.compute_device_mesh_grid()
        solid_temperature = solid_temperature_fn(*mesh_grid, physical_simulation_time)
        return solid_temperature


