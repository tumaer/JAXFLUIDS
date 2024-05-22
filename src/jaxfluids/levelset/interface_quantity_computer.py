from typing import Tuple
import types

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.levelset.quantity_extender import QuantityExtender
from jaxfluids.data_types.case_setup import SolidPropertiesSetup
from jaxfluids.config import precision

class InterfaceQuantityComputer:
    """The InterfaceQuantityComputer class 
    1) solves the two-material Riemann problem, i.e., computes the interface velocity
        and interface pressure for FLUID-FLUID interface interactions
    2) computes the solid velocity for FLUID-SOLID-DYNAMIC interface interactions
    3) computes the solid temperature for FLUID-SOLID interface interactions
    3) computes the rigid body acceleration for FLUID-SOLID-DYNAMIC-COUPLED interface interactions
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            solid_properties: SolidPropertiesSetup,
            extender_interface: QuantityExtender, 
            numerical_setup: NumericalSetup,
            ) -> None:

        self.eps = precision.get_eps()

        self.extender_interface = extender_interface
        self.material_manager = material_manager
        self.domain_information = domain_information
        self.equation_information = material_manager.equation_information
        self.numerical_setup = numerical_setup

        self.solid_velocity = solid_properties.velocity
        self.solid_density = solid_properties.density
        self.solid_temperature = solid_properties.temperature

        is_viscous_flux = numerical_setup.active_physics.is_viscous_flux
        if is_viscous_flux:
            derivative_stencil = numerical_setup.levelset.interface_flux.derivative_stencil
            self.derivative_stencil : SpatialDerivative = derivative_stencil(
                        nh = domain_information.nh_conservatives,
                        inactive_axes = domain_information.inactive_axes)

        self.aperture_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

    def compute_solid_temperature(
            self,
            physical_simulation_time: float
            ) -> Array:
        """Computes the solid temperature. If
        solid temperature is None, then
        immersed solid boundary is adiabatic.

        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """
        mesh_grid = self.domain_information.compute_device_mesh_grid(mesh_grid)
        solid_temperature = self.solid_temperature.value(*mesh_grid, physical_simulation_time)
        return solid_temperature
    
    def compute_inviscid_solid_acceleration_xi(
            self,
            primitives: Array,
            volume_fraction: Array,
            apertures: Array,
            axis: int,
            gravity: Array = None
            ) -> Array: 

        # DOMAIN INFORMATION
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives

        is_convective_flux = self.numerical_setup.active_physics.is_convective_flux
        is_volume_force = self.numerical_setup.active_physics.is_volume_force
        is_geometric_source = self.numerical_setup.active_physics.is_geometric_source
        is_parallel = self.domain_information.is_parallel

        if is_geometric_source:
            raise NotImplementedError
        
        if is_parallel:
            raise NotImplementedError

        # MASS
        cell_volume = self.domain_information.get_device_cell_volume()
        mass_solid = jnp.sum(self.solid_density * (1.0 - volume_fraction[...,nhx_,nhy_,nhz_]) * cell_volume)

        # BUFFER
        acceleration_xi = jnp.zeros((3,1,1,1))
        
        # DELTA APERTURE
        delta_aperture = apertures[axis][self.aperture_slices[axis][1]][...,nhx_,nhy_,nhz_] - \
                            apertures[axis][self.aperture_slices[axis][0]][...,nhx_,nhy_,nhz_]

        # INVISCID CONTRIBUTION
        if is_convective_flux:
            cell_face_areas = self.domain_information.get_device_cell_face_areas()
            pressure = primitives[4,nhx,nhy,nhz]
            inviscid_contribution_xi = jnp.sum( pressure * delta_aperture * cell_face_areas[axis])
            acceleration_xi = acceleration_xi.at[axis].set(inviscid_contribution_xi)

        # GRAVITY CONTRIBUTION
        if is_volume_force:
            gravity_force = mass_solid * gravity[axis]
            acceleration_xi = acceleration_xi.at[axis].add(gravity_force)

        acceleration_xi *= 1./mass_solid

        return acceleration_xi

    def compute_viscous_solid_acceleration(
            self,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            apertures: Array,
            friction: Array,
            ) -> Array:

        # DOMAIN INFORMATION
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        is_parallel = self.domain_information.is_parallel

        # MASS
        cell_volume = self.domain_information.get_device_cell_volume()
        mass_solid = jnp.sum(self.solid_density * (1.0 - volume_fraction[...,nhx_,nhy_,nhz_]) * cell_volume)
        
        # VISCOUS CONTRIBUTION
        viscous_force_vector = jnp.sum(-friction, axis=(-1,-2,-3), keepdims=True)
        acceleration = viscous_force_vector/mass_solid

        return acceleration

    def compute_solid_velocity(
            self,
            physical_simulation_time: float
            ) -> Array:
        """Computes the solid interface velocity for
        FLUID-SOLID-DYNAMIC interface interactions.

        :param physical_simulation_time: Current physical simulation time  
        :type physical_simulation_time: float
        :return: Solid interface velocity
        :rtype: Array
        """

        mesh_grid = self.domain_information.compute_device_mesh_grid()

        is_callable = self.solid_velocity.is_callable
        is_blocks = self.solid_velocity.is_blocks

        if is_blocks:
            solid_velocity = 0.0
            for block in self.solid_velocity.blocks:
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
            solid_velocity = []
            velocity_callable = self.solid_velocity.velocity_callable
            for field in velocity_callable._fields:
                velocity_xi_callable = getattr(velocity_callable, field)
                velocity_xi = velocity_xi_callable(*mesh_grid, physical_simulation_time)
                solid_velocity.append(velocity_xi)
            solid_velocity = jnp.stack(solid_velocity)
        else:
            raise NotImplementedError

        return solid_velocity


    def solve_interface_interaction(
            self,
            primitives: Array,
            normal: Array,
            curvature: Array
            ) -> Tuple[Array, Array]:
        """Solves the two-material Riemann problem for FLUID-FLUID interface interactions.

        :param primitives: Primitive variable buffer
        :type primitives: Array
        :param normal: Interface normal buffer
        :type normal: Array
        :param curvature: Interface curvature buffer
        :type curvature: Array
        :return: Interface velocity and interface pressure
        :rtype: Tuple[Array, Array]
        """

        is_surface_tension = self.numerical_setup.active_physics.is_surface_tension

        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        energy_ids = self.equation_information.energy_ids
        mass_ids = self.equation_information.mass_ids
        velocity_ids = self.equation_information.velocity_ids

        pressure = primitives[energy_ids,...,nhx__,nhy__,nhz__]
        density = primitives[mass_ids,...,nhx__,nhy__,nhz__]
        velocity = primitives[velocity_ids,...,nhx__,nhy__,nhz__]
    
        velocity_normal_projection = jnp.einsum('ijklm, ijklm -> jklm', velocity, jnp.expand_dims(normal, axis=1) )
        speed_of_sound = self.material_manager.get_speed_of_sound(pressure=pressure, density=density)
        impendance = speed_of_sound * density
        inverse_impendace_sum = 1.0 / ( impendance[0] + impendance[1] + self.eps )

        # CAPILLARY PRESSURE JUMP
        if is_surface_tension:
            delta_p = self.material_manager.get_sigma() * curvature
        else:
            delta_p = 0.0

        # INTERFACE QUANTITIES
        interface_velocity = ( impendance[1] * velocity_normal_projection[1] + impendance[0] * velocity_normal_projection[0] + \
                                pressure[1] - pressure[0] - delta_p ) * inverse_impendace_sum
        interface_pressure_positive = (impendance[1] * pressure[0] + impendance[0] * (pressure[1] - delta_p) + \
                                        impendance[0] * impendance[1] * (velocity_normal_projection[1] - velocity_normal_projection[0]) ) * inverse_impendace_sum
        interface_pressure_negative = (impendance[1] * (pressure[0] + delta_p) + impendance[0] * pressure[1] + \
                                        impendance[0] * impendance[1] * (velocity_normal_projection[1] - velocity_normal_projection[0]) ) * inverse_impendace_sum

        interface_pressure = jnp.stack([interface_pressure_positive, interface_pressure_negative], axis=0)

        return interface_velocity, interface_pressure



    def compute_interface_quantities(
            self,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            normal: Array,
            curvature: Array,
            steps: int = None,
            CFL: float = None,
            interface_velocity_old: Array = None,
            interface_pressure_old: Array = None,
            ) -> Tuple[Array, Array]:
        """Computes interface velocity and pressure for
        FLUID-FLUID interface interaction and
        extends the values into the narrowband_computation.
        If step/CFL is provided, they will be used for the
        extension procedure, otherwise the values specified
        in the numerical setup will be used.
        If interface_velocity/pressure_old buffer is
        provided, they will be used as starting values
        in the extension procedure, otherwise 0.0 will
        be used as starting values.

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param normal: _description_
        :type normal: Array
        :param curvature: _description_
        :type curvature: Array
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        is_convective_flux = self.numerical_setup.active_physics.is_convective_flux
        is_viscous_flux = self.numerical_setup.active_physics.is_viscous_flux

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx__, nhy__, nhz__ = self.domain_information.domain_slices_conservatives_to_geometry
        nh_offset = self.domain_information.nh_offset
        smallest_cell_size = self.domain_information.smallest_cell_size
        
        levelset_setup = self.numerical_setup.levelset
        narrowband_setup = levelset_setup.narrowband
        narrowband_computation = narrowband_setup.computation_width

        force_steps = True if steps != None else False
        CFL = levelset_setup.extension.CFL_interface if CFL == None else CFL
        steps = levelset_setup.extension.steps_interface if steps == None else steps

        if is_convective_flux or is_viscous_flux:

            normalized_levelset = jnp.abs(levelset[nhx,nhy,nhz])/smallest_cell_size
            mask_narrowband = jnp.where(normalized_levelset <= narrowband_computation, 1, 0)
            normal_extend = normal * jnp.sign(levelset[nhx__,nhy__,nhz__])
            # cut_cell_mask = compute_cut_cell_mask(levelset, nh_offset)
            cut_cell_mask = (volume_fraction > 0.0) & (volume_fraction < 1.0)
            inverse_cut_cell_mask = 1 - cut_cell_mask
            mask_extend = inverse_cut_cell_mask[nhx_,nhy_,nhz_] * mask_narrowband

            # COMPUTE INTERFACE QUANTITIES
            interface_velocity, interface_pressure = self.solve_interface_interaction(
                primitives, normal, curvature)

            interface_velocity *= cut_cell_mask
            interface_pressure *= cut_cell_mask

            if interface_velocity_old != None:
                interface_velocity = interface_velocity + interface_velocity_old * (1-cut_cell_mask)
            if interface_pressure_old != None:
                interface_pressure = interface_pressure + interface_pressure_old * (1-cut_cell_mask)

            interface_quantities = jnp.concatenate([
                jnp.expand_dims(interface_velocity, axis=0),
                interface_pressure], axis=0)
            
            # EXTEND INTERFACE QUANTITIES
            interface_quantities, step_count = self.extender_interface.extend(
                interface_quantities, normal_extend, mask_extend,
                0.0, CFL, steps, force_steps=force_steps)

            # CUT OFF NARROW BAND
            interface_quantities = interface_quantities.at[...,nhx_,nhy_,nhz_].mul(mask_narrowband)

            interface_velocity = interface_quantities[0]
            interface_pressure = interface_quantities[1:]

        else:

            interface_velocity = jnp.zeros_like(levelset[nhx__,nhy__,nhz__])
            interface_pressure = jnp.ones((2,) + levelset[nhx__,nhy__,nhz__].shape)

        return interface_velocity, interface_pressure, step_count


