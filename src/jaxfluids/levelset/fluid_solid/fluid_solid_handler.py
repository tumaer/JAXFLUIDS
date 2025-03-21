from typing import Dict, Union, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.information import LevelsetPositivityInformation, LevelsetProcedureInformation
from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.numerical_setup import ActivePhysicsSetup, LevelsetSetup
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.levelset.mixing.solids_mixer import SolidsMixer
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered
from jaxfluids.initialization.helper_functions import create_field_buffer
from jaxfluids.levelset.fluid_solid.interface_quantities import compute_thermal_interface_state
from jaxfluids.levelset.fluid_solid.helper_functions import get_solid_velocity_and_temperature
from jaxfluids.levelset.fluid_solid.interface_flux_contributions import convective_interface_flux, viscous_interface_flux, heat_interface_flux
from jaxfluids.config import precision
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField, LevelsetSolidCellIndices

Array = jax.Array

class FluidSolidLevelsetHandler:


    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            halo_manager: HaloManager,
            geometry_calculator: GeometryCalculator,
            levelset_setup: LevelsetSetup,
            solid_properties_manager: SolidPropertiesManager,
            extender: IterativeExtender = None
            ) -> None:

        self.material_manager = material_manager
        self.domain_information = domain_information
        self.halo_manager = halo_manager
        self.equation_information = material_manager.equation_information
        self.levelset_setup = levelset_setup
        self.geometry_calculator = geometry_calculator

        self.solid_properties_manager = solid_properties_manager

        solid_coupling = self.equation_information.solid_coupling
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        self.aperture_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

    def compute_interface_flux_xi(
            self,
            primitives: Array,
            volume_fraction: Array,
            levelset: Array,
            apertures: Tuple[Array],
            axis_index: int,
            gravity: Array,
            physical_simulation_time: float,
            solid_velocity: Array = None
            ) -> Array:
        """Computes the dimension-by-dimension
        interface flux contributions, i.e.,
        convective flux. For coupled fluid solid
        level-set models, computes acceleration on solid
        induced by pressure field and gravity.
        

        :param primitives: _description_
        :type primitives: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param apertures: _description_
        :type apertures: Array
        :param axis_index: _description_
        :type axis_index: int
        :param gravity: _description_
        :type gravity: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param solid_velocity: _description_, defaults to None
        :type solid_velocity: Array, optional
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """

        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives

        is_parallel = self.domain_information.is_parallel
        
        interface_flux_setup = self.levelset_setup.interface_flux
        active_physics_setup = self.equation_information.active_physics
        is_convective_flux = active_physics_setup.is_convective_flux
        is_geometric_source = active_physics_setup.is_geometric_source
        is_volume_force = active_physics_setup.is_volume_force
        levelset_model = self.levelset_setup.model
        is_interpolate_pressure = interface_flux_setup.is_interpolate_pressure

        is_moving_levelset = self.equation_information.is_moving_levelset
        solid_coupling = self.equation_information.solid_coupling

        s_energy = self.equation_information.s_energy
        s_momentum_xi = self.equation_information.s_momentum_xi
        s_xi = s_momentum_xi[axis_index]

        interface_flux_xi = jnp.zeros(primitives[...,nhx,nhy,nhz].shape)

        s1 = self.aperture_slices[axis_index][0]
        s2 = self.aperture_slices[axis_index][1]
        apertures_xi = apertures[axis_index]
        delta_aperture = apertures_xi[s1][...,nhx_,nhy_,nhz_] - apertures_xi[s2][...,nhx_,nhy_,nhz_]
        
        # TODO aaron move all this into compute interface_flux ??
        if solid_coupling.dynamic == "TWO-WAY":
                
            if is_geometric_source:
                raise NotImplementedError
            
            if is_parallel:
                raise NotImplementedError

            solid_density = self.solid_properties_manager.get_solid_density()
            cell_volume = self.domain_information.get_device_cell_volume()
            mass_solid = jnp.sum(solid_density * (1.0 - volume_fraction[...,nhx_,nhy_,nhz_]) * cell_volume)

            acceleration_xi = jnp.zeros((3,1,1,1))

            # pressure acting on solid
            if is_convective_flux and not is_interpolate_pressure:
                pressure = primitives[s_energy,nhx,nhy,nhz]
                cell_face_areas = self.domain_information.get_device_cell_face_areas()
                momentum_flux_xi = - pressure * delta_aperture * cell_face_areas[axis_index]
                inviscid_contribution_xi = jnp.sum( momentum_flux_xi ) 
                acceleration_xi = acceleration_xi.at[axis_index].set(inviscid_contribution_xi)

            # gravity acting on solid
            if is_volume_force:
                gravity_force = mass_solid * gravity[axis_index]
                acceleration_xi = acceleration_xi.at[axis_index].add(gravity_force)

            acceleration_xi *= 1./mass_solid

        else:
            acceleration_xi = None

        return interface_flux_xi, acceleration_xi


    def compute_interface_flux(
            self,
            primitives: Array,
            solid_velocity: Array,
            solid_temperature: Array,
            levelset: Array,
            volume_fraction: Array,
            apertures: Tuple[Array],
            physical_simulation_time: float,
            interface_cells: LevelsetSolidCellIndicesField = None,
            ml_setup: MachineLearningSetup = None,
            is_return_interpolated_temperature: bool = False
            ) -> Array:
        """Computes interface flux
        contributions, i.e., convective, viscous and heat flux.
        For coupled fluid solid level-set models,
        also computes the acceleration induced by
        viscous fluxes on the solid, as well as
        heat flux into solid.


        :param primitives: _description_
        :type primitives: Array
        :param solid_velocity: _description_
        :type solid_velocity: Array
        :param solid_temperature: _description_
        :type solid_temperature: Array
        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param apertures: _description_
        :type apertures: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
        nx,ny,nz = self.domain_information.device_number_of_cells
        dim = self.domain_information.dim
        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices
        is_parallel = self.domain_information.is_parallel

        no_primes = self.equation_information.no_primes
        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset

        active_physics_setup = self.equation_information.active_physics
        is_viscous_flux = active_physics_setup.is_viscous_flux
        is_heat_flux = active_physics_setup.is_heat_flux
        is_convective_flux = active_physics_setup.is_convective_flux

        interface_flux_setup = self.levelset_setup.interface_flux
        material_properties_averaging = interface_flux_setup.material_properties_averaging
        is_interpolate_pressure = interface_flux_setup.is_interpolate_pressure
        is_cell_based_computation = interface_flux_setup.is_cell_based_computation
        is_cell_based_computation = is_cell_based_computation and interface_cells is not None

        if is_cell_based_computation:
            interface_cells_indices = interface_cells.indices
            interface_cells_mask = interface_cells.mask

        solid_coupling = self.equation_information.solid_coupling

        ids_mass = self.equation_information.ids_mass
        ids_energy = self.equation_information.ids_energy
        s_velocity = self.equation_information.s_velocity

        s_mass = self.equation_information.s_mass
        s_energy = self.equation_information.s_energy
        s_velocity = self.equation_information.s_velocity

        if solid_coupling.dynamic == "ONE-WAY":
            solid_velocity = self.solid_properties_manager.compute_imposed_solid_velocity(
                physical_simulation_time,
                ml_setup
            )
        elif solid_coupling.dynamic == "TWO-WAY":
            solid_velocity = solid_velocity[...,nhx,nhy,nhz]
        else:
            pass

        if solid_coupling.thermal == "ONE-WAY":
            solid_temperature = self.solid_properties_manager.compute_imposed_solid_temperature(physical_simulation_time)
        else:
            pass

        interface_flux = jnp.zeros(primitives[...,nhx,nhy,nhz].shape)
        solid_heat_flux = None
        acceleration = None

        interpolation_dh = self.levelset_setup.interface_flux.interpolation_dh
        dh = interpolation_dh * cell_size

        # NOTE aperture based normal must be used for consistent convective flux
        # TODO what about dissipative fluxes ?
        cell_centers_halos = self.domain_information.get_device_cell_centers_halos()
        normal = self.geometry_calculator.compute_normal(levelset)
        normal_aperture_based = self.geometry_calculator.compute_normal_apertures_based(apertures)
        interface_length = self.geometry_calculator.compute_interface_length(apertures)

        mesh_grid = self.domain_information.compute_device_mesh_grid()
        mesh_grid = jnp.array(mesh_grid)    

        # NOTE for static solids, we only compute flux in interface cells, i.e., the
        # buffers with _cc have shape (...,N_if), where N_if is the number of interface cells
        if is_cell_based_computation:
            s_if = (...,) + interface_cells_indices
            buffers_tuple = (primitives[...,nhx,nhy,nhz], levelset[...,nhx,nhy,nhz],
                            normal_aperture_based[...,nhx_,nhy_,nhz_], normal[...,nhx_,nhy_,nhz_],
                            interface_length[...,nhx_,nhy_,nhz_], mesh_grid)
            primitives_cc, levelset_cc, normal_aperture_based_cc, \
                normal_cc, interface_length_cc, mesh_grid_cc = tuple([buffer[s_if] for buffer in buffers_tuple])
            
            if solid_coupling.dynamic == "ONE-WAY":
                solid_velocity_cc = solid_velocity[s_if]
            else:
                solid_velocity_cc = None

            if solid_coupling.thermal == "ONE-WAY":
                solid_temperature_cc = solid_temperature[s_if]
            else:
                solid_temperature_cc = None

        else:

            primitives_cc, levelset_cc = primitives[...,nhx,nhy,nhz], levelset[...,nhx,nhy,nhz]
            normal_cc, normal_aperture_based_cc = normal[...,nhx_,nhy_,nhz_], normal_aperture_based[...,nhx_,nhy_,nhz_]
            interface_length_cc = interface_length[...,nhx_,nhy_,nhz_]
            mesh_grid_cc = mesh_grid
            solid_velocity_cc = solid_velocity
            solid_temperature_cc = solid_temperature

        if is_convective_flux:

            momentum_flux, energy_flux = convective_interface_flux(
                primitives_cc, levelset_cc, normal_aperture_based_cc,
                interface_length_cc, solid_velocity_cc, mesh_grid_cc, dh,
                is_cell_based_computation, is_interpolate_pressure,
                self.material_manager, self.domain_information)
            
            if is_cell_based_computation:
                if is_parallel:
                    interface_flux = interface_flux.at[(s_velocity,)+s_if].add(momentum_flux*interface_cells_mask)
                    interface_flux = interface_flux.at[(ids_energy,)+s_if].add(energy_flux*interface_cells_mask)
                else:
                    interface_flux = interface_flux.at[(s_velocity,)+s_if].add(momentum_flux)
                    interface_flux = interface_flux.at[(ids_energy,)+s_if].add(energy_flux)
            else:
                interface_flux = interface_flux.at[s_velocity].add(momentum_flux)
                interface_flux = interface_flux.at[ids_energy].add(energy_flux)


        # NOTE compute interpolated state
        if is_viscous_flux or is_heat_flux:

            # NOTE fluid interpolation point
            s_ = jnp.s_[active_axes_indices,...]
            ip_fluid = mesh_grid_cc + normal_cc[s_] * (-levelset_cc + dh)

            if not is_cell_based_computation:
                ip_fluid = ip_fluid.reshape(dim,-1)
            primitives_ip = linear_interpolation_scattered(
                jnp.swapaxes(ip_fluid, -1, 0), primitives,
                cell_centers_halos)
            if not is_cell_based_computation:
                primitives_ip = primitives_ip.reshape(no_primes,nx,ny,nz)

            # NOTE solid interpolation point
            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

            elif solid_coupling.thermal == "ONE-WAY":
                solid_temperature_ip = solid_temperature_cc
                ip_solid = None

        if is_viscous_flux:

            momentum_flux, energy_flux = viscous_interface_flux(
                primitives_ip, normal_cc, interface_length_cc,
                solid_velocity_cc, dh, self.material_manager,
                self.domain_information)

            if is_cell_based_computation:
                if is_parallel:
                    interface_flux = interface_flux.at[(s_velocity,)+s_if].add(momentum_flux*interface_cells_mask)
                    interface_flux = interface_flux.at[(ids_energy,)+s_if].add(energy_flux*interface_cells_mask)
                else:
                    interface_flux = interface_flux.at[(s_velocity,)+s_if].add(momentum_flux)
                    interface_flux = interface_flux.at[(ids_energy,)+s_if].add(energy_flux)
            else:
                interface_flux = interface_flux.at[s_velocity].add(momentum_flux)
                interface_flux = interface_flux.at[ids_energy].add(energy_flux)

        if is_heat_flux and solid_coupling.thermal:
            
            energy_flux = heat_interface_flux(
                primitives_ip, solid_temperature_ip, ip_solid,
                interface_length_cc, dh,
                material_properties_averaging, ml_setup,
                self.material_manager, self.solid_properties_manager,
                self.domain_information)

            if is_cell_based_computation:
                if is_parallel:
                    interface_flux = interface_flux.at[(ids_energy,)+s_if].add(energy_flux*interface_cells_mask)
                else:
                    interface_flux = interface_flux.at[(ids_energy,)+s_if].add(energy_flux)
            else:
                interface_flux = interface_flux.at[ids_energy].add(energy_flux)

            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

        if solid_coupling.dynamic == "TWO-WAY":
            is_parallel = self.domain_information.is_parallel
            if is_parallel:
                raise NotImplementedError
            cell_volume = self.domain_information.get_device_cell_volume()
            solid_density = self.solid_properties_manager.get_solid_density()
            mass_solid = jnp.sum(solid_density * (1.0 - volume_fraction[...,nhx_,nhy_,nhz_]) * cell_volume)
            viscous_force_vector = jnp.sum(-momentum_flux, axis=(-1,-2,-3), keepdims=True)
            acceleration = viscous_force_vector/mass_solid

        if is_return_interpolated_temperature:
            return interface_flux, acceleration, solid_heat_flux, primitives_ip, solid_temperature_ip
        else:
            return interface_flux, acceleration, solid_heat_flux
    
    # NOTE wrapper function
    def compute_thermal_interface_state(
            self,
            primitives: Array,
            solid_temperature: Array,
            levelset: Array,
            ) -> Tuple[Array, Array]:
        
        normal = self.geometry_calculator.compute_normal(levelset)
        interface_heat_flux, interface_temperature = compute_thermal_interface_state(
            primitives, solid_temperature, levelset, normal,
            self.levelset_setup.interface_flux, self.material_manager,
            self.solid_properties_manager, self.domain_information)
        
        return interface_heat_flux, interface_temperature