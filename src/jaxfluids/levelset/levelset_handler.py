from functools import partial
from typing import List, Tuple, Union, Dict
import types

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks, compute_cut_cell_mask
from jaxfluids.levelset.helper_functions import linear_filtering
from jaxfluids.levelset.interface_quantity_computer import InterfaceQuantityComputer
from jaxfluids.levelset.interface_flux_computer import InterfaceFluxComputer 
from jaxfluids.levelset.reinitialization.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.levelset.ghost_cell_handler import GhostCellHandler
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.levelset.geometry_calculator import GeometryCalculator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.equation_manager import EquationManager
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.information import LevelsetPositivityInformation, LevelsetResidualInformation
from jaxfluids.levelset.residual_computer import ResidualComputer
from jaxfluids.data_types.information import LevelsetPositivityInformation
from jaxfluids.levelset.mixing.conservative_mixer import ConservativeMixer
from jaxfluids.data_types.case_setup import SolidPropertiesSetup
from jaxfluids.levelset.quantity_extender import QuantityExtender
from jaxfluids.config import precision

class LevelsetHandler():

    """ The LevelsetHandler class manages computations to perform two-phase
    simulations using the levelset method.
    The main functionality includes
        - Transformation of the conservative states from volume-averages
            to actual conserved quantities according to the volume fraction
        - Weighting of the cell face fluxes according to the apertures
        - Computation of the interface fluxes
        - Computation of the levelset advection right-hand-side 
        - LevelsetExtensionSetup of the primitive state from the real fluid cells to the ghost fluid cells
        - Mixing of the integrated conservative states
        - Computation of geometrical quantities, i.e., volume fraction,
            apertures and real fluid/cut cell masks
    """
    
    def __init__(
            self,
            domain_information: DomainInformation,
            numerical_setup: NumericalSetup,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            halo_manager: HaloManager,
            solid_properties: SolidPropertiesSetup
            ) -> None:

        self.eps = precision.get_eps()

        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.halo_manager = halo_manager
        self.domain_information = domain_information
        self.equation_information = equation_manager.equation_information
        self.levelset_setup = numerical_setup.levelset

        levelset_setup = numerical_setup.levelset
        self.geometry_calculator = GeometryCalculator(
                domain_information=domain_information,
                levelset_setup=levelset_setup)

        self.interface_flux_computer = InterfaceFluxComputer(
            domain_information = domain_information,
            material_manager = material_manager,
            numerical_setup = numerical_setup,
            solid_temperature_model = solid_properties.temperature.model)
        
        levelset_setup = numerical_setup.levelset
        extender_interface = QuantityExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            extension_setup = levelset_setup.extension,
            narrowband_setup = levelset_setup.narrowband,
            extension_quantity = "interface")

        extender_primes = QuantityExtender(
            domain_information = domain_information,
            halo_manager = halo_manager,
            extension_setup = levelset_setup.extension,
            narrowband_setup = levelset_setup.narrowband,
            extension_quantity = "primitives")
        
        nh_geometry = levelset_setup.halo_cells
        narrowband_setup = levelset_setup.narrowband
        reinitialization_setup = levelset_setup.reinitialization_runtime
        reinitializer = reinitialization_setup.type
        self.reinitializer: LevelsetReinitializer = reinitializer(
            domain_information = domain_information,
            halo_manager = halo_manager,
            reinitialization_setup = reinitialization_setup,
            halo_cells = nh_geometry,
            narrowband_setup = narrowband_setup)

        self.ghost_cell_handler = GhostCellHandler(
            domain_information = domain_information,
            halo_manager = halo_manager,
            extender_primes = extender_primes,
            equation_manager = equation_manager,
            levelset_setup = levelset_setup)

        self.interface_quantity_computer = InterfaceQuantityComputer(
            domain_information = domain_information,
            material_manager = material_manager,
            solid_properties = solid_properties,
            extender_interface = extender_interface,
            numerical_setup = numerical_setup)
        
        self.residual_computer = ResidualComputer(
            domain_information = domain_information,
            levelset_reinitializer = self.reinitializer,
            extender_primes = extender_primes,
            extender_interface = extender_interface,
            levelset_setup = levelset_setup)

        mixer = levelset_setup.mixing.type
        self.mixer: ConservativeMixer = mixer(
            domain_information = domain_information,
            equation_information = equation_manager.equation_information,
            levelset_setup = levelset_setup,
            halo_manager = halo_manager,
            material_manager = material_manager,
            equation_manager = equation_manager)

        levelset_advection_stencil = levelset_setup.levelset_advection_stencil
        self.levelset_advection_stencil: SpatialDerivative = levelset_advection_stencil(
            nh = domain_information.nh_conservatives,
            inactive_axes = domain_information.inactive_axes)

        # ACTIVE PHYSICAL FLUXES
        self.is_viscous_flux = numerical_setup.active_physics.is_viscous_flux
        self.is_surface_tension = numerical_setup.active_physics.is_surface_tension
        self.is_convective_flux = numerical_setup.active_physics.is_convective_flux

        active_axes_indices = domain_information.active_axes_indices
        index_pairs = [(0,1), (0,2), (1,2)]
        self.index_pairs_mixing = [] 
        for pair in index_pairs:
            if pair[0] in active_axes_indices and pair[1] in active_axes_indices:
                self.index_pairs_mixing.append(pair)

    def compute_volume_fraction_and_apertures(
            self,
            levelset: Array
            ) -> Tuple[Array, Tuple]:
        """Wrapper function for the linear 
        interface reconstruction performed by
        the geometry calculator.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Tuple containing the volume fraction
            buffer and the aperture buffers
        :rtype: Tuple[Array, Tuple]
        """
        volume_fraction, apertures = \
        self.geometry_calculator.interface_reconstruction(levelset)
        return volume_fraction, apertures

    def transform_to_conservatives(
            self,
            conservatives: Array,
            volume_fraction: Array
            ) -> Array:
        """Transforms the volume-averaged conservatives
        to actual conservatives that can be integrated.

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: Array
        :return: Buffer of actual conservative variables
        :rtype: Array
        """
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        levelset_model = self.levelset_setup.model
        if levelset_model == "FLUID-FLUID":
            volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
        conservatives = conservatives.at[...,nhx,nhy,nhz].mul(volume_fraction[...,nhx_,nhy_,nhz_])
        return conservatives

    def weight_cell_face_flux_xi(
            self,
            flux_xi: Array,
            apertures: Array
            ) -> Array:
        """Weights the cell face fluxes according to the apertures.

        :param flux_xi: Cell face flux at xi
        :type flux_xi: Array
        :param apertures: Aperture buffer
        :type apertures: Array
        :return: Weighted cell face flux at xi
        :rtype: Array
        """

        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        levelset_model = self.levelset_setup.model
        if levelset_model == "FLUID-FLUID": 
            apertures = jnp.stack([apertures, 1.0 - apertures], axis=0)
        flux_xi *= apertures[...,nhx_,nhy_,nhz_]
        return flux_xi

    def weight_volume_force(
            self,
            volume_force: Array,
            volume_fraction: Array
            ) -> Array:
        """Weights the volume forces according to the volume fraction.

        :param volume_force: Volume force buffer
        :type volume_force: Array
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: Array
        :return: Weighted volume force
        :rtype: Array
        """
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        levelset_model = self.levelset_setup.model
        if levelset_model == "FLUID-FLUID":
            volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
        volume_force *= volume_fraction[...,nhx_,nhy_,nhz_]
        return volume_force

    def compute_solid_velocity(
            self,
            physical_simulation_time: float
            ) -> Array:
        """Wrapper to compute solid velocity.

        :param physical_simulation_time: Current physical simulation time
        :type physical_simulation_time: float
        :return: Interface velocity buffer
        :rtype: Array
        """
        solid_velocity = self.interface_quantity_computer.compute_solid_velocity(
            physical_simulation_time)
        return solid_velocity

    def compute_solid_temperature(
            self,
            physical_simulation_time: float
            ) -> Array:
        """Wrapper to compute solid temperature.

        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """
        solid_temperature = \
        self.interface_quantity_computer.compute_solid_temperature(
            physical_simulation_time)
        return solid_temperature

    def compute_inviscid_solid_acceleration_xi(
            self,
            primitives: Array,
            volume_fraction: Array,
            apertures: Array,
            axis: int,
            gravity: Array = None
            ) -> Array:
        """Computes the rigid body acceleration vector for
        FLUID-SOLID-DYNAMIC-COUPLED interface interaction.

        :param primitives: Primitive variable buffer
        :type primitives: Array
        :param apertures: Aperture buffer
        :type apertures: Tuple
        :param physical_simulation_time: Current physical simulation time
        :type physical_simulation_time: float
        :return: Rigid body accerlation
        :rtype: Array
        """
        rigid_body_acceleration_xi = \
        self.interface_quantity_computer.compute_inviscid_solid_acceleration_xi(
            primitives, volume_fraction, apertures,  axis, gravity)
        return rigid_body_acceleration_xi


    def compute_viscous_solid_acceleration(
            self,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            apertures: Array,
            friction: Array,
            ) -> Array:
        viscous_rigid_body_acceleration = \
        self.interface_quantity_computer.compute_viscous_solid_acceleration(
            primitives, levelset, volume_fraction, apertures, friction)
        return viscous_rigid_body_acceleration

    def compute_interface_quantities(
            self,
            primitives: Array,
            levelset: Array,
            volume_fraction: Array,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            steps: int = None,
            CFL: float = None,
            ) -> Tuple[Array, Array, float]:
        """Wrapper to compute interface quantities.

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param interface_velocity: _description_, defaults to None
        :type interface_velocity: Array, optional
        :param interface_pressure: _description_, defaults to None
        :type interface_pressure: Array, optional
        :param steps: _description_, defaults to None
        :type steps: int, optional
        :param CFL: _description_, defaults to None
        :type CFL: float, optional
        :return: _description_
        :rtype: Tuple[Array, Array, float]
        """
        normal = self.geometry_calculator.compute_normal(levelset)
        if self.is_surface_tension:
            curvature = self.geometry_calculator.compute_curvature(levelset)
        else:
            curvature = None
        interface_velocity, interface_pressure, step_count = \
        self.interface_quantity_computer.compute_interface_quantities(
            primitives, levelset, volume_fraction, normal, curvature, steps, CFL,
            interface_velocity, interface_pressure)
        
        return interface_velocity, interface_pressure, step_count

    def compute_viscous_solid_force(
            self,
            primitives: Array,
            interface_velocity: Array,
            levelset: Array,
            apertures: Tuple[Array]
            ) -> Array:
        """Wrapper to compute the solid viscous
        interface flux according to Meyer 2010

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param apertures: _description_
        :type apertures: Array
        :return: _description_
        :rtype: Array
        """
        normal = self.geometry_calculator.compute_normal(levelset)
        interface_length = self.geometry_calculator.compute_interface_length(apertures)
        interface_flux = self.interface_flux_computer.compute_viscous_solid_force(
            primitives, interface_velocity, levelset, normal, interface_length)
        return interface_flux

    def compute_interface_flux_xi(
            self,
            primitives: Array,
            levelset: Array,
            interface_velocity: Array,
            interface_pressure: Array,
            volume_fraction: Array,
            apertures: Array,
            axis: int,
            temperature: Array,
            ) -> Array:
        """Computes the interface flux depending on
        the present interface interaction type.

        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param interface_velocity: _description_
        :type interface_velocity: Array
        :param interface_pressure: _description_
        :type interface_pressure: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param apertures: _description_
        :type apertures: Array
        :param axis: _description_
        :type axis: int
        :param temperature: _description_
        :type temperature: Array
        :return: _description_
        :rtype: Array
        """
        normal = self.geometry_calculator.compute_normal(levelset)
        interface_flux_xi = self.interface_flux_computer.compute_interface_flux_xi(
            primitives, interface_velocity, interface_pressure,
            volume_fraction, apertures, normal, axis,
            temperature)
        return interface_flux_xi

    def compute_levelset_advection_rhs_xi(
            self,
            levelset: Array,
            interface_velocity: Array,
            axis: int
            ) -> Array:
        """Computes the right-hand-side of the
        levelset advection equation.

        :param levelset: Levelset buffer
        :type levelset: Array
        :param interface_velocity: Interface velocity buffer
        :type interface_velocity: Array
        :param axis: Current axis
        :type axis: int
        :return: right-hand-side contribution for current axis
        :rtype: Array
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        smallest_cell_size = self.domain_information.smallest_cell_size
        cell_size = self.domain_information.smallest_cell_size

        levelset_model = self.levelset_setup.model
        narrowband_computation = self.levelset_setup.narrowband.computation_width

        if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            interface_velocity = interface_velocity[...,nhx,nhy,nhz]
        elif levelset_model == "FLUID-FLUID":
            interface_velocity = interface_velocity[...,nhx_,nhy_,nhz_]

        # GEOMETRICAL QUANTITIES
        normal = self.geometry_calculator.compute_normal(levelset)
        if levelset_model == "FLUID-FLUID":
            normalized_levelset = jnp.abs(levelset[nhx,nhy,nhz])/smallest_cell_size
            mask_narrowband = jnp.where(normalized_levelset <= narrowband_computation, 1, 0)
        else:
            mask_narrowband = jnp.ones_like(levelset[nhx,nhy,nhz])

        # DERIVATIVE
        derivative_L = self.levelset_advection_stencil.derivative_xi(levelset, 1.0, axis, 0)
        derivative_R = self.levelset_advection_stencil.derivative_xi(levelset, 1.0, axis, 1)

        # UPWINDING
        if levelset_model == "FLUID-FLUID":
            velocity = interface_velocity * normal[axis,nhx_,nhy_,nhz_]
        else:
            velocity = interface_velocity[axis]
        mask_L = jnp.where(velocity >= 0.0, 1.0, 0.0)
        mask_R = 1.0 - mask_L

        # SUM RHS
        rhs_contribution  = - velocity * (mask_L * derivative_L + mask_R * derivative_R) / cell_size
        rhs_contribution *= mask_narrowband

        return rhs_contribution

    def compute_residuals(
            self,
            primitives: Array,
            volume_fraction: Array,
            levelset: Array = None,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            reinitialization_step_count: int = None,
            prime_extension_step_count: int = None,
            interface_extension_step_count: int = None,
            ) -> LevelsetResidualInformation:
        """Wrapper function to compute
        the residuals  of the reinitialization
        and extension procedure.

        :param levelset: _description_
        :type levelset: Array
        :param primitives: _description_
        :type primitives: Array
        :param conservatives: _description_
        :type conservatives: Array
        :return: _description_
        :rtype: Tuple[float]
        """

        normal = self.geometry_calculator.compute_normal(levelset)
        levelset_residuals_info = self.residual_computer.compute_residuals(
            primitives, volume_fraction, levelset, normal,
            interface_velocity, interface_pressure,
            reinitialization_step_count, prime_extension_step_count,
            interface_extension_step_count)
    
        return levelset_residuals_info

    def treat_integrated_levelset(
            self,
            levelset: Array,
            perform_reinitialization: bool,
            is_last_stage: bool
            ) -> Tuple[Array, Array, Array, int]:
        """Treats the integrated levelset field.
        1) Reinitialization (only for last stage)
        2) Halo update and interface recontruction
        3) Remove small structures (only for last stage)

        :param levelset: _description_
        :type levelset: Array
        :param perform_reinitialization: _description_
        :type perform_reinitialization: bool
        :param is_last_stage: _description_
        :type is_last_stage: bool
        :return: _description_
        :rtype: Tuple[Array]
        """
        
        reinitialization_setup = self.levelset_setup.reinitialization_runtime
        remove_underresolved = reinitialization_setup.remove_underresolved
        perform_cutoff = self.levelset_setup.narrowband.perform_cutoff
        
        CFL = reinitialization_setup.CFL
        steps = reinitialization_setup.steps

        levelset = self.halo_manager.perform_halo_update_levelset(
            levelset, True, True)

        if perform_reinitialization and steps > 0 and is_last_stage:
            mask_reinitialize = self.reinitializer.compute_reinitialization_mask(levelset)
            levelset, step_count = self.reinitializer.perform_reinitialization(
                levelset, CFL, steps, mask_reinitialize)
            if perform_cutoff:
                levelset = self.reinitializer.set_levelset_cutoff(levelset)
            levelset = self.halo_manager.perform_halo_update_levelset(
                levelset, True, True, False, False)
        else:
            step_count = 0

        volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(
            levelset)

        if remove_underresolved and is_last_stage:
            levelset = self.reinitializer.remove_underresolved_structures(
                levelset, volume_fraction)
            levelset = self.halo_manager.perform_halo_update_levelset(
                levelset, True, True)
            volume_fraction, apertures = self.geometry_calculator.interface_reconstruction(
                levelset)
            
        return levelset, volume_fraction, apertures, step_count

    def get_reinitialization_flag(
            self,
            simulation_step: jnp.int32
            ) -> bool:
        """Decides whether to perform 
        reinitialization for the current
        simulation step based on the 
        the reinitialization interval.

        :param simulation_step: _description_
        :type simulation_step: jnp.int32
        :return: _description_
        :rtype: bool
        """
        reinitialization_setup = self.levelset_setup.reinitialization_runtime
        interval_reinitialization = reinitialization_setup.interval
        if simulation_step % interval_reinitialization == 0:
            perform_reinitialization = True
        else:
            perform_reinitialization = False
        return perform_reinitialization

    def treat_integrated_material_fields(
            self,
            conservatives: Array,
            primitives: Array,
            levelset: Array,
            volume_fraction_new: Array,
            volume_fraction_old: Array,
            physical_simulation_time: Array,
            solid_velocity: Array = None
            ) -> Tuple[Array, Array,
            LevelsetPositivityInformation]:
        """Treats the integrated material fields.
        1) Halo update for integrated conservatives
        2) Mixing procedure on integrated conservatives
        3) Compute primitives from conservatives
            in real fluid
        4) Halo update for primes in real fluid
        5) Perform ghost cell treatment

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param levelset: _description_
        :type levelset: Array
        :param volume_fraction_new: _description_
        :type volume_fraction_new: Array
        :param volume_fraction_old: _description_
        :type volume_fraction_old: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: Array
        :param solid_velocity: _description_, defaults to None
        :type solid_velocity: Array, optional
        :return: _description_
        :rtype: Tuple[Array]
        """


        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        levelset_model = self.equation_information.levelset_model

        normal = self.geometry_calculator.compute_normal(levelset)
        conservatives = self.halo_manager.perform_halo_update_conservatives_mixing(
            conservatives)
        conservatives, invalid_cells, mixing_invalid_cell_count = self.mixer.perform_mixing(
            conservatives, levelset, normal, volume_fraction_new,
            volume_fraction_old)

        primitives = self.compute_primitives_from_conservatives_in_real_fluid(
            conservatives, primitives, volume_fraction_new)
        primitives = self.halo_manager.perform_halo_update_material(
            primitives, physical_simulation_time, False, False)

        if levelset_model == "FLUID-SOLID-DYNAMIC":
            solid_velocity = self.compute_solid_velocity(physical_simulation_time)
        elif levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            solid_velocity = solid_velocity[...,nhx,nhy,nhz]
        else:
            solid_velocity = None

        conservatives, primitives, extension_invalid_cell_count, extension_step_count = \
        self.ghost_cell_handler.perform_ghost_cell_treatment(
            conservatives, primitives, levelset, volume_fraction_new,
            physical_simulation_time, normal, solid_velocity,
            invalid_cells)

        levelset_positivity_info = LevelsetPositivityInformation(
            mixing_invalid_cell_count,
            extension_invalid_cell_count)

        return conservatives, primitives, levelset_positivity_info, extension_step_count

    def compute_primitives_from_conservatives_in_real_fluid(
            self,
            conservatives: Array,
            primitives: Array,
            volume_fraction: Array,
            ) -> Array:
        """Computes the primitive variables from the
        integrated mixed conservatives within the real fluid.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param invalid_cells: _description_
        :type invalid_cells: Array
        :return: _description_
        :rtype: Array
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry

        levelset_model = self.levelset_setup.model

        mask_real = compute_fluid_masks(volume_fraction, levelset_model)
        mask_real = mask_real[...,nhx_,nhy_,nhz_]
        mask_ghost = 1 - mask_real

        primes_in_real = self.equation_manager.get_primitives_from_conservatives(
            conservatives[...,nhx,nhy,nhz]) 
        primes_in_real *= mask_real 

        primitives = primitives.at[...,nhx,nhy,nhz].mul(mask_ghost)
        primitives = primitives.at[...,nhx,nhy,nhz].add(primes_in_real)

        return primitives