from functools import partial
from typing import Dict, List, Union, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.buffers import ForcingBuffers, IntegrationBuffers
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler
from jaxfluids.solvers.source_term_solver import SourceTermSolver
from jaxfluids.config import precision

class SpaceSolver:
    """The Space Solver class manages the calculation of
    the righ-hand-side (i.e., fluxes) of the NSE
    and, for two-phase simulations, manages the calculation
    of the rhs of the level-set advection.

    Depending on the numerical setup, the calculation
    of the fluxes for the NSE has contributions from:
    1) Convective flux
    2) Viscous flux
    3) Heat flux
    4) Interface exchange flux
    5) Volume force flux
    6) External forcing 
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            halo_manager: HaloManager,
            numerical_setup: NumericalSetup,
            gravity: Array,
            geometric_source: Union[Dict, None],
            levelset_handler: LevelsetHandler = None,
            diffuse_interface_handler: DiffuseInterfaceHandler = None,
            positivity_handler: PositivityHandler = None
            ) -> None:
        
        self.eps = precision.get_eps()

        convective_fluxes_setup = numerical_setup.conservatives.convective_fluxes
        positivity_setup = numerical_setup.conservatives.positivity
        diffuse_interface_setup = numerical_setup.diffuse_interface

        convective_solver = convective_fluxes_setup.convective_solver
        diffuse_interface_setup = numerical_setup.diffuse_interface
        self.convective_flux_solver: ConvectiveFluxSolver = convective_solver(
            convective_fluxes_setup=convective_fluxes_setup,
            positivity_setup=positivity_setup,
            diffuse_interface_setup=diffuse_interface_setup,
            material_manager=material_manager,
            domain_information=domain_information,
            equation_manager=equation_manager,
            diffuse_interface_handler=diffuse_interface_handler,
            positivity_handler=positivity_handler)

        self.gravity = gravity
        self.source_term_solver = SourceTermSolver(
            numerical_setup=numerical_setup,
            material_manager=material_manager,
            equation_manager=equation_manager,
            gravity=gravity,
            geometric_source=geometric_source,
            domain_information=domain_information)

        self.levelset_handler = levelset_handler
        self.diffuse_interface_handler = diffuse_interface_handler
        self.positivity_handler = positivity_handler
        self.is_flux_limiter = not positivity_setup.flux_limiter in (None, False)
        self.is_interpolation_limiter = positivity_setup.is_interpolation_limiter
        self.is_thinc_interpolation_limiter = positivity_setup.is_thinc_interpolation_limiter
        self.is_thinc_reconstruction = numerical_setup.diffuse_interface.thinc.is_thinc_reconstruction
        self.is_acdi_flux_limiter = positivity_setup.is_acdi_flux_limiter
        
        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.halo_manager = halo_manager
        self.numerical_setup = numerical_setup
        self.domain_information = domain_information

        self.active_forcings = jnp.array(
            [getattr(numerical_setup.active_forcings, forcing) for forcing in numerical_setup.active_forcings._fields]
            ).any() 

        self.dim = domain_information.dim
        self.active_axes_indices = domain_information.active_axes_indices
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nhx__, self.nhy__, self.nhz__ = \
            domain_information.domain_slices_conservatives_to_geometry
        self.is_parallel = domain_information.is_parallel
        self.split_factors = domain_information.split_factors
        self.is_mesh_stretching = domain_information.is_mesh_stretching

        self.equation_type = self.equation_information.equation_type
        self.mass_ids = self.equation_information.mass_ids
        self.mass_slices = self.equation_information.mass_slices
        self.vel_ids = self.equation_information.velocity_ids
        self.vel_slices = self.equation_information.velocity_slices
        self.energy_ids = self.equation_information.energy_ids
        self.energy_slices = self.equation_information.energy_slices
        self.vf_ids = self.equation_information.vf_ids
        self.vf_slices = self.equation_information.vf_slices
        self.species_ids = self.equation_information.species_ids
        self.species_slices = self.equation_information.species_slices
        self.momentum_and_energy_slices = self.equation_information.momentum_and_energy_slices

        self.flux_slices = [
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

        shape_equations = self.equation_information.shape_equations
        nx_device, ny_device, nz_device = domain_information.device_number_of_cells
        self.flux_shapes = [
            shape_equations + (nx_device + 1, ny_device, nz_device),
            shape_equations + (nx_device, ny_device + 1, nz_device),
            shape_equations + (nx_device, ny_device, nz_device + 1),
        ]

        self.is_double_precision = numerical_setup.precision.is_double_precision_compute
        self.dtype = jnp.float64 if self.is_double_precision else jnp.float32

    def compute_rhs(
            self,
            conservatives: Array,
            primitives: Array,
            physical_simulation_time: float,
            physical_timestep_size: float,
            levelset: Array = None,
            volume_fraction: Array = None,
            apertures: Tuple = None,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            forcing_buffers: ForcingBuffers = None,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            is_feedforward: bool = False
            ) -> Tuple[IntegrationBuffers, jnp.float32, jnp.int32, jnp.int32]:
        """Computes the right-hand-side of the Navier-Stokes equations
        depending  on active physics and active axis. For levelset
        simulations with FLUID-FLUID or FLUID-SOLID-DYNAMIC
        interface interactions, also computes the right-hand-side
        of the levelset advection.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param physical_timestep_size: _description_
        :type physical_timestep_size: float
        :param levelset: _description_, defaults to None
        :type levelset: Array, optional
        :param volume_fraction: _description_, defaults to None
        :type volume_fraction: Array, optional
        :param apertures: _description_, defaults to None
        :type apertures: Tuple, optional
        :param interface_velocity: _description_, defaults to None
        :type interface_velocity: Array, optional
        :param interface_pressure: _description_, defaults to None
        :type interface_pressure: Array, optional
        :param forcing_buffers: _description_, defaults to None
        :type forcing_buffers: ForcingBuffers, optional
        :param ml_parameters_dict: _description_, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: _description_, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[IntegrationBuffers, jnp.float32, jnp.int32, jnp.int32]
        """

        is_convective_flux = self.numerical_setup.active_physics.is_convective_flux
        is_viscous_flux = self.numerical_setup.active_physics.is_viscous_flux
        is_heat_flux = self.numerical_setup.active_physics.is_heat_flux
        is_volume_force = self.numerical_setup.active_physics.is_volume_force
        is_geometric_source = self.numerical_setup.active_physics.is_geometric_source
        is_surface_tension = self.numerical_setup.active_physics.is_surface_tension
        is_solid_levelset = self.equation_information.is_solid_levelset
        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset

        is_interface_regularization = self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening

        # COMPUTE FLUID TEMPERATURE
        if is_viscous_flux or is_heat_flux:
            temperature = self.material_manager.get_temperature(primitives)
                 
            if is_solid_levelset and is_heat_flux:
                interface_quantity_computer = self.levelset_handler.interface_quantity_computer
                solid_temperature_model = interface_quantity_computer.solid_temperature.model
                if solid_temperature_model == "CUSTOM":
                    solid_temperature = self.levelset_handler.compute_solid_temperature(
                        physical_simulation_time)
                    mask_real = compute_fluid_masks(
                        volume_fraction[self.nhx_,self.nhy_,self.nhz_],
                        self.equation_information.levelset_model)
                    solid_temperature *= (1 - mask_real)
                    temperature = temperature.at[self.nhx,self.nhy,self.nhz].mul(mask_real)
                    temperature = temperature.at[self.nhx,self.nhy,self.nhz].add(
                        solid_temperature)
                    if self.is_parallel:
                        temperature = self.halo_manager.perform_inner_halo_update_material(
                            temperature)
                
            temperature = self.halo_manager.perform_outer_halo_update_temperature(
                temperature, physical_simulation_time)

        else:
            temperature = None
        
        # VOLUME FRACTION EDGE HALOS
        if self.equation_type in ("DIFFUSE-INTERFACE-5EQM"):
            is_thinc = self.numerical_setup.diffuse_interface.thinc.is_thinc_reconstruction
            is_diffusion_sharpening = self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening
            is_interface_compression = self.numerical_setup.diffuse_interface.interface_compression.is_interface_compression
            flag = any((is_surface_tension, is_thinc, is_diffusion_sharpening, is_interface_compression))
            #TODO @aaron this seems a bit messy???
            if flag:
                volume_fraction = primitives[self.vf_slices]
                volume_fraction = self.halo_manager.perform_halo_update_material(
                    volume_fraction, physical_simulation_time, True, False, None, False)
                # TODO conservatives not updated??
                primitives = primitives.at[self.vf_slices].set(volume_fraction)
                conservatives = conservatives.at[self.vf_slices].set(volume_fraction)

        # LEVEL-SET SPECIFIC
        if levelset_model == "FLUID-SOLID-DYNAMIC":
            interface_velocity = self.levelset_handler.compute_solid_velocity(
                physical_simulation_time)

        # DIFFUSE-INTERFACE SPECIFIC
        # TODO deniz
        # Compute curvature
        # TODO deniz: also compute normal here?
        curvature = None
        if self.equation_information.diffuse_interface_model == "5EQM" \
            and is_surface_tension:                        
            curvature = self.diffuse_interface_handler.compute_curvature(
                primitives[self.vf_ids])

        # INITIALIZE RHS VARIABLES
        rhs_conservatives = 0.0
        rhs_levelset = 0.0 if is_moving_levelset else None
        rhs_solid_interface_velocity = 0.0 if self.equation_information.levelset_model \
            == "FLUID-SOLID-DYNAMIC-COUPLED" else None

        # INITIALIZE POSITIVITY COUNTERS
        positivity_count_flux = 0 if self.is_flux_limiter else None
        positivity_count_interpolation = 0 if self.is_interpolation_limiter else None
        positivity_count_thinc = 0 if self.is_thinc_interpolation_limiter and \
            self.is_thinc_reconstruction else None
        positivity_count_acdi = 0 if self.is_acdi_flux_limiter and \
            self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening else None
        count_acdi = 0 if self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening \
            else None
        thinc_count = 0 if self.is_thinc_reconstruction else None

        one_cell_size = self.domain_information.get_device_one_cell_sizes()
        # CELL FACE FLUX
        for axis in self.active_axes_indices:
            flux_xi = jnp.zeros(self.flux_shapes[axis], dtype=self.dtype)
            one_cell_size_xi = one_cell_size[axis]

            # CONVECTIVE CONTRIBUTION
            if is_convective_flux:
                flux_xi_convective, u_hat_xi, alpha_hat_xi, \
                positivity_count_interpolation_xi, \
                positivity_count_thinc_xi \
                = self.convective_flux_solver.compute_flux_xi(
                    primitives, conservatives, axis,
                    curvature=curvature,
                    apertures=apertures,
                    ml_parameters_dict=ml_parameters_dict,
                    ml_networks_dict=ml_networks_dict)
                if self.is_interpolation_limiter \
                    and positivity_count_interpolation_xi is not None:
                    positivity_count_interpolation += positivity_count_interpolation_xi
                if self.is_thinc_interpolation_limiter \
                    and positivity_count_thinc_xi is not None:
                    positivity_count_thinc += positivity_count_thinc_xi

                # POSITIVITY FIX
                if self.is_flux_limiter:
                    # TODO INCLUDE INTERFACE FLUX LEVELSET
                    flux_xi_convective, u_hat_xi, alpha_hat_xi, positivity_count_xi \
                    = self.positivity_handler.compute_positivity_preserving_flux(
                        flux_xi_convective, u_hat_xi, alpha_hat_xi,
                        primitives, conservatives, levelset, volume_fraction,
                        apertures[axis] if apertures is not None else None, 
                        curvature, physical_timestep_size,
                        axis, ml_parameters_dict, ml_networks_dict)
                    positivity_count_flux += positivity_count_xi
                flux_xi += flux_xi_convective

            # DIFFUSE INTERFACE REGULARIZATION
            if is_interface_regularization:
                # TODO numerical dissipation
                interface_regularization_flux_xi, count_acdi_xi \
                    = self.diffuse_interface_handler.compute_diffusion_sharpening_flux_xi(
                        conservatives, primitives, axis,
                        numerical_dissipation=alpha_hat_xi)
                if count_acdi_xi is not None:
                    count_acdi += count_acdi_xi
                if self.is_acdi_flux_limiter:
                    interface_regularization_flux_xi, positivity_count_acdi_xi = \
                        self.positivity_handler.compute_positivity_preserving_sharpening_flux(
                        flux_xi_convective, u_hat_xi, interface_regularization_flux_xi,
                        primitives, conservatives, physical_timestep_size, axis)
                    positivity_count_acdi += positivity_count_acdi_xi

                flux_xi -= interface_regularization_flux_xi

            # VISCOUS CONTRIBUTION
            if is_viscous_flux:
                viscous_flux_xi = self.source_term_solver.compute_viscous_flux_xi(
                    primitives, temperature, axis)
                flux_xi = flux_xi.at[self.momentum_and_energy_slices].add(-viscous_flux_xi)

            # HEAT CONTRIBUTION
            if is_heat_flux:
                heat_flux_xi = self.source_term_solver.compute_heat_flux_xi(
                    temperature, primitives, axis)
                flux_xi = flux_xi.at[self.energy_ids].add(heat_flux_xi)

            # WEIGHT FLUXES
            if levelset_model:
                flux_xi = self.levelset_handler.weight_cell_face_flux_xi(flux_xi, apertures[axis])

            # SUM RIGHT HAND SIDE
            if any((is_convective_flux, is_viscous_flux, is_heat_flux)):
                rhs_conservatives += one_cell_size_xi * (
                    flux_xi[self.flux_slices[axis][1]] - flux_xi[self.flux_slices[axis][0]])

            # DIFFUSE INTERFACE MODEL
            # DIVERGENCE CONTRIBUTION VOLUME FRACTION
            if is_convective_flux \
                and self.equation_type == "DIFFUSE-INTERFACE-5EQM":
                rhs_volume_fraction_contribution = \
                    self.diffuse_interface_handler.compute_volume_fraction_source_term(
                        u_hat_xi, primitives[self.vf_slices],
                        one_cell_size_xi, axis)
                rhs_conservatives = rhs_conservatives.at[self.vf_slices].add(
                    -rhs_volume_fraction_contribution)

            # SURFACE TENSION TERM
            if is_convective_flux and is_surface_tension \
                and self.equation_type == "DIFFUSE-INTERFACE-5EQM":
                rhs_momentum_contribution, rhs_energy_contribution = \
                    self.diffuse_interface_handler.compute_surface_tension_source_term_xi(
                        u_hat_xi, alpha_hat_xi, primitives[self.vf_ids],
                        curvature, one_cell_size_xi, axis)
                rhs_conservatives = rhs_conservatives.at[self.vel_ids[axis]].add(
                    -rhs_momentum_contribution)
                rhs_conservatives = rhs_conservatives.at[self.energy_ids].add(
                    -rhs_energy_contribution)

            # INTERFACE FLUXES
            if levelset_model:
                interface_flux_xi = self.levelset_handler.compute_interface_flux_xi(
                    primitives, levelset, interface_velocity, interface_pressure,
                    volume_fraction, apertures[axis], axis, temperature)
                rhs_conservatives += one_cell_size_xi * interface_flux_xi

            # LEVELSET ADVECTION
            if is_moving_levelset:
                levelset_advection_rhs_contribution_xi = \
                    self.levelset_handler.compute_levelset_advection_rhs_xi(
                        levelset, interface_velocity, axis)
                rhs_levelset += levelset_advection_rhs_contribution_xi

            # RIGID BODY ACCELERATION
            if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
                inviscid_acceleration_xi = \
                    self.levelset_handler.compute_inviscid_solid_acceleration_xi(
                        primitives, volume_fraction, apertures[axis], axis,
                        self.gravity)
                rhs_solid_interface_velocity += inviscid_acceleration_xi

        # INTERFACE FRICTION MEYER
        viscous_flux_method = self.numerical_setup.levelset.interface_flux.viscous_flux_method
        if is_solid_levelset and is_viscous_flux and viscous_flux_method == "MEYER":
            interface_friction = self.levelset_handler.compute_viscous_solid_force(
                primitives, interface_velocity, levelset, apertures)
            dV = self.domain_information.get_device_cell_volume()
            rhs_conservatives = rhs_conservatives.at[self.vel_slices].add(
                interface_friction / dV)
            if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
                viscous_solid_acceleration = self.levelset_handler.compute_viscous_solid_acceleration(
                    primitives, levelset, volume_fraction, apertures, interface_friction)
                rhs_solid_interface_velocity += viscous_solid_acceleration

        # VOLUME FORCES
        if is_volume_force:
            volume_forces = self.source_term_solver.compute_gravity_forces(conservatives)
            if self.equation_information.levelset_model:
                volume_forces = self.levelset_handler.weight_volume_force(
                    volume_forces, volume_fraction)
            rhs_conservatives = rhs_conservatives.at[self.momentum_and_energy_slices].add(volume_forces)

        # GEOMETRIC SOURCE TERMS
        if is_geometric_source:
            geometric_source_terms = self.source_term_solver.compute_geometric_source_terms(
                conservatives, primitives, temperature)
            rhs_conservatives += geometric_source_terms

        # FORCINGS
        if forcing_buffers is not None:
            for field in forcing_buffers._fields:
                force = getattr(forcing_buffers, field)
                if force is not None:
                    if self.equation_information.levelset_model:
                        force = self.levelset_handler.weight_volume_force(force, volume_fraction)
                    rhs_conservatives += force

        # SUM POSITIVITY COUNTS PARALLEL 
        is_logging = self.numerical_setup.conservatives.positivity.is_logging
        if is_logging and not is_feedforward:
            if self.is_parallel and self.is_flux_limiter:
                positivity_count_flux = jax.lax.psum(positivity_count_flux, axis_name="i")
            if self.is_parallel and self.is_interpolation_limiter:
                # TODO check when interpolation limiter counter is None
                positivity_count_interpolation = jax.lax.psum(positivity_count_interpolation, axis_name="i")
            if self.is_parallel and self.is_thinc_interpolation_limiter and self.is_thinc_reconstruction:
                # TODO check when interpolation limiter counter is None
                positivity_count_thinc = jax.lax.psum(positivity_count_thinc, axis_name="i")
            if self.is_parallel and self.is_acdi_flux_limiter and \
                self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening:
                positivity_count_acdi = jax.lax.psum(positivity_count_acdi, axis_name="i")
            if self.is_parallel and self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening:
                count_acdi = jax.lax.psum(count_acdi, axis_name="i")

        rhs_buffers = IntegrationBuffers(
            rhs_conservatives, rhs_levelset,
            rhs_solid_interface_velocity)

        return rhs_buffers, \
            positivity_count_flux, \
            positivity_count_interpolation, \
            positivity_count_thinc, \
            positivity_count_acdi, \
            count_acdi
