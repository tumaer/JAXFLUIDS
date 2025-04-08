from functools import partial
from typing import Dict, List, Union, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.data_types.buffers import ForcingBuffers, IntegrationBuffers, EulerIntegrationBuffers
from jaxfluids.data_types.information import PositivityCounter, DiscretizationCounter
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.levelset.helper_functions import (
    weight_cell_face_flux_xi, weight_volume_force)
from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.math.sum_consistent import sum3_consistent
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler
from jaxfluids.solvers.source_term_solver import SourceTermSolver
from jaxfluids.config import precision
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField

Array = jax.Array

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
        self.ids_mass = self.equation_information.ids_mass
        self.s_mass = self.equation_information.s_mass
        self.vel_ids = self.equation_information.ids_velocity
        self.vel_slices = self.equation_information.s_velocity
        self.ids_energy = self.equation_information.ids_energy
        self.s_energy = self.equation_information.s_energy
        self.ids_volume_fraction = self.equation_information.ids_volume_fraction
        self.s_volume_fraction = self.equation_information.s_volume_fraction
        self.ids_species = self.equation_information.ids_species
        self.s_species = self.equation_information.s_species
        self.s_momentum_and_energy = self.equation_information.s_momentum_and_energy

        self.flux_slices = [
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

        shape_equations = self.equation_information.shape_equations
        nx_device, ny_device, nz_device = domain_information.device_number_of_cells
        self.rhs_shapes = [
            shape_equations + (nx_device, ny_device, nz_device),
            shape_equations + (nx_device, ny_device, nz_device),
            shape_equations + (nx_device, ny_device, nz_device),
        ]
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
            temperature: Array,
            physical_simulation_time: float,
            physical_timestep_size: float,
            levelset: Array = None,
            volume_fraction: Array = None,
            apertures: Tuple = None,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            solid_velocity: Array = None,
            solid_temperature: Array = None,
            interface_cells: LevelsetSolidCellIndicesField = None,
            forcing_buffers: ForcingBuffers = None,
            ml_setup: MachineLearningSetup = None,
            is_feed_forward: bool = False
        ) -> Tuple[IntegrationBuffers, PositivityCounter, DiscretizationCounter]:
        """Computes the right-hand-side of the Navier-Stokes equations
        depending  on active physics and active axis.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param temperature: _description_
        :type temperature: Array
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
        :param solid_velocity: _description_, defaults to None
        :type solid_velocity: Array, optional
        :param solid_temperature: _description_, defaults to None
        :type solid_temperature: Array, optional
        :param interface_cells: _description_, defaults to None
        :type interface_cells: LevelsetSolidCellIndicesField, optional
        :param forcing_buffers: _description_, defaults to None
        :type forcing_buffers: ForcingBuffers, optional
        :param ml_setup: _description_, defaults to None
        :type ml_setup: MachineLearningSetup, optional
        :param is_feed_forward: _description_, defaults to False
        :type is_feed_forward: bool, optional
        :return: _description_
        :rtype: Tuple[IntegrationBuffers, PositivityCounter, DiscretizationCounter]
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
        solid_coupling = self.equation_information.solid_coupling
        is_interface_regularization = self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening

        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry

        # VOLUME FRACTION EDGE HALOS
        if self.equation_type in ("DIFFUSE-INTERFACE-4EQM", "DIFFUSE-INTERFACE-5EQM"):
            is_thinc = self.numerical_setup.diffuse_interface.thinc.is_thinc_reconstruction
            is_diffusion_sharpening = self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening
            is_interface_compression = self.numerical_setup.diffuse_interface.interface_compression.is_interface_compression
            flag = any((is_surface_tension, is_thinc, is_diffusion_sharpening, is_interface_compression))
            #TODO @aaron this seems a bit messy???
            # TODO conservatives not updated??
            if flag:
                volume_fraction = primitives[self.s_volume_fraction]
                volume_fraction = self.halo_manager.perform_halo_update_material(
                    volume_fraction, physical_simulation_time, True, True, None, False,
                    ml_setup=ml_setup)
                primitives = primitives.at[self.s_volume_fraction].set(volume_fraction)
                conservatives = conservatives.at[self.s_volume_fraction].set(volume_fraction)

        # DIFFUSE INTERFACE 4-EQUATION MODEL SPECIFIC
        # TODO
        if self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            volume_fraction = self.material_manager.diffuse_4eqm_mixture.get_volume_fractions_from_pressure_temperature(
                primitives[self.s_mass], primitives[self.ids_energy], temperature)

        # DIFFUSE-INTERFACE SPECIFIC
        # Compute curvature
        # TODO deniz: also compute normal here?
        if is_surface_tension and self.equation_information.diffuse_interface_model == "5EQM":
            curvature = self.diffuse_interface_handler.compute_curvature(primitives[self.ids_volume_fraction])
        else:
            curvature = None

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

        # INITIALIZE RHS VARIABLES
        rhs_conservatives = 0.0

        if is_moving_levelset:
            rhs_levelset = 0.0
        else:
            rhs_levelset = None

        if solid_coupling.dynamic == "TWO-WAY":
            rhs_solid_velocity = 0.0
        else:
            rhs_solid_velocity = None

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        else:
            rhs_solid_energy = None

        if precision.is_consistent_summation and self.domain_information.dim == 3:
            rhs_conservatives_list = []
            rhs_levelset_list = []
            rhs_solid_velocity_list = []
            rhs_solid_energy_list = []

        for axis in self.active_axes_indices:
            rhs_buffers_euler_xi, positivity_counter_xi, discretization_counter_xi = \
            self.compute_rhs_xi(
                conservatives, primitives, temperature, axis,
                physical_simulation_time, physical_timestep_size,
                levelset, volume_fraction, apertures,
                interface_velocity, interface_pressure,
                solid_velocity, solid_temperature,
                curvature,
                ml_setup)

            if precision.is_consistent_summation and self.domain_information.dim == 3:
                # NOTE consistent summation of the directional fluxes
                rhs_conservatives_list.append(rhs_buffers_euler_xi.conservatives)

                if is_moving_levelset:
                    rhs_levelset_list.append(rhs_buffers_euler_xi.levelset)

                if solid_coupling.dynamic == "TWO-WAY":
                    rhs_solid_velocity_list.append(rhs_buffers_euler_xi.solid_velocity)
            
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError

            else:
                rhs_conservatives += rhs_buffers_euler_xi.conservatives

                if is_moving_levelset:
                    rhs_levelset += rhs_buffers_euler_xi.levelset
                
                if solid_coupling.dynamic == "TWO-WAY":
                    rhs_solid_velocity += rhs_buffers_euler_xi.solid_velocity
                        
                if solid_coupling.thermal == "TWO-WAY":
                    raise NotImplementedError


            if self.is_interpolation_limiter and positivity_counter_xi.interpolation_limiter is not None:
                positivity_count_interpolation += positivity_counter_xi.interpolation_limiter

            if self.is_thinc_interpolation_limiter and positivity_counter_xi.thinc_limiter is not None:
                positivity_count_thinc += positivity_counter_xi.thinc_limiter

            if self.is_flux_limiter and positivity_counter_xi.flux_limiter is not None:
                positivity_count_flux += positivity_counter_xi.flux_limiter

            if is_interface_regularization:
                if discretization_counter_xi.acdi is not None:
                    count_acdi += discretization_counter_xi.acdi

                if self.is_acdi_flux_limiter and positivity_counter_xi.acdi_limiter is not None:
                    positivity_count_acdi += positivity_counter_xi.acdi_limiter


        if precision.is_consistent_summation and self.domain_information.dim == 3:
            rhs_conservatives = sum3_consistent(*rhs_conservatives_list)

            if is_moving_levelset:
                rhs_levelset = sum3_consistent(*rhs_levelset_list)
            
            if solid_coupling.dynamic == "TWO-WAY":
                rhs_solid_velocity = sum3_consistent(*rhs_solid_velocity_list)


        # INTERFACE FLUXES
        if levelset_model == "FLUID-FLUID":
            fluid_fluid_handler = self.levelset_handler.fluid_fluid_handler
            interface_flux = fluid_fluid_handler.compute_interface_flux(
                primitives, levelset, interface_velocity,
                apertures, temperature)
            dV = self.domain_information.get_device_cell_volume()
            rhs_conservatives += interface_flux/dV
        elif is_solid_levelset:
            fluid_solid_handler = self.levelset_handler.fluid_solid_handler
            interface_flux, solid_acceleration, heat_flux_solid \
            = fluid_solid_handler.compute_interface_flux(
                primitives, solid_velocity, solid_temperature,
                levelset, volume_fraction, apertures,
                physical_simulation_time, interface_cells,
                ml_setup=ml_setup)

            dV = self.domain_information.get_device_cell_volume()
            rhs_conservatives += interface_flux/dV
            if solid_coupling.dynamic == "TWO-WAY":
                rhs_solid_velocity += solid_acceleration
            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

        # VOLUME FORCES
        if is_volume_force:
            volume_forces = self.source_term_solver.compute_gravity_forces(conservatives)
            if self.equation_information.levelset_model:
                volume_forces = weight_volume_force(
                    volume_forces, volume_fraction, 
                    self.domain_information, levelset_model)
            rhs_conservatives = rhs_conservatives.at[self.s_momentum_and_energy].add(volume_forces)


        # GEOMETRIC SOURCE TERMS
        if is_geometric_source:
            geometric_source_terms = self.source_term_solver.compute_geometric_source_terms(
                conservatives, primitives, temperature)
            rhs_conservatives += geometric_source_terms


        # FORCINGS
        active_forcings = self.equation_information.active_forcings
        if any(active_forcings._asdict().values()) and forcing_buffers is not None:
            for field in forcing_buffers._fields:
                force = getattr(forcing_buffers, field)
                if force is not None:
                    if field == "solid_temperature_force":
                        volume_fraction_solid = 1.0 - volume_fraction[nhx_,nhy_,nhz_]
                        force *= volume_fraction_solid
                        rhs_solid_energy += force
                    else:
                        if self.equation_information.levelset_model:
                            force = weight_volume_force(
                                force, volume_fraction,
                                self.domain_information,
                                levelset_model)
                        rhs_conservatives += force


        # SUM POSITIVITY COUNTS PARALLEL 
        is_logging = self.numerical_setup.output.logging.is_positivity
        if is_logging and not is_feed_forward:
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

        rhs_euler_buffers = EulerIntegrationBuffers(
            rhs_conservatives,
            rhs_levelset,
            rhs_solid_velocity,
            rhs_solid_energy
        )

        rhs_buffers = IntegrationBuffers(
            rhs_euler_buffers,
        )

        positivity_counter = PositivityCounter(
            positivity_count_interpolation,
            positivity_count_thinc,
            positivity_count_flux,
            positivity_count_acdi,
            None
        )

        discretization_counter = DiscretizationCounter(count_acdi, None)

        # TODO CALLBACK on_rhs_axis

        return rhs_buffers, positivity_counter, discretization_counter


    def compute_rhs_xi(
            self, 
            conservatives: Array,
            primitives: Array,
            temperature: Array,
            axis: int,
            physical_simulation_time: float,
            physical_timestep_size: float,
            levelset: Array = None,
            volume_fraction: Array = None,
            apertures: Tuple = None,
            interface_velocity: Array = None,
            interface_pressure: Array = None,
            solid_velocity: Array = None,
            solid_temperature: Array = None,
            curvature: Array = None,
            ml_setup: MachineLearningSetup = None,
        ) -> Tuple[EulerIntegrationBuffers, PositivityCounter, DiscretizationCounter]:

        is_convective_flux = self.numerical_setup.active_physics.is_convective_flux
        is_viscous_flux = self.numerical_setup.active_physics.is_viscous_flux
        is_heat_flux = self.numerical_setup.active_physics.is_heat_flux
        is_volume_force = self.numerical_setup.active_physics.is_volume_force
        is_geometric_source = self.numerical_setup.active_physics.is_geometric_source
        is_surface_tension = self.numerical_setup.active_physics.is_surface_tension
        is_solid_levelset = self.equation_information.is_solid_levelset
        levelset_model = self.equation_information.levelset_model
        is_moving_levelset = self.equation_information.is_moving_levelset
        solid_coupling = self.equation_information.solid_coupling
        is_interface_regularization = self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening

        
        # rhs_conservatives_xi = 0.0
        rhs_conservatives_xi = jnp.zeros(self.rhs_shapes[axis], dtype=self.dtype)

        if is_moving_levelset:
            rhs_levelset_xi = 0.0
        else:
            rhs_levelset_xi = None

        if solid_coupling.dynamic == "TWO-WAY":
            rhs_solid_velocity_xi = 0.0
        else:
            rhs_solid_velocity_xi = None

        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        else:
            rhs_solid_energy_xi = None

        positivity_count_interpolation_xi = None
        positivity_count_thinc_xi = None
        positivity_count_flux_xi = None
        positivity_count_acdi_xi = None

        count_thinc_xi = None
        count_acdi_xi = None

        one_cell_size = self.domain_information.get_device_one_cell_sizes()
        one_cell_size_xi = one_cell_size[axis]

        flux_xi = jnp.zeros(self.flux_shapes[axis], dtype=self.dtype)

        # CONVECTIVE CONTRIBUTION
        if is_convective_flux:
            flux_xi_convective, u_hat_xi, alpha_hat_xi, \
            positivity_count_interpolation_xi, positivity_count_thinc_xi \
            = self.convective_flux_solver.compute_flux_xi(
                primitives, conservatives, axis,
                curvature=curvature,
                volume_fraction=volume_fraction[0] if self.equation_type == "DIFFUSE-INTERFACE-4EQM" else None,
                apertures=apertures,
                ml_setup=ml_setup
            )

            # POSITIVITY FIX
            if self.is_flux_limiter:
                # TODO INCLUDE INTERFACE FLUX LEVELSET
                flux_xi_convective, u_hat_xi, alpha_hat_xi, positivity_count_flux_xi \
                = self.positivity_handler.compute_positivity_preserving_flux(
                    flux_xi_convective, u_hat_xi, alpha_hat_xi,
                    primitives, conservatives, temperature, levelset, volume_fraction,
                    apertures[axis] if apertures is not None else None, 
                    curvature, physical_timestep_size,
                    axis, ml_setup
                )

            flux_xi += flux_xi_convective


        # DIFFUSE INTERFACE REGULARIZATION
        if is_interface_regularization:
            # TODO numerical dissipation
            interface_regularization_flux_xi, count_acdi_xi \
                = self.diffuse_interface_handler.compute_diffusion_sharpening_flux_xi(
                    conservatives, primitives, axis,
                    volume_fraction=volume_fraction[0] if self.equation_type == "DIFFUSE-INTERFACE-4EQM" else None,
                    numerical_dissipation=alpha_hat_xi)

            if self.is_acdi_flux_limiter:
                interface_regularization_flux_xi, positivity_count_acdi_xi = \
                    self.positivity_handler.compute_positivity_preserving_sharpening_flux(
                    flux_xi_convective, u_hat_xi, interface_regularization_flux_xi,
                    primitives, conservatives, physical_timestep_size, axis)

            flux_xi -= interface_regularization_flux_xi


        # VISCOUS CONTRIBUTION
        if is_viscous_flux:
            if self.numerical_setup.conservatives.dissipative_fluxes.is_laplacian:
                viscous_term_xi = self.source_term_solver.compute_viscous_term_xi(
                    primitives, temperature, axis)
                rhs_conservatives_xi = rhs_conservatives_xi.at[self.s_momentum_and_energy].add(viscous_term_xi)
            else:
                viscous_flux_xi = self.source_term_solver.compute_viscous_flux_xi(
                    primitives, temperature, axis)
                flux_xi = flux_xi.at[self.s_momentum_and_energy].add(-viscous_flux_xi)


        # HEAT CONTRIBUTION
        if is_heat_flux:
            if self.numerical_setup.conservatives.dissipative_fluxes.is_laplacian:
                heat_term_xi = self.source_term_solver.compute_heat_term_xi(
                    temperature, primitives, axis)
                rhs_conservatives_xi = rhs_conservatives_xi.at[self.ids_energy].add(heat_term_xi)
            else:
                heat_flux_xi = self.source_term_solver.compute_heat_flux_xi(
                    temperature, primitives, axis)
                flux_xi = flux_xi.at[self.ids_energy].add(heat_flux_xi)


        # WEIGHT FLUXES
        if levelset_model:
            flux_xi = weight_cell_face_flux_xi(
                flux_xi, apertures[axis],
                self.domain_information,
                levelset_model)


        # SUM RIGHT HAND SIDE
        if any((is_convective_flux, is_viscous_flux, is_heat_flux)):
            rhs_conservatives_xi += one_cell_size_xi * (
                flux_xi[self.flux_slices[axis][1]] - flux_xi[self.flux_slices[axis][0]])


        # DIFFUSE INTERFACE MODEL
        # DIVERGENCE CONTRIBUTION VOLUME FRACTION
        if is_convective_flux \
            and self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            # TODO deniz: add Kapila term
            rhs_volume_fraction_contribution = \
                self.diffuse_interface_handler.compute_volume_fraction_source_term(
                    u_hat_xi, primitives[self.s_volume_fraction],
                    one_cell_size_xi, axis)
            rhs_conservatives_xi = rhs_conservatives_xi.at[self.s_volume_fraction].add(
                -rhs_volume_fraction_contribution)


        # SURFACE TENSION TERM
        # TODO 4EQM
        if is_convective_flux and is_surface_tension \
            and self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rhs_momentum_contribution, rhs_energy_contribution = \
                self.diffuse_interface_handler.compute_surface_tension_source_term_xi(
                    u_hat_xi, alpha_hat_xi, primitives[self.ids_volume_fraction],
                    curvature, one_cell_size_xi, axis)
            rhs_conservatives_xi = rhs_conservatives_xi.at[self.vel_ids[axis]].add(
                -rhs_momentum_contribution)
            rhs_conservatives_xi = rhs_conservatives_xi.at[self.ids_energy].add(
                -rhs_energy_contribution)


        # INTERFACE FLUXES
        if levelset_model == "FLUID-FLUID":
            fluid_fluid_handler = self.levelset_handler.fluid_fluid_handler
            interface_flux_xi = fluid_fluid_handler.compute_interface_flux_xi(
                primitives, levelset, interface_velocity, interface_pressure,
                volume_fraction, apertures[axis], axis, temperature)
            rhs_conservatives_xi += one_cell_size_xi * interface_flux_xi
        elif is_solid_levelset:
            fluid_solid_handler = self.levelset_handler.fluid_solid_handler
            interface_flux_xi, solid_acceleration_xi = fluid_solid_handler.compute_interface_flux_xi(
                primitives, volume_fraction, levelset,
                apertures, axis, self.gravity, physical_simulation_time,
                solid_velocity)
            rhs_conservatives_xi += one_cell_size_xi * interface_flux_xi
            if solid_coupling.dynamic == "TWO-WAY":
                rhs_solid_velocity_xi += solid_acceleration_xi


        # LEVELSET ADVECTION
        if is_moving_levelset:
            levelset_advection_rhs_contribution_xi = \
                self.levelset_handler.compute_levelset_advection_rhs_xi(
                    levelset, interface_velocity, solid_velocity,
                    axis, physical_simulation_time, ml_setup)
            rhs_levelset_xi += levelset_advection_rhs_contribution_xi


        # SOLID TEMPERATURE
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

        # CREATE BUFFERS
        rhs_buffers_euler_xi = EulerIntegrationBuffers(
            rhs_conservatives_xi, rhs_levelset_xi,
            rhs_solid_velocity_xi, rhs_solid_energy_xi)

        positivity_counter_xi = PositivityCounter(
            positivity_count_interpolation_xi,
            positivity_count_thinc_xi,
            positivity_count_flux_xi,
            positivity_count_acdi_xi,
            None)

        discretization_counter_xi = DiscretizationCounter(count_acdi_xi, None)

        return rhs_buffers_euler_xi, positivity_counter_xi, discretization_counter_xi
        
