#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from functools import partial
import time
from typing import List, Tuple, Union, Dict

import jax
from jax.config import config
import jax.numpy as jnp

from jaxfluids.boundary_condition import BoundaryCondition
from jaxfluids.domain_information import DomainInformation
from jaxfluids.forcing.forcing import Forcing
from jaxfluids.input_reader import InputReader
from jaxfluids.io_utils.logger import Logger
from jaxfluids.io_utils.output_writer import OutputWriter
from jaxfluids.levelset.interface_quantity_computer import InterfaceQuantityComputer
from jaxfluids.materials.material_manager import Material
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.eigendecomposition import Eigendecomposition
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.levelset.levelset_handler import LevelsetHandler 
from jaxfluids.levelset.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.levelset.geometry_calculator import GeometryCalculator
from jaxfluids.space_solver import SpaceSolver
from jaxfluids.stencils import DICT_FIRST_DERIVATIVE_CENTER
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.time_integration import DICT_TIME_INTEGRATION
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.utilities import get_primitives_from_conservatives, get_conservatives_from_primitives
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.turb.turb_stats_manager import TurbStatsManager

class SimulationManager:
    """ The SimulationManager is the top-level class in JAX-FLUIDS. It
    provides functionality to perform conventional CFD simulations
    as well as end-to-end optimization of ML models.

    The most important methods of the SimulationManager are:
    1) simulate()               -   Performs conventional CFD simulation.
    2) feedforward()            -   Feedforward of a batch of data, i.e.,
        advances a batch of initial conditions in time for a fixed amount of steps
    3) do_integration_step()    -   Performs a single integration step
    """

    def __init__(self, input_reader: InputReader) -> None:

        self.input_reader       = input_reader
        self.numerical_setup    = self.input_reader.numerical_setup

        config.update("jax_enable_x64", self.numerical_setup["output"]["is_double_precision_compute"])

        # SET EPSILON IN CLASSES
        self.eps    = jnp.finfo(jnp.float64).eps if self.numerical_setup["output"]["is_double_precision_compute"] \
            else jnp.finfo(jnp.float32).eps
        classes     = [Material, SpatialReconstruction, SpatialDerivative, Eigendecomposition, RiemannSolver,
            GeometryCalculator, LevelsetHandler, LevelsetReinitializer, InterfaceQuantityComputer]
        for c in classes:
            c.eps = self.eps

        self.eps_time = 1e-12

        self.unit_handler = UnitHandler( **input_reader.nondimensionalization_parameters )

        self.domain_information = DomainInformation(
            dim                 = input_reader.dim,
            nx                  = input_reader.nx,
            ny                  = input_reader.ny,
            nz                  = input_reader.nz,
            nh_conservatives    = self.numerical_setup["conservatives"]["halo_cells"],
            nh_geometry         = self.numerical_setup["levelset"]["halo_cells"] if input_reader.levelset_type != None else None,
            domain_size         = self.unit_handler.non_dimensionalize_domain_size(input_reader.domain_size)
        )

        self.material_manager = MaterialManager(
            unit_handler        = self.unit_handler,
            material_properties = input_reader.material_properties,
            levelset_type       = input_reader.levelset_type
            )

        self.boundary_condition = BoundaryCondition(    
            domain_information      = self.domain_information,
            material_manager        = self.material_manager,
            unit_handler            = self.unit_handler,
            boundary_types          = input_reader.boundary_location_types,
            wall_velocity_functions = input_reader.wall_velocity_functions,
            dirichlet_functions     = input_reader.dirichlet_functions,
            neumann_functions       = input_reader.neumann_functions,
            levelset_type           = input_reader.levelset_type
        )

        # TIME CONTROL
        if "fixed_timestep" in self.numerical_setup["conservatives"]["time_integration"].keys():
            self.fixed_timestep = self.unit_handler.non_dimensionalize(self.numerical_setup["conservatives"]["time_integration"]["fixed_timestep"], "time")
        else:
            self.fixed_timestep = False
        self.end_time       = self.unit_handler.non_dimensionalize(input_reader.end_time, "time")
        self.CFL            = self.numerical_setup["conservatives"]["time_integration"]["CFL"]

        self.time_integrator : TimeIntegrator = DICT_TIME_INTEGRATION[self.numerical_setup["conservatives"]["time_integration"]["integrator"]](nh=self.domain_information.nh_conservatives, inactive_axis=self.domain_information.inactive_axis)

        # LEVELSET HANDLER
        if self.input_reader.levelset_type != None:
            self.levelset_handler    = LevelsetHandler(
                domain_information          = self.domain_information,   
                numerical_setup             = self.numerical_setup,
                material_manager            = self.material_manager,
                unit_handler                = self.unit_handler,
                solid_interface_velocity    = self.input_reader.solid_interface_velocity,
                boundary_condition          = self.boundary_condition,
                )

        # SPACE SOLVER
        self.space_solver = SpaceSolver(
            domain_information  = self.domain_information,    
            material_manager    = self.material_manager,
            numerical_setup     = self.numerical_setup,
            gravity             = self.unit_handler.non_dimensionalize(input_reader.gravity, "gravity"),
            levelset_type       = self.input_reader.levelset_type,
            levelset_handler    = self.levelset_handler if self.input_reader.levelset_type else None
            )

        # TURBULENT STATISTICS 
        if self.input_reader.is_turb_init:
            self.turb_stats_manager = TurbStatsManager(
                domain_information  = self.domain_information,
                material_manager    = self.material_manager,
            )

        # FORCINGS
        if self.input_reader.active_forcings:
            self.forcings_computer = Forcing( 
                domain_information      = self.domain_information,
                material_manager        = self.material_manager, 
                unit_handler            = self.unit_handler,
                levelset_handler        = self.levelset_handler if self.input_reader.levelset_type != None else None,
                levelset_type           = self.input_reader.levelset_type,
                is_mass_flow_forcing    = self.numerical_setup["active_forcings"]["is_mass_flow_forcing"],
                is_temperature_forcing  = self.numerical_setup["active_forcings"]["is_temperature_forcing"],
                is_turb_hit_forcing     = self.numerical_setup["active_forcings"]["is_turb_hit_forcing"],
                mass_flow_target        = self.input_reader.mass_flow_target,
                flow_direction          = self.input_reader.mass_flow_direction,
                temperature_target      = self.input_reader.temperature_target 
                )

        # OUTPUT WRITER
        self.output_writer = OutputWriter(  
            input_reader                        = input_reader,
            unit_handler                        = self.unit_handler,
            domain_information                  = self.domain_information,
            material_manager                    = self.material_manager,
            levelset_handler                    = self.levelset_handler if self.input_reader.levelset_type != None else None,
            derivative_stencil_conservatives    = DICT_FIRST_DERIVATIVE_CENTER[self.numerical_setup["output"]["derivative_stencil"]](nh=self.domain_information.nh_conservatives, inactive_axis=self.domain_information.inactive_axis),
            derivative_stencil_geometry         = DICT_FIRST_DERIVATIVE_CENTER[self.numerical_setup["output"]["derivative_stencil"]](nh=self.domain_information.nh_geometry, inactive_axis=self.domain_information.inactive_axis) if self.input_reader.levelset_type != None else None,
            )

        self.logger = Logger("", logging_level=self.numerical_setup["output"]["logging"]) 

    def simulate(self, buffer_dictionary: Dict[str, Dict[str, Union[jnp.DeviceArray, float]]]) -> None:
        """Performs a conventional CFD simulation.

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict[str, Dict[str, Union[jnp.DeviceArray, float]]]
        """
        self.initialize(buffer_dictionary)
        self.advance(buffer_dictionary)

    def initialize(self, buffer_dictionary: Dict[str, Dict[str, Union[jnp.DeviceArray, float]]]) -> None:
        """ Initializes the simulation, i.e., creates the output directory,
        logs the numerical and case setup, and writes the initial output.

        :param buffer_dictionary: Dictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        """

        # CREATE OUTPUT FOLDER, CASE SETUP AND NUMERICAL SETUP
        self.output_writer.create_folder()

        # CONFIGURE LOGGER AND LOG NUMERICAL SETUP AND CASE SETUP
        self.logger.configure_logger(self.output_writer.save_path_case)
        self.logger.log_initialization()
        self.logger.log_numerical_setup_and_case_setup(*self.input_reader.info())

        # LOG TURBULENT STATS
        if self.input_reader.is_turb_init:
            nhx, nhy, nhz       = self.domain_information.domain_slices_conservatives
            turbulent_statistics_dict = self.turb_stats_manager.get_turbulent_statistics(
                buffer_dictionary["material_fields"]["primes"][...,nhx,nhy,nhz])
            self.logger.log_turbulent_stats_at_start(turbulent_statistics_dict)

        # WRITE INITIAL OUTPUT
        current_time = buffer_dictionary["time_control"]["current_time"]
        self.output_writer.next_timestamp += current_time
        self.output_writer.write_output(buffer_dictionary, force_output=True)

    def advance(self, buffer_dictionary: Dict[str, Dict[str, Union[jnp.DeviceArray, float]]]) -> None:
        """ Advances the initial buffers in time.

        :param buffer_dictionary: ictionary containing the material field buffers,
        levelset quantitiy buffers, time control and mass flow forcing parameters
        :type buffer_dictionary: Dict
        """

        # LOG SIMULATION START
        self.logger.log_sim_start()

        # START LOOP
        start_loop      = time.time()
        current_step    = 0
        current_time    = buffer_dictionary["time_control"]["current_time"]
        while current_time < self.end_time - self.eps_time:

            start_iteration = time.time()

            # COMPUTE TIMESTEP 
            if self.fixed_timestep:
                timestep_size = self.fixed_timestep
            else:
                timestep_size = self.compute_timestep(
                        **buffer_dictionary["material_fields"],
                        **buffer_dictionary["levelset_quantities"]
                        )
            buffer_dictionary["time_control"]["timestep_size"] = timestep_size

            # COMPUTE FORCINGS
            if self.input_reader.active_forcings:
                if self.numerical_setup["active_forcings"]["is_turb_hit_forcing"]:
                    material_fields, _, _ = self.do_integration_step(
                            **buffer_dictionary["material_fields"],
                            **buffer_dictionary["time_control"])
                    buffer_dictionary["material_fields"]["primes_dash"] = material_fields["primes"]
                forcings_dictionary = self.forcings_computer.compute_forcings(
                    **buffer_dictionary["material_fields"],
                    **buffer_dictionary["time_control"],
                    **buffer_dictionary["levelset_quantities"],
                    **buffer_dictionary["mass_flow_forcing"],
                    logger=self.logger
                    )
                if self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]:
                    buffer_dictionary["mass_flow_forcing"].update(forcings_dictionary["mass_flow"])
            else:
                forcings_dictionary = None

            # INTEGRATION STEP
            if self.input_reader.levelset_type != None:
                reinitialize = True if current_step % self.levelset_handler.interval_reinitialization == 0 else False
            else:
                reinitialize = False
            material_fields, levelset_quantities, residuals = self.do_integration_step(
                **buffer_dictionary["material_fields"],
                **buffer_dictionary["time_control"],
                **buffer_dictionary["levelset_quantities"],
                **buffer_dictionary["machinelearning_modules"],
                forcings_dictionary=forcings_dictionary,
                reinitialize=reinitialize
            )
            buffer_dictionary["material_fields"].update(material_fields)
            buffer_dictionary["levelset_quantities"].update(levelset_quantities)

            # INCREMENT PHYSICAL SIMULATION TIME
            current_time += timestep_size
            buffer_dictionary["time_control"]["current_time"] = current_time
            
            # FORCE PYTHON TO WAIT FOR JAX COMPUTATIONS TO COMPLETE
            buffer_dictionary["material_fields"]["cons"].block_until_ready()

            # WRITE H5 OUTPUT
            self.output_writer.write_output(buffer_dictionary, force_output=False)

            # COMPUTE WALL CLOCK FOR TIME STEP
            wall_clock_step = time.time() - start_iteration
            wall_clock_step_cell =  wall_clock_step / self.domain_information.resolution
            mean_wall_clock_step = (wall_clock_step + mean_wall_clock_step*(current_step - 1))/current_step if current_step > 3 else wall_clock_step
            mean_wall_clock_step_cell = (wall_clock_step_cell + mean_wall_clock_step_cell*(current_step - 1))/current_step if current_step > 3 else wall_clock_step_cell
            
            # INCREMENT CURRENT STEP
            current_step += 1

            # LOG TERMINAL
            print_list = [
                'CURRENT TIME                   = %4.4e' % (self.unit_handler.dimensionalize(current_time, "time")),
                'CURRENT DT                     = %4.4e' % (self.unit_handler.dimensionalize(timestep_size, "time")),
                'CURRENT STEP                   = %6d'   % current_step,
                'WALL CLOCK TIMESTEP            = %4.4e' % wall_clock_step,
                'MEAN WALL CLOCK TIMESTEP       = %4.4e' % mean_wall_clock_step,
                'WALL CLOCK TIMESTEP CELL       = %4.4e' % (wall_clock_step_cell),
                'MEAN WALL CLOCK TIMESTEP CELL  = %4.4e' % (mean_wall_clock_step_cell)
            ]

            if self.input_reader.levelset_type != None:
                print_list += [ 'RESIDUAL EXTENSION PRIMES      = %4.4e'    % residuals["extension_primes"] ]
            if self.input_reader.levelset_type == "FLUID-FLUID":
                print_list += [ 'RESIDUAL EXTENSION INTERFACE   = %4.4e'    % residuals["extension_interface"] ]
                print_list += [ 'RESIDUAL REINITIALIZATION      = %4.4e'    % residuals["reinitialization"] ]
                
            self.logger.log_end_time_step(print_list)


        # FINAL OUTPUT
        self.output_writer.write_output(buffer_dictionary, force_output=True, simulation_finish=True)

        # LOG SIMULATION FINISH
        self.logger.log_sim_finish(time.time() - start_loop)

    @partial(jax.jit, static_argnums=(0, 8, 11))
    def do_integration_step(self, cons: jnp.DeviceArray, primes: jnp.DeviceArray, timestep_size: float, current_time: float,
        levelset: Union[jnp.DeviceArray, None] = None, volume_fraction: Union[jnp.DeviceArray, None] = None, apertures: Union[List, None] = None,
        reinitialize: bool = False, forcings_dictionary: Union[Dict, None] = None, 
        ml_parameters_dict: Union[Dict, None] = None, ml_networks_dict: Union[Dict, None] = None, **kwargs) -> Tuple[Dict, Dict, Dict]:
        """Performs an integration step using the specified integration scheme. For twophase simulations 
        a single RK stage consists of the following:
        1) Compute right-hand-side of Navier-Stokes and levelset advection equation
        2) Transform volume-averaged conservatives to actual conservatives
            that can be integrated according to volume fraction
        3) Prepare the conservative and levelset buffer for integration according
            to the present integration scheme
        4) Integrate conservatives
        5) Integrate levelset + reinitialize levelset + fill levelset boundaries
        6) Compute volume fraction and apertures from integrated levelset quantities
        7) Apply the mixing procedure to the integrated conservative variables
        8) Transform mixed conservative variables to volume-averaged conservative variables and compute 
            corresponding primitive variables
        9) Extend primitive variables into ghost cells and compute conservative variables in ghost cells from extended primitive
            variables
        10) Fill material boundaries

        :param cons: Buffer of conservative variables
        :type cons: jnp.DeviceArray
        :param primes: Buffer of primitive variables
        :type primes: jnp.DeviceArray
        :param timestep_size: Current physical time step size
        :type timestep_size: float
        :param current_time: Current physical simulation time
        :type current_time: float
        :param levelset: Levelset buffer, defaults to None
        :type levelset: Union[jnp.DeviceArray, None], optional
        :param volume_fraction: Volume fraction buffer, defaults to None
        :type volume_fraction: Union[jnp.DeviceArray, None], optional
        :param apertures: Aperture buffers, defaults to None
        :type apertures: Union[List, None], optional
        :param reinitialize: Flag indicating whether to reinitialize levelset in the present time step, defaults to False
        :type reinitialize: bool, optional
        :param forcings_dictionary: Dictionary containing forcing buffers, defaults to None
        :type forcings_dictionary: Union[Dict, None], optional
        :param ml_parameters_dict: Dictionary containing NN weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary containing NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Tuple of material fields dictionary, levelset quantities dictionary and residual dictionary
        :rtype: Tuple[Dict, Dict, Dict]
        """
        
        # DEFAULT VALUES FOR RESIDUALS
        residual_primes = None if self.input_reader.levelset_type == None else 0.0
        residual_reinit = None if self.input_reader.levelset_type != "FLUID-FLUID" else 0.0
        
        # INIT BUFFER FOR RUNGE KUTTA SCHEME
        if self.time_integrator.no_stages > 1:
            init_cons       = self.levelset_handler.transform_to_conservatives(cons, volume_fraction) if self.input_reader.levelset_type != None else jnp.array(cons, copy=True)
            init_levelset   = jnp.array(levelset, copy=True) if self.input_reader.levelset_type in ["FLUID-FLUID", "FLUID-SOLID-DYNAMIC"] else None

        current_time_stage = current_time

        # LOOP STAGES
        for stage in range( self.time_integrator.no_stages ):

            # RIGHT HAND SIDE
            rhs_cons, rhs_levelset, residual_interface = self.space_solver.compute_rhs(
                cons, primes, current_time_stage, 
                levelset, volume_fraction, apertures, 
                forcings_dictionary, 
                ml_parameters_dict, ml_networks_dict)

            # TRANSFORM TO CONSERVATIVES
            if self.input_reader.levelset_type != None:
                cons = self.levelset_handler.transform_to_conservatives(cons, volume_fraction)

            # PREPARE BUFFER FOR RUNGE KUTTA INTEGRATION
            if stage > 0:
                cons = self.time_integrator.prepare_buffer_for_integration(cons, init_cons, stage)
                if self.input_reader.levelset_type in ["FLUID-FLUID", "FLUID-SOLID-DYNAMIC"]:
                    levelset = self.time_integrator.prepare_buffer_for_integration(levelset, init_levelset, stage)

            # INTEGRATE
            cons = self.time_integrator.integrate(cons, rhs_cons, timestep_size, stage)
            if self.input_reader.levelset_type in ["FLUID-FLUID", "FLUID-SOLID-DYNAMIC"]:
                levelset_new = self.time_integrator.integrate(levelset, rhs_levelset, timestep_size, stage)
                
                # REINITIALIZE
                if self.input_reader.levelset_type == "FLUID-FLUID" and stage == self.time_integrator.no_stages - 1 and reinitialize:
                    levelset_new, residual_reinit   = self.levelset_handler.reinitialize(levelset_new, False)
                else:
                    residual_reinit = 0.0

                # LEVELSET BOUNDARIES AND INTERFACE RECONSTRUCTION
                levelset_new                        = self.boundary_condition.fill_boundary_levelset(levelset_new)
                volume_fraction_new, apertures_new  = self.levelset_handler.compute_volume_fraction_and_apertures(levelset_new)
            
            elif self.input_reader.levelset_type == "FLUID-SOLID-STATIC":
                levelset_new, volume_fraction_new, apertures_new = levelset, volume_fraction, apertures

            current_time_stage = current_time + timestep_size*self.time_integrator.timestep_increment_factor[stage]

            # MIXING AND PRIME EXTENSION
            if self.input_reader.levelset_type != None:
                cons, mask_small_cells = self.levelset_handler.mixing(cons, levelset_new, volume_fraction_new, volume_fraction)
                cons    = self.levelset_handler.transform_to_volume_averages(cons, volume_fraction_new)
                primes  = self.levelset_handler.compute_primitives_from_conservatives_in_real_fluid(cons, primes, levelset_new, volume_fraction_new, mask_small_cells)
                cons, primes, residual_primes           = self.levelset_handler.extend_primes(cons, primes, levelset_new, volume_fraction_new, current_time_stage, mask_small_cells)
                levelset, volume_fraction, apertures    = levelset_new, volume_fraction_new, apertures_new
            else:
                primes = get_primitives_from_conservatives(cons, self.material_manager)
            
            # FILL BOUNDARIES
            cons, primes = self.boundary_condition.fill_boundary_primes(cons, primes, current_time_stage)

        # CREATE DICTIONARIES
        material_fields     = {"cons": cons, "primes": primes}
        levelset_quantities = {"levelset": levelset, "volume_fraction": volume_fraction, "apertures": apertures}
        residuals           = {"extension_primes": residual_primes, "extension_interface": residual_interface, "reinitialization": residual_reinit}

        return material_fields, levelset_quantities, residuals

    @partial(jax.jit, static_argnums=(0))
    def compute_timestep(self, primes: jnp.DeviceArray, levelset: jnp.DeviceArray,
            volume_fraction: jnp.DeviceArray, **kwargs) -> float:
        """Computes the physical time step size depending on the active physics.

        :param primes: Buffer of primitive variables
        :type primes: jnp.DeviceArray
        :param levelset: Levelset buffer
        :type levelset: jnp.DeviceArray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.DeviceArray
        :return: Time step size
        :rtype: float
        """
        
        # DOMAIN INFORMATION
        nhx, nhy, nhz       = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_    = self.domain_information.domain_slices_geometry
        cell_sizes          = self.domain_information.cell_sizes
        min_cell_size       = jnp.min(jnp.array([jnp.min(dxi) for dxi in cell_sizes]))

        # COMPUTE TEMPERATURE
        if self.numerical_setup["active_physics"]["is_viscous_flux"] or self.numerical_setup["active_physics"]["is_heat_flux"]:
            temperature = self.material_manager.get_temperature(primes[4,...,nhx,nhy,nhz], primes[0,...,nhx,nhy,nhz])

        # COMPUTE MASKS
        if self.input_reader.levelset_type != None:
            mask_real, _ = self.levelset_handler.compute_masks(levelset, volume_fraction)

        # CONVECTIVE CONTRIBUTION
        speed_of_sound  = self.material_manager.get_speed_of_sound(p=primes[4,...,nhx,nhy,nhz], rho=primes[0,...,nhx,nhy,nhz])
        abs_velocity    = 0.0
        for i in range(1,4):
            abs_velocity += (jnp.abs(primes[i,...,nhx,nhy,nhz]) + speed_of_sound)
        if self.input_reader.levelset_type != None:
            abs_velocity *= mask_real[...,nhx_,nhy_,nhz_]
        dt = min_cell_size / ( jnp.max(abs_velocity) + self.eps )

        # VISCOUS CONTRIBUTION
        if self.numerical_setup["active_physics"]["is_viscous_flux"]:
            const = 3.0 / 14.0
            kinematic_viscosity = self.material_manager.get_dynamic_viscosity(temperature) / primes[0,...,nhx,nhy,nhz]
            if self.input_reader.levelset_type != None:
                kinematic_viscosity = kinematic_viscosity * mask_real[..., nhx_,nhy_,nhz_]
            dt = jnp.minimum(dt, const * ( min_cell_size * min_cell_size ) / jnp.max(kinematic_viscosity) )

        # HEAT TRANSFER CONTRIBUTION
        if self.numerical_setup["active_physics"]["is_heat_flux"]:
            const = 0.1
            thermal_diffusivity = self.material_manager.get_thermal_conductivity(temperature) / primes[0,...,nhx,nhy,nhz]
            if self.input_reader.levelset_type != None:
                thermal_diffusivity = thermal_diffusivity * mask_real[..., nhx_,nhy_,nhz_]
            dt = jnp.minimum(dt, const * ( min_cell_size * min_cell_size ) / jnp.max(thermal_diffusivity) )

        dt = self.CFL * dt

        return dt

    def _feed_forward(self, primes_init: jnp.DeviceArray, levelset_init: jnp.DeviceArray, n_steps: int, timestep_size: float, 
        t_start: float, output_freq: int = 1, ml_parameters_dict: Union[Dict, None] = None,
        ml_networks_dict: Union[Dict, None] = None) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        """Advances the initial buffers in time for a fixed amount of steps and returns the
        entire trajectory. This function is differentiable and
        must therefore be used to end-to-end optimize ML models within the JAX-FLUIDS simulator.

        :param primes_init: Initial primitive variables buffer
        :type primes_init: jnp.DeviceArray
        :param levelset_init: Initial levelset buffer
        :type levelset_init: jnp.DeviceArray
        :param n_steps: Number of time steps
        :type n_steps: int
        :param timestep_size: Physical time step size
        :type timestep_size: float
        :param t_start: Physical start time
        :type t_start: float
        :param output_freq: Frequency in time steps for output, defaults to 1
        :type output_freq: int, optional
        :param ml_parameters_dict: _description_, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: _description_, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[jnp.DeviceArray, jnp.DeviceArray]
        """
        # CREATE BUFFER
        nh               = self.domain_information.nh_conservatives
        nx, ny, nz       = self.domain_information.number_of_cells
        nhx, nhy, nhz    = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry

        # CREATE BUFFER
        if self.input_reader.levelset_type == "FLUID-FLUID":
            primes      = jnp.ones((5, 2, nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))
            cons        = jnp.zeros((5, 2, nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))
            levelset    = jnp.zeros((nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))
        else:
            primes      = jnp.ones((5, nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))
            cons        = jnp.zeros((5, nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))
            levelset    = None

        # PRIME & LEVELSET BUFFER
        primes       = primes.at[..., nhx, nhy, nhz].set(primes_init)
        if self.input_reader.levelset_type == "FLUID-FLUID":
            levelset        = levelset.at[nhx, nhy, nhz].set(levelset_init)
            levelset        = self.boundary_condition.fill_boundary_levelset(levelset)
            levelset, _     = self.levelset_handler.reinitialize(levelset, True)
            levelset        = self.boundary_condition.fill_boundary_levelset(levelset)
            volume_fraction, apertures  = self.levelset_handler.compute_volume_fraction_and_apertures(levelset)
            _, primes, _     = self.levelset_handler.extend_primes(cons, primes, levelset, volume_fraction, t_start)
        else:
            volume_fraction = None
            apertures       = None

        # CONSERVATIVES
        cons = get_conservatives_from_primitives(primes, self.material_manager)

        # BOUNDARIES
        cons, primes = self.boundary_condition.fill_boundary_primes(cons, primes, 0.0)

        # TODO FEED FORWARD FOR LEVELSET
        if self.input_reader.levelset_type != None:
            primes_real = self.output_writer.compute_real_buffer(primes[...,nhx,nhy,nhz], volume_fraction[nhx_,nhy_,nhz_])
            out         = jnp.concatenate([primes_real, jnp.expand_dims(volume_fraction[nhx_,nhy_,nhz_], axis=0)], axis = 0)
        else:
            primes_real = primes[:,nhx,nhy,nhz]
            out         = primes_real

        # INITIAL OUTPUT
        solution_list = [out]
        times_list    = [t_start]

        forcings_dictionary = {}

        current_time = t_start
        current_step = 0

        # LOOP OVER STEPS
        for step in range(n_steps):
            if self.input_reader.levelset_type != None:
                reinitialize = True if current_step % self.levelset_handler.interval_reinitialization == 0 else False
            else:
                reinitialize = False

            material_fields, levelset_quantities, residuals = self.do_integration_step(
                cons, primes, timestep_size, current_time, 
                levelset, volume_fraction, apertures, reinitialize, 
                forcings_dictionary, ml_parameters_dict, ml_networks_dict)

            primes, cons = material_fields["primes"], material_fields["cons"]
            levelset, volume_fraction, apertures = levelset_quantities["levelset"], levelset_quantities["volume_fraction"], levelset_quantities["apertures"]

            current_time += timestep_size
            current_step += 1

            # APPEND OUTPUT
            if current_step % output_freq == 0:
                if self.input_reader.levelset_type != None:
                    cons_real   = self.output_writer.compute_real_buffer(cons[...,nhx,nhy,nhz], volume_fraction[nhx_,nhy_,nhz_])
                    primes_real = self.output_writer.compute_real_buffer(primes[...,nhx,nhy,nhz], volume_fraction[nhx_,nhy_,nhz_])
                    out         = jnp.concatenate([primes_real, jnp.expand_dims(volume_fraction[nhx_,nhy_,nhz_], axis=0)], axis = 0)
                else:
                    primes_real = primes[:,nhx,nhy,nhz]
                    out         = primes_real

                solution_list.append(out)
                times_list.append(current_time)

        solution_array = jnp.stack(solution_list)
        times_array    = jnp.stack(times_list)
        return solution_array, times_array

    def feed_forward(self, batch_primes_init: jnp.DeviceArray, batch_levelset_init: jnp.DeviceArray, n_steps: int, timestep_size: float, 
        t_start: float, output_freq: int = 1, ml_parameters_dict: Union[Dict, None] = None, 
        ml_networks_dict: Union[Dict, None] = None) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        """Vectorized version of the _feed_forward() method.

        :param batch_primes_init: batch of initial primitive variable buffers
        :type batch_primes_init: jnp.DeviceArray
        :param batch_levelset_init: batch of initial levelset buffers
        :type batch_levelset_init: jnp.DeviceArray
        :param n_steps: Number of integration steps
        :type n_steps: int
        :param timestep: Physical time step size
        :type timestep: float
        :param t_start: Physical start time
        :type t_start: float
        :param output_freq: Frequency in time steps for output, defaults to 1
        :type output_freq: int, optional
        :param ml_parameters_dict: NN weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: _description_
        :rtype: Tuple[jnp.DeviceArray, jnp.DeviceArray]
        """

        return jax.vmap(self._feed_forward, in_axes=(0,0,None,None,None,None,None,None), out_axes=(0,0,))(
            batch_primes_init, batch_levelset_init, n_steps, timestep_size, t_start, output_freq, ml_parameters_dict, ml_networks_dict)
