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

import types
from typing import Union, Dict

import h5py
from jax import config
import jax.numpy as jnp
import numpy as np

from jaxfluids.boundary_condition import BoundaryCondition
from jaxfluids.domain_information import DomainInformation
from jaxfluids.input_reader import InputReader
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.levelset.levelset_creator import LevelsetCreator
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.utilities import get_conservatives_from_primitives
from jaxfluids.turb.turb_init_cond import TurbInitManager

class Initializer:
    """The Initializer class implements functionality to create a dictionary of initial buffers that is 
    passed to the simulate() method of the SimulationManager class. The initialization() method returns this
    dictionary. The initial buffers are created in one of the following ways: 
    1) From a restart file that is specified in the case setup.
    2) From turbulent initial condition parameters that are specified in the case setup
    3) From the initial primitive buffer that is passed to the initialization() method
    4) From the initial conditions for primitive variables specified in case setup
    Note that if multiple of the above are provided, the priority is 1) - 4).
    """

    def __init__(self, input_reader: InputReader) -> None:

        self.input_reader       = input_reader
        self.numerical_setup    = self.input_reader.numerical_setup

        config.update("jax_enable_x64", self.input_reader.numerical_setup["output"]["is_double_precision_compute"])

        self.eps = jnp.finfo(jnp.float64).eps if self.input_reader.numerical_setup["output"]["is_double_precision_compute"] else jnp.finfo(jnp.float32).eps

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

        if input_reader.levelset_type != None:

            self.levelset_handler    = LevelsetHandler(
                    domain_information          = self.domain_information,   
                    numerical_setup             = self.numerical_setup,
                    material_manager            = self.material_manager,
                    unit_handler                = self.unit_handler,
                    solid_interface_velocity    = self.input_reader.solid_interface_velocity,
                    boundary_condition          = self.boundary_condition,
            )

            self.levelset_creator   = LevelsetCreator(
                    domain_information          = self.domain_information,
                    unit_handler                = self.unit_handler,
                    initial_levelset            = input_reader.initial_levelset,
                    narrow_band_cutoff          = self.numerical_setup["levelset"]["narrow_band_cutoff"]
            )

        if self.input_reader.is_turb_init:
            self.turb_init_manager = TurbInitManager(self.input_reader, self.domain_information, self.material_manager)

    def initialization(self, user_prime_init: Union[np.ndarray, jnp.ndarray] = None) -> Dict[str, Dict[str, Union[jnp.ndarray, float]]]:
        """Creates a dictionary of initial buffers. The buffers are either created from the case setup file 
        or from the user_prime_init argument.

        :param user_prime_init: User specified initial buffer for the primitive variables, defaults to None
        :type user_prime_init: Union[np.ndarray, jnp.ndarray], optional
        :return: Dictionary of initial buffers
        :rtype: Dict[str, Dict[str, Union[jnp.ndarray, float]]]
        """

        # DOMAIN INFORMATION
        nh                  = self.domain_information.nh_conservatives
        nx, ny, nz          = self.domain_information.number_of_cells
        nhx, nhy, nhz       = self.domain_information.domain_slices_conservatives
        active_axis_indices = self.domain_information.active_axis_indices

        # CREATE INITIAL BUFFER
        if self.input_reader.levelset_type == "FLUID-FLUID":
            primes = jnp.ones((5, 2, nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz)) * self.eps
        else:
            primes = jnp.ones((5, 1, nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz)) * self.eps

        # FILL BUFFERS FROM RESTART FILE
        if self.input_reader.restart_flag:

            h5file = h5py.File(self.input_reader.restart_file_path, "r")

            # PHYSICAL SIMULATION TIME
            current_time = self.unit_handler.non_dimensionalize(h5file["time"][()], "time")

            # CHECK IF END TIME IS LARGER THAN TIME OF RESTART
            assert self.unit_handler.non_dimensionalize(self.input_reader.end_time, "time") > current_time, "The end time %.4e is not larger than the restart time %.4e" % (self.input_reader.end_time, current_time)

            # LEVELSET FLUID FLUID INTERFACE INTERACTION TYPE
            if self.input_reader.levelset_type == "FLUID-FLUID":
                primes_fluid_fluid = ["density_0", "velocity_0", "pressure_0", "density_1", "velocity_1", "pressure_1"]
                density_0   = self.unit_handler.non_dimensionalize(h5file["primes/density_0"][:].T, "density")
                density_1   = self.unit_handler.non_dimensionalize(h5file["primes/density_1"][:].T, "density")
                density     = jnp.stack([density_0, density_1], axis=0)
                pressure_0  = self.unit_handler.non_dimensionalize(h5file["primes/pressure_0"][:].T, "pressure")
                pressure_1  = self.unit_handler.non_dimensionalize(h5file["primes/pressure_1"][:].T, "pressure")
                pressure    = jnp.stack([pressure_0, pressure_1], axis=0)
                if "velocity_0" in h5file["primes"].keys():
                    velocity_0  = self.unit_handler.non_dimensionalize(h5file["primes/velocity_0"][:].T, "velocity")
                    velocity_1  = self.unit_handler.non_dimensionalize(h5file["primes/velocity_1"][:].T, "velocity")
                else:
                    velocityX_0 = self.unit_handler.non_dimensionalize(h5file["primes/velocityX_0"][:].T, "velocity") if 0 in active_axis_indices else jnp.zeros_like(density_0)
                    velocityX_1 = self.unit_handler.non_dimensionalize(h5file["primes/velocityX_1"][:].T, "velocity") if 0 in active_axis_indices else jnp.zeros_like(density_0)
                    velocityX   = jnp.stack([velocityX_0, velocityX_1], axis=0)
                    velocityY_0 = self.unit_handler.non_dimensionalize(h5file["primes/velocityY_0"][:].T, "velocity") if 1 in active_axis_indices else jnp.zeros_like(density_0) 
                    velocityY_1 = self.unit_handler.non_dimensionalize(h5file["primes/velocityY_1"][:].T, "velocity") if 1 in active_axis_indices else jnp.zeros_like(density_0)
                    velocityX   = jnp.stack([velocityY_0, velocityY_1], axis=0)
                    velocityZ_0 = self.unit_handler.non_dimensionalize(h5file["primes/velocityZ_0"][:].T, "velocity") if 2 in active_axis_indices else jnp.zeros_like(density_0)
                    velocityZ_1 = self.unit_handler.non_dimensionalize(h5file["primes/velocityZ_1"][:].T, "velocity") if 2 in active_axis_indices else jnp.zeros_like(density_0)
                    velocityZ   = jnp.stack([velocityZ_0, velocityZ_1], axis=0)
                    velocity    = jnp.stack([velocityX, velocityY, velocityZ], axis=0)
                primes_init     = jnp.vstack([jnp.expand_dims(density, axis=0), velocity, jnp.expand_dims(pressure, axis=0)])
                primes          = primes.at[...,nhx,nhy,nhz].set(primes_init)
            # ELSE
            else:  
                density     = self.unit_handler.non_dimensionalize(h5file["primes/density"][:].T, "density")
                pressure    = self.unit_handler.non_dimensionalize(h5file["primes/pressure"][:].T, "pressure")
                if "velocity" in h5file["primes"].keys():
                    velocity = self.unit_handler.non_dimensionalize(h5file["primes/velocity"][:].T, "velocity")
                else:
                    velocityX   = self.unit_handler.non_dimensionalize(h5file["primes/velocityX"][:].T, "velocity") if 0 in active_axis_indices else jnp.zeros_like(density)
                    velocityY   = self.unit_handler.non_dimensionalize(h5file["primes/velocityY"][:].T, "velocity") if 1 in active_axis_indices else jnp.zeros_like(density) 
                    velocityZ   = self.unit_handler.non_dimensionalize(h5file["primes/velocityZ"][:].T, "velocity") if 2 in active_axis_indices else jnp.zeros_like(density)
                    velocity    = jnp.stack([velocityX, velocityY, velocityZ], axis=0)

                primes_init     = jnp.vstack([jnp.expand_dims(density, axis=0), velocity, jnp.expand_dims(pressure, axis=0)])
                primes          = primes.at[...,0,nhx,nhy,nhz].set(primes_init)
                primes          = jnp.squeeze(primes, axis=1)
            
            # COMPUTE CONSERVATIVES AND FILL BOUNDARIES
            cons            = get_conservatives_from_primitives(primes, self.material_manager)
            cons, primes    = self.boundary_condition.fill_boundary_primes(cons, primes, current_time)
            
            # CHECK IF RESOLUTION OF RESTART FILE AND CASE SETUP MATCH
            if self.input_reader.levelset_type == "FLUID-FLUID":
                assert (2, nx, ny, nz) == density.shape, "Resolution in case setup %s is not same as in restart file %s" % ((nx, ny, nz), density.shape)
            else:
                assert (nx, ny, nz) == density.shape, "Resolution in case setup %s is not same as in restart file %s" % ((nx, ny, nz), density.shape)
            
            # FORCINGS
            if self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]:
                PID_e_int       = h5file["mass_flow_forcing/PID_e_int"][()]
                PID_e_new       = h5file["mass_flow_forcing/PID_e_new"][()]

            # LEVELSET QUANTITIE
            if self.input_reader.levelset_type != None:
                assert "levelset" in h5file.keys(), "No levelset in restart file."
                assert "levelset" in h5file["levelset"].keys(), "No levelset in restart file."
                levelset                    = jnp.zeros((nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))
                levelset_init               = self.unit_handler.non_dimensionalize(h5file["levelset/levelset"][:].T, "length")
                levelset                    = levelset.at[...,nhx,nhy,nhz].set(levelset_init)
                levelset                    = self.boundary_condition.fill_boundary_levelset(levelset)
                volume_fraction, apertures  = self.levelset_handler.compute_volume_fraction_and_apertures(levelset)

        # FILL BUFFERS FROM TURBULENT INITIAL CONDITIONS
        elif self.input_reader.is_turb_init:

            # PHYSICAL SIMULATION TIME
            current_time = 0.0

            # GET INITIAL PRIMITIVE VARIABLES
            density, velocityX, velocityY, velocityZ, pressure = self.turb_init_manager.get_turbulent_initial_condition() 
            primes_init     = jnp.stack([density, velocityX, velocityY, velocityZ, pressure])
            primes          = primes.at[:, 0, nhx, nhy, nhz].set(primes_init)

            if self.input_reader.levelset_type != "FLUID-FLUID":
                primes = jnp.squeeze(primes, axis=1)

            # CONSERVATIVES
            cons = get_conservatives_from_primitives(primes, self.material_manager)

            # BOUNDARIES
            cons, primes  = self.boundary_condition.fill_boundary_primes(cons, primes, 0.0)

        # USER SPECIFIED INIT PRIME BUFFER
        elif user_prime_init is not None:
            # TODO LEVELSET
        
            # PHYSICAL SIMULATION TIME
            current_time = 0.0
            
            # NONDIM
            user_prime_init[0]      = self.unit_handler.non_dimensionalize(user_prime_init[0], "density")
            user_prime_init[1:4]    = self.unit_handler.non_dimensionalize(user_prime_init[1:4], "velocity")
            user_prime_init[4]      = self.unit_handler.non_dimensionalize(user_prime_init[4], "pressure")

            # PRIMES
            primes = primes.at[..., nhx, nhy, nhz].set(user_prime_init)
            if self.input_reader.levelset_type != "FLUID-FLUID":
                primes = jnp.squeeze(primes, axis=1)

            # CONSERVATIVES
            cons = get_conservatives_from_primitives(primes, self.material_manager)

            # BOUNDARIES
            cons, primes  = self.boundary_condition.fill_boundary_primes(cons, primes, 0.0)

        # FILL BUFFERS USING THE INITIAL CONDITION FOR THE PRIMITIVE VARIABLES SPECIFIED IN THE CASE SETUP
        else:
            
            # PHYSICAL SIMULATION TIME
            current_time = 0.0

            # GENERATE MESHGRID
            fluid_phases    = 2 if self.input_reader.levelset_type == "FLUID-FLUID" else 1
            mesh_grid       = [jnp.meshgrid(*self.domain_information.cell_centers, indexing="ij")[i] for i in active_axis_indices]

            # DIMENSIONALIZE FOR LAMBDA FUNCTION
            for i in range(len(mesh_grid)):
                mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")
            
            # GENERATE INITIAL CONDITION
            primes_init = []
            for i in range(fluid_phases):
                initial_condition = self.input_reader.initial_condition[["positive", "negative"][i]] if self.input_reader.levelset_type == "FLUID-FLUID" else self.input_reader.initial_condition
                primes_init_fluid = []
                for key in initial_condition:
                    function = initial_condition[key]
                    if type(function) in [float, np.float64, np.float32]:
                        prime_state_init = function*np.ones(mesh_grid[0].shape)
                    elif type(function) == types.LambdaType:
                        prime_state_init = function(*mesh_grid)
                    else:
                        assert False, "Initial condition must be float or lambda function"
                    prime_state_init = self.unit_handler.non_dimensionalize(prime_state_init, key)
                    primes_init_fluid.append( prime_state_init )
                primes_init_fluid = jnp.stack(primes_init_fluid, axis=0)
                primes = primes.at[..., i, nhx, nhy, nhz].set(primes_init_fluid)

            if self.input_reader.levelset_type != "FLUID-FLUID":
                primes = jnp.squeeze(primes, axis=1)

            # CONSERVATIVES
            cons = get_conservatives_from_primitives(primes, self.material_manager)

            # LEVELSET
            if self.input_reader.levelset_type != None:
                levelset        = self.levelset_creator.create_levelset()
                levelset        = self.boundary_condition.fill_boundary_levelset(levelset)
                levelset, _     = self.levelset_handler.reinitialize(levelset, True)
                levelset        = self.boundary_condition.fill_boundary_levelset(levelset)
                volume_fraction, apertures  = self.levelset_handler.compute_volume_fraction_and_apertures(levelset)
                cons, primes, _ = self.levelset_handler.extend_primes(cons, primes, levelset, volume_fraction, current_time)

            # BOUNDARIES
            cons, primes  = self.boundary_condition.fill_boundary_primes(cons, primes, 0.0)

        buffer_dictionary = {
            "material_fields": {},
            "levelset_quantities": {},
            "mass_flow_forcing": {},
            "machinelearning_modules": {},
            "time_control": {}
        }
        
        # MATERIAL FIELD BUFFERS
        buffer_dictionary["material_fields"].update({
            "cons": cons, "primes": primes,
        })
        
        # TIME CONTROL
        buffer_dictionary["time_control"].update({
            "current_time": current_time, "timestep_size": 0.0,
        })

        # MASS FLOW FORCING
        if self.input_reader.restart_flag and self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]:
            buffer_dictionary["mass_flow_forcing"].update({
                "PID_e_int": PID_e_int, "PID_e_new": PID_e_new, "scalar_value": 0.0
            })
        elif self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]:
            buffer_dictionary["mass_flow_forcing"].update({
                "PID_e_int": 0.0, "PID_e_new": 0.0, "scalar_value": 0.0
            })
        else:
            buffer_dictionary["mass_flow_forcing"].update({
                "PID_e_int": None, "PID_e_new": None, "scalar_value": None
            })

        # LEVELSET QUANTITIES FIELD BUFFERS
        if self.input_reader.levelset_type != None:
            buffer_dictionary["levelset_quantities"].update({
                "levelset": levelset, "volume_fraction": volume_fraction, "apertures": apertures
            })
        else:
            buffer_dictionary["levelset_quantities"].update({
                "levelset": None, "volume_fraction": None, "apertures": None
            })

        # MACHINE LEARNING
        buffer_dictionary["machinelearning_modules"].update({
            "ml_parameters_dict": None, "ml_networks_dict": None,
        })

        return buffer_dictionary