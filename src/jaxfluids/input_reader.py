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

import os
import types
from typing import Dict, Union, Tuple

import numpy as np
import jax.numpy as jnp
import json

from jaxfluids.solvers.riemann_solvers import DICT_RIEMANN_SOLVER, DICT_SIGNAL_SPEEDS
from jaxfluids.stencils import DICT_DERIVATIVE_REINITIALIZATION, DICT_DERIVATIVE_LEVELSET_ADVECTION, DICT_SPATIAL_RECONSTRUCTION, DICT_DERIVATIVE_QUANTITY_EXTENDER, DICT_DERIVATIVE_FACE, DICT_FIRST_DERIVATIVE_CENTER, DICT_CENTRAL_RECONSTRUCTION
from jaxfluids.time_integration import DICT_TIME_INTEGRATION
from jaxfluids.materials import DICT_MATERIAL

class InputReader:
    """ The InputReader class to reads a case setup
    and a numerical setup for setting up a JAX-FLUIDS simulation. Case setup
    and numerical setup can be provided as either a path to a json file or 
    as a preloaded dictionary.
    """

    def __init__(self, case_setup: Union[str, Dict], numerical_setup: Union[str, Dict]) -> None:

        # READ CASE SETUP AND NUMERICAL_SETUP 
        if type(case_setup) == str:        
            self.case_setup = json.load(open(case_setup))
        elif type(case_setup) == dict:
            self.case_setup = case_setup

        if type(numerical_setup) == str:
            self.numerical_setup = json.load(open(numerical_setup))
        elif type(numerical_setup) == dict:
            self.numerical_setup = numerical_setup

        # GENERAL
        self.case_name = self.case_setup["general"]["case_name"]
        self.end_time   = self.case_setup["general"]["end_time"]
        self.save_path  = self.case_setup["general"]["save_path"]
        self.save_dt    = self.case_setup["general"]["save_dt"]

        # LEVELSET INTERFACE INTERACTION TYPE
        if "levelset" in self.numerical_setup.keys():
            self.levelset_type = self.numerical_setup["levelset"]["interface_interaction"]
        else:
            self.levelset_type = None

        # DOMAIN
        self.nx                     = self.case_setup["domain"]["x"]["cells"]
        self.ny                     = self.case_setup["domain"]["y"]["cells"]
        self.nz                     = self.case_setup["domain"]["z"]["cells"]
        self.nh_conservatives       = self.numerical_setup["conservatives"]["halo_cells"]
        self.nh_geometry            = self.numerical_setup["levelset"]["halo_cells"] if self.levelset_type != None else None
        self.number_of_cells        = jnp.array([self.nx, self.ny, self.nz])
        self.dim                    = sum([1 if n > 1 else 0 for n in self.number_of_cells])
        self.domain_size            = { "x": self.case_setup["domain"]["x"]["range"],
                                        "y": self.case_setup["domain"]["y"]["range"],
                                        "z": self.case_setup["domain"]["z"]["range"] }
        self.axis                   = np.array(["x", "y", "z"])
        self.active_axis            = []
        self.inactive_axis          = []
        self.active_axis_indices    = []
        self.inactive_axis_indices  = []
        for i, axis in enumerate(["x", "y", "z"]):
            self.active_axis.append( axis ) if self.number_of_cells[i] > 1 else None
            self.inactive_axis.append( axis ) if self.number_of_cells[i] == 1 else None
            self.active_axis_indices.append( i ) if self.number_of_cells[i] > 1 else None
            self.inactive_axis_indices.append( i ) if self.number_of_cells[i] == 1 else None

        # BOUNDARIES    -   SYMMETRY, PERIODIC, WALL, DIRICHLET, NEUMANN, INACTIVE
        self.boundary_location_types    = self.case_setup["boundary_condition"]["types"]
        self.location_to_wall_velocity  = {     "east":     {"v": 0.0, "w": 0.0},   "west":     {"v": 0.0, "w": 0.0},
                                                "north":    {"u": 0.0, "w": 0.0},   "south":    {"u": 0.0, "w": 0.0},
                                                "top":      {"u": 0.0, "v": 0.0},   "bottom":   {"u": 0.0, "v": 0.0}    }
        self.wall_velocity_functions    = dict((location, self.location_to_wall_velocity[location]) for location in ["east", "west", "north", "south", "top", "bottom"]) 
        self.dirichlet_functions        = dict((location, {"rho": 1.0, "u": 0.0, "v": 0.0, "w": 0.0, "p": 1.0}) for location in ["east", "west", "north", "south", "top", "bottom"])
        self.neumann_functions          = dict((location, {"rho": 0.0, "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}) for location in ["east", "west", "north", "south", "top", "bottom"])

        # TODO SPLIT UP ALL OF THIS
        
        for location in ["east", "west", "north", "south", "top", "bottom"]:

            # TRANSFORM WALL VELOCITY FUNCTIONS
            if "wall_velocity_functions" in self.case_setup["boundary_condition"].keys():
                if location in self.case_setup["boundary_condition"]["wall_velocity_functions"].keys():
                    wall_velocity_functions = self.case_setup["boundary_condition"]["wall_velocity_functions"][location]

                    # MULTIPLE BOUNDARY TYPES AT LOCATION
                    if type(wall_velocity_functions) == list:
                        self.wall_velocity_functions[location] = []
                        for functions in wall_velocity_functions:
                            wall_velocity_dict = {}
                            for velocity in functions.keys():
                                wall_velocity_dict[velocity] = eval(functions[velocity]) if type(functions[velocity]) == str else functions[velocity] 
                            self.wall_velocity_functions[location].append(wall_velocity_dict)

                    # SINGLE BOUNDARY TYPE AT LOCATION
                    else:
                        self.wall_velocity_functions[location] = {}
                        for velocity in wall_velocity_functions.keys():
                            if type(wall_velocity_functions[velocity]) == str:
                                self.wall_velocity_functions[location][velocity] = eval(wall_velocity_functions[velocity])
                            else:
                                self.wall_velocity_functions[location][velocity] = wall_velocity_functions[velocity]

            # TRANSFORM DIRICHLET FUNCTIONS
            if "dirichlet_functions" in self.case_setup["boundary_condition"].keys():
                if location in self.case_setup["boundary_condition"]["dirichlet_functions"].keys():
                    dirichlet_functions = self.case_setup["boundary_condition"]["dirichlet_functions"][location]

                    # MULTIPLE BOUNDARY TYPES AT LOCATION
                    if type(dirichlet_functions) == list:
                        self.dirichlet_functions[location] = []
                        for functions in dirichlet_functions:
                            prime_dict = {}
                            for prime in ["rho", "u", "v", "w", "p"]:
                                prime_dict[prime] = eval(functions[prime]) if type(functions[prime]) == str else functions[prime] 
                            self.dirichlet_functions[location].append(prime_dict)

                    # SINGLE BOUNDARY TYPE AT LOCATION
                    else:
                        for prime in ["rho", "u", "v", "w", "p"]:
                            if type(dirichlet_functions[prime]) == str:
                                self.dirichlet_functions[location][prime] = eval(dirichlet_functions[prime])
                            else:
                                self.dirichlet_functions[location][prime] = dirichlet_functions[prime]

            # TRANSFORM NEUMANN FUNCTIONS
            if "neumann_functions" in self.case_setup["boundary_condition"].keys():
                if location in self.case_setup["boundary_condition"]["neumann_functions"].keys():
                    neumann_functions = self.case_setup["boundary_condition"]["neumann_functions"][location]
                    
                    # MULTIPLE BOUNDARY TYPES AT LOCATION
                    if type(neumann_functions) == list:
                        self.neumann_functions[location] = []
                        for functions in neumann_functions:
                            prime_dict = {}
                            for prime in ["rho", "u", "v", "w", "p"]:
                                prime_dict[prime] = eval(functions[prime]) if type(functions[prime]) == str else functions[prime] 
                            self.neumann_functions[location].append(prime_dict)

                    # SINGLE BOUNDARY TYPE AT LOCATION
                    else:
                        for prime in ["rho", "u", "v", "w", "p"]:
                            if type(neumann_functions[prime]) == str:
                                self.neumann_functions[location][prime] = eval(neumann_functions[prime])
                            else:
                                self.neumann_functions[location][prime] = neumann_functions[prime]

        # INITIAL CONDITION FROM RESTART FILE
        self.restart_flag       = self.case_setup["restart"]["flag"] if "restart" in self.case_setup.keys() else False
        self.restart_file_path  = self.case_setup["restart"]["file_path"] if "restart" in self.case_setup.keys() else None

        # TURBULENT INITIAL CONDITION
        self.is_turb_init       = True if "turb_init_params" in self.case_setup["initial_condition"].keys() else False
        self.turb_init_params   = self.case_setup["initial_condition"]["turb_init_params"] if "turb_init_params" in self.case_setup["initial_condition"].keys() else None
        assert not (self.is_turb_init and self.levelset_type != None), "Turbulent initialization not possible with active levelset."

        # INITIAL CONDITION FROM CASE SETUP FILE - ONLY USED IF RESTART AND TURBULENT INITIAL CONDITION ARE FALSE
        self.initial_condition_from_case_setup = False
        if self.is_turb_init == False and self.restart_flag == False:

            self.initial_condition_from_case_setup = True
            
            # INITIAL PRIMITIVE VARIABLES
            if self.levelset_type != None:
                assert "primes" in self.case_setup["initial_condition"].keys(), "No primes in initial condition"
                if self.levelset_type == "FLUID-FLUID":
                    self.initial_condition = {}
                    for fluid in ["positive", "negative"]:
                        self.initial_condition[fluid] = dict((prime, eval(value)) if type(value) == str else (prime, value) for prime, value in self.case_setup["initial_condition"]["primes"][fluid].items())     
                elif self.levelset_type in ["FLUID-SOLID-STATIC", "FLUID-SOLID-DYNAMIC"]:
                    self.initial_condition = dict((prime, eval(value)) if type(value) == str else (prime, value) for prime, value in self.case_setup["initial_condition"]["primes"].items())     
            else:
                self.initial_condition = dict((prime, eval(value)) if type(value) == str else (prime, value) for prime, value in self.case_setup["initial_condition"].items())     

            # INITIAL LEVELSET
            if self.levelset_type != None:
                assert "levelset" in self.case_setup["initial_condition"].keys(), "Levelset is active, however, no initial levelset is provided."
                if type(self.case_setup["initial_condition"]["levelset"]) == str:
                    self.initial_levelset = eval(self.case_setup["initial_condition"]["levelset"])
                elif type(self.case_setup["initial_condition"]["levelset"] == list):
                    self.initial_levelset = []
                    for i, levelset_object in enumerate(self.case_setup["initial_condition"]["levelset"]):
                        temp = {}
                        temp["shape"] = levelset_object["shape"]
                        temp["parameters"] = levelset_object["parameters"]
                        if type(levelset_object["bounding_domain"]) == str:
                            temp["bounding_domain"] = eval(levelset_object["bounding_domain"])
                        else:
                            assert False, "Wrong type for bounding_domain of solid levelset object %d. Must be lambda (str)." % (i)
                        self.initial_levelset.append(temp)
                else:
                    assert False, "Type of initial_levelset must be lambda (str) or a list of solid levelset objects."
            else:
                self.initial_levelset = None
        
        # SOLID INTERFACE VELOCITY
        if self.levelset_type == "FLUID-SOLID-DYNAMIC":
            assert "solid_interface_velocity" in self.case_setup.keys(), "Levelset is FLUID-SOLID-DYNAMIC, however, no solid interface velocity is provided."
            if type(self.case_setup["solid_interface_velocity"]) == str: 
                self.solid_interface_velocity = eval(self.case_setup["solid_interface_velocity"])
            elif type(self.case_setup["solid_interface_velocity"] == list):
                self.solid_interface_velocity = []
                for i, velocity_object in enumerate(self.case_setup["solid_interface_velocity"]):
                    temp = {}
                    if type(velocity_object["function"]) == str:
                        temp["function"] = eval(velocity_object["function"])
                    else:
                        assert False, "Wrong type for function of solid velocity object %d. Must be lambda (str)." % (i)
                    if type(velocity_object["bounding_domain"]) == str:
                        temp["bounding_domain"] = eval(velocity_object["bounding_domain"])
                    else:
                        assert False, "Wrong type for bounding_domain of solid velocity object %d. Must be lambda (str)." % (i)
                    self.solid_interface_velocity.append(temp)
            else:
                assert False, "Type of solid_levelset must be lambda (str) or a list of solid velocity objects."
        else:
            self.solid_interface_velocity = None

        # PHYSICAL PARAMETERS
        self.gravity = jnp.array(self.case_setup["gravity"]) if "gravity" in self.case_setup.keys() else jnp.array([0.0, 0.0, 0.0])

        # MATERIAL PROPERTIES
        self.material_properties = {}
        if self.levelset_type == "FLUID-FLUID":
            for fluid in ["positive", "negative"]:
                self.material_properties[fluid] = {}
                self.material_properties[fluid]["type"]                     = self.case_setup["material_properties"][fluid]["type"]
                self.material_properties[fluid]["dynamic_viscosity"]        = eval(self.case_setup["material_properties"][fluid]["dynamic_viscosity"]) if type(self.case_setup["material_properties"][fluid]["dynamic_viscosity"]) == str and self.case_setup["material_properties"][fluid]["dynamic_viscosity"] != "Sutherland" else self.case_setup["material_properties"][fluid]["dynamic_viscosity"]
                self.material_properties[fluid]["sutherland_parameters"]    = self.case_setup["material_properties"][fluid]["sutherland_parameters"] if "sutherland_parameters" in self.case_setup["material_properties"][fluid].keys() else None
                self.material_properties[fluid]["bulk_viscosity"]           = self.case_setup["material_properties"][fluid]["bulk_viscosity"]
                self.material_properties[fluid]["thermal_conductivity"]     = eval(self.case_setup["material_properties"][fluid]["thermal_conductivity"]) if type(self.case_setup["material_properties"][fluid]["thermal_conductivity"]) == str and self.case_setup["material_properties"][fluid]["thermal_conductivity"] != "Prandtl" else self.case_setup["material_properties"][fluid]["thermal_conductivity"]
                self.material_properties[fluid]["prandtl_number"]           = self.case_setup["material_properties"][fluid]["prandtl_number"] if "prandtl_number" in self.case_setup["material_properties"][fluid].keys() else None
                self.material_properties[fluid]["specific_heat_ratio"]      = self.case_setup["material_properties"][fluid]["specific_heat_ratio"]
                self.material_properties[fluid]["specific_gas_constant"]    = self.case_setup["material_properties"][fluid]["specific_gas_constant"]
                self.material_properties[fluid]["background_pressure"]      = self.case_setup["material_properties"][fluid]["background_pressure"] if "background_pressure" in self.case_setup["material_properties"][fluid].keys() else None
            self.material_properties["pairing"] = {}
            self.material_properties["pairing"]["surface_tension_coefficient"] = 0.0
            if "pairing" in self.case_setup["material_properties"].keys():
                if "surface_tension_coefficient" in self.case_setup["material_properties"]["pairing"].keys():
                    self.material_properties["pairing"]["surface_tension_coefficient"] = self.case_setup["material_properties"]["pairing"]["surface_tension_coefficient"]

        else:
            self.material_properties = {}
            self.material_properties["type"]                     = self.case_setup["material_properties"]["type"]
            self.material_properties["dynamic_viscosity"]        = eval(self.case_setup["material_properties"]["dynamic_viscosity"]) if type(self.case_setup["material_properties"]["dynamic_viscosity"]) == str and self.case_setup["material_properties"]["dynamic_viscosity"] != "Sutherland" else self.case_setup["material_properties"]["dynamic_viscosity"]
            self.material_properties["sutherland_parameters"]    = self.case_setup["material_properties"]["sutherland_parameters"] if "sutherland_parameters" in self.case_setup["material_properties"].keys() else None
            self.material_properties["bulk_viscosity"]           = self.case_setup["material_properties"]["bulk_viscosity"]
            self.material_properties["thermal_conductivity"]     = eval(self.case_setup["material_properties"]["thermal_conductivity"]) if type(self.case_setup["material_properties"]["thermal_conductivity"]) == str and self.case_setup["material_properties"]["thermal_conductivity"] != "Prandtl" else self.case_setup["material_properties"]["thermal_conductivity"]
            self.material_properties["prandtl_number"]           = self.case_setup["material_properties"]["prandtl_number"] if "prandtl_number" in self.case_setup["material_properties"].keys() else None
            self.material_properties["specific_heat_ratio"]      = self.case_setup["material_properties"]["specific_heat_ratio"]
            self.material_properties["specific_gas_constant"]    = self.case_setup["material_properties"]["specific_gas_constant"]
            self.material_properties["background_pressure"]      = self.case_setup["material_properties"]["background_pressure"] if "background_pressure" in self.case_setup["material_properties"].keys() else None

        # NONDIMENSIONALIZATION PARAMETERS
        self.nondimensionalization_parameters = {}
        self.nondimensionalization_parameters["density_reference"]      = self.case_setup["nondimensionalization_parameters"]["density_reference"]
        self.nondimensionalization_parameters["length_reference"]       = self.case_setup["nondimensionalization_parameters"]["length_reference"]
        self.nondimensionalization_parameters["velocity_reference"]     = self.case_setup["nondimensionalization_parameters"]["velocity_reference"]
        self.nondimensionalization_parameters["temperature_reference"]  = self.case_setup["nondimensionalization_parameters"]["temperature_reference"]

        # FORCINGS
        if "forcings" in self.case_setup.keys():
            self.mass_flow_target       = eval(self.case_setup["forcings"]["mass_flow_target"]) if "mass_flow_target" in self.case_setup["forcings"].keys() and type(self.case_setup["forcings"]["mass_flow_target"]) == str else self.case_setup["forcings"]["mass_flow_target"] if "mass_flow_target" in self.case_setup["forcings"].keys() else None
            self.mass_flow_direction    = self.case_setup["forcings"]["mass_flow_direction"] if "mass_flow_direction" in self.case_setup["forcings"].keys() else None 
            self.temperature_target     = eval(self.case_setup["forcings"]["temperature_target"]) if "temperature_target" in self.case_setup["forcings"].keys() and type(self.case_setup["forcings"]["temperature_target"]) == str else self.case_setup["forcings"]["temperature_target"] if "temperature_target" in self.case_setup["forcings"].keys() else None
        else:
            self.mass_flow_target       = None
            self.mass_flow_direction    = None
            self.temperature_target     = None

        # NUMERICAL SETUP DEFAULT VALUES
        self.numerical_setup["output"]["logging"] = "INFO" if not "logging" in self.numerical_setup["output"].keys() else self.numerical_setup["output"]["logging"]

        if "active_forcings" in self.numerical_setup.keys():
            self.numerical_setup["active_forcings"]["is_mass_flow_forcing"] = False if not "is_mass_flow_forcing" in self.numerical_setup["active_forcings"].keys() else self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]
            self.numerical_setup["active_forcings"]["is_temperature_forcing"] = False if not "is_temperature_forcing" in self.numerical_setup["active_forcings"].keys() else self.numerical_setup["active_forcings"]["is_temperature_forcing"]
            self.numerical_setup["active_forcings"]["is_turb_hit_forcing"] = False if not "is_turb_hit_forcing" in self.numerical_setup["active_forcings"].keys() else self.numerical_setup["active_forcings"]["is_turb_hit_forcing"]
        else:
            self.numerical_setup["active_forcings"] = {}
            self.numerical_setup["active_forcings"]["is_mass_flow_forcing"] = False
            self.numerical_setup["active_forcings"]["is_temperature_forcing"] = False
            self.numerical_setup["active_forcings"]["is_turb_hit_forcing"] = False
        self.active_forcings = jnp.array([self.numerical_setup["active_forcings"][forcing] for forcing in self.numerical_setup["active_forcings"]]).any() 

        self.numerical_setup["output"]["is_double_precision_compute"] = True if not "is_double_precision_compute" in self.numerical_setup["output"].keys() else self.numerical_setup["output"]["is_double_precision_compute"]
        self.numerical_setup["output"]["is_double_precision_output"] = True if not "is_double_precision_output" in self.numerical_setup["output"].keys() else self.numerical_setup["output"]["is_double_precision_output"]
        self.numerical_setup["output"]["is_xdmf"] = False if not "is_xdmf" in self.numerical_setup["output"].keys() else self.numerical_setup["output"]["is_xdmf"]
        self.numerical_setup["output"]["derivative_stencil"] = "DC4" if not "derivative_stencil" in self.numerical_setup["output"].keys() else self.numerical_setup["output"]["derivative_stencil"]
        self.numerical_setup["output"]["quantities"] =  {"primes": ["density", "velocityX", "velocityY", "velocityZ", "pressure", "temperature"]} if not "quantities" in self.numerical_setup["output"].keys() else self.numerical_setup["output"]["quantities"]
        
        # AVAILABLE OUTPUT QUANTITIES
        self.available_quantities = {
            "primes" : ["density", "velocity", "velocityX", "velocityY", "velocityZ", "pressure", "temperature"],
            "cons": ["mass", "momentum", "momentumX", "momentumY", "momentumZ", "energy"],
            "levelset": ["levelset", "volume_fraction", "mask_real", "normal", "interface_pressure", "interface_velocity"],
            "real_fluid": [ "density", "velocity", "velocityX", "velocityY", "velocityZ", "pressure", "temperature",
                            "mass", "momentum", "momentumX", "momentumY", "momentumZ", "energy" ],
            "miscellaneous": ["mach_number", "schlieren", "absolute_velocity", "vorticity", "absolute_vorticity"],
        }

        # THOROUGH SANITY CHECK
        self.sanity_check()

    def info(self) -> Tuple[Dict, Dict]:

        numerical_setup_dict = {}
        for key0, item0 in self.numerical_setup.items():
            if isinstance(item0, dict):
                key0_ = str(key0).replace("_", " ").upper()
                numerical_setup_dict[key0_] = {}
                for key1, item1 in item0.items():
                    key1_ = str(key1).replace("_", " ").upper()
                    if isinstance(item1, dict):
                        numerical_setup_dict[key0_][key1_] = {}
                        for key2, item2 in item1.items():
                            key2_ = str(key2).replace("_", " ").upper()
                            numerical_setup_dict[key0_][key1_][key2_] = item2
                    else:
                        numerical_setup_dict[key0_][key1_] = item1 

        case_setup_dict = {}
        for key0, item0 in self.case_setup.items():
            if isinstance(item0, dict):
                key0_ = str(key0).replace("_", " ").upper()
                case_setup_dict[key0_] = {}
                for key1, item1 in item0.items():
                    key1_ = str(key1).replace("_", " ").upper()
                    if isinstance(item1, dict):
                        case_setup_dict[key0_][key1_] = {}
                        for key2, item2 in item1.items():
                            key2_ = str(key2).replace("_", " ").upper()
                            if isinstance(item2, dict):
                                case_setup_dict[key0_][key1_][key2_] = {}
                                for key3, item3 in item2.items():
                                    key3_ = str(key3).replace("_", " ").upper()
                                    case_setup_dict[key0_][key1_][key2_][key3_] = item3
                            else:
                                case_setup_dict[key0_][key1_][key2_] = item2
                    else:
                        case_setup_dict[key0_][key1_] = item1

        return numerical_setup_dict, case_setup_dict

    def sanity_check(self) -> None:
        """Checks if the case setup and numerical setup is consistent.
        """

        # DOMAIN
        for i in self.active_axis_indices:
            assert self.number_of_cells[i] >= self.nh_conservatives, "Number of cells in %d direction must be >= halo cells"
        if self.levelset_type != None:
            assert self.nh_conservatives > self.nh_geometry, "halo cells for conservatives must be > halo cells for geometry"

        # TIME INTEGRATOR 
        assert self.numerical_setup["conservatives"]["time_integration"]["integrator"] in DICT_TIME_INTEGRATION.keys(), "Time integrator %s not implemented. Choose from %s." %(self.numerical_setup["conservatives"]["time_integration"]["time_integrator"], ", ".join(DICT_TIME_INTEGRATION.keys()))
        assert type(self.numerical_setup["conservatives"]["time_integration"]["CFL"]) in [float, np.float32, np.float64, jnp.float32, jnp.float64], "CFL type must be float."
        assert self.numerical_setup["conservatives"]["time_integration"]["CFL"] > 0.0, "CFL number must be > 0.0."
        if "fixed_timestep" in self.numerical_setup["conservatives"]["time_integration"].keys():
            assert type(self.numerical_setup["conservatives"]["time_integration"]["fixed_timestep"]) in [float, np.float32, np.float64, jnp.float32, jnp.float64], "CFL type must be float."
            assert self.numerical_setup["conservatives"]["time_integration"]["fixed_timestep"] > 0.0, "Fixed timestep must be > 0.0."

        # CONVECTIVE SOLVER
        assert self.numerical_setup["conservatives"]["convective_fluxes"]["convective_solver"] in ["GODUNOV", "FLUX-SPLITTING", "ALDM"], "Convective solver %s not implemented. Choose from GODUNOV, FLUX-SPLITTING, ALDM." %(self.numerical_setup["conservatives"]["convective_fluxes"]["convective_solver"])

        # GODUNOV 
        if self.numerical_setup["conservatives"]["convective_fluxes"]["convective_solver"] == "GODUNOV":
            assert self.numerical_setup["conservatives"]["convective_fluxes"]["riemann_solver"] in DICT_RIEMANN_SOLVER.keys(), "Riemann solver %s not implemented for GODUNOV convective solver. Choose from %s." %(self.numerical_setup["conservatives"]["convective_fluxes"]["riemann_solver"], ", ".join(DICT_RIEMANN_SOLVER.keys()))
            assert self.numerical_setup["conservatives"]["convective_fluxes"]["signal_speed"] in DICT_SIGNAL_SPEEDS.keys(), "Signal speed %s not implemented for GODUNOV convective solver. Choose from %s." %(self.numerical_setup["conservatives"]["convective_fluxes"]["signal_speed"], ", ".join(DICT_SIGNAL_SPEEDS.keys()))
            assert self.numerical_setup["conservatives"]["convective_fluxes"]["reconstruction_var"] in ["PRIMITIVE", "CONSERVATIVE", "CHAR-PRIMITIVE", "CHAR-CONSERVATIVE"], "Reconstruction var %s not implemented. Choose from %s." %(self.numerical_setup["conservatives"]["convective_fluxes"]["reconstruction_var"], ", ".join(["PRIMITIVE", "CONSERVATIVE", "CHAR-PRIMITIVE", "CHAR-CONSERVATIIVE"]))

        # FLUX SPLITTING 
        elif self.numerical_setup["conservatives"]["convective_fluxes"]["convective_solver"] == "FLUX-SPLITTING":
            assert self.numerical_setup["conservatives"]["convective_fluxes"]["flux_splitting"] in ["ROE", "CLLF", "LLF", "GLF"], "Flux splitting %s not implemented for FLUX-SPLITTING convective solver. Choose from %s." %(self.numerical_setup["conservatives"]["convective_fluxes"]["riemann_solver"], ", ".join(["ROE", "CLLF", "LLF", "GLF"]))

        # SPATIAL RECONSTRUCTOR
        if self.numerical_setup["conservatives"]["convective_fluxes"]["convective_solver"] not in ["ALDM"]:
            assert self.numerical_setup["conservatives"]["convective_fluxes"]["spatial_reconstructor"] in DICT_SPATIAL_RECONSTRUCTION.keys(), "Spatial reconstruction %s not implemented. Choose from %s." %(self.numerical_setup["conservatives"]["convective_fluxes"]["spatial_reconstructor"], ", ".join(DICT_SPATIAL_RECONSTRUCTION))

        # CHECK DERIVATIVE STENCILS
        if self.numerical_setup["active_physics"]["is_viscous_flux"] or self.numerical_setup["active_physics"]["is_heat_flux"]:
            assert self.numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_face"] in DICT_DERIVATIVE_FACE.keys(), "Derivative stencil %s for cell-face derivative not implemented. Choose from %s." % (self.numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_face"], ", ".join(DICT_DERIVATIVE_FACE))
            assert self.numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_center"] in DICT_FIRST_DERIVATIVE_CENTER.keys(), "Derivative stencil %s for cell-center derivative not implemented. Choose from %s." %(self.numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_center"], ", ".join(DICT_FIRST_DERIVATIVE_CENTER))
            assert self.numerical_setup["conservatives"]["dissipative_fluxes"]["reconstruction_stencil"] in DICT_CENTRAL_RECONSTRUCTION.keys(), "Central reconstruction %s not implemented. Choose from %s." %(self.numerical_setup["conservatives"]["dissipative_fluxes"]["reconstruction_stencil"], ", ".join(DICT_CENTRAL_RECONSTRUCTION))

        assert self.numerical_setup["output"]["derivative_stencil"] in DICT_FIRST_DERIVATIVE_CENTER.keys(), "Derivative stencil %s for outut writer not implemented. Choose from %s." %(self.numerical_setup["output"]["derivative_stencil"], ", ".join(DICT_FIRST_DERIVATIVE_CENTER))

        # CELLS AND DOMAIN LENGTH CHECK
        for i, n in enumerate(self.number_of_cells):
            if n < 1:
                assert False, "Number of cells in %s direction is smaller than 1" % self.axis[i]

        for key, item in self.domain_size.items():
            assert item[0] < item[1], "Domain bound %s_max is smaller than %s_min." % (key, key)

        # DIMENSION CHECK
        assert self.dim == len([n for n in self.number_of_cells if n > 1]), "Dimension and number of cells do not match"
        
        # GENERAL BOUNDARY TYPE CHECK
        for i in range(2 if self.levelset_type != None else 1):

            quantity = ["primes", "levelset"][i]
            boundary_location_types = self.boundary_location_types[quantity] if self.levelset_type != None else self.boundary_location_types

            for location_types in boundary_location_types.items():
                
                location            = location_types[0]
                types_and_ranges    = location_types[1]

                # MULTIPLE BOUNDARIES AT SAME LOCATION - TYPE AND DIMENSION CHECK
                if type(types_and_ranges) == list:
                    assert quantity != "levelset", "Multiple boundary types at same location for levelset not implemented."
                    assert self.dim == 2, "Multiple boundary types at same location only implemented in 2D."
                    b_types = types_and_ranges[0]
                    for b_type in b_types:
                        assert b_type in ["symmetry", "periodic", "dirichlet", "neumann", "wall", "inactive"], "Boundary type %s does not exist." % b_type
                        assert b_type != "periodic", "It is not possible to use boundary type periodic at a location with multiple types."
                else:
                    b_type = location_types[1]
                    
                    # GENERAL TYPE CHECK
                    if quantity == "primes":
                        assert b_type in ["symmetry", "periodic", "dirichlet", "neumann", "wall", "inactive"], "Boundary type %s does not exist for primes" % b_type
                    elif quantity == "levelset":
                        assert b_type in ["symmetry", "periodic", "neumann", "inactive"], "Boundary type %s does not exist for levelset" % b_type
                    
                    # PERIODIC AT OPPOSITE LOCATION CHECK
                    if b_type == "periodic":
                        if location in ["east", "west"]:
                            assert boundary_location_types["east"] == boundary_location_types["west"], "%s %s boundary is periodic, however the opposite boundary is not." % (quantity, location)
                        if location in ["north", "south"]:
                            assert boundary_location_types["north"] == boundary_location_types["south"], "%s %s boundary is periodic, however the opposite boundary is not." % (quantity, location)
                        if location in ["top", "bottom"]:
                            assert boundary_location_types["top"] == boundary_location_types["bottom"], "%s %s boundary is periodic, however the opposite boundary is not." % (quantity, location)

            # INACTIVE AXIS BOUNDARY TYPE CHECK
            for axis in self.inactive_axis:
                if axis == "x":
                    for location in ["east", "west"]:
                        if boundary_location_types[location] != "inactive":
                            assert False, "Axis %s is inactive, however the boundary type for the %s boundary is %s. Change to inactive" % (axis, location, boundary_location_types[location])
                if axis == "y":
                    for location in ["north", "south"]:
                        if boundary_location_types[location] != "inactive":
                            assert False, "Axis %s is inactive, however the boundary type for the %s boundary is %s. Change to inactive" % (axis, location, boundary_location_types[location])
                if axis == "z":
                    for location in ["top", "bottom"]:
                        if boundary_location_types[location] != "inactive":
                            assert False, "Axis %s is inactive, however the boundary type for the %s boundary is %s. Change to inactive" % (axis, location, boundary_location_types[location])

            # ACTIVE AXIS BOUNDARY TYPE CHECK
            for axis in self.active_axis:
                if axis == "x":
                    for location in ["east", "west"]:
                        if boundary_location_types[location] == "inactive":
                            assert False, "Axis %s is active, however the boundary type for the %s boundary is inactive." % (axis, location)
                if axis == "y":
                    for location in ["north", "south"]:
                        if boundary_location_types[location] == "inactive":
                            assert False, "Axis %s is active, however the boundary type for the %s boundary is inactive." % (axis, location)
                if axis == "z":
                    for location in ["top", "bottom"]:
                        if boundary_location_types[location] == "inactive":
                            assert False, "Axis %s is active, however the boundary type for the %s boundary is inactive." % (axis, location)
        
        
        boundary_types_location = self.boundary_location_types["primes"] if self.levelset_type != None else self.boundary_location_types

        # WALL VELOCITY FUNCTION CHECK
        for location in self.wall_velocity_functions:
            wall_velocity_functions = self.wall_velocity_functions[location]
            if type(wall_velocity_functions) == list:
                assert type(boundary_types_location[location]) == list, "Amount of wall velocity functions and wall types at location %s does not match." % location
                assert boundary_types_location[location][0].count("wall") == len(wall_velocity_functions), "Amount of wall velocity functions and wall types at location %s does not match." % location
                for function in wall_velocity_functions:
                    assert function.keys() == self.location_to_wall_velocity[location].keys(), "Wall velocity function at location %s has keys %s, however, the keys must be %s" % (location, function.keys(), self.location_to_wall_velocity[location].keys())
                    for velocity in function:
                        if type(function[velocity]) == types.LambdaType:
                            argcount = function[velocity].__code__.co_argcount
                            assert argcount == 1, "Wall velocity lambda at location %s for velocity %s takes %d arguments, however, must take 1 argument (time). Use float for constant wall velocity." % (location, velocity, argcount)
            else:
                for velocity in wall_velocity_functions:
                    assert wall_velocity_functions.keys() == self.location_to_wall_velocity[location].keys(), "Wall velocity function at location %s has keys %s, however, the keys must be %s" % (location, wall_velocity_functions.keys(), self.location_to_wall_velocity[location].keys())
                    if type(wall_velocity_functions[velocity]) == types.LambdaType:
                        argcount = wall_velocity_functions[velocity].__code__.co_argcount
                        assert argcount == 1, "Wall velocity lambda at location %s for velocity %s takes %d arguments, however, must take 1 argument (time). Use float for constant wall velocity." % (location, velocity, argcount)

        # DIRICHLET TYPE AND FUNCTION CHECK
        for location in self.dirichlet_functions:
            dirichlet_functions = self.dirichlet_functions[location]
            if type(dirichlet_functions) == list:
                assert type(boundary_types_location[location]) == list, "Amount of dirichlet functions and dirichlet types at location %s does not match." % location
                assert boundary_types_location[location][0].count("dirichlet") == len(dirichlet_functions), "Amount of dirichlet functions and dirichlet types at location %s does not match." % location
                for function in dirichlet_functions:
                    for prime in function:
                        if type(function[prime]) == types.LambdaType:
                            argcount = function[prime].__code__.co_argcount
                            if argcount != self.dim:
                                assert False, "Dirichlet lambda at location %s for prime state %s takes %d arguments, however the dimension is %d." % (location, prime, argcount, self.dim)
            else:
                for prime in dirichlet_functions:
                    if type(dirichlet_functions[prime]) == types.LambdaType:
                        argcount = dirichlet_functions[prime].__code__.co_argcount
                        if argcount != self.dim:
                            assert False, "Dirichlet lambda at location %s for prime state %s takes %d arguments, however the dimension is %d." % (location, prime, argcount, self.dim)

        # NEUMANN TYPE AND FUNCTION CHECK
        for location in self.neumann_functions:
            neumann_functions = self.neumann_functions[location]
            if type(neumann_functions) == list:
                assert type(boundary_types_location[location]) == list, "Amount of neumann functions and neumann types at location %s does not match." % location
                assert boundary_types_location[location][0].count("neumann") == len(neumann_functions), "Amount of neumann functions and neumann types at location %s does not match." % location
                for function in neumann_functions:
                    for prime in function:
                        if type(function[prime]) == types.LambdaType:
                            argcount = function[prime].__code__.co_argcount
                            if argcount != self.dim:
                                assert False, "Neumann lambda at location %s for prime state %s takes %d arguments, however the dimension is %d." % (location, prime, argcount, self.dim)
            else:
                for prime in neumann_functions:
                    if type(neumann_functions[prime]) == types.LambdaType:
                        argcount = neumann_functions[prime].__code__.co_argcount
                        if argcount != self.dim:
                            assert False, "Neumann lambda at location %s for prime state %s takes %d arguments, however the dimension is %d." % (location, prime, argcount, self.dim)

        # ARGUMENTS INITIAL CONDITION CHECK
        if self.initial_condition_from_case_setup:
            no_phases = 2 if self.levelset_type == "FLUID-FLUID" else 1
            for i in range(no_phases):
                initial_condition = self.initial_condition[["positive", "negative"][i]] if self.levelset_type == "FLUID-FLUID" else self.initial_condition
                for prime in initial_condition:
                    initial_prime_value = initial_condition[prime]
                    if type(initial_prime_value) == types.LambdaType:
                        argcount = initial_prime_value.__code__.co_argcount
                        assert argcount == self.dim, "Initial condition lambda for prime state %s takes %d arguments, however the dimension is %d" % (key, argcount, self.dim)
                    else:
                        assert type(initial_prime_value) in [float, np.float32, np.float64, jnp.float32, jnp.float64], "Type of initial condition must be float or lambda (str)"

        # MATERIAL PARAMETER CHECK
        for fluid in ["positive", "negative"]:

            material_properties = self.material_properties[fluid] if self.levelset_type == "FLUID-FLUID" else self.material_properties

            assert material_properties["type"] in DICT_MATERIAL.keys(), "Material type %s not implemented. Choose from %s" % (material_properties["type"], DICT_MATERIAL.keys())
            assert material_properties["specific_gas_constant"]     >= 0.0, "Specific heat ratio must be >= 0.0."
            assert material_properties["specific_heat_ratio"]       >= 0.0, "Specific heat ratio must be >= 0.0."
            if material_properties["type"] == "StiffenedGas":
                assert material_properties["background_pressure"] != None, "Material is StiffenedGas, but no background pressure is provided."

            if type(material_properties["dynamic_viscosity"]) in [float, np.float32, np.float64]:
                assert material_properties["dynamic_viscosity"] >= 0.0, "Dynamic viscosity must be >= 0.0."
            
            elif material_properties["dynamic_viscosity"] == "Sutherland":
                assert type(material_properties["sutherland_parameters"]) == list, "Sutherland parameters must be a list with [mu_0, T_0, C]."
                assert len(material_properties["sutherland_parameters"]) == 3, "Sutherland parameters must be a list with [mu_0, T_0, C]."
            
            else:
                assert type(material_properties["dynamic_viscosity"]) == types.LambdaType, "Dynamic viscosity model must either be float, 'Sutherland', LambdaType."


            if type(material_properties["thermal_conductivity"]) in [float, np.float32, np.float64]:
                assert material_properties["thermal_conductivity"] >= 0.0, "Thermal conductivity must be >= 0.0."

            elif material_properties["thermal_conductivity"] == "Prandtl":
                assert material_properties["prandtl_number"] >= 0.0 if type(material_properties["prandtl_number"]) != None else False, "Prandtl number must be >= 0.0."  
            
            else:
                assert type(material_properties["thermal_conductivity"]) == types.LambdaType, "Thermal conductivity model must either be float, 'Prandtl', LambdaType."

            assert material_properties["bulk_viscosity"] >= 0.0, "Bulk viscosity must be >= 0.0."

        # TURB INIT CHECK
        if self.is_turb_init:
            assert self.turb_init_params["turb_case"] in ["RISTORCELLI", "CHANNEL", "TGV"], "Turbulent initial condition %s does not exist." %(self.turb_init_params["turb_case"])
            assert self.turb_init_params is not None, "Please provide parameters for the chosen turbulent initial condition."


        # MASS FLOW FORCING 
        if self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]:
            assert self.mass_flow_target != None, "Mass flow forcing is true, however, mass flow target is None."
            if type(self.mass_flow_target) == types.LambdaType:
                argcount = self.mass_flow_target.__code__.co_argcount
                assert argcount == 1, "Mass flow target function has more than one input argument" % (argcount)

        # TEMPERATURE FORCING
        if self.numerical_setup["active_forcings"]["is_temperature_forcing"]:
            assert self.temperature_target != None, "Temperature forcing is true, however, temperature target is None."
            if type(self.temperature_target) == types.LambdaType:
                argcount = self.temperature_target.__code__.co_argcount
                assert argcount == self.dim + 1, "Temperature forcing is a lambda function. Dimension is %d, however, the amount of arguments for the lambda is %d." % (self.dim, argcount)

        # RESTART
        if self.restart_flag:
            assert os.path.exists(self.restart_file_path), "Restart flag is true, however restart file path does not exist."

        # LEVELSET
        if self.levelset_type != None:
            
            # GENERAL
            assert self.numerical_setup["levelset"]["interface_interaction"] in [None, "FLUID-SOLID-STATIC", "FLUID-SOLID-DYNAMIC", "FLUID-FLUID"], "interface_interaction must be in [None, FLUID-SOLID-STATIC, FLUID-SOLID-DYNAMIC, FLUID-FLUID]"
            
            assert self.initial_levelset != None, "Levelset is active, however, initial_levelset is None."
            if type(self.initial_levelset) == types.LambdaType:
                argcount = self.initial_levelset.__code__.co_argcount
                assert argcount == self.dim, "Initial levelset lambda takes %d arguments, however the dimension is %d." % (argcount, self.dim)
            elif type(self.initial_levelset) == list:
                for i, levelset_object in enumerate(self.initial_levelset):
                    argcount = levelset_object["bounding_domain"].__code__.co_argcount
                    assert argcount == self.dim, "Initial levelset object %d bounding domain lambda takes %d arguments, however the dimension is %d." % (i, argcount, self.dim)
            
            assert self.numerical_setup["levelset"]["geometry_calculator_stencil"] in DICT_FIRST_DERIVATIVE_CENTER.keys(), "Geometry calculator stencil %s not implemented. Choose from %s." %(self.numerical_setup["geometry_calculator_stencil"], ", ".join(DICT_FIRST_DERIVATIVE_CENTER))
            assert self.numerical_setup["levelset"]["volume_fraction_threshold"] >= 0.0, "Volume fraction threshold must be > 0.0."
            assert self.numerical_setup["levelset"]["levelset_advection_stencil"] in DICT_DERIVATIVE_LEVELSET_ADVECTION.keys(), "Levelset advection stencil %s not implemented. Choose from %s." % (self.numerical_setup["levelset_advection_stencil"], ", ".join(DICT_DERIVATIVE_LEVELSET_ADVECTION))
            assert self.numerical_setup["levelset"]["narrow_band_cutoff"] > 0, "Levelset cutoff narrow band must be > 0." 
            assert self.numerical_setup["levelset"]["narrow_band_computations"] > 0, "Levelset computations narrow band must be > 0." 
            
            # QUANTITY EXTENDER
            assert self.numerical_setup["levelset"]["extension"]["time_integrator"] in DICT_TIME_INTEGRATION.keys(), "Quantity extender time integrator %s not implemented. Choose from %s." %(self.numerical_setup["quantity_extender_time_integrator"], ", ".join(DICT_TIME_INTEGRATION.keys()))
            assert self.numerical_setup["levelset"]["extension"]["spatial_stencil"] in DICT_DERIVATIVE_QUANTITY_EXTENDER.keys(), "Quantity extender stencil %s not implemented. Choose from %s." %(self.numerical_setup["quantity_extender_stencil"], ", ".join(DICT_DERIVATIVE_QUANTITY_EXTENDER))
            assert self.numerical_setup["levelset"]["extension"]["CFL_primes"] > 0.0, "Primes extension CFL must be > 0."  
            assert self.numerical_setup["levelset"]["extension"]["CFL_interface"] > 0.0, "Interface extension CFL must be > 0."  
            assert self.numerical_setup["levelset"]["extension"]["steps_primes"] >= 0, "Primes extension steps must be > 0."  
            assert self.numerical_setup["levelset"]["extension"]["steps_interface"] >= 0, "Interface extension steps must be > 0."  

            # LEVELSET REINITIALIZER
            assert self.numerical_setup["levelset"]["reinitialization"]["time_integrator"] in DICT_TIME_INTEGRATION.keys(), "Levelset reinitializr time integrator %s not implemented. Choose from %s." %(self.numerical_setup["levelset_reinitializer_time_integrator"], ", ".join(DICT_TIME_INTEGRATION.keys()))
            assert self.numerical_setup["levelset"]["reinitialization"]["spatial_stencil"] in DICT_DERIVATIVE_REINITIALIZATION.keys(), "Levelset reinitializr stencil %s not implemented. Choose from %s." %(self.numerical_setup["levelset_reinitializer_stencil"], ", ".join(DICT_DERIVATIVE_REINITIALIZATION))
            assert self.numerical_setup["levelset"]["reinitialization"]["CFL"] > 0.0, "Reinitialization CFL must be > 0."  
            assert self.numerical_setup["levelset"]["reinitialization"]["interval"] > 0, "Reinitialization interval must be > 0."  
            assert self.numerical_setup["levelset"]["reinitialization"]["steps"] >= 0, "Reinitialization steps must be >= 0."  

            # SOLID VELOCITY
            dummy_cell_centers  = [jnp.zeros(self.number_of_cells[i]) for i in range(3)]
            dummy_mesh_grid     = [jnp.meshgrid(*dummy_cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
            if type(self.solid_interface_velocity) == types.LambdaType:
                argcount = self.solid_interface_velocity.__code__.co_argcount
                assert argcount == self.dim + 1, "Solid velocity lambda takes %d arguments, however the dimension is %d." % (argcount, self.dim)
                output = self.solid_interface_velocity(*dummy_mesh_grid, 0.0)
                assert type(output) == tuple, "Solid velocity lambda output must be tuple with velocities for each active axis."
                assert len(output) == self.dim, "Solid velocity lambda outputs %d velocities, however the dimension is %d." % (len(output), self.dim)
            elif type(self.solid_interface_velocity) == list:
                for i, velocity_object in enumerate(self.solid_interface_velocity):
                    velocity_function = velocity_object["function"]
                    argcount = velocity_function.__code__.co_argcount
                    assert argcount == self.dim + 1, "Solid velocity object %d lambda takes %d arguments, however the dimension is %d." % (i, argcount, self.dim)
                    output = velocity_function(*dummy_mesh_grid, 0.0)
                    assert type(output) == tuple, "Solid velocity object %d lambda output must be tuple with velocities for each active axis." % (i)
                    assert len(output) == self.dim, "Solid velocity object %d lambda outputs %d velocities, however the dimension is %d." % (i, len(output), self.dim)
                    argcount = velocity_object["bounding_domain"].__code__.co_argcount
                    assert argcount == self.dim + 1, "Solid velocity object %d bounding domain lambda takes %d arguments, however the dimension is %d." % (i, argcount, self.dim)

        # OUTPUT WRITER
        quantity_type_list = ["primes", "cons", "real_fluid", "miscellaneous", "levelset"]
        for quantity_type in self.numerical_setup["output"]["quantities"].keys():
            assert quantity_type in quantity_type_list, "Output quantity type must be in %s." % quantity_type_list
            if quantity_type == "real_fluid":
                assert self.levelset_type == "FLUID-FLUID", "Requested real fluid output, however, levelset_type is not FLUID-FLUID."
            for quantity in self.numerical_setup["output"]["quantities"][quantity_type]:
                if quantity == "interface_pressure":
                    assert self.levelset_type == "FLUID-FLUID", "Requested interface_pressure output, however, levelset_type is not FLUID-FLUID."
                if quantity == "interface_velocity":
                    assert self.levelset_type != "FLUID-SOLID-STATIC", "Requested interface_velocity output, however, levelset_type is FLUID-SOLID-STATIC."
                assert quantity in self.available_quantities[quantity_type], "Requested output quantity %s of type %s does not exist. Choose from %s." % (quantity, quantity_type, self.available_quantities[quantity_type]) 

        # LOGGER
        assert self.numerical_setup["output"]["logging"] in ["NONE", "INFO", "DEBUG", "INFO_TO_FILE", "DEBUG_TO_FILE"], "Logging must be in [NONE, INFO, DEBUG, INFO_TO_FILE, DEBUG_TO_FILE]"