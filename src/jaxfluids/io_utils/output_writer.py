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
import json
from typing import Dict, Tuple, Union
from functools import partial

import h5py
import jax
import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.input_reader import InputReader
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class OutputWriter:
    """Output writer for JAX-FLUIDS. The OutputWriter class can write h5 and xdmf 
    files. h5 and xdmf files can be visualized in paraview. Xdmf output is activated
    via the is_xdmf_file flag under the output keyword in the numerical setup.

    If the xdmf option is activated, a single xdmf file is written for each h5 file.
    Additionally, at the end of the simulation, an time series xdmf file is written
    which summarizes the whole simulation. This enables loading the entire timeseries
    into Paraview.
    """
    def __init__(self, input_reader: InputReader, unit_handler: UnitHandler, domain_information: DomainInformation,
        material_manager: MaterialManager, levelset_handler: LevelsetHandler, derivative_stencil_conservatives: SpatialDerivative,
        derivative_stencil_geometry : Union[SpatialDerivative, None]) -> None:

        # GENERAL
        self.case_name      = input_reader.case_name
        self.next_timestamp = unit_handler.non_dimensionalize(input_reader.save_dt, "time")
        self.save_dt        = unit_handler.non_dimensionalize(input_reader.save_dt, "time")
        self.save_path      = input_reader.save_path

        # JSON
        self.case_setup         = input_reader.case_setup
        self.numerical_setup    = input_reader.numerical_setup

        # NUMERICAL SETUP
        self.is_xdmf                            = self.numerical_setup["output"]["is_xdmf"]
        self.is_double                          = self.numerical_setup["output"]["is_double_precision_output"]
        self.output_quantities                  = self.numerical_setup["output"]["quantities"]
        self.is_mass_flow_forcing               = self.numerical_setup["active_forcings"]["is_mass_flow_forcing"]
        self.levelset_type                      = input_reader.levelset_type
        self.derivative_stencil_conservatives   = derivative_stencil_conservatives
        self.derivative_stencil_geometry        = derivative_stencil_geometry

        # MATERIAL, UNITHANDLER
        self.material_manager       = material_manager
        self.unit_handler           = unit_handler
        self.levelset_handler       = levelset_handler

        # DOMAIN INFORMATION
        self.number_of_cells                    = domain_information.number_of_cells
        self.cell_centers                       = domain_information.cell_centers
        self.cell_faces                         = domain_information.cell_faces
        self.cell_sizes                         = domain_information.cell_sizes
        self.nhx, self.nhy, self.nhz            = domain_information.domain_slices_conservatives
        self.nhx_, self.nhy_, self.nhz_         = domain_information.domain_slices_geometry
        self.nhx__, self.nhy__, self.nhz__      = domain_information.domain_slices_conservatives_to_geometry
        self.active_axis_indices                = domain_information.active_axis_indices

        # QUANTITIES
        self.quantity_index = {
            "primes": {"density": 0, "velocityX": 1, "velocityY": 2, "velocityZ": 3, "velocity": jnp.s_[1:4],  "pressure": 4, "temperature": 5},
            "cons": {"mass": 0, "momentumX": 1, "momentumY": 2, "momentumZ": 3, "momentum": jnp.s_[1:4], "energy": 4}
        }

        self.output_timeseries  = []
        self.xdmf_timeseries    = []

        self.save_path_case, self.save_path_domain = self.get_folder_name()

    def create_folder(self) -> None:
        """Sets up a folder for the simulation. Dumps the numerical setup and
        cas setup into the simulation folder and creates an output folder
        within in simulation folder into which simulation output is saved.

        simulation_folder
        ---- Numerical setup
        ---- Case setup
        ---- domain
        """
        os.mkdir(self.save_path_case)
        os.mkdir(self.save_path_domain)

        with open(os.path.join(self.save_path_case, self.case_name + ".json"), "w") as json_file:
            json.dump(self.case_setup, json_file, ensure_ascii=False, indent=4)
        with open(os.path.join(self.save_path_case, "numerical_setup.json"), "w") as json_file:
            json.dump(self.numerical_setup, json_file, ensure_ascii=False, indent=4)

    def get_folder_name(self) -> Tuple[str, str]:
        """Returns a name for the simulation folder based on the case name.

        :return: Path to the simulation folder and path to domain folder within
            simulation folder.
        :rtype: Tuple[str, str]
        """

        case_name_folder    = self.case_name

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        create_directory = True
        i = 1
        while create_directory:
            if os.path.exists(os.path.join(self.save_path, case_name_folder)):
                case_name_folder = self.case_name + "-%d" % i
                i += 1
            else:
                save_path_case     = os.path.join(self.save_path, case_name_folder)
                save_path_domain   = os.path.join(save_path_case, "domain")
                create_directory   = False

        return save_path_case, save_path_domain

    def write_output(self, buffer_dictionary: Dict[str, Dict[str, Union[jnp.ndarray, float]]], 
        force_output: bool, simulation_finish: bool = False) -> None:
        """Writes h5 and (optional) xdmf output.

        :param buffer_dictionary: Dictionary with flow field buffers
        :type buffer_dictionary: Dict[str, Dict[str, Union[jnp.ndarray, float]]]
        :param force_output: Flag which forces an output.
        :type force_output: bool
        :param simulation_finish: Flag that indicates the simulation finish -> 
            then timeseries xdmf is written, defaults to False
        :type simulation_finish: bool, optional
        """
        current_time = buffer_dictionary["time_control"]["current_time"]
        if force_output:
            self.write_h5file(buffer_dictionary)
            if self.is_xdmf:
                self.write_xdmffile(current_time)
        else:
            diff = current_time - self.next_timestamp
            if diff >= -jnp.finfo(jnp.float64).eps:
                self.write_h5file(buffer_dictionary)
                if self.is_xdmf:
                    self.write_xdmffile(current_time)
                self.next_timestamp += self.save_dt

        if simulation_finish and self.is_xdmf:
            self.write_timeseries_xdmffile()

    def write_h5file(self, buffer_dictionary: Dict[str, Dict[str, jnp.ndarray]]) -> None:

        current_time = buffer_dictionary["time_control"]["current_time"]

        filename = "data_%.6f.h5" % self.unit_handler.dimensionalize(current_time, "time")

        with h5py.File(os.path.join(self.save_path_domain, filename), "w") as h5file:

            # MESH DATA
            h5file.create_group(name="mesh")
            h5file.create_dataset(name="mesh/gridX",        data=self.unit_handler.dimensionalize(self.cell_centers[0], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/gridY",        data=self.unit_handler.dimensionalize(self.cell_centers[1], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/gridZ",        data=self.unit_handler.dimensionalize(self.cell_centers[2], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/gridFX",       data=self.unit_handler.dimensionalize(self.cell_faces[0], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/gridFY",       data=self.unit_handler.dimensionalize(self.cell_faces[1], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/gridFZ",       data=self.unit_handler.dimensionalize(self.cell_faces[2], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/cellsizeX",    data=self.unit_handler.dimensionalize(self.cell_sizes[0], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/cellsizeY",    data=self.unit_handler.dimensionalize(self.cell_sizes[1], "length"), dtype="f8")
            h5file.create_dataset(name="mesh/cellsizeZ",    data=self.unit_handler.dimensionalize(self.cell_sizes[2], "length"), dtype="f8")

            # CURRENT TIME
            h5file.create_dataset(name="time", data=self.unit_handler.dimensionalize(current_time, "time"), dtype="f8")

            # COMPUTE TEMPERATURE
            primes          = buffer_dictionary["material_fields"]["primes"]
            temperature     = jnp.expand_dims(self.material_manager.get_temperature(primes[4], primes[0]), axis=0)
            material_fields = {"primes": jnp.vstack([primes, temperature]), "cons": buffer_dictionary["material_fields"]["cons"]}

            # CONSERAVITVES AND PRIMITIVES
            for key in ["primes", "cons"]:
                if key in self.output_quantities.keys():
                    h5file.create_group(name=key)
                    for quantity in self.output_quantities[key]:
                        if self.levelset_type == "FLUID-FLUID":
                            for i in range(2):
                                quantity_name = "%s_%d" % (quantity, i)
                                buffer = self.unit_handler.dimensionalize(material_fields[key][self.quantity_index[key][quantity], i, self.nhx, self.nhy, self.nhz], quantity)
                                h5file.create_dataset(name="/".join([key, quantity_name]), data=buffer.T, dtype="f8" if self.is_double else "f4")
                        else:
                            buffer = self.unit_handler.dimensionalize(material_fields[key][self.quantity_index[key][quantity], self.nhx, self.nhy, self.nhz], quantity)
                            h5file.create_dataset(name="/".join([key, quantity]), data=buffer.T, dtype="f8" if self.is_double else "f4")


            if self.levelset_type != None:
                
                # LEVELSET QUANTITIES
                if "levelset" in self.output_quantities.keys():
                    h5file.create_group(name="levelset")
                    levelset_quantities = {}
                    levelset, volume_fraction   = buffer_dictionary["levelset_quantities"]["levelset"], buffer_dictionary["levelset_quantities"]["volume_fraction"]
                    normal                      = self.levelset_handler.geometry_calculator.compute_normal(levelset)
                    mask_real, _                = self.levelset_handler.compute_masks(levelset, volume_fraction)
                    if self.levelset_type == "FLUID-FLUID":
                        interface_velocity, interface_pressure, _ = self.levelset_handler.compute_interface_quantities(material_fields["primes"], levelset, volume_fraction)
                        levelset_quantities["interface_velocity"] = self.unit_handler.dimensionalize(interface_velocity, "velocity")
                        levelset_quantities["interface_pressure"] = self.unit_handler.dimensionalize(interface_pressure, "pressure")
                        mask_real = mask_real[0]
                    elif self.levelset_type == "FLUID-SOLID-DYNAMIC":
                        interface_velocity = self.levelset_handler.compute_solid_interface_velocity(current_time)
                        levelset_quantities["interface_velocity"] = self.unit_handler.dimensionalize(interface_velocity, "velocity")
                    levelset_quantities.update({
                        "levelset": self.unit_handler.dimensionalize(levelset[self.nhx, self.nhy, self.nhz], "length"),
                        "volume_fraction": volume_fraction[...,self.nhx_,self.nhy_,self.nhz_], "mask_real": mask_real[...,self.nhx_,self.nhy_,self.nhz_],
                        "normal": normal[...,self.nhx_,self.nhy_,self.nhz_]
                        })
                    for quantity in self.output_quantities["levelset"]:
                        h5file.create_dataset(name="levelset/" + quantity, data=levelset_quantities[quantity].T, dtype="f8" if self.is_double else "f4")
                
                # CONSERVATIVES AND PRIMITIVES FOR REAL FLUID
                if "real_fluid" in self.output_quantities.keys():
                    h5file.create_group(name="real_fluid")
                    for key in ["cons", "primes"]:
                        real_buffer = self.compute_real_buffer(material_fields[key][...,self.nhx,self.nhy,self.nhz], buffer_dictionary["levelset_quantities"]["volume_fraction"][self.nhx_,self.nhy_,self.nhz_])
                        for quantity in [quant for quant in self.output_quantities["real_fluid"] if quant in self.quantity_index[key].keys()]:
                            real_state = self.unit_handler.dimensionalize(real_buffer[self.quantity_index[key][quantity]], quantity)
                            h5file.create_dataset(name="real_fluid/" + quantity, data=real_state.T, dtype="f8" if self.is_double else "f4")

            # MISCELLANEOUS - ALWAYS COMPUTED FOR REAL FLUID
            if "miscellaneous" in self.output_quantities.keys():
                h5file.create_group(name="miscellaneous")
                for quantity in self.output_quantities["miscellaneous"]:
                    computed_quantity = self.compute_miscellaneous(material_fields["primes"], quantity, buffer_dictionary["levelset_quantities"]["volume_fraction"] if self.levelset_type != None else None)
                    h5file.create_dataset(name="miscellaneous/" + quantity, data=computed_quantity.T, dtype="f8" if self.is_double else "f4")

            # MASS FLOW FORCING
            if self.is_mass_flow_forcing:
                h5file.create_group(name="mass_flow_forcing")
                h5file.create_dataset(name="mass_flow_forcing/scalar_value", data=buffer_dictionary["mass_flow_forcing"]["scalar_value"], dtype="f8" if self.is_double else "f4")
                h5file.create_dataset(name="mass_flow_forcing/PID_e_int", data=buffer_dictionary["mass_flow_forcing"]["PID_e_int"], dtype="f8" if self.is_double else "f4")
                h5file.create_dataset(name="mass_flow_forcing/PID_e_new", data=buffer_dictionary["mass_flow_forcing"]["PID_e_new"], dtype="f8" if self.is_double else "f4")

    def write_xdmffile(self, current_time: float) -> None:
        """Writes an xdmf file for the current time step.
        The xdmf file corresponds to an h5 file which holds the 
        data buffers.

        :param current_time: Current simulation time.
        :type current_time: float
        """

        filename = "data_%.6f" % self.unit_handler.dimensionalize(current_time, "time")

        h5file_name   = filename + ".h5"
        xdmffile_path = filename + ".xdmf"
        h5file_path   = os.path.join(self.save_path_domain, h5file_name)
        xdmffile_path = os.path.join(self.save_path_domain, xdmffile_path)

        xdmf_str = ""

        # XDMF START
        xdmf_preamble ='''<?xml version="1.0" ?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0">
        <Domain>
            <Grid Name="TimeStep" GridType="Collection" CollectionType="Temporal">'''
        
        xdmf_str_start = '''
                <Grid Name="SpatialData_%e" GridType="Uniform">
                    <Time TimeType="Single" Value="%e" />

                    <Geometry Type="VXVYVZ">
                        <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i">%s:mesh/gridFX</DataItem>
                        <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i">%s:mesh/gridFY</DataItem>
                        <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i">%s:mesh/gridFZ</DataItem>
                    </Geometry>
                    <Topology Dimensions="%i %i %i" Type="3DRectMesh"/>''' %( # 1 512 128
                        current_time, current_time,
                        8 if self.is_double else 4, len(self.cell_faces[0]), h5file_name, 
                        8 if self.is_double else 4, len(self.cell_faces[1]), h5file_name, 
                        8 if self.is_double else 4, len(self.cell_faces[2]), h5file_name,
                        len(self.cell_faces[0]), len(self.cell_faces[1]), len(self.cell_faces[2]))

        # XDMF QUANTITIES
        xdmf_quants = []

        # CONSERVATIVES AND PRIMITIVES 
        for key in ["cons", "primes"]: 
            if key in self.output_quantities.keys():
                for quantity in self.output_quantities[key]:
                    no_phases = 2 if self.levelset_type == "FLUID-FLUID" else 1
                    for i in range(no_phases):
                        quantity_name = "%s_%d" % (quantity, i) if self.levelset_type == "FLUID-FLUID" else quantity
                        xdmf_quants.append(self.get_xdmf(key, quantity_name, h5file_name, *self.number_of_cells))


        # REAL FLUID AND MISCELLANEOUS
        for key in ["real_fluid", "miscellaneous"]: 
            if key in self.output_quantities.keys():
                for quantity in self.output_quantities[key]:
                    xdmf_quants.append(self.get_xdmf(key, quantity, h5file_name, *self.number_of_cells))

        xdmf_str_end = '''</Grid>'''

        # XDMF END
        xdmf_postamble = '''</Grid>
        </Domain>
        </Xdmf>'''

        # APPEND XDMF SPATIAL TO TIMESERIES
        if current_time not in self.output_timeseries:
            self.output_timeseries.append(current_time)
            self.xdmf_timeseries.append("\n".join([xdmf_str_start] + xdmf_quants + [xdmf_str_end]))

        # JOIN FINAL XDMF STR AND WRITE TO FILE
        xdmf_str = "\n".join([xdmf_preamble, xdmf_str_start] + xdmf_quants + [xdmf_str_end, xdmf_postamble])
        with open(xdmffile_path, "w") as xdmf_file:
            xdmf_file.write(xdmf_str)

    def write_timeseries_xdmffile(self) -> None:
        """Write xdmffile for the complete time series so that visualization 
        tools like Paraview can load the complete time series at once. This is
        done once at the end of a simulation when every output time stamp is 
        known. 
        """
        xdmffile_path = os.path.join(self.save_path_domain, "data_time_series.xdmf")

        xdmf_str = ""

        # XDMF START
        xdmf_str_start ='''<?xml version="1.0" ?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0">
        <Domain>
            <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">'''

        # XDMF END
        xdmf_str_end = '''</Grid>
        </Domain>
        </Xdmf>'''

        # JOIN FINAL XDMF STR AND WRITE TO FILE
        xdmf_str = "\n".join([xdmf_str_start] + self.xdmf_timeseries + [xdmf_str_end])
        with open(xdmffile_path, "w") as xdmf_file:
            xdmf_file.write(xdmf_str)

    def get_xdmf(self, group: str, quant: str, h5file_name: str, Nx: int, Ny: int, Nz: int) -> str:
        """Returns the string for the xdmf file for the given output quantity.

        :param group: Group name in h5 file under which the quantity is stored.
        :type group: str
        :param quant: Name of the output quantity.
        :type quant: str
        :param h5file_name: Name of the corresponding h5 file.
        :type h5file_name: str
        :param Nx: Resolution in x direction.
        :type Nx: int
        :param Ny: Resolution in y direction.
        :type Ny: int
        :param Nz: Resolution in z direction.
        :type Nz: int
        :return: Xdmf string for the specified quantity.
        :rtype: str
        """
        if quant in ["velocity", "momentum", "vorticity"]:
            xdmf ='''<Attribute Name="%s" AttributeType="Vector" Center="Cell">
            <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i %i %i %i">%s:%s/%s</DataItem>
            </Attribute>''' %(quant, 8 if self.is_double else 4, Nz, Ny, Nx, 3, h5file_name, group, quant)
        else:
            xdmf ='''<Attribute Name="%s" AttributeType="Scalar" Center="Cell">
                <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i %i %i">%s:%s/%s</DataItem>
            </Attribute>''' %(quant, 8 if self.is_double else 4, Nz, Ny, Nx, h5file_name, group, quant)
        return xdmf

    @partial(jax.jit, static_argnums=(0))
    def compute_real_buffer(self, buffer: jnp.ndarray, volume_fraction: jnp.ndarray) -> jnp.ndarray:
        """ For two-phase simulations, merges the two separate phase buffers 
        into a single real buffer. Calculation is done as a arithmetic average 
        based on the volume fraction. 

        :param buffer: Data buffer.
        :type buffer: jnp.ndarray
        :param volume_fraction: Buffer of the volume fraction.
        :type volume_fraction: jnp.ndarray
        :return: Combined data buffer of the 'real' fluid.
        :rtype: jnp.ndarray
        """
        
        volume_fraction    = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)       
        conservatives_real = buffer[...,0,:,:,:] * volume_fraction[0] + buffer[...,1,:,:,:] * volume_fraction[1]
        return conservatives_real

    @partial(jax.jit, static_argnums=(0,2))
    def compute_miscellaneous(self, primes: jnp.ndarray, quantity: str, volume_fraction: Union[jnp.ndarray, None]) -> jnp.ndarray:
        """Compute miscellaneous output fields for h5 output.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param quantity: String identifier of the quantity to be computed.
        :type quantity: str
        :param volume_fraction: Buffer of the volume fraction field, 
            only for two-phase simulations. Otherwise None.
        :type volume_fraction: Union[jnp.ndarray, None]
        :return: Computed phyiscal output quantity.
        :rtype: jnp.ndarray
        """
        if self.levelset_type == "FLUID-FLUID":
            primes_real = self.compute_real_buffer(primes[...,self.nhx__,self.nhy__,self.nhz__], volume_fraction)
        else:
            primes_real = primes

        if quantity == "schlieren":
            computed_quantity = self.compute_schlieren(primes_real[0:1])
        elif quantity == "vorticity":
            computed_quantity = self.compute_vorticity(primes_real[1:4])
        elif quantity == "absolute_vorticity":
            computed_quantity = self.compute_absolute_vorticity(primes_real[1:4])
        elif quantity == "absolute_velocity":
            computed_quantity = self.compute_absolute_velocity(primes_real[1:4])
        elif quantity == "mach_number":
            computed_quantity = self.compute_mach_number(primes, volume_fraction)
        
        return computed_quantity

    @partial(jax.jit, static_argnums=(0))
    def compute_absolute_velocity(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Computes the absolute velocity field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: jnp.ndarray
        :return: Buffer of absolute velocity.
        :rtype: jnp.ndarray
        """
        absolute_velocity = jnp.sqrt( jnp.sum( jnp.square(velocity), axis=0) )
        if self.levelset_type == "FLUID-FLUID":
            absolute_velocity = absolute_velocity[...,self.nhx_,self.nhy_,self.nhz_]
        else:
            absolute_velocity = absolute_velocity[...,self.nhx,self.nhy,self.nhz]
        return absolute_velocity

    @partial(jax.jit, static_argnums=(0))
    def compute_mach_number(self, primes: jnp.ndarray, volume_fraction: Union[jnp.ndarray, None]) -> jnp.ndarray:
        """Computes the Mach number field for h5 output.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param volume_fraction: Buffer of volume fraction.
        :type volume_fraction: Union[jnp.ndarray, None]
        :return: Buffer of Mach number.
        :rtype: jnp.ndarray
        """
        absolute_velocity   = jnp.sqrt( jnp.sum( jnp.square(primes[1:4]), axis=0) )
        speed_of_sound      = self.material_manager.get_speed_of_sound(primes[4], primes[0])
        mach_number         = absolute_velocity/speed_of_sound
        if self.levelset_type == "FLUID-FLUID":
            mach_number = self.compute_real_buffer(mach_number[...,self.nhx__,self.nhy__,self.nhz__], volume_fraction)[...,self.nhx_,self.nhy_,self.nhz_]
        else:
            mach_number = mach_number[...,self.nhx,self.nhy,self.nhz]
        return mach_number            

    @partial(jax.jit, static_argnums=(0))
    def compute_schlieren(self, density: jnp.ndarray) -> jnp.ndarray:
        """Computes numerical schlieren field for h5 output.

        :param density: Buffer of density.
        :type density: jnp.ndarray
        :return: Buffer of schlieren.
        :rtype: jnp.ndarray
        """
        schlieren = []
        for i in range(3):
            if self.levelset_type == "FLUID-FLUID":
                schlieren.append( self.derivative_stencil_geometry.derivative_xi(density, self.cell_sizes[i], i) if i in self.active_axis_indices else jnp.zeros(density[...,self.nhx_,self.nhy_,self.nhz_].shape) )
            else:
                schlieren.append( self.derivative_stencil_conservatives.derivative_xi(density, self.cell_sizes[i], i) if i in self.active_axis_indices else jnp.zeros(density[...,self.nhx,self.nhy,self.nhz].shape) )
        schlieren = jnp.linalg.norm(jnp.vstack(schlieren), axis=0, ord=2)
        return schlieren

    @partial(jax.jit, static_argnums=(0))
    def compute_vorticity(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Computes vorticity field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: jnp.ndarray
        :return: Buffer of vorticity.
        :rtype: jnp.ndarray
        """

        if self.levelset_type == "FLUID-FLUID":
            velocity_grad = jnp.stack([self.derivative_stencil_geometry.derivative_xi(velocity, self.cell_sizes[k], k) if k in self.active_axis_indices else jnp.zeros(velocity[...,self.nhx_,self.nhy_,self.nhz_].shape) for k in range(3)], axis=1)
        else:
            velocity_grad = jnp.stack([self.derivative_stencil_conservatives.derivative_xi(velocity, self.cell_sizes[k], k) if k in self.active_axis_indices else jnp.zeros(velocity[...,self.nhx,self.nhy,self.nhz].shape) for k in range(3)], axis=1)
        
        du_dy, du_dz = velocity_grad[0,1], velocity_grad[0,2]
        dv_dx, dv_dz = velocity_grad[1,0], velocity_grad[1,2]
        dw_dx, dw_dy = velocity_grad[2,0], velocity_grad[2,1]

        vorticity = jnp.stack([
            dw_dy - dv_dz,
            du_dz - dw_dx,
            dv_dx - du_dy
        ], axis=0)

        return vorticity

    @partial(jax.jit, static_argnums=(0))
    def compute_absolute_vorticity(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Computes absolute vorticity field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: jnp.ndarray
        :return: Buffer of absolute vorticity.
        :rtype: jnp.ndarray
        """
        
        absolute_vorticity = jnp.linalg.norm(self.compute_vorticity(velocity), axis=0, ord=2)
        return absolute_vorticity