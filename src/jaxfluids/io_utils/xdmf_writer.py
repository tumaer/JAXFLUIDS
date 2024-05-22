import os
from typing import Dict

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.output import OutputQuantitiesSetup
from jaxfluids.data_types.buffers import TimeControlVariables

class XDMFWriter():
    def __init__(
            self,
            domain_information: DomainInformation,
            unit_handler: UnitHandler, 
            equation_information: EquationInformation,
            quantities_setup: OutputQuantitiesSetup,
            is_double: str,
            ) -> None:

        self.domain_information = domain_information
        self.unit_handler = unit_handler
        self.save_path_domain = None
        self.quantities_setup = quantities_setup
        self.is_double = is_double
        self.equation_information = equation_information
        self.num_digits_output = 10
        
        self.is_multihost = domain_information.is_multihost
        self.process_id = domain_information.process_id

        self.output_timeseries  = []
        self.xdmf_timeseries    = []

    def set_save_path_domain(self, save_path_domain: str) -> None:
        self.save_path_domain = save_path_domain

    def write_file(
            self,
            time_control_variables: TimeControlVariables,
            is_write_step: bool = False
            ) -> None:
        """Writes an xdmf file for the current time step.
        The xdmf file corresponds to an h5 file which holds the 
        data buffers.

        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param is_write_step: _description_, defaults to False
        :type is_write_step: bool, optional
        """
        if is_write_step:
            physical_simulation_time = time_control_variables.physical_simulation_time
            simulation_step = time_control_variables.simulation_step
            if self.is_multihost:
                filename = f"data_proc{self.process_id:d}_{simulation_step:d}"
            else:
                filename = f"data_{simulation_step:d}" 
        else:
            physical_simulation_time = time_control_variables.physical_simulation_time
            current_time = self.unit_handler.dimensionalize(physical_simulation_time, "time")
            if self.is_multihost:
                filename = f"data_proc{self.process_id:d}_{current_time:.{self.num_digits_output}f}"
            else:
                filename = f"data_{current_time:.{self.num_digits_output}f}"

        # cell_faces = self.domain_information.get_global_cell_faces()
        number_of_cells = self.domain_information.global_number_of_cells
        number_of_cell_faces = self.domain_information.global_number_of_cell_faces
        levelset_model = self.equation_information.levelset_model

        h5file_name   = filename + ".h5"
        xdmffile_path = filename + ".xdmf"
        xdmffile_path = os.path.join(self.save_path_domain, xdmffile_path)

        xdmf_str = ""

        # XDMF START
        xdmf_preamble ='''<?xml version="1.0" ?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0">
        <Domain>
            <Grid Name="TimeStep" GridType="Collection" CollectionType="Temporal">'''
        
        precision = 8 if self.is_double else 4
        xdmf_str_start = f'''
                <Grid Name="SpatialData_{physical_simulation_time:e}" GridType="Uniform">
                    <Time TimeType="Single" Value="{physical_simulation_time:e}" />

                    <Geometry Type="VXVYVZ">
                        <DataItem Format="HDF" NumberType="Float" Precision="{precision:d}" Dimensions="{number_of_cell_faces[0]:d}">{h5file_name:s}:domain/gridFX</DataItem>
                        <DataItem Format="HDF" NumberType="Float" Precision="{precision:d}" Dimensions="{number_of_cell_faces[1]:d}">{h5file_name:s}:domain/gridFY</DataItem>
                        <DataItem Format="HDF" NumberType="Float" Precision="{precision:d}" Dimensions="{number_of_cell_faces[2]:d}">{h5file_name:s}:domain/gridFZ</DataItem>
                    </Geometry>
                    <Topology Dimensions="{number_of_cell_faces[0]:d} {number_of_cell_faces[1]:d} {number_of_cell_faces[2]:d}" Type="3DRectMesh"/>'''

        # XDMF QUANTITIES
        xdmf_quants = []

        # CONSERVATIVES AND PRIMITIVES 
        for key in ["conservatives", "primitives"]: 
            if getattr(self.quantities_setup, key) != None:
                for quantity in getattr(self.quantities_setup, key):
                    no_phases = 2 if levelset_model == "FLUID-FLUID" else 1
                    for i in range(no_phases):
                        quantity_name = f"{quantity:s}_{i:d}" if levelset_model == "FLUID-FLUID" else quantity
                        xdmf_quants.append(self.get_xdmf(key, quantity_name, h5file_name, *number_of_cells))

        # REAL FLUID AND MISCELLANEOUS
        # TODO write real fluid only for fluid-fluid???
        for key in ["real_fluid", "miscellaneous"]: 
            if getattr(self.quantities_setup, key) != None:
                for quantity in getattr(self.quantities_setup, key):
                    xdmf_quants.append(self.get_xdmf(key, quantity, h5file_name, *number_of_cells))

        # TODO write levelset ??? 
        if levelset_model != None:
            for key in ["levelset"]:
                if getattr(self.quantities_setup, key) != None:
                    for quantity in getattr(self.quantities_setup, key):
                        xdmf_quants.append(self.get_xdmf(key, quantity, h5file_name, *number_of_cells))

        xdmf_str_end = '''</Grid>'''

        # XDMF END
        xdmf_postamble = '''</Grid>
        </Domain>
        </Xdmf>'''

        # APPEND XDMF SPATIAL TO TIMESERIES
        if physical_simulation_time not in self.output_timeseries:
            self.output_timeseries.append(physical_simulation_time)
            self.xdmf_timeseries.append("\n".join([xdmf_str_start] + xdmf_quants + [xdmf_str_end]))

        # JOIN FINAL XDMF STR AND WRITE TO FILE
        xdmf_str = "\n".join([xdmf_preamble, xdmf_str_start] + xdmf_quants + [xdmf_str_end, xdmf_postamble])
        with open(xdmffile_path, "w") as xdmf_file:
            xdmf_file.write(xdmf_str)

    def write_timeseries(self) -> None:
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
        precision = 8 if self.is_double else 4
        if quant in ["velocity", "momentum", "vorticity",
                     "velocity_0", "momentum_0", "velocity_1", "momentum_1",
                     "interface_pressure"]:
            if quant == "vorticity":
                # TODO should also be dim, but must be changed accordingly in hdf5 writer
                # see comment in hdf5 writer
                # In 1D, vorticity is not defined
                # In 2D, vorticity has 1 component
                # In 3D, vorticity has 3 components
                dim = self.domain_information.dim
                if dim == 2:
                    dim = 1
            elif quant == "interface_pressure":
                dim = 2
            else:
                dim = self.domain_information.dim
            xdmf =f'''<Attribute Name="{quant:s}" AttributeType="Vector" Center="Cell">
            <DataItem Format="HDF" NumberType="Float" Precision="{precision:d}" Dimensions="{Nz:d} {Ny:d} {Nx:d} {dim:d}">{h5file_name:s}:{group:s}/{quant:s}</DataItem>
            </Attribute>'''
        else:
            xdmf =f'''<Attribute Name="{quant:s}" AttributeType="Scalar" Center="Cell">
                <DataItem Format="HDF" NumberType="Float" Precision="{precision:d}" Dimensions="{Nz:d} {Ny:d} {Nx:d}">{h5file_name:s}:{group:s}/{quant:s}</DataItem>
            </Attribute>'''
        return xdmf

