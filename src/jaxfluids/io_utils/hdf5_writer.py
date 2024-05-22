import os
import json
from typing import Dict, Tuple, Union
from functools import partial

import h5py
import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.data_types.buffers import ForcingParameters, SimulationBuffers, \
    MaterialFieldBuffers, LevelsetFieldBuffers, TimeControlVariables
from jaxfluids.data_types.information import WallClockTimes
from jaxfluids.data_types.case_setup.output import OutputQuantitiesSetup
from jaxfluids.data_types.numerical_setup.output import OutputSetup
from jaxfluids.data_types.numerical_setup import ActiveForcingsSetup

from jaxfluids.stencils import DICT_SECOND_DERIVATIVE_CENTER
from jaxfluids.math.differential_operator.laplacian import Laplacian

class HDF5Writer():
    def __init__(
            self,
            domain_information: DomainInformation,
            unit_handler: UnitHandler, 
            material_manager: MaterialManager,
            levelset_handler: LevelsetHandler,
            derivative_stencil: SpatialDerivative,
            quantities_setup: OutputQuantitiesSetup,
            is_double: str,
            output_setup: OutputSetup,
            ) -> None:

        # MEMBER
        self.domain_information = domain_information
        self.material_manager = material_manager
        self.equation_information = material_manager.equation_information
        self.unit_handler = unit_handler
        self.levelset_handler = levelset_handler
        self.domain_information = domain_information
        self.derivative_stencil = derivative_stencil
        self.quantities_setup = quantities_setup
        self.is_double = is_double
        self.is_domain = output_setup.is_domain
        self.is_wall_clock_times = output_setup.is_wall_clock_times
        self.is_metadata = output_setup.is_metadata
        self.is_time = output_setup.is_time
        self.save_path_domain = None

        self.is_parallel = domain_information.is_parallel
        self.num_digits_output = 10

        active_forcings = self.equation_information.active_forcings
        self.is_mass_flow_forcing = active_forcings.is_mass_flow_forcing
        self.is_turb_hit_forcing = active_forcings.is_turb_hit_forcing
        self.is_temperature_forcing = active_forcings.is_temperature_forcing

        self.is_multihost = domain_information.is_multihost
        self.process_id = domain_information.process_id

        self.material_field_indices = self.equation_information.get_material_field_indices(
            domain_information.active_axes_indices)
        
        self.laplacian = Laplacian(DICT_SECOND_DERIVATIVE_CENTER["CENTRAL2"], domain_information)
    
    def set_save_path_domain(self, save_path_domain: str) -> None:
        self.save_path_domain = save_path_domain

    def write_file(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            wall_clock_times: WallClockTimes,
            forcing_parameters: ForcingParameters = None,
            is_write_step: bool = False
            ) -> None:
        """Saves the specified output fields in a .h5 file.

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_, defaults to None
        :type forcing_parameters: ForcingParameters, optional
        """
        
        physical_simulation_time = time_control_variables.physical_simulation_time
        material_fields = simulation_buffers.material_fields
        levelset_fields = simulation_buffers.levelset_fields

        if is_write_step:
            simulation_step = time_control_variables.simulation_step
            if self.is_multihost:
                filename = f"data_proc{self.process_id:d}_{simulation_step:d}.h5"
            else:
                filename = f"data_{simulation_step:d}.h5"
        else:
            current_time_ = self.unit_handler.dimensionalize(physical_simulation_time, "time")
            if self.is_multihost:
                filename = f"data_proc{self.process_id:d}_{current_time_:.{self.num_digits_output}f}.h5"
            else:
                filename = f"data_{current_time_:.{self.num_digits_output}f}.h5"

        dtype = "f8" if self.is_double else "f4"

        with h5py.File(os.path.join(self.save_path_domain, filename), "w") as h5file:

            # DOMAIN INFORMATION
            cell_centers = self.domain_information.get_local_cell_centers()
            cell_faces = self.domain_information.get_local_cell_faces()
            cell_sizes = self.domain_information.get_local_cell_sizes()

            cell_centers_h5 = []
            cell_faces_h5 = []
            cell_sizes_h5 = []
            for i in range(3):
                xi = self.unit_handler.dimensionalize(cell_centers[i], "length")
                fxi = self.unit_handler.dimensionalize(cell_faces[i], "length")
                dxi = self.unit_handler.dimensionalize(cell_sizes[i], "length")
                cell_centers_h5.append(jnp.squeeze(xi))
                cell_faces_h5.append(jnp.squeeze(fxi))
                cell_sizes_h5.append(jnp.squeeze(dxi))

            dim = self.domain_information.dim
            split_factors = self.domain_information.split_factors
            is_parallel = self.domain_information.is_parallel
            is_multihost = self.domain_information.is_multihost
            host_count = self.domain_information.host_count
            process_id = self.domain_information.process_id
            local_device_count = self.domain_information.local_device_count
            global_device_count = self.domain_information.global_device_count
            
            # EQUATION INFORMATION
            number_fluids = self.equation_information.no_fluids
            fluid_names = self.equation_information.fluid_names
            diffuse_interface_model = self.equation_information.diffuse_interface_model
            levelset_model = self.equation_information.levelset_model

            # METADATA
            if self.is_metadata:
                h5file.create_group(name="metadata")

                h5file.create_dataset(name="metadata/is_parallel", data=is_parallel)
                h5file.create_dataset(name="metadata/is_multihost", data=is_multihost)
                h5file.create_dataset(name="metadata/process_id", data=process_id)
                h5file.create_dataset(name="metadata/local_device_count", data=local_device_count)
                h5file.create_dataset(name="metadata/global_device_count", data=global_device_count)

                h5file.create_dataset(name="metadata/fluid_names", data=fluid_names)
                h5file.create_dataset(name="metadata/number_fluids", data=number_fluids)
                h5file.create_dataset(name="metadata/levelset_model", data=levelset_model)
                h5file.create_dataset(name="metadata/diffuse_interface_model", data=diffuse_interface_model)

                h5file.create_dataset(name="metadata/is_double_precision", data=self.is_double)
                
                h5file["metadata"].create_group("available_quantities")
                for field in self.quantities_setup._fields:
                    quantity_list = getattr(self.quantities_setup, field)
                    if quantity_list != None:
                        h5file["metadata/available_quantities"].create_dataset(data=quantity_list, name=field)

            # DOMAIN DATA
            if self.is_domain:
                h5file.create_group(name="domain")
                h5file.create_dataset(name="domain/dim", data=dim)
                h5file.create_dataset(name="domain/gridX", data=cell_centers_h5[0], dtype=dtype)
                h5file.create_dataset(name="domain/gridY", data=cell_centers_h5[1], dtype=dtype)
                h5file.create_dataset(name="domain/gridZ", data=cell_centers_h5[2], dtype=dtype)
                h5file.create_dataset(name="domain/gridFX", data=cell_faces_h5[0], dtype=dtype)
                h5file.create_dataset(name="domain/gridFY", data=cell_faces_h5[1], dtype=dtype)
                h5file.create_dataset(name="domain/gridFZ", data=cell_faces_h5[2], dtype=dtype)
                h5file.create_dataset(name="domain/cellsizeX", data=cell_sizes_h5[0], dtype=dtype)
                h5file.create_dataset(name="domain/cellsizeY", data=cell_sizes_h5[1], dtype=dtype)
                h5file.create_dataset(name="domain/cellsizeZ", data=cell_sizes_h5[2], dtype=dtype)
                h5file.create_dataset(name="domain/split_factors", data=jnp.array(split_factors), dtype="i8")
            
            if self.is_wall_clock_times:
                h5file.create_group(name="wall_clock")
                h5file.create_dataset(name="wall_clock/step", data=wall_clock_times.step, dtype="f8")
                h5file.create_dataset(name="wall_clock/step_per_cell", data=wall_clock_times.step_per_cell, dtype="f8")
                h5file.create_dataset(name="wall_clock/mean_step", data=wall_clock_times.mean_step, dtype="f8")
                h5file.create_dataset(name="wall_clock/mean_step_per_cell", data=wall_clock_times.mean_step_per_cell, dtype="f8")


            # CURRENT TIME
            if self.is_time:
                h5file.create_dataset(name="time", data=self.unit_handler.dimensionalize(physical_simulation_time, "time"), dtype=dtype)

            # CONSERAVITVES AND PRIMITIVES
            for key in ["conservatives", "primitives"]:
                if getattr(self.quantities_setup, key) != None:
                    h5file.create_group(name=key)
                    for quantity in getattr(self.quantities_setup, key):
                        if levelset_model == "FLUID-FLUID":
                            for phase in range(2):
                                buffer = self.prepare_material_field_buffer_for_h5dump(material_fields, key, quantity, "conservatives", phase)
                                quantity_name = "%s_%d" % (quantity, phase)
                                h5file.create_dataset(name="/".join([key, quantity_name]), data=buffer, dtype=dtype)
                        else:
                            buffer = self.prepare_material_field_buffer_for_h5dump(material_fields, key, quantity, "conservatives")
                            h5file.create_dataset(name="/".join([key, quantity]), data=buffer, dtype=dtype)

            # LEVELSET RELATED FIELDS
            if getattr(self.quantities_setup, "levelset") != None:
                h5file.create_group(name="levelset")
                for quantity in self.quantities_setup.levelset:
                    buffer = self.prepare_levelset_field_buffer(levelset_fields, quantity)
                    h5file.create_dataset(name="levelset/" + quantity, data=buffer, dtype=dtype)

            # DIFFUSE INTERFACE GEOMETRICAL QUANTITIES
            # TODO deniz
            
            # REAL FLUID FIELDS
            if getattr(self.quantities_setup, "real_fluid") != None:
                h5file.create_group(name="real_fluid")
                volume_fraction = levelset_fields.volume_fraction
                for quantity in self.quantities_setup.real_fluid:
                    real_buffer = self.prepare_real_material_field_buffer_for_h5dump(material_fields, quantity, volume_fraction)
                    h5file.create_dataset(name="real_fluid/" + quantity, data=real_buffer, dtype=dtype)
                    
            # MISCELLANEOUS FIELDS
            if getattr(self.quantities_setup, "miscellaneous") != None:
                h5file.create_group(name="miscellaneous")
                primitives = material_fields.primitives
                volume_fraction = levelset_fields.volume_fraction if \
                    levelset_model == "FLUID-FLUID" else None
                for quantity in self.quantities_setup.miscellaneous:
                    miscellaneous_buffer = self.compute_miscellaneous(primitives, quantity, volume_fraction)
                    h5file.create_dataset(name="miscellaneous/" + quantity, data=miscellaneous_buffer, dtype=dtype)

            # FORCINGS
            if getattr(self.quantities_setup, "forcings") != None:
                h5file.create_group(name="forcings")
                if self.is_mass_flow_forcing:
                    h5file["forcings"].create_group(name="mass_flow")
                    PID_e_int = forcing_parameters.mass_flow_controller_params.integral_error
                    PID_e_new = forcing_parameters.mass_flow_controller_params.current_error
                    h5file["forcings/mass_flow"].create_dataset(name="PID_e_int", data=PID_e_int, dtype=dtype)
                    h5file["forcings/mass_flow"].create_dataset(name="PID_e_new", data=PID_e_new, dtype=dtype)
                if self.is_turb_hit_forcing:
                    h5file["forcings"].create_group(name="turb_hit")
                    ek_ref = forcing_parameters.hit_ek_ref
                    h5file["forcings/turb_hit"].create_dataset(name="ek_ref", data=ek_ref, dtype=dtype)

    def _prepare_material_field_buffer_for_h5dump(
            self,
            material_fields: MaterialFieldBuffers,
            key: str,
            quantity: str,
            slices_buffer: Tuple,
            phase: int = None
            ) -> Array:
        """Prepares buffer for h5 dump, i.e.,
        slices and dimensionalizes the buffer.

        :param buffer: _description_
        :type buffer: Array
        :param quantity: _description_
        :type quantity: str
        :param slices_buffer: _description_
        :type slices_buffer: Tuple
        :param phase: _description_, defaults to None
        :type phase: int, optional
        :return: _description_
        :rtype: Array
        """

        if slices_buffer == "conservatives":
            nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        elif slices_buffer == "geometry":
            nhx, nhy, nhz = self.domain_information.domain_slices_geometry
        elif slices_buffer == None:
            nhx, nhy, nhz = (jnp.s_[:], jnp.s_[:], jnp.s_[:])

        # MATERIAL FIELDS
        if quantity in self.material_field_indices[key].keys():
            index = self.material_field_indices[key][quantity]
            buffer = getattr(material_fields, key)[index]

        # MATERIAL FIELD RELATED QUANTITIES WHICH ARE NOT DIRECTLY GIVEN
        # IN PRIMITIVES OR CONSERVATIVES (E.G., DENSITY (for diffuse),
        else:
            equation_type = self.equation_information.equation_type
            primitives = material_fields.primitives
            mass_ids = self.equation_information.mass_ids
            energy_ids = self.equation_information.energy_ids
            mass_slices = self.equation_information.mass_slices
            vf_slices = self.equation_information.vf_slices
            component_slices = self.equation_information.species_slices
            if quantity == "temperature":
                buffer = self.material_manager.get_temperature(primitives)
                    
            elif quantity in ["density", "mass"]:
                if equation_type in ("DIFFUSE-INTERFACE-5EQM"):
                    buffer = self.material_manager.get_density(primitives)
                else:
                    raise NotImplementedError
                
        if phase != None:
            slice_object = jnp.s_[...,phase,nhx,nhy,nhz]
        else:
            slice_object = jnp.s_[...,nhx,nhy,nhz]

        buffer = buffer[slice_object]
        buffer = self.unit_handler.dimensionalize(buffer, quantity)
        return buffer.T

    @partial(jax.pmap, static_broadcasted_argnums=(0,2,3,4,5),
             in_axes=(None,0,None,None,None,None), axis_name="i")
    def _prepare_material_field_buffer_for_h5dump_pmap(
            self,
            material_fields: MaterialFieldBuffers,
            key: str,
            quantity: str,
            slices_buffer: str,
            phase: int = None
            ) -> Array:
        return self._prepare_material_field_buffer_for_h5dump(
            material_fields, key, quantity, slices_buffer, phase)

    @partial(jax.jit, static_argnums=(0,2,3,4,5))
    def _prepare_material_field_buffer_for_h5dump_jit(
            self,
            material_fields: MaterialFieldBuffers,
            key: str,
            quantity: str,
            slices_buffer: str,
            phase: int = None
            ) -> Array:
        return self._prepare_material_field_buffer_for_h5dump(
            material_fields, key, quantity, slices_buffer, phase)

    def prepare_material_field_buffer_for_h5dump(
            self,
            material_fields: MaterialFieldBuffers,
            key: str,
            quantity: str,
            slices_buffer: str,
            phase: int = None
            ) -> Array:
        if self.domain_information.is_parallel:
            return self._prepare_material_field_buffer_for_h5dump_pmap(
                material_fields, key, quantity, slices_buffer, phase)
        else:
            return self._prepare_material_field_buffer_for_h5dump_jit(
                material_fields, key, quantity, slices_buffer, phase)







    def _prepare_real_material_field_buffer_for_h5dump(
            self,
            material_fields: MaterialFieldBuffers,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        """Computes the real fluid buffer for FLUID-FLUID levelset simulations
        from the material fields and prepares it for h5 dump.

        :param material_fields: _description_
        :type material_fields: MaterialFieldBuffers
        :param quantity: _description_
        :type quantity: str
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :return: _description_
        :rtype: Array
        """

        if quantity == "temperature":
            buffer = self.material_manager.get_temperature(material_fields.primitives)
        else:
            key = "primitives" if quantity in self.material_field_indices["primitives"].keys() else "conservatives" 
            index = self.material_field_indices[key][quantity]
            buffer = getattr(material_fields, key)[index]
        buffer = self._compute_real_buffer(buffer, volume_fraction, "conservatives", "geometry") 
        buffer = self.unit_handler.dimensionalize(buffer, quantity)
        return buffer.T

    @partial(jax.pmap, static_broadcasted_argnums=(0,2),
             in_axes=(None,0,None,0), axis_name="i")
    def _prepare_real_material_field_buffer_for_h5dump_pmap(
            self,
            material_fields: MaterialFieldBuffers,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        return self._prepare_real_material_field_buffer_for_h5dump(
            material_fields, quantity, volume_fraction)

    @partial(jax.jit, static_argnums=(0,2))
    def _prepare_real_material_field_buffer_for_h5dump_jit(
            self,
            material_fields: MaterialFieldBuffers,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        return self._prepare_real_material_field_buffer_for_h5dump(
            material_fields, quantity, volume_fraction)

    def prepare_real_material_field_buffer_for_h5dump(
            self,
            material_fields: MaterialFieldBuffers,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        if self.domain_information.is_parallel:
            return self._prepare_real_material_field_buffer_for_h5dump_pmap(
                material_fields, quantity, volume_fraction)
        else:
            return self._prepare_real_material_field_buffer_for_h5dump_jit(
                material_fields, quantity, volume_fraction)








    def _prepare_levelset_field_buffer(
            self,
            levelset_fields: LevelsetFieldBuffers,
            quantity: str,
            ) -> Dict:
        """Computes levelset related fields for h5 dump.

        :param levelset_fields: _description_
        :type levelset_fields: LevelsetFieldBuffers
        :param quantity: _description_
        :type quantity: str
        :return: _description_
        :rtype: Dict
        """
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        domain_slices = {
            "levelset": jnp.s_[nhx,nhy,nhz],
            "volume_fraction": jnp.s_[nhx_,nhy_,nhz_],
            "interface_velocity": jnp.s_[...,nhx_,nhy_,nhz_],
            "interface_pressure": jnp.s_[...,nhx_,nhy_,nhz_]
            }
        if self.equation_information.levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            active_axes_indices = jnp.array(self.domain_information.active_axes_indices)
            domain_slices["interface_velocity"] = jnp.s_[active_axes_indices,nhx,nhy,nhz]
        slice_object = domain_slices[quantity]
        buffer = getattr(levelset_fields, quantity)
        buffer = self.unit_handler.dimensionalize(buffer[slice_object], quantity)
        return buffer.T

    @partial(jax.pmap, static_broadcasted_argnums=(0,2),
             in_axes=(None,0,None), axis_name="i")
    def _prepare_levelset_field_buffer_pmap(
            self,
            levelset_fields: LevelsetFieldBuffers,
            quantity: str
            ) -> Array:
        return self._prepare_levelset_field_buffer(
            levelset_fields, quantity)

    @partial(jax.jit, static_argnums=(0,2))
    def _prepare_levelset_field_buffer_jit(
            self,
            levelset_fields: LevelsetFieldBuffers,
            quantity: str,
            ) -> Array:
        return self._prepare_levelset_field_buffer(
            levelset_fields, quantity)

    def prepare_levelset_field_buffer(
            self,
            levelset_fields: LevelsetFieldBuffers,
            quantity: str
            ) -> Array:
        if self.domain_information.is_parallel:
            return self._prepare_levelset_field_buffer_pmap(
                levelset_fields, quantity)
        else:
            return self._prepare_levelset_field_buffer_jit(
                levelset_fields, quantity)





    def _compute_real_buffer(
            self,
            buffer: Array,
            volume_fraction: Array,
            slices_buffer: str,
            slices_vf: str
            ) -> Array:
        """For two-phase simulations, merges the two separate phase buffers 
        into a single real buffer. Calculation is done as a arithmetic average 
        based on the volume fraction. 

        :param buffer: _description_
        :type buffer: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param slices_buffer: _description_
        :type slices_buffer: str
        :param slices_vf: _description_
        :type slices_vf: str
        :return: _description_
        :rtype: Array
        """
  
        if slices_buffer == "conservatives":
            nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        elif slices_buffer == "geometry":
            nhx, nhy, nhz = self.domain_information.domain_slices_geometry
        elif slices_buffer == "offset":
            nhx, nhy, nhz = self.domain_information.domain_slices_conservatives_to_geometry
        elif slices_buffer == None:
            nhx, nhy, nhz = (jnp.s_[:], jnp.s_[:], jnp.s_[:])
        buffer = buffer[...,nhx,nhy,nhz]

        if slices_vf == "geometry":
            nhx, nhy, nhz = self.domain_information.domain_slices_geometry
        elif slices_vf == None:
            nhx, nhy, nhz = (jnp.s_[:], jnp.s_[:], jnp.s_[:])
        volume_fraction = volume_fraction[...,nhx,nhy,nhz]

        volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)    
        buffer_real = buffer[...,0,:,:,:] * volume_fraction[0] + buffer[...,1,:,:,:] * volume_fraction[1]
        return buffer_real






    def _compute_miscellaneous(
            self,
            primitives: Array,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        """Compute miscellaneous output fields for h5 output.

        :param primitives: _description_
        :type primitives: Array
        :param quantity: _description_
        :type quantity: str
        :return: _description_
        :rtype: Array
        """

        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry

        velocity_slices = self.equation_information.velocity_slices
        equation_type = self.equation_information.equation_type

        energy_slices = self.equation_information.energy_ids
        mass_slices = self.equation_information.mass_slices
        vf_slices = self.equation_information.vf_slices

        mass_id = self.equation_information.mass_ids
        energy_id = self.equation_information.energy_ids

        if quantity == "mach_number":
            speed_of_sound = self.material_manager.get_speed_of_sound(primitives)
            if equation_type == "TWO-PHASE-LS":
                speed_of_sound = self._compute_real_buffer(
                    speed_of_sound, volume_fraction, "conservatives", "geometry")
            
        if equation_type == "TWO-PHASE-LS":
            primitives = self._compute_real_buffer(
                primitives, volume_fraction, "offset", None)

        if quantity == "schlieren":
            density = self.material_manager.get_density(primitives)
            computed_quantity = self._compute_schlieren(density)

        elif quantity == "vorticity":
            velocity = primitives[velocity_slices]
            computed_quantity = self._compute_vorticity(velocity)

        elif quantity == "absolute_vorticity":
            velocity = primitives[velocity_slices]
            computed_quantity = self._compute_absolute_vorticity(velocity)

        elif quantity == "absolute_velocity":
            if equation_type == "TWO-PHASE-LS":
                velocity = primitives[velocity_slices,nhx_,nhy_,nhz_]
            else:
                velocity = primitives[velocity_slices,nhx,nhy,nhz]
            computed_quantity = self._compute_absolute_velocity(velocity)

        elif quantity == "mach_number":
            if equation_type == "TWO-PHASE-LS":
                velocity = primitives[velocity_slices,nhx_,nhy_,nhz_]
            else:
                velocity = primitives[velocity_slices,nhx,nhy,nhz]
                speed_of_sound = speed_of_sound[...,nhx,nhy,nhz]
            computed_quantity = self._compute_mach_number(
                velocity, speed_of_sound)
            
        elif quantity == "qcriterion":
            velocity = primitives[velocity_slices]
            computed_quantity = self._compute_qcriterion(velocity)

        elif quantity == "dilatation":
            velocity = primitives[velocity_slices]
            computed_quantity = self._compute_dilatation(velocity) 
            
        else:
            raise NotImplementedError

        computed_quantity = self.unit_handler.dimensionalize(computed_quantity, quantity)
        return computed_quantity.T

    @partial(jax.pmap, static_broadcasted_argnums=(0,2),
             in_axes=(None,0,None,0), axis_name="i")
    def _compute_miscellaneous_pmap(
            self,
            primitives: Array,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        return self._compute_miscellaneous(
            primitives, quantity, volume_fraction)

    @partial(jax.jit, static_argnums=(0,2))
    def _compute_miscellaneous_jit(
            self,
            primitives: Array,
            quantity: str,
            volume_fraction: Array
            ) -> Array:
        return self._compute_miscellaneous(
            primitives, quantity, volume_fraction)

    def compute_miscellaneous(
            self,
            primitives: Array,
            quantity: str,
            volume_fraction: Array = None
            ) -> Array:
        if self.domain_information.is_parallel:
            return self._compute_miscellaneous_pmap(
                primitives, quantity, volume_fraction)
        else:
            return self._compute_miscellaneous_jit(
                primitives, quantity, volume_fraction)

    def _compute_absolute_velocity(
            self,
            velocity: Array
            ) -> Array:
        """Computes the absolute velocity field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: Array
        :return: Buffer of absolute velocity.
        :rtype: Array
        """
        absolute_velocity = jnp.sqrt(jnp.sum(jnp.square(velocity), axis=0))
        return absolute_velocity

    def _compute_mach_number(
            self,
            velocity: Array,
            speed_of_sound: Array
            ) -> Array:
        """Computes the Mach number field for h5 output.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param volume_fraction: Buffer of volume fraction.
        :type volume_fraction: Array
        :return: Buffer of Mach number.
        :rtype: Array
        """
        absolute_velocity = self._compute_absolute_velocity(velocity)
        mach_number = absolute_velocity/speed_of_sound
        return mach_number

    def _compute_schlieren(
            self,
            density: Array
            ) -> Array:
        """Computes numerical schlieren field for h5 output.

        :param density: Buffer of density.
        :type density: Array
        :return: Buffer of schlieren.
        :rtype: Array
        """
        cell_sizes = self.domain_information.get_device_cell_sizes()
        active_axes_indices = self.domain_information.active_axes_indices
        equation_type = self.equation_information.equation_type
        if equation_type == "TWO-PHASE-LS":
            nhx,nhy,nhz = self.domain_information.domain_slices_geometry
        else:
            nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        shape = density[...,nhx,nhy,nhz].shape

        schlieren = []
        for axis_index in range(3):
            if axis_index in active_axes_indices:
                derivative = self.derivative_stencil.derivative_xi(
                    density, cell_sizes[axis_index], axis_index)
            else:
                derivative = jnp.zeros(shape)
            schlieren.append(derivative)
        schlieren = jnp.stack(schlieren, axis=0)
        schlieren = jnp.sqrt(jnp.sum(schlieren*schlieren, axis=0) + 1e-10) # TODO EPS
        return schlieren

    def _compute_vorticity(
            self,
            velocity: Array
            ) -> Array:
        """Computes vorticity field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: Array
        :return: Buffer of vorticity.
        :rtype: Array
        """
        # TODO: should vorticity in 2D sim only return 1D vector with
        # active component and in 1D sim should be caught by sanity check
        # somewhere??
        velocity_grad = self.compute_velocity_gradient(velocity)
        du_dy, du_dz = velocity_grad[0,1], velocity_grad[0,2]
        dv_dx, dv_dz = velocity_grad[1,0], velocity_grad[1,2]
        dw_dx, dw_dy = velocity_grad[2,0], velocity_grad[2,1]
        vorticity = jnp.stack([
            dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy
            ], axis=0)
        if self.domain_information.dim == 3:
            pass
        elif self.domain_information.dim == 2:
            if all(xi in self.domain_information.active_axes for xi in ("x", "y")):
                vorticity = vorticity[2:3]
            elif all(xi in self.domain_information.active_axes for xi in ("x", "z")):
                vorticity = vorticity[1:2]
            elif all(xi in self.domain_information.active_axes for xi in ("y", "z")):
                vorticity = vorticity[0:1]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        return vorticity

    def _compute_absolute_vorticity(
            self,
            velocity: Array
            ) -> Array:
        """Computes absolute vorticity field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: Array
        :return: Buffer of absolute vorticity.
        :rtype: Array
        """
        vorticity = self._compute_vorticity(velocity)
        # TODO eps ad
        absolute_vorticity = jnp.sqrt(jnp.sum(vorticity * vorticity, axis=0) + 1e-100)
        return absolute_vorticity

    def _compute_qcriterion(
            self,
            velocity: Array
            ) -> Array:
        """Computes Q-criterion field for h5 output.

        :param velocity: Buffer of velocities.
        :type velocity: Array
        :return: Buffer of vorticity.
        :rtype: Array
        """
        velocity_grad = self.compute_velocity_gradient(velocity)
        # Rate_of_strain_tensor
        S = 0.5 * (velocity_grad + jnp.transpose(velocity_grad, axes=(1,0,2,3,4)))
        # Vorticity_tensor
        W = 0.5 * (velocity_grad - jnp.transpose(velocity_grad, axes=(1,0,2,3,4)))
        # Q-criterion
        Q = 0.5 * (jnp.linalg.norm(W, axis=(0,1))**2 - jnp.linalg.norm(S, axis=(0,1))**2)
        return Q

    def _compute_dilatation(
            self,
            velocity: Array
            ) -> Array:
        velocity_grad = self.compute_velocity_gradient(velocity)
        dilatation = velocity_grad[0,0] + velocity_grad[1,1] + velocity_grad[2,2]
        return dilatation

    def compute_velocity_gradient(
        self,
        velocity: Array
        ) -> Array:
        cell_sizes = self.domain_information.get_device_cell_sizes()
        active_axes_indices = self.domain_information.active_axes_indices
        equation_type = self.equation_information.equation_type
        if equation_type == "TWO-PHASE-LS":
            nhx,nhy,nhz = self.domain_information.domain_slices_geometry
        else:
            nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        shape = velocity[...,nhx,nhy,nhz].shape
        velocity_grad = []
        for i in range(3):
            velocity_grad.append(
                self.derivative_stencil.derivative_xi(velocity, cell_sizes[i], i
                ) if i in active_axes_indices else jnp.zeros(shape)
            )
        velocity_grad = jnp.stack(velocity_grad, axis=1)
        return velocity_grad