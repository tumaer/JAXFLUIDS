from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, TYPE_CHECKING, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.buffers import SimulationBuffers, \
    MaterialFieldBuffers, LevelsetFieldBuffers, \
    ForcingParameters, TimeControlVariables
from jaxfluids.data_types.information import StepInformation
from jaxfluids.initialization.helper_functions import create_field_buffer

if TYPE_CHECKING:
    from jaxfluids.simulation_manager import SimulationManager

class ScanFields(NamedTuple):
    simulation_buffers: SimulationBuffers
    time_control_variables: TimeControlVariables
    forcing_parameters: ForcingParameters
    ml_paramters: Dict


def initialize_fields_for_feedforward(
        sim_manager: SimulationManager,
        primes_init: Array,
        physical_timestep_size: float,
        t_start: float,
        levelset_init: Array,
        solid_interface_velocity_init: Array,
        ) -> Tuple[SimulationBuffers, TimeControlVariables, ForcingParameters]:

    # DOMAIN/EQUATION INFORMATION
    nh = sim_manager.domain_information.nh_conservatives
    nhx, nhy, nhz = sim_manager.domain_information.domain_slices_conservatives
    split_factors = sim_manager.domain_information.split_factors
    device_number_of_cells = sim_manager.domain_information.device_number_of_cells
    equation_type = sim_manager.equation_information.equation_type
    no_primes = sim_manager.equation_information.no_primes
    levelset_model = sim_manager.equation_information.levelset_model
    is_viscous_flux = sim_manager.numerical_setup.active_physics.is_viscous_flux
    
    if sim_manager.numerical_setup.precision.is_double_precision_compute:
        dtype = jnp.float64 
    else:
        dtype = jnp.float32

    if equation_type == "TWO-PHASE-LS":
        leading_dim = (5,2)
    else:
        leading_dim = no_primes

    # INITIALIZE MATERIAL FIELDS
    primitives = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim)
    primitives = primitives.at[..., nhx, nhy, nhz].set(primes_init)
    conservatives = sim_manager.equation_manager.get_conservatives_from_primitives(primitives)
    primitives, conservatives = sim_manager.halo_manager.perform_halo_update_material(
        primitives, t_start, is_viscous_flux, False, conservatives)

    # INITIALIZE LEVELSET FIELD
    if levelset_model:
        assert_str = (
            "Consistency error while initializing fields for feed foward. "
            f"Level-set model {levelset_model} is active, however levelset_init is "
            f"of type {type(levelset_init)}. levelset_init must be a jax.Array or "
            "a np.ndarray."
        )
        assert isinstance(levelset_init, (Array, np.ndarray)), assert_str

        levelset_handler = sim_manager.levelset_handler
        geometry_calculator = levelset_handler.geometry_calculator
        ghost_cell_handler = levelset_handler.ghost_cell_handler
        interface_quantity_computer = levelset_handler.interface_quantity_computer

        levelset = create_field_buffer(nh, device_number_of_cells, dtype)
        levelset = levelset.at[nhx, nhy, nhz].set(levelset_init)
        levelset = sim_manager.levelset_handler.reinitializer.set_levelset_cutoff(levelset)
        levelset = sim_manager.halo_manager.perform_halo_update_levelset(levelset, True, True)
        volume_fraction, apertures = sim_manager.levelset_handler.compute_volume_fraction_and_apertures(levelset)

        if levelset_model == "FLUID-SOLID-DYNAMIC-COUPLED":
            interface_velocity = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim=3)
            interface_velocity = interface_velocity.at[...,nhx,nhy,nhz].set(solid_interface_velocity_init)
            solid_interface_velocity = interface_velocity[...,nhx,nhy,nhz]
            interface_pressure = None
        elif levelset_model == "FLUID-SOLID-DYNAMIC":
            interface_velocity = None
            solid_interface_velocity = interface_quantity_computer.compute_solid_velocity(t_start)
            interface_pressure = None
        elif levelset_model == "FLUID-FLUID":
            interface_velocity, interface_pressure = levelset_handler.compute_interface_quantities(
                primitives, levelset, None, None, 200, 0.3)
            solid_interface_velocity = None
        else:
            interface_velocity = None
            solid_interface_velocity = None
            interface_pressure = None
           
        normal = geometry_calculator.compute_normal(levelset)
        conservatives, primitives, *_ = ghost_cell_handler.perform_ghost_cell_treatment(
            conservatives, primitives, levelset, volume_fraction,
            t_start, normal, solid_interface_velocity)
        primitives, conservatives = sim_manager.halo_manager.perform_halo_update_material(
            primitives, t_start, is_viscous_flux, False, conservatives)

    else:
        levelset, volume_fraction, apertures = None, None, None
        interface_velocity, interface_pressure = None, None

    material_fields = MaterialFieldBuffers(
        conservatives=conservatives, 
        primitives=primitives)

    levelset_fields = LevelsetFieldBuffers(
        levelset=levelset, volume_fraction=volume_fraction,
        apertures=apertures, interface_velocity=interface_velocity,
        interface_pressure=interface_pressure)

    simulation_buffers = SimulationBuffers(
        material_fields, levelset_fields)

    time_control_variables = TimeControlVariables(
        physical_simulation_time=t_start, 
        simulation_step=0,
        physical_timestep_size=physical_timestep_size)

    forcing_parameters = ForcingParameters()

    return simulation_buffers, time_control_variables, forcing_parameters

def configure_multistep(
        do_integration_step_fn: Callable,
        post_process_fn: Callable,
        outer_steps: int,
        inner_steps: int,
        is_scan: bool,
        is_checkpoint: bool,
        is_include_t0: bool,
        ml_networks_dict: Dict
        ) -> Callable:
    """_summary_

    :param do_integration_step_fn: _description_
    :type do_integration_step_fn: Callable
    :param post_process_fn: _description_
    :type post_process_fn: Callable
    :param outer_steps: _description_
    :type outer_steps: int
    :param inner_steps: _description_
    :type inner_steps: int
    :param is_scan: _description_
    :type is_scan: bool
    :param is_checkpoint: _description_
    :type is_checkpoint: bool
    :param is_include_t0: _description_
    :type is_include_t0: bool
    :param ml_networks_dict: _description_
    :type ml_networks_dict: Dict
    :return: _description_
    :rtype: Callable
    """

    def wrapped_do_integration_step(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,    
            ml_parameters_dict: Dict
        ) -> Tuple[SimulationBuffers, TimeControlVariables, 
                   ForcingParameters, StepInformation]:

        perform_compression = False
        perform_reinitialization = True
        return do_integration_step_fn(
            simulation_buffers, 
            time_control_variables,
            forcing_parameters, 
            perform_reinitialization=perform_reinitialization,
            perform_compression=perform_compression,
            ml_parameters_dict=ml_parameters_dict, 
            ml_networks_dict=ml_networks_dict,
            is_feedforward=True)

    def wrapped_do_integration_step_scan(scan_fields: ScanFields, _: Any) -> Tuple[ScanFields, Any]:
        simulation_buffers = scan_fields.simulation_buffers
        time_control_variables = scan_fields.time_control_variables
        forcing_parameters = scan_fields.forcing_parameters
        ml_parameters_dict = scan_fields.ml_paramters

        simulation_buffers, time_control_variables, \
        forcing_parameters, step_information = wrapped_do_integration_step(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters_dict=ml_parameters_dict)

        scan_fields = ScanFields(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters_dict)

        return scan_fields, _

    def inner_step_unroll(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters_dict: Dict
            ) -> Tuple[SimulationBuffers, TimeControlVariables, 
        ForcingParameters, StepInformation]:
        for _ in range(inner_steps):
            simulation_buffers, time_control_variables, \
            forcing_parameters, step_information = wrapped_do_integration_step(
                simulation_buffers, time_control_variables,
                forcing_parameters, ml_parameters_dict=ml_parameters_dict,
            )
        return simulation_buffers, time_control_variables, \
            forcing_parameters, step_information

    if is_checkpoint:
        inner_step_unroll = jax.checkpoint(inner_step_unroll)

    def outer_step_unroll(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters_dict: Dict
        ) -> Tuple[Array, Array]:
        if is_include_t0:
            out_buffer_list = [post_process_fn(simulation_buffers)]
            out_times_list = [time_control_variables.physical_simulation_time]
        else:
            out_buffer_list, out_times_list = [], []

        for _ in range(outer_steps):
            # print("HERE", forcings_dictionary)
            simulation_buffers, time_control_variables, \
            forcing_parameters, step_information = inner_step_unroll(
                simulation_buffers, time_control_variables,
                forcing_parameters, ml_parameters_dict)

            current_time = time_control_variables.physical_simulation_time
            current_step = time_control_variables.simulation_step

            out_buffer = post_process_fn(simulation_buffers)
            out_buffer_list.append(out_buffer)
            out_times_list.append(current_time)

        no_out_buffers = len(out_buffer_list)
        no_out_buffer_fields = len(out_buffer_list[0])
        out_buffer = tuple(
            jnp.array([
                out_buffer_list[i][j] for i in range(no_out_buffers)]
            ) for j in range(no_out_buffer_fields)
        )
        out_times = jnp.array(out_times_list)
        return out_buffer, out_times
            
    def inner_step_scan(scan_fields: ScanFields, _: Any) -> Tuple[Dict, Dict]:
        scan_fields, _ = jax.lax.scan(wrapped_do_integration_step_scan,
            scan_fields, xs=None, length=inner_steps)
        
        simulation_buffers = scan_fields.simulation_buffers
        current_time = scan_fields.time_control_variables.physical_simulation_time
        out_fields = {
            "out_buffer": post_process_fn(simulation_buffers),
            "out_times": current_time}
        return scan_fields, out_fields

    if is_checkpoint:
        inner_step_scan = jax.checkpoint(inner_step_scan)

    def outer_step_scan(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters_dict: Dict
        ) -> Tuple[Array, Array]:
        if is_include_t0:
            out_buffer_init = post_process_fn(simulation_buffers)
            out_times_init = time_control_variables.physical_simulation_time

        scan_fields = ScanFields(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters_dict)
        _, out_fields = jax.lax.scan(inner_step_scan, scan_fields, xs=None,
            length=outer_steps)

        if is_include_t0:
            no_out_buffers = len(out_fields["out_buffer"])
            out_buffer = tuple(
                jnp.vstack(
                    [out_buffer_init[i][None], out_fields["out_buffer"][i]]
                ) for i in range(no_out_buffers)
            )
            out_times = jnp.concatenate([jnp.array([out_times_init]), out_fields["out_times"]])
        else:
            out_buffer, out_times = out_fields["out_buffer"], out_fields["out_times"]
        return out_buffer, out_times

    def multistep(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            ml_parameters_dict: Dict
        ) -> Tuple[Array, Array]:
        if is_scan:
            out_buffer, out_times = outer_step_scan(
                simulation_buffers, time_control_variables, 
                forcing_parameters, ml_parameters_dict)
        else:
            out_buffer, out_times = outer_step_unroll(
                simulation_buffers, time_control_variables, 
                forcing_parameters, ml_parameters_dict)
        return out_buffer, out_times

    return multistep