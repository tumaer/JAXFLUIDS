from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from jaxfluids.levelset.fluid_fluid.interface_quantities import compute_interface_quantities
from jaxfluids.data_types.buffers import SimulationBuffers, \
    MaterialFieldBuffers, LevelsetFieldBuffers, \
    ForcingParameters, TimeControlVariables, SolidFieldBuffers
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.ml_buffers import MachineLearningSetup, CallablesSetup, ParametersSetup
from jaxfluids.levelset.extension.material_fields.extension_handler import ghost_cell_extension_material_fields
from jaxfluids.data_types.information import StepInformation
from jaxfluids.initialization.helper_functions import create_field_buffer
from jaxfluids.levelset.fluid_solid.interface_quantities import compute_thermal_interface_state
from jaxfluids.data_types.buffers import LevelsetSolidCellIndicesField
from jaxfluids.data_types.buffers import ControlFlowParameters

from jaxfluids.feed_forward.data_types import FeedForwardSetup, ScanFields

if TYPE_CHECKING:
    from jaxfluids.simulation_manager import SimulationManager

Array = jax.Array


def initialize_fields_feed_forward(
        sim_manager: SimulationManager,
        primes_init: Array,
        physical_timestep_size: float,
        t_start: float,
        levelset_init: Array,
        solid_temperature_init: Array,
        solid_velocity_init: Array,
        ml_setup: MachineLearningSetup
    ) -> Tuple[SimulationBuffers, TimeControlVariables, ForcingParameters]:

    # DOMAIN/EQUATION INFORMATION
    domain_information = sim_manager.domain_information
    equation_information = sim_manager.equation_information
    halo_manager = sim_manager.halo_manager
    material_manager = sim_manager.material_manager
    unit_handler = sim_manager.unit_handler

    smallest_cell_size = domain_information.smallest_cell_size
    nh = domain_information.nh_conservatives
    nhx, nhy, nhz = domain_information.domain_slices_conservatives
    split_factors = domain_information.split_factors
    device_number_of_cells = domain_information.device_number_of_cells
    equation_type = equation_information.equation_type
    no_primes = equation_information.no_primes
    levelset_model = equation_information.levelset_model
    numerical_setup = sim_manager.numerical_setup

    fill_edge_halos_material = halo_manager.fill_edge_halos_material
    fill_vertex_halos_material = halo_manager.fill_vertex_halos_material
    
    if sim_manager.numerical_setup.precision.is_double_precision_compute:
        dtype = jnp.float64 
    else:
        dtype = jnp.float32

    if equation_type == "TWO-PHASE-LS":
        leading_dim = (5,2)
    else:
        leading_dim = no_primes


    physical_timestep_size = unit_handler.non_dimensionalize(physical_timestep_size, "time")
    t_start = unit_handler.non_dimensionalize(t_start, "time")

    # NOTE material fields
    primitives = create_field_buffer(nh, device_number_of_cells, dtype, leading_dim)
    primitives = primitives.at[...,nhx,nhy,nhz].set(primes_init)
    quantity_list = equation_information.primitive_quantities
    primitives = unit_handler.non_dimensionalize(primitives, "specified", quantity_list)
    conservatives = sim_manager.equation_manager.get_conservatives_from_primitives(primitives)
    primitives, conservatives = sim_manager.halo_manager.perform_halo_update_material(
        primitives, t_start, fill_edge_halos_material, 
        fill_vertex_halos_material, conservatives,
        ml_setup=ml_setup)
    if equation_information.is_compute_temperature:
        temperature = material_manager.get_temperature(primitives)
        temperature = halo_manager.perform_outer_halo_update_temperature(
            temperature, t_start)
    else:
        temperature = None


    if levelset_model:

        levelset_setup = numerical_setup.levelset
        extension_setup_primitives = levelset_setup.extension.primitives
        extension_setup_solids = levelset_setup.extension.solids

        mixing_setup_conservatives = levelset_setup.mixing.conservatives
        mixing_setup_solids = levelset_setup.mixing.solids
        interface_flux_setup = levelset_setup.interface_flux

        solid_coupling = equation_information.solid_coupling

        # TODO cell based computation not supported for feed foward currently
        if any((
            extension_setup_primitives.interpolation.is_cell_based_computation,
            extension_setup_solids.interpolation.is_cell_based_computation,
            mixing_setup_conservatives.is_cell_based_computation,
            mixing_setup_solids.is_cell_based_computation,
            interface_flux_setup.is_cell_based_computation,
            )):
            assert False, "is_cell_based_computation must be False when using feed_forward"


        levelset_handler = sim_manager.levelset_handler
        extender_primes = levelset_handler.extender_primes
        geometry_calculator = levelset_handler.geometry_calculator
        equation_manager = sim_manager.equation_manager
        solid_properties_manager = sim_manager.solid_properties_manager

        # NOTE levelset fields
        levelset = create_field_buffer(nh, device_number_of_cells, dtype)
        levelset = levelset.at[nhx, nhy, nhz].set(levelset_init)
        levelset = unit_handler.non_dimensionalize(levelset, "length")
        levelset = sim_manager.levelset_handler.reinitializer.set_levelset_cutoff(levelset)
        levelset = sim_manager.halo_manager.perform_halo_update_levelset(levelset, True, True)
        volume_fraction, apertures = geometry_calculator.interface_reconstruction(levelset)
        normal = geometry_calculator.compute_normal(levelset)

        # NOTE solid fields
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError
        else:
            solid_temperature = None
            solid_energy = None

        if solid_coupling.dynamic == "TWO-WAY":
            raise NotImplementedError # TODO
        else:
            solid_velocity = None
        
        # NOTE interpolation based extension requires all halo regions
        if extension_setup_primitives.method == "INTERPOLATION" and \
            not all((fill_vertex_halos_material, fill_edge_halos_material)):
            primitives = halo_manager.perform_halo_update_material(
                primitives, t_start, True, True, None, False,
                ml_setup=ml_setup)

        # interface heat flux/temperature required for extension for conjugate heat
        if solid_coupling.thermal == "TWO-WAY" and any((extension_setup_primitives.method == "INTERPOLATION",
                                                        extension_setup_solids.method == "INTERPOLAPTION")):
            raise NotImplementedError
        else:
            interface_heat_flux = None
            interface_temperature = None
        
        extension_setup = levelset_setup.extension
        narrowband_setup = levelset_setup.narrowband
        
        # ghost cell extension material fields
        conservatives, primitives, _, _ = ghost_cell_extension_material_fields(
            conservatives, primitives, levelset, volume_fraction,
            normal, solid_temperature, solid_velocity,
            interface_heat_flux, interface_temperature,
            None, t_start, extension_setup.primitives,
            narrowband_setup, None,
            extender_primes, equation_manager,
            solid_properties_manager,
            is_initialization=True,
            ml_setup=ml_setup
        )

        primitives, conservatives = halo_manager.perform_halo_update_material(
            primitives, t_start, fill_edge_halos_material,
            fill_vertex_halos_material, conservatives,
            ml_setup=ml_setup)
        if equation_information.is_compute_temperature:
            temperature = material_manager.get_temperature(primitives)
            temperature = halo_manager.perform_outer_halo_update_temperature(
                temperature, t_start)
        else:
            temperature = None

        # interface quantities
        if levelset_model == "FLUID-FLUID":
            fluid_fluid_handler = levelset_handler.fluid_fluid_handler
            extender_interface = fluid_fluid_handler.extender_interface
            curvature = geometry_calculator.compute_curvature(levelset)
            interface_velocity, interface_pressure, _ = \
            compute_interface_quantities(
                primitives, levelset, volume_fraction, normal, curvature, 
                material_manager, extender_interface,
                extension_setup.interface.iterative, narrowband_setup,
                is_initialization=True, ml_setup=ml_setup)
        else:
            interface_velocity, interface_pressure = None, None

        # NOTE ghost cell extension solids
        if solid_coupling.thermal == "TWO-WAY":
            raise NotImplementedError

    else:
        levelset, volume_fraction, apertures = None, None, None
        interface_velocity, interface_pressure = None, None
        solid_energy, solid_temperature, solid_velocity = None, None, None

    # CREATE CONTAINERS
    material_fields = MaterialFieldBuffers(
        conservatives, primitives, temperature)

    levelset_fields = LevelsetFieldBuffers(
        levelset, volume_fraction, apertures, interface_velocity,
        interface_pressure)

    solid_fields = SolidFieldBuffers(
        solid_velocity, solid_energy, solid_temperature)

    simulation_buffers = SimulationBuffers(
        material_fields, levelset_fields,
        solid_fields)

    time_control_variables = TimeControlVariables(
        t_start, 0, physical_timestep_size)

    forcing_parameters = ForcingParameters()

    return simulation_buffers, time_control_variables, forcing_parameters

def configure_multistep(
        do_integration_step_fn: Callable,
        post_process_fn: Callable,
        feed_forward_setup: FeedForwardSetup,
        ml_callables: CallablesSetup
    ) -> Callable:
    """_summary_

    # NOTE Callables are not valid JAX types
    and cannot be passsed directly to scanned
    functions
    
    :param do_integration_step_fn: _description_
    :type do_integration_step_fn: Callable
    :param post_process_fn: _description_
    :type post_process_fn: Callable
    :param feed_forward_setup: _description_
    :type feed_forward_setup: FeedForwardSetup
    :param ml_networks_dict: _description_
    :type ml_networks_dict: Dict
    :return: _description_
    :rtype: Callable
    """

    outer_steps = feed_forward_setup.outer_steps
    inner_steps = feed_forward_setup.inner_steps
    is_scan = feed_forward_setup.is_scan
    is_checkpoint_inner_step = feed_forward_setup.is_checkpoint_inner_step
    is_checkpoint_integration_step = feed_forward_setup.is_checkpoint_integration_step
    is_include_t0 = feed_forward_setup.is_include_t0

    def wrapped_do_integration_step(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,    
            ml_parameters: ParametersSetup
        ) -> Tuple[JaxFluidsBuffers, Dict]:

        # TODO hard coded flags
        jxf_buffers = JaxFluidsBuffers(
            simulation_buffers,
            time_control_variables,
            forcing_parameters,
            StepInformation()
        )
        control_flow_params = ControlFlowParameters(
            perform_reinitialization=True,
            perform_compression=False,
            is_cumulative_statistics=False,
            is_logging_statistics=False,
            is_feed_foward=True
        )
        
        return do_integration_step_fn(
            jxf_buffers,
            control_flow_params,
            ml_parameters=ml_parameters, 
            ml_callables=ml_callables
        )

    def wrapped_do_integration_step_scan(scan_fields: ScanFields, _: Any) -> Tuple[ScanFields, Any]:
        simulation_buffers = scan_fields.simulation_buffers
        time_control_variables = scan_fields.time_control_variables
        forcing_parameters = scan_fields.forcing_parameters
        ml_parameters = scan_fields.ml_parameters

        jxf_buffers, callback_dict \
        = wrapped_do_integration_step(
            simulation_buffers,
            time_control_variables,
            forcing_parameters,
            ml_parameters
        )

        scan_fields = ScanFields(
            jxf_buffers.simulation_buffers,
            jxf_buffers.time_control_variables,
            jxf_buffers.forcing_parameters,
            ml_parameters
        )

        return scan_fields, _

    def inner_step_unroll(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters: ParametersSetup
        ) -> Tuple[SimulationBuffers, TimeControlVariables, 
        ForcingParameters, StepInformation]:
        for _ in range(inner_steps):
            if is_checkpoint_integration_step:
                simulation_buffers, time_control_variables, \
                forcing_parameters, step_information = jax.checkpoint(wrapped_do_integration_step)(
                    simulation_buffers, time_control_variables,
                    forcing_parameters, ml_parameters,
                )
            else:
                simulation_buffers, time_control_variables, \
                forcing_parameters, step_information = wrapped_do_integration_step(
                    simulation_buffers, time_control_variables,
                    forcing_parameters, ml_parameters,
                )
        return simulation_buffers, time_control_variables, \
            forcing_parameters, step_information

    if is_checkpoint_inner_step:
        inner_step_unroll = jax.checkpoint(inner_step_unroll)

    def outer_step_unroll(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters: ParametersSetup
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
                forcing_parameters, ml_parameters)

            current_time = time_control_variables.physical_simulation_time
            current_step = time_control_variables.simulation_step

            out_buffer = post_process_fn(simulation_buffers)
            out_buffer_list.append(out_buffer)
            out_times_list.append(current_time)

        no_out_buffers = len(out_buffer_list)
        keys = out_buffer_list[0].keys()
        out_buffer = {
            key: jnp.array([out_buffer_list[i][key] for i in range(no_out_buffers)])
            for key in keys
        }
        out_times = jnp.array(out_times_list)
        return out_buffer, out_times
            
    def inner_step_scan(scan_fields: ScanFields, _: Any) -> Tuple[Dict, Dict]:
        
        if is_checkpoint_integration_step:
            scan_fields, _ = jax.lax.scan(jax.checkpoint(wrapped_do_integration_step_scan),
                scan_fields, xs=None, length=inner_steps)
        else:
            scan_fields, _ = jax.lax.scan(wrapped_do_integration_step_scan,
                scan_fields, xs=None, length=inner_steps)
        
        simulation_buffers = scan_fields.simulation_buffers
        current_time = scan_fields.time_control_variables.physical_simulation_time
        simulation_step = scan_fields.time_control_variables.simulation_step
        out_fields = {
            "out_buffer": post_process_fn(simulation_buffers),
            "out_times": current_time
        }
        return scan_fields, out_fields

    if is_checkpoint_inner_step:
        inner_step_scan = jax.checkpoint(inner_step_scan)

    def outer_step_scan(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters: ParametersSetup
        ) -> Tuple[Array, Array]:

        if is_include_t0:
            out_buffer_init = post_process_fn(simulation_buffers)
            out_times_init = time_control_variables.physical_simulation_time

        scan_fields = ScanFields(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters)
        _, out_fields = jax.lax.scan(inner_step_scan, scan_fields, xs=None,
            length=outer_steps)

        if is_include_t0:
            out_buffer = {
                key: jnp.concatenate(
                    [out_buffer_init[key][None], out_fields["out_buffer"][key]]
                ) for key in out_fields["out_buffer"].keys()
            }
            out_times = jnp.concatenate([jnp.array([out_times_init]), out_fields["out_times"]])

        else:
            out_buffer, out_times = out_fields["out_buffer"], out_fields["out_times"]

        return out_buffer, out_times





    # TODO
    def wrapped_do_integration_step_jaxforloop(
            _: int,
            fields: ScanFields
            ) -> ScanFields:
        simulation_buffers = fields.simulation_buffers
        time_control_variables = fields.time_control_variables
        forcing_parameters = fields.forcing_parameters
        ml_parameters = fields.ml_parameters
        simulation_buffers, time_control_variables, \
        forcing_parameters, _ = wrapped_do_integration_step(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters)
        fields = ScanFields(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters)
        return fields
    
    def inner_step_jaxforloop(
            i : int,
            fields: ScanFields
            ) -> ScanFields:
        fields: ScanFields = jax.lax.fori_loop(0, inner_steps, wrapped_do_integration_step_jaxforloop, fields)
        return fields


    if is_checkpoint_inner_step:
        inner_step_jaxforloop = jax.checkpoint(inner_step_jaxforloop)

    def outer_step_jaxforloop(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables, 
            forcing_parameters: ForcingParameters,
            ml_parameters: ParametersSetup
        ) -> Tuple[Array, Array]:

        # NOTE call post_process_fn() to infer shape of out buffer
        out_fields_init = {
            "out_buffer": post_process_fn(simulation_buffers),
            "out_times": jnp.array(time_control_variables.physical_simulation_time)}
        
        if is_include_t0:
            no = inner_steps*outer_steps+1
        else:
            no = inner_steps*outer_steps

        out_fields = jax.tree.map(lambda buffer: jnp.zeros((no,)+buffer.shape), out_fields_init)
        if is_include_t0:
            out_fields = jax.tree.map(lambda buffer, buffer_i: buffer.at[0].set(buffer_i), out_fields, out_fields_init)

        fields = ScanFields(
            simulation_buffers, time_control_variables,
            forcing_parameters, ml_parameters)

        def body(
                index: int,
                args: Tuple[ScanFields, Dict]):
            fields, out_fields = args
            fields: ScanFields = jax.lax.fori_loop(0, inner_steps, inner_step_jaxforloop, fields)
            simulation_buffers = fields.simulation_buffers
            current_time = fields.time_control_variables.physical_simulation_time
            out_fields_i = {
                "out_buffer": post_process_fn(simulation_buffers),
                "out_times": jnp.array(current_time)}
            index += 1 if is_include_t0 else 0
            out_fields = jax.tree.map(lambda buffer, buffer_i: buffer.at[index].set(buffer_i), out_fields, out_fields_i)
            return (fields, out_fields)
        
        fields, out_fields = jax.lax.fori_loop(0, outer_steps, body, (fields, out_fields))

        out_buffer, out_times = out_fields["out_buffer"], out_fields["out_times"]
        
        return out_buffer, out_times




    def multistep(
            simulation_buffers: SimulationBuffers, 
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            ml_parameters: ParametersSetup
        ) -> Tuple[Array, Array]:
        if is_scan:
            out_buffer, out_times = outer_step_scan(
                simulation_buffers, time_control_variables, 
                forcing_parameters, ml_parameters)
        else:
            out_buffer, out_times = outer_step_unroll(
                simulation_buffers, time_control_variables, 
                forcing_parameters, ml_parameters)
            
        return out_buffer, out_times
    
    return multistep