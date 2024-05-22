from functools import partial
from time import time
from typing import Tuple, Union, Dict, Callable
import types

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.data_types.information import ForcingInformation, \
    MassFlowForcingInformation, TemperatureForcingInformation
from jaxfluids.forcing.pid_control import PIDControl
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.levelset.geometry_calculator import compute_fluid_masks
from jaxfluids.turb.statistics.utilities import real_wavenumber_grid, energy_spectrum_physical, energy_spectrum_physical_parallel, wavenumber_grid_parallel
from jaxfluids.data_types.buffers import ForcingBuffers, ForcingParameters, \
    LevelsetFieldBuffers, MassFlowControllerParameters, MaterialFieldBuffers, \
    SimulationBuffers, TimeControlVariables
from jaxfluids.data_types.case_setup import ForcingSetup
from jaxfluids.data_types.numerical_setup import ActiveForcingsSetup, ActivePhysicsSetup
from jaxfluids.domain.helper_functions import reassemble_buffer, \
    split_and_shard_buffer, split_buffer, split_subdomain_dimensions
from jaxfluids.math.parallel_fft import parallel_fft, parallel_ifft
from jaxfluids.config import precision

class Forcing:
    """Class that manages the computation of external forcing terms.

    Currently implemented are:
    1) Mass flow rate forcing
    2) Temperature forcing
    3) Homogeneous isotropic turbulence forcing 
    """

    def __init__(self,
        domain_information: DomainInformation,
        equation_information: EquationInformation,
        material_manager: MaterialManager, 
        unit_handler: UnitHandler,
        forcing_setup: ForcingSetup,
        active_forcings_setups: ActiveForcingsSetup,
        active_physics: ActivePhysicsSetup
        ) -> None:

        self.eps = precision.get_eps()

        self.material_manager = material_manager
        self.unit_handler = unit_handler
        self.domain_information = domain_information
        self.equation_information = equation_information

        self.is_mass_flow_forcing = active_forcings_setups.is_mass_flow_forcing
        self.is_temperature_forcing = active_forcings_setups.is_temperature_forcing
        self.is_turb_hit_forcing = active_forcings_setups.is_turb_hit_forcing
        self.is_acoustic_forcing = active_forcings_setups.is_acoustic_forcing
        self.is_custom_forcing = active_forcings_setups.is_custom_forcing

        self.number_of_cells = domain_information.global_number_of_cells

        if self.is_turb_hit_forcing:
            self.hit_forcing_cutoff = forcing_setup.hit_forcing_cutoff
            self.temperature_target = forcing_setup.temperature_target

        if self.is_mass_flow_forcing:
            flow_direction = forcing_setup.mass_flow_direction
            if flow_direction == "x":
                self.vec    = jnp.array([1.0, 0.0, 0.0])
                self.int_ax = (-1,-2)
                self.index  = 1
            elif flow_direction == "y":
                self.vec    = jnp.array([0.0, 1.0, 0.0])
                self.int_ax = (-3,-1)
                self.index  = 2
            elif flow_direction == "z":
                self.vec    = jnp.array([0.0, 0.0, 1.0])
                self.int_ax = (-3,-2)
                self.index  = 3

            self.mass_flow_target = forcing_setup.mass_flow_target
            self.PID_mass_flow_forcing = PIDControl(
                K_P=5e-1, K_I=5, K_D=0, T_N=5, T_V=1)

        # TEMPERATURE FORCING
        if self.is_temperature_forcing:
            self.temperature_target = forcing_setup.temperature_target

        # ACOUSTIC FORCING
        if self.is_acoustic_forcing:
            acoustic_forcing = forcing_setup.acoustic_forcing 
            self.acoustic_forcing_type = acoustic_forcing.type
            self.acoustic_forcing_axis = acoustic_forcing.axis
            self.acoustic_forcing_axis_id = self.domain_information.axis_to_axis_id[acoustic_forcing.axis]
            self.acoustic_forcing_plane_value = acoustic_forcing.plane_value
            self.acoustic_forcing_function = acoustic_forcing.forcing
    
        self.is_geometric_source = active_physics.is_geometric_source
        if self.is_geometric_source:
            self.symmetry_type = forcing_setup.geometric_source.symmetry_type
            self.symmetry_axis = forcing_setup.geometric_source.symmetry_axis
            self.symmetry_axis_id = domain_information.axis_to_axis_id[self.symmetry_axis]
            self.radial_axis = [axis for axis in domain_information.active_axes if axis != self.symmetry_axis][0]
            self.radial_axis_id = domain_information.axis_to_axis_id[self.radial_axis]

            self.cell_face_slices = [
                [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
                [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
                [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
            ]

        if self.is_custom_forcing:
            self.custom_force_callable = forcing_setup.custom_forcing

    def compute_forcings(
            self,
            simulation_buffers: SimulationBuffers,
            time_control_variables: TimeControlVariables,
            forcing_parameters: ForcingParameters,
            do_runge_kutta_stages: Callable,
            ml_parameters_dict: Union[None, Dict] = None,
            ml_networks_dict: Union[None, Dict] = None,
            ) -> Tuple[ForcingBuffers, ForcingParameters,
                       ForcingInformation]:
        """Computes forcings for temperature, mass flow
        and turbulence kinetic energy.

        :param simulation_buffers: _description_
        :type simulation_buffers: SimulationBuffers
        :param time_control_variables: _description_
        :type time_control_variables: TimeControlVariables
        :param forcing_parameters: _description_
        :type forcing_parameters: ForcingParameters
        :param do_runge_kutta_stages: _description_
        :type do_runge_kutta_stages: Callable
        :return: _description_
        :rtype: Tuple[ForcingBuffers, ForcingParameters, ForcingInformation]
        """

        material_fields = simulation_buffers.material_fields
        levelset_fields = simulation_buffers.levelset_fields

        physical_simulation_time = time_control_variables.physical_simulation_time
        physical_timestep_size = time_control_variables.physical_timestep_size

        conservatives = material_fields.conservatives
        primitives = material_fields.primitives

        levelset = levelset_fields.levelset
        volume_fraction = levelset_fields.volume_fraction

        if self.is_mass_flow_forcing:
            mass_flow_forcing, mass_flow_forcing_infos, \
            mass_flow_controller_params = self.compute_mass_flow_forcing(
                conservatives, primitives, volume_fraction, 
                physical_simulation_time, physical_timestep_size, 
                forcing_parameters.mass_flow_controller_params.current_error, 
                forcing_parameters.mass_flow_controller_params.integral_error)
        else:
            mass_flow_forcing, mass_flow_forcing_infos, \
            mass_flow_controller_params = None, None, None

        if self.is_temperature_forcing:
            temperature_forcing, temperature_forcing_infos = \
            self.compute_temperature_forcing(
                primitives, volume_fraction,
                physical_simulation_time,
                physical_timestep_size)
        else:
            temperature_forcing, temperature_forcing_infos = None, None

        if self.is_turb_hit_forcing:
            material_fields_dash, *_ = do_runge_kutta_stages(
                material_fields, time_control_variables,
                levelset_fields, ml_parameters_dict=ml_parameters_dict,
                ml_networks_dict=ml_networks_dict)
            primes_dash = material_fields_dash.primitives
            turb_hit_forcing = self.compute_turb_hit_forcing(
                primitives, primes_dash, physical_timestep_size,
                physical_simulation_time, forcing_parameters.hit_ek_ref
            )
        else:
            turb_hit_forcing = None

        if self.is_acoustic_forcing:
            acoustic_forcing = self.compute_acoustic_forcing(
                primitives=primitives,
                physical_simulation_time=physical_simulation_time
            )
        else:
            acoustic_forcing = None

        if self.is_custom_forcing:
            custom_forcing = self.compute_custom_forcing(
                primitives=primitives,
                physical_simulation_time=physical_simulation_time
            )
        else:
            custom_forcing = None

        forcing_buffers = ForcingBuffers(
            mass_flow_forcing, turb_hit_forcing,
            temperature_forcing, acoustic_forcing,
            custom_forcing)

        # TODO add acoustic forcing
        forcing_params = ForcingParameters(
            mass_flow_controller_params, 
            forcing_parameters.hit_ek_ref)
        
        # TODO add acoustic forcing
        forcing_infos = ForcingInformation(
            mass_flow_forcing_infos, 
            temperature_forcing_infos)

        return forcing_buffers, forcing_params, forcing_infos
        
    def compute_temperature_forcing(
            self,
            primitives: Array,
            volume_fraction: Array,
            physical_simulation_time: float,
            physical_timestep_size: float
            ) -> Tuple[Array, TemperatureForcingInformation]:
        """Computes temperature forcing.

        :param primitives: _description_
        :type primitives: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param physical_timestep_size: _description_
        :type physical_timestep_size: float
        :return: _description_
        :rtype: Tuple[Array, TemperatureForcingInformation]
        """
        # DOMAIN INFORMATION
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry

        # EQUATION INFORMATION
        mass_ids = self.equation_information.mass_ids
        mass_slices = self.equation_information.mass_slices
        vf_slices = self.equation_information.vf_slices
        species_slices = self.equation_information.species_slices
        energy_ids = self.equation_information.energy_ids
        levelset_model = self.equation_information.levelset_model
        equation_type = self.equation_information.equation_type
        no_fluids = self.equation_information.no_fluids

        # COMPUTE TEMPERATURE
        pressure = primitives[energy_ids,...,nhx,nhy,nhz]
        rho = self.material_manager.get_density(primitives[...,nhx,nhy,nhz])

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            alpha_i = primitives[(vf_slices,) + (...,nhx,nhy,nhz)]
        else:
            None

        temperature = self.material_manager.get_temperature(
            primitives[...,nhx,nhy,nhz],
            density=rho)

        mesh_grid = self.domain_information.compute_device_mesh_grid()
        temperature_target = self.temperature_target(*mesh_grid, physical_simulation_time)

        temperature_error = (temperature_target - temperature)
        
        # print(jnp.squeeze(temperature))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(ncols=2)
        # ax[0].imshow(jnp.squeeze(temperature))
        # ax[1].imshow(jnp.squeeze(temperature_error))
        # plt.show()

        if levelset_model:
            mask_real = compute_fluid_masks(volume_fraction, self.equation_information.levelset_model)
            temperature_error *= mask_real[...,nhx_,nhy_,nhz_]

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            R = self.material_manager.get_specific_gas_constant()
            gamma = self.material_manager.get_gamma(alpha_i=alpha_i)

        else:
            R = self.material_manager.get_specific_gas_constant()
            gamma = self.material_manager.get_gamma()

        forcing = rho * R / (gamma - 1) * temperature_error / physical_timestep_size
        mean_absolute_error = jnp.mean(jnp.abs(temperature_error))

        # BUILD FORCING VECTOR
        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            forcing = [jnp.zeros_like(forcing) for i in range(no_fluids+3)] \
                + [forcing] + [jnp.zeros_like(forcing) for i in range(no_fluids-1)]
        
        elif equation_type in ("SINGLE-PHASE", 
                               "SINGLE-PHASE-SOLID-LS",
                               "TWO-PHASE-LS"):            
            forcing = [jnp.zeros_like(forcing) for i in range(4)] + [forcing]
        
        else:
            raise NotImplementedError

        temperature_forcing_infos = TemperatureForcingInformation(
            mean_absolute_error)

        forcing = jnp.stack(forcing, axis=0)

        return forcing, temperature_forcing_infos

    def compute_mass_flow_forcing(
            self,
            conservatives: Array,
            primitives: Array,
            volume_fraction: Array,
            physical_simulation_time: float,
            physical_timestep_size: float,
            PID_e_new: float,
            PID_e_int: float
            ) -> Tuple[Array, MassFlowForcingInformation,
                       MassFlowControllerParameters]:
        """Computes mass flow forcing.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param physical_timestep_size: _description_
        :type physical_timestep_size: float
        :param PID_e_new: _description_
        :type PID_e_new: float
        :param PID_e_int: _description_
        :type PID_e_int: float
        :return: _description_
        :rtype: Tuple[Array, MassFlowForcingInformation, MassFlowControllerParameters]
        """
        # DOMAIN INFORMATION
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        is_parallel = self.domain_information.is_parallel
        split_factors = self.domain_information.split_factors

        # EQUATION INFORMATION
        levelset_model = self.equation_information.levelset_model

        mass_flow_target = self.mass_flow_target(physical_simulation_time)

        # COMPUTE CURRENT MASS FLOW
        if levelset_model:
            momentum = conservatives[self.index,...,nhx,nhy,nhz] * volume_fraction[...,nhx_,nhy_,nhz_]
        else:
            momentum = conservatives[self.index,...,nhx,nhy,nhz]


        if self.is_geometric_source:
            if is_parallel:
                raise NotImplementedError
            if self.symmetry_type == "AXISYMMETRIC":
                cell_faces = self.domain_information.get_device_cell_faces()[self.radial_axis_id]
                cell_faces = cell_faces * cell_faces
                cell_face_slices = self.cell_face_slices[self.radial_axis_id]
                cell_face_area = jnp.pi * (cell_faces[cell_face_slices[0]] - cell_faces[cell_face_slices[1]])
                # cell_face_area = jnp.pi * (cell_faces[1:] - cell_faces[:-1])
                # cell_face_area = cell_face_area.reshape(np.roll(np.array([-1,1,1]), shift=self.radial_axis_id))
            else:
                raise NotImplementedError
        else:
            cell_face_area = self.domain_information.get_device_cell_face_areas()[self.index-1]
        
        mass_flow_current = jnp.mean(jnp.sum(cell_face_area * momentum, axis=self.int_ax), axis=-1)
        if levelset_model == "FLUID-FLUID":
            mass_flow_current = jnp.sum(mass_flow_current)

        if is_parallel:
            mass_flow_current = jax.lax.all_gather(mass_flow_current, axis_name="i")
            mass_flow_current = split_subdomain_dimensions(mass_flow_current, split_factors)
            mass_flow_current = jnp.mean(jnp.sum(mass_flow_current, axis=self.int_ax))

        # COMPUTE MASS FLOW FORCING
        mass_flow_forcing_scalar, PID_e_new, PID_e_int = \
        self.PID_mass_flow_forcing.compute_output(
            mass_flow_current, mass_flow_target,
            physical_timestep_size, PID_e_new, PID_e_int)
        
        mass_flow_forcing = mass_flow_forcing_scalar * self.vec

        density = primitives[0:1,...,nhx,nhy,nhz]
        vels    = primitives[1:4,...,nhx,nhy,nhz]
        
        body_force_momentum = jnp.einsum("ij..., jk...->ik...", mass_flow_forcing.reshape(3,1), jnp.ones(density.shape))
        body_force_energy = jnp.einsum("ij..., jk...->ik...", mass_flow_forcing.reshape(1,3), vels)

        body_force = jnp.vstack([jnp.zeros(body_force_energy.shape), body_force_momentum, body_force_energy])
        
        mass_flow_forcing_infos = MassFlowForcingInformation(
            mass_flow_current, mass_flow_target,
            mass_flow_forcing_scalar)

        mass_flow_controller_parameters = MassFlowControllerParameters(
            PID_e_new, PID_e_int)
        
        return body_force, mass_flow_forcing_infos, mass_flow_controller_parameters

    def compute_turb_hit_forcing(
            self,
            primitives: Array,
            primes_dash: Array,
            timestep: float,
            physical_simulation_time: float,
            ek_ref: Array = None
            ) -> Array:
        """Computes forcing for HIT 

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param primes_dash: Buffer of intermediate primitive variables which are obtained
            by integrating primitives without forcing term.
        :type primes_dash: Array
        :param timestep: Current time step.
        :type timestep: float
        :return: Buffer of the forcing vector.
        :rtype: Array
        """


        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        mass_ids = self.equation_information.mass_ids
        energy_ids = self.equation_information.energy_ids

        primitives = primitives[:,nhx,nhy,nhz]
        primes_dash = primes_dash[:,nhx,nhy,nhz]

        Tbar = jnp.mean(self.material_manager.get_temperature(pressure=primitives[energy_ids], density=primitives[mass_ids]))

        s_0 = jnp.zeros(primitives[0].shape)
        s_1, s_2, s_3 = self.calculate_velocity_forcing_vector(primitives[1:4], primes_dash[1:4], self.hit_forcing_cutoff, timestep, ek_ref)
        R = self.material_manager.get_specific_gas_constant()
        gamma = self.material_manager.get_gamma()

        mesh_grid = self.domain_information.compute_device_mesh_grid()
        temperature_target = self.temperature_target(*mesh_grid, physical_simulation_time)

        s_4 = primitives[1] * s_1 + primitives[2] * s_2 + primitives[3] * s_3 + (temperature_target - Tbar) * R / (gamma - 1)
        force = [s_0, s_1, s_2, s_3, s_4]
        return primitives[0] * jnp.stack(force)

    def calculate_velocity_forcing_vector(
            self,
            vels: Array,
            vels_dash: Array,
            eta_s: int,
            timestep: float,
            ek_ref: Array
            ) -> Array:
        """Calculates the velocity forcing vector for HIT forcing.

        :param vels: Buffer of velocities.
        :type vels: Array
        :param vels_dash: Buffer of intermediate velocities which are obtained
            by integrating primitives without forcing term.
        :type vels_dash: Array
        :param eta_s: Cut-off wavenumber up to which forcing is applied.
        :type eta_s: int
        :param timestep: Current time step.
        :type timestep: float
        :return: Buffer of the velocity forcing vector.
        :rtype: Array
        """

        is_parallel = self.domain_information.is_parallel
        number_of_cells = self.domain_information.global_number_of_cells
        split_factors = self.domain_information.split_factors
        if is_parallel:
            ek_dash = energy_spectrum_physical_parallel(
                vels_dash, split_factors, multiplicative_factor=0.5)
            split_axis_in = np.argmax(split_factors)
            split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
            split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])
            N = number_of_cells[0]

            vels_hat = parallel_fft(vels, split_factors, split_axis_out)

            k_field = wavenumber_grid_parallel(number_of_cells, split_factors_out)
            one_k2_field = 1.0 / (jnp.sum(jnp.square(k_field), axis=0) + self.eps)
            k_vec = jnp.arange(N)
            k_mag_vec = jnp.sqrt( k_vec**2 + self.eps )
            # TODO eps AD
            shell = (jnp.linalg.norm(k_field,axis=0,ord=2) + 0.5).astype(int)

            Cs_eta = 0.5 / (ek_dash + 1e-10) * (ek_dash - ek_ref) / timestep * (k_mag_vec <= eta_s)
            div_u = k_field[0] * vels_hat[0] + k_field[1] * vels_hat[1] + k_field[2] * vels_hat[2]
            Cs = Cs_eta[shell]
            s_hat = - Cs * (vels_hat - k_field * one_k2_field * div_u)
            
            s_real = parallel_ifft(s_hat, split_factors_out, split_axis_in)

        else:
            ek_dash = energy_spectrum_physical(vels_dash, multiplicative_factor=0.5)
            vels_hat = jnp.fft.rfftn(vels, axes=(-1,-2,-3))
            k_field, k_vec, _ = real_wavenumber_grid(self.number_of_cells[0])
            one_k2_field = 1.0 / (jnp.sum(jnp.square(k_field), axis=0) + self.eps)
            k_mag_vec = jnp.sqrt( k_vec**2 + self.eps )
            # TODO eps AD
            shell = (jnp.linalg.norm(k_field,axis=0,ord=2) + 0.5).astype(int)

            Cs_eta = 0.5 / (ek_dash + 1e-10) * (ek_dash - ek_ref) / timestep * (k_mag_vec <= eta_s)
            div_u = k_field[0] * vels_hat[0] + k_field[1] * vels_hat[1] + k_field[2] * vels_hat[2]
            Cs = Cs_eta[shell]

            s_hat = - Cs * (vels_hat - k_field * one_k2_field * div_u)
            s_real = jnp.fft.irfftn(s_hat, axes=(-1,-2,-3))

        return s_real

    def compute_acoustic_forcing(
            self,
            primitives: Array,
            physical_simulation_time: float
        ) -> Array:
        
        # DOMAIN INFORMATION
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        cell_centers = self.domain_information.get_device_cell_centers()

        # EQUATION INFORMATION
        mass_ids = self.equation_information.mass_ids
        energy_ids = self.equation_information.energy_ids
        
        primitives = primitives[...,nhx,nhy,nhz]
        pressure = primitives[energy_ids]
        density = primitives[mass_ids]

        if self.acoustic_forcing_type == "PLANAR":
            physical_simulation_time = self.unit_handler.dimensionalize(physical_simulation_time, "time")
            ft = self.acoustic_forcing_function(physical_simulation_time)
            ft = self.unit_handler.non_dimensionalize(ft, "pressure")
            
            cell_center_i = cell_centers[self.acoustic_forcing_axis_id]
            sigma = jnp.min(self.domain_information.get_device_cell_sizes()[self.acoustic_forcing_axis_id])
            h = jnp.abs(cell_center_i - self.acoustic_forcing_plane_value)
            delta_fun = 1.0 / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(-0.5 * (h/sigma)**2)
            delta_fun = jnp.moveaxis(delta_fun.reshape(-1,1,1), 0, self.acoustic_forcing_axis_id)

            # TODO: Do we use local speed of sound? Or initial value?
            # c0 = jnp.array([343.2488418652865]).reshape(1,1,1)
            c0 = self.material_manager.get_speed_of_sound(pressure=pressure, density=density)
            gamma = self.material_manager.get_gamma()
            if self.acoustic_forcing_axis == "x":
                omega = jnp.stack([ft / c0, ft * jnp.ones_like(c0), jnp.zeros_like(c0), jnp.zeros_like(c0), c0 / (gamma - 1) * ft])
            if self.acoustic_forcing_axis == "y":
                omega = jnp.stack([ft / c0, jnp.zeros_like(c0), ft * jnp.ones_like(c0), jnp.zeros_like(c0), c0 / (gamma - 1) * ft])
            if self.acoustic_forcing_axis == "z":
                omega = jnp.stack([ft / c0, jnp.zeros_like(c0), jnp.zeros_like(c0), ft * jnp.ones_like(c0), c0 / (gamma - 1) * ft])

            source = omega * delta_fun

        else:
            raise NotImplementedError

        return source

    def compute_custom_forcing(
            self,
            primitives: Array,
            physical_simulation_time: float
        ) -> Array:

        meshgrid = self.domain_information.compute_device_mesh_grid()

        custom_force_list = []
        primes_tuple = self.equation_information.primes_tuple
        for prime_state in primes_tuple:
            prime_state_callable = getattr(self.custom_force_callable, prime_state)
            prime_force = prime_state_callable(*meshgrid, physical_simulation_time)
            custom_force_list.append(prime_force)
        custom_force = jnp.stack(custom_force_list, axis=0) 

        if self.equation_information.levelset_model == "FLUID-FLUID":
            custom_force = jnp.stack([
                custom_force, custom_force], axis=1)

        return custom_force
