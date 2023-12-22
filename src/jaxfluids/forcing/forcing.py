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
from typing import Tuple, Union, Dict
import types

import jax
import jax.numpy as jnp
from jaxfluids import levelset

from jaxfluids.forcing.pid_control import PIDControl
from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.io_utils.logger import Logger
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.turb.turb_stats_manager import TurbStatsManager

class Forcing:
    """Class that manages the computation of external forcing terms.

    Currently implemented are:
    1) Mass flow rate forcing
    2) Temperature forcing
    3) Homogeneous isotropic turbulence forcing 
    """

    def __init__(self, domain_information: DomainInformation, material_manager: MaterialManager, unit_handler: UnitHandler,
        levelset_handler: Union[LevelsetHandler, None], levelset_type: str, is_mass_flow_forcing: bool, is_temperature_forcing: bool, is_turb_hit_forcing: bool,
        mass_flow_target: Union[float, types.LambdaType], flow_direction: str, temperature_target: Union[float, types.LambdaType]) -> None:

        # BOOLS FORCINGS
        self.is_mass_flow_forcing   = is_mass_flow_forcing
        self.is_temperature_forcing = is_temperature_forcing
        self.is_turb_hit_forcing    = is_turb_hit_forcing

        # MATERIAL AND UNIT HANDLER
        self.material_manager   = material_manager
        self.unit_handler       = unit_handler
        self.levelset_handler   = levelset_handler
        if is_turb_hit_forcing:
            self.turb_stats_manager = TurbStatsManager(domain_information, material_manager)
            self.k_mag_vec      = self.turb_stats_manager.k_mag_vec
            self.k_field        = self.turb_stats_manager.k_field
            self.one_k2_field   = self.turb_stats_manager.one_k2_field
            self.shell          = self.turb_stats_manager.shell
        
        # LEVELSET INTERFACE INTERACTION TYPE
        self.levelset_type = levelset_type

        # DOMAIN INFORMATION
        self.inactive_axis_indices      = [{"x": 0, "y": 1, "z": 2}[axis] for axis in domain_information.inactive_axis]
        self.nhx, self.nhy, self.nhz    = domain_information.domain_slices_conservatives
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nx, self.ny, self.nz       = domain_information.number_of_cells
        dx, dy, dz                      = domain_information.cell_sizes
        self.dim                        = domain_information.dim
        self.cell_centers               = domain_information.cell_centers
        self.active_axis_indices        = domain_information.active_axis_indices

        # BOOLS FORCINGS
        self.is_mass_flow_forcing   = is_mass_flow_forcing
        self.is_temperature_forcing = is_temperature_forcing
        self.is_turb_hit_forcing    = is_turb_hit_forcing

        # MASS FLOW FORCING
        self.mass_flow_target       = mass_flow_target 
        self.PID_mass_flow_forcing  = PIDControl(K_P = 5e-1, K_I = 5, K_D = 0, T_N = 5, T_V = 1)

        if flow_direction == "x":
            self.vec    = jnp.array([1.0, 0.0, 0.0])
            self.int_ax = (-1,-2)
            self.index  = 1
            self.dA     = dy * dz
        elif flow_direction == "y":
            self.vec    = jnp.array([0.0, 1.0, 0.0])
            self.int_ax = (-3,-1)
            self.index  = 2
            self.dA     = dx * dz
        elif flow_direction == "z":
            self.vec    = jnp.array([0.0, 0.0, 1.0])
            self.int_ax = (-3,-2)
            self.index  = 3
            self.dA     = dx * dy

        # TEMPERATURE FORCING
        self.temperature_target = temperature_target
    
    def compute_forcings(self, primes: jnp.ndarray, cons: jnp.ndarray, levelset: Union[jnp.ndarray, None],
        volume_fraction: Union[jnp.ndarray, None], current_time: float, timestep_size: float, PID_e_new: float, PID_e_int: float,
        logger: Logger, primes_dash: Union[jnp.ndarray, None] = None, **kwargs) -> Dict:
        
        """Computes forcings for temperature, mass flow and turbulence kinetic energy.

        :param primes: buffer of primitive variables
        :type primes: jnp.ndarray
        :param primes_dash: buffer of primitive variables for next time step without forcing
        :type primes_dash: Union[jnp.ndarray, None]
        :param cons: buffer of conservative variables
        :type cons: jnp.ndarray
        :param volume_fraction: buffer of volume fractions
        :type volume_fraction: Union[jnp.ndarray, None]
        :param mask_real: mask indicating the real fluid
        :type mask_real: Union[jnp.ndarray, None]
        :param current_time: current physical simulation time
        :type current_time: float
        :param timestep_size: current physical time step size
        :type timestep_size: float
        :param PID_e_new: Error of previous timestep for PID controller 
        :type PID_e_new: float
        :param PID_e_int: Accumalated error for PID controller
        :type PID_e_int: float
        :param logger: Logger for terminal output
        :type logger: Logger
        :return: Dictionary containing buffers of forcings
        :rtype: Dict
        """

        forcings_dictionary = {}

        if self.is_mass_flow_forcing:
            mass_flow_forcing, mass_flow_current, mass_flow_target, PID_e_new, PID_e_int = self.compute_mass_flow_forcing(cons, primes, volume_fraction, current_time, timestep_size, PID_e_new, PID_e_int)
            forcings_dictionary.update({
                "mass_flow": {
                    "force": mass_flow_forcing,
                    "PID_e_new": PID_e_new,
                    "PID_e_int": PID_e_int
                },
            })
            logger.log_start_time_step([
                'PID CONTROL',
                'MASS FLOW TARGET   = %4.4e' %(mass_flow_target),
                'MASS FLOW CURRENT  = %4.4e' %(mass_flow_current),
            ])
        
        if self.is_temperature_forcing:
            temperature_forcing, temperature_error = self.compute_temperature_forcing(primes, levelset, volume_fraction, current_time, timestep_size)
            forcings_dictionary.update({
                "temperature": {
                    "force": temperature_forcing,
                },
            })
            logger.log_start_time_step([
                'TEMPERATURE CONTROL',
                'TEMPERATURE ERROR  = %4.4e' % temperature_error,
            ])

        if self.is_turb_hit_forcing:
            turb_hit_forcing = self.compute_turb_hit_forcing(primes, primes_dash, timestep_size)
            forcings_dictionary.update({
                "turbulence": {
                    "force": turb_hit_forcing
                }
            })

        return forcings_dictionary
        
    @partial(jax.jit, static_argnums=(0))
    def compute_temperature_forcing(self, primes: jnp.ndarray, levelset: Union[jnp.ndarray, None], volume_fraction: Union[jnp.ndarray, None],
            current_time: float, timestep_size: float) -> Tuple[jnp.ndarray, float]:
        """Computes temperature forcing.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param levelset: Buffer of level-set field.
        :type levelset: Union[jnp.ndarray, None]
        :param volume_fraction: Buffer of volume fraction field.
        :type volume_fraction: Union[jnp.ndarray, None]
        :param current_time: Current simulation time.
        :type current_time: float
        :param timestep_size: Current integration time step.
        :type timestep_size: float
        :return: Buffer of the forcing vector and the mean absolute error wrt the temperature target.
        :rtype: Tuple[jnp.ndarray, float]
        """
       
        # COMPUTE TEMPERATURE
        temperature     = self.material_manager.get_temperature(primes[4,...,self.nhx,self.nhy,self.nhz], primes[0,...,self.nhx,self.nhy,self.nhz])

        # COMPUTE LAMBDA INPUTS
        mesh_grid = [jnp.meshgrid(*self.cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
        for i in range(len(mesh_grid)):
            mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")
        current_time = self.unit_handler.dimensionalize(current_time, "time")

        # COMPUTE TEMPERATURE TARGET
        if type(self.temperature_target) == types.LambdaType:
            temperature_target = self.temperature_target(*mesh_grid, current_time)
            for axis in self.inactive_axis_indices:
                temperature_target = jnp.expand_dims(temperature_target, axis)
        else:
            temperature_target = self.temperature_target
        temperature_target = self.unit_handler.non_dimensionalize(temperature_target, "temperature")

        # COMPUTE REAL FLUID MASK
        if self.levelset_type != None:
            mask_real, _ = self.levelset_handler.compute_masks(levelset, volume_fraction)

        # COMPUTE TEMPERATURE FORCING
        R, gamma, rho       = self.material_manager.R, self.material_manager.gamma, primes[0,...,self.nhx,self.nhy,self.nhz]
        temperature_error   = (temperature_target - temperature) * mask_real[...,self.nhx_,self.nhy_,self.nhz_] if self.levelset_type != None else temperature_target - temperature
        forcing             = rho * R * gamma/(gamma - 1) * (temperature_error) / timestep_size
        mean_absolute_error = jnp.mean(jnp.abs(temperature_error))
        forcing             = [jnp.zeros_like(forcing) for i in range(4)] + [forcing]

        return jnp.stack(forcing, axis=0), mean_absolute_error

    @partial(jax.jit, static_argnums=(0))
    def compute_mass_flow_forcing(self, cons: jnp.ndarray, primes: jnp.ndarray, volume_fraction: Union[jnp.ndarray, None], 
        current_time: float, timestep_size: float, PID_e_new: float, PID_e_int: float) -> Tuple[jnp.ndarray, float, float, float, float]:
        """Computes mass flow forcing

        :param cons: Buffer of the conservative variables.
        :type cons: jnp.ndarray
        :param primes: Buffer of the primitive variables.
        :type primes: jnp.ndarray
        :param volume_fraction: Buffer of the volume fraction in two-phase flows.
        :type volume_fraction: Union[jnp.ndarray, None]
        :param current_time: Current simulation time.
        :type current_time: float
        :param timestep_size: Current time step.
        :type timestep_size: float
        :param PID_e_new: Current PID error 
        :type PID_e_new: float
        :param PID_e_int: Current PID integral error
        :type PID_e_int: float
        :return: Buffer of the body force, current mass flow, mass flow target, PID error, PID integral error
        :rtype: Tuple[jnp.ndarray, float, float]
        """
        # COMPUTE MASS FLOW TARGET
        if type(self.mass_flow_target) == types.LambdaType:
            mass_flow_target = self.mass_flow_target(self.unit_handler.dimensionalize(current_time, "time"))
        else:
            mass_flow_target = self.mass_flow_target
        mass_flow_target = self.unit_handler.non_dimensionalize(mass_flow_target, "mass_flow")

        # COMPUTE CURRENT MASS FLOW
        momentum = cons[self.index, ..., self.nhx, self.nhy, self.nhz] * volume_fraction[...,self.nhx_,self.nhy_,self.nhz_] if self.levelset_type != None else cons[self.index, ..., self.nhx, self.nhy, self.nhz]
        mass_flow_current = jnp.mean(jnp.sum(self.dA * momentum, axis=self.int_ax), axis=-1)
        mass_flow_current = jnp.sum(mass_flow_current) if self.levelset_type == "FLUID-FLUID" else mass_flow_current

        # COMPUTE MASS FLOW FORCING
        mass_flow_forcing_scalar, PID_e_new, PID_e_int  = self.PID_mass_flow_forcing.compute_output(mass_flow_current, mass_flow_target, timestep_size, PID_e_new, PID_e_int)
        mass_flow_forcing = mass_flow_forcing_scalar * self.vec

        density = primes[0:1,...,self.nhx,self.nhy,self.nhz]
        vels    = primes[1:4,...,self.nhx,self.nhy,self.nhz]
        
        body_force_momentum = jnp.einsum("ij..., jk...->ik...", mass_flow_forcing.reshape(3,1), jnp.ones(density.shape))
        body_force_energy   = jnp.einsum("ij..., jk...->ik...", mass_flow_forcing.reshape(1,3), vels)

        body_force          = jnp.vstack([jnp.zeros(body_force_energy.shape), body_force_momentum, body_force_energy])

        return body_force, mass_flow_current, mass_flow_target, PID_e_new, PID_e_int

    @partial(jax.jit, static_argnums=(0))
    def compute_turb_hit_forcing(self, primes: jnp.ndarray, primes_dash: jnp.ndarray, timestep: float) -> jnp.ndarray:
        """Computes forcing for HIT 

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param primes_dash: Buffer of intermediate primitive variables which are obtained
            by integrating primes without forcing term.
        :type primes_dash: jnp.ndarray
        :param timestep: Current time step.
        :type timestep: float
        :return: Buffer of the forcing vector.
        :rtype: jnp.ndarray
        """
        primes      = primes[:,self.nhx,self.nhy,self.nhz]
        primes_dash = primes_dash[:,self.nhx,self.nhy,self.nhz]

        # TODO MAKE eta_s user-specified parameter
        eta_s = 2
        Tbar = jnp.mean(self.material_manager.get_temperature(p=primes[4], rho=primes[0]))

        s_0 = jnp.zeros(primes[0].shape)
        s_1, s_2, s_3 = self.calculate_velocity_forcing_vector(primes[1:4], primes_dash[1:4], eta_s, timestep)
        s_4 = primes[1] * s_1 + primes[2] * s_2 + primes[3] * s_3 + (self.temperature_target - Tbar) * self.material_manager.R / (self.material_manager.gamma - 1)
        force = [s_0, s_1, s_2, s_3, s_4]
        return primes[0] * jnp.stack(force)

    def calculate_velocity_forcing_vector(self, vels: jnp.ndarray, vels_dash: jnp.ndarray, eta_s: int, timestep: float) -> jnp.ndarray:
        """Calculates the velocity forcing vector for HIT forcing.

        :param vels: Buffer of velocities.
        :type vels: jnp.ndarray
        :param vels_dash: Buffer of intermediate velocities which are obtained
            by integrating primes without forcing term.
        :type vels_dash: jnp.ndarray
        :param eta_s: Cut-off wavenumber up to which forcing is applied.
        :type eta_s: int
        :param timestep: Current time step.
        :type timestep: float
        :return: Buffer of the velocity forcing vector.
        :rtype: jnp.ndarray
        """

        vels_hat      = jnp.stack([jnp.fft.rfftn(vels[ii], axes=(2,1,0)) for ii in range(3)])
        vels_dash_hat = jnp.stack([jnp.fft.rfftn(vels_dash[ii], axes=(2,1,0)) for ii in range(3)])

        ek      = self.turb_stats_manager.energy_spectrum_spectral(vels_hat)
        ek_dash = self.turb_stats_manager.energy_spectrum_spectral(vels_dash_hat)

        Cs_eta = 0.5 / (ek_dash + 1e-10) * (ek_dash - ek) / timestep * (self.k_mag_vec <= eta_s)
        div_u = self.k_field[0] * vels_hat[0] + self.k_field[1] * vels_hat[1] + self.k_field[2] * vels_hat[2]
        Cs = Cs_eta[self.shell]

        s_hat = [
            -Cs * (vels_hat[0] - self.k_field[0] * self.one_k2_field * div_u),
            -Cs * (vels_hat[1] - self.k_field[1] * self.one_k2_field * div_u),
            -Cs * (vels_hat[2] - self.k_field[2] * self.one_k2_field * div_u),
        ]

        return jnp.stack([jnp.fft.irfftn(s_hat[ii], axes=(2,1,0)) for ii in range(3)])

