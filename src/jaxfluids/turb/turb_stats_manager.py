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

from typing import Dict, Tuple

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager

class TurbStatsManager:
    """ Provides functionality to calculate statistics of turbulent flows.
    The TurbStatsManager provides turbulent statistics of the initial flow 
    field as well as cumulative statistics over the course of a simulation.
    """
    # TODO Include functionality for postprocessing.
    
    def __init__(self, domain_information: DomainInformation, material_manager: MaterialManager = None) -> None:

        self.domain_information = domain_information
        self.material_manager   = material_manager

        self.eps = 1e-8

        self.Nx, self.Ny, self.Nz = domain_information.number_of_cells
        assert (self.Nx == self.Ny and self.Ny == self.Nz), "The present implementation only works for cubic domains."

        self.initialize()

    def initialize(self) -> None:
        self.k_field, self.k_vec = self._get_real_wavenumber_grid(self.Nx)
        self.k2_field       = self.k_field[0]*self.k_field[0] + self.k_field[1]*self.k_field[1] + self.k_field[2]*self.k_field[2]
        self.one_k2_field   = 1.0 / (self.k2_field + self.eps)
        self.k_mag_vec      = jnp.sqrt( self.k_vec**2 )
        self.k_mag_field    = jnp.sqrt( self.k2_field )
        self.shell          = (self.k_mag_field + 0.5).astype(int)
        self.fact           = 2 * (self.k_field[0] > 0) * (self.k_field[0] < self.Nx//2) + 1 * (self.k_field[0] == 0) + 1 * (self.k_field[0] == self.Nx//2)

    def _get_real_wavenumber_grid(self, N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initializes wavenumber grid and wavenumber vector.

        :param N: Resolution.
        :type N: int
        :return: Wavenumber grid and wavenumber vector.
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        Nf = N//2 + 1
        k = jnp.fft.fftfreq(N, 1./N) # for y and z direction
        kx = k[:Nf]
        kx = kx.at[-1].mul(-1)
        k_field = jnp.array(jnp.meshgrid(kx, k, k, indexing="ij"), dtype=int)
        return k_field, k 

    def energy_spectrum_spectral(self, velocity_hat: jnp.ndarray) -> jnp.ndarray:
        """Calculates the three-dimensional spectral energy spectrum of the input velocity.

        :param velocity_hat: Velocity vector in spectral space.
        :type velocity_hat: jnp.ndarray
        :return: Spectral energy spectrum.
        :rtype: jnp.ndarray
        """

        ek          = jnp.zeros(self.Nx)
        n_samples    = jnp.zeros(self.Nx) 

        uu = self.fact * 0.5 * (jnp.abs(velocity_hat[0]*velocity_hat[0]) + jnp.abs(velocity_hat[1]*velocity_hat[1]) + jnp.abs(velocity_hat[2]*velocity_hat[2]))

        ek        = ek.at[self.shell.flatten()].add(uu.flatten())
        n_samples = n_samples.at[self.shell.flatten()].add(self.fact.flatten()) 

        ek = ek * 4 * jnp.pi * self.k_vec**2 / (n_samples + self.eps) / (self.Nx**3) / (self.Nx**3) 
        return ek

    def energy_spectrum_physical(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Calculates the three-dimensional spectral energy spectrum of the input velocity.
        Wrapper around self.energy_spectrum_spectral

        :param velocity: Velocity vector in physical space.
        :type velocity: jnp.ndarray
        :return: Spectral energy spectrum.
        :rtype: jnp.ndarray
        """

        velocity_hat = jnp.stack([jnp.fft.rfftn(velocity[ii], axes=(2,1,0)) for ii in range(3)])
        return self.energy_spectrum_spectral(velocity_hat)

    def get_turbulent_statistics(self, primes: jnp.ndarray) -> Dict:
        """Computes the turbulent statistics for the given primitive buffer.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :return: Dictionary with turbulent statistics.
        :rtype: Dict
        """
        # TODO Include other setups besides HIT
        return self.hit_statistics(primes)

    def hit_statistics(self, primes: jnp.ndarray) -> Dict:
        """Calculates statistics for homogeneous isotropic turbulence.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :return: Dictionary with information on the HIT statistics.
        :rtype: Dict
        """

        rho = primes[0]
        velocity = primes[1:4]
        pressure = primes[4]
        velocity_hat = jnp.stack([jnp.fft.rfftn(velocity[ii], axes=(2,1,0)) for ii in range(3)])

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        
        temperature = self.material_manager.get_temperature(p=pressure, rho=rho)
        
        rho_mean = jnp.mean(rho)
        rho_rms  = jnp.std(rho)
        
        p_mean = jnp.mean(pressure)
        p_rms  = jnp.std(pressure)

        T_mean = jnp.mean(temperature) 
        T_rms  = jnp.std(temperature) 

        speed_of_sound        = self.material_manager.get_speed_of_sound(p=pressure, rho=rho)
        speed_of_sound_mean_1 = self.material_manager.get_speed_of_sound(p=p_mean, rho=rho_mean)
        speed_of_sound_mean_2 = jnp.mean(speed_of_sound)

        mu      = self.material_manager.get_dynamic_viscosity(temperature) 
        mu_mean = jnp.mean(mu)
        nu      = mu / rho
        nu_mean = jnp.mean(nu)

        u_rms = u_prime = jnp.sqrt( jnp.mean( velocity[0]**2 + velocity[1]**2 + velocity[2]**2 ) /3)
        q_rms = jnp.sqrt( jnp.mean( velocity[0]**2 + velocity[1]**2 + velocity[2]**2 ) )

        Ma_t   = jnp.sqrt( jnp.mean( velocity[0]**2 + velocity[1]**2 + velocity[2]**2 ) ) / speed_of_sound_mean_1
        Ma_rms = jnp.sqrt( jnp.mean( (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) / speed_of_sound ) )

        TKE   = 0.5 * jnp.mean(rho * (velocity[0]**2 + velocity[1]**2 + velocity[2]**2))

        duidj      = self.calculate_sheartensor_spectral(velocity_hat)
        S_ij       = 0.5 * ( duidj + jnp.transpose(duidj, axes=(1,0,2,3,4)) )
        SijSij_bar = jnp.mean(jnp.sum(S_ij**2, axis=(0,1)))
        eps        = 2 * mu_mean / rho_mean * SijSij_bar 

        vorticity       = self.calculate_vorticity_spectral(velocity_hat)
        vorticity_rms   = jnp.sqrt(jnp.mean(vorticity[0]**2 + vorticity[1]**2 + vorticity[2]**2)/3)
        enstrophy_mean  = jnp.mean(vorticity[0]**2 + vorticity[1]**2 + vorticity[2]**2)

        dilatation     = self.calculate_dilatation_spectral(velocity_hat)
        dilatation_rms = jnp.std(dilatation)
        divergence_rms = jnp.sqrt(jnp.mean( duidj[0,0]**2 + duidj[1,1]**2 + duidj[2,2]**2 ))

        ek = self.energy_spectrum_spectral(velocity_hat)

        # LENGTH SCALES 
        # Integral length scale
        int_Ek        = jnp.trapz(ek)
        int_Ek_over_k = jnp.trapz(ek / (self.k_vec + 1e-10))
        L_I           = 0.75 * jnp.pi * int_Ek_over_k / int_Ek

        # Longitudinal and lateral Taylor length scale
        lambda_  = jnp.sqrt( u_rms**2 / jnp.mean(duidj[0,0]**2) )
        lambda_g = jnp.sqrt(2 * u_rms**2 / jnp.mean(duidj[0,0]**2) )
        lambda_f = jnp.sqrt(2 * u_rms**2 / jnp.mean(duidj[0,1]**2) )

        # Kolmogorov
        eta      = (nu_mean**3 / eps)**0.25
        kmax     = jnp.sqrt(2) * Nx / 3
        eta_kmax = eta * kmax

        # TIME SCALES
        tau_LI     = L_I / u_rms            # Large-eddy-turnover time
        tau_lambda = lambda_f / u_rms       # Eddy turnover time

        # REYNOLDS NUMBER
        Re_turb     = rho_mean * q_rms**4 / eps / mu_mean
        Re_lambda   = lambda_f * u_rms * rho_mean / mu_mean
        Re_lambda_2 = lambda_ * u_rms * rho_mean / mu_mean
        Re_int      = u_rms * L_I * rho_mean / mu_mean

        turb_stats_dict = {
            "FLOW STATE": {
                "P MEAN": p_mean, "p_rms": p_rms,
                "RHO MEAN": rho_mean, "rho_rms": rho_rms,
                "T MEAN": T_mean, "T_rms": T_rms,
                "C MEAN": speed_of_sound_mean_1,
                "MU MEAN": mu_mean, 
                "Ma T": Ma_t, "Ma_rms": Ma_rms,
                "U RMS": u_rms,
                "TKE": TKE,
                "ENSTROPHY MEAN": enstrophy_mean,
                "DILATATION RMS": dilatation_rms, 
                "DIVERGENCE RMS": divergence_rms,
            },

            "REYNOLDS NUMBERS": {
                "RE LAMBDA": Re_lambda, 
                "RE LAMBDA2": Re_lambda_2,
                "RE INTEGRAL": Re_int,
                "RE TURBULENT": Re_turb,
            },

            "LENGTH SCALES": {
                "INTEGRAL": L_I, 
                "TAYLOR MICRO": lambda_, 
                "LONGITUDINAL TAYLOR": lambda_g, 
                "LATERAL TAYLOR": lambda_f,
                "KOLMOGOROV": eta, 
                "KOLMOGOROV RESOLUTION": eta_kmax,
            },

            "TIME SCALES": {
                "LARGE EDDY TURNOVER": tau_LI,
                "EDDY TURNOVER": tau_lambda,
            },
        }

        return turb_stats_dict

    def calculate_vorticity_spectral(self, velocity_hat: jnp.ndarray) -> jnp.ndarray:
        """Calculates the vortiticy of the input velocity field. Calculation done in 
        spectral space.

        omega = (du3/dx2 - du2/dx3, du1/dx3 - du3/dx1, du2/dx1 - du1/dx2)

        :param velocity_hat: Buffer of velocities in spectral space.
        :type velocity_hat: jnp.ndarray
        :return: Vorticity vector in physical space.
        :rtype: jnp.ndarray
        """

        omega_0 = jnp.fft.irfftn(1j * (self.k_field[1] * velocity_hat[2] - self.k_field[2] * velocity_hat[1]), axes=(2,1,0))
        omega_1 = jnp.fft.irfftn(1j * (self.k_field[2] * velocity_hat[0] - self.k_field[0] * velocity_hat[2]), axes=(2,1,0))
        omega_2 = jnp.fft.irfftn(1j * (self.k_field[0] * velocity_hat[1] - self.k_field[1] * velocity_hat[0]), axes=(2,1,0))
        omega   = jnp.stack([omega_0, omega_1, omega_2])
        return omega 

    def calculate_vorticity(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Calculates the vortiticy of the input velocity field. Calculation done in 
        spectral space.

        omega = [   du3/dx2 - du2/dx3
                    du1/dx3 - du3/dx1
                    du2/dx1 - du1/dx2]

        :param velocity: Buffer of velocities in physical space.
        :type velocity: jnp.ndarray
        :return: Vorticity vector in physical space.
        :rtype: jnp.ndarray
        """

        velocity_hat = jnp.stack([jnp.fft.rfftn(velocity[ii], axes=(2,1,0)) for ii in range(3)])
        return self.calculate_vorticity_spectral(velocity_hat)

    def calculate_sheartensor_spectral(self, velocity_hat: jnp.ndarray) -> jnp.ndarray:
        """Calculates the shear tensor in spectral space.

        dui/dxj = IFFT ( 1j * k_j * u_i_hat  )

        :param velocity_hat: Buffer of velocities in spectral space.
        :type velocity_hat: jnp.ndarray
        :return: Buffer of the shear tensor.
        :rtype: jnp.ndarray
        """

        duidj = [[], [], []]
        for ii in range(3):
            for jj in range(3):
                duidj[ii].append(jnp.fft.irfftn(1j * self.k_field[jj] * velocity_hat[ii], axes=(2,1,0)))
        return jnp.array(duidj)

    def calculate_sheartensor(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Calculates the shear tensor in spectral space. Wrapper around 
        self.calculate_sheartensor_spectral().

        duidj = [
            du1/dx1 du1/dx2 du1/dx3
            du2/dx1 du2/dx2 du2/dx3
            du3/dx1 du3/dx2 du3/dx3
        ]

        :param velocity: Buffer of velocities in physical space.
        :type velocity: jnp.ndarray
        :return: Buffer of the shear tensor.
        :rtype: jnp.ndarray
        """
    
        velocity_hat = jnp.stack([jnp.fft.rfftn(velocity[ii], axes=(2,1,0)) for ii in range(3)])
        return self.calculate_sheartensor_spectral(velocity_hat)

    def calculate_dilatation_spectral(self, velocity_hat: jnp.ndarray) -> jnp.ndarray:
        """Calculates the dilatation of the given velocity field in spectral space.

        :param velocity_hat: Buffer of velocities in spectral space.
        :type velocity_hat: jnp.ndarray
        :return: Buffer of the dilatational field.
        :rtype: jnp.ndarray
        """

        dilatation_spectral = 1j * (self.k_field[0] * velocity_hat[0] + self.k_field[1] * velocity_hat[1] + self.k_field[2] * velocity_hat[2])
        dilatation_real     = jnp.fft.irfftn(dilatation_spectral, axes=(2,1,0))
        return dilatation_real

    def calculate_dilatation(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """_summary_

        Calculates dilatation in spectral space 

        velocity: (3, Nx, Ny, Nz) array

        dilatation: (Nx, Ny, Nz) array
        dilatation = du1/dx1 + du2/dx2 + du3/dx3

        :param velocity: Buffer of velocities in physical space.
        :type velocity: jnp.ndarray
        :return: Buffer of dilatational field.
        :rtype: jnp.ndarray
        """
        
        velocity_hat = jnp.stack([jnp.fft.rfftn(velocity[ii], axes=(2,1,0)) for ii in range(3)])
        return self.calculate_dilatation_spectral(velocity_hat)

    def calculate_strain(duidj: jnp.ndarray) -> jnp.ndarray:
        """Calculates the strain given the velocity gradient tensor.

        :param duidj: Buffer of velocity gradient.
        :type duidj: jnp.ndarray
        :return: Buffer of strain tensor.
        :rtype: jnp.ndarray
        """

        S_ij = 0.5 * ( duidj + jnp.transpose(duidj, axes=(1,0,2,3,4)) )
        return S_ij