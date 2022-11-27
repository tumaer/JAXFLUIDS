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

import jax
import jax.numpy as jnp

import numpy as np

from jaxfluids.turb.turb_utils import velocity_rfftn, get_real_wavenumber_grid, calculate_sheartensor, calculate_dilatation, calculate_vorticity
from jaxfluids.materials.material_manager import MaterialManager

def get_energy_spectrum(vel):
    Nx, Ny, Nz = vel.shape[1:]
    assert (Nx == Ny and Ny == Nz)
    
    N = Nx
    Nf = N//2 + 1

    k_field, k = get_real_wavenumber_grid(N=Nx)
    k_mag = np.sqrt( k_field[0]*k_field[0] + k_field[1]*k_field[1] + k_field[2]*k_field[2] )

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + 1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)

    # Fourier transform 
    vel_hat = jnp.stack([jnp.fft.rfftn(vel[ii], axes=(2,1,0)) for ii in range(3)])

    ek = np.zeros(N)
    n_samples = np.zeros(N)

    uu = fact * 0.5 * (jnp.abs(vel_hat[0]*vel_hat[0]) + jnp.abs(vel_hat[1]*vel_hat[1]) + jnp.abs(vel_hat[2]*vel_hat[2]))

    np.add.at(ek, shell.flatten(), uu.flatten())
    np.add.at(n_samples, shell.flatten(), 1)
    ek *= 4 * np.pi * k**2 / (n_samples + 1e-10)
    ek *= 1/(N**3)

    return ek

def get_energy_spectrum_np(vel):
    Nx, Ny, Nz = vel.shape[1:]
    assert (Nx == Ny and Ny == Nz)
    
    N = Nx
    Nf = N//2 + 1

    k_field, k = get_real_wavenumber_grid(N=Nx)
    k_mag = np.sqrt( k_field[0]*k_field[0] + k_field[1]*k_field[1] + k_field[2]*k_field[2] )

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + 1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)

    # Fourier transform 
    vel_hat = velocity_rfftn(vel)

    ek = np.zeros(N)
    n_samples = np.zeros(N)

    uu = fact * 0.5 * (np.abs(vel_hat[0]*vel_hat[0]) + jnp.abs(vel_hat[1]*vel_hat[1]) + jnp.abs(vel_hat[2]*vel_hat[2]))

    np.add.at(ek, shell.flatten(), uu.flatten())
    np.add.at(n_samples, shell.flatten(), fact.flatten())

    ek *= 4 * np.pi * k**2 / (n_samples + 1e-10)
    ek *= 1 / (N**3) / (N**3)

    energy_phys = 0.5 * np.sum(vel[0]**2 + vel[1]**2 + vel[2]**2) / N**3
    energy_spec = np.sum(uu) / N**3 / N**3
    energy_ek   = np.sum(ek)

    # print("ENERGY PHYS: %4.3e, ENERGY SPEC: %4.3e, ENERGY EK: %4.3e" %(energy_phys, energy_spec, energy_ek))
    return ek, k

def get_scalar_energy_spectrum(vel):
    Nx, Ny, Nz = vel.shape
    assert (Nx == Ny and Ny == Nz)
    
    N = Nx
    Nf = N//2 + 1

    k_field, k = get_real_wavenumber_grid(N=Nx)
    k_mag = np.sqrt( k_field[0]*k_field[0] + k_field[1]*k_field[1] + k_field[2]*k_field[2] )

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + 1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)

    # Fourier transform 
    vel_hat = jnp.fft.rfftn(vel, axes=(2,1,0))

    ek = np.zeros(N)
    n_samples = np.zeros(N)

    uu = fact * 0.5 * jnp.abs(vel_hat * vel_hat)

    np.add.at(ek, shell.flatten(), uu.flatten())
    np.add.at(n_samples, shell.flatten(), fact)
    ek *= 4 * np.pi * k**2 / (n_samples + 1e-10)
    ek *= 1 / (N**3) / (N**3)

    return ek

def get_stats_hit(rho: jnp.array, velocityX: jnp.array, velocityY: jnp.array, velocityZ: jnp.array, pressure: jnp.array, material_manager: MaterialManager):
    Nx, Ny, Nz = rho.shape
    
    U = np.stack([velocityX, velocityY, velocityZ])
    
    temperature = material_manager.get_temperature(p=pressure, rho=rho)
    
    rho_mean = np.mean(rho)
    rho_rms = np.std(rho)
    
    p_mean = np.mean(pressure)
    p_rms  = np.std(pressure)

    T_mean = np.mean(temperature) 
    T_rms  = np.std(temperature) 

    speed_of_sound        = material_manager.get_speed_of_sound(p=pressure, rho=rho)
    speed_of_sound_mean_1 = material_manager.get_speed_of_sound(p=p_mean, rho=rho_mean)
    speed_of_sound_mean_2 = np.mean(speed_of_sound)

    mu      = material_manager.get_dynamic_viscosity(temperature) 
    mu_mean = np.mean(mu)
    nu      = mu / rho
    nu_mean = np.mean(nu)

    u_rms = u_prime = np.sqrt( np.mean( velocityX**2 + velocityY**2 + velocityZ**2 ) /3)
    q_rms = np.sqrt( np.mean( velocityX**2 + velocityY**2 + velocityZ**2 ) )

    Ma_t   = np.sqrt( np.mean( velocityX**2 + velocityY**2 + velocityZ**2 ) ) / speed_of_sound_mean_1
    Ma_rms = np.sqrt( np.mean( (velocityX**2 + velocityY**2 + velocityZ**2) / speed_of_sound ) )

    tke   = 0.5 * np.mean(rho * (velocityX**2 + velocityY**2 + velocityZ**2))

    duidj = calculate_sheartensor(U)
    S_ij  = 0.5 * ( duidj + np.transpose(duidj, axes=(1,0,2,3,4)) )
    SijSij_bar = np.mean(np.sum(S_ij**2, axis=(0,1)))
    eps = 2 * mu_mean / rho_mean * SijSij_bar 

    vorticity       = calculate_vorticity(U)
    vorticity_rms   = np.sqrt(np.mean(vorticity[0]**2 + vorticity[1]**2 + vorticity[2]**2)/3)
    enstrophy_mean  = np.mean(vorticity[0]**2 + vorticity[1]**2 + vorticity[2]**2)

    dil = calculate_dilatation(U)
    dilatation_rms = np.std(dil)
    divergence_rms = np.sqrt(np.mean( duidj[0,0]**2 + duidj[1,1]**2 + duidj[2,2]**2 ))

    ek, k = get_energy_spectrum_np(U)

    # LENGTH SCALES 
    # Integral length scale
    int_Ek = np.trapz(ek)
    int_Ek_over_k = np.trapz(ek / (k + 1e-10))
    L_I = 0.75 * np.pi * int_Ek_over_k / int_Ek

    # Longitudinal and lateral Taylor length scale
    lambda_  = np.sqrt( u_rms**2 / np.mean(duidj[0,0]**2) )
    lambda_g = np.sqrt(2 * u_rms**2 / np.mean(duidj[0,0]**2) )
    lambda_f = np.sqrt(2 * u_rms**2 / np.mean(duidj[0,1]**2) )

    # Kolmogorov
    eta = (nu_mean**3 / eps)**0.25
    kmax = np.sqrt(2) * Nx / 3
    eta_kmax = eta * kmax

    # TIME SCALES
    tau_LI = L_I / u_rms                # Large-eddy-turnover time
    tau_lambda = lambda_f / u_rms       # Eddy turnover time

    # REYNOLDS NUMBER
    Re_turb     = rho_mean * q_rms**4 / eps / mu_mean
    Re_lambda   = lambda_f * u_rms * rho_mean / mu_mean
    Re_lambda_2 = u_rms * lambda_ * rho_mean / mu_mean
    Re_int      = u_rms * L_I * rho_mean / mu_mean

    turb_stats_dict = {
        "FlowState": {
            "p_mean": p_mean, "p_rms": p_rms,
            "rho_mean": rho_mean, "rho_rms": rho_rms,
            "T_mean": T_mean, "T_rms": T_rms,
            "c_mean": speed_of_sound_mean_1,
            "mu_mean": mu_mean, 
            "Ma_t": Ma_t, "Ma_rms": Ma_rms,
            "U_rms": u_rms,
            "TKE": tke,
            "Enstrophy_mean": enstrophy_mean,
            "Dilatation_rms": dilatation_rms, "Divergence_rms": divergence_rms,
        },

        "ReynoldsNumbers": {
            "Re_lambda": Re_lambda, 
            "Re_lambda_2": Re_lambda_2,
            "Re_int": Re_int,
            "Re_turb": Re_turb,
        },

        "LengthScales": {
            "Integral length": L_I, 
            "Taylor microscale": lambda_, 
            "Longitudinal TM": lambda_g, 
            "Lateral TM": lambda_f,
            "Kolmogorov": eta, 
            "Kolmogorov resolution": eta_kmax,
        },

        "TimeScales": {
            "Large-eddy-turnover time": tau_LI,
            "Eddy-turnover time": tau_lambda,
        },
    }

    return turb_stats_dict


def get_quick_stats_hit(rho, velocityX, velocityY, velocityZ, pressure, mu: float, gamma: float = 1.4):
    print("\n\n********************\nTURB STATS HIT")
    U = np.stack([velocityX, velocityY, velocityZ])

    rho_mean = np.mean(rho)
    rho_rms = np.std(rho)
    
    p_mean = np.mean(pressure)
    p_rms = np.std(pressure)

    # TODO make function of material?
    c = np.sqrt(gamma * pressure / rho)

    c_mean = np.sqrt(gamma * p_mean / rho_mean)
    print("p_mean = %4.3f, p_rms = %4.3f" %(p_mean, p_rms))
    print("rho_mean = %4.3f, rho_rms = %4.3f" %(rho_mean, rho_rms))
    print("c_mean = %4.3f, c_mean_2 = %4.3f" %(c_mean, np.mean(c)))

    Ma_t   = np.sqrt( np.mean( U[0]**2 + U[1]**2 + U[2]**2 ) ) / c_mean
    Ma_rms = np.sqrt( np.mean( (U[0]**2 + U[1]**2 + U[2]**2) / c ) )
    print("Ma_t = %4.3f, Ma_rms = %4.3f" %(Ma_t, Ma_rms))

    u_rms = u_prime = np.sqrt(np.mean(U[0]**2 + U[1]**2 + U[2]**2)/3)
    tke   = 0.5 * np.mean(rho * (U[0]**2 + U[1]**2 + U[2]**2))
    q_rms = np.sqrt(3) * u_rms

    print("REFERENCE VALUES: U_RMS=", u_rms)

    duidj = calculate_sheartensor(U)
    S_ij  = 0.5 * ( duidj + np.transpose(duidj, axes=(1,0,2,3,4)) )
    SijSij_bar = np.mean(np.sum(S_ij**2, axis=(0,1)))
    eps = 2 * mu / rho_mean * SijSij_bar 
    Re_turb   = rho_mean * q_rms**4 / eps / mu

    vorticity = calculate_vorticity(U)
    enstrophy = vorticity[0]**2 + vorticity[1]**2 + vorticity[2]**2
    enstrophy_mean = np.mean(enstrophy)
    print("AVERAGED ENSTROPHY:", enstrophy_mean)

    dil = calculate_dilatation(U)
    divergence_rms = np.sqrt(np.mean( duidj[0,0]**2 + duidj[1,1]**2 + duidj[2,2]**2 ))
    print("RMS DILATATION:", np.std(dil))
    print("RMS DIVERGENCE:", divergence_rms)

    ek, k = get_energy_spectrum_np(U)

    # Integral length scale
    int_Ek = np.trapz(ek)
    int_Ek_over_k = np.trapz(ek / (k + 1e-10))
    L_I = 0.75 * np.pi * int_Ek_over_k / int_Ek

    print("INTEGRAL LENGTH SCALE = %4.3f" %(L_I))

    # Longitudinal and lateral Taylor length scale
    lambda_  = np.sqrt( u_prime**2 / np.mean(duidj[0,0]**2) )
    lambda_g = np.sqrt(2 * u_prime**2 / np.mean(duidj[0,0]**2) )
    lambda_f = np.sqrt(2 * u_prime**2 / np.mean(duidj[0,1]**2) )
    print("Taylor microscale = %4.3e \nLongitudinal Taylor microscale = %4.3e \nLateral Taylor microscale = %4.3e" %(lambda_, lambda_g, lambda_f))
    print("tau_re = ", lambda_f / u_rms)

    # Kolmogorov
    eta = ((mu / rho_mean)**3 / eps)**0.25
    # kmax = np.sqrt(3) * Nf
    # eta_kmax = eta * kmax
    print("Kolmogorov = %4.3e" %(eta))

    # Large-eddy-turnover time
    tau = L_I / u_rms 
    print("LARGE-EDDY-TURNOVER TIME:", tau)

    Re_lambda   = lambda_f * u_rms * rho_mean / mu
    Re_lambda_2 = u_rms * lambda_ * rho_mean / mu
    Re_int      = u_rms * L_I * rho_mean / mu
    print("Re_lambda = %3.2f, Re_lambda_2 = %3.2f" %(Re_lambda, Re_lambda_2))
    print("Re_int = %3.2e" %(Re_int))
    print("RE_turb = ", Re_turb, "mu = ", mu)
    print("tau_re = ", lambda_f / u_prime)
    print("tau_eddy = ", np.sqrt(15 * mu / rho_mean / eps))
    print("TKE_0 = ", tke)


def calculate_sheartensor_OLD(U_field: np.array) -> np.array:
    """
    Calculate shear tensor in spectral space 

    U_field: (3, Nx, Ny, Nz) array

    duidj = [
        du1/dx1 du1/dx2 du1/dx3
        du2/dx1 du2/dx2 du2/dx3
        du3/dx1 du3/dx2 du3/dx3
    ]
    """
    Nx, Ny, Nz = U_field.shape[1:]
    assert (Nx == Ny and Ny == Nz)

    N = Nx
    Nf = N//2 + 1

    k_field, k = get_real_wavenumber_grid(N=Nx)

    U_hat = velocity_rfftn(U_field)

    # dui/dxj = IFFT ( 1j * k_j * u_i_hat  )
    duidj = np.zeros((3,3,N,N,N), dtype=float)
    for ii in range(3):
        for jj in range(3):
            duidj[ii,jj] = jnp.fft.irfftn(1j * k_field[jj] * U_hat[ii], axes=(2,1,0))

    return duidj

def get_energy_spectrum_np_OLD(vel):
    """
    SHOWS equality of energy in physical and spectral space
    and also via integration of energy spectrum
    """
    Nx, Ny, Nz = vel.shape[1:]
    assert (Nx == Ny and Ny == Nz)

    N = Nx
    Nf = N//2 + 1
    k = np.fft.fftfreq(N, 1./N) # for y and z direction
    kx = k[:Nf].copy()
    kx[-1] *= -1
    k_field = np.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    k2_field = k_field[0]*k_field[0] + k_field[1]*k_field[1] + k_field[2]*k_field[2]
    k_mag = np.sqrt(k2_field)
    # one_k2_field = 1.0 / (k2_field + 1e-10)

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + 1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)

    # Fourier transform 
    vel_hat = np.zeros((3,Nf,Ny,Nz), dtype=complex)
    for ii in range(3):
       vel_hat[ii] = np.fft.rfftn(vel[ii], axes=(2,1,0))

    ek = np.zeros(N)
    n_samples = np.zeros(N)

    uu = fact * 0.5 * (np.abs(vel_hat[0]*vel_hat[0]) + jnp.abs(vel_hat[1]*vel_hat[1]) + jnp.abs(vel_hat[2]*vel_hat[2]))

    np.add.at(ek, shell.flatten(), uu.flatten())
    np.add.at(n_samples, shell.flatten(), fact.flatten())
    print(n_samples)
    ek *= 4 * np.pi * k**2 / (n_samples + 1e-10)
    ek *= 1/(N**3)/(N**3)

    energy_phys = 0.5 * np.mean(vel[0]**2 + vel[1]**2 + vel[2]**2)
    vel_hat_2 = np.zeros((3,Nx,Ny,Nz), dtype=complex)
    for ii in range(3):
       vel_hat_2[ii] = np.fft.fftn(vel[ii])
    energy_spec = 0.5 * np.mean(vel_hat_2[0]*np.conj(vel_hat_2[0]) + vel_hat_2[1]*np.conj(vel_hat_2[1]) + vel_hat_2[2]*np.conj(vel_hat_2[2])) / N**3
    
    k_field_2 = np.array(np.meshgrid(k, k, k, indexing="ij"), dtype=int)
    k_mag_2 = np.sqrt( k_field_2[0]*k_field_2[0] + k_field_2[1]*k_field_2[1] + k_field_2[2]*k_field_2[2] )
    shell_2 = (k_mag_2 + 0.5).astype(int)

    ek_2 = np.zeros(N)
    n_samples_2 = np.zeros(N)
    uu_2 = 0.5 * (np.abs(vel_hat_2[0]*vel_hat_2[0]) + jnp.abs(vel_hat_2[1]*vel_hat_2[1]) + jnp.abs(vel_hat_2[2]*vel_hat_2[2]))

    np.add.at(ek_2, shell_2.flatten(), uu_2.flatten())
    np.add.at(n_samples_2, shell_2.flatten(), 1)
    ek_2 *= 4 * np.pi * k**2 / (n_samples_2 + 1e-10)
    ek_2 *= 1/(N**3)/(N**3)

    import matplotlib.pyplot as plt
    plt.plot(kx, ek[:Nf], marker="x", mfc="None")
    plt.plot(kx, ek_2[:Nf], marker="o", mfc="None")
    plt.plot(kx, 0.013 * kx**4 * np.exp(-2.0 * kx**2 / 4**2) )
    plt.show()

    energy_ek   = np.sum(ek)
    energy_ek_2   = np.trapz(ek_2)
    print("ENERGY PHYS:", energy_phys)
    print("ENERGY SPEC:", energy_spec)
    print("ENERGY EK:", energy_ek, energy_ek_2)

    return ek, k
