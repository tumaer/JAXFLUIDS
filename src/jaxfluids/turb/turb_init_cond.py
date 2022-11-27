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

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from numpy.fft import rfftn, irfftn

from jaxfluids.input_reader import InputReader
from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager

class TurbInitManager:
    """ The TurbInitManager implements functionality for the initialization
    of turbulent flow fields. The main function of the TurbInitManager is 
    the get_turbulent_initial_condition method which returns a randomly 
    initialized turbulent flow field according to the user-specified initial
    conditions. Currently there are four different options available:
    
    1) HIT flow field according to Ristorcelli
    2) Taylor-Green vortex
    3) Turbulent channel flow (under construction)
    """

    def __init__(self, input_reader: InputReader, domain_information: DomainInformation, material_manager: MaterialManager) -> None:
        self.input_reader       = input_reader
        self.domain_information = domain_information
        self.material_manager   = material_manager

        self.turb_init_params = input_reader.turb_init_params 
        
        self.N = domain_information.number_of_cells[0]

        self.turbulent_random_seed = self.input_reader.turb_init_params["seed"] if "seed" in self.input_reader.turb_init_params.keys() else 0
        np.random.seed(self.turbulent_random_seed)

    def get_turbulent_initial_condition(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:     
        """Calculates turbulent primitive variables.
        
        Initialization is based on the turbulent case
        specified in the self.turb_init_params dictionary.

        :return: Primitive variables: density, velocity vector, pressure
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        mesh = self.domain_information.cell_centers
        coordinates = [coord for coord in mesh if len(coord) > 1]
        mesh_grid = jnp.meshgrid(*coordinates, indexing="ij")

        if self.turb_init_params["turb_case"] == "RISTORCELLI":
            density, velocityX, velocityY, velocityZ, pressure = turb_init_ristorcelli(
                N               = self.N, 
                gamma           = self.material_manager.gamma,
                T_ref           = self.turb_init_params["T_ref"],
                rho_ref         = self.turb_init_params["rho_ref"],
                energy_spectrum = self.turb_init_params["energy_spectrum"],
                ma_target       = self.turb_init_params["ma_target"],
                xi_0            = self.turb_init_params["xi_0"],
                ic_type         = self.turb_init_params["ic_type"])
        elif self.turb_init_params["turb_case"] == "TGV":
            density, velocityX, velocityY, velocityZ, pressure = turb_init_TGV(
                X               = mesh_grid, 
                gamma           = self.material_manager.gamma,
                Ma              = self.turb_init_params["Ma"], 
                rho_ref         = self.turb_init_params["rho_ref"],
                V_ref           = self.turb_init_params["V_ref"],
                L               = self.turb_init_params["L_ref"])
        elif self.turb_init_params["turb_case"] == "CHANNEL":
            density, velocityX, velocityY, velocityZ, pressure = turb_init_channel()

        return density, velocityX, velocityY, velocityZ, pressure

def get_target_spectrum(energy_spectrum: str, nx: int, xi_0: int = None) -> np.ndarray:
    """Returns the user-specified energy spectrum which is 
    later used for initialization of a turbulent velocity field.

    :param energy_spectrum: String identifier for the energy spectrum.
    :type energy_spectrum: str
    :param nx: Spatial resolution.
    :type nx: int
    :param xi_0: Wavenumber of energy spectrum peak, defaults to None
    :type xi_0: int, optional
    :raises NotImplementedError: Raises error if specified energy spectrum
        is not implemented.
    :return: Array with energy in spectral space. 
    :rtype: np.ndarray
    """
    
    # Energy spectrum and wavenumber vector
    ek = np.zeros((nx,))
    kx = np.arange(nx) 

    if   energy_spectrum.upper() == "ISOTROPIC_FORCED":
        ek[1:nx//2] = 0.5 * kx[1:nx//2]**(-5.0/3.0)
    
    elif energy_spectrum.upper() == "EXPONENTIAL":
        A = 0.013
        ek[1:] = A * kx[1:]**4 * np.exp(-2.0 * kx[1:]**2 / xi_0**2)

    elif energy_spectrum.upper() == "BOX":
        ek[1:xi_0+1] = 1

    else:
        raise NotImplementedError

    return ek

def turb_init_TGV(X: np.ndarray, gamma: float, Ma: float, rho_ref: float, V_ref: float, L: float) -> np.ndarray:
    """Implements initial conditions for compressible Taylor-Green vortex (TGV).

    :param X: Buffer of cell center coordinats.
    :type X: np.ndarray
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param Ma: Mach number of the flow.
    :type Ma: float
    :param rho_ref: Reference density scale.
    :type rho_ref: float
    :param V_ref: Reference velocity scale..
    :type V_ref: float
    :param L: Reference length scale.
    :type L: float
    :return: Buffer with TGV initial conditions in terms of primitive variables.
    :rtype: np.ndarray
    """
    
    density   =  rho_ref * np.ones(X[0].shape)
    velocityX =  V_ref * np.sin(X[0] / L) * np.cos(X[1] / L) * np.cos(X[2] / L)
    velocityY = -V_ref * np.cos(X[0] / L) * np.sin(X[1] / L) * np.cos(X[2] / L)
    velocityZ =  np.zeros(X[0].shape)
    pressure  =  rho_ref * V_ref**2 * (1 / gamma / Ma**2 + 1/16.0 * ((np.cos(2 * X[0] / L) + np.cos(2 * X[1] / L)) * (np.cos(2 * X[2] / L) + 2)))
    return density, velocityX, velocityY, velocityZ, pressure

def _turb_init_spyropoulos_v2(N: int, energy_spectrum: str, ma_target: float, psi_target: float, xi_0: int, is_compressible: bool = True) -> np.ndarray:
    '''
    Incompressible/compressible random initial velocity field with prescribed energy spectrum
    '''
    # Reference values 
    gamma = 1.4
    T_ref = rho_ref = 1
    p_ref = rho_ref * T_ref / gamma / ma_target**2
    c_ref = np.sqrt(gamma * p_ref / rho_ref)
    print("Reference Values: p_ref = %3.2e, rho_ref = %3.2e, T_ref = %3.2e, c_ref = %3.2e" %(p_ref, rho_ref, T_ref, c_ref))

    # Get target energy spectrum
    ek_target = get_target_spectrum(energy_spectrum, N, xi_0)

    # Generate random velocity field
    U1 = 2 * np.pi * np.random.uniform(size=(3,N,N,N))  # For solenoidal field
    U2 = 2 * np.pi * np.random.uniform(size=(3,N,N,N))  # For dilatational field

    for itr in range(3):
        print("Iteration:", itr, "\n")
        # Rescale fluctuations according to target spectrum
        U1 = rescale_field(U1, ek_target)
        U2 = rescale_field(U2, ek_target)
        # div u = 0 by Helmholtz projection
        U1 = get_solenoidal_field(U1)
        U2 = U2 - get_solenoidal_field(U2)

    # Adjust ratio of compressible kinetic energy
    U1_rms = np.sqrt(np.mean(U1[0]**2 + U1[1]**2 + U1[2]**2))
    U2_rms = np.sqrt(np.mean(U2[0]**2 + U2[1]**2 + U2[2]**2))
    U2 = np.sqrt( psi_target / (1 - psi_target) ) * U1_rms / U2_rms * U2
    U = U1 + U2
    
    U1_rms = np.sqrt(np.mean(U1[0]**2 + U1[1]**2 + U1[2]**2))
    U2_rms = np.sqrt(np.mean(U2[0]**2 + U2[1]**2 + U2[2]**2))
    U_rms = np.sqrt(np.mean(U[0]**2 + U[1]**2 + U[2]**2))
    print("q_s = %4.3e, q_d = %4.3e, psi = (q_d/q)^2 = %4.3e" %(U1_rms, U2_rms, (U2_rms/U_rms)**2) )
    
    # Adjust level of fluctuations to match turbulent Mach number
    Ma_t = np.sqrt(np.mean(U[0]**2 + U[1]**2 + U[2]**2)) / c_ref
    print("MA_T_PRE =", Ma_t)
    U *= ma_target / Ma_t
    Ma_t = np.sqrt(np.mean(U[0]**2 + U[1]**2 + U[2]**2)) / c_ref
    print("Ma_T_INIT = ", Ma_t, "\n")

    if is_compressible:
        # TODO
        U, p, rho = get_dilatational_field(U, ma_target, T_ref, rho_ref, gamma)
        print("STATS OF COMPRESSIBLE FIELD:")
    else:
        # Uniform pressure and density
        p   = p_ref * np.ones_like(U[0])
        rho = rho_ref * np.ones_like(U[0])

    return rho, U[0], U[1], U[2], p

def turb_init_ristorcelli(N: int, gamma: float, T_ref: float, rho_ref: float, energy_spectrum: str, ma_target: float, xi_0: 
        int, ic_type: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """    Ristorcelli & Blaisdell 1997
    Incompressible/compressible random initial velocity field with prescribed energy spectrum

    Currently three variants are implemented which can be selected via the ic_type argument:
    IC1) Solenoidal velocity field with uniform pressure and density
    IC2) Solenoidal velocity field with pressure obtained from Poisson equation and uniform density
    IC3) Compressible velocity field with pressure and density obtained from Poisson equations

    :param N: Spatial resolution.
    :type N: int
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param T_ref: Temperature reference. 
    :type T_ref: float
    :param rho_ref: Density reference.
    :type rho_ref: float
    :param energy_spectrum: Target energy spectrum.
    :type energy_spectrum: str
    :param ma_target: Target Mach number.
    :type ma_target: float
    :param xi_0: Wavenumber at which energy spectrum is maximal.
    :type xi_0: int
    :param ic_type: String identifier which determines the type of 
        thermodynamic fluctuations. 
    :type ic_type: int
    :return: Initial density, velocity, and pressure fields. 
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    
    p_ref = rho_ref * T_ref / gamma / ma_target**2
    c_ref = np.sqrt(gamma * p_ref / rho_ref)

    # Get target energy spectrum
    ek_target = get_target_spectrum(energy_spectrum, N, xi_0)

    # Generate random velocity field
    velocity = 2 * np.pi * np.random.uniform(size=(3,N,N,N))

    for itr in range(3):
        # Rescale fluctuations according to target spectrum
        velocity = rescale_field(velocity, ek_target)
        # div u = 0 by Helmholtz projection
        velocity = get_solenoidal_field(velocity)

    # Rescale to match Mach number
    q_rms = np.sqrt(np.mean(velocity[0]**2 + velocity[1]**2 + velocity[2]**2))
    Ma_t  = q_rms / c_ref
    velocity *= ma_target / Ma_t

    if ic_type == "IC1":
        # Zero density and pressure fluctuations, solenoidal velocity
        pressure   = p_ref * np.ones_like(velocity[0])
        density    = rho_ref * np.ones_like(velocity[0])
    elif ic_type == "IC2":
        # Zero density fluctuations, fluctuating pressure field obtained from Poisson euqation, solenoidal velocity
        _, pressure, _ = get_dilatational_field(velocity, ma_target, T_ref, rho_ref, gamma)
        pressure       = np.abs(pressure)
        density        = rho_ref * np.ones_like(velocity[0])
    elif ic_type == "IC3":
        # Pressure and density fluctuations, velocity has dilatational component
        velocity, pressure, density = get_dilatational_field(velocity, ma_target, T_ref, rho_ref, gamma)
        pressure = np.abs(pressure)
    
    return density, velocity[0], velocity[1], velocity[2], pressure

def turb_init_channel():
    """Implements initial conditions for a turbulent channel flow.
    # TODO
    """
    raise NotImplementedError

def get_solenoidal_field(U: np.ndarray) -> np.ndarray:
    """Performs a Helmholtz decomposition of the given velocity,
    i.e., calculates a solenoidal velocity field.
    
    Note that the input velocity field has to be sufficiently smooth,
    and that the domain has to be [0, 2pi] x [0, 2pi] x [0, 2pi].

    :param U: Input velocity field.
    :type U: np.ndarray
    :return: Projected, solenoidal velocity field.
    :rtype: np.ndarray
    """

    nx, ny, nz = U.shape[1:]
    assert nx == ny
    assert ny == nz

    N = nx
    Nf = N//2 + 1

    ky = kz = np.fft.fftfreq(N, 1./N)
    kx = ky[:Nf].copy()
    kx[-1] *= -1
    k_field = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=int)

    # Create buffers
    U_sol     = np.zeros((3,N,N,N))
    U_hat     = np.zeros((3,Nf,N,N), dtype=complex)
    U_sol_hat = np.zeros((3,Nf,N,N), dtype=complex)

    for ii in range(3):
        U_hat[ii] = np.fft.rfftn(U[ii], axes=(2,1,0))


    one_k2_field = 1.0 / (k_field[0]**2 + k_field[1]**2 + k_field[2]**2 + 1e-99)
    div = k_field[0] * U_hat[0] + k_field[1] * U_hat[1] + k_field[2] * U_hat[2] 

    for ii in range(3):
        U_sol_hat[ii] = U_hat[ii] - k_field[ii] * one_k2_field * div
        U_sol[ii]     = np.fft.irfftn(U_sol_hat[ii], axes=(2,1,0))

    print("Divergence in spectral space:", np.sum(np.abs(k_field[0] * U_sol_hat[0] + k_field[1] * U_sol_hat[1] + k_field[2] * U_sol_hat[2])))

    return U_sol

def get_dilatational_field(U_field: np.array, Ma: float, T_ref: float, rho_ref: float, 
    gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes dilatational part of the velocity field and consistent pressure
    and density fluctuations. Pressure and density fluctuations are introduced 
    by solving two Poisson equations in spectral space.
    Routine based on Ristorecelli & Blaisdell 1997.
    
    :param U_field: Solenoidal input velocity field.
    :type U_field: np.array
    :param Ma: Target Mach number.
    :type Ma: float
    :param T_ref: Reference temperature.
    :type T_ref: float
    :param rho_ref: Reference density.
    :type rho_ref: float
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :return: Dilatational velocity field, fluctuating density and pressure.
    :rtype: np.ndarray
    """

    nx, ny, nz = U_field.shape[1:]
    assert nx == ny
    assert ny == nz

    N = nx
    Nf = N//2 + 1

    ky = kz = np.fft.fftfreq(N, 1./N)
    kx = ky[:Nf].copy()
    kx[-1] *= -1
    k_field = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=int)
    # km_field = 1 / (2 * np.pi / N) * (16/12 * np.sin(2 * np.pi * k_field / N) - 2/12 * np.sin(4 * np.pi * k_field / N))
    one_k2_field = 1.0 / (k_field[0]**2 + k_field[1]**2 + k_field[2]**2 + 1e-99)

    # Create buffers
    f = np.zeros((3,N,N,N))
    w = np.zeros((3,N,N,N))

    ## Solve Poisson equation for p1
    p1_hat = 0
    for ii in range(3):
        for jj in range(3):
            p1_hat += k_field[ii] * k_field[jj] * rfftn(U_field[ii] * U_field[jj], axes=(2,1,0)) 
    p1_hat *= -one_k2_field
    p1 = irfftn(p1_hat, axes=(2,1,0))

    ## Calculate pressure and density field
    eps = np.sqrt(gamma * Ma**2)
    press = T_ref * rho_ref / gamma / Ma**2 + p1
    dens  = rho_ref + Ma**2 * p1

    ## Solve Poisson equation for dp1dt
    # 1) Calculate (vk vi),k + p1,i
    for ii in range(3):
        f[ii] = irfftn(
               1j * k_field[0]  * rfftn(U_field[0] * U_field[ii], axes=(2,1,0)) \
            +  1j * k_field[1]  * rfftn(U_field[1] * U_field[ii], axes=(2,1,0)) \
            +  1j * k_field[2]  * rfftn(U_field[2] * U_field[ii], axes=(2,1,0)) \
            +  1j * k_field[ii] * rfftn(p1, axes=(2,1,0)),
            axes=(2,1,0)
            )
    # 2) Solve Poisson equation and FFT back >> dp1/dt
    dp1dt_hat = 0.0

    for ii in range(3):
        for jj in range(3):
            dp1dt_hat -= 2 * k_field[ii] * k_field[jj] * rfftn(f[ii] * U_field[jj], axes=(2,1,0))
    dp1dt_hat *= -one_k2_field
    dp1dt = irfftn(dp1dt_hat, axes=(2,1,0))


    ## Solve for dilatation: d = -Mt**2 (dp/dt + v_k dp/dk)
    udp1_hat = 0
    for ii in range(3):
        udp1_hat += 1j * k_field[ii] * rfftn(U_field[ii] * p1, axes=(2,1,0))

    d_hat = -Ma**2 * (dp1dt_hat + udp1_hat)
    # d_hat *= -1 / gamma
        
    # Construct dilatational velocity field
    for ii in range(3):
        w[ii] = irfftn(-1j * k_field[ii] * one_k2_field * d_hat, axes=(2,1,0))

    # Calculate solenoidal + dilatational field
    U_final = U_field + w

    return U_final, press, dens

def rescale_field(U_field: np.array, ek_target: np.array) -> np.ndarray:
    """Rescales the input velocity field such that it matches 
    the specified target energy spectrum. Rescaling is done in
    spectral space.

    :param U_field: Input velocity field which has to be rescaled.
    :type U_field: np.array
    :param ek_target: Target energy spectrum.
    :type ek_target: np.array
    :return: Rescaled velocity field.
    :rtype: np.ndarray
    """

    ndof, Nx, Ny, Nz = U_field.shape
    Mx = Nx // 2 + 1
    My = Ny // 2 + 1
    Mz = Nz // 2 + 1

    U_hat = np.zeros((3,Mx,Ny,Nz), dtype=np.complex128)
    U_scaled = np.zeros((3,Nx,Ny,Nz))

    # Factor to account for fft normalization
    fftfactor = 1 / Nx / Ny / Nz

    for ii in range(3):
        U_hat[ii] = np.fft.rfftn(U_field[ii], axes=(2,1,0))
        U_hat[ii,-1,:,:] = 0
        U_hat[ii,:,My,:] = 0
        U_hat[ii,:,:,Mz] = 0
    U_hat *= fftfactor

    # START ENERGY SPECTRUM
    ek = np.zeros(Nx)
    nsamples = np.zeros(Nx)

    k = np.arange(Nx)
    kx = np.fft.fftfreq(Nx, 1./Nx).astype(int)
    kx = kx[:Mx]; kx[-1] *= -1
    ky = np.fft.fftfreq(Ny, 1./Ny).astype(int)
    kz = np.fft.fftfreq(Nz, 1./Nz).astype(int)

    K = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=int)
    Kmag = np.sqrt( K[0]**2 + K[1]**2 + K[2]**2 )
    Shell = ( Kmag + 0.5 ).astype(int) 

    Fact = 2 * (K[0] > 0) * (K[0] < Nx//2) + 1 * (K[0]==0) + 1 * (K[0] == Nx//2)
    Fact[-1,:,:] = 0
    Fact[:,My,:] = 0
    Fact[:,:,Mz] = 0

    UU = Fact * 0.5 * ( np.abs(U_hat[0])**2 + np.abs(U_hat[1])**2 + np.abs(U_hat[2])**2 )
    np.add.at(ek, Shell.flatten(), UU.flatten())
    # np.add.at(nsamples, Shell.flatten(), 1)
    np.add.at(nsamples, Shell.flatten(), Fact.flatten())
    ek *= 4 * np.pi * k**2 / (nsamples + 1e-10)

    # END ENERGY SPECTRUM

    #sum_ek  = np.sum(ek)
    #sum_ek0 = np.sum(ek_target)
    # print("Energy is:", sum_ek, ", Energy target:", sum_ek0)

    scale = np.sqrt(ek_target / (ek + 1e-10))

    U_hat *= scale[Shell] * 1/fftfactor

    # Transform back to real space
    for ii in range(3):
        U_scaled[ii] = np.fft.irfftn(U_hat[ii], axes=(2,1,0))

    return U_scaled