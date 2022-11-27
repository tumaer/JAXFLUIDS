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

import jax.numpy as jnp
import numpy as np

################################################################################

# BASICS FOR FFT/IFFT

def is_power_of_two(a):
    return np.log2(a).is_integer()

def velocity_rfftn(velocity: np.array) -> np.array:
    """
    RFFTN 
    velocity : (3, Nx, Ny, Nz)
    velocity_hat: (3, Nfx, Ny, Nz)
    """
    _, Nx, Ny, Nz = velocity.shape
    assert (is_power_of_two(Nx) and is_power_of_two(Ny) and is_power_of_two(Nz))
    
    Nxf = Nx // 2 + 1
    velocity_hat = np.zeros((3, Nxf, Ny, Nz), dtype=np.complex128)
    for ii in range(3):
        velocity_hat[ii] = np.fft.rfftn(velocity[ii], axes=(2,1,0))
    return velocity_hat

def get_real_wavenumber_grid(N):
    Nf = N//2 + 1
    k = np.fft.fftfreq(N, 1./N) # for y and z direction
    kx = k[:Nf].copy()
    kx[-1] *= -1
    k_field = np.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    return k_field, k 

################################################################################

# FLOW FIELD DATA

def calculate_vorticity_spectral(velocity_hat: np.array, k_field: np.array, dtype: np.dtype = np.float64) -> np.array:
    Nxf, Ny, Nz = velocity_hat.shape[1:]
    Nx = 2 * (Nxf - 1)

    omega = np.zeros((3, Nx, Ny, Nz), dtype=dtype)

    # omega = (du3/dx2 - du2/dx3, du1/dx3 - du3/dx1, du2/dx1 - du1/dx2)
    omega[0] = jnp.fft.irfftn(1j * (k_field[1] * velocity_hat[2] - k_field[2] * velocity_hat[1]), axes=(2,1,0))
    omega[1] = jnp.fft.irfftn(1j * (k_field[2] * velocity_hat[0] - k_field[0] * velocity_hat[2]), axes=(2,1,0))
    omega[2] = jnp.fft.irfftn(1j * (k_field[0] * velocity_hat[1] - k_field[1] * velocity_hat[0]), axes=(2,1,0))
    
    return omega 

def calculate_vorticity(velocity: np.array) -> np.array:
    """
    Calculates vorticity in spectral space
    velocity: (3, Nx, Ny, Nz)

    omega: (3, Nx, Ny, Nz)
    omega = [   du3/dx2 - du2/dx3
                du1/dx3 - du3/dx1
                du2/dx1 - du1/dx2]
    """
    Nx, Ny, Nz = velocity.shape[1:]

    k_field, _   = get_real_wavenumber_grid(N=Nx)
    velocity_hat = velocity_rfftn(velocity)

    return calculate_vorticity_spectral(velocity_hat, k_field)

def calculate_sheartensor_spectral(velocity_hat: np.array, k_field: np.array, dtype: np.dtype = np.float64) -> np.array:
    Nxf, Ny, Nz = velocity_hat.shape[1:]
    Nx = 2 * (Nxf - 1)

    # dui/dxj = IFFT ( 1j * k_j * u_i_hat  )
    duidj = np.zeros((3, 3, Nx, Ny, Nz), dtype=dtype)
    for ii in range(3):
        for jj in range(3):
            duidj[ii,jj] = np.fft.irfftn(1j * k_field[jj] * velocity_hat[ii], axes=(2,1,0))
    
    return duidj

def calculate_sheartensor(velocity: np.array) -> np.array:
    """
    Calculates shear tensor in spectral space 

    velocity: (3, Nx, Ny, Nz) array

    duidj: (3, 3, Nx, Ny, Nz) array
    duidj = [
        du1/dx1 du1/dx2 du1/dx3
        du2/dx1 du2/dx2 du2/dx3
        du3/dx1 du3/dx2 du3/dx3
    ]
    """
    Nx, Ny, Nz = velocity.shape[1:]
    
    k_field, _   = get_real_wavenumber_grid(N=Nx)
    velocity_hat = velocity_rfftn(velocity)

    return calculate_sheartensor_spectral(velocity_hat, k_field)

def calculate_dilatation_spectral(velocity_hat: np.array, k_field: np.array, dtype: np.dtype = np.float64) -> np.array:
    dilatation_spectral = 1j * (k_field[0] * velocity_hat[0] + k_field[1] * velocity_hat[1] + k_field[2] * velocity_hat[2])
    dilatation_real     = np.fft.irfftn(dilatation_spectral, axes=(2,1,0))
    return dilatation_real

def calculate_dilatation(velocity):
    """
    Calculates dilatation in spectral space 

    velocity: (3, Nx, Ny, Nz) array

    dilatation: (Nx, Ny, Nz) array
    dilatation = du1/dx1 + du2/dx2 + du3/dx3
    """
    Nx, Ny, Nz = velocity.shape[1:]
    
    k_field, _   = get_real_wavenumber_grid(N=Nx)
    velocity_hat = velocity_rfftn(velocity)

    return calculate_dilatation_spectral(velocity_hat, k_field)

def calculate_strain(duidj: np.array) -> np.array:
    """ duidj: (3,3,Nx,Ny,Nz) """
    S_ij = 0.5 * ( duidj + np.transpose(duidj, axes=(1,0,2,3,4)) )
    return S_ij
