from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.turbulence.statistics.utilities import (
    energy_spectrum_physical, energy_spectrum_spectral,
    energy_spectrum_spectral_real_parallel)
from jaxfluids.math.fft import rfft3D_np, irfft3D_np, parallel_fft, parallel_ifft, \
    rfft3D, irfft3D, parallel_rfft, parallel_irfft, wavenumber_grid_parallel, \
    real_wavenumber_grid_parallel, real_wavenumber_grid_np, \
    real_wavenumber_grid
from jaxfluids.config import precision
from jaxfluids.data_types.case_setup.initial_conditions import HITParameters
from jaxfluids.domain.helper_functions import reassemble_buffer, split_buffer

Array = jax.Array

def initialize_hit(
        mesh_grid: Tuple[Array],
        split_factors: Tuple[int],
        gamma: float,
        R: float, 
        parameters: HITParameters
    ) -> Array:
    """Turbulent initial conditions for compresible HIT.
    
    Ristorcelli & Blaisdell 1997
    Incompressible/compressible random initial velocity field with prescribed energy spectrum

    Currently three variants are implemented which can be selected via the ic_type argument:
    IC1) Solenoidal velocity field with uniform pressure and density
    IC2) Solenoidal velocity field with pressure obtained from Poisson equation and uniform density
    IC3) Compressible velocity field with pressure and density obtained from Poisson equations
    IC4) 

    # TODO should get material manager

    :param mesh_grid: Device mesh grid
    :type mesh_grid: Tuple[Array]
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param gamma: _description_
    :type gamma: float
    :param R: _description_
    :type R: float
    :param parameters: _description_
    :type parameters: HITParameters
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Array
    """

    ic_type = parameters.ic_type
    rho_ref = parameters.rho_ref
    T_ref = parameters.T_ref
    ma_target = parameters.ma_target

    # Ideally R = 1 / gamma / Ma**2
    p_ref = rho_ref * R * T_ref
    c_ref = np.sqrt(gamma * p_ref / rho_ref)

    velocity = compute_solenoidal_velocity_field(
        mesh_grid, split_factors,
        gamma, R, parameters)

    if ic_type == "IC1":
        # Zero density and pressure fluctuations, solenoidal velocity
        pressure = p_ref * jnp.ones_like(velocity[0])
        density = rho_ref * jnp.ones_like(velocity[0])

    elif ic_type == "IC2":
        #Zero density fluctuations, fluctuating pressure field obtained
        # from Poisson equation, solenoidal velocity
        _, pressure, _ = get_dilatational_field(
            velocity=velocity, Ma=ma_target, T_ref=T_ref, 
            rho_ref=rho_ref, p_ref=p_ref, gamma=gamma, R=R,
            split_factors=split_factors)
        density = rho_ref * jnp.ones_like(velocity[0])

    elif ic_type == "IC3":
        # Pressure and density fluctuations according to Ristorcelli,
        # velocity has dilatational component
        velocity, pressure, density = get_dilatational_field(
            velocity=velocity, Ma=ma_target, T_ref=T_ref, 
            rho_ref=rho_ref, p_ref=p_ref, gamma=gamma, R=R,
            split_factors=split_factors)

    elif ic_type == "IC4":
        # Pressure from Poisson equation, density via isentropy relation,
        # solenoidal velocity field
        _, pressure, _ = get_dilatational_field(
            velocity=velocity, Ma=ma_target, T_ref=T_ref, 
            rho_ref=rho_ref, p_ref=p_ref, gamma=gamma, R=R,
            split_factors=split_factors)
        density = rho_ref * (pressure/p_ref)**(1/gamma)
    
    else:
        raise NotImplementedError

    primitives_init = jnp.concatenate([
        jnp.expand_dims(density, axis=0),
        velocity,
        jnp.expand_dims(pressure, axis=0)
    ], axis=0)

    return primitives_init


def compute_solenoidal_velocity_field(
        mesh_grid: Tuple[Array],
        split_factors: Tuple[int],
        gamma: float,
        R: float, 
        parameters: HITParameters
    ) -> Array:

    split_axis = np.argmax(np.array(split_factors))
    split = split_factors[split_axis]
    device_resolution = mesh_grid[0].shape
    Nx,Ny,Nz = device_resolution
    N = int(device_resolution[split_axis]*split)
    is_parallel = True if np.prod(np.array(split_factors)) > 1 else False

    is_velocity_spectral = parameters.is_velocity_spectral
    energy_spectrum = parameters.energy_spectrum
    T_ref = parameters.T_ref
    rho_ref = parameters.rho_ref
    xi_0 = parameters.xi_0
    xi_1 = parameters.xi_1
    ma_target = parameters.ma_target

    if is_velocity_spectral:
        assert_str = ("For velocity initialization in spectral space, "
            "choose exponential energy spectrum.")
        assert energy_spectrum == "EXPONENTIAL", assert_str

    # Ideally R = 1 / gamma / Ma**2
    p_ref = rho_ref * R * T_ref
    c_ref = np.sqrt(gamma * p_ref / rho_ref)

    ek_fun = get_target_spectrum(energy_spectrum)

    if is_velocity_spectral:
        velocity = solenoidal_velocity_init_spectral_space(
            N=N, gamma=gamma, R=R, T_ref=T_ref,
            rho_ref=rho_ref, ek_fun=ek_fun,
            ma_target=ma_target, xi_0=xi_0, xi_1=xi_1,
            split_factors=split_factors)

    else:
        # Get 1-dimensional energy spectrum
        k_vec = jnp.arange(N)
        ek_target = ek_fun(k_vec, xi_0=xi_0, xi_1=xi_1, u_rms=1.0,)

        # Generate random velocity field
        if is_parallel:
            device_id = jax.lax.axis_index(axis_name="i")
            key = jax.random.PRNGKey(device_id)
            velocity = 2 * jnp.pi * jax.random.uniform(key, shape=(3,Nx,Ny,Nz))
        else:
            velocity = 2 * np.pi * np.random.uniform(size=(3,Nx,Ny,Nz))           

        for _ in range(3):
            # Rescale fluctuations according to target spectrum
            velocity = rescale_field(velocity, ek_target, split_factors)
            # div u = 0 by Helmholtz projection
            velocity = get_solenoidal_field(velocity, split_factors)

        # Rescale to match Mach number
        q_rms = jnp.mean(jnp.sum(velocity * velocity, axis=0))
        if is_parallel:
            q_rms = jax.lax.pmean(q_rms, axis_name="i")
        q_rms = jnp.sqrt(q_rms)
        Ma_t  = q_rms / c_ref
        # u_rms = q_rms / jnp.sqrt(3)
        # Ma_t  = u_rms / c_ref
        velocity *= ma_target / Ma_t
    
    return velocity

def get_target_spectrum(energy_spectrum: str) -> Callable:
    """Returns the user-specified energy spectrum function which is 
    later used for initialization of a turbulent velocity field. 
    Currently implemented are the following energy spectra:

    1) Kolmogorov-5/3 law
    2) Exponential energy spectrum
    3) Box spectrum

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
    
    if energy_spectrum.upper() == "KOLMOGOROV":
        # ek[1:nx//2] = 0.5 * kx[1:nx//2]**(-5.0/3.0)

        def ek_fun(k: Array, xi_0: int, xi_1: int, 
                   **kwargs) -> Array:
            k = k + jnp.where(k == 0, precision.get_eps(), 0)
            ek = (k >= xi_0) * (k < xi_1) * (0.5 * k**(-5/3))
            return ek
    
    elif energy_spectrum.upper() == "EXPONENTIAL":
        # A = 0.013
        # ek[1:] = A * kx[1:]**4 * np.exp(-2.0 * kx[1:]**2 / xi_0**2)

        def ek_fun(k: Array, xi_0: int, u_rms: float, 
                   **kwargs) -> Array:
            A = u_rms**2 * 16 * jnp.sqrt(2 / jnp.pi)
            ek = A * k**4 / xi_0**5 * jnp.exp(- 2 * k**2 / xi_0**2)
            return ek

    elif energy_spectrum.upper() == "BOX":
        # ek[1:xi_0+1] = 1

        def ek_fun(k: Array, xi_0:int, xi_1: int, 
                   **kwargs) -> Array:
            ek = (k >= xi_0) * (k < xi_1) * 1.0
            return ek

    else:
        raise NotImplementedError

    return ek_fun

def _turb_init_spyropoulos_v2(
        N: int, 
        energy_spectrum: str, 
        ma_target: float, 
        psi_target: float, 
        xi_0: int, 
        is_compressible: bool = True
    ) -> np.ndarray:
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

def apply_conjugate_symmetry_2D(A: np.ndarray) -> np.ndarray:
    """Applies conjugate symmetry to the last two
    axes of the input array. I.e.,
    A(k1,k2) = A*(-k1,-k2)

    :param A: Input array of shape (...,N,N)
    :type A: np.ndarray
    :return: Output array of shape (...,N,N)
    :rtype: np.ndarray
    """
    N1, N2 = A.shape[-2:]

    assert_string = "Only implemented for square matrices."
    assert N1 == N2, assert_string

    Nf_ = N1 // 2
    s_negative_k = np.s_[-Nf_+1:] # (-Nf_+1,...,-2,-1)
    s_positive_k = np.s_[1:Nf_]   # (1,2,...,Nf-1)
    A[...,s_negative_k,s_negative_k] = np.flip(np.conj(A[...,s_positive_k,s_positive_k]), axis=(-2,-1))
    A[...,s_positive_k,s_negative_k] = np.flip(np.conj(A[...,s_negative_k,s_positive_k]), axis=(-2,-1))
    A[...,0,s_negative_k] = np.flip(np.conj(A[...,0,s_positive_k]), axis=-1)
    A[...,s_negative_k,0] = np.flip(np.conj(A[...,s_positive_k,0]), axis=-1)
    return A

def apply_conjugate_symmetry_2D_jnp(A: Array) -> Array:
    N1, N2 = A.shape[-2:]

    assert_string = "Only implemented for square matrices."
    assert N1 == N2, assert_string

    Nf_ = N1 // 2
    s_negative_k = jnp.s_[-Nf_+1:] # (-Nf_+1,...,-2,-1)
    s_positive_k = jnp.s_[1:Nf_]   # (1,2,...,Nf-1)
    A = A.at[...,s_negative_k,s_negative_k].set(jnp.flip(jnp.conj(A[...,s_positive_k,s_positive_k]), axis=(-2,-1)))
    A = A.at[...,s_positive_k,s_negative_k].set(jnp.flip(jnp.conj(A[...,s_negative_k,s_positive_k]), axis=(-2,-1)))
    A = A.at[...,0,s_negative_k].set(jnp.flip(jnp.conj(A[...,0,s_positive_k]), axis=-1))
    A = A.at[...,s_negative_k,0].set(jnp.flip(jnp.conj(A[...,s_positive_k,0]), axis=-1))
    return A

def solenoidal_velocity_init_spectral_space(
        N: int, 
        gamma: float,
        R: float, 
        T_ref: float, 
        rho_ref: float,
        ek_fun: Callable, 
        ma_target: float, 
        xi_0: int,
        xi_1: int,
        split_factors: Tuple[int]
    ) -> np.ndarray:
    """Initial condition for compressible HIT with solenoidal velocity field
    and uniform temperature and density. Initialization in spectral space,
    see Johnsen et al. (2010) - Assessment of high-resolution methods for 
    numerical simulations of compressible turbulence with shock waves
    for details.

    :param N: _description_
    :type N: int
    :param gamma: _description_
    :type gamma: float
    :param R: _description_
    :type R: float
    :param T_ref: _description_
    :type T_ref: float
    :param rho_ref: _description_
    :type rho_ref: float
    :param energy_spectrum: _description_
    :type energy_spectrum: str
    :param ma_target: _description_
    :type ma_target: float
    :param xi_0: _description_
    :type xi_0: int
    :return: _description_
    :rtype: np.ndarray
    """
    
    # TODO combine parallel and serial

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False
    
    p_ref = rho_ref * R * T_ref 
    c_ref = np.sqrt(gamma * R * T_ref)
    u_rms = ma_target / np.sqrt(3) * c_ref

    if not is_parallel:
        k_field = real_wavenumber_grid_np(N)
        k_mag = np.sqrt(np.sum(k_field * k_field, axis=0))
        k12_mag = np.sqrt(k_field[0]*k_field[0] + k_field[1]*k_field[1])
        k_mag[0,0,0] = precision.get_eps() # Dummy value to prevent division by zero

        # k_mag_int = (k_mag + 0.5).astype(int)
        # Ek = (k_mag_int >= 1) * (k_mag_int <= 80)
        # Ek = u_rms**2 * 16 * np.sqrt(2 / np.pi) * k_mag_int**4 / xi_0**5 * np.exp(- 2 * k_mag_int**2 / xi_0**2)
        # amplitude = np.sqrt(2 * Ek / (4 * np.pi * k_mag_int * k_mag_int))

        # Ek = u_rms**2 * 16 * np.sqrt(2 / np.pi) * k_mag**4 / xi_0**5 * np.exp(- 2 * k_mag**2 / xi_0**2)    
        Ek = ek_fun(k_mag, u_rms=u_rms, xi_0=xi_0, xi_1=xi_1)
        amplitude = np.sqrt(2 * Ek / (4 * np.pi * k_mag * k_mag))
        phi = 2 * np.pi * np.random.uniform(size=k_field.shape)
        a = amplitude * np.exp(1j * phi[0]) * np.cos(phi[2])
        b = amplitude * np.exp(1j * phi[1]) * np.sin(phi[2])

        one_k_mag = 1.0 / k_mag
        k1_over_k12mag = k_field[0] / (k12_mag + 1e-100)
        k2_over_k12mag = k_field[1] / (k12_mag + 1e-100)
        #k1_over_k12mag[0,0,:] = 0.0    
        k2_over_k12mag[0,0,:] = 1.0

        velocity_hat = np.array([
            k2_over_k12mag * a + k1_over_k12mag * k_field[2] * one_k_mag * b,
            k2_over_k12mag * k_field[2] * one_k_mag * b - k1_over_k12mag * a,
            -k12_mag * one_k_mag * b,
        ], dtype=np.complex128)
        velocity_hat[:,0,0,0] = 0.0
        velocity_hat[:,N//2,:,:] = 0.0
        velocity_hat[:,:,N//2,:] = 0.0
        velocity_hat[:,:,:,-1] = 0.0

        # Apply conjugate symmetry for k3=0, i.e., u_hat(k1,k2,0) = u_hat*(-k1,-k2,0),
        # for k3>0 this is inherently given by rfft/irfft
        velocity_hat[...,0] = apply_conjugate_symmetry_2D(velocity_hat[...,0])

        # Correct scaling of FFT, only if norm="backward" is used
        velocity_hat *= N**3    
        velocity = irfft3D_np(velocity_hat)

    else:
        number_of_cells = [N,N,N]

        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1]),-1)[split_axis_in]
        split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])

        k_field = real_wavenumber_grid_parallel(number_of_cells, split_factors_out)
        k_mag = jnp.sqrt(jnp.sum(k_field * k_field, axis=0))
        k_mag = k_mag.at[0,0,0].set(precision.get_eps())    # Dummy value to prevent division by zero
        k12_mag = jnp.sqrt(k_field[0]*k_field[0] + k_field[1]*k_field[1])

        Ek = ek_fun(k_mag, u_rms=u_rms, xi_0=xi_0, xi_1=xi_1)
        amplitude = jnp.sqrt(2 * Ek / (4 * jnp.pi * k_mag * k_mag))
        phi = 2 * jnp.pi * jnp.array(np.random.uniform(size=k_field.shape))
        a = amplitude * jnp.exp(1j * phi[0]) * jnp.cos(phi[2])
        b = amplitude * jnp.exp(1j * phi[1]) * jnp.sin(phi[2])

        one_k_mag = 1.0 / k_mag
        k1_over_k12mag = k_field[0] / (k12_mag + 1e-100)
        k2_over_k12mag = k_field[1] / (k12_mag + 1e-100)
        #k1_over_k12mag[0,0,:] = 0.0    
        k2_over_k12mag = k2_over_k12mag.at[0,0,:].set(1.0)

        velocity_hat = jnp.array([
            k2_over_k12mag * a + k1_over_k12mag * k_field[2] * one_k_mag * b,
            k2_over_k12mag * k_field[2] * one_k_mag * b - k1_over_k12mag * a,
            -k12_mag * one_k_mag * b,
        ], dtype=jnp.complex128)
        velocity_hat = velocity_hat.at[:,0,0,0].set(0.0)
        velocity_hat = velocity_hat.at[:,N//2,:,:].set(0.0)
        velocity_hat = velocity_hat.at[:,:,N//2,:].set(0.0)
        velocity_hat = velocity_hat.at[:,:,:,-1].set(0.0)

        # TODO this section is very memory intensive;
        # we need a dedicated apply_conjugate_symmetry method
        # for parallel settings
        velocity_hat = jax.lax.all_gather(velocity_hat, axis_name="i")
        velocity_hat = reassemble_buffer(velocity_hat, split_factors_out)

        # Apply conjugate symmetry for k3=0, i.e., u_hat(k1,k2,0) = u_hat*(-k1,-k2,0),
        # for k3>0 this is inherently given by rfft/irfft
        velocity_hat = velocity_hat.at[...,0].set(apply_conjugate_symmetry_2D_jnp(velocity_hat[...,0]))

        velocity_hat = split_buffer(velocity_hat, split_factors_out)
        device_id = jax.lax.axis_index(axis_name="i")
        velocity_hat = velocity_hat[device_id]

        # Correct scaling of FFT, only if norm="backward" is used
        velocity_hat *= N**3    
        velocity = parallel_irfft(velocity_hat, split_factors_out, split_axis_in)

    return velocity

def get_solenoidal_field(velocity: np.ndarray, split_factors: Tuple[int]) -> np.ndarray:
    """Performs a Helmholtz decomposition of the given velocity,
    i.e., calculates a solenoidal velocity field.
    
    Note that the input velocity field has to be sufficiently smooth,
    and that the domain has to be [0, 2pi] x [0, 2pi] x [0, 2pi].

    :param velocity: Input velocity field.
    :type velocity: np.ndarray
    :return: Projected, solenoidal velocity field.
    :rtype: np.ndarray
    """

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False
    eps = precision.get_eps()

    if is_parallel:

        number_of_cells_device = velocity.shape[-3:]
        number_of_cells = [int(number_of_cells_device[i]*split_factors[i]) for i in range(3)]

        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1]),-1)[split_axis_in]
        split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])

        nx, ny, nz = number_of_cells
        assert (nx == ny) & (ny == nz)

        k_field = real_wavenumber_grid_parallel(number_of_cells, split_factors_out)
        one_k2_field = 1.0 / (jnp.sum(k_field * k_field, axis=0) + eps)

        velocity_hat = parallel_rfft(velocity, split_factors, split_axis_out)
        div = jnp.sum(k_field * velocity_hat, axis=0) 
        velocity_sol_hat = velocity_hat - k_field * one_k2_field * div
        velocity_sol = parallel_irfft(velocity_sol_hat, split_factors_out, split_axis_in)
            
    else:

        nx, ny, nz = velocity.shape[-3:]
        assert (nx == ny) & (ny == nz)

        k_field = real_wavenumber_grid(nx)
        one_k2_field = 1.0 / (jnp.sum(k_field * k_field, axis=0) + eps)

        velocity_hat = rfft3D(velocity)
        div = jnp.sum(k_field * velocity_hat, axis=0)
        velocity_sol_hat = velocity_hat - k_field * one_k2_field * div
        velocity_sol = irfft3D(velocity_sol_hat)        
        
    return velocity_sol

def get_dilatational_field(
        velocity: np.array, 
        Ma: float, 
        T_ref: float, 
        rho_ref: float,
        p_ref: float, 
        gamma: float,
        R: float,
        split_factors: Tuple[int]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes dilatational part of the velocity field and consistent pressure
    and density fluctuations. Pressure and density fluctuations are introduced 
    by solving two Poisson equations in spectral space.
    Routine based on Ristorecelli & Blaisdell 1997.
    
    :param velocity: Solenoidal input velocity field.
    :type velocity: np.array
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

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False
    eps = precision.get_eps()

    if is_parallel:
        number_of_cells_device = velocity.shape[-3:]
        Nx_in, Ny_in, Nz_in = number_of_cells_device
        number_of_cells = [int(number_of_cells_device[i]*split_factors[i]) for i in range(3)]

        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
        split_factors_out = tuple([split_factors[split_axis_in]
                                   if i == split_axis_out else 1 for i in range(3)])
        
        nx,ny,nz = number_of_cells
    
    else:
        nx,ny,nz = velocity.shape[-3:]

    assert_str = "Routine only implemented for homogeneous isotropic grid."
    assert (nx == ny) & (ny == nz), assert_str

    
    if is_parallel:
        k_field = real_wavenumber_grid_parallel(number_of_cells, split_factors_out)
    else:
        N = nx
        k_field = real_wavenumber_grid(N)
    # Modified wavenumber
    # k_field = 1 / (2 * jnp.pi / N) * (16/12 * jnp.sin(2 * jnp.pi * k_field / N) - 2/12 * jnp.sin(4 * jnp.pi * k_field / N))
    
    one_k2_field = 1.0 / (jnp.sum(k_field * k_field, axis=0) + eps)

    c_ref = jnp.sqrt(gamma * p_ref / rho_ref)
    q_rms = jnp.mean(jnp.sum(velocity * velocity, axis=0))
    if is_parallel:
        q_rms = jax.lax.pmean(q_rms, axis_name="i")
    q_rms = jnp.sqrt(q_rms)
    u_rms = q_rms / jnp.sqrt(3)
    u_ref = u_rms
    # u_ref = q_rms

    Ma = u_ref / c_ref
    eps_squared = gamma * Ma**2

    # NOTE this may unintentionally rescale the velocity field
    # in IC2 settings
    # velocity /= u_ref


    # NOTE Solve Poisson equation for p1
    # - k^2 p_hat = k_i k_j (u_i u_j)_hat
    p1_hat = 0.0
    for ii in range(3):
        for jj in range(3):
            if is_parallel:
                p1_hat += k_field[ii] * k_field[jj] * parallel_rfft(
                    velocity[ii] * velocity[jj],
                    split_factors, split_axis_out)
            else:
                p1_hat += k_field[ii] * k_field[jj] * rfft3D(
                    velocity[ii] * velocity[jj])
    # NOTE last term due to dimensional calculation
    p1_hat *= -one_k2_field * (rho_ref / eps_squared)
    if is_parallel:
        p1 = parallel_irfft(p1_hat, split_factors_out, split_axis_in)
    else:
        p1 = irfft3D(p1_hat)


    # NOTE Solve Poisson equation for dp1dt
    # 1) Calculate f = (vk vi),k + p1,i
    # NOTE (rho_ref / eps_squared) is scaling due to dimensional calculation
    if is_parallel:
        f = jnp.zeros((3,Nx_in,Ny_in,Nz_in))
        for ii in range(3):
            f_ii = 1j * k_field[ii] * parallel_rfft(p1, split_factors, split_axis_out)
            for jj in range(3):
                f_ii += 1j * (rho_ref / eps_squared) * k_field[jj] \
                    * parallel_rfft(velocity[jj] * velocity[ii],
                                    split_factors, split_axis_out)
            f = f.at[ii].set(parallel_irfft(f_ii, split_factors_out, split_axis_in))
    else:
        f = jnp.zeros((3,N,N,N))
        for ii in range(3):
            f_ii = 1j * k_field[ii] * rfft3D(p1)
            for jj in range(3):
                f_ii += 1j * (rho_ref / eps_squared) * k_field[jj] \
                    * rfft3D(velocity[jj] * velocity[ii])
            f = f.at[ii].set(irfft3D(f_ii))

    # 2) Solve Poisson equation and FFT back >> dp1/dt
    # k^2 dpdt_hat = 2 k_i k_j (f_i u_j)_hat
    dp1dt_hat = 0.0
    for ii in range(3):
        for jj in range(3):
            if is_parallel:
                dp1dt_hat += 2 * k_field[ii] * k_field[jj] * parallel_rfft(
                    f[ii] * velocity[jj],
                    split_factors, split_axis_out)
            else:
                dp1dt_hat += 2 * k_field[ii] * k_field[jj] * rfft3D(
                    f[ii] * velocity[jj])
    dp1dt_hat *= one_k2_field

    # NOTE Solve for dilatation: d = -Mt**2 (dp/dt + v_k dp/dk)
    # Dimensional form: -gamma * p_ref * d = p1_t + v_k * p1_k
    udp1_hat = 0.0
    for ii in range(3):
        if is_parallel:
            udp1_hat += 1j * k_field[ii] * parallel_rfft(
                velocity[ii] * p1, split_factors, split_axis_out)        
        else:
            udp1_hat += 1j * k_field[ii] * rfft3D(velocity[ii] * p1)

    # d_hat = -Ma**2 * (dp1dt_hat + udp1_hat)
    d_hat = -1.0 / (gamma * p_ref) * (dp1dt_hat + udp1_hat)
        
    # Construct dilatational velocity field
    velocity_dil_hat = -1j * k_field * one_k2_field * d_hat
    if is_parallel:
        velocity_dil = parallel_irfft(velocity_dil_hat, split_factors_out, split_axis_in)
    else:
        velocity_dil = irfft3D(velocity_dil_hat)

    # Calculate pressure and density field
    press = p_ref + eps_squared * p1
    dens = rho_ref + eps_squared * (p1 / c_ref**2)    # rho_1 = p_1 / gamma
    # Calculate solenoidal + dilatational field
    U_final = velocity + eps_squared * velocity_dil

    # # Calculate pressure and density field
    # press = p_ref + p1
    # dens  = rho_ref + Ma**2 * p1
    # # Calculate solenoidal + dilatational field
    # U_final = velocity + w

    return U_final, press, dens

def rescale_field(
        velocity: jnp.array,
        ek_target: jnp.array,
        split_factors: Tuple[int],
    ) -> Array:
    """Rescales the input velocity field such that it matches 
    the specified target energy spectrum. Rescaling is done in
    spectral space.

    :param velocity: Input velocity field which has to be rescaled.
    :type velocity: np.array
    :param ek_target: Target energy spectrum.
    :type ek_target: np.array
    :return: Rescaled velocity field.
    :rtype: np.ndarray
    """

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False
    eps = precision.get_eps()

    if is_parallel:
        number_of_cells_device = velocity.shape[-3:]
        number_of_cells = [int(number_of_cells_device[i]*split_factors[i]) for i in range(3)]

        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1]),-1)[split_axis_in]
        split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])

        N = number_of_cells[0]
        k_field = real_wavenumber_grid_parallel(number_of_cells, split_factors_out)
        kmag2_field = jnp.sum(jnp.square(k_field), axis=0)
        shell = (jnp.sqrt(kmag2_field + eps) + 0.5).astype(int)

        velocity_hat = parallel_rfft(velocity, split_factors, split_axis_out)
        ek_current = energy_spectrum_spectral_real_parallel(velocity_hat, split_factors_out, multiplicative_factor=0.5)
        scale = jnp.sqrt(ek_target / (ek_current + eps))
        velocity_hat *= scale[shell] * N**3
        velocity_scaled = parallel_irfft(velocity_hat, split_factors_out, split_axis_in)

    else:

        number_of_cells = velocity.shape[-3:]
        N = number_of_cells[0]
        k_field = real_wavenumber_grid(N)
        kmag2_field = jnp.sum(jnp.square(k_field), axis=0)
        shell = (jnp.sqrt(kmag2_field + eps) + 0.5).astype(int)

        velocity_hat = rfft3D(velocity)
        ek_current = energy_spectrum_spectral(velocity_hat, number_of_cells, multiplicative_factor=0.5)
        scale = jnp.sqrt(ek_target / (ek_current + eps))
        velocity_hat *= scale[shell] * N**3
        velocity_scaled = irfft3D(velocity_hat)

    return velocity_scaled