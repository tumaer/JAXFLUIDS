from typing import Callable, List, Tuple

import jax.numpy as jnp
from jax import Array
import jax
import numpy as np

from jaxfluids.math.parallel_fft import parallel_fft, parallel_ifft
from jaxfluids.turb.statistics.utilities import (wavenumber_grid_parallel, energy_spectrum_physical,
                                                 rfft3D_np, irfft3D_np)
from jaxfluids.config import precision
from jaxfluids.data_types.case_setup.initial_conditions import HITParameters

def turb_init_hit(
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

    :param mesh_grid: _description_
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

    velocity = compute_solenoidal_velocity_field(mesh_grid, split_factors,
                                                 gamma, R, parameters)

    if ic_type == "IC1":
        # Zero density and pressure fluctuations, solenoidal velocity
        pressure = p_ref * jnp.ones_like(velocity[0])
        density = rho_ref * jnp.ones_like(velocity[0])

    elif ic_type == "IC2":
        #Zero density fluctuations, fluctuating pressure field obtained
        # from Poisson equation, solenoidal velocity
        _, pressure, _ = get_dilatational_field(
            U_field=velocity, Ma=ma_target, T_ref=T_ref, 
            rho_ref=rho_ref, p_ref=p_ref, gamma=gamma, R=R,
            split_factors=split_factors)
        pressure = jnp.abs(pressure)
        density = rho_ref * jnp.ones_like(velocity[0])

    elif ic_type == "IC3":
        # Pressure and density fluctuations according to Ristorcelli,
        # velocity has dilatational component
        velocity, pressure, density = get_dilatational_field(
            U_field=velocity, Ma=ma_target, T_ref=T_ref, 
            rho_ref=rho_ref, p_ref=p_ref, gamma=gamma, R=R,
            split_factors=split_factors)
        pressure = jnp.abs(pressure)

    elif ic_type == "IC4":
        # Pressure from Poisson equation, density via isentropy relation,
        # solenoidal velocity field
        _, pressure, _ = get_dilatational_field(
            U_field=velocity, Ma=ma_target, T_ref=T_ref, 
            rho_ref=rho_ref, p_ref=p_ref, gamma=gamma, R=R,
            split_factors=split_factors)
        density = pressure**(1/gamma)
    
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
        if not is_parallel:
            velocity = 2 * np.pi * np.random.uniform(size=(3,Nx,Ny,Nz))
        else:
            device_id = jax.lax.axis_index(axis_name="i")
            key = jax.random.PRNGKey(device_id)
            velocity = 2 * jnp.pi * jax.random.uniform(key, shape=(3,Nx,Ny,Nz))

        for itr in range(3):
            print(itr)
            # Rescale fluctuations according to target spectrum
            velocity = rescale_field(velocity, ek_target, split_factors)
            # div u = 0 by Helmholtz projection
            velocity = get_solenoidal_field(velocity, split_factors)

        # Rescale to match Mach number
        q_rms = jnp.mean(jnp.sum(velocity * velocity, axis=0))
        if is_parallel:
            q_rms = jax.lax.pmean(q_rms, axis_name="i")
        q_rms = jnp.sqrt(q_rms)
        u_rms = q_rms / jnp.sqrt(3)
        Ma_t  = q_rms / c_ref
        # Ma_t  = u_rms / c_ref
        velocity *= ma_target / Ma_t
    
    return velocity

def get_target_spectrum(
        energy_spectrum: str,
    ) -> Callable:
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

        def ek_fun(k: Array, xi_0: int, xi_1: int, **kwargs) -> Array:
            k = k + jnp.where(k == 0, precision.get_eps(), 0)
            ek = (k >= xi_0) * (k < xi_1) * (0.5 * k**(-5/3))
            return ek
    
    elif energy_spectrum.upper() == "EXPONENTIAL":
        # A = 0.013
        # ek[1:] = A * kx[1:]**4 * np.exp(-2.0 * kx[1:]**2 / xi_0**2)

        def ek_fun(k: Array, xi_0: int, u_rms: float, **kwargs) -> Array:
            A = u_rms**2 * 16 * jnp.sqrt(2 / jnp.pi)
            ek = A * k**4 / xi_0**5 * jnp.exp(- 2 * k**2 / xi_0**2)
            return ek

    elif energy_spectrum.upper() == "BOX":
        # ek[1:xi_0+1] = 1

        def ek_fun(k: Array, xi_0:int, xi_1: int, **kwargs) -> Array:
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

def apply_conjugate_symmetry_2D(A):
    N = A.shape[0]
    Nf_ = N // 2
    A[-Nf_+1:,-Nf_+1:] = np.flip(np.conj(A[1:Nf_,1:Nf_]), axis=(0,1))
    A[1:Nf_,-Nf_+1:] = np.flip(np.conj(A[-Nf_+1:,1:Nf_]), axis=(0,1))
    A[0,-Nf_+1:] = np.flip(np.conj(A[0,1:Nf_]))
    A[-Nf_+1:,0] = np.flip(np.conj(A[1:Nf_,0]))
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

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False
    
    if not is_parallel:
            
        Nf = N//2 + 1
        ky = kz = np.fft.fftfreq(N, 1./N)
        kx = ky[:Nf].copy()
        kx[-1] *= -1

        k_field = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=int) # (3, Nf, N, N)
        k_mag = np.sqrt(np.sum(k_field * k_field, axis=0))
        k_12_mag = np.sqrt(k_field[0]*k_field[0] + k_field[1]*k_field[1])

        k_mag[0,0,0] = precision.get_eps() # Dummy value to prevent division by zero

        p_ref = rho_ref * R * T_ref 
        c_ref = np.sqrt(gamma * R * T_ref)
        u_rms = ma_target / np.sqrt(3) * c_ref

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
        k1_over_k12 = np.zeros_like(k_12_mag)
        k1_over_k12[:,1:,:] = k_field[0,:,1:,:] / k_12_mag[:,1:,:]
        k1_over_k12[1:,:,:] = k_field[0,1:,:,:] / k_12_mag[1:,:,:]
        #k1_over_k12[0,0,:] = 0.0    
        k2_over_k12 = np.zeros_like(k_12_mag)
        k2_over_k12[:,1:,:] = k_field[1,:,1:,:] / k_12_mag[:,1:,:]
        k2_over_k12[1:,:,:] = k_field[1,1:,:,:] / k_12_mag[1:,:,:]
        k2_over_k12[0,0,:] = 1.0

        velocity_hat = np.array([
            k2_over_k12 * a + k1_over_k12 * k_field[2] * one_k_mag * b,
            k2_over_k12 * k_field[2] * one_k_mag * b - k1_over_k12 * a,
            -k_12_mag * one_k_mag * b,
        ], dtype=np.complex128)
        velocity_hat[:,0,0,0] = 0.0
        velocity_hat[:,-1,:,:] = 0.0
        velocity_hat[:,:,N//2,:] = 0.0
        velocity_hat[:,:,:,N//2] = 0.0
        for ii in range(3):
            velocity_hat[ii,0] = apply_conjugate_symmetry_2D(velocity_hat[ii,0])

        # Correct scaling of FFT, only if norm="backward" is used
        # velocity_hat *= N**3    
        velocity = irfft3D_np(velocity_hat, axes=(-1,-2,-3), norm="forward")

    else:
        # TODO turbulent
        raise NotImplementedError

    return velocity

def get_solenoidal_field(
        U: np.ndarray,
        split_factors: Tuple[int]
        ) -> np.ndarray:
    """Performs a Helmholtz decomposition of the given velocity,
    i.e., calculates a solenoidal velocity field.
    
    Note that the input velocity field has to be sufficiently smooth,
    and that the domain has to be [0, 2pi] x [0, 2pi] x [0, 2pi].

    :param U: Input velocity field.
    :type U: np.ndarray
    :return: Projected, solenoidal velocity field.
    :rtype: np.ndarray
    """

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False

    if not is_parallel:
            
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

        one_k2_field = 1.0 / (k_field[0]**2 + k_field[1]**2 + k_field[2]**2 + precision.get_eps())
        div = k_field[0] * U_hat[0] + k_field[1] * U_hat[1] + k_field[2] * U_hat[2] 

        for ii in range(3):
            U_sol_hat[ii] = U_hat[ii] - k_field[ii] * one_k2_field * div
            U_sol[ii]     = np.fft.irfftn(U_sol_hat[ii], axes=(2,1,0))

        # print("Divergence in spectral space:", np.mean(np.abs(k_field[0] * U_sol_hat[0] + k_field[1] * U_sol_hat[1] + k_field[2] * U_sol_hat[2])))

    else:
        
        resolution_in = U.shape[-3:]
        number_of_cells = [int(resolution_in[i]*split_factors[i]) for i in range(3)]
        nx,ny,nz = number_of_cells
        assert nx == ny
        assert ny == nz

        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
        split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])
        k_field = wavenumber_grid_parallel(number_of_cells, split_factors_out)

        U_hat = parallel_fft(U, split_factors, split_axis_out)

        resolution_out = U_hat.shape[-3:]
        Nx_in,Ny_in,Nz_in = resolution_in
        Nx_out,Ny_out,Nz_out = resolution_out

        U_sol_hat = jnp.zeros((3,Nx_out,Ny_out,Nz_out), dtype=complex)
        U_sol = jnp.zeros((3,Nx_in,Ny_in,Nz_in))

        one_k2_field = 1.0 / (k_field[0]**2 + k_field[1]**2 + k_field[2]**2 + precision.get_eps())
        div = k_field[0] * U_hat[0] + k_field[1] * U_hat[1] + k_field[2] * U_hat[2] 

        for ii in range(3):
            U_sol_hat_ii = U_hat[ii] - k_field[ii] * one_k2_field * div
            U_sol_hat = U_sol_hat.at[ii].set(U_sol_hat_ii)
            U_sol_ii = parallel_ifft(U_sol_hat_ii, split_factors_out, split_axis_in)
            U_sol = U_sol.at[ii].set(U_sol_ii)

        # print("Divergence in spectral space:", jnp.mean(jnp.abs(k_field[0] * U_sol_hat[0] + k_field[1] * U_sol_hat[1] + k_field[2] * U_sol_hat[2])))

    return U_sol

def get_dilatational_field(
        U_field: np.array, 
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

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False

    if not is_parallel:
            
        nx, ny, nz = U_field.shape[1:]
        assert nx == ny
        assert ny == nz

        N = nx
        Nf = N//2 + 1

        ky = kz = np.fft.fftfreq(N, 1./N)
        kx = ky[:Nf].copy()
        kx[-1] *= -1
        k_field = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=int)
        # Modified wavenumber
        # k_field = 1 / (2 * np.pi / N) * (16/12 * np.sin(2 * np.pi * k_field / N) - 2/12 * np.sin(4 * np.pi * k_field / N))
        one_k2_field = 1.0 / (np.sum(k_field * k_field, axis=0) + precision.get_eps())
    
        c_ref = np.sqrt(gamma * p_ref / rho_ref)
        q_rms = np.sqrt(np.mean(np.sum(U_field * U_field, axis=0)))
        u_rms = q_rms / np.sqrt(3)
        u_ref = u_rms
        # u_ref = q_rms

        Ma = u_ref / c_ref
        eps_squared = gamma * Ma**2
        U_field /= u_ref

        ## Solve Poisson equation for p1
        # - k^2 p_hat = k_i k_j (u_i u_j)_hat
        p1_hat = 0
        for ii in range(3):
            for jj in range(3):
                p1_hat += k_field[ii] * k_field[jj] * rfft3D_np(U_field[ii] * U_field[jj]) 
        p1_hat *= -one_k2_field
        p1 = irfft3D_np(p1_hat)

        # Solve Poisson equation for dp1dt
        # 1) Calculate f = (vk vi),k + p1,i
        f = np.zeros((3,N,N,N))
        for ii in range(3):
            f[ii] = irfft3D_np(
                1j * k_field[0]  * rfft3D_np(U_field[0] * U_field[ii]) \
                +  1j * k_field[1]  * rfft3D_np(U_field[1] * U_field[ii]) \
                +  1j * k_field[2]  * rfft3D_np(U_field[2] * U_field[ii]) \
                +  1j * k_field[ii] * rfft3D_np(p1),)

        # 2) Solve Poisson equation and FFT back >> dp1/dt
        # - k^2 dpdt_hat = -2 k_i k_j (f_i u_j)_hat
        dp1dt_hat = 0.0
        for ii in range(3):
            for jj in range(3):
                dp1dt_hat -= 2 * k_field[ii] * k_field[jj] * rfft3D_np(f[ii] * U_field[jj])
        dp1dt_hat *= -one_k2_field

        # Solve for dilatation: d = -Mt**2 (dp/dt + v_k dp/dk)
        udp1_hat = 0
        for ii in range(3):
            udp1_hat += 1j * k_field[ii] * rfft3D_np(U_field[ii] * p1)

        # d_hat = -Ma**2 * (dp1dt_hat + udp1_hat)
        d_hat = -1 / gamma * (dp1dt_hat + udp1_hat)
            
        # Construct dilatational velocity field
        w = np.zeros((3,N,N,N))
        for ii in range(3):
            w[ii] = irfft3D_np(-1j * k_field[ii] * one_k2_field * d_hat)

        # Calculate pressure and density field
        press = p_ref + p_ref * eps_squared * p1
        dens  = rho_ref + rho_ref * eps_squared * (p1 / gamma)    # rho_1 = p_1 / gamma
        # Calculate solenoidal + dilatational field
        U_final = u_ref * U_field + u_ref * eps_squared * w

        # # Calculate pressure and density field
        # press = p_ref + p1
        # dens  = rho_ref + Ma**2 * p1
        # # Calculate solenoidal + dilatational field
        # U_final = U_field + w

    else:

        resolution_in = U_field.shape[-3:]
        Nx_in,Ny_in,Nz_in = resolution_in
        number_of_cells = [int(resolution_in[i]*split_factors[i]) for i in range(3)]
        nx,ny,nz = number_of_cells
        assert nx == ny
        assert ny == nz

        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
        split_factors_out = tuple([split_factors[split_axis_in]
                                   if i == split_axis_out else 1 for i in range(3)])

        k_field = wavenumber_grid_parallel(number_of_cells, split_factors_out)
        one_k2_field = 1.0 / (jnp.sum(k_field * k_field, axis=0) + precision.get_eps())

        c_ref = jnp.sqrt(gamma * p_ref / rho_ref)
        q_rms = jnp.mean(jnp.sum(U_field * U_field, axis=0))
        q_rms = jax.lax.pmean(q_rms, axis_name="i")
        q_rms = jnp.sqrt(q_rms)
        u_rms = q_rms / jnp.sqrt(3)
        u_ref = u_rms

        Ma = u_ref / c_ref
        eps_squared = gamma * Ma**2
        U_field /= u_ref

        ## Solve Poisson equation for p1
        # - k^2 p_hat = k_i k_j (u_i u_j)_hat
        p1_hat = 0
        for ii in range(3):
            for jj in range(3):
                p1_hat += k_field[ii] * k_field[jj] * parallel_fft(U_field[ii] * U_field[jj],
                                                                   split_factors, split_axis_out) 
        p1_hat *= -one_k2_field
        p1 = parallel_ifft(p1_hat, split_factors_out, split_axis_in)

        # Solve Poisson equation for dp1dt
        # 1) Calculate f = (vk vi),k + p1,i
        f = jnp.zeros((3,Nx_in,Ny_in,Nz_in))
        for ii in range(3):
            f_ii = 1j * k_field[0]  * parallel_fft(U_field[0] * U_field[ii],
                                                   split_factors, split_axis_out)
            f_ii +=  1j * k_field[1]  * parallel_fft(U_field[1] * U_field[ii],
                                                     split_factors, split_axis_out)
            f_ii +=  1j * k_field[2]  * parallel_fft(U_field[2] * U_field[ii],
                                                     split_factors, split_axis_out)
            f_ii +=  1j * k_field[ii]  * parallel_fft(p1, split_factors, split_axis_out)
            f_ii = parallel_ifft(f_ii, split_factors_out, split_axis_in)
            f = f.at[ii].set(f_ii)

        # 2) Solve Poisson equation and FFT back >> dp1/dt
        # - k^2 dpdt_hat = -2 k_i k_j (f_i u_j)_hat
        dp1dt_hat = 0.0
        for ii in range(3):
            for jj in range(3):
                dp1dt_hat -= 2 * k_field[ii] * k_field[jj] * parallel_fft(f[ii] * U_field[jj],
                                                                          split_factors, split_axis_out)
        dp1dt_hat *= -one_k2_field

        # Solve for dilatation: d = -Mt**2 (dp/dt + v_k dp/dk)
        udp1_hat = 0
        for ii in range(3):
            udp1_hat += 1j * k_field[ii] * parallel_fft(U_field[ii] * p1, split_factors, split_axis_out)

        d_hat = -1 / gamma * (dp1dt_hat + udp1_hat)
            
        # Construct dilatational velocity field
        w = jnp.zeros((3,Nx_in,Ny_in,Nz_in))
        for ii in range(3):
            w_ii = -1j * k_field[ii] * one_k2_field * d_hat
            w_ii = parallel_ifft(w_ii, split_factors_out, split_axis_in)
            w = w.at[ii].set(w_ii)

        # Calculate pressure and density field
        press = p_ref + p_ref * eps_squared * p1
        dens  = rho_ref + rho_ref * eps_squared * (p1 / gamma)    # rho_1 = p_1 / gamma
        # Calculate solenoidal + dilatational field
        U_final = u_ref * U_field + u_ref * eps_squared * w

    return U_final, press, dens

def rescale_field(
        U_field: np.array,
        ek_target: np.array,
        split_factors: Tuple[int],
        ) -> np.ndarray:
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

    no_subdomains = np.prod(np.array(split_factors))
    is_parallel = True if no_subdomains > 1 else False

    if not is_parallel:
            
        _, Nx, Ny, Nz = U_field.shape
        Mx = Nx // 2 + 1
        My = Ny // 2 + 1
        Mz = Nz // 2 + 1

        # Factor to account for fft normalization
        fftfactor = 1 / Nx / Ny / Nz

        # FORWARD FFT
        U_hat = rfft3D_np(U_field)
        U_hat[:,-1,:,:] = 0
        U_hat[:,:,My-1,:] = 0
        U_hat[:,:,:,Mz-1] = 0
        U_hat *= fftfactor

        # START ENERGY SPECTRUM
        ek = np.zeros(Nx)
        nsamples = np.zeros(Nx)
        
        # print(Nx)
        # exit()
        k = np.arange(Nx)
        kx = np.fft.fftfreq(Nx, 1./Nx).astype(int)
        kx = kx[:Mx]; kx[-1] *= -1
        ky = np.fft.fftfreq(Ny, 1./Ny).astype(int)
        kz = np.fft.fftfreq(Nz, 1./Nz).astype(int)

        K = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=int)
        Kmag = np.sqrt(np.sum(K*K, axis=0))
        Shell = ( Kmag + 0.5 ).astype(int) 

        Fact = 2 * (K[0] > 0) * (K[0] < Nx//2) + 1 * (K[0]==0) + 1 * (K[0] == Nx//2)
        Fact[-1,:,:] = 0
        Fact[:,My-1,:] = 0
        Fact[:,:,Mz-1] = 0

        # # TODO update in vectorial notation
        UU = Fact * 0.5 * (np.sum(np.abs(U_hat)**2, axis=0))
        np.add.at(ek, Shell.flatten(), UU.flatten())
        # np.add.at(nsamples, Shell.flatten(), 1)
        np.add.at(nsamples, Shell.flatten(), Fact.flatten())
        ek *= 4 * np.pi * k**2 / (nsamples + precision.get_eps())
        ek = energy_spectrum_physical(U_field, multiplicative_factor=0.5) #TODO deniz this ek and above ek not the same, why ?

        # END ENERGY SPECTRUM
        scale = np.sqrt(ek_target / (ek + precision.get_eps()))
        U_hat *= scale[Shell] * 1/fftfactor

        # Transform back to real space
        U_scaled = irfft3D_np(U_hat)

    else:

        number_of_cells_device = U_field.shape[-3:]
        number_of_cells = [int(number_of_cells_device[i]*split_factors[i]) for i in range(3)]
        
        split_axis_in = np.argmax(np.array(split_factors))
        split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
        U_hat = parallel_fft(U_field, split_factors, split_axis_out)

        # WAVENUMBER GRID
        N = number_of_cells[0]
        split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])
        k_field = wavenumber_grid_parallel(number_of_cells, split_factors_out)
        k_vec = jnp.arange(N)
        k_mag_field = jnp.linalg.norm(k_field, axis=0, ord=2)
        shell = (k_mag_field + 0.5).astype(int)

        # ENERGY SPECTRUM
        fftfactor = 1 / N**3
        U_hat *= fftfactor
        abs_energy = jnp.sum(jnp.real(U_hat * jnp.conj(U_hat)), axis=(-4))
        abs_energy *= 0.5
        n_samples = jnp.zeros(N)
        n_samples = n_samples.at[shell.flatten()].add(1.0)
        n_samples = jax.lax.psum(n_samples, axis_name="i")
        ek = jnp.zeros(N)
        ek = ek.at[shell.flatten()].add(abs_energy.flatten())
        ek = jax.lax.psum(ek, axis_name="i")
        ek *= 4 * jnp.pi * k_vec**2 / (n_samples + precision.get_eps())

        # RESCALE
        scale = jnp.sqrt(ek_target / (ek + precision.get_eps()))
        U_hat *= scale[shell] * 1/fftfactor
        split_factors = tuple([split_factors[split_axis_in]
                               if i == split_axis_out else 1 for i in range(3)])
        U_scaled = parallel_ifft(U_hat, split_factors, split_axis_in)

    return U_scaled

