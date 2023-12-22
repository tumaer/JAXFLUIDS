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

import jax
import jax.numpy as jnp 

from jaxfluids.materials.material_manager import MaterialManager

def get_conservatives_from_primitives(primes:jnp.ndarray, material_manager: MaterialManager) -> jnp.ndarray:
    """Converts primitive variables to conservative variables.

    :param primes: Buffer of primitive variables
    :type primes: jnp.ndarray
    :param material_manager: Class that calculats material quantities 
    :type material_manager: MaterialManager
    :return: Buffer of conservative variables
    :rtype: jnp.ndarray
    """
    e    = material_manager.get_energy(p = primes[4], rho = primes[0])
    rho  = primes[0] # = rho
    rhou = primes[0] * primes[1] # = rho * u
    rhov = primes[0] * primes[2] # = rho * v
    rhow = primes[0] * primes[3] # = rho * w
    E    = primes[0] * (.5 * ( primes[1] * primes[1] + primes[2] * primes[2] + primes[3] * primes[3] ) + e)  # E = rho * (1/2 u^2 + e)
    cons = jnp.stack([rho, rhou, rhov, rhow, E], axis=0)
    return cons

def get_primitives_from_conservatives(cons: jnp.ndarray, material_manager: MaterialManager) -> jnp.ndarray:
    """Converts conservative variables to primitive variables.

    :param cons: Buffer of conservative variables
    :type cons: jnp.ndarray
    :param material_manager: Class that calculats material quantities
    :type material_manager: MaterialManager
    :return: Buffer of primitive variables
    :rtype: jnp.ndarray
    """
    rho =  cons[0]  # rho = rho
    u =  cons[1] / (cons[0] + jnp.finfo(float).eps)  # u = rho*u / rho
    v =  cons[2] / (cons[0] + jnp.finfo(float).eps)  # v = rho*v / rho
    w =  cons[3] / (cons[0] + jnp.finfo(float).eps)  # w = rho*w / rho
    e = cons[4] / (cons[0] + jnp.finfo(float).eps) - 0.5 * (u * u + v * v + w * w)
    p = material_manager.get_pressure(e, cons[0]) # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)
    primes = jnp.stack([rho, u, v, w, p], axis=0)
    return primes

def get_fluxes_xi(primes: jnp.ndarray, cons: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Computes the physical flux in a specified spatial direction.
    Cf. Eq. (3.65) in Toro.

    :param primes: Buffer of primitive variables
    :type primes: jnp.ndarray
    :param cons: Buffer of conservative variables
    :type cons: jnp.ndarray
    :param axis: Spatial direction along which fluxes are calculated
    :type axis: int
    :return: Physical fluxes in axis direction
    :rtype: jnp.ndarray
    """
    rho_ui      = cons[axis+1] # (rho u_i)
    rho_ui_u1   = cons[axis+1] * primes[1] # (rho u_i) * u_1
    rho_ui_u2   = cons[axis+1] * primes[2] # (rho u_i) * u_2
    rho_ui_u3   = cons[axis+1] * primes[3] # (rho u_i) * u_3
    ui_Ep       = primes[axis+1] * ( cons[4] + primes[4] ) 
    if axis == 0:
        rho_ui_u1 += primes[4]
    elif axis == 1:
        rho_ui_u2 += primes[4]
    elif axis == 2:
        rho_ui_u3 += primes[4]
    flux_xi = jnp.stack([rho_ui, rho_ui_u1, rho_ui_u2, rho_ui_u3, ui_Ep], axis=0)
    return flux_xi