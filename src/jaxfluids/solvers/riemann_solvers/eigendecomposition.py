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
from typing import Tuple, Union

import jax 
from jax import vmap
import jax.numpy as jnp
import numpy as np

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.domain_information import DomainInformation
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class Eigendecomposition:
    """ The Eigendecomposition class implements functionality for
    eigendecomposition of the Jacobian matrix. Eigendecomposition
    can be done based on primitive or conservative variables. The 
    frozen state can be calculated from arithmetic or Roe averages.

    eigendecomp_prim only returns left and right eigenvectors,
    while eigendecomp_cons additionally returns eigenvalues according
    to a user-specified flux-splitting.

    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, material_manager: MaterialManager, stencil_size: int, 
        domain_information: DomainInformation, flux_splitting: str = None) -> None:
        self.nh             = domain_information.nh_conservatives
        self.number_cells   = domain_information.number_of_cells

        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives

        # FROZEN STATE x_{i+1/2} is computed either via arithmetic mean or via Roe average
        # self.frozen_state = "ARITHMETIC" 
        self.frozen_state = "ROE" 

        # SLICES TO GET EACH STENCIL
        self.s1_ = [
            np.arange(stencil_size).reshape(1,-1) + np.arange(self.nh - stencil_size//2, self.number_cells[0] + self.nh - stencil_size//2 + 1).reshape(-1,1),
            np.arange(stencil_size).reshape(1,-1) + np.arange(self.nh - stencil_size//2, self.number_cells[1] + self.nh - stencil_size//2 + 1).reshape(-1,1),
            np.arange(stencil_size).reshape(1,-1) + np.arange(self.nh - stencil_size//2, self.number_cells[2] + self.nh - stencil_size//2 + 1).reshape(-1,1),
        ]

        self.stencil_slices = [
            jnp.s_[..., self.s1_[0], self.nhy, self.nhz],
            jnp.s_[..., self.nhx, self.s1_[1], self.nhz],
            jnp.s_[..., self.nhx, self.nhy, self.s1_[2]],
        ]

        # SLICES TO GET CELL I AND I+1
        self.s2_ = [
            [jnp.s_[..., stencil_size//2-1, None:None, None:None], jnp.s_[..., stencil_size//2, None:None, None:None]],
            [jnp.s_[..., None:None, stencil_size//2-1, None:None], jnp.s_[..., None:None, stencil_size//2, None:None]],
            [jnp.s_[..., None:None, None:None, stencil_size//2-1], jnp.s_[..., None:None, None:None, stencil_size//2]],
        ]

        self.material_manager = material_manager

        self.flux_splitting = flux_splitting

    def get_stencil_window(self, var: jnp.ndarray, axis: int) -> jnp.ndarray:
        return var[self.stencil_slices[axis]]

    def compute_frozen_state(self, primes: jnp.ndarray, axis: int) -> jnp.ndarray:
        primes_L   = primes[self.s2_[axis][0]]
        primes_R   = primes[self.s2_[axis][1]]
        
        if self.frozen_state == "ARITHMETIC":
            primes_ave      = 0.5 * ( primes_L + primes_R )
            grueneisen_ave  = self.material_manager.get_grueneisen(rho=primes_ave[0])  # TODO ASSUMES ONE MATERIAL
            enthalpy_ave    = self.material_manager.get_total_enthalpy(p=primes_ave[4], rho=primes_ave[0], u=primes_ave[1], v=primes_ave[2], w=primes_ave[3])

            c_ave           = self.material_manager.get_speed_of_sound(primes_ave[4], primes_ave[0])
            cc_ave          = c_ave * c_ave
            velocity_square = primes_ave[1] * primes_ave[1] + primes_ave[2] * primes_ave[2] + primes_ave[3] * primes_ave[3]
        
            return primes_ave, enthalpy_ave, grueneisen_ave, c_ave, cc_ave, velocity_square

        if self.frozen_state == "ROE":
            # TODO Better way to calculate primes_ave
            primes_ave     = self.compute_roe_cons(primes_L, primes_R)
            primes_ave.at[0].set(jnp.sqrt(primes_L[0] * primes_R[0]))

            rho_sqrt_L, rho_sqrt_R = jnp.sqrt(primes_L[0]), jnp.sqrt(primes_R[0])
            rho_div = 1.0 / ( rho_sqrt_L + rho_sqrt_R )
            
            enthalpy_L     = self.material_manager.get_total_enthalpy(p=primes_L[4], rho=primes_L[0], u=primes_L[1], v=primes_L[2], w=primes_L[3])
            enthalpy_R     = self.material_manager.get_total_enthalpy(p=primes_R[4], rho=primes_R[0], u=primes_R[1], v=primes_R[2], w=primes_R[3])
            enthalpy_ave   = (rho_sqrt_L * enthalpy_L + rho_sqrt_R * enthalpy_R) * rho_div
            
            psi_L          = self.material_manager.get_psi(p=primes_L[4], rho=primes_L[0])
            psi_R          = self.material_manager.get_psi(p=primes_R[4], rho=primes_R[0])
            psi_ave        = (rho_sqrt_L * psi_L + rho_sqrt_R * psi_R) * rho_div

            grueneisen_L    = self.material_manager.get_grueneisen(rho=primes_L[0])
            grueneisen_R    = self.material_manager.get_grueneisen(rho=primes_R[0])
            grueneisen_ave  = (rho_sqrt_L * grueneisen_L + rho_sqrt_R * grueneisen_R) * rho_div

            squared_velocity_difference = (primes_R[1] - primes_L[1]) * (primes_R[1] - primes_L[1]) + \
                (primes_R[2] - primes_L[2]) * (primes_R[2] - primes_L[2]) + (primes_R[3] - primes_L[3]) * (primes_R[3] - primes_L[3]) 

            p_over_rho_ave  = (rho_sqrt_L * primes_L[4]/primes_L[0] + rho_sqrt_R * primes_R[4]/primes_R[0]) * rho_div \
                            + 0.5 * primes_ave[0] * rho_div * rho_div * squared_velocity_difference

            velocity_square = primes_ave[1] * primes_ave[1] + primes_ave[2] * primes_ave[2] + primes_ave[3] * primes_ave[3]
            
            # cc_ave = (self.material_manager.gamma - 1) * (enthalpy_ave - 0.5 * velocity_square)
            cc_ave = psi_ave + grueneisen_ave * p_over_rho_ave
            
            c_ave  = jnp.sqrt( cc_ave )

            return primes_ave, enthalpy_ave, grueneisen_ave, c_ave, cc_ave, velocity_square

    def compute_roe_cons(self, prime_L: jnp.ndarray, prime_R: jnp.ndarray) -> jnp.ndarray:
        """Computes the Roe averaged conservative state.

        :param prime_L: Buffer of primitive variables left of a cell face.
        :type prime_L: jnp.ndarray
        :param prime_R: Buffer of primitive variables right of a cell face.
        :type prime_R: jnp.ndarray
        :return: Buffer of Roe averaged quantities at the cell face.
        :rtype: jnp.ndarray
        """
        roe_cons = (jnp.sqrt(prime_L[0]) * prime_L + jnp.sqrt(prime_R[0]) * prime_R) / (jnp.sqrt(prime_L[0]) + jnp.sqrt(prime_R[0]) + self.eps)
        return roe_cons

    def eigendecomp_prim(self, primes: jnp.ndarray, axis: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the eigendecomposition of the Jacobian matrix wrt primitive variables.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param axis: Direction of the cell face at which the eigendecomposition is to be performed.
        :type axis: int
        :return: Buffer of left and right eigenvectors.
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        _s = primes_ave[0].shape

        primes_ave, enthalpy_ave, grueneisen_ave, c_ave, cc_ave, velocity_square  = self.compute_frozen_state(primes, axis)
        # X - DIRECTION
        if axis == 0:
            right_eigs = jnp.array([
                [primes_ave[0]         , jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), primes_ave[0]         ],
                [-c_ave                , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), c_ave                 ],
                [jnp.zeros(_s)         , jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s)         ],
                [jnp.zeros(_s)         , jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s)         ],
                [cc_ave * primes_ave[0], jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), cc_ave * primes_ave[0]],
                ])

            left_eigs  = jnp.array([
                [jnp.zeros(_s), -0.5 / c_ave , jnp.zeros(_s), jnp.zeros(_s), 0.5 / cc_ave / primes_ave[0]],
                [jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), -1.0 / cc_ave               ],
                [jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s)               ],
                [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s)               ],
                [jnp.zeros(_s), 0.5 / c_ave  , jnp.zeros(_s), jnp.zeros(_s), 0.5 / cc_ave / primes_ave[0]],
            ])
        # Y - DIRECTION
        elif axis == 1:
            right_eigs = jnp.array([
                [primes_ave[0]         , jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), primes_ave[0]         ],
                [jnp.zeros(_s)         , jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s)         ],
                [-c_ave                , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), c_ave                 ],
                [jnp.zeros(_s)         , jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s)         ],
                [cc_ave * primes_ave[0], jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), cc_ave * primes_ave[0]],
                ])

            left_eigs  = jnp.array([
                [jnp.zeros(_s), jnp.zeros(_s), -0.5 / c_ave , jnp.zeros(_s), 0.5 / cc_ave / primes_ave[0]],
                [jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), -1.0 / cc_ave               ],
                [jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)               ],
                [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s)               ],
                [jnp.zeros(_s), jnp.zeros(_s), 0.5 / c_ave  , jnp.zeros(_s), 0.5 / cc_ave / primes_ave[0]],
            ])
        # Z - DIRECTION
        elif axis == 2:
            right_eigs = jnp.array([
                [primes_ave[0]         , jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), primes_ave[0]         ],
                [jnp.zeros(_s)         , jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s)         ],
                [jnp.zeros(_s)         , jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s)         ],
                [-c_ave                , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), c_ave                 ],
                [cc_ave * primes_ave[0], jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), cc_ave * primes_ave[0]],
                ])

            left_eigs  = jnp.array([
                [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), -0.5 / c_ave , 0.5 / cc_ave / primes_ave[0]],
                [jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), -1.0 / cc_ave               ],
                [jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)               ],
                [jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s), jnp.zeros(_s)               ],
                [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), 0.5 / c_ave  , 0.5 / cc_ave / primes_ave[0]],
            ])

        return right_eigs, left_eigs

    def _eigendecomp_cons(self, primes: jnp.ndarray, axis: int) -> Union[Tuple[jnp.ndarray, jnp.ndarray], 
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Computes eigendecomposition of the Jacobian matrix for conservative variables.
        Formulation only valid for an ideal gas. Implementation according to Rohde 2001.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param axis: Direction of the cell face at which the eigendecomposition is to be performed.
        :type axis: int
        :return: Buffer of left, right eigenvectors and the eigenvalues.
        :rtype: Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
        """

        primes_ave, enthalpy_ave, grueneisen_ave, c_ave, cc_ave, velocity_square  = self.compute_frozen_state(primes, axis)
        ek = 0.5 * velocity_square

        _s = primes_ave[0].shape

        # X - DIRECTION
        if axis == 0:
            right_eigs = jnp.array([
                [jnp.ones(_s)                        , jnp.ones(_s) ,  jnp.zeros(_s), jnp.zeros(_s), jnp.ones(_s)                        ],
                [primes_ave[1] - c_ave               , primes_ave[1],  jnp.zeros(_s), jnp.zeros(_s), primes_ave[1] + c_ave               ],
                [primes_ave[2]                       , primes_ave[2], -jnp.ones(_s) , jnp.zeros(_s), primes_ave[2]                       ],
                [primes_ave[3]                       , primes_ave[3],  jnp.zeros(_s), jnp.ones(_s) , primes_ave[3]                       ],
                [enthalpy_ave - primes_ave[1] * c_ave, ek           , -primes_ave[2], primes_ave[3], enthalpy_ave + primes_ave[1] * c_ave],
                ])

            left_eigs  = grueneisen_ave / 2 / cc_ave * jnp.array([
                [ek + c_ave / grueneisen_ave * primes_ave[1]  , -primes_ave[1] - c_ave / grueneisen_ave, -primes_ave[2]               , -primes_ave[3]             , jnp.ones(_s)     ],
                [2 / grueneisen_ave * cc_ave - velocity_square, 2 * primes_ave[1]                      , 2*primes_ave[2]              , 2*primes_ave[3]            , -2 * jnp.ones(_s)],
                [ 2 * cc_ave / grueneisen_ave * primes_ave[2] , jnp.zeros(_s)                          , - 2 * cc_ave / grueneisen_ave, jnp.zeros(_s)              , jnp.zeros(_s)    ],
                [-2 * cc_ave / grueneisen_ave * primes_ave[3] , jnp.zeros(_s)                          , jnp.zeros(_s)                , 2 * cc_ave / grueneisen_ave, jnp.zeros(_s)    ],
                [ek - c_ave / grueneisen_ave * primes_ave[1]  , -primes_ave[1] + c_ave / grueneisen_ave, -primes_ave[2]               , -primes_ave[3]             , jnp.ones(_s)     ],
            ])
        # Y - DIRECTION
        elif axis == 1:
            right_eigs = jnp.array([
                [jnp.ones(_s)                        , jnp.zeros(_s), jnp.ones(_s) , jnp.zeros(_s) , jnp.ones(_s)                        ],
                [primes_ave[1]                       , jnp.ones(_s) , primes_ave[1], jnp.zeros(_s) , primes_ave[1]                       ],
                [primes_ave[2] - c_ave               , jnp.zeros(_s), primes_ave[2], jnp.zeros(_s) , primes_ave[2] + c_ave               ],
                [primes_ave[3]                       , jnp.zeros(_s), primes_ave[3], -jnp.ones(_s) , primes_ave[3]                       ],
                [enthalpy_ave - primes_ave[2] * c_ave, primes_ave[1], ek           , -primes_ave[3], enthalpy_ave + primes_ave[2] * c_ave],
                ])

            left_eigs  = grueneisen_ave / 2 / cc_ave * jnp.array([
                [ek + c_ave / grueneisen_ave * primes_ave[2]  , -primes_ave[1]             , -primes_ave[2] - c_ave / grueneisen_ave, -primes_ave[3]              , jnp.ones(_s)     ],
                [-2 * cc_ave / grueneisen_ave * primes_ave[1] , 2 * cc_ave / grueneisen_ave, jnp.zeros(_s)                          , jnp.zeros(_s)               , jnp.zeros(_s)    ],
                [2 / grueneisen_ave * cc_ave - velocity_square, 2*primes_ave[1]            , 2 * primes_ave[2]                      , 2 * primes_ave[3]           , -2 * jnp.ones(_s)],
                [2 * cc_ave / grueneisen_ave * primes_ave[3]  , jnp.zeros(_s)              , jnp.zeros(_s)                          , -2 * cc_ave / grueneisen_ave, jnp.zeros(_s)    ],
                [ek - c_ave / grueneisen_ave * primes_ave[2]  , -primes_ave[1]             , -primes_ave[2] + c_ave / grueneisen_ave, -primes_ave[3]              , jnp.ones(_s)     ],
            ])
        # Z - DIRECTION
        elif axis == 2:
            right_eigs = jnp.array([
                [jnp.ones(_s)                        , jnp.zeros(_s) , jnp.zeros(_s), jnp.ones(_s) , jnp.ones(_s)                        ],
                [primes_ave[1]                       , -jnp.ones(_s) , jnp.zeros(_s), primes_ave[1], primes_ave[1]                       ],
                [primes_ave[2]                       , jnp.zeros(_s) , jnp.ones(_s) , primes_ave[2], primes_ave[2]                       ],
                [primes_ave[3] - c_ave               , jnp.zeros(_s) , jnp.zeros(_s), primes_ave[3], primes_ave[3] + c_ave               ],
                [enthalpy_ave - primes_ave[3] * c_ave, -primes_ave[1], primes_ave[2], ek           , enthalpy_ave + primes_ave[3] * c_ave],
                ])

            left_eigs  = grueneisen_ave / 2 / cc_ave * jnp.array([
                [ek + c_ave / grueneisen_ave * primes_ave[3] , -primes_ave[1]              , -primes_ave[2]             , -primes_ave[3] - c_ave / grueneisen_ave, jnp.ones(_s)     ],
                [ 2 * cc_ave / grueneisen_ave * primes_ave[1], -2 * cc_ave / grueneisen_ave, jnp.zeros(_s)              , jnp.zeros(_s)                          , jnp.zeros(_s)    ],
                [-2 * cc_ave / grueneisen_ave * primes_ave[2], jnp.zeros(_s)               , 2 * cc_ave / grueneisen_ave, jnp.zeros(_s)                          , jnp.zeros(_s)    ],
                [2 / grueneisen_ave*cc_ave - velocity_square , 2 * primes_ave[1]           , 2 * primes_ave[2]          , 2 * primes_ave[3]                      , -2 * jnp.ones(_s)],
                [ek - c_ave / grueneisen_ave * primes_ave[3] , -primes_ave[1]              , -primes_ave[2]             , -primes_ave[3] + c_ave / grueneisen_ave, jnp.ones(_s)     ],
            ])

        if not self.flux_splitting:
            return right_eigs, left_eigs

        # EIGENVALUES FOR FLUX-SPLITTING

        # ROE EIGENVALUES
        if self.flux_splitting == "ROE":
            gamma_1     = jnp.abs(primes_ave[axis+1] - c_ave)
            gamma_234   = jnp.abs(primes_ave[axis+1])
            gamma_5     = jnp.abs(primes_ave[axis+1] + c_ave) 

        # cLLF EIGENVALUES
        if self.flux_splitting == "CLLF":
            gamma_1     = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1] - self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0])), jnp.abs(primes[self.s2_[axis][1]][axis+1] - self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][1]][4], rho=primes[self.s2_[axis][1]][0])))
            gamma_234   = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1]), jnp.abs(primes[self.s2_[axis][1]][axis+1]))
            gamma_5     = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1] + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0])), jnp.abs(primes[self.s2_[axis][1]][axis+1] + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][1]][4], rho=primes[self.s2_[axis][1]][0])))

        # LLF EIGENVALUES
        if self.flux_splitting == "LLF":
            gamma_1 = gamma_234 = gamma_5 = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1]) + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0]), jnp.abs(primes[self.s2_[axis][1]][axis+1]) + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][1]][4], rho=primes[self.s2_[axis][1]][0]))

        # GLF EIGENVALUES
        if self.flux_splitting == "GLF":
            gamma_12345 = jnp.max(jnp.abs(primes[self.s2_[axis][0]][axis+1]) + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0]))
            gamma_1 = gamma_234 = gamma_5 = gamma_12345 * jnp.ones(_s)

        eigvals = jnp.array([
            [gamma_1      , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), gamma_234    , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), gamma_234    , jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), gamma_234    , jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), gamma_5      ],
        ])

        return right_eigs, left_eigs, eigvals     

    def eigendecomp_cons(self, primes: jnp.ndarray, axis: int) -> Union[Tuple[jnp.ndarray, jnp.ndarray], 
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Computes eigendecomposition of the Jacobian matrix for conservative variables.
        Formulation for a general equation of state. Implementation according to Fedkiv et al.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param axis: Direction of the cell face at which the eigendecomposition is to be performed.
        :type axis: int
        :return: Buffer of left, right eigenvectors and the eigenvalues.
        :rtype: Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
        """

        primes_ave, enthalpy_ave, grueneisen_ave, c_ave, cc_ave, velocity_square  = self.compute_frozen_state(primes, axis)
        one_cc_ave  = 1.0 / cc_ave
        one_rho_ave = 1.0 / primes_ave[0]
        _s = primes_ave[0].shape

        # LEFT AND RIGHT EIGENVECTORS ACCORDING TO FEDKIW ET AL 1999
        # X - DIRECTION
        if axis == 0:
            right_eigs = jnp.array([
                [jnp.ones(_s)                        , grueneisen_ave                        ,  jnp.zeros(_s)                , jnp.zeros(_s)                , jnp.ones(_s)                        ],
                [primes_ave[1] - c_ave               , grueneisen_ave * primes_ave[1]        ,  jnp.zeros(_s)                , jnp.zeros(_s)                , primes_ave[1] + c_ave               ],
                [primes_ave[2]                       , grueneisen_ave * primes_ave[2]        , -primes_ave[0]                , jnp.zeros(_s)                , primes_ave[2]                       ],
                [primes_ave[3]                       , grueneisen_ave * primes_ave[3]        ,  jnp.zeros(_s)                , primes_ave[0]                , primes_ave[3]                       ],
                [enthalpy_ave - primes_ave[1] * c_ave, grueneisen_ave * enthalpy_ave - cc_ave, -primes_ave[0] * primes_ave[2], primes_ave[0] * primes_ave[3], enthalpy_ave + primes_ave[1] * c_ave],
                ])

            left_eigs  = jnp.array([
                [0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave + (primes_ave[1] + c_ave) * c_ave ) , 0.5 * one_cc_ave * (-primes_ave[1] * grueneisen_ave - c_ave), 0.5 * one_cc_ave * (-primes_ave[2] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[3] * grueneisen_ave), 0.5 * one_cc_ave * grueneisen_ave],
                [one_cc_ave * (enthalpy_ave - velocity_square), primes_ave[1] * one_cc_ave, primes_ave[2] * one_cc_ave, primes_ave[3] * one_cc_ave, -one_cc_ave  ],
                [primes_ave[2] * one_rho_ave                  , jnp.zeros(_s)             , -one_rho_ave              , jnp.zeros(_s)             , jnp.zeros(_s)],
                [-primes_ave[3] * one_rho_ave                 , jnp.zeros(_s)             , jnp.zeros(_s)             , one_rho_ave               , jnp.zeros(_s)],
                [0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave - (primes_ave[1] - c_ave) * c_ave ) , 0.5 * one_cc_ave * (-primes_ave[1] * grueneisen_ave + c_ave), 0.5 * one_cc_ave * (-primes_ave[2] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[3] * grueneisen_ave), 0.5 * one_cc_ave * grueneisen_ave],
            ])
        # Y - DIRECTION
        elif axis == 1:
            right_eigs = jnp.array([
                [jnp.ones(_s)                        , jnp.zeros(_s)                , grueneisen_ave                        ,  jnp.zeros(_s)                , jnp.ones(_s)                        ],
                [primes_ave[1]                       , primes_ave[0]                , grueneisen_ave * primes_ave[1]        ,  jnp.zeros(_s)                , primes_ave[1]                       ],
                [primes_ave[2] - c_ave               , jnp.zeros(_s)                , grueneisen_ave * primes_ave[2]        ,  jnp.zeros(_s)                , primes_ave[2] + c_ave               ],
                [primes_ave[3]                       , jnp.zeros(_s)                , grueneisen_ave * primes_ave[3]        , -primes_ave[0]                , primes_ave[3]                       ],
                [enthalpy_ave - primes_ave[2] * c_ave, primes_ave[0] * primes_ave[1], grueneisen_ave * enthalpy_ave - cc_ave, -primes_ave[0] * primes_ave[3], enthalpy_ave + primes_ave[2] * c_ave],
                ])

            left_eigs  = jnp.array([
                [0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave + (primes_ave[2] + c_ave) * c_ave ) , 0.5 * one_cc_ave * (-primes_ave[1] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[2] * grueneisen_ave - c_ave), 0.5 * one_cc_ave * (-primes_ave[3] * grueneisen_ave), 0.5 * one_cc_ave * grueneisen_ave],
                [-primes_ave[1] * one_rho_ave                 , one_rho_ave               , jnp.zeros(_s)             , jnp.zeros(_s)             , jnp.zeros(_s)],
                [one_cc_ave * (enthalpy_ave - velocity_square), primes_ave[1] * one_cc_ave, primes_ave[2] * one_cc_ave, primes_ave[3] * one_cc_ave, -one_cc_ave  ],
                [primes_ave[3] * one_rho_ave                  , jnp.zeros(_s)             , jnp.zeros(_s)             , -one_rho_ave              , jnp.zeros(_s)],
                [0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave - (primes_ave[2] - c_ave) * c_ave ) , 0.5 * one_cc_ave * (-primes_ave[1] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[2] * grueneisen_ave + c_ave), 0.5 * one_cc_ave * (-primes_ave[3] * grueneisen_ave), 0.5 * one_cc_ave * grueneisen_ave],
            ])
        # Z - DIRECTION
        elif axis == 2:
            right_eigs = jnp.array([
                [jnp.ones(_s)                        ,  jnp.zeros(_s)                , jnp.zeros(_s)                , grueneisen_ave                        , jnp.ones(_s)                        ],
                [primes_ave[1]                       , -primes_ave[0]                , jnp.zeros(_s)                , grueneisen_ave * primes_ave[1]        , primes_ave[1]                       ],
                [primes_ave[2]                       ,  jnp.zeros(_s)                , primes_ave[0]                , grueneisen_ave * primes_ave[2]        , primes_ave[2]                       ],
                [primes_ave[3] - c_ave               ,  jnp.zeros(_s)                , jnp.zeros(_s)                , grueneisen_ave * primes_ave[3]        , primes_ave[3] + c_ave               ],
                [enthalpy_ave - primes_ave[3] * c_ave, -primes_ave[0] * primes_ave[1], primes_ave[0] * primes_ave[2], grueneisen_ave * enthalpy_ave - cc_ave, enthalpy_ave + primes_ave[3] * c_ave],
                ])

            left_eigs  = jnp.array([
                [0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave + (primes_ave[3] + c_ave) * c_ave ) , 0.5 * one_cc_ave * (-primes_ave[1] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[2] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[3] * grueneisen_ave - c_ave), 0.5 * one_cc_ave * grueneisen_ave],
                [primes_ave[1] * one_rho_ave                  , -one_rho_ave              , jnp.zeros(_s)             , jnp.zeros(_s)             , jnp.zeros(_s)],
                [-primes_ave[2] * one_rho_ave                 , jnp.zeros(_s)             , one_rho_ave               , jnp.zeros(_s)             , jnp.zeros(_s)],
                [one_cc_ave * (enthalpy_ave - velocity_square), primes_ave[1] * one_cc_ave, primes_ave[2] * one_cc_ave, primes_ave[3] * one_cc_ave, -one_cc_ave  ],
                [0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave - (primes_ave[3] - c_ave) * c_ave ) , 0.5 * one_cc_ave * (-primes_ave[1] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[2] * grueneisen_ave), 0.5 * one_cc_ave * (-primes_ave[3] * grueneisen_ave + c_ave), 0.5 * one_cc_ave * grueneisen_ave],
            ])

        if not self.flux_splitting:
            return right_eigs, left_eigs

        # EIGENVALUES FOR FLUX-SPLITTING

        # ROE EIGENVALUES
        if self.flux_splitting == "ROE":
            gamma_1     = jnp.abs(primes_ave[axis+1] - c_ave)
            gamma_234   = jnp.abs(primes_ave[axis+1])
            gamma_5     = jnp.abs(primes_ave[axis+1] + c_ave) 

        # cLLF EIGENVALUES
        if self.flux_splitting == "CLLF":
            gamma_1     = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1] - self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0])), jnp.abs(primes[self.s2_[axis][1]][axis+1] - self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][1]][4], rho=primes[self.s2_[axis][1]][0])))
            gamma_234   = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1]), jnp.abs(primes[self.s2_[axis][1]][axis+1]))
            gamma_5     = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1] + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0])), jnp.abs(primes[self.s2_[axis][1]][axis+1] + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][1]][4], rho=primes[self.s2_[axis][1]][0])))

        # LLF EIGENVALUES
        if self.flux_splitting == "LLF":
            gamma_1 = gamma_234 = gamma_5 = jnp.maximum(jnp.abs(primes[self.s2_[axis][0]][axis+1]) + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0]), jnp.abs(primes[self.s2_[axis][1]][axis+1]) + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][1]][4], rho=primes[self.s2_[axis][1]][0]))

        # GLF EIGENVALUES
        if self.flux_splitting == "GLF":
            gamma_12345 = jnp.max(jnp.abs(primes[self.s2_[axis][0]][axis+1]) + self.material_manager.get_speed_of_sound(p=primes[self.s2_[axis][0]][4], rho=primes[self.s2_[axis][0]][0]))
            gamma_1 = gamma_234 = gamma_5 = gamma_12345 * jnp.ones(_s)

        eigvals = jnp.array([
            [gamma_1      , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), gamma_234    , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), gamma_234    , jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), gamma_234    , jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), gamma_5      ],
        ])

        return right_eigs, left_eigs, eigvals     

    def transformtochar(self, stencil: jnp.ndarray, left_eig: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Transforms the stencil from physical to characteristic space.

        :param stencil: Buffer with variables in physical space.
        :type stencil: jnp.ndarray
        :param left_eig: Buffer of left eigenvalues.
        :type left_eig: jnp.ndarray
        :param axis: Spatial direction along which transformation has to be performed. 
        :type axis: int
        :return: Buffer with variables in characteristic space.
        :rtype: jnp.ndarray
        """
        left_eig = jnp.expand_dims(left_eig, axis=axis-3)
        return jnp.einsum("ij...,j...->i...", left_eig, stencil)

    def transformtophysical(self, stencil: jnp.ndarray, right_eig: jnp.ndarray) -> jnp.ndarray:
        """Transforms the stencil from characteristic to physical space.

        :param stencil: Buffer with variables in characteristic space.
        :type stencil: jnp.ndarray
        :param right_eig: Buffer of right eigenvalues.
        :type right_eig: jnp.ndarray
        :return: Buffer with variables in physical space.
        :rtype: jnp.ndarray
        """
        return jnp.einsum("ij...,j...->i...", right_eig, stencil)