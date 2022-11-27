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

from jaxfluids.solvers.riemann_solvers.HLL import HLL
from jaxfluids.solvers.riemann_solvers.HLLC import HLLC
from jaxfluids.solvers.riemann_solvers.HLLCLM import HLLCLM
from jaxfluids.solvers.riemann_solvers.Rusanov import Rusanov
from jaxfluids.solvers.riemann_solvers.RusanovNN import RusanovNN

from jaxfluids.solvers.riemann_solvers.signal_speeds import signal_speed_Arithmetic, signal_speed_Rusanov, signal_speed_Davis, signal_speed_Davis_2,\
    signal_speed_Einfeldt, signal_speed_Toro

DICT_RIEMANN_SOLVER ={
    'HLL': HLL,
    'HLLC': HLLC,
    'HLLCLM': HLLCLM,
    'RUSANOV': Rusanov,
    'RUSANOVNN': RusanovNN,
}

DICT_SIGNAL_SPEEDS ={
    'ARITHMETIC': signal_speed_Arithmetic,
    'RUSANOV': signal_speed_Rusanov,
    'DAVIS': signal_speed_Davis,
    'DAVIS2': signal_speed_Davis_2,
    'EINFELDT': signal_speed_Einfeldt,
    'TORO': signal_speed_Toro,
}