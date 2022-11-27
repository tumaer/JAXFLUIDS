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

from jaxfluids.stencils.reconstruction.weno1_js import WENO1
from jaxfluids.stencils.reconstruction.weno3_js import WENO3
from jaxfluids.stencils.reconstruction.weno3_n import WENO3N
from jaxfluids.stencils.reconstruction.weno3_nn_opt1 import WENO3NNOPT1
from jaxfluids.stencils.reconstruction.weno3_nn_opt2 import WENO3NNOPT2
from jaxfluids.stencils.reconstruction.weno3_z import WENO3Z
from jaxfluids.stencils.reconstruction.weno3_fp import WENO3FP
from jaxfluids.stencils.reconstruction.weno5_js import WENO5
from jaxfluids.stencils.reconstruction.weno5_z import WENO5Z
from jaxfluids.stencils.reconstruction.weno6_cu import WENO6CU
from jaxfluids.stencils.reconstruction.weno6_cum1 import WENO6CUM1
from jaxfluids.stencils.reconstruction.weno6_cum2 import WENO6CUM2
from jaxfluids.stencils.reconstruction.weno6_cumd import WENO6CUMD
from jaxfluids.stencils.reconstruction.weno7_js import WENO7
from jaxfluids.stencils.reconstruction.teno5 import TENO5

DICT_SPATIAL_RECONSTRUCTION = {
    'WENO1':            WENO1,
    'WENO3-JS':         WENO3,
    'WENO3-N':          WENO3N,
    'WENO3-NN-OPT1':    WENO3NNOPT1,
    'WENO3-NN-OPT2':    WENO3NNOPT2,
    'WENO3-Z':          WENO3Z,
    'WENO3-FP':         WENO3FP,
    'WENO5-JS':         WENO5,
    'WENO5-Z':          WENO5Z,
    'WENO6-CU':         WENO6CU,
    'WENO6-CUM1':       WENO6CUM1,
    'WENO6-CUM2':       WENO6CUM2,
    'WENO6-CUMD':       WENO6CUMD,
    'WENO7-JS':         WENO7,
    'TENO5':            TENO5,
}

from jaxfluids.stencils.reconstruction.central_second_order import CentralSecondOrderReconstruction
from jaxfluids.stencils.reconstruction.central_fourth_order import CentralFourthOrderReconstruction

DICT_CENTRAL_RECONSTRUCTION = {
    'R2':   CentralSecondOrderReconstruction,
    'R4':   CentralFourthOrderReconstruction,
}

from jaxfluids.stencils.derivative.deriv_second_order_center import DerivativeSecondOrderCenter
from jaxfluids.stencils.derivative.deriv_fourth_order_center import DerivativeFourthOrderCenter
from jaxfluids.stencils.derivative.second_deriv_second_order_center import SecondDerivativeSecondOrderCenter
from jaxfluids.stencils.derivative.second_deriv_fourth_order_center import SecondDerivativeFourthOrderCenter
from jaxfluids.stencils.derivative.deriv_second_order_face import DerivativeSecondOrderFace
from jaxfluids.stencils.derivative.deriv_fourth_order_face import DerivativeFourthOrderFace

DICT_FIRST_DERIVATIVE_CENTER = {
    'DC2':  DerivativeSecondOrderCenter,
    'DC4':  DerivativeFourthOrderCenter,
}

DICT_SECOND_DERIVATIVE_CENTER = {
    'DC2':  SecondDerivativeSecondOrderCenter,
    'DC4':  SecondDerivativeFourthOrderCenter,
}

DICT_DERIVATIVE_FACE = {
    'DF2':  DerivativeSecondOrderFace,
    'DF4':  DerivativeFourthOrderFace, 
}

from jaxfluids.stencils.levelset.houc3 import HOUC3
from jaxfluids.stencils.levelset.houc5 import HOUC5
from jaxfluids.stencils.levelset.houc7 import HOUC7
from jaxfluids.stencils.levelset.weno3_deriv import WENO3DERIV
from jaxfluids.stencils.levelset.weno5_deriv import WENO5DERIV
from jaxfluids.stencils.levelset.deriv_first_order import DerivativeFirstOrderSided
from jaxfluids.stencils.levelset.deriv_first_order_subcell_fix import DerivativeFirstOrderSidedSubcellFix

DICT_DERIVATIVE_QUANTITY_EXTENDER = {
    'FIRSTORDER': DerivativeFirstOrderSided, 
}

DICT_DERIVATIVE_LEVELSET_ADVECTION = {
    'HOUC3': HOUC3,
    'HOUC5': HOUC5,
    'HOUC7': HOUC7,
    'FIRSTORDER': DerivativeFirstOrderSided, 
}

DICT_DERIVATIVE_REINITIALIZATION = {
    'WENO3DERIV': WENO3DERIV,
    'WENO5DERIV': WENO5DERIV,
    'FIRSTORDER': DerivativeFirstOrderSided, 
    'FIRSTORDERSUBCELL': DerivativeFirstOrderSidedSubcellFix, 
}