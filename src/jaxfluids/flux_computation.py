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

from typing import Dict, Union

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.iles.ALDM import ALDM
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers import DICT_RIEMANN_SOLVER, DICT_SIGNAL_SPEEDS
from jaxfluids.solvers.high_order_godunov import HighOrderGodunov
from jaxfluids.solvers.flux_splitting_scheme import FluxSplittingScheme
from jaxfluids.stencils import DICT_SPATIAL_RECONSTRUCTION

class FluxComputer:
    """The Flux Computer sets up the user-specified flux function
    for the calculation of the convective terms. The flux calculation
    is called in the space solver by compute_convective_flux_xi().

    There are three general options for the convective flux function.
    1) High-order Godunov Scheme
    2) Flux-splitting Scheme
    3) ALDM Scheme
    """

    def __init__(self, numerical_setup: Dict, material_manager: MaterialManager, domain_information: DomainInformation) -> None:
        
        self.convective_solver = numerical_setup["conservatives"]["convective_fluxes"]["convective_solver"] 

        if self.convective_solver == "GODUNOV":
            self.flux_computer = HighOrderGodunov(
                material_manager        = material_manager, 
                domain_information      = domain_information, 
                riemann_solver          = DICT_RIEMANN_SOLVER[numerical_setup["conservatives"]["convective_fluxes"]["riemann_solver"]](material_manager=material_manager, signal_speed=DICT_SIGNAL_SPEEDS[numerical_setup["conservatives"]["convective_fluxes"]["signal_speed"]]), 
                reconstruction_stencil  = DICT_SPATIAL_RECONSTRUCTION[numerical_setup["conservatives"]["convective_fluxes"]["spatial_reconstructor"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis), 
                reconstruction_var      = numerical_setup["conservatives"]["convective_fluxes"]["reconstruction_var"], 
                is_safe_reconstruction  = numerical_setup["conservatives"]["convective_fluxes"]["is_safe_reconstruction"]
                )

        elif self.convective_solver == "FLUX-SPLITTING":
            self.flux_computer = FluxSplittingScheme(
                material_manager        = material_manager, 
                domain_information      = domain_information, 
                flux_splitting          = numerical_setup["conservatives"]["convective_fluxes"]["flux_splitting"], 
                reconstruction_stencil  = DICT_SPATIAL_RECONSTRUCTION[numerical_setup["conservatives"]["convective_fluxes"]["spatial_reconstructor"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis)
                )

        elif self.convective_solver == "ALDM":
            self.flux_computer = ALDM(
                domain_information  = domain_information, 
                material_manager    = material_manager
                )

    def compute_convective_flux_xi(self, primes: jnp.ndarray, cons: jnp.ndarray, axis: int,
        ml_parameters_dict: Union[Dict, None] = None, ml_networks_dict: Union[Dict, None] = None) -> jnp.ndarray:
        """Computes the convective fluxes. 

        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param cons: Conservative variable buffer
        :type cons: jnp.ndarray
        :param axis: Spatial direction
        :type axis: int
        :param ml_parameters_dict: Dictionary of neural network weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary of neural network architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Convective fluxes in axis direction
        :rtype: jnp.ndarray
        """
        
        fluxes_xi = self.flux_computer.compute_fluxes_xi(primes, cons, axis, 
            ml_parameters_dict=ml_parameters_dict, ml_networks_dict=ml_networks_dict)
        
        return fluxes_xi