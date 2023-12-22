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
from typing import List, Tuple, Union, Dict
import types

import jax
import jax.numpy as jnp

from jaxfluids.boundary_condition import BoundaryCondition
from jaxfluids.domain_information import DomainInformation
from jaxfluids.levelset.helper_functions import move_source_to_target_ii, move_source_to_target_ij, move_source_to_target_ijk, move_target_to_source_ii, move_target_to_source_ij, move_target_to_source_ijk
from jaxfluids.levelset.interface_quantity_computer import InterfaceQuantityComputer
from jaxfluids.levelset.interface_flux_computer import InterfaceFluxComputer 
from jaxfluids.levelset.levelset_reinitializer import LevelsetReinitializer
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.levelset.quantity_extender import QuantityExtender
from jaxfluids.levelset.geometry_calculator import GeometryCalculator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.utilities import get_conservatives_from_primitives, get_primitives_from_conservatives
from jaxfluids.stencils import DICT_FIRST_DERIVATIVE_CENTER, DICT_SECOND_DERIVATIVE_CENTER, DICT_DERIVATIVE_LEVELSET_ADVECTION, DICT_DERIVATIVE_QUANTITY_EXTENDER, DICT_DERIVATIVE_REINITIALIZATION
from jaxfluids.time_integration import DICT_TIME_INTEGRATION

class LevelsetHandler():

    """ The LevelsetHandler class manages computations to perform two-phase simulations using the levelset method.
    The main functionality includes
        - Transformation of the conservative states from volume-averages to actual conserved quantities according to the volume fraction
        - Weighting of the cell face fluxes according to the apertures
        - Computation of the interface fluxes
        - Computation of the levelset advection right-hand-side 
        - Extension of the primitive state from the real fluid cells to the ghost fluid cells
        - Mixing of the integrated conservatives
        - Computation of geometrical quantities, i.e., volume fraction, apertures and real fluid/cut cell masks
    """
    eps = jnp.finfo(jnp.float64).eps
    
    def __init__(self, domain_information: DomainInformation, numerical_setup: Dict, material_manager: MaterialManager,
        unit_handler: UnitHandler, solid_interface_velocity: Union[List, types.LambdaType], boundary_condition: BoundaryCondition) -> None:

        self.geometry_calculator     = GeometryCalculator(
                domain_information          = domain_information,
                first_derivative_stencil    = DICT_FIRST_DERIVATIVE_CENTER[numerical_setup["levelset"]["geometry_calculator_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis, offset=numerical_setup["levelset"]["halo_cells"]),
                second_derivative_stencil   = DICT_SECOND_DERIVATIVE_CENTER[numerical_setup["levelset"]["geometry_calculator_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis, offset=numerical_setup["levelset"]["halo_cells"]),
                subcell_reconstruction      = numerical_setup["levelset"]["subcell_reconstruction"]
                )

        self.interface_quantity_computer = InterfaceQuantityComputer(
            domain_information          = domain_information,
            material_manager            = material_manager,
            unit_handler                = unit_handler,
            solid_interface_velocity    = solid_interface_velocity,
            numerical_setup             = numerical_setup
        )

        self.interface_flux_computer = InterfaceFluxComputer(
            domain_information  = domain_information,
            material_manager    = material_manager,
            numerical_setup     = numerical_setup
        )

        self.extender_primes   = QuantityExtender(
                domain_information      = domain_information,
                boundary_condition      = boundary_condition,
                time_integrator         = DICT_TIME_INTEGRATION[numerical_setup["levelset"]["extension"]["time_integrator"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis),
                spatial_stencil         = DICT_DERIVATIVE_QUANTITY_EXTENDER[numerical_setup["levelset"]["extension"]["spatial_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis),
                is_interface            = False,
                )

        self.extender_interface   = QuantityExtender(
                domain_information      = domain_information,
                boundary_condition      = boundary_condition,
                time_integrator         = DICT_TIME_INTEGRATION[numerical_setup["levelset"]["extension"]["time_integrator"]](nh=domain_information.nh_geometry, inactive_axis=domain_information.inactive_axis),
                spatial_stencil         = DICT_DERIVATIVE_QUANTITY_EXTENDER[numerical_setup["levelset"]["extension"]["spatial_stencil"]](nh=domain_information.nh_geometry, inactive_axis=domain_information.inactive_axis),
                is_interface            = True,
        )

        self.levelset_reinitializer = LevelsetReinitializer(
                domain_information      = domain_information,
                boundary_condition      = boundary_condition,
                time_integrator         = DICT_TIME_INTEGRATION[numerical_setup["levelset"]["reinitialization"]["time_integrator"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis),
                derivative_stencil      = DICT_DERIVATIVE_REINITIALIZATION[numerical_setup["levelset"]["reinitialization"]["spatial_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis),
            )

        self.levelset_reinitializer_init = LevelsetReinitializer(
                domain_information  = domain_information,
                boundary_condition  = boundary_condition,
                time_integrator     = DICT_TIME_INTEGRATION[numerical_setup["levelset"]["reinitialization"]["time_integrator_init"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis),
                derivative_stencil  = DICT_DERIVATIVE_REINITIALIZATION[numerical_setup["levelset"]["reinitialization"]["spatial_stencil_init"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis),
        )

        self.levelset_advection_stencil  : SpatialDerivative = DICT_DERIVATIVE_LEVELSET_ADVECTION[numerical_setup["levelset"]["levelset_advection_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis)
        
        self.material_manager   = material_manager

        # DOMAIN INFORMATION PARAMETERS
        self.dim                            = domain_information.dim
        self.cell_sizes                     = domain_information.cell_sizes
        self.cell_centers                   = domain_information.cell_centers
        self.nhx, self.nhy, self.nhz        = domain_information.domain_slices_conservatives
        self.nhx_, self.nhy_, self.nhz_     = domain_information.domain_slices_geometry
        self.nh                             = domain_information.nh_conservatives
        self.nh_                            = domain_information.nh_geometry
        self.nhx__, self.nhy__, self.nhz__  = domain_information.domain_slices_conservatives_to_geometry
        self.active_axis_indices            = domain_information.active_axis_indices
        self.inactive_axis_indices          = domain_information.inactive_axis_indices
        self.smallest_cell_size             = jnp.min(jnp.array([self.cell_sizes[i] for i in self.active_axis_indices]))

        # GENERAL LEVELSET
        self.levelset_type                  = numerical_setup["levelset"]["interface_interaction"]
        self.volume_fraction_threshold      = numerical_setup["levelset"]["volume_fraction_threshold"]
        self.narrow_band_cutoff             = numerical_setup["levelset"]["narrow_band_cutoff"]
        self.narrow_band_computations       = numerical_setup["levelset"]["narrow_band_computations"]
        self.mixing_targets                 = numerical_setup["levelset"]["mixing_targets"] if "mixing_targets" in numerical_setup["levelset"].keys() else self.dim

        # EXTENSION PARAMETERS
        self.steps_extension_primes         = numerical_setup["levelset"]["extension"]["steps_primes"]
        self.CFL_extension_primes           = numerical_setup["levelset"]["extension"]["CFL_primes"]
        self.steps_extension_interface      = numerical_setup["levelset"]["extension"]["steps_interface"]
        self.CFL_extension_interface        = numerical_setup["levelset"]["extension"]["CFL_interface"]

        # REINITIALIZATION PARAMETERS
        self.steps_reinitialization         = numerical_setup["levelset"]["reinitialization"]["steps"]
        self.steps_reinitialization_init    = numerical_setup["levelset"]["reinitialization"]["steps_init"]
        self.CFL_reinitialization           = numerical_setup["levelset"]["reinitialization"]["CFL"]
        self.CFL_reinitialization_init      = numerical_setup["levelset"]["reinitialization"]["CFL_init"]
        self.interval_reinitialization      = numerical_setup["levelset"]["reinitialization"]["interval"]
        self.cut_cell_reinitialization      = numerical_setup["levelset"]["reinitialization"]["cut_cell"]

        # ACTIVE PHYSICAL FLUXES
        self.is_viscous_flux                = numerical_setup["active_physics"]["is_viscous_flux"]
        self.is_surface_tension             = numerical_setup["active_physics"]["is_surface_tension"]

        index_pairs = [(0,1), (0,2), (1,2)]
        self.index_pairs_mixing = [] 
        for pair in index_pairs:
            if pair[0] in self.active_axis_indices and pair[1] in self.active_axis_indices:
                self.index_pairs_mixing.append(pair)

    def compute_cut_cell_mask(self, volume_fraction: jnp.ndarray) -> jnp.ndarray:
        """Computes the cut cell mask, i.e., cells where the volume fraction is > 0.0 and < 1.0

        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :return: Cut cell mask
        :rtype: jnp.ndarray
        """
        mask_cut_cells = jnp.where( (volume_fraction > 0.0) & (volume_fraction < 1.0), 1, 0)
        return mask_cut_cells

    @partial(jax.jit, static_argnums=(0))
    def compute_masks(self, levelset: jnp.ndarray,
            volume_fraction: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the real mask and the cut cell mask

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :return: Tuple containing the real mask and the cut cell mask
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        if self.levelset_type == "FLUID-FLUID":
            mask_positive   = jnp.where( levelset[self.nhx__, self.nhy__, self.nhz__] > 0.0, 1, 0 )
            mask_negative   = jnp.where( levelset[self.nhx__, self.nhy__, self.nhz__] < 0.0, 1, 0 )
            mask_cut_cells  = self.compute_cut_cell_mask(volume_fraction)
            mask_positive   = jnp.maximum(mask_positive, mask_cut_cells)
            mask_negative   = jnp.maximum(mask_negative, mask_cut_cells)
            mask_real       = jnp.stack([mask_positive, mask_negative], axis=0)
        else:
            mask_positive   = jnp.where( levelset[self.nhx__, self.nhy__, self.nhz__] > 0.0, 1, 0 )
            mask_cut_cells  = self.compute_cut_cell_mask(volume_fraction)
            mask_real       = jnp.maximum(mask_positive, mask_cut_cells)
        return mask_real, mask_cut_cells

    def compute_volume_fraction_and_apertures(self, levelset: jnp.ndarray) -> Tuple[jnp.ndarray, List]:
        """Computes the volume fraction and apertures via linear interface reconstruction

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :return: Tuple containing the volume fraction buffer and the aperture buffers
        :rtype: Tuple[jnp.ndarray, List]
        """
        volume_fraction, apertures = self.geometry_calculator.linear_interface_reconstruction(levelset)
        return volume_fraction, apertures

    def extend_primes(self, cons: jnp.ndarray, primes: jnp.ndarray, levelset: jnp.ndarray,
            volume_fraction: jnp.ndarray, current_time: float,
            mask_small_cells: jnp.ndarray = None) -> Tuple[jnp.ndarray, float]:
        """Extends the primitives from the real fluid cells into the ghost fluid cells within the narrow_band_compute.
        Subsequently, the corresponding conservatives are computed from the extended primitives.

        :param cons: Buffer of conservative variables
        :type cons: jnp.ndarray
        :param primes: Buffer of primitive variables
        :type primes: jnp.ndarray
        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :param current_time: Current physical simulation time
        :type current_time: float
        :param mask_small_cells: Mask indicating small negative cells, defaults to None
        :type mask_small_cells: jnp.ndarray, optional
        :return: Tuple of primitive and conservative buffer and maximum extension residual
        :rtype: Tuple[jnp.ndarray, float]
        """
        
        # GEOMETRICAL QUANTITIES - WE EXTEND INTO GHOST CELLS PLUS SMALL CELLS INSIDE THE NARROW BAND
        normal              = self.geometry_calculator.compute_normal(levelset)[...,self.nhx_,self.nhy_,self.nhz_]
        normal_extend       = jnp.stack([-normal, normal], axis=1)  if self.levelset_type == "FLUID-FLUID" else -normal
        mask_real, _        = self.compute_masks(levelset, volume_fraction)
        mask_ghost          = 1 - mask_real[...,self.nhx_,self.nhy_,self.nhz_]
        mask_narrow_band    = jnp.where(jnp.abs(levelset[self.nhx,self.nhy,self.nhz])/self.smallest_cell_size < self.narrow_band_computations, 1, 0)
        mask_extend         = jnp.maximum(mask_ghost * mask_narrow_band, mask_small_cells) if mask_small_cells != None else mask_ghost * mask_narrow_band

        # EXTEND PRIMES
        primes, max_residual = self.extender_primes.extend(primes, normal_extend, mask_extend, self.CFL_extension_primes, self.steps_extension_primes)

        # VELOCITY TREATMENT FOR SOLIDS
        if self.levelset_type in ["FLUID-SOLID-STATIC", "FLUID-SOLID-DYNAMIC"]:
            if self.is_viscous_flux:
                primes = primes.at[1:4, self.nhx, self.nhy, self.nhz].mul(jnp.where(mask_ghost == 1.0, -1.0, 1.0))
            else:
                velocities  = primes[1:4,...,self.nhx,self.nhy,self.nhz] * mask_ghost
                velocities  = velocities - normal * 2 * jnp.einsum('ijkl, ijkl -> jkl', normal, velocities)
                primes      = primes.at[1:4, ..., self.nhx, self.nhy, self.nhz].mul(mask_real[...,self.nhx_,self.nhy_,self.nhz_])
                primes      = primes.at[1:4, ..., self.nhx, self.nhy, self.nhz].add(velocities)
        if self.levelset_type == "FLUID-SOLID-DYNAMIC":
            interface_velocity = self.compute_solid_interface_velocity(current_time)
            primes = primes.at[1:4, self.nhx, self.nhy, self.nhz].add(mask_ghost * interface_velocity)

        # CUT OFF PRIMES IN GHOST CELLS IN WHICH WE DO NOT EXTEND, I.E. ALL GHOST CELLS OUTSIDE THE NARROW BAND
        mask_cut_off = (1 - mask_narrow_band) * mask_ghost
        primes = primes.at[...,self.nhx,self.nhy,self.nhz].mul(1 - mask_cut_off)
        primes = primes.at[[0,4],...,self.nhx,self.nhy,self.nhz].add(mask_cut_off * self.eps)

        # UPDATE CONSERVATIVES IN EXTENDED CELLS, I.E. GHOST CELLS PLUS SMALL CELLS INSIDE THE NARROW BAND
        cons_in_extend = get_conservatives_from_primitives(primes[...,self.nhx,self.nhy,self.nhz], self.material_manager) * mask_extend
        cons = cons.at[...,self.nhx,self.nhy,self.nhz].mul(1 - mask_extend)
        cons = cons.at[...,self.nhx,self.nhy,self.nhz].add(cons_in_extend)
        return cons, primes, max_residual
        
    def transform_to_conservatives(self, cons: jnp.ndarray, volume_fraction: jnp.ndarray) -> jnp.ndarray:
        """Transforms the volume-averaged conservatives to actual conservatives that can be integrated
        according to the volume fraction.

        :param cons: Buffer of conservative variables
        :type cons: jnp.ndarray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :return: Buffer of actual conservative variables
        :rtype: jnp.ndarray
        """
        if self.levelset_type == "FLUID-FLUID":
            volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
        cons = cons.at[...,self.nhx,self.nhy,self.nhz].mul(volume_fraction[...,self.nhx_,self.nhy_,self.nhz_])
        return cons

    def transform_to_volume_averages(self, cons: jnp.ndarray, volume_fraction: jnp.ndarray) -> jnp.ndarray:
        """Transforms the mixed conservatives to volume-averaged conservatives.
        Emtpy cells are filled with eps. Negative small cells (which may occur after mixing) are filled with eps.
        We extend the integrated primitive state into the negative small cells cells.

        :param cons: Buffer of mixed conservative variables
        :type cons: jnp.ndarray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :return: Volume-averaged conservatives
        :rtype: jnp.ndarray
        """

        if self.levelset_type == "FLUID-FLUID": 
            volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)

        # TRANSFORM TO VOLUME AVERAGES
        mask = jnp.where(volume_fraction[...,self.nhx_,self.nhy_,self.nhz_] == 0.0, 1, 0)
        cons = cons.at[...,self.nhx,self.nhy,self.nhz].mul(1.0/(volume_fraction[...,self.nhx_,self.nhy_,self.nhz_] + mask * self.eps))

        # SET NEGATIVE AND EMPTY CELLS TO EPS - THEY ARE EITHER GHOST CELLS OR VERY SMALL REAL CELLS - WE EXTEND INTO BOTH
        mask = jnp.maximum( mask, jnp.where((cons[0,...,self.nhx,self.nhy,self.nhz] <= 0.0) | (cons[4,...,self.nhx,self.nhy,self.nhz] <= 0.0), 1, 0) ) 
        cons = cons.at[...,self.nhx,self.nhy,self.nhz].mul( 1 - mask )
        cons = cons.at[...,self.nhx,self.nhy,self.nhz].add( mask * self.eps )

        return cons

    def weight_cell_face_flux_xi(self, flux_xi: jnp.ndarray, apertures: jnp.ndarray) -> jnp.ndarray:
        """Weights the cell face fluxes according to the apertures.

        :param flux_xi: Cell face flux at xi
        :type flux_xi: jnp.ndarray
        :param apertures: Aperture buffer
        :type apertures: jnp.ndarray
        :return: Weighted cell face flux at xi
        :rtype: jnp.ndarray
        """
        if self.levelset_type == "FLUID-FLUID": 
            apertures = jnp.stack([apertures, 1.0 - apertures], axis=0)
        flux_xi *= apertures[...,self.nhx_,self.nhy_,self.nhz_]
        return flux_xi

    def weight_volume_force(self, volume_force: jnp.ndarray, volume_fraction: jnp.ndarray) -> jnp.ndarray:
        """Weights the volume forces according to the volume fraction.

        :param volume_force: Volume force buffer
        :type volume_force: jnp.ndarray
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :return: Weighted volume force
        :rtype: jnp.ndarray
        """
        if self.levelset_type == "FLUID-FLUID":
            volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
        volume_force *= volume_fraction[...,self.nhx_,self.nhy_,self.nhz_]
        return volume_force

    @partial(jax.jit, static_argnums=(0))
    def compute_solid_interface_velocity(self, current_time: float) -> jnp.ndarray:
        """Computes the interface velocity for FLUID-SOLID-DYNAMIC interface interaction.

        :param current_time: Current physical simulation time
        :type current_time: float
        :return: Interface velocity buffer
        :rtype: jnp.ndarray
        """
        interface_velocity = self.interface_quantity_computer.compute_solid_interface_velocity(current_time)
        return interface_velocity

    @partial(jax.jit, static_argnums=(0))
    def compute_interface_quantities(self, primes: jnp.ndarray, levelset: jnp.ndarray,
            volume_fraction: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Computes interface velocity and pressure for FLUID-FLUID interface interaction and
        extends the values into the narrow_band_compute.

        :param primes: Buffer of primitive variables
        :type primes: jnp.ndarray
        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param volume_fraction: Volume fractio buffer
        :type volume_fraction: jnp.ndarray
        :return: Tuple of interface velocity, interface pressure and maximum residual of the extension
        :rtype: Tuple[jnp.ndarray, jnp.ndarray, float]
        """

        # GEOMTRICAL QUANTITIES
        normal                  = self.geometry_calculator.compute_normal(levelset)
        curvature               = self.geometry_calculator.compute_curvature(levelset) if self.is_surface_tension else None
        mask_cut_cells          = self.compute_cut_cell_mask(volume_fraction)

        # COMPUTE INTERFACE QUANTITIES
        interface_velocity, interface_pressure  = self.interface_quantity_computer.solve_interface_interaction(primes, normal, curvature)
        interface_quantities                    = jnp.vstack([jnp.expand_dims(interface_velocity, axis=0), interface_pressure])

        # EXTEND INTERFACE QUANTITIES
        inverse_cut_cell_mask                   = 1 - mask_cut_cells[...,self.nhx_,self.nhy_,self.nhz_]
        mask_narrow_band                        = jnp.where(jnp.abs(levelset[self.nhx,self.nhy,self.nhz])/self.smallest_cell_size < self.narrow_band_computations, 1, 0)
        mask_extend                             = inverse_cut_cell_mask * mask_narrow_band
        normal_extend                           = normal[...,self.nhx_,self.nhy_,self.nhz_] * jnp.sign(levelset[self.nhx,self.nhy,self.nhz])
        interface_quantities, max_residual      = self.extender_interface.extend(interface_quantities, normal_extend, mask_extend, self.CFL_extension_interface, self.steps_extension_interface)

        # CUT OFF NARROW BAND
        interface_quantities = interface_quantities[...,self.nhx_,self.nhy_,self.nhz_] * mask_narrow_band

        return interface_quantities[0], interface_quantities[1:], max_residual

    def compute_interface_flux_xi(self, primes: jnp.ndarray, levelset: jnp.ndarray, interface_velocity: jnp.ndarray,
            interface_pressure: Union[jnp.ndarray, None], volume_fraction: jnp.ndarray, apertures: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes the interface flux depending on the present interface interaction type.

        :param primes: Buffer of primitive variables
        :type primes: jnp.ndarray
        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param interface_velocity: Interface velocity buffer
        :type interface_velocity: jnp.ndarray
        :param interface_pressure: Interface pressure buffer
        :type interface_pressure: Union[jnp.ndarray, None]
        :param volume_fraction: Volume fraction buffer
        :type volume_fraction: jnp.ndarray
        :param apertures: Aperture buffer
        :type apertures: jnp.ndarray
        :param axis: Current axis
        :type axis: int
        :return: Interface fluxes
        :rtype: jnp.ndarray
        """
        normal              = self.geometry_calculator.compute_normal(levelset)
        interface_flux_xi   = self.interface_flux_computer.compute_interface_flux_xi(primes, interface_velocity, interface_pressure, volume_fraction, apertures, normal, axis)
        return interface_flux_xi

    def compute_levelset_advection_rhs(self, levelset: jnp.ndarray, interface_velocity: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Computes the right-hand-side of the levelset advection equation.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param interface_velocity: Interface velocity buffer
        :type interface_velocity: jnp.ndarray
        :param axis: Current axis
        :type axis: int
        :return: right-hand-side contribution for current axis
        :rtype: jnp.ndarray
        """
        
        # GEOMETRICAL QUANTITIES
        normal              = self.geometry_calculator.compute_normal(levelset)
        if self.levelset_type == "FLUID-FLUID":
            mask_narrow_band = jnp.where(jnp.abs(levelset[self.nhx,self.nhy,self.nhz])/self.smallest_cell_size < self.narrow_band_computations, 1, 0)
        else:
            mask_narrow_band = jnp.ones_like(levelset[self.nhx,self.nhy,self.nhz])

        # DERIVATIVE
        derivative_L = self.levelset_advection_stencil.derivative_xi(levelset, 1.0, axis, 0)
        derivative_R = self.levelset_advection_stencil.derivative_xi(levelset, 1.0, axis, 1)

        # UPWINDING
        velocity            = interface_velocity * normal[axis,self.nhx_,self.nhy_,self.nhz_] if self.levelset_type == "FLUID-FLUID" else interface_velocity[axis]
        mask_L              = jnp.where(velocity >= 0.0, 1.0, 0.0)
        mask_R              = 1.0 - mask_L

        # SUM RHS
        rhs_contribution  = - velocity * (mask_L * derivative_L + mask_R * derivative_R) / self.cell_sizes[axis]
        rhs_contribution *= mask_narrow_band

        return rhs_contribution

    # @partial(jax.jit, static_argnums=(0, 2))
    def reinitialize(self, levelset: jnp.ndarray, initializer: bool) -> Tuple[jnp.ndarray, float]:
        """Reinitializes the levelset buffer and subsequently applies cut off 
        to values that lie outside the narrow_band_cutoff. If the initializer flag is True,
        cut cells are always reinitialized. This is required to initialize
        levelset fields that are provided as no signed distance functions, e.g., an ellipse.
        If cut_cell_reinitialization is True (specified in numerical setup), then cut cells are also reinitialized.
        The residual is only computed in the narrow_band_computations.

        :param levelset: Levelset buffer
        :type levelset: jnp.ndarray
        :param initializer: Bool indicating if the reinitialization is performed in the
            jaxfluids initializer or the jaxfluids simulation manager
        :type initializer: bool
        :return: Reinitialized levelset buffer and the corresponding maximum residual
        :rtype: Tuple[jnp.ndarray, float]
        """
        
        # TODO SPLIT UP INITIAL REINITIALIZATION AND SET CUT OFF
        
        # REINITIALIZE
        if initializer:
            mask_reinitialize       = jnp.ones_like(levelset[self.nhx,self.nhy,self.nhz], dtype=jnp.uint32)
            mask_residual           = jnp.ones_like(levelset[self.nhx,self.nhy,self.nhz], dtype=jnp.uint32)
            levelset, max_residual  = self.levelset_reinitializer_init.reinitialize(levelset, mask_reinitialize, mask_residual, self.CFL_reinitialization_init, self.steps_reinitialization_init)
            # levelset, levelset_out  = self.levelset_reinitializer_init.reinitialize_debug(levelset, mask_reinitialize, mask_residual, self.CFL_reinitialization_init, self.steps_reinitialization_init)
        else:
            mask_reinitialize       = jnp.ones_like(levelset[self.nhx,self.nhy,self.nhz], dtype=jnp.uint32) \
                    if self.cut_cell_reinitialization else \
                    jnp.where( jnp.abs(levelset[self.nhx,self.nhy,self.nhz])/self.smallest_cell_size > jnp.sqrt(2.0)/2, 1, 0 )
            mask_residual           = jnp.where(jnp.abs(levelset[self.nhx,self.nhy,self.nhz]/self.smallest_cell_size) < self.narrow_band_computations, 1, 0)
            mask_residual           = mask_residual * mask_reinitialize
            levelset, max_residual  = self.levelset_reinitializer.reinitialize(levelset, mask_reinitialize, mask_residual, self.CFL_reinitialization, self.steps_reinitialization)

        # CUT OFF MASKS
        mask_cut_off_positive = jnp.where(levelset[self.nhx,self.nhy,self.nhz]/self.smallest_cell_size > self.narrow_band_cutoff, 1, 0)
        mask_cut_off_negative = jnp.where(levelset[self.nhx,self.nhy,self.nhz]/self.smallest_cell_size < -self.narrow_band_cutoff, 1, 0)

        # SET CUTOFF VALUES
        levelset = levelset.at[self.nhx,self.nhy,self.nhz].mul(1 - mask_cut_off_positive)
        levelset = levelset.at[self.nhx,self.nhy,self.nhz].mul(1 - mask_cut_off_negative)
        levelset = levelset.at[self.nhx,self.nhy,self.nhz].add(mask_cut_off_positive * self.narrow_band_cutoff * self.smallest_cell_size)
        levelset = levelset.at[self.nhx,self.nhy,self.nhz].add(-mask_cut_off_negative * self.narrow_band_cutoff * self.smallest_cell_size)

        return levelset, max_residual

    def mixing(self, cons: jnp.ndarray, levelset_new: jnp.ndarray, volume_fraction_new: jnp.ndarray,
            volume_fraction_old: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Performs a mixing procedure on the integrated conservatives enabling stable integration of
        small cut cells with the CFL criterion for full cells. For small cells that are in the process of vanishing 
        (becoming ghost cells), this procedure may produce negative mass/energy. To prevent numerical instability,
        we track these cells with the mask_small_cells buffer and perform a prime extension into these cells.

        :param cons: Buffer of integrated conservative variables
        :type cons: jnp.ndarray
        :param levelset_new: Integrated levelset buffer
        :type levelset_new: jnp.ndarray
        :param volume_fraction_new: Integrated volume fraction buffer
        :type volume_fraction_new: jnp.ndarray
        :param volume_fraction_old: Volume fraction buffer of previous time step (RK stage)
        :type volume_fraction_old: jnp.ndarray
        :return: Tuple containing the mixed conservatives and a mask indicating small cells with negative energy/mass.
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """

        # GEOMETRICAL QUANTITIES
        normal                              = self.geometry_calculator.compute_normal(levelset_new)
        if self.levelset_type == "FLUID-FLUID":
            normal                              = jnp.stack([normal, -normal], axis=1)
            volume_fraction_new                 = jnp.stack([volume_fraction_new, 1.0 - volume_fraction_new], axis=0)
            volume_fraction_old                 = jnp.stack([volume_fraction_old, 1.0 - volume_fraction_old], axis=0)
        normal_sign                             = jnp.sign(normal)

        # CREATE SOURCE MASK - SMALL CELLS, NEWLY CREATED, AND VANISHED CELLS
        small_cells         = jnp.where((volume_fraction_new < self.volume_fraction_threshold) & (volume_fraction_new > 0.0), 1, 0)
        newly_created_cells = jnp.where((volume_fraction_new > 0.0) & (volume_fraction_old == 0.0), 1, 0)
        vanished_cells      = jnp.where((volume_fraction_new == 0.0) & (volume_fraction_old > 0.0), 1, 0)
        mask_source         = jnp.maximum(small_cells, jnp.maximum(vanished_cells, newly_created_cells))

        # MIXING WEIGHTS II
        mixing_weight_sum = 0.0
        mixing_weight_ii_list = []
        for i in self.active_axis_indices:
            mixing_weight_ii                    = jnp.square(normal[i])
            volume_fraction_target_at_source    = move_target_to_source_ii(volume_fraction_new, normal_sign, i)
            mixing_weight_ii                    *= volume_fraction_target_at_source
            mixing_weight_sum                   += mixing_weight_ii
            mixing_weight_ii_list.append( mixing_weight_ii )
        mixing_weight_ii = jnp.stack(mixing_weight_ii_list, axis=0)

        # MIXING WEIGHTS IJ
        if self.dim > 1 and self.mixing_targets > 1:
            mixing_weight_ij_list = []
            for (i, j) in self.index_pairs_mixing:
                mixing_weight_ij                    = jnp.abs(normal[i]*normal[j])
                volume_fraction_target_at_source    = move_target_to_source_ij(volume_fraction_new, normal_sign, i, j)
                mixing_weight_ij                    *= volume_fraction_target_at_source
                mixing_weight_sum                   += mixing_weight_ij
                mixing_weight_ij_list.append( mixing_weight_ij )
            mixing_weight_ij = jnp.stack(mixing_weight_ij_list, axis=0)

        # MIXING WEIGHTS IJK
        if self.dim == 3 and self.mixing_targets == 3:
            mixing_weight_ijk                       = jnp.abs(normal[0]*normal[1]*normal[2])**(2/3)
            volume_fraction_target_at_source        = move_target_to_source_ijk(volume_fraction_new, normal_sign)
            mixing_weight_ijk                       *= volume_fraction_target_at_source
            mixing_weight_sum                       += mixing_weight_ijk

        # NORMALIZATION
        mixing_weight_ii  = mixing_weight_ii/(mixing_weight_sum + self.eps)
        mixing_weight_ij  = mixing_weight_ij/(mixing_weight_sum + self.eps) if self.dim > 1 and self.mixing_targets > 1 else None
        mixing_weight_ijk = mixing_weight_ijk/(mixing_weight_sum + self.eps) if self.dim == 3 and self.mixing_targets == 3 else None

        Mixing_fluxes_source = []
        Mixing_fluxes_target = []
        
        # MIXING CONTRIBUTIONS II
        for i, axis in enumerate(self.active_axis_indices):
            # MOVE TARGET VALUES TO SOURCE POSITION
            volume_fraction_target_at_source = move_target_to_source_ii(volume_fraction_new, normal_sign, axis)
            cons_target_at_source = move_target_to_source_ii(cons[...,self.nhx__,self.nhy__,self.nhz__], normal_sign, axis)
            # MIXING FLUXES
            M_xi_source = mixing_weight_ii[i]/(volume_fraction_new * mixing_weight_ii[i] + volume_fraction_target_at_source + self.eps) * (cons_target_at_source * volume_fraction_new - cons[...,self.nhx__,self.nhy__,self.nhz__] * volume_fraction_target_at_source) * mask_source
            M_xi_target = move_source_to_target_ii(-M_xi_source, normal_sign, axis)
            Mixing_fluxes_source.append(M_xi_source)
            Mixing_fluxes_target.append(M_xi_target)

        # MIXING CONTRIBUTIONS IJ
        if self.dim > 1 and self.mixing_targets > 1:
            for k, (axis_i, axis_j) in enumerate(self.index_pairs_mixing):
                # MOVE TARGET VALUES TO SOURCE POSITION
                volume_fraction_target_at_source = move_target_to_source_ij(volume_fraction_new, normal_sign, axis_i, axis_j)
                cons_target_at_source = move_target_to_source_ij(cons[...,self.nhx__,self.nhy__,self.nhz__], normal_sign, axis_i, axis_j)
                # MIXING FLUXES
                M_xi_source = mixing_weight_ij[k]/(volume_fraction_new * mixing_weight_ij[k] + volume_fraction_target_at_source + self.eps) * (cons_target_at_source * volume_fraction_new - cons[...,self.nhx__,self.nhy__,self.nhz__] * volume_fraction_target_at_source) * mask_source
                M_xi_target = move_source_to_target_ij(-M_xi_source, normal_sign, axis_i, axis_j)
                Mixing_fluxes_source.append(M_xi_source)
                Mixing_fluxes_target.append(M_xi_target)

        # MIXING CONTRIBUTIONS IJK
        if self.dim == 3 and self.mixing_targets == 3:
            volume_fraction_target_at_source = move_target_to_source_ijk(volume_fraction_new, normal_sign)
            cons_target_at_source = move_source_to_target_ijk(cons[...,self.nhx__,self.nhy__,self.nhz__], normal_sign)
            # MIXING FLUXES
            M_xi_source = mixing_weight_ijk/(volume_fraction_new * mixing_weight_ijk + volume_fraction_target_at_source + self.eps) * (cons_target_at_source * volume_fraction_new - cons[...,self.nhx__,self.nhy__,self.nhz__] * volume_fraction_target_at_source) * mask_source
            M_xi_target = move_source_to_target_ijk(-M_xi_source, normal_sign)
            Mixing_fluxes_source.append(M_xi_source)
            Mixing_fluxes_target.append(M_xi_target)

        cons = cons.at[...,self.nhx,self.nhy,self.nhz].add(sum(Mixing_fluxes_source)[...,self.nhx_,self.nhy_,self.nhz_] + sum(Mixing_fluxes_target)[...,self.nhx_,self.nhy_,self.nhz_])

        mask_small_cells = jnp.where((cons[0,...,self.nhx,self.nhy,self.nhz] < 0.0) | (cons[4,...,self.nhx,self.nhy,self.nhz] < 0.0), 1, 0) * jnp.where((volume_fraction_new[...,self.nhx_,self.nhy_,self.nhz_] != 0.0), 1, 0)
        
        return cons, mask_small_cells

    def compute_primitives_from_conservatives_in_real_fluid(self, cons: jnp.ndarray, primes: jnp.ndarray,
            levelset: jnp.ndarray, volume_fraction: jnp.ndarray, mask_small_cells: jnp.ndarray) -> jnp.ndarray:
        """Computes the primitive variables from the mixed conservatives within the real fluid.

        :param cons: Buffer of primitive variables
        :type cons: jnp.ndarray
        :param primes: _description_
        :type primes: jnp.ndarray
        :param levelset: _description_
        :type levelset: jnp.ndarray
        :param volume_fraction: _description_
        :type volume_fraction: jnp.ndarray
        :param mask_small_cells: _description_
        :type mask_small_cells: jnp.ndarray
        :return: _description_
        :rtype: jnp.ndarray
        """
        # WE COMPUTE PRIMES ONLY IN REAL CELLS IN WHICH WE DO NOT EXTEND, I.E. REAL CELLS MINUS SMALL CELLS
        mask_real, _    = self.compute_masks(levelset, volume_fraction)
        mask_real       = mask_real[...,self.nhx_,self.nhy_,self.nhz_] * (1 - mask_small_cells)
        mask_ghost      = 1 - mask_real
        primes_in_real  = get_primitives_from_conservatives(cons[...,self.nhx,self.nhy,self.nhz], self.material_manager) * mask_real
        primes          = primes.at[...,self.nhx,self.nhy,self.nhz].mul(mask_ghost)
        primes          = primes.at[...,self.nhx,self.nhy,self.nhz].add(primes_in_real)
        return primes