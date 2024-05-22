from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
import jax

from jaxfluids.config import precision
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager

from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup, PositivitySetup
from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup, THINCSetup

from jaxfluids.solvers.convective_fluxes.high_order_godunov import HighOrderGodunov
from jaxfluids.solvers.riemann_solvers import HLLC, signal_speed_Einfeldt
from jaxfluids.stencils import WENO1


class PositivityLimiterFlux:
    """The PositivityLimiterFlux class implementes functionality
    which ensures that numerical fluxes lead to admissible states
    after temporal integration.

    #TODO rewrite such that integrate_positivity is called at a single
    place for all types of equations, only check_admissability() then
    distinguishes between equation types

    #TODO 'invert' definition of theta, such that counter = sum(theta) 
    and not sum(1 - theta)
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            numerical_setup: NumericalSetup,
            diffuse_interface_handler: DiffuseInterfaceHandler = None,
            ) -> None:
        self.eps = precision.get_flux_limiter_eps()
        
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.material_manager = material_manager
        self.domain_information = domain_information
        
        positivity_setup = numerical_setup.conservatives.positivity
        self.positivity_flux_limiter = positivity_setup.flux_limiter
        
        # SETTING UP HIGH-ORDER GODUNOV FLUX SOLVER WITH WENO1
        _convective_fluxes_setup = ConvectiveFluxesSetup(
            convective_solver=HighOrderGodunov,
            riemann_solver=HLLC,
            signal_speed=signal_speed_Einfeldt,
            reconstruction_stencil=WENO1,
            split_reconstruction=None,
            flux_splitting=None,
            reconstruction_variable="PRIMITIVE", 
            frozen_state=None,
            iles_setup=None,
            catum_setup=None)

        _positivity_setup = PositivitySetup(
            flux_limiter=None,
            is_interpolation_limiter=False,
            is_thinc_interpolation_limiter=False,
            is_volume_fraction_limiter=False,
            is_acdi_flux_limiter=False,
            is_logging=False)
        
        _thinc_setup = THINCSetup(
            is_thinc_reconstruction=False,
            thinc_type=None,
            interface_treatment=None,
            interface_projection=None,
            interface_parameter=None,
            volume_fraction_threshold=None)

        _diffuse_interface_setup = DiffuseInterfaceSetup(
            None, None, False, None,
            None, _thinc_setup, None)

        self.convective_flux_solver_positive = HighOrderGodunov(
            convective_fluxes_setup=_convective_fluxes_setup,
            positivity_setup=_positivity_setup,
            diffuse_interface_setup=_diffuse_interface_setup,
            material_manager=material_manager,
            domain_information=domain_information,
            equation_manager=equation_manager,
            diffuse_interface_handler=None,
            positivity_handler=None)
        
        self.equation_type = self.equation_information.equation_type
        self.partition_type = "UNIFORM"

        self.mass_ids = self.equation_information.mass_ids
        self.mass_slices = self.equation_information.mass_slices
        self.vel_ids = self.equation_information.velocity_ids
        self.vel_slices = self.equation_information.velocity_slices
        self.energy_ids = self.equation_information.energy_ids
        self.energy_slices = self.equation_information.energy_slices
        self.vf_ids = self.equation_information.vf_ids
        self.vf_slices = self.equation_information.vf_slices

        nh_cons = domain_information.nh_conservatives
        self.dim = domain_information.dim
        self.active_axes_indices = domain_information.active_axes_indices
        self.is_parallel = domain_information.is_parallel
        self.split_factors = domain_information.split_factors
        self.is_mesh_stretching = domain_information.is_mesh_stretching
        self.is_parallel = domain_information.is_parallel
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nhx__, self.nhy__, self.nhz__ = domain_information.domain_slices_conservatives_to_geometry

        # SLICES FOR MINUS, PLUS
        if self.equation_information.levelset_model:
            nh_geometry = domain_information.nh_geometry
            self.cons_positivity_slices = [ 
                [jnp.s_[...,nh_geometry:-nh_geometry+1,self.nhy_,self.nhz_], jnp.s_[...,nh_geometry-1:-nh_geometry,self.nhy_,self.nhz_]],
                [jnp.s_[...,self.nhx_,nh_geometry:-nh_geometry+1,self.nhz_], jnp.s_[...,self.nhx_,nh_geometry-1:-nh_geometry,self.nhz_]],
                [jnp.s_[...,self.nhx_,self.nhy_,nh_geometry:-nh_geometry+1], jnp.s_[...,self.nhx_,self.nhy_,nh_geometry-1:-nh_geometry]],
            ]
        else:
            self.cons_positivity_slices = [ 
                [jnp.s_[...,nh_cons:-nh_cons+1,self.nhy,self.nhz], jnp.s_[...,nh_cons-1:-nh_cons,self.nhy,self.nhz]],
                [jnp.s_[...,self.nhx,nh_cons:-nh_cons+1,self.nhz], jnp.s_[...,self.nhx,nh_cons-1:-nh_cons,self.nhz]],
                [jnp.s_[...,self.nhx,self.nhy,nh_cons:-nh_cons+1], jnp.s_[...,self.nhx,self.nhy,nh_cons-1:-nh_cons]],
            ]

        self.mesh_positivity_slices = [ 
            [jnp.s_[...,nh_cons:-nh_cons+1,:,:], jnp.s_[...,nh_cons-1:-nh_cons,:,:]],
            [jnp.s_[...,:,nh_cons:-nh_cons+1,:], jnp.s_[...,:,nh_cons-1:-nh_cons,:]],
            [jnp.s_[...,:,:,nh_cons:-nh_cons+1], jnp.s_[...,:,:,nh_cons-1:-nh_cons]],
        ]

    def compute_positivity_preserving_flux(
            self, 
            flux_xi_convective: Array, 
            u_hat_xi: Array,
            alpha_hat_xi: Array,
            primitives: Array,
            conservatives: Array, 
            levelset: Array,
            volume_fraction: Array, 
            apertures: Array, 
            curvature: Array, 
            physical_timestep_size: float, 
            axis: int, 
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None
            ) -> Tuple[Array, Array]:
        """Computes a positivity-preserving flux if the current convective flux 
        violates positivity. Two positivity-preserving methdods are implemented:

        1) Hu, Adams & Shu (HAS)
        2) Simplified version of HAS 
        3) Wong et al. NASA

        Consists of two main steps:

        1)  a) Pseudo-integration
            b) Check on density/partial density/volume_fraction
            c) Flux limiting
        2)  a) Pseudo-integration
            b) Check on pressure/internal energy/speed of sound
            c) Flux limiting

        :param flux_xi_convective: Buffer of the convectivce flux in xi direction.
        :type flux_xi_convective: Array
        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param physical_timestep_size: Current physical time step size
        :type physical_timestep_size: float
        :param axis: Spatial direction
        :type axis: int
        :param ml_parameters_dict: Dictionary of ML parameters, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary of ML network architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Buffer of positivity-preserving flux
        :rtype: Array
        """
        slice_minus = self.cons_positivity_slices[axis][0]  # i+1 -
        slice_plus = self.cons_positivity_slices[axis][1]   #  i  +

        positivity_count = 0

        one_cell_sizes_halos = self.domain_information.get_device_one_cell_sizes_halos()
        one_cell_sizes_halos_xi = one_cell_sizes_halos[axis]
        lambda_pos = physical_timestep_size * one_cell_sizes_halos_xi * self.dim

        # POSITIVITY FIX BASED ON HU, ADAMS, SHU
        if self.positivity_flux_limiter == "HAS":
            # TODO NEEDS UPDATE

            flux_xi_convective_pos, *_ = \
            self.convective_flux_solver_positive.compute_flux_xi(
                primitives, conservatives,
                axis, ml_parameters_dict,
                ml_networks_dict)

            # FIRST INTEGRATION CHECK - DENSITY
            U_minus = conservatives[slice_minus] + 2 * lambda_pos * flux_xi_convective
            U_minus_LF = conservatives[slice_minus] + 2 * lambda_pos * flux_xi_convective_pos
            theta_minus = jnp.where(U_minus[0] < self.eps.density,
                (self.eps.density - U_minus_LF[0]) / (U_minus[0] - U_minus_LF[0]),
                1.0)

            U_plus = conservatives[slice_plus] - 2 * lambda_pos * flux_xi_convective
            U_plus_LF = conservatives[slice_plus] - 2 * lambda_pos * flux_xi_convective_pos
            theta_plus = jnp.where(U_plus[0] < self.eps.density, 
                (self.eps.density - U_plus_LF[0]) / (U_plus[0] - U_plus_LF[0]), 
                1.0)
            
            theta_rho = jnp.minimum(theta_minus, theta_plus)
            
            flux_xi_convective = (1 - theta_rho) * flux_xi_convective_pos + theta_rho * flux_xi_convective

            # SECOND INTEGRATION CHECK - PRESSURE
            U_minus = conservatives[slice_minus] + 2 * lambda_pos * flux_xi_convective
            W_minus = self.equation_manager.get_primitives_from_conservatives(U_minus)
            W_minus_LF = self.equation_manager.get_primitives_from_conservatives(U_minus_LF)
            theta_minus = jnp.where(W_minus[4] < self.eps.pressure, (self.eps.pressure - W_minus_LF[4]) / (W_minus[4] - W_minus_LF[4]), 1.0)

            U_plus = conservatives[slice_plus] - 2 * lambda_pos * flux_xi_convective
            W_plus = self.equation_manager.get_primitives_from_conservatives(U_plus)
            W_plus_LF = self.equation_manager.get_primitives_from_conservatives(U_plus_LF)
            theta_plus = jnp.where(W_plus[4] < self.eps.pressure, (self.eps.pressure - W_plus_LF[4]) / (W_plus[4] - W_plus_LF[4]), 1.0)
            
            theta_p = jnp.minimum(theta_minus, theta_plus)
            positivity_count = jnp.sum(1 - jnp.minimum(theta_rho, theta_p))
            
            flux_xi_convective = (1 - theta_p) * flux_xi_convective_pos + theta_p * flux_xi_convective

        # SIMPLE - SIMPLIFIED POSITIVITY FIX BASED ON HU ET AL. - BINARY SWITCH BETWEEN HIGH-ORDER AND POSITIVE FLUX
        # NASA   - POSITIVITY FIX ACCORDING TO WONG ET AL. 2021 - BINARY SWITCH BETWEEN HIGH-ORDER AND POSITIVE FLUX
        if self.positivity_flux_limiter in ["SIMPLE", "NASA"]:
            # Evaluate first-order positivity preserving flux
            flux_xi_convective_pos, u_hat_xi_pos, alpha_hat_xi_pos, *_ \
            = self.convective_flux_solver_positive.compute_flux_xi(
                primitives, conservatives, axis, curvature,
                ml_parameters_dict, ml_networks_dict)

            if self.equation_type == "TWO-PHASE-LS":
                # Volume fraction & apertures are stacked such that they have
                # shape = (2,Nx,Ny,Nz)
                volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
                apertures = apertures[...,self.nhx_,self.nhy_,self.nhz_]
                apertures = jnp.stack([apertures, 1.0 - apertures], axis=0)
            elif self.equation_type == "SINGLE-PHASE-SOLID-LS":
                apertures = apertures[...,self.nhx_,self.nhy_,self.nhz_]
            else:
                volume_fraction = apertures = None

            # FIRST INTEGRATION CHECK - DENSITY & PARTIAL DENSITIES
            U_minus, U_plus = self.integrate_positivity(
                conservatives=conservatives,
                primitives=primitives,
                lambda_pos=lambda_pos,
                flux_xi_convective=flux_xi_convective,
                u_hat_xi=u_hat_xi,
                axis=axis,
                volume_fraction=volume_fraction,
                apertures=apertures)

            theta = self.check_admissibility_for_density_and_volume_fraction(
                U_minus, U_plus, apertures)

            flux_xi_convective, u_hat_xi, alpha_hat_xi, positivity_count = \
            self.apply_limiter_to_high_order_flux(flux_xi_convective,
                                                  flux_xi_convective_pos,
                                                  theta, positivity_count,
                                                  u_hat_xi, u_hat_xi_pos,
                                                  alpha_hat_xi, alpha_hat_xi_pos)

            # SECOND INTEGRATION CHECK - PRESSURE & SPEED OF SOUND
            U_minus, U_plus = self.integrate_positivity(
                conservatives=conservatives,
                primitives=primitives,
                lambda_pos=lambda_pos,
                flux_xi_convective=flux_xi_convective,
                u_hat_xi=u_hat_xi,
                axis=axis,
                volume_fraction=volume_fraction,
                apertures=apertures)

            theta = self.check_admissibility_pressure_speed_of_sound(
                U_minus, U_plus, apertures)

            flux_xi_convective, u_hat_xi, alpha_hat_xi, positivity_count = \
            self.apply_limiter_to_high_order_flux(flux_xi_convective,
                                                  flux_xi_convective_pos,
                                                  theta, positivity_count,
                                                  u_hat_xi, u_hat_xi_pos,
                                                  alpha_hat_xi, alpha_hat_xi_pos)

        return flux_xi_convective, u_hat_xi, alpha_hat_xi, positivity_count
    
    def compute_positivity_preserving_sharpening_flux(
            self,
            flux_xi_convective: Array,
            u_hat_xi: Array,
            interface_regularization_flux_xi: Array,
            primitives: Array,
            conservatives: Array,
            physical_timestep_size: float,
            axis: int
            ) -> Tuple[Array, int]:
        positivity_count = 0            

        one_cell_sizes_halos = self.domain_information.get_device_one_cell_sizes_halos()
        one_cell_sizes_halos_xi = one_cell_sizes_halos[axis]
        lambda_pos = physical_timestep_size * one_cell_sizes_halos_xi * self.dim

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            test_flux = flux_xi_convective - interface_regularization_flux_xi
            
            # CHECK PARTIAL DENSITIES AND VOLUME FRACTIONS
            U_minus, U_plus = self.integrate_positivity(
                conservatives=conservatives,
                primitives=primitives,
                lambda_pos=lambda_pos,
                flux_xi_convective=test_flux,
                u_hat_xi=u_hat_xi,
                axis=axis)

            theta = jnp.where(
                (jnp.minimum(U_minus[self.mass_slices], U_plus[self.mass_slices]) < self.eps.density        ).any(axis=0) |
                (jnp.minimum(U_minus[self.vf_slices], U_plus[self.vf_slices])     < self.eps.volume_fraction).any(axis=0) |
                (jnp.maximum(U_minus[self.vf_slices], U_plus[self.vf_slices])     > 1.0 - self.eps.volume_fraction).any(axis=0),
                0, 1)
            
            positivity_count += jnp.sum(1 - theta)
        
            interface_regularization_flux_xi = (1 - theta) * jnp.zeros_like(interface_regularization_flux_xi) \
                + theta * interface_regularization_flux_xi
            test_flux = flux_xi_convective - interface_regularization_flux_xi

            # CHECK PSEUDO SPEED OF SOUND
            U_minus, U_plus = self.integrate_positivity(
                conservatives=conservatives,
                primitives=primitives,
                lambda_pos=lambda_pos,
                flux_xi_convective=test_flux,
                u_hat_xi=u_hat_xi,
                axis=axis)

            W_minus = self.equation_manager.get_primitives_from_conservatives(U_minus)
            W_plus = self.equation_manager.get_primitives_from_conservatives(U_plus)

            pb_minus = self.material_manager.get_background_pressure(W_minus[self.vf_slices])
            pb_plus = self.material_manager.get_background_pressure(W_plus[self.vf_slices])

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            theta = jnp.where(
                (jnp.minimum(W_minus[self.energy_ids] + pb_minus, W_plus[self.energy_ids] + pb_plus) < self.eps.pressure),
                0, 1)

            # print("SUM THETA ALPHA/RHOALPHA = {}".format(jnp.sum(1 - theta)))

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # rho_minus  = self.material_manager.get_density(U_minus)
            # rho_plus   = self.material_manager.get_density(U_plus)
            # rhoe_minus = rho_minus * self.material_manager.get_specific_energy(W_minus[self.energy_ids], rho=rho_minus, alpha_i=W_minus[self.vf_slices])
            # rhoe_plus  = rho_plus  * self.material_manager.get_specific_energy(W_plus[self.energy_ids], rho=rho_plus, alpha_i=W_plus[self.vf_slices])

            # theta   = jnp.where( 
            #     (jnp.minimum(rhoe_minus - pb_minus, rhoe_plus - pb_plus) < self.eps.pressure), 0, 1
            # )

            positivity_count += jnp.sum(1 - theta)

            interface_regularization_flux_xi = \
                (1 - theta) * jnp.zeros_like(interface_regularization_flux_xi) \
                + theta * interface_regularization_flux_xi
        
        else:
            raise NotImplementedError

        return interface_regularization_flux_xi, positivity_count

    def integrate_positivity(
            self,
            conservatives: Array,
            primitives: Array,
            lambda_pos: jnp.float32,
            flux_xi_convective: Array,
            u_hat_xi: Array,
            axis: jnp.int32,
            volume_fraction: Array = None,
            apertures: Array = None
            ) -> Tuple[Array, Array]:
        """Performs pseudo-integration for the positivity check.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param lambda_pos: _description_
        :type lambda_pos: jnp.float32
        :param flux_xi_convective: _description_
        :type flux_xi_convective: Array
        :param u_hat_xi: _description_
        :type u_hat_xi: Array
        :param axis: _description_
        :type axis: jnp.int32
        :return: _description_
        :rtype: Tuple[Array, Array]
        """
        slice_minus = self.cons_positivity_slices[axis][0]
        slice_plus = self.cons_positivity_slices[axis][1]

        mesh_slice_minus = self.mesh_positivity_slices[axis][0]
        mesh_slice_plus = self.mesh_positivity_slices[axis][1]

        if self.is_mesh_stretching[axis]:
            lambda_pos_minus = lambda_pos[mesh_slice_minus]
            lambda_pos_plus = lambda_pos[mesh_slice_plus]
        else:
            lambda_pos_minus = lambda_pos_plus = lambda_pos

        # DETERMINE U_MINUS AND U_PLUS STATES
        if self.equation_type in ("SINGLE-PHASE",
                                  "DIFFUSE-INTERFACE-5EQM"):
            if self.positivity_flux_limiter == "SIMPLE":
                conservatives_minus = conservatives[slice_minus] + 2.0 * lambda_pos_minus * flux_xi_convective
                conservatives_plus = conservatives[slice_plus] - 2.0 * lambda_pos_plus * flux_xi_convective

            if self.positivity_flux_limiter == "NASA":
                flux_xi_simple = self.equation_manager.get_fluxes_xi(primitives, conservatives, axis)
                if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
                    # SET VOLUME FRACTION FLUX IN THE SIMPLE FLUX TO ZERO
                    flux_xi_simple = flux_xi_simple.at[self.vf_slices].set(0.0)

                conservatives_minus = conservatives[slice_minus] \
                    + 2.0 * lambda_pos_minus * (flux_xi_convective - flux_xi_simple[slice_minus])
                conservatives_plus = conservatives[slice_plus] \
                    - 2.0 * lambda_pos_plus * (flux_xi_convective - flux_xi_simple[slice_plus])

            if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
                conservatives_minus = conservatives_minus.at[self.vf_slices].add(
                    -2.0 * lambda_pos_minus * conservatives[(self.vf_slices,)+(slice_minus)] * u_hat_xi)
                conservatives_plus = conservatives_plus.at[self.vf_slices].add(
                    2.0 * lambda_pos_plus * conservatives[(self.vf_slices,)+(slice_plus)] * u_hat_xi)

        elif self.equation_type in ("SINGLE-PHASE-SOLID-LS", "TWO-PHASE-LS"):
            # VOLUME-AVERAGED CONS --> CONS # TODO WHAT ABOUT INTERFACE FLUX ?
            alpha_cons = volume_fraction * conservatives[...,self.nhx__,self.nhy__,self.nhz__]
            flux_xi_simple = self.equation_manager.get_fluxes_xi(primitives, conservatives, axis)
            flux_xi_simple = flux_xi_simple[...,self.nhx__,self.nhy__,self.nhz__]

            alpha_cons_minus = alpha_cons[slice_minus] + \
                2.0 * lambda_pos_minus * apertures * (flux_xi_convective - flux_xi_simple[slice_minus])
            alpha_cons_plus = alpha_cons[slice_plus] - \
                2.0 * lambda_pos_plus * apertures * (flux_xi_convective - flux_xi_simple[slice_plus])

            conservatives_minus = alpha_cons_minus / (volume_fraction[slice_minus] + 1e-10)
            conservatives_plus = alpha_cons_plus / (volume_fraction[slice_plus] + 1e-10)

        else:
            raise NotImplementedError

        return conservatives_minus, conservatives_plus

    def check_admissibility_for_density_and_volume_fraction(
            self,
            U_minus: Array,
            U_plus: Array,
            apertures: Array = None
            ) -> Array:
        if self.equation_type == "SINGLE-PHASE":
            # Check density > eps_density
            theta = jnp.where(
                (jnp.minimum(U_minus[self.mass_slices], U_plus[self.mass_slices]) < self.eps.density).any(axis=0), 
                0, 1)

        elif self.equation_type == "SINGLE-PHASE-SOLID-LS":
            # Check density > eps_density at all cell faces
            # with apertures > 0, i.e., all fluid cell faces.
            theta = jnp.where(
                (apertures > 0.0) & (jnp.minimum(U_minus[self.mass_ids], U_plus[self.mass_ids]) < self.eps.density), 
                0, 1)

        elif self.equation_type == "TWO-PHASE-LS":
            # Check density > eps_density at all cell faces
            # with apertures > 0. Here, apertures are wrt
            # to each fluid phase, i.e., apertures have shape
            # (2,Nx,Ny,Nz)
            theta = jnp.where(
                (apertures > 0.0) & (jnp.minimum(U_minus[self.mass_ids], U_plus[self.mass_ids]) < self.eps.density), 
                0, 1)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            # Check partial densities and volume fractions
            theta = jnp.where(
                (jnp.minimum(U_minus[self.mass_slices], U_plus[self.mass_slices]) < self.eps.density).any(axis=0) |
                (jnp.minimum(U_minus[self.vf_slices], U_plus[self.vf_slices]) < self.eps.volume_fraction).any(axis=0) |
                (jnp.maximum(U_minus[self.vf_slices], U_plus[self.vf_slices]) > 1.0 - self.eps.volume_fraction).any(axis=0),
                0, 1)

        else:
            raise NotImplementedError

        return theta

    def check_admissibility_pressure_speed_of_sound(
            self,
            U_minus: Array,
            U_plus: Array,
            apertures: Array = None
            ) -> Array:
        W_minus = self.equation_manager.get_primitives_from_conservatives(U_minus)
        W_plus = self.equation_manager.get_primitives_from_conservatives(U_plus)
        pressure_minus = W_minus[self.energy_ids]
        pressure_plus = W_plus[self.energy_ids]

        if self.equation_type == "SINGLE-PHASE":
            # Check speed of sound via 1) pressure or 2) internal energy
            pb_minus = self.material_manager.get_background_pressure(W_minus[self.vf_slices])
            pb_plus = self.material_manager.get_background_pressure(W_plus[self.vf_slices])

            # Option 1 - check PRESSURE DIRECTLY
            theta = jnp.where(
                (jnp.minimum(pressure_minus + pb_minus, pressure_plus + pb_plus) < self.eps.pressure),
                0, 1)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # rho_minus  = self.material_manager.get_density(U_minus)
            # rho_plus   = self.material_manager.get_density(U_plus)
            # rhoe_minus = rho_minus * self.material_manager.get_specific_energy(pressure_minus, rho=rho_minus, alpha_i=W_minus[self.vf_slices])
            # rhoe_plus  = rho_plus  * self.material_manager.get_specific_energy(pressure_plus, rho=rho_plus, alpha_i=W_plus[self.vf_slices])

            # theta   = jnp.where( 
            #     (jnp.minimum(rhoe_minus - pb_minus, rhoe_plus - pb_plus) < self.eps.pressure), 0, 1
            # )

        elif self.equation_type == "SINGLE-PHASE-SOLID-LS":
            pb_minus = self.material_manager.get_background_pressure(W_minus[self.vf_slices])
            pb_plus = self.material_manager.get_background_pressure(W_plus[self.vf_slices])

            theta = jnp.where( 
                (apertures > 0.0) & (jnp.minimum(pressure_minus + pb_minus, pressure_plus + pb_plus) < self.eps.pressure),
                0, 1)

        elif self.equation_type == "TWO-PHASE-LS":
            pb_minus = self.material_manager.get_background_pressure(W_minus[self.vf_slices])
            pb_plus = self.material_manager.get_background_pressure(W_plus[self.vf_slices])

            theta = jnp.where( 
                (apertures > 0.0) & (jnp.minimum(pressure_minus + pb_minus, pressure_plus + pb_plus) < self.eps.pressure),
                0, 1)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            # CHECK PSEUDO SPEED OF SOUND
            pb_minus = self.material_manager.get_background_pressure(W_minus[self.vf_slices])
            pb_plus = self.material_manager.get_background_pressure(W_plus[self.vf_slices])

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            theta = jnp.where((jnp.minimum(pressure_minus + pb_minus, pressure_plus + pb_plus) < self.eps.pressure),
                0, 1)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # rho_minus = self.material_manager.get_density(U_minus)
            # rho_plus = self.material_manager.get_density(U_plus)
            # rhoe_minus = rho_minus * self.material_manager.get_specific_energy(
            #     pressure_minus, rho=rho_minus, alpha_i=W_minus[self.vf_slices])
            # rhoe_plus = rho_plus * self.material_manager.get_specific_energy(
            #     pressure_plus, rho=rho_plus, alpha_i=W_plus[self.vf_slices])

            # theta   = jnp.where( 
            #     (jnp.minimum(rhoe_minus - pb_minus, rhoe_plus - pb_plus) < self.eps.pressure), 0, 1
            # )

        else:
            raise NotImplementedError

        return theta

    def compute_partition(
            self,
            primitives: Array,
            axis: int
        ) -> Array:
        if self.partition_type == "UNIFORM":
            one_sigma_axis = self.dim
        
        elif self.partition_type == "GEOMETRIC":
            one_sigma_axis = self.one_cell_size_halos[axis] \
                / sum(self.one_cell_size_halos[axis_id] for axis_id in self.active_axes_indices)

        elif self.partition_type == "WAVESPEED":
            vel_ids = self.vel_ids
            speed_of_sound = self.material_manager.get_speed_of_sound(primitives)
            max_wave_propagation = []
            for i in range(3):
                if i in self.active_axes_indices:
                    max_wave_propagation_i = jnp.max(jnp.abs(primitives[vel_ids[i]]) + speed_of_sound)
                    if self.is_parallel:
                        max_wave_propagation_i = jax.lax.all_gather(max_wave_propagation_i, axis_name="i")
                        max_wave_propagation_i = jnp.max(max_wave_propagation_i)
                    max_wave_propagation.append(max_wave_propagation_i)
                else:
                    max_wave_propagation.append(0.0)
            one_sigma_axis = max_wave_propagation[axis] * self.one_cell_size_halos[axis] \
                / sum(max_wave_propagation[axis_id] * self.one_cell_size_halos[axis_id] for axis_id in self.active_axes_indices)

        else:
            raise NotImplementedError
        
        return one_sigma_axis

    def apply_limiter_to_high_order_flux(
            self,
            flux_xi_convective: Array,
            flux_xi_convective_pos: Array,
            theta: Array,
            positivity_count: jnp.int32,
            u_hat_xi: Array = None,
            u_hat_xi_pos: Array = None,
            alpha_hat_xi: Array = None,
            alpha_hat_xi_pos: Array = None,
            ) -> Tuple[Array, Array, Array, jnp.int32]:
        """Limit high-order fluxes and corresponding cell-face values

        :param flux_xi_convective: _description_
        :type flux_xi_convective: Array
        :param flux_xi_convective_pos: _description_
        :type flux_xi_convective_pos: Array
        :param theta: _description_
        :type theta: Array
        :param positivity_count: _description_
        :type positivity_count: jnp.int32
        :param u_hat_xi: _description_, defaults to None
        :type u_hat_xi: Array, optional
        :param u_hat_xi_pos: _description_, defaults to None
        :type u_hat_xi_pos: Array, optional
        :param alpha_hat_xi: _description_, defaults to None
        :type alpha_hat_xi: Array, optional
        :param alpha_hat_xi_pos: _description_, defaults to None
        :type alpha_hat_xi_pos: Array, optional
        :return: _description_
        :rtype: Tuple[Array, Array, Array, jnp.int32]
        """
        positivity_count += jnp.sum(1 - theta)
        flux_xi_convective = (1 - theta) * flux_xi_convective_pos + theta * flux_xi_convective

        if self.equation_type in ("DIFFUSE-INTERFACE-5EQM"):
            u_hat_xi = (1 - theta) * u_hat_xi_pos + theta * u_hat_xi

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM" and alpha_hat_xi is not None:
            alpha_hat_xi = (1 - theta) * alpha_hat_xi_pos + theta * alpha_hat_xi
        
        return flux_xi_convective, u_hat_xi, alpha_hat_xi, positivity_count