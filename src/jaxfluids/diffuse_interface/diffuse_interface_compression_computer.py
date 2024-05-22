from typing import Tuple
import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.diffuse_interface.diffuse_interface_geometry_calculator import DiffuseInterfaceGeometryCalculator
from jaxfluids.diffuse_interface.helper_functions import smoothed_interface_function, \
    heaviside
from jaxfluids.config import precision

class DiffuseInterfaceCompressionComputer:
    """The DiffuseInterfaceCompressionComputer class implements functionality
    to perform interface compression for two-phase simulations with the 
    diffuse-interface method.
    """

    def __init__(
            self, 
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            material_manager: MaterialManager,
            halo_manager: HaloManager,
            diffuse_interface_setup: DiffuseInterfaceSetup,
            geometry_calculator: DiffuseInterfaceGeometryCalculator
            ) -> None:

        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.equation_manager = equation_manager
        self.halo_manager = halo_manager
        self.material_manager = material_manager
        self.geometry_calculator = geometry_calculator
        
        self.smallest_cell_size = domain_information.smallest_cell_size
        self.largest_cell_size = domain_information.largest_cell_size
        self.nhx_, self.nhy_, self.nhz_ = domain_information.domain_slices_geometry
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.active_axes_indices = domain_information.active_axes_indices
        self.active_axes = domain_information.active_axes
        
        equation_information = equation_manager.equation_information
        self.is_surface_tension = equation_information.active_physics.is_surface_tension
        self.mass_ids = equation_information.mass_ids
        self.mass_slices = equation_information.mass_slices
        self.vel_ids = equation_information.velocity_ids
        self.vel_slices = equation_information.velocity_slices
        self.energy_ids = equation_information.energy_ids
        self.energy_slices = equation_information.energy_slices
        self.vf_ids = equation_information.vf_ids
        self.vf_slices = equation_information.vf_slices
        
        self.flux_slices = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

        self.domain_slices_normal = self.geometry_calculator.domain_slices_normal
        self.interface_smoothing_parameter = diffuse_interface_setup.geometry_calculation.interface_smoothing
        
        # INTERFACE COMPRESSION SETUP
        interface_compression_setup = diffuse_interface_setup.interface_compression        
        
        self.time_integrator: TimeIntegrator = interface_compression_setup.time_integrator(
            nh=domain_information.nh_conservatives,
            inactive_axes=domain_information.inactive_axes)
        
        self.first_derivative_cell_center = geometry_calculator.first_derivative_cell_center
        self.reconstruction_stencil = geometry_calculator.reconstruction_stencil_conservatives
                
        self.heaviside_parameter = interface_compression_setup.heaviside_parameter
        interface_thickness_parameter = interface_compression_setup.interface_thickness_parameter
        self.epsilon_h = interface_thickness_parameter * self.largest_cell_size
        
        self.sigma = self.material_manager.get_sigma()
        # TODO
        equation_type = equation_information.equation_type
        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            self.gamma_l = self.material_manager.diffuse_5eqm_mixture.gamma[0]
            self.gamma_g = self.material_manager.diffuse_5eqm_mixture.gamma[1]
            self.Gamma_l = self.material_manager.diffuse_5eqm_mixture.one_gamma_[0]
            self.Gamma_g = self.material_manager.diffuse_5eqm_mixture.one_gamma_[1]
            self.Pi_l = self.material_manager.diffuse_5eqm_mixture.gamma_pb_[0]
            self.Pi_g = self.material_manager.diffuse_5eqm_mixture.gamma_pb_[1]
    
        self.delta_gamma = self.gamma_l - self.gamma_l
        self.delta_Gamma = self.Gamma_l - self.Gamma_g
        self.delta_Pi = self.Pi_l - self.Pi_g    
        
    def compute_R_star(
            self, 
            rho_alpha: Array, 
            alpha: Array, 
            normal_cell_center: Array
            ) -> Array:
        """Computes the interface compression operator R_star

        :param rho_alpha: _description_
        :type rho_alpha: Array
        :param alpha: _description_
        :type alpha: Array
        :param normal_cell_center: _description_
        :type normal_cell_center: Array
        :return: _description_
        :rtype: Array
        """
        print("ALPHA SHAPE =", alpha.shape, "RHO_ALPHA SHAPE =", rho_alpha.shape)

        one_cell_sizes = self.domain_information.get_device_one_cell_sizes()
        cell_sizes = self.domain_information.get_device_cell_sizes()

        R_star = 0
        for i in self.active_axes_indices:
            normal_cell_face_xi = self.geometry_calculator.compute_normal_(
                smoothed_interface_function(alpha, self.interface_smoothing_parameter),
                location="FACE", axis=i)
            print("NORMAL AT CELL FACE SHAPE =", normal_cell_face_xi.shape)
            
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # x_tmp = jnp.arange(normal_cell_face_xi.shape[1])
            # ax.plot(x_tmp, normal_cell_face_xi[0,:,0,0])
            # ax.plot(0.5*(x_tmp[1:]+x_tmp[:-1]), normal_cell_center[0,:,0,0], linestyle="--")
            # plt.savefig("debug_normal_cc.png")
            # plt.close()
            
            gradient_rhoalpha_cell_face_xi = self.geometry_calculator.compute_gradient_at_cell_face_xi(
                rho_alpha, i)
            print("GRADIENT RHO ALPHA AT CELL FACE SHAPE =", gradient_rhoalpha_cell_face_xi.shape)

            f_star = self.epsilon_h * jnp.sum(normal_cell_face_xi * gradient_rhoalpha_cell_face_xi, axis=0)
            print("F STAR SHAPE =", f_star.shape)
            delta_fstar = one_cell_sizes[i] \
                * (f_star[self.flux_slices[i][0]] - f_star[self.flux_slices[i][1]])
            print("DELTA F STAR SHAPE =", delta_fstar.shape)
            
            gradient_rhoalpha_cell_center = self.first_derivative_cell_center.derivative_xi(
                rho_alpha, cell_sizes[i], i)
            print("GRADIENT RHOALPHA SHAPE =", gradient_rhoalpha_cell_center.shape)
            
            R_star += normal_cell_center[i] * (
                delta_fstar \
                - (1.0 - 2.0 * alpha[self.nhx,self.nhy,self.nhz]) * gradient_rhoalpha_cell_center
                )
        
        R_star *= heaviside(alpha[self.nhx,self.nhy,self.nhz], self.heaviside_parameter)
        print("R_STAR SHAPE =", R_star.shape)
        return R_star
        
    def compute_R(
            self, 
            volume_fraction: Array,
            normal_cell_center: Array
            ) -> Array:
        """Computes the interface compression operator R

        R = n \cdot ( \nabla(eps_h * |\nabla phi_l|) - phi_l (1 - phi_l))

        Implementation following to Garrick et al. 2017

        :param volume_fraction: _description_
        :type volume_fraction: Array
        :param normal_cell_center: _description_
        :type normal_cell_center: Array
        :return: _description_
        :rtype: Array
        """
        one_cell_sizes = self.domain_information.get_device_one_cell_sizes()
        R = 0.0
        for i in self.active_axes_indices:
            volume_fraction_cf = self.reconstruction_stencil.reconstruct_xi(volume_fraction, axis=i)
            # Eq. 65
            gradient_smoothed_vf_at_cf = self.geometry_calculator.compute_gradient_at_cell_face_xi(
                smoothed_interface_function(volume_fraction, self.interface_smoothing_parameter),
                axis=i)
            vf_gradient_cell_face = self.compute_volume_fraction_gradient_at_cell_face(
                gradient_smoothed_vf_at_cf, volume_fraction_cf, self.interface_smoothing_parameter)            
            # Eq. (58)
            f = self.epsilon_h * vf_gradient_cell_face \
                - volume_fraction_cf * (1.0 - volume_fraction_cf)
            # Eq. (56)
            R += normal_cell_center[i] * one_cell_sizes[i] \
                * (f[self.flux_slices[i][0]] - f[self.flux_slices[i][1]])
        
        return R
    
    def compute_volume_fraction_gradient_at_cell_face(
            self,
            gradient_smoothed_vf: Array,
            vf_at_cf: Array,
            alpha: float,
            ) -> Array:
        # Eq. 65
        # TODO eps AD
        gradient_vf = 1.0 / alpha \
            * (vf_at_cf * (1.0 - vf_at_cf))**(1.0 - alpha) \
            * (vf_at_cf**alpha + (1.0 - vf_at_cf)**alpha) * (vf_at_cf**alpha + (1.0 - vf_at_cf)**alpha) \
            * jnp.linalg.norm(gradient_smoothed_vf, axis=0)
        return gradient_vf 
    
    def perform_interface_compression(
            self,
            conservatives: Array,
            primitives: Array,
            CFL: float,
            steps: int,
            physical_simulation_time: float
            ) -> Tuple[Array, Array]:
        # TODO do we need limiting if eps in vf initialization???
        conservatives = self.limit_volume_fraction(conservatives)
        primitives = self.equation_manager.get_primitives_from_conservatives(conservatives)
        
        volume_fraction = conservatives[self.vf_ids[0]]
        fictitious_timestep_size = CFL * self.smallest_cell_size
        
        for i in range(steps):
            print("\n INTERFACE COMPRESSION STEP =", i)
            if self.is_surface_tension:
                curvature = self.geometry_calculator.compute_curvature(
                    volume_fraction)[self.nhx_,self.nhy_,self.nhz_]
                # curvature = jnp.ones_like(curvature)
                print("CURAVTURE IS NAN = ", jnp.isnan(curvature).any())
            else:
                curvature = 0.0
            
            conservatives, primitives = self.do_integration_step(
                conservatives, primitives, curvature, 
                physical_simulation_time, fictitious_timestep_size)
            print("MIN / MAX VOLUME FRACTION")
            print(jnp.min(conservatives[-1]))
            print(jnp.max(conservatives[-1]))
        
        return conservatives, primitives
    
    def do_integration_step(
            self,
            conservatives: Array,
            primitives: Array,
            curvature: Array,
            physical_simulation_time: float,
            fictitious_timestep_size: float
            ) -> Array:
                
        if self.time_integrator.no_stages > 1:
            init = jnp.array(conservatives, copy=True)
        for stage in range(self.time_integrator.no_stages):
            rhs = self.compute_rhs(
                conservatives, primitives, curvature, fictitious_timestep_size)
            if stage > 0:
                conservatives = self.time_integrator.prepare_buffer_for_integration(
                    conservatives, init, stage)
            
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(ncols=2, nrows=2)
            # ax[0,0].plot(conservatives[0,:,0,0])
            # ax[0,1].plot(conservatives[1,:,0,0])
            # ax[1,0].plot(conservatives[-2,:,0,0])
            # ax[1,1].plot(conservatives[-1,:,0,0])
            # plt.savefig("debug_cons.png")
            # plt.close()
            
            # fig, ax = plt.subplots(ncols=2, nrows=2)
            # ax[0,0].plot(rhs[0,:,0,0])
            # ax[0,1].plot(rhs[1,:,0,0])
            # ax[1,0].plot(rhs[-2,:,0,0])
            # ax[1,1].plot(rhs[-1,:,0,0])
            # plt.savefig("debug_rhs.png")
            # plt.close()
            
            # input()

            conservatives = self.time_integrator.integrate(
                conservatives, rhs, fictitious_timestep_size, stage)
            # TODO do we need limiting if eps in vf initialization
            conservatives = self.limit_volume_fraction(conservatives)
            
            primitives = self.equation_manager.get_primitives_from_conservatives(conservatives)
            primitives, conservatives = self.halo_manager.perform_halo_update_material(
                primitives, physical_simulation_time, False, False, conservatives) # TODO HALOS

        return conservatives, primitives
    
    def compute_rhs(
            self, 
            conservatives: Array, 
            primitives: Array,
            curvature: Array,
            timestep: float
            ) -> Array:
        
        alpha_l = primitives[self.vf_ids[0]]
        alpha_g = 1.0 - primitives[self.vf_ids[0]]
        rhoalpha_l = primitives[self.mass_ids[0]]
        rhoalpha_g = primitives[self.mass_ids[1]]
        velocities = primitives[self.vel_slices][...,self.nhx,self.nhy,self.nhz]
        pressure = primitives[self.energy_ids][self.nhx,self.nhy,self.nhz]
        kappa_e = jnp.sum(velocities * velocities, axis=0)
                
        print("CONSERVATIVES SHAPE = ", conservatives.shape)
        
        normal_cell_center = self.geometry_calculator.compute_normal_(
            smoothed_interface_function(alpha_l, self.interface_smoothing_parameter),
            location="CENTER")[self.domain_slices_normal]
        print("NORMAL CC SHAPE =", normal_cell_center.shape)
        print("NORMAL CC IS NAN =", jnp.isnan(normal_cell_center).any())
        
        R_hat_l = self.compute_R_star(rhoalpha_l, alpha_l, normal_cell_center)
        R_hat_g = self.compute_R_star(rhoalpha_g, alpha_l, normal_cell_center)
        R_hat = R_hat_l + R_hat_g
        R = self.compute_R(alpha_l, normal_cell_center)
        R_u = velocities * R_hat
        R_E = kappa_e * R_hat \
            + (pressure * self.delta_Gamma + self.delta_Pi) * R \
            + self.sigma * curvature \
                * (self.gamma_l * alpha_g[self.nhx, self.nhy, self.nhz] + self.gamma_g * alpha_l[self.nhx, self.nhy, self.nhz] \
                    + self.delta_gamma * timestep * R - 1.0) * self.Gamma_l * self.Gamma_g * R
        
        print(R_hat_l.shape, R_hat_g.shape, R_u.shape, R_E.shape, R.shape)

        rhs = jnp.stack([R_hat_l, R_hat_g, R_u[0], R_u[1], R_u[2], R_E, R], axis=0)
        print(rhs.shape)
        return rhs

    def limit_volume_fraction(self, conservatives: Array) -> Array:
        volume_fraction = conservatives[self.vf_slices]
        volume_fraction = jnp.minimum(jnp.maximum(volume_fraction, 0.0), 1.0)
        conservatives = conservatives.at[self.vf_slices].set(volume_fraction)
        return conservatives
    