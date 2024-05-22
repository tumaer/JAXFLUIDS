import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
from jaxfluids.diffuse_interface.diffuse_interface_geometry_calculator import DiffuseInterfaceGeometryCalculator
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.config import precision

class DiffuseInterfacePDERegularizationComputer:
    """Implements functionality for the evaluation of ACDI/CDI-based
    interface diffusion and sharpening fluxes for the 5-equation
    diffuse-interface model.
    """
    
    def __init__(
            self,
            domain_information: DomainInformation,
            equation_information: EquationInformation,
            material_manager: MaterialManager,
            diffuse_interface_setup: DiffuseInterfaceSetup,
            geometry_calculator: DiffuseInterfaceGeometryCalculator
            ) -> None:

        self.eps = precision.get_eps()

        self.equation_information = equation_information
        self.geometry_calculator = geometry_calculator
        self.material_manager = material_manager
        self.domain_information = domain_information
        
        diffusion_sharpening_setup = diffuse_interface_setup.diffusion_sharpening        
        self.diffusion_sharpening_model = diffusion_sharpening_setup.model
        self.density_model = diffusion_sharpening_setup.density_model
        self.incompressible_density = diffusion_sharpening_setup.incompressible_density.reshape(-1,1,1,1)
        self.interface_thickness_parameter = diffusion_sharpening_setup.interface_thickness_parameter
        self.interface_velocity_parameter = diffusion_sharpening_setup.interface_velocity_parameter
        self.mobility_model = diffusion_sharpening_setup.mobility_model
        self.vf_threshold = diffusion_sharpening_setup.volume_fraction_threshold
        self.acdi_threshold = diffusion_sharpening_setup.acdi_threshold
        self.is_acdi_mask = self.acdi_threshold > 0.0

        self.dim = domain_information.dim
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.active_axes_indices = domain_information.active_axes_indices
        self.smallest_cell_size = domain_information.smallest_cell_size
        self.is_parallel = domain_information.is_parallel
        
        self.gradient_at_cell_face_shape = geometry_calculator.normal_at_cell_face_shape
        self.normal_to_geometry = geometry_calculator.normal_to_geometry
        
        self.reconstruction_stencil_conservatives = geometry_calculator.reconstruction_stencil_conservatives
        self.reconstruction_stencil_geometry = geometry_calculator.reconstruction_stencil_geometry
        self.derivative_stencil_conservatives_center = geometry_calculator.first_derivative_cell_center
        self.derivative_stencil_conservatives_face = geometry_calculator.first_derivative_stencil_cell_face

        nh = domain_information.nh_conservatives
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        self.slices_LR = [
            [jnp.s_[..., nh-1:-nh, nhy, nhz], jnp.s_[..., nh:-nh+1, nhy, nhz]],
            [jnp.s_[..., nhx, nh-1:-nh, nhz], jnp.s_[..., nhx, nh:-nh+1, nhz]],
            [jnp.s_[..., nhx, nhy, nh-1:-nh], jnp.s_[..., nhx, nhy, nh:-nh+1]]
        ]

        # Slices to give i-1, i, i+1
        self.slices_3 = [
            [jnp.s_[..., nh-2:-nh, nhy, nhz], jnp.s_[..., nh-1:-nh+1, nhy, nhz], jnp.s_[..., nh:-nh+2, nhy, nhz]],
            [jnp.s_[..., nhx, nh-2:-nh, nhz], jnp.s_[..., nhx, nh-1:-nh+1, nhz], jnp.s_[..., nhx, nh:-nh+2, nhz]],
            [jnp.s_[..., nhx, nhy, nh-2:-nh], jnp.s_[..., nhx, nhy, nh-1:-nh+1], jnp.s_[..., nhx, nhy, nh:-nh+2]]
        ]
        
        self.slices_2 = [
            [jnp.s_[..., :-1, :, :], jnp.s_[..., 1:, :, :]],
            [jnp.s_[..., :, :-1, :], jnp.s_[..., :, 1:, :]],
            [jnp.s_[..., :, :, :-1], jnp.s_[..., :, :, 1:]],
        ]

        self.reconstruction_type = "CENTRAL"
        self.reconstruction_type_kinetic_energy = "CENTRAL"

    def compute_diffusion_sharpening_flux_xi(
            self,
            conservatives: Array,
            primitives: Array,
            axis: int,
            volume_fraction: Array = None,
            numerical_dissipation: Array = None
            ) -> Array:

        equation_type = self.equation_information.equation_type
        vel_slices = self.equation_information.velocity_slices
        vf_ids = self.equation_information.vf_ids
        slice_p, slice_pp = self.slices_LR[axis]

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            volume_fraction = conservatives[vf_ids]
    
        # INTERFACE FLUX
        velocity_max = self._compute_maximal_velocity(primitives)
        # velocity_max = 0.25
        interface_regularization_flux_xi = self.compute_interface_regularization_flux(
            volume_fraction, velocity_max, axis, numerical_dissipation)

        # MASK
        if self.is_acdi_mask:
            mask = jnp.where(
                (jnp.minimum(volume_fraction[slice_p], volume_fraction[slice_pp]) > self.acdi_threshold) &
                (jnp.maximum(volume_fraction[slice_p], volume_fraction[slice_pp]) < 1.0 - self.acdi_threshold),
                1, 0)
            interface_regularization_flux_xi *= mask
            count_acdi_xi = jnp.sum(mask)
        else:
            count_acdi_xi = None
        
        # DEGENERATE MOBILITIY
        if self.mobility_model:
            interface_regularization_flux_xi *= self._compute_mobility(volume_fraction, axis)
        
        # slice_im, slice_i, slice_ip = self.slices_3[axis]
        # slice_0, slice_1 = self.slices_2[axis]
        # marker = (volume_fraction[slice_ip] - volume_fraction[slice_i]) * (volume_fraction[slice_i] - volume_fraction[slice_im])
        # mask = jnp.where((marker[slice_0] > 0) & (marker[slice_1] > 0), 1, 0)
        # interface_regularization_flux_xi *= mask

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # ax[0].plot(jnp.squeeze(volume_fraction[nhx,nhy,nhz]), marker=".")
        # ax[1].plot(jnp.squeeze(interface_regularization_flux[0]), marker="o", mfc="None")
        # ax[1].plot(jnp.squeeze(interface_regularization_flux[1]), marker="x", mfc="None")
        # ax[1].plot(jnp.squeeze(interface_regularization_flux[2]), marker="s", mfc="None")
        # plt.show()

        if self.diffusion_sharpening_model in ("CDI", "ACDI"):
            # MASS
            phasic_densities_cf = self.compute_phasic_density_cf_xi(primitives, axis)
            mass_regularization_flux = phasic_densities_cf * interface_regularization_flux_xi

            # MOMENTUM
            velocity_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(primitives[vel_slices], axis)
            momentum_regularization_flux = (mass_regularization_flux[0] - mass_regularization_flux[1]) * velocity_cf

            # ENERGY
            kinetic_energy_cf = self.compute_kinetic_energy_cf_xi(primitives, axis)
            phasic_volume_specific_enthalpy_cf \
                = self.compute_phasic_phasic_volume_specific_enthalpy_cf_xi(
                primitives, axis)
            energy_regularization_flux = \
                (mass_regularization_flux[0] - mass_regularization_flux[1]) * kinetic_energy_cf \
                + (phasic_volume_specific_enthalpy_cf[0] - phasic_volume_specific_enthalpy_cf[1]) \
                    * interface_regularization_flux_xi
        
        elif self.diffusion_sharpening_model in ("MODELB", "MODELD"):
            mass_slices = self.equation_information.mass_slices
            
            # # CENTRAL
            # phi_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(volume_fraction, axis)
            # phi_cf_threshold = (phi_cf - self.vf_threshold)
            # one_minus_phi_cf_threshold = (1.0 - self.vf_threshold - phi_cf)

            # UPWINDING
            slice_R, slice_L = self.slices_LR[axis]
            phi_cf = jnp.where(interface_regularization_flux_xi >= 0.0, volume_fraction[slice_L], volume_fraction[slice_R])
            one_minus_cf = jnp.where(interface_regularization_flux_xi >= 0.0, 1.0 - volume_fraction[slice_R], 1.0 - volume_fraction[slice_L])
            phi_cf_threshold = (phi_cf - self.vf_threshold)
            one_minus_phi_cf_threshold = (one_minus_cf - self.vf_threshold)

            # MASS
            rhoalpha_cf = self.compute_rhoalpha_cf_xi(primitives[mass_slices],
                                                      axis,
                                                      interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)
            mass_regularization_flux = [
                rhoalpha_cf[0] * one_minus_phi_cf_threshold * interface_regularization_flux_xi,
                rhoalpha_cf[1] * phi_cf_threshold * interface_regularization_flux_xi,]
            
            # MOMENTUM
            velocity_cf = self.compute_velocity_cf_xi(
                primitives[vel_slices],
                axis,
                interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)
            momentum_regularization_flux = (mass_regularization_flux[0] - mass_regularization_flux[1]) * velocity_cf
            
            # ENERGY
            kinetic_energy_cf = self.compute_kinetic_energy_cf_xi(
                primitives,
                axis,
                interface_regularization_flux_xi if self.reconstruction_type_kinetic_energy == "UPWIND" else None)
            phasic_volume_specific_enthalpy_cf = self.compute_phasic_phasic_volume_specific_enthalpy_cf_xi(
                primitives,
                axis,
                interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)
            energy_regularization_flux = \
                (mass_regularization_flux[0] - mass_regularization_flux[1]) * kinetic_energy_cf \
                + (phasic_volume_specific_enthalpy_cf[0] - phasic_volume_specific_enthalpy_cf[1]) \
                    * phi_cf_threshold * one_minus_phi_cf_threshold \
                    * interface_regularization_flux_xi
            
            # INTERFACE
            # interface_regularization_flux_xi *= phi_cf * (1.0 - phi_cf)
            interface_regularization_flux_xi *= phi_cf_threshold * one_minus_phi_cf_threshold

        elif self.diffusion_sharpening_model == "MODELC":
            # Analytically introduce degenerate mobility of form 4 * phi * (1-phi)
            # to avoid division by volume fraction.
            mass_slices = self.equation_information.mass_slices
            
            interface_regularization_flux_xi *= 4.0

            # phi_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(volume_fraction, axis)
            phi_cf = self.compute_volume_fraction_cf_xi(volume_fraction,
                                                        axis,
                                                        interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)

            # MASS
            # rhoalpha_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(primitives[mass_slices], axis)
            rhoalpha_cf = self.compute_rhoalpha_cf_xi(primitives[mass_slices],
                                                      axis,
                                                      interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)
            mass_regularization_flux = [
                rhoalpha_cf[0] * (1.0 - self.vf_threshold - phi_cf) * interface_regularization_flux_xi,
                rhoalpha_cf[1] * (phi_cf - self.vf_threshold) * interface_regularization_flux_xi,]
            
            # MOMENTUM
            # velocity_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(primitives[vel_slices], axis)
            velocity_cf = self.compute_velocity_cf_xi(
                primitives[vel_slices],
                axis,
                interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)
            momentum_regularization_flux = (mass_regularization_flux[0] - mass_regularization_flux[1]) * velocity_cf
            
            # ENERGY
            kinetic_energy_cf = self.compute_kinetic_energy_cf_xi(
                primitives,
                axis,
                interface_regularization_flux_xi if self.reconstruction_type_kinetic_energy == "UPWIND" else None)
            phasic_volume_specific_enthalpy_cf = self.compute_phasic_phasic_volume_specific_enthalpy_cf_xi(
                primitives,
                axis,
                interface_regularization_flux_xi if self.reconstruction_type == "UPWIND" else None)
            energy_regularization_flux = \
                (mass_regularization_flux[0] - mass_regularization_flux[1]) * kinetic_energy_cf \
                + (phasic_volume_specific_enthalpy_cf[0] - phasic_volume_specific_enthalpy_cf[1]) \
                    * (phi_cf - self.vf_threshold) * (1.0 - self.vf_threshold - phi_cf) \
                    * interface_regularization_flux_xi
            
            # INTERFACE
            # interface_regularization_flux_xi *= phi_cf * (1.0 - phi_cf)
            interface_regularization_flux_xi *= (phi_cf - self.vf_threshold) * (1.0 - self.vf_threshold - phi_cf)

        else:
            raise NotImplementedError

        # ASSEMBLE FINAL FLUX
        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            regularization_flux = jnp.stack([
                mass_regularization_flux[0],
                -mass_regularization_flux[1],
                momentum_regularization_flux[0],
                momentum_regularization_flux[1],
                momentum_regularization_flux[2],
                energy_regularization_flux,
                interface_regularization_flux_xi
                ], axis=0)
        else:
            raise NotImplementedError
        
        return regularization_flux, count_acdi_xi
    
    def _compute_maximal_velocity(self, primitives: Array) -> Array:
        nhx, nhy, nhz = self.nhx, self.nhy, self.nhz
        vel_ids = self.equation_information.velocity_ids

        primitives = primitives[...,nhx,nhy,nhz]
        
        # speed_of_sound = self.material_manager.get_speed_of_sound(primitives)
        # velocity_max = speed_of_sound * speed_of_sound
        velocity_max = 0.0
        for i in self.active_axes_indices:
            velocity_max += primitives[vel_ids[i]] * primitives[vel_ids[i]]
        velocity_max = jnp.sqrt(jnp.max(velocity_max))
        
        if self.is_parallel:
            velocity_max = jax.lax.all_gather(velocity_max, axis_name="i")
            velocity_max = jnp.max(velocity_max)

        return velocity_max
    
    def _compute_maximal_dilatation(self, primitives: Array) -> Array:
        cell_sizes = self.domain_information.get_device_cell_sizes()
        vel_ids = self.equation_information.velocity_ids
        dilatation = 0.0
        for axis_i in self.active_axes_indices:
            dilatation += self.derivative_stencil_conservatives_center.derivative_xi(
                primitives[vel_ids[axis_i]],
                cell_sizes[axis_i],
                axis_i)
        
        dilatation = jnp.max(dilatation)
        
        if self.is_parallel:
            dilatation = jax.lax.all_gather(dilatation, axis_name="i")
            dilatation = jnp.max(dilatation)

        return dilatation

    def _compute_mobility(self, phi: Array, axis: int) -> Array:
        slice_p, slice_pp = self.slices_LR[axis]
        if self.mobility_model == "OPT1":
            mobility = 4.0 * 0.5 * (
                (phi[slice_p] - 0.5) * (phi[slice_p] - 0.5) \
                + (phi[slice_pp] - 0.5) * (phi[slice_pp] - 0.5))

        elif self.mobility_model == "OPT2":
            mobility = 4.0 * 0.5 * (
                jnp.abs(phi[slice_p] * (1.0 - phi[slice_p])) \
                + jnp.abs(phi[slice_pp] * (1.0 - phi[slice_pp])))

        else:
            raise NotImplementedError

        return mobility
    
    def compute_interface_regularization_flux(
            self,
            phi: Array,
            velocity_max: float,
            axis: int,
            numerical_dissipation: Array = None
            ) -> Array:
        """Computes the interface regularization flux based
        on the CDI/ACDI method. The regularization flux is
        a diffusion & sharpening flux.
        
        Jain et al. - 2020 - A conservative diffuse-interface
        method for compressible two-phase flows
        
        Jain - 2022 - Accurate conservative phase-field method
        for simulation of two-phase flows
        
        Eq. (12) in Jain et al. 2020
        F_interface = \Gamma * {\epsilon \nabla \phi - \phi (1 - \phi) \nabla \phi / |\nabla \phi| }

        Eq. (13) in Jain 2022
        F_interface = \Gamma * {\epsilon \nabla \phi - 1/4 (1 - \tanh^2(\psi / 2 / \epsilon)) \nabla \psi / |\nabla \psi| }

        :param phi: Volume fraction buffer, (Nx+2nh,Ny+2nh,Nz+2nh)
        :type phi: Array
        :param velocity_max: Maximal absolute velocity in the domain
        :type velocity_max: float
        :param axis: Direction of the cell-face
        :type axis: int
        :return: Interface regularization flux buffer, e.g., for x-direction,
            (Nx+1,Ny,Nz)
        :rtype: Array
        """
        cell_sizes = self.domain_information.get_device_cell_sizes()
        phi = jnp.clip(phi, self.vf_threshold, 1.0 - self.vf_threshold)
        interface_thickness = self.interface_thickness_parameter * self.smallest_cell_size
        interface_velocity = self.interface_velocity_parameter * velocity_max
        # interface_velocity = 1.0
        
        derivative_phi_cf = self.derivative_stencil_conservatives_face.derivative_xi(
            phi, cell_sizes[axis], axis)
        
        if self.diffusion_sharpening_model in ("CDI", "MODELD"):
            normal_cc = self.geometry_calculator.compute_normal_(phi, location="CENTER")[self.normal_to_geometry]

        elif self.diffusion_sharpening_model in ("ACDI", "MODELB", "MODELC"):
            psi = self.geometry_calculator.compute_signed_distance_function(phi) # NOTE: this is actually psi / epsilon
            normal_cc = self.geometry_calculator.compute_normal_(psi, location="CENTER")[self.normal_to_geometry]

        else:
            raise NotImplementedError

        # AXIS-COMPONENT OF NORMAL AT CELL-FACE
        normal_cf = self.reconstruction_stencil_geometry.reconstruct_xi(normal_cc, axis)
        normal_cf_axis = normal_cf[axis]

        if self.diffusion_sharpening_model in ("CDI", "MODELD"):
            phi_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(phi, axis)
            flux_xi = interface_velocity * (
                interface_thickness * derivative_phi_cf \
                - (phi_cf - self.vf_threshold) * (1.0 - self.vf_threshold - phi_cf) * normal_cf_axis)
        
        elif self.diffusion_sharpening_model in ("ACDI", "MODELC"): 
            psi_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(psi, axis)
            psi_cf = jnp.tanh(0.5 * psi_cf)
            psi_cf *= psi_cf
            
            flux_xi = interface_velocity * (
                interface_thickness * derivative_phi_cf \
                - (0.25 * (1.0 - psi_cf) - self.vf_threshold * (1.0 - self.vf_threshold)) * normal_cf_axis)

        elif self.diffusion_sharpening_model == "MODELB":
            derivative_psi_cf = self.derivative_stencil_conservatives_face.derivative_xi(
                psi, cell_sizes[axis], axis)
            # flux_xi = interface_velocity * (jnp.sign(derivative_phi_cf) * jnp.abs(derivative_psi_cf) - normal_cf_axis)
            flux_xi = interface_velocity * (interface_thickness * derivative_psi_cf - normal_cf_axis)

        else:
            raise NotImplementedError
        
        return flux_xi

    def compute_timestep(
            self,
            primitives: Array
            ) -> Array:
        """Computes the time step criterion for CDI/ACDI
        regularization terms.

        :param primitives: _description_
        :type primitives: Array
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        velocity_max = self._compute_maximal_velocity(primitives)
        dilatation_max = self._compute_maximal_dilatation(primitives)
        interface_thickness = self.interface_thickness_parameter * self.smallest_cell_size
        interface_velocity = self.interface_velocity_parameter * velocity_max
        
        if self.diffusion_sharpening_model == "CDI":
            diffusive_condition = 2 * self.dim * interface_velocity * interface_thickness \
                / (self.smallest_cell_size * self.smallest_cell_size)
            dilatation_condition = dilatation_max
            dt = 1.0 / (diffusive_condition -  dilatation_condition)

        elif self.diffusion_sharpening_model in ("ACDI", "MODELB"):
            # TODO !!!
            diffusive_condition = 2 * self.dim * interface_velocity * interface_thickness \
                / (self.smallest_cell_size * self.smallest_cell_size)
            dilatation_condition = dilatation_max
            dt = 1.0 / (diffusive_condition -  dilatation_condition + self.eps)

        else:
            diffusive_condition = 2 * self.dim * interface_velocity * interface_thickness \
                / (self.smallest_cell_size * self.smallest_cell_size)
            dilatation_condition = dilatation_max
            dt = 1.0 / (diffusive_condition -  dilatation_condition + self.eps)
        
        return dt

    def compute_volume_fraction_cf_xi(
            self,
            volume_fraction: Array,
            axis: int,
            regularization_flux: Array = None
        ) -> Array:

        if self.reconstruction_type == "CENTRAL":
            phi_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(volume_fraction, axis)
        elif self.reconstruction_type == "UPWIND":
            slice_L, slice_R = self.slices_LR[axis]
            phi_cf = jnp.where(regularization_flux <= 0.0, volume_fraction[slice_L], volume_fraction[slice_R])
        else:
            raise NotImplementedError
        
        return phi_cf

    def compute_rhoalpha_cf_xi(
            self,
            rhoalpha: Array,
            axis: int,
            regularization_flux: Array = None
            ) -> Array:
        if self.reconstruction_type == "CENTRAL":
            rhoalpha_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(rhoalpha, axis)
        elif self.reconstruction_type == "UPWIND":
            slice_L, slice_R = self.slices_LR[axis]
            rhoalpha_cf = jnp.where(regularization_flux <= 0.0, rhoalpha[slice_L], rhoalpha[slice_R])
        else:
            raise NotImplementedError
        return rhoalpha_cf

    def compute_phasic_density_cf_xi(
            self,
            primitives: Array,
            axis: int
            ) -> Array:
        equation_type = self.equation_information.equation_type
        mass_slices = self.equation_information.mass_slices
        energy_ids = self.equation_information.energy_ids
        vf_slices = self.equation_information.vf_slices

        # PHASIC DENSITIES
        if self.density_model == "INCOMPRESSIBLE":
            phasic_densities_cf = self.incompressible_density

        elif self.density_model == "COMPRESSIBLE":
            rhoalpha_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(
                primitives[mass_slices], axis)
            alpha_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(
                primitives[vf_slices], axis)
            phasic_densities_cf = self.material_manager.get_phasic_density(
                alpha_rho_i=rhoalpha_cf, alpha_i=alpha_cf)
            # phasic_densities = self.material_manager.get_phasic_density(
            #     alpha_rho_i=primitives[mass_slices], alpha_i=primitives[vf_slices])
            # phasic_densities_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(
            #     phasic_densities, axis)

        else:
            raise NotImplementedError

        return phasic_densities_cf

    def compute_velocity_cf_xi(
            self,
            velocities: Array,
            axis: int,
            regularization_flux: Array = None
            ) -> Array:
        if self.reconstruction_type == "CENTRAL":
            velocity_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(velocities, axis)
        elif self.reconstruction_type == "UPWIND":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return velocity_cf

    def compute_phasic_phasic_volume_specific_enthalpy_cf_xi(
            self,
            primitives: Array,
            axis: int,
            regularization_flux: Array = None
            ) -> Array:
        equation_type = self.equation_information.equation_type
        mass_slices = self.equation_information.mass_slices
        energy_ids = self.equation_information.energy_ids
        vf_slices = self.equation_information.vf_slices
    

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            phasic_volume_specific_enthalpy = self.material_manager.get_phasic_volume_specific_enthalpy(
                primitives[energy_ids])

        else:
            raise NotImplementedError

        if self.reconstruction_type == "CENTRAL":
            phasic_volume_specific_enthalpy_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(
                phasic_volume_specific_enthalpy, axis)
        elif self.reconstruction_type == "UPWIND":
            slice_L, slice_R = self.slices_LR[axis]
            phasic_volume_specific_enthalpy_cf = jnp.where(
                regularization_flux <= 0.0,
                phasic_volume_specific_enthalpy[slice_L],
                phasic_volume_specific_enthalpy[slice_R])
        else:
            raise NotImplementedError
        
        return phasic_volume_specific_enthalpy_cf

    def compute_kinetic_energy_cf_xi(
            self,
            primitives: Array,
            axis: int,
            regularization_flux: Array = None
            ) -> Array:

        slice_p, slice_pp = self.slices_LR[axis]
        vel_ids = self.equation_information.velocity_ids
        vel_slices = self.equation_information.velocity_slices  

        if self.reconstruction_type_kinetic_energy == "CENTRAL":
            # Kinetic energy: k_{i+1/2} = 0.5 * (k_i + k_{i+1})
            #                           = 0.5 * (0.5 * u_i * u_i + 0.5 * u_{i+1} * u_{i+1})
            # kinetic_energy = 0.0
            # for i in self.active_axes_indices:
            #     kinetic_energy += primitives[vel_ids[i]] * primitives[vel_ids[i]]
            # kinetic_energy *= 0.5
            # kinetic_energy_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(kinetic_energy, axis)

            velocity_cf = self.reconstruction_stencil_conservatives.reconstruct_xi(primitives[vel_slices], axis)
            kinetic_energy_cf = 0.5 * jnp.sum(velocity_cf * velocity_cf, axis=0)
        
        elif self.reconstruction_type_kinetic_energy == "MIXED":
            # Kinetic energy: k_{i+1/2} = 0.5 * u_{i} * u_{i+1}
            kinetic_energy_cf = 0.0
            for i in self.active_axes_indices:
                kinetic_energy_cf += primitives[vel_ids[i]][slice_p] * primitives[vel_ids[i]][slice_pp]
            kinetic_energy_cf *= 0.5
        
        elif self.reconstruction_type_kinetic_energy == "UPWIND":
            raise NotImplementedError

        else:
            raise NotImplementedError

        return kinetic_energy_cf
