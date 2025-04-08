from functools import partial
from typing import Dict, Union

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.domain.helper_functions import trim_cell_sizes
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup.forcings import GeometricSourceSetup

Array = jax.Array

class SourceTermSolver:
    """ The SourceTermSolver class manages source term contributions
    on the right-hand-side of NSE. Contributions have to be activated
    in the active_physics section in the numerical setup file.

    This includes amongst others:
    1) Gravitational force
    2) Heat flux
    3) Viscous flux
    4) Species diffusion flux
    """

    def __init__(
            self, 
            numerical_setup: NumericalSetup,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            gravity: Array,
            geometric_source: GeometricSourceSetup,
            domain_information: DomainInformation,
            ) -> None:

        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.domain_information = domain_information

        self.equation_type = self.equation_information.equation_type
        active_physics = numerical_setup.active_physics
        self.is_convective_flux = active_physics.is_convective_flux
        self.is_viscous_flux = active_physics.is_viscous_flux
        self.is_heat_flux = active_physics.is_heat_flux
        self.is_volume_force = active_physics.is_volume_force
        self.is_geometric_source = active_physics.is_geometric_source
        self.is_surface_tension = active_physics.is_surface_tension
        self.is_viscous_heat_production = active_physics.is_viscous_heat_production

        global_cell_sizes_halos = domain_information.get_global_cell_sizes_halos()

        if numerical_setup.active_physics.is_viscous_flux:
            # NOTE stencils for computing cell-face tangential derivatives
            # at cell-face locations. Used for off-diagonal components of
            # the velocity gradient tensor.
            dissipative_fluxes_setup = numerical_setup.conservatives.dissipative_fluxes
            derivative_stencil_center = dissipative_fluxes_setup.derivative_stencil_center
            reconstruct_stencil_duidxi = dissipative_fluxes_setup.reconstruction_stencil
            offset = reconstruct_stencil_duidxi.required_halos
            self.derivative_stencil_center: SpatialDerivative = derivative_stencil_center(
                nh=domain_information.nh_conservatives, 
                inactive_axes=domain_information.inactive_axes, 
                offset=offset,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=global_cell_sizes_halos)

            trim = domain_information.nh_conservatives - offset
            global_cell_sizes_halos_trimmed = trim_cell_sizes(
                global_cell_sizes_halos, trim)
            self.reconstruct_stencil_duidxi: SpatialReconstruction = reconstruct_stencil_duidxi(
                nh=offset,
                inactive_axes=domain_information.inactive_axes, 
                offset=0,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=global_cell_sizes_halos_trimmed)
            
            if dissipative_fluxes_setup.is_laplacian:
                first_derivative_stencil_center = dissipative_fluxes_setup.derivative_stencil_center
                self.first_derivative_stencil_center_domain: SpatialDerivative = first_derivative_stencil_center(
                    nh=domain_information.nh_conservatives, 
                    inactive_axes=domain_information.inactive_axes, 
                    is_mesh_stretching=domain_information.is_mesh_stretching,
                    cell_sizes=global_cell_sizes_halos)
                
                second_derivative_stencil_center = dissipative_fluxes_setup.second_derivative_stencil_center
                self.second_derivative_stencil_center_domain: SpatialDerivative = second_derivative_stencil_center(
                    nh=domain_information.nh_conservatives, 
                    inactive_axes=domain_information.inactive_axes, 
                    is_mesh_stretching=domain_information.is_mesh_stretching,
                    cell_sizes=global_cell_sizes_halos) 


        else:
            derivative_stencil_center = None
            reconstruct_stencil_duidxi = None
        
        if any((self.is_viscous_flux, self.is_heat_flux)):
            dissipative_fluxes_setup = numerical_setup.conservatives.dissipative_fluxes
            derivative_stencil_face = dissipative_fluxes_setup.derivative_stencil_face
            self.derivative_stencil_face: SpatialDerivative = derivative_stencil_face(
                nh=domain_information.nh_conservatives, 
                inactive_axes=domain_information.inactive_axes, 
                offset=0,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=global_cell_sizes_halos)
            
            reconstruct_stencil_ui = dissipative_fluxes_setup.reconstruction_stencil
            self.reconstruct_stencil_ui: SpatialReconstruction = reconstruct_stencil_ui(
                nh=domain_information.nh_conservatives, 
                inactive_axes=domain_information.inactive_axes, 
                offset=0,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=global_cell_sizes_halos)
        else:
            derivative_stencil_face = None
            reconstruct_stencil_ui = None

        self.gravity = gravity
        
        if self.is_geometric_source:
            if self.equation_type in ("DIFFUSE-INTERFACE-4EQM"):
                raise NotImplementedError
            self.symmetry_type = geometric_source.symmetry_type
            self.symmetry_axis = geometric_source.symmetry_axis
            self.symmetry_axis_id = domain_information.axis_to_axis_id[self.symmetry_axis]
            self.radial_axis = [axis for axis in domain_information.active_axes if axis != self.symmetry_axis][0]
            self.radial_axis_id = domain_information.axis_to_axis_id[self.radial_axis]

            dissipative_fluxes_setup = numerical_setup.conservatives.dissipative_fluxes
            derivative_stencil_center = dissipative_fluxes_setup.derivative_stencil_center
            self.derivative_stencil_center_geometric_source_term: SpatialDerivative = derivative_stencil_center(
                nh=domain_information.nh_conservatives, 
                inactive_axes=domain_information.inactive_axes, 
                offset=0,
                is_mesh_stretching=domain_information.is_mesh_stretching,
                cell_sizes=global_cell_sizes_halos) 

        nx, ny, nz = domain_information.global_number_of_cells
        sx, sy, sz = domain_information.split_factors
        if self.equation_information.levelset_model == "FLUID-FLUID":
            self.shape_fluxes = [
                (3, 2, int(nx/sx+1), int(ny/sy), int(nz/sz)),
                (3, 2, int(nx/sx), int(ny/sy+1), int(nz/sz)),
                (3, 2, int(nx/sx), int(ny/sy), int(nz/sz+1))]
        else:
            self.shape_fluxes = [
                (3, int(nx/sx+1), int(ny/sy), int(nz/sz)),
                (3, int(nx/sx), int(ny/sy+1), int(nz/sz)),
                (3, int(nx/sx), int(ny/sy), int(nz/sz+1))]

        self.ids_mass = self.equation_information.ids_mass
        self.s_mass = self.equation_information.s_mass
        self.ids_velocity = self.equation_information.ids_velocity
        self.s_velocity = self.equation_information.s_velocity
        self.ids_energy = self.equation_information.ids_energy
        self.s_energy = self.equation_information.s_energy
        self.ids_volume_fraction = self.equation_information.ids_volume_fraction
        self.s_volume_fraction = self.equation_information.s_volume_fraction

    def compute_gravity_forces(
            self,
            conservatives: Array
            ) -> Array:
        """Computes flux due to gravitational force.

        :param conservatives: Buffer of conservative variables.
        :type conservatives: Array
        :return: Buffer with gravitational forces.
        :rtype: Array
        """
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives

        density = self.material_manager.get_density(conservatives[...,nhx,nhy,nhz])
        density = jnp.expand_dims(density, axis=0)
        momentum = conservatives[self.s_velocity,...,nhx,nhy,nhz]

        momentum_contribution = jnp.einsum("ij..., jk...->ik...", self.gravity.reshape(3,1), density)
        energy_contribution = jnp.einsum("ij..., jk...->ik...", self.gravity.reshape(1,3), momentum)
        gravity_forces = jnp.concatenate([momentum_contribution, energy_contribution], axis=0)

        return gravity_forces

    def compute_heat_flux_xi(
            self,
            temperature: Array,
            primitives: Array,
            axis: int,
        ) -> Array:
        """Computes the heat flux in axis direction.
        
        q = - \lambda \nabla T

        :param temperature: Buffer with temperature.
        :type temperature: Array
        :param axis: Spatial direction along which the heat flux is calculated.
        :type axis: int
        :return: Heat flux in axis direction.
        :rtype: Array
        """
        cell_sizes = self.domain_information.get_device_cell_sizes()
        equation_type = self.equation_information.equation_type

        temperature_at_cf = self.reconstruct_stencil_ui.reconstruct_xi(temperature, axis)
        volume_fraction_at_cf = None
        mass_fraction_at_cf = None

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            volume_fraction_at_cf = self.reconstruct_stencil_ui.reconstruct_xi(
                primitives[self.s_volume_fraction], axis)

        thermal_conductivity = self.material_manager.get_thermal_conductivity(
            temperature_at_cf,
            None,
            None,
            None,
            volume_fraction_at_cf,
            mass_fraction_at_cf)

        temperature_grad = self.derivative_stencil_face.derivative_xi(
            temperature, cell_sizes[axis], axis)
        heat_flux_xi = -thermal_conductivity * temperature_grad

        return heat_flux_xi

    def compute_heat_term_xi(
            self,
            temperature: Array,
            primitives: Array,
            axis: int,
            ) -> Array:
        raise NotImplementedError

    def compute_viscous_flux_xi(
            self,
            primitives: Array,
            temperature: Array,
            axis: int,
        ) -> Array:
        """Computes viscous flux in one spatial direction

        vel_grad = [
            du/dx du/dy du/dz
            dv/dx dv/dy dv/dz
            dw/dx dw/dy dw/dz
        ]

        :param temperature: Buffer of temperature.
        :type temperature: Array
        :param axis: Axis along which the viscous flux is computed.
        :type axis: int
        :return: Viscous flux along axis direction.
        :rtype: Array
        """
        active_axes_indices = self.domain_information.active_axes_indices
        equation_type = self.equation_information.equation_type
        no_fluids = self.equation_information.no_fluids

        temperature_at_cf = self.reconstruct_stencil_ui.reconstruct_xi(temperature, axis)
        volume_fraction_at_cf = None
        mass_fraction_at_cf = None

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            volume_fraction_at_cf = self.reconstruct_stencil_ui.reconstruct_xi(
                primitives[self.s_volume_fraction], axis)            

        dynamic_viscosity = self.material_manager.get_dynamic_viscosity(
            temperature_at_cf, 
            None,
            None,
            volume_fraction_at_cf,
            mass_fraction_at_cf)
        bulk_viscosity = self.material_manager.get_bulk_viscosity(
            temperature_at_cf,
            None,
            None,
            volume_fraction_at_cf,
            mass_fraction_at_cf)

        velocity = primitives[self.s_velocity]
        velocity_gradient = self.compute_velocity_gradient(velocity, axis)
        tau_j = self.compute_tau(velocity_gradient, dynamic_viscosity, bulk_viscosity, axis)
        velocity_at_xj = self.reconstruct_stencil_ui.reconstruct_xi(velocity, axis)

        if self.is_viscous_heat_production:
            vel_tau_at_xj = 0.0
            for k in active_axes_indices:
                vel_tau_at_xj += tau_j[k] * velocity_at_xj[k]
        else:
            vel_tau_at_xj = jnp.zeros_like(tau_j[0])

        fluxes_xj = jnp.stack([*tau_j, vel_tau_at_xj])

        return fluxes_xj

    def compute_viscous_term_xi(
            self,
            primitives: Array,
            temperature: Array,
            axis: int,
            ) -> Array:

        active_axes_indices = self.domain_information.active_axes_indices
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        cell_sizes = self.domain_information.get_device_cell_sizes()
        cell_sizes_xi = cell_sizes[axis]

        dynamic_viscosity = self.material_manager.get_dynamic_viscosity(
            temperature, primitives) 
        bulk_viscosity = self.material_manager.get_bulk_viscosity(
            temperature, primitives)

        mu_1 = dynamic_viscosity
        mu_2 = bulk_viscosity - 2.0 / 3.0 * dynamic_viscosity

        velocity = primitives[self.s_velocity]
        velocity_gradient = self.compute_velocity_gradient_cc(velocity)
        second_derivative_tensor_slice = self.compute_second_derivative_tensor_slice_cc(velocity, axis)
        velocity = velocity[...,nhx,nhy,nhz]
       
        # NOTE shear contribution
        if isinstance(mu_1, float):
            dmu_1_dxi = 0.0
        else:
            dmu_1_dxi = self.first_derivative_stencil_center_domain.derivative_xi(mu_1, cell_sizes_xi, axis)
            mu_1 = mu_1[nhx,nhy,nhz]
        shear_contribution = \
            (velocity_gradient[:,axis] + velocity_gradient[axis,:]) * dmu_1_dxi \
            + (second_derivative_tensor_slice[:,axis] + second_derivative_tensor_slice[axis,:]) * mu_1
        viscous_source_term = shear_contribution

        # NOTE volumetric contribution
        if isinstance(mu_2, float):
            dmu_2_dxi = 0.0
        else:
            dmu_2_dxi = self.first_derivative_stencil_center_domain.derivative_xi(mu_2, cell_sizes_xi, axis)
            mu_2 = mu_2[nhx,nhy,nhz]
        volumetric_contribution = \
            sum([velocity_gradient[ii,ii] for ii in active_axes_indices]) * dmu_2_dxi \
            + sum([second_derivative_tensor_slice[ii,ii] for ii in active_axes_indices]) * mu_2
        viscous_source_term = viscous_source_term.at[axis].add(volumetric_contribution)

        if self.is_viscous_heat_production:
            vel_tau = 0.0
            for ii in active_axes_indices:
                vel_tau += viscous_source_term[ii] * velocity[ii]
                vel_tau += (velocity_gradient[ii,axis] + velocity_gradient[axis,ii]) * mu_1 * velocity_gradient[ii,axis]
            vel_tau += sum([velocity_gradient[ii,ii] for ii in active_axes_indices]) * mu_2 * velocity_gradient[axis,axis]

        else:
            vel_tau = jnp.zeros_like(viscous_source_term[0])

        viscous_source_term = jnp.stack([*viscous_source_term, vel_tau])

        return viscous_source_term


    def compute_velocity_gradient(
            self,
            velocity: Array,
            axis: int
            ) -> Array:
        """Computes the velocity gradient at cell faces
        in axis direction.

        Here, the velocity gradient is defined as

        \partial u_i / \partial x_j

        velocity_gradient = [
            du/dx du/dy du/dz
            dv/dx dv/dy dv/dz
            dw/dx dw/dy dw/dz
        ]

        NOTE Often the velocity gradient tensor is defined
        as \partial u_j / \partial x_i, which is simply the
        transpose of the definition used here.

        :return: _description_
        :rtype: _type_
        """

        active_axes_indices = self.domain_information.active_axes_indices
        velocity_gradient = []
        for i in range(3):
            if i in active_axes_indices:
                velocity_gradient.append(self.compute_xi_derivatives_at_xj(velocity, i, axis))
            else:
                velocity_gradient.append(jnp.zeros(self.shape_fluxes[axis]))
        velocity_gradient = jnp.stack(velocity_gradient, axis=1)
        return velocity_gradient

    def compute_velocity_gradient_cc(self, velocity: Array) -> Array:
        """Computes the velocity gradient at cell centers.

        :param velocity: _description_
        :type velocity: Array
        :return: _description_
        :rtype: Array
        """

        active_axes_indices = self.domain_information.active_axes_indices
        cell_sizes = self.domain_information.get_device_cell_sizes()
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives

        velocity_gradient = []
        for axis in range(3):
            if axis in active_axes_indices:
                velocity_gradient.append(self.first_derivative_stencil_center_domain.derivative_xi(
                    velocity, cell_sizes[axis], axis))
            else:
                velocity_gradient.append(jnp.zeros_like(velocity[...,nhx,nhy,nhz]))
        velocity_gradient = jnp.stack(velocity_gradient, axis=1)

        return velocity_gradient

    def compute_second_derivative_tensor_slice_cc(self, velocity: Array, axis_j: int) -> Array:
        """

        :param velocity: _description_
        :type velocity: Array
        :param axis_j: _description_
        :type axis_j: int
        :return: _description_
        :rtype: Array
        """

        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        cell_sizes = self.domain_information.get_device_cell_sizes()

        second_derivative_tensor_slice = []
        for axis_i in range(3):
            if axis_i in self.domain_information.active_axes_indices:
                if axis_i == axis_j:
                    u_xi_xj = self.second_derivative_stencil_center_domain.derivative_xi(
                        velocity, cell_sizes[axis_i], axis_i)
                else:
                    u_xi_xj = self.second_derivative_stencil_center_domain.derivative_xi_xj(
                        velocity, cell_sizes[axis_i], cell_sizes[axis_j], axis_i, axis_j)
            else:
                u_xi_xj = jnp.zeros_like(velocity[...,nhx,nhy,nhz])
            second_derivative_tensor_slice.append(u_xi_xj)
        second_derivative_tensor_slice = jnp.stack(second_derivative_tensor_slice, axis=1)

        return second_derivative_tensor_slice

    def compute_tau(
            self,
            vel_grad: Array,
            dynamic_viscosity: Array, 
            bulk_viscosity: Array,
            axis: int
            ) -> Array:
        """Computes the stress tensor at a cell face in axis direction.
        tau_axis = [tau_axis0, tau_axis1, tau_axis2]

        :param vel_grad: Buffer of velocity gradient. Shape is 3 x 3 (x 2) x Nx x Ny x Nz 
        :type vel_grad: Array
        :param dynamic_viscosity: Buffer of dynamic viscosity.
        :type dynamic_viscosity: Array
        :param bulk_viscosity: Buffer of bulk viscosity.
        :type bulk_viscosity: Array
        :param axis: Cell face direction at which viscous stresses are calculated.
        :type axis: int
        :return: Buffer of viscous stresses.
        :rtype: Array
        """

        active_axes_indices = self.domain_information.active_axes_indices

        mu_1 = dynamic_viscosity
        mu_2 = bulk_viscosity - 2.0 / 3.0 * dynamic_viscosity
        tau_list = [
            mu_1 * (vel_grad[axis,0] + vel_grad[0,axis]) if axis in active_axes_indices and 0 in active_axes_indices else jnp.zeros(vel_grad.shape[2:]), 
            mu_1 * (vel_grad[axis,1] + vel_grad[1,axis]) if axis in active_axes_indices and 1 in active_axes_indices else jnp.zeros(vel_grad.shape[2:]),
            mu_1 * (vel_grad[axis,2] + vel_grad[2,axis]) if axis in active_axes_indices and 2 in active_axes_indices else jnp.zeros(vel_grad.shape[2:]) 
        ]
        tau_list[axis] += mu_2 * sum([vel_grad[k,k] for k in active_axes_indices])
        return jnp.stack(tau_list)

    def compute_xi_derivatives_at_xj(
            self,
            primitives: Array,
            axis_i: int,
            axis_j: int
            ) -> Array:
        """Computes the spatial derivative in axis_i direction
        at the cell face in axis_j direction.

        If the spatial direction of the cell face at which the derivative is evaluated
        corresponds with the direction of the derivative, direct evaluation via FD
        at the cell-face.

        If the spatial direction of the cell face at which the derivative is evaluated
        does not correspond with the direction of the derivative:
        1) Compute derivative at cel center
        2) Interpolate derivative from cell center to cell face

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param axis_i: Spatial direction wrt which the derivative is taken.
        :type axis_i: int
        :param axis_j: Spatial direction along which derivative is evaluated.
        :type axis_j: int
        :return: Derivative wrt axis_i direction at cell face in axis_j direction.
        :rtype: Array
        """

        cell_sizes = self.domain_information.get_device_cell_sizes()
        cell_sizes_xi = cell_sizes[axis_i]

        if self.domain_information.is_mesh_stretching[axis_i]:
            cell_centers_difference = self.domain_information.get_device_cell_centers_difference()
            cell_centers_difference_xi = cell_centers_difference[axis_i]
        else:
            cell_centers_difference_xi = cell_sizes_xi
        
        if axis_i == axis_j:
            deriv_xi_at_xj = self.derivative_stencil_face.derivative_xi(primitives, cell_centers_difference_xi, axis_i)
        else:
            deriv_xi_at_c = self.derivative_stencil_center.derivative_xi(primitives, cell_sizes_xi, axis_i)
            deriv_xi_at_xj = self.reconstruct_stencil_duidxi.reconstruct_xi(deriv_xi_at_c, axis_j)
        return deriv_xi_at_xj
    
    def compute_geometric_source_terms(
            self,
            conservatives: Array,
            primitives: Array,
            temperature: Array,
            ) -> Array:
        """Computes geometric source terms due to symmetries.
        1) Axisymmetric (2D)
        2) Cylindrical symmetry (2D)
        3) Spherical symmetry (1D)

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param temperature: _description_
        :type temperature: Array
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        cell_sizes = self.domain_information.get_device_cell_sizes()
        radial_axis_id = self.radial_axis_id
        symmetry_axis_id = self.symmetry_axis_id
        active_axes_indices = self.domain_information.active_axes_indices

        geometric_source = jnp.zeros_like(conservatives[...,nhx,nhy,nhz])

        if self.symmetry_type == "AXISYMMETRIC":
            # Cylindrical formulations with u_theta = 0 which greatly
            # simplifies the resulting expressions
            radial_coord = self.domain_information.get_device_cell_centers()[radial_axis_id]
            one_radial_coord = 1.0 / radial_coord

            if self.is_convective_flux:
                radial_velocity = primitives[self.ids_velocity[radial_axis_id],...,nhx,nhy,nhz]
                tmp = conservatives.at[self.ids_energy].add(primitives[self.ids_energy])[...,nhx,nhy,nhz]
                geometric_source -= one_radial_coord * radial_velocity * tmp

                if self.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM":
                    # NOTE there is no source term for the advection equation
                    # of the volume fraction field due to u * grad(alpha)
                    geometric_source = geometric_source.at[self.ids_volume_fraction].set(0.0)
            
            if self.is_viscous_flux:
                u_r = primitives[self.ids_velocity[radial_axis_id]]
                u_z = primitives[self.ids_velocity[symmetry_axis_id]]
                mu_1 = self.material_manager.get_dynamic_viscosity(temperature, primitives)
                mu_2 = self.material_manager.get_bulk_viscosity(temperature, primitives) - 2.0 / 3.0 * mu_1

                du_r_dr = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    u_r, cell_sizes[radial_axis_id], radial_axis_id)
                du_r_dz = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    u_r, cell_sizes[symmetry_axis_id], symmetry_axis_id)
                du_z_dr = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    u_z, cell_sizes[radial_axis_id], radial_axis_id)
                du_z_dz = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    u_z, cell_sizes[symmetry_axis_id], symmetry_axis_id)
                dmu_2_dr = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    mu_2, cell_sizes[radial_axis_id], radial_axis_id)
                dmu_2_dz = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    mu_2, cell_sizes[symmetry_axis_id], symmetry_axis_id)

                u_r = u_r[...,nhx,nhy,nhz]
                u_z = u_z[...,nhx,nhy,nhz]
                mu_1 = mu_1[...,nhx,nhy,nhz]
                mu_2 = mu_2[...,nhx,nhy,nhz]

                geometric_source = geometric_source.at[self.ids_velocity[radial_axis_id]].add(
                    one_radial_coord * ((2 * mu_1 + mu_2) * du_r_dr + u_r * (dmu_2_dr - one_radial_coord * (2 * mu_1 + mu_2)))
                )
                geometric_source = geometric_source.at[self.ids_velocity[symmetry_axis_id]].add(
                    one_radial_coord * ((mu_1 + mu_2) * du_r_dz + mu_1 * du_z_dr + u_r * dmu_2_dz)
                )
                geometric_source = geometric_source.at[self.ids_energy].add(
                    one_radial_coord * (
                    u_r * ((2 * mu_1 + 3 * mu_2) * du_r_dr + 2 * mu_2 * du_z_dz) \
                    + u_z * ((mu_1 + mu_2) * du_r_dz + mu_1 * du_z_dr) \
                    + u_r * (u_r * dmu_2_dr + u_z * dmu_2_dz))
                )

            if self.is_heat_flux:
                thermal_conductivity = self.material_manager.get_thermal_conductivity(temperature, primitives)
                temperature_dr = self.derivative_stencil_center_geometric_source_term.derivative_xi(
                    temperature, cell_sizes[radial_axis_id], radial_axis_id)
                geometric_source = geometric_source.at[self.ids_energy].add(
                    one_radial_coord * thermal_conductivity * temperature_dr)

            if self.is_surface_tension:
                raise NotImplementedError

        elif self.symmetry_type == "CYLINDRICAL":
            # TODO symmetry
            # TODO u_theta as function of (r,z)
            if self.is_convective_flux:
                radial_axis_id = active_axes_indices[0]
                radial_velocity = primitives[self.ids_velocity[radial_axis_id],...,nhx,nhy,nhz]
                tmp = conservatives.at[self.ids_energy].add(primitives[self.ids_energy])[...,nhx,nhy,nhz]
                geometric_source -= one_radial_coord * radial_velocity * tmp

            if self.is_heat_flux:
                raise NotImplementedError

            if self.is_surface_tension:
                raise NotImplementedError

        elif self.symmetry_type == "SPHERICAL":
            # TODO symmetry
            if self.is_convective_flux:
                radial_axis_id = active_axes_indices[0]
                radial_velocity = primitives[self.ids_velocity[radial_axis_id],...,nhx,nhy,nhz]
                tmp = conservatives.at[self.ids_energy].add(primitives[self.ids_energy])[...,nhx,nhy,nhz]
                geometric_source -= 2 * one_radial_coord * radial_velocity * tmp

            if self.is_heat_flux:
                raise NotImplementedError

            if self.is_surface_tension:
                raise NotImplementedError

        else:
            raise NotImplementedError

        return geometric_source
