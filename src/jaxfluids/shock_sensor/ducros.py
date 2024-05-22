import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.shock_sensor.shock_sensor import ShockSensor
from jaxfluids.stencils.derivative.deriv_center_2 import DerivativeSecondOrderCenter
from jaxfluids.stencils.derivative.deriv_center_adap_2 import DerivativeSecondOrderAdapCenter
from jaxfluids.stencils.reconstruction.central_adap_2 import CentralSecondOrderAdapReconstruction


class Ducros(ShockSensor):
    """Ducros Shock Sensor

    fs = jnp.where(div / (div + curl + self.epsilon_s) >= 0.95, 1, 0)

    """

    def __init__(self, domain_information: DomainInformation) -> None:
        super().__init__(domain_information)

        nh_conservatives = domain_information.nh_conservatives
        inactive_axes = domain_information.inactive_axes
        is_mesh_stretching = domain_information.is_mesh_stretching
        cell_sizes_halos = domain_information.get_global_cell_sizes_halos()

        offset=1
        self.derivative_stencil_center = DerivativeSecondOrderAdapCenter(
            nh=nh_conservatives,
            inactive_axes=inactive_axes,
            offset=offset,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_halos)
        
        self.reconstruction_stencil_face = CentralSecondOrderAdapReconstruction(
            nh=nh_conservatives,
            inactive_axes=inactive_axes,
            offset=offset,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes_halos)

        nx, ny, nz = domain_information.global_number_of_cells
        sx, sy, sz = domain_information.split_factors
        active_axes_indices = domain_information.active_axes_indices
        self.shape_vel_grad = (3, 
            int(nx/sx)+2*offset if 0 in active_axes_indices else 1, 
            int(ny/sy)+2*offset if 1 in active_axes_indices else 1, 
            int(nz/sz)+2*offset if 2 in active_axes_indices else 1
            )

        nhx = jnp.s_[:] if "x" in domain_information.inactive_axes else jnp.s_[offset:-offset]    
        nhy = jnp.s_[:] if "y" in domain_information.inactive_axes else jnp.s_[offset:-offset]    
        nhz = jnp.s_[:] if "z" in domain_information.inactive_axes else jnp.s_[offset:-offset]

        self.s_ = [
            [jnp.s_[:-offset, nhy     , nhz     ], jnp.s_[offset:, nhy    , nhz    ]],
            [jnp.s_[nhx     , :-offset, nhz     ], jnp.s_[nhx    , offset:, nhz    ]],
            [jnp.s_[nhx     , nhy     , :-offset], jnp.s_[nhx    , nhy    , offset:]],
        ]

    def compute_sensor_function(self, vels: Array, axis: int) -> Array:
        if len(self.active_axes_indices) == 1:
            fs = 1.0
        else:
            vel_grad = self.compute_velocity_derivatives(vels)
            # EVALUATE DIV AND CURL AT CELL CENTER
            div = vel_grad[0,0] + vel_grad[1,1] + vel_grad[2,2]
            curl_1 = vel_grad[1,2] - vel_grad[2,1]
            curl_2 = vel_grad[2,0] - vel_grad[0,2]
            curl_3 = vel_grad[0,1] - vel_grad[1,0]            

            # CALCULATE DIV AND CURL AT CELL FACE
            div = div[self.s_[axis][0]] + div[self.s_[axis][1]]
            curl_1 = curl_1[self.s_[axis][0]] + curl_1[self.s_[axis][1]]
            curl_2 = curl_2[self.s_[axis][0]] + curl_2[self.s_[axis][1]]
            curl_3 = curl_3[self.s_[axis][0]] + curl_3[self.s_[axis][1]]
            # div = self.reconstruction_stencil_face.reconstruct_xi(div, axis)
            # curl_1 = self.reconstruction_stencil_face.reconstruct_xi(curl_1, axis)
            # curl_2 = self.reconstruction_stencil_face.reconstruct_xi(curl_2, axis)
            # curl_3 = self.reconstruction_stencil_face.reconstruct_xi(curl_3, axis)

            div  = jnp.abs(div)
            curl = jnp.sqrt(curl_1 * curl_1 + curl_2 * curl_2 + curl_3 * curl_3) 

            fs = jnp.where(div / (div + curl + self.eps) >= 0.95, 1, 0)

        return fs    

    def compute_velocity_derivatives(self, vels: Array) -> Array:
        """Computes the velocity gradient.
        Note that velocity gradients and especially curl and divergence
        are often used to determine the presence of shocks.

        vel_grad = [ du/dx dv/dx dw/dx
                     du/dy dv/dy dw/dy
                     du/dz dv/dz dw/dz ]

        :param vels: Buffer of velocities.
        :type vels: Array
        :return: Buffer of the velocity gradient.
        :rtype: Array
        """

        vel_grad = []
        for i in range(3):
            if i in self.active_axes_indices:
                vel_grad_i = self.derivative_stencil_center.derivative_xi(vels, self.cell_sizes[i], i)
            else:
                vel_grad_i = jnp.zeros(self.shape_vel_grad)
            
            vel_grad.append(vel_grad_i)

        vel_grad = jnp.stack(vel_grad)
        # vel_grad = jnp.stack([self.derivative_stencil_center.derivative_xi(vels, self.cell_sizes[i], i) if i in self.active_axes_indices else jnp.zeros(self.shape_vel_grad) for i in range(3)])
        return vel_grad