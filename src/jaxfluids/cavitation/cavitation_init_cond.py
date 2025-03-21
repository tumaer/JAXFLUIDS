from typing import Tuple, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.domain.helper_functions import split_and_shard_buffer_np
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionCavitation

Array = jax.Array

class CavitationInitializationManager:
    """
    """
    # TODO: @deniz check this, implemented by Andre

    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            initial_condition_cavitation: InitialConditionCavitation,
            ) -> None:

        self.domain_information = domain_information
        self.cavitation_material = material_manager.material

        self.initial_condition_cavitation = initial_condition_cavitation
        
    def get_cavitation_initial_condition(
            self,
            mesh_grid: List
            ) -> Array:
        """_summary_

        :param mesh_grid: _description_
        :type mesh_grid: List
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        is_parallel = self.domain_information.is_parallel
        cell_centers = self.domain_information.get_local_cell_centers()
        domain_size = self.domain_information.get_local_domain_size()
        active_axes_indices = self.domain_information.active_axes_indices
        split_factors = self.domain_information.split_factors
    
        cavitation_case = self.initial_condition_cavitation.case
        parameters = self.initial_condition_cavitation.parameters

        if cavitation_case == "SINGLE_BUBBLE_2D":
            R_0 = parameters.bubble_radius
            x_0 = parameters.bubble_origin_x 
            y_0 = parameters.bubble_origin_y 
            z_0 = parameters.bubble_origin_z 
            alpha_bubble = parameters.vapor_volume_fraction
            p_inf = parameters.driving_pressure
            is_one_r = parameters.is_one_r
            is_barotropic = parameters.is_barotropic

            X, Y = mesh_grid
            mask_bubble = jnp.where((X - x_0)**2 + (Y - y_0)**2 <= R_0**2, 1.0, 0.0)
            density_bubble = self.cavitation_material.get_mixture_density(alpha_bubble)
            pressure_bubble, _ = self.cavitation_material.get_pressure(None, density_bubble)
            r = jnp.sqrt((X - x_0)**2 + (Y - y_0)**2)

            pressure_liquid = p_inf
            if is_one_r:
                pressure_liquid += (pressure_bubble - p_inf) * R_0 / (r + 1e-10)
            pressure_init = mask_bubble * pressure_bubble + (1 - mask_bubble) * pressure_liquid
            
            density_liquid = self.cavitation_material.get_density(pressure_liquid)
            density_init = mask_bubble * density_bubble + (1 - mask_bubble) * density_liquid
            
            velX_init = velY_init = velZ_init = jnp.zeros_like(X)
            primitives_init = jnp.stack([density_init, velX_init, velY_init, velZ_init, pressure_init], axis=0)
            
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(ncols=3)
            # im0 = ax[0].pcolormesh(X[:,:,0], Y[:,:,0], density_init[:,:,0])
            # im1 = ax[1].pcolormesh(X[:30,:30,0], Y[:30,:30,0], density_init[:30,:30,0])
            # im2 = ax[2].pcolormesh(X[:,:,0], Y[:,:,0], pressure_init[:,:,0])
            # for axi in ax:
            #     axi.set_box_aspect(1.0)
            # plt.savefig("test.png")
            # plt.show()
            # plt.close()
            # print(primitives_init.shape)
            # print(jnp.min(density_init), jnp.max(density_init))
            # print(jnp.min(pressure_init), jnp.max(pressure_init))
            # print(self.domain_information.smallest_cell_size)
            # print(jnp.max(self.cavitation_material.get_speed_of_sound(pressure_init, density_init)))
            # exit()

        elif cavitation_case == "SINGLE_BUBBLE_3D":
            R_0 = parameters.bubble_radius
            x_0 = parameters.bubble_origin_x 
            y_0 = parameters.bubble_origin_y 
            z_0 = parameters.bubble_origin_z 
            alpha_bubble = parameters.vapor_volume_fraction
            p_inf = parameters.driving_pressure
            is_one_r = parameters.is_one_r
            is_barotropic = parameters.is_barotropic

            X, Y, Z = mesh_grid
            mask_bubble = jnp.where((X - x_0)**2 + (Y - y_0)**2 + (Z - z_0)**2 <= R_0**2, 1.0, 0.0)
            density_bubble = self.cavitation_material.get_mixture_density(alpha_bubble)
            pressure_bubble, _ = self.cavitation_material.get_pressure(None, density_bubble)
            r = jnp.sqrt((X - x_0)**2 + (Y - y_0)**2 + (Z - z_0)**2)

            pressure_liquid = p_inf
            if is_one_r:
                pressure_liquid += (pressure_bubble - p_inf) * R_0 / (r + 1e-10)
            pressure_init = mask_bubble * pressure_bubble + (1 - mask_bubble) * pressure_liquid
            
            density_liquid = self.cavitation_material.get_density(pressure_liquid)
            density_init = mask_bubble * density_bubble + (1 - mask_bubble) * density_liquid
            
            velX_init = velY_init = velZ_init = jnp.zeros_like(X)
            primitives_init = jnp.stack([density_init, velX_init, velY_init, velZ_init, pressure_init], axis=0)

        # elif cavitation_case == "CLOUD_BUBBLE_3D":

        #     R_0 = self.unit_handler.non_dimensionalize(self.cavitation_init_params["bubble_radius"], "length")
        #     nx, ny, nz = self.cavitation_init_params["bubble_domain_origin"]
        #     n_size = self.cavitation_init_params["bubble_domain_size"]
        #     bubble_count = self.cavitation_init_params["bubble_count"]
        #     bubble_size = self.cavitation_init_params["bubble_size_range"]
        #     bubble_size = self.unit_handler.non_dimensionalize(bubble_size, "length")
        #     alpha_bubble = self.cavitation_init_params["vapor_volume_fraction"]
        #     p_inf = self.unit_handler.non_dimensionalize(self.cavitation_init_params["driving_pressure"], "pressure")
        #     is_one_r = self.cavitation_init_params["is_one_r"]
        #     is_barotropic = self.cavitation_init_params["is_barotropic"]
        #     X, Y, Z = mesh_grid
        #     mask_bubble = jnp.zeros((X,Y,Z))

        #     for i in range(bubble_count):
        #         x_0 = np.random.uniform(low=nx-n_size, high=nx+n_size)
        #         y_0 = np.random.uniform(low=ny-n_size, high=ny+n_size)
        #         z_0 = np.random.uniform(low=nz-n_size, high=nz+n_size)
        #         #bubble_origin = [bubble_x,bubble_y,bubble_z]
        #         R_0 = np.random.uniform(low=bubble_size[0],high=bubble_size[1])
        #         bubble = jnp.where((X - x_0)**2 + (Y - y_0)**2 + (Z - z_0)**2 <= R_0**2, 1.0, 0.0)
        #         if ((bubble + mask_bubble) <= 1).all():
        #             mask_bubble = mask_bubble + bubble
        #         else:
        #             i-=1
                    
        #     density_bubble = self.cavitation_material.get_mixture_density(alpha_bubble)
        #     pressure_bubble = self.cavitation_material.get_pressure(None, density_bubble)
        #     r = jnp.sqrt((X - x_0)**2 + (Y - y_0)**2 + (Z - z_0)**2)
        #     pressure_liquid = p_inf + (pressure_bubble - p_inf) * R_0 / (r + 1e-10)
        #     #TODO
        #     pressure_init = mask_bubble * pressure_bubble + (1 - mask_bubble) * pressure_liquid
            
        #     density_liquid = self.cavitation_material.get_density(pressure_liquid)
        #     density_init = mask_bubble * density_bubble + (1 - mask_bubble) * density_liquid
            
        #     velX_init = velY_init = velZ_init = jnp.zeros_like(X)
        #     primitives_init = jnp.stack([density_init, velX_init, velY_init, velZ_init, pressure_init], axis=0)

        #     raise NotImplementedError

        else:
            raise NotImplementedError

        return primitives_init