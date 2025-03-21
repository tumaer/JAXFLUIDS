from functools import partial
from re import S
from typing import List, Union, Dict, Callable, Tuple

import jax
import jax.numpy as jnp
from numpy import maximum

from jaxfluids.levelset.creation.NACA_helper_functions import thickness_distribution, four_digit_airfoil, five_digit_airfoil

Array = jax.Array


class NACA_airfoils:

    def __init__(self, is_double_precision: bool) -> None:

        self.is_double_precision = is_double_precision

        self.digit_function_dict: Dict[str, Callable] = {
            4: four_digit_airfoil,
            5: five_digit_airfoil
        }

    @partial(jax.jit, static_argnums=(0,1,5))
    def NACA_airfoil(
            self, 
            digit_code: str, 
            scaling_factor: float, 
            chord_line: Array,
            mesh_coordinates: Array, 
            task: str
        ) -> Array | Tuple[Array]:
        """Computes the minimum distance to the upper/lower airfoil profile 
        for the specified chord line values for a NACA airfoil.

        :param digit_code: _description_
        :type digit_code: str
        :param chord_line: _description_
        :type chord_line: Array
        :param mesh_coordinates: _description_
        :type mesh_coordinates: Array
        :param task: _description_
        :type task: str
        :return: _description_
        :rtype: Array
        """

        chord_line = jnp.clip(chord_line, 0.0, 1.0)
        camber_line, theta = self.digit_function_dict[len(digit_code)](digit_code, chord_line)

        thickness = int(digit_code[-2:])/100.0
        y_t = thickness_distribution(thickness, chord_line)
        
        airfoil_y_upper = camber_line + y_t * jnp.cos(theta)
        airfoil_x_upper = chord_line - y_t * jnp.sin(theta)
        airfoil_y_lower = camber_line - y_t * jnp.cos(theta)
        airfoil_x_lower = chord_line + y_t * jnp.sin(theta)

        airfoil_y_upper *= scaling_factor
        airfoil_x_upper *= scaling_factor
        airfoil_y_lower *= scaling_factor
        airfoil_x_lower *= scaling_factor

        if task == "ESTIMATE":
    
            coordinates_airfoil_upper = jnp.stack([airfoil_x_upper, airfoil_y_upper], axis=1)
            coordinates_airfoil_lower = jnp.stack([airfoil_x_lower, airfoil_y_lower], axis=1) # TODO
            
            coordinates_airfoil_upper = coordinates_airfoil_upper.reshape(-1,2,1,1,1)
            coordinates_airfoil_lower = coordinates_airfoil_lower.reshape(-1,2,1,1,1)

            distance_vector_upper = coordinates_airfoil_upper - mesh_coordinates
            distance_vector_lower = coordinates_airfoil_lower - mesh_coordinates

            distance_upper = jnp.sqrt(jnp.sum(jnp.square(distance_vector_upper), axis=1))
            distance_lower = jnp.sqrt(jnp.sum(jnp.square(distance_vector_lower), axis=1))

            distance_upper_min = jnp.min(distance_upper, axis=0)
            distance_lower_min = jnp.min(distance_lower, axis=0)

            index_minimum_upper = jnp.argmin(distance_upper, axis=0)
            index_minimum_lower = jnp.argmin(distance_lower, axis=0)
            
            mask = jnp.where(distance_upper_min < distance_lower_min, 1, 0)
            index_minimum = mask * index_minimum_upper + (1 - mask) * index_minimum_lower

            chord_line_estimate = chord_line[index_minimum]

            distance_minimum = jnp.minimum(distance_upper_min, distance_lower_min)

            return chord_line_estimate, distance_minimum

        else:

            coordinates_airfoil_upper = jnp.stack([airfoil_x_upper, airfoil_y_upper], axis=0)
            coordinates_airfoil_lower = jnp.stack([airfoil_x_lower, airfoil_y_lower], axis=0)

            if task == "COORDINATES":
                return coordinates_airfoil_upper, coordinates_airfoil_lower

            elif task == "DISTANCE": 
                distance_vector_upper = coordinates_airfoil_upper - mesh_coordinates
                distance_vector_lower = coordinates_airfoil_lower - mesh_coordinates
                distance_upper = jnp.sqrt(jnp.sum(jnp.square(distance_vector_upper), axis=0))
                distance_lower = jnp.sqrt(jnp.sum(jnp.square(distance_vector_lower), axis=0))
                distance = jnp.minimum(distance_upper, distance_lower)

                return distance

    def compute_distance_estimate(
            self, 
            airfoil: str, 
            mesh_grid: Tuple[Array], 
            resolution: int,
            batch_size: int
            ) -> Tuple[Array, Array]:
        """_summary_

        :param airfoil: _description_
        :type airfoil: str
        :param mesh_grid: _description_
        :type mesh_grid: List
        :param resolution: _description_
        :type resolution: int
        :param batch_size: _description_
        :type batch_size: int
        :return: _description_
        :rtype: Array
        """
        
        digit_code = airfoil.split("_")[1]
        scaling_factor = float(airfoil.split("_")[2])
        mesh_coordinates = jnp.stack([mesh_grid[0], mesh_grid[1]], axis=0)

        # ESTIMATE - STARTING VALUES
        no_batches = int(jnp.ceil(resolution/batch_size))

        # DISCRETE CHORD LINE
        chord_line = jnp.linspace(0, 1, resolution)

        # BUFFER
        chord_line_estimate = jnp.zeros_like(mesh_grid[0])
        distance_estimate = 100 * jnp.ones_like(mesh_grid[0])

        print("COMPUTING CHORD LINE ESTIMATE")
        for i in range(no_batches):
            chord_line_input = chord_line[i*batch_size:(i+1)*batch_size]
            chord_line_estimate_new, distance_estimate_new = self.NACA_airfoil(
                digit_code, scaling_factor, chord_line_input, mesh_coordinates, "ESTIMATE")
            mask = jnp.where(distance_estimate_new < distance_estimate, 1, 0)
            chord_line_estimate = chord_line_estimate_new * mask + chord_line_estimate * (1 - mask)
            distance_estimate = distance_estimate_new * mask + distance_estimate * (1 - mask)
            if i % 10 == 0:
                print(f"PROCESSING BATCH {i:5d}/{no_batches:5d}")
        print(f"MIN DISTANCE ESTIMATE {jnp.min(distance_estimate):.5e}\n")

        return chord_line_estimate, distance_estimate

    def optimize_distance_estimate(
            self,
            airfoil: str,
            chord_line_estimate: Array,
            distance_estimate: Array,
            mesh_grid: Tuple[Array],
            eps: float,
            learning_rate: float,
            steps: int
        ) -> Array:
        """_summary_

        :param airfoil: _description_
        :type airfoil: str
        :param chord_line_estimate: _description_
        :type chord_line_estimate: Array
        :param distance_estimate: _description_
        :type distance_estimate: Array
        :param mesh_grid: _description_
        :type mesh_grid: List
        :param eps: _description_
        :type eps: float
        :param learning_rate: _description_
        :type learning_rate: float
        :param steps: _description_
        :type steps: int
        :return: _description_
        :rtype: Array
        """

        digit_code = airfoil.split("_")[1]
        scaling_factor = float(airfoil.split("_")[2])
        mesh_coordinates = jnp.stack([mesh_grid[0], mesh_grid[1]], axis=0)

        chord_line = chord_line_estimate

        print("OPTIMIZE DISTANCE WITH GRADIENT DESCENT")
        for i in range(steps):
            
            # COMPUTE DISTANCE DERIVATIVE W.R.T. CHORD LINE POSITION
            # TODO why not use AD?
            distance_1 = self.NACA_airfoil(digit_code, scaling_factor, chord_line + eps, mesh_coordinates, "DISTANCE")
            distance_2 = self.NACA_airfoil(digit_code, scaling_factor, chord_line - eps, mesh_coordinates, "DISTANCE")
            distance_first_derivative = (distance_1 - distance_2)/(2*eps)

            # GRADIENT DESCENT
            new_chord_line = chord_line - learning_rate * distance_first_derivative
            new_chord_line = jnp.clip(new_chord_line, 0.0, 1.0)

            chord_line_deviation = jnp.mean(jnp.abs(chord_line - new_chord_line))
            chord_line = new_chord_line
            distance = self.NACA_airfoil(digit_code, scaling_factor, chord_line, mesh_coordinates, "DISTANCE")
            distance_deviation = jnp.mean(jnp.abs(distance_estimate - distance))
            
            if i % 100 == 0:
                print(f"CHORD LINE DEVIATION: {chord_line_deviation:.16e}, DISTANCE DEVIATION TO ESTIMATE {distance_deviation:.16e}, ITERATION: {i:d}")

        # CHECK CONVERGENCE
        max_distance_deviation = jnp.max(jnp.abs(distance_estimate - distance))
        print(f"MAX DISTANCE DEVITATION TO ESTIMATE {max_distance_deviation:.5e}\n")
        assert_string = (
            "Newton scheme for NACA levelset generation failed. "
            f"The maximum deviation to the estimate is {max_distance_deviation:.5e}. "
            "Need better estimate for start value."
        )
        assert max_distance_deviation < 1e-2, assert_string
        deviation = jnp.clip(jnp.abs(distance-distance_estimate), 1e-16, 1e10)
        return distance

    def apply_sign(self, airfoil: str, distance: Array, mesh_grid: Tuple[Array], resolution: int) -> Array:
        """_summary_

        :param airfoil: _description_
        :type airfoil: str
        :param distance: _description_
        :type distance: Array
        :param mesh_grid: _description_
        :type mesh_grid: List
        :param resolution: _description_
        :type resolution: int
        :return: _description_
        :rtype: Array
        """

        digit_code = airfoil.split("_")[1]
        scaling_factor = float(airfoil.split("_")[2])

        chord_line = jnp.linspace(0,1,resolution)
        coordinates_upper, coordinates_lower = self.NACA_airfoil(digit_code, scaling_factor, chord_line, mesh_grid, "COORDINATES")  

        cell_centers_x = mesh_grid[0][:,0,0] # TODO hardcoded for chord_line = x axis

        # LOCATE CLOSEST AIRFOIL COORDINATES TO MESH CELL CENTERS
        argmin_upper_x = jnp.argmin(jnp.abs(coordinates_upper[0,:].reshape(-1,1) - cell_centers_x), axis=0)  
        argmin_lower_x = jnp.argmin(jnp.abs(coordinates_lower[0,:].reshape(-1,1) - cell_centers_x), axis=0)  
        upper_y = coordinates_upper[1,:][argmin_upper_x] * jnp.where((cell_centers_x >= 0) &  (cell_centers_x <= scaling_factor), 1, 0)
        lower_y = coordinates_lower[1,:][argmin_lower_x] * jnp.where((cell_centers_x >= 0) &  (cell_centers_x <= scaling_factor), 1, 0)

        # CHECK IF MESH CELL CENTER LIES WITHIN AIRFOIL 
        sign = jnp.where((mesh_grid[1] < upper_y.reshape(-1,1,1)) & (mesh_grid[1] > lower_y.reshape(-1,1,1)), -1, 1)
        signed_distance = sign * distance

        return signed_distance

    def compute_levelset(self, airfoil: str, mesh_grid: Tuple[Array], smallest_cell_size: float) -> Array:
        """Computes the levelset field for the specified airfoil.

        :return: _description_
        :rtype: Array
        """

        scaling_factor = float(airfoil.split("_")[2])

        resolution_for_estimate = int(10.0 * scaling_factor / smallest_cell_size) 
        batch_size = 50
        chord_line_estimate, distance_estimate = self.compute_distance_estimate(
            airfoil, mesh_grid, resolution_for_estimate, batch_size)
        
        eps = 1e-8 if self.is_double_precision else 1e-6
        learning_rate = 1e-8 if self.is_double_precision else 1e-6
        steps = 5000
        distance = self.optimize_distance_estimate(
            airfoil, chord_line_estimate, distance_estimate,
            mesh_grid, eps, learning_rate, steps)

        resolution_for_sign = int(12.0 * scaling_factor / smallest_cell_size) 
        levelset = self.apply_sign(airfoil, distance, mesh_grid, resolution_for_sign)

        # import matplotlib.pyplot as plt
        # import matplotlib
        # from mpl_toolkits.axes_grid1 import make_axes_locatable

        # deviation = jnp.clip(jnp.abs(distance_estimate - distance), 1e-16, 1e16)
        # norm = matplotlib.colors.LogNorm(vmin=jnp.min(deviation), vmax=jnp.max(deviation))
        # X,Y = jnp.squeeze(mesh_grid[0]),jnp.squeeze(mesh_grid[1])
        # fig, ax = plt.subplots(2,2)
        # ax[0,0].pcolormesh(X, Y, jnp.clip(jnp.squeeze(distance_estimate/smallest_cell_size), -10, 10), cmap="jet")
        # ax[0,1].pcolormesh(X, Y, jnp.clip(jnp.squeeze(distance/smallest_cell_size), -10, 10), cmap="jet")
        # ax[1,0].pcolormesh(X, Y, jnp.squeeze(deviation), cmap="jet", norm=norm)
        # ax[1,1].pcolormesh(X, Y, jnp.clip(jnp.squeeze(levelset/smallest_cell_size), -10, 10), cmap="jet")
        # for axi in ax.flatten():
        #     axi.set_aspect("equal")
        # plt.show()
        # plt.savefig("test.png")
        # plt.close()

        return levelset