from functools import partial
from typing import List, Union, Dict, Callable, Tuple
import types
import os

import h5py
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.creation.NACA_airfoils import NACA_airfoils
from jaxfluids.levelset.creation.generic_shapes import (
    get_circle, get_sphere, get_rectangle, get_square, get_diamond,
    get_ellipse, get_ellipsoid)
from jaxfluids.domain.helper_functions import split_buffer
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionLevelset
from jaxfluids.unit_handler import UnitHandler

class LevelsetCreator:

    """The LevelsetCreator implements functionality to create
    initial levelset fields. The initial
    levelset field in one of two ways:
    1) Lambda function
    2) List of building blocks. A single building block includes a shape
        and a lambda function for the bounding domain.
    3) .h5 file
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            unit_handler: UnitHandler,
            initial_condition_levelset: InitialConditionLevelset,
            is_double_precision: bool
            ) -> None:

        self.initial_condition_levelset = initial_condition_levelset
        self.unit_handler = unit_handler
        self.domain_information = domain_information

        self.shape_function_dict: Dict[str, Callable] = {
            "circle": get_circle,
            "rectangle": get_rectangle,
            "square": get_square,
            "sphere": get_sphere,
            "diamond": get_diamond,
            "ellipse": get_ellipse,
            "ellipsoid": get_ellipsoid
            }

        self.NACA_airfoils = NACA_airfoils(is_double_precision)

    def create_levelset(
            self,
            levelset: Array,
            mesh_grid: Tuple[Array]
            ) -> Array:
        """Creates the levelset field.

        :param levelset: _description_
        :type levelset: Array
        :param mesh_grid: _description_
        :type mesh_grid: List
        :return: _description_
        :rtype: Array
        """

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives  
        active_axes_indices = self.domain_information.active_axes_indices
        smallest_cell_size = self.domain_information.smallest_cell_size

        is_callable = self.initial_condition_levelset.is_callable
        is_blocks = self.initial_condition_levelset.is_blocks
        is_NACA = self.initial_condition_levelset.is_NACA

        if is_callable:
            levelset_callable = self.initial_condition_levelset.levelset_callable
            levelset_init = levelset_callable(*mesh_grid)
            levelset = levelset.at[nhx,nhy,nhz].set(levelset_init)
        
        elif is_NACA:
            levelset_init = self.NACA_airfoils.compute_levelset(
                self.initial_condition_levelset.NACA_profile,
                mesh_grid, smallest_cell_size)
            levelset = levelset.at[nhx,nhy,nhz].set(levelset_init)

        elif is_blocks:
            for levelset_block in self.initial_condition_levelset.blocks:
                shape = levelset_block.shape
                parameters = levelset_block.parameters
                bounding_domain_callable = levelset_block.bounding_domain_callable
                
                levelset_callable = self.shape_function_dict[shape]
                levelset_init = levelset_callable(mesh_grid, parameters, active_axes_indices)
                mask = bounding_domain_callable(*mesh_grid)
                levelset = levelset.at[nhx,nhy,nhz].mul(1.0 - mask)
                levelset = levelset.at[nhx,nhy,nhz].add(levelset_init * mask)                         

        else:
            raise NotImplementedError

        return levelset
    
    # TODO AARON SOME SORT OF CHECK IF INTERFACE IS ON SMALLEST CELL SIZE
    def sanity_check():
        pass