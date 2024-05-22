from __future__ import annotations
from typing import Dict, Union, TYPE_CHECKING
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup
    
class ConvectiveFluxSolver(ABC):

    eps = jnp.finfo(jnp.float64).eps

    """The Flux Computer sets up the user-specified flux function
    for the calculation of the convective terms. The flux calculation
    is called in the space solver by compute_convective_flux_xi().

    There are three general options for the convective flux function.
    1) High-order Godunov Scheme
    2) Flux-splitting Scheme
    3) ALDM Scheme
    """

    def __init__(
            self,
            convective_fluxes_setup: ConvectiveFluxesSetup,
            material_manager: MaterialManager,
            domain_information: DomainInformation,
            equation_manager: EquationManager
            ) -> None:

        self.domain_information = domain_information
        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        
    @abstractmethod
    def compute_flux_xi(
            self,
            primitives: Array,
            conservatives: Array,
            axis: int,
            curvature: Array = None,
            volume_fraction: Array = None,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            **kwargs,
            ) -> Array:
        """Computes the convective fluxes. 

        :param primitives: Primitive variable buffer
        :type primitives: Array
        :param conservatives: Conservative variable buffer
        :type conservatives: Array
        :param axis: Spatial direction
        :type axis: int
        :param ml_parameters_dict: Dictionary of neural network weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary of neural network architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Convective fluxes in axis direction
        :rtype: Array
        """
        pass