from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager
from jaxfluids.config import precision

Array = jax.Array

class RiemannSolver(ABC):
    """Abstract base class for Riemann solvers.

    RiemannSolver has two fundamental attributes: a material manager and a signal speed.
    The solve_riemann_problem_xi method solves the one-dimensional Riemann problem.
    """

    def __init__(
            self,
            material_manager: MaterialManager, 
            equation_manager: EquationManager,
            signal_speed: Callable,
            **kwargs
            ) -> None:

        self.eps = precision.get_eps()

        self.material_manager = material_manager
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.signal_speed = signal_speed

        # MINOR AXIS DIRECTIONS 
        self.velocity_minor = self.equation_information.velocity_minor_axes

        self.equation_type = self.equation_information.equation_type
        self.is_surface_tension = self.equation_information.active_physics.is_surface_tension

        self.ids_mass = self.equation_manager.equation_information.ids_mass
        self.s_mass = self.equation_manager.equation_information.s_mass
        self.ids_velocity = self.equation_manager.equation_information.ids_velocity
        self.s_velocity = self.equation_manager.equation_information.s_velocity
        self.ids_energy = self.equation_manager.equation_information.ids_energy
        self.s_energy = self.equation_manager.equation_information.s_energy
        self.ids_volume_fraction = self.equation_manager.equation_information.ids_volume_fraction
        self.s_volume_fraction = self.equation_manager.equation_information.s_volume_fraction
        self.ids_species = self.equation_manager.equation_information.ids_species
        self.s_species = self.equation_manager.equation_information.s_species
    
    def solve_riemann_problem_xi(
            self, 
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """Solves one-dimensional Riemann problem in the direction as specified 
        by the axis argument. Wrapper function which calls, depending on the equation type,
        one of

        1) _solve_riemann_problem_xi_single_phase
        2) _solve_riemann_problem_xi_diffuse_four_equation
        3) _solve_riemann_problem_xi_diffuse_five_equation

        :param primitives_L: primtive variable buffer left of cell face
        :type primitives_L: Array
        :param primitives_R: primtive variable buffer right of cell face
        :type primitives_R: Array
        :param conservatives_L: conservative variable buffer left of cell face
        :type conservatives_L: Array
        :param conservatives_R: conservative variable buffer right of cell face
        :type conservatives_R: Array
        :param axis: Spatial direction along which Riemann problem is solved.
        :type axis: int
        :return: _description_
        :rtype: Tuple[Array, Union[Array, None], Union[Array, None]]
        """

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            return self._solve_riemann_problem_xi_single_phase(
                primitives_L, primitives_R,
                conservatives_L, conservatives_R,
                axis, **kwargs)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            return self._solve_riemann_problem_xi_diffuse_four_equation(
                primitives_L, primitives_R,
                conservatives_L, conservatives_R,
                axis, **kwargs)

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            return self._solve_riemann_problem_xi_diffuse_five_equation(
                primitives_L, primitives_R,
                conservatives_L, conservatives_R,
                axis, **kwargs)

        else:
            raise NotImplementedError

    @abstractmethod
    def _solve_riemann_problem_xi_single_phase(
            self,
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """Solves one-dimensional single-phase Riemann problem
        in the direction as specified by the axis argument.

        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param axis: _description_
        :type axis: int
        :return: _description_
        :rtype: Tuple[Array, Union[Array, None], Union[Array, None]]
        """
        pass

    @abstractmethod
    def _solve_riemann_problem_xi_diffuse_four_equation(
            self,
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """Solves one-dimensional Riemann problem for the diffuse-interface
        four-equation model in the direction as specified by the axis argument.

        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param axis: _description_
        :type axis: int
        :return: _description_
        :rtype: Tuple[Array, Union[Array, None], Union[Array, None]]
        """
        pass

    @abstractmethod
    def _solve_riemann_problem_xi_diffuse_five_equation(
            self,
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """Solves one-dimensional Riemann problem for the diffuse-interface
        five-equation model in the direction as specified by the axis argument.

        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param axis: _description_
        :type axis: int
        :return: _description_
        :rtype: Tuple[Array, Union[Array, None], Union[Array, None]]
        """
        pass
