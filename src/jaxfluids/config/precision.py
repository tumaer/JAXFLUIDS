from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array 

class Epsilons(NamedTuple):
    density: float
    pressure: float
    volume_fraction: float

def check_value(value: float, name: str):
    assert_string = (
        f"Precision value for {name:s} must be of type float.")
    assert isinstance(value, float), assert_string
    assert_string = (
        f"Precision value for {name:s} must be greater than 0.0.")
    assert value > 0.0, assert_string

class PrecisionConfig:

    def __init__(self):
        jax.config.update("jax_default_matmul_precision", "highest")
        if jax.config.read("jax_enable_x64"):
            self.enable_double_precision()
        else:
            self.enable_single_precision()

    def enable_single_precision(self) -> None:
        """Enables single precision as default for 
        jax computations. Epsilons for e.g. interpolation
        or flux limiter are set accordingly.
        """
        jax.config.update("jax_enable_x64", False)
        self.__eps = jnp.finfo(jnp.float32).eps
        self.__smallest_normal = jnp.finfo(jnp.float32).tiny
        self.__fmax = jnp.finfo(jnp.float32).max
        self.__spatial_stencil_eps = 1e-30
        self.__interpolation_limiter_eps = Epsilons(1e-6, 1e-6, 1e-6)
        self.__flux_limiter_eps = Epsilons(1e-6, 1e-6, 1e-6)
        self.__thinc_limiter_eps = Epsilons(1e-6, 1e-6, 1e-6)

    def enable_double_precision(self) -> None:
        """Enables double precision as default for 
        jax computations. Epsilons for e.g. interpolation
        or flux limiter are set accordingly.
        """
        jax.config.update("jax_enable_x64", True)
        self.__eps = jnp.finfo(jnp.float64).eps
        self.__smallest_normal = jnp.finfo(jnp.float64).tiny
        self.__fmax = jnp.finfo(jnp.float64).max
        self.__spatial_stencil_eps = 1e-30
        self.__interpolation_limiter_eps = Epsilons(1e-12, 1e-10, 1e-12)
        self.__flux_limiter_eps = Epsilons(1e-12, 1e-10, 1e-12)
        self.__thinc_limiter_eps = Epsilons(1e-11, 1e-9, 1e-11)
    
    def set_eps(self, eps: float) -> None:
        """Sets the general epsilon used in jaxfluids.

        :param eps: _description_
        :type eps: float
        """
        check_value(eps, "eps")
        self.__eps = eps

    def set_smallest_normal(self, smallest_normal: float) -> None:
        """Sets the smallest_normal.
        
        :param smallest_normal: _description_
        :type smallest_normal: float
        """
        check_value(smallest_normal, "tiny")
        self.__smallest_normal = smallest_normal

    def set_fmax(self, fmax: float) -> None:
        """Sets the general fmax used in 
        jaxfluids.

        :param fmax: _description_
        :type fmax: float
        """
        check_value(fmax, "fmax")
        self.__fmax = fmax

    def set_spatial_stencil_eps(self, eps: float) -> None:
        """Sets the epsilon used for spatial stencils
        to prevent numerical errors like division by zero.

        :param eps: _description_
        :type eps: float
        """
        check_value(eps, "spatial stencil eps")
        self.__spatial_stencil_eps = eps     

    def set_interpolation_limiter_eps(
            self,
            density: float,
            pressure: float,
            volume_fraction: float
            ) -> None:
        """Sets the epsilons used to activate
        the interpolation limiter for the spatial
        reconstruction.

        :param density: _description_
        :type density: float
        :param pressure: _description_
        :type pressure: float
        :param volume_fraction: _description_
        :type volume_fraction: float
        """
        check_value(density, "interplation limiter density eps")
        check_value(pressure, "interplation limiter pressure eps")
        check_value(volume_fraction, "interplation limiter volume fraction eps")
        self.__interpolation_limiter_eps = Epsilons(density, pressure, volume_fraction)

    def set_flux_limiter_eps(
            self,
            density: float,
            pressure: float,
            volume_fraction: float
            ) -> None:
        """Sets the epsilons used to activate the
        flux limiter for the convective flux computation
        for the high-order godunov scheme.

        :param density: _description_
        :type density: float
        :param pressure: _description_
        :type pressure: float
        :param volume_fraction: _description_
        :type volume_fraction: float
        """
        check_value(density, "flux limiter density eps")
        check_value(pressure, "flux limiter pressure eps")
        check_value(volume_fraction, "flux limiter volume fraction eps")
        self.__flux_limiter_eps = Epsilons(density, pressure, volume_fraction)

    def set_thinc_limiter_eps(
            self,
            density: float,
            pressure: float,
            volume_fraction: float
            ) -> None:
        """Sets the epsilons used to activate the
        thinc limiter for the thinc reconstruction
        in the diffuse interface model. 

        :param density: _description_
        :type density: float
        :param pressure: _description_
        :type pressure: float
        :param volume_fraction: _description_
        :type volume_fraction: float
        """
        check_value(density, "thinc limiter density eps")
        check_value(pressure, "thinc limiter pressure eps")
        check_value(volume_fraction, "thinc limiter volume fraction eps")
        self.__thinc_limiter_eps = Epsilons(density, pressure, volume_fraction)

    def get_eps(self) -> float:
        """Gets the general epsilon used in
        jaxfluids to prevent numerical errors
        like division by zero.

        :return: _description_
        :rtype: _type_
        """
        return self.__eps
    
    def get_smallest_normal(self) -> float:
        """Gets the smallest normal.

        :return: _description_
        :rtype: _type_
        """
        return self.__smallest_normal

    def get_fmax(self) -> float:
        """Gets the general fmax used in jaxfluids.

        :return: _description_
        :rtype: _type_
        """
        return self.__fmax

    def get_spatial_stencil_eps(self) -> float:
        """Sets the epsilon used for spatial stencils
        to prevent numerical errors like division by zero.

        :param eps: _description_
        :type eps: float
        :return: _description_
        :rtype: _type_
        """
        return self.__spatial_stencil_eps

    def get_interpolation_limiter_eps(self) -> Epsilons:
        """Sets the epsilons used to activate
        the interpolation limiter for the spatial
        reconstruction.

        :return: _description_
        :rtype: Epsilons
        """
        return self.__interpolation_limiter_eps
    
    def get_flux_limiter_eps(self) -> Epsilons:
        """Sets the epsilons used to activate the
        flux limiter for the convective flux computation
        for the high-order godunov scheme.

        :return: _description_
        :rtype: Epsilons
        """
        return self.__flux_limiter_eps
    
    def get_thinc_limiter_eps(self) -> Epsilons:
        """Sets the epsilons used to activate the
        thinc limiter for the thinc reconstruction
        in the diffuse interface model.

        :return: _description_
        :rtype: Epsilons
        """
        return self.__thinc_limiter_eps
    

