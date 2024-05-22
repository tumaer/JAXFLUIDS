from functools import partial
from typing import Callable, Dict, Union, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

class UnitHandler:
    """The UnitHandler class implements functionaly
    to solve the NSE in non-dimensional form.
    """

    # UNIVERSAL CONSTANTS
    _universal_gas_constant = 8.31446261815324   # J / (K mol)
    _avogadro_constant = 6.02214076e+23          # 1 / mol
    _p_ref_global = 1.01325e5                    # Pa
    _T_ref_global = 298.15                       # K

    def __init__(
            self,
            density_reference: float,
            length_reference: float,
            velocity_reference: float,
            temperature_reference: float,
            amount_of_substance_reference: float = 1.0
            ) -> None:

        self.density_reference = density_reference
        self.length_reference = length_reference
        self.velocity_reference = velocity_reference
        self.temperature_reference = temperature_reference
        self.amount_of_substance_reference = amount_of_substance_reference

        self.mass_reference = density_reference * length_reference**3
        self.time_reference = length_reference / velocity_reference
        self.pressure_reference = density_reference * velocity_reference**2
        self.momentum_reference = density_reference * velocity_reference

        self.dynamic_viscosity_reference = density_reference * velocity_reference * length_reference
        self.thermal_conductivity_reference = \
            density_reference * velocity_reference**3 * length_reference / temperature_reference
        self.mass_diffusivity_reference = velocity_reference * length_reference
        self.surface_tension_coefficient_reference = \
            density_reference * velocity_reference * velocity_reference * length_reference

        self.gas_constant_reference = self.mass_reference * velocity_reference**2 / temperature_reference
        self.specific_gas_constant_reference = velocity_reference**2 / temperature_reference
        self.enthalpy_mass_reference = velocity_reference**2
        self.enthalpy_molar_reference = \
            self.mass_reference * velocity_reference**2 / amount_of_substance_reference
        self.entropy_molar_reference = self.enthalpy_molar_reference / temperature_reference
        self.energy_translation_factor_reference = density_reference * self.pressure_reference
        self.thermal_energy_factor_reference = velocity_reference**2 / density_reference

        self.gravity_reference = velocity_reference**2 / length_reference
        self.mass_flow_reference = self.mass_reference / self.time_reference

        self.velocity_gradient_reference = self.velocity_reference / self.length_reference

        self.reference_dict = {

            # MATERIAL FIELDS
            "density": density_reference,
            "velocity": velocity_reference,
            "temperature": temperature_reference,
            "pressure": self.pressure_reference,
            "momentum": self.momentum_reference,
            "energy": self.pressure_reference,
            "amount_of_substance": self.amount_of_substance_reference,
            "one_amount_of_substance": 1.0 / self.amount_of_substance_reference,

            # LEVELSET FIELDS
            "levelset": self.length_reference,
            "interface_pressure": self.pressure_reference,
            "interface_velocity": self.velocity_reference,
            "mask_real": 1.0,
            "volume_fraction": 1.0,
            "normal": 1.0,

            # MATERIAL PROPERTIES
            "dynamic_viscosity": self.dynamic_viscosity_reference,
            "thermal_conductivity": self.thermal_conductivity_reference,
            "mass_diffusivity": self.mass_diffusivity_reference,
            "gas_constant": self.gas_constant_reference,
            "specific_gas_constant": self.specific_gas_constant_reference,
            "specific_heat_capacity": self.specific_gas_constant_reference,
            "enthalpy_mass": self.enthalpy_mass_reference,
            "enthalpy_molar": self.enthalpy_molar_reference,
            "entropy_molar": self.entropy_molar_reference,
            "surface_tension_coefficient": self.surface_tension_coefficient_reference,
            "energy_translation_factor": self.energy_translation_factor_reference,
            "thermal_energy_factor": self.thermal_energy_factor_reference,

            # PHYSICAL QUANTITIES
            "length": length_reference,
            "time": self.time_reference,
            "gravity": self.gravity_reference,
            "mass": self.mass_reference,
            "mass_flow": self.mass_flow_reference,

            # OUTPUT
            "absolute_velocity": self.velocity_reference,
            "vorticity": self.velocity_reference/self.length_reference,
            "absolute_vorticity": self.velocity_reference/self.length_reference,
            "schlieren": self.density_reference/self.length_reference,
            "mach_number": 1.0,
            "qcriterion": self.velocity_gradient_reference**2,
            "dilatation": 1.0/self.time_reference,

            "none": 1.0,
            "None": 1.0
        }
        
        self.universal_gas_constant_nondim = self.non_dimensionalize(self._universal_gas_constant, "gas_constant")
        self.avogadro_constant_nondim = self.non_dimensionalize(self._avogadro_constant, "one_amount_of_substance")
        self.p_ref_global_nondim = self.non_dimensionalize(self._p_ref_global, "pressure")
        self.T_ref_global_nondim = self.non_dimensionalize(self._T_ref_global, "temperature")

    def non_dimensionalize(
            self,
            value: Union[Array, float],
            quantity: Union[str, Callable],
            quantity_list: Tuple = None,
            is_spatial_derivative: bool = False,
            is_temporal_derivative: bool = False
            ) -> Union[Array, float]:
        """Non-dimensionalizes the given buffer w.r.t. the specified quantity.

        :param value: Dimensional quantity buffer
        :type value: Union[Array, float]
        :param quantity: Quantity name
        :type quantity: str
        :return: Non-dimensional quantity buffer
        :rtype: Union[Array, float]
        """

        if isinstance(quantity, Callable):
            reference = quantity(
                self.density_reference,
                self.length_reference,
                self.velocity_reference,
                self.temperature_reference,
                self.amount_of_substance_reference)
            value_nondim = value / reference

        elif isinstance(quantity, str):
            if quantity == "specified":
                reference = jnp.stack([self.reference_dict[quant] for quant in quantity_list])
                reference = reference.reshape((-1,) + (1,) * (value.ndim - 1))
                value_nondim = value / reference
            
            elif quantity in ["None", "none"]:
                value_nondim = value

            else:
                quantity = self.convert_name(quantity)
                reference = self.reference_dict[quantity]
                value_nondim = value / reference

        else:
            raise NotImplementedError

        if is_spatial_derivative:
            value_nondim = value_nondim * self.length_reference

        if is_temporal_derivative:
            value_nondim = value_nondim * self.time_reference

        return value_nondim

    def dimensionalize(
            self,
            value: Union[Array, float],
            quantity: Union[str, Callable],
            quantity_list: Tuple = None
            ) -> Union[Array, float]:
        """Dimensionalizes the given quantity buffer w.r.t. the specified quanty.

        :param value: Non-dimensional quantity buffer
        :type value: Union[Array, float]
        :param quantity: Quantity name
        :type quantity: str
        :return: Dimensional quantity buffer
        :rtype: Union[Array, float]
        """
        if isinstance(quantity, Callable):
            reference = quantity(
                self.density_reference,
                self.length_reference,
                self.velocity_reference,
                self.temperature_reference,)
            value_dim = value * reference

        elif isinstance(quantity, str):
            if quantity == "specified":
                reference = jnp.stack([self.reference_dict[quant] for quant in quantity_list])
                reference = reference.reshape((-1,) + (1,) * (value.ndim - 1))
                value_dim = value * reference
            
            elif quantity in ("None", "none",):
                value_dim = value

            else:
                quantity = self.convert_name(quantity)
                reference = self.reference_dict[quantity]
                value_dim = value * reference

        else:
            raise NotImplementedError

        return value_dim

    def convert_name(self, quantity: str) -> str:
        """Converts the name of the quantity as specified during
        computation to the appropriate name for non-dimensionalization.

        :param quantity: _description_
        :type quantity: str
        :return: _description_
        :rtype: str
        """
        if quantity == "rho":
            quantity = "density"
        elif quantity in ("u", "v", "w",):
            quantity = "velocity"
        elif quantity in ("rhou", "rhov", "rhow",):
            quantity = "momentum"
        elif quantity in ("p",):
            quantity = "pressure"
        elif quantity == "T":
            quantity = "temperature"
        elif quantity.startswith("alpharho_"):
            quantity = "density"
        elif quantity.startswith("alpha_"):
            quantity = "volume_fraction"
        elif quantity.startswith("rho_"):
            quantity = "density"       
        return quantity
