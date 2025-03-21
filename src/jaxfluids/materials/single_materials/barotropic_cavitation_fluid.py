from typing import List, Union
import types

import jax
import jax.numpy as jnp

from jaxfluids.materials.single_materials.material import Material
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup
from jaxfluids.materials.single_materials import Tait

Array = jax.Array

class BarotropicCavitationFluid(Material):
    """Implements a water like barotropic material for cavitation simulations.
    There exist several options for how the pure liquid phase is model
    and how the vapor/liquid mixture is modeled.
    
    Liquid phase models:
    1) Linear
    2) Tait
    
    Mixture phase models:
    1) Linear
    2) Frozen
    3) Equilibrium liquid
    4) Equilibrium full
    """
    def __init__(
            self,
            unit_handler: UnitHandler,
            material_setup: MaterialPropertiesSetup
            ) -> None:

        super().__init__(unit_handler, material_setup)

        cavitation_fluid_setup = material_setup.eos.barotropic_cavitation_fluid_setup
        
        self.liquid_phase_model = cavitation_fluid_setup.liquid_phase_model
        self.mixture_phase_model = cavitation_fluid_setup.mixture_phase_model
        self.is_linear = (self.liquid_phase_model == "LINEAR") and (self.mixture_phase_model == "LINEAR")

        self.T_ref = cavitation_fluid_setup.temperature_ref
        self.rho_ref_l = cavitation_fluid_setup.density_liquid_ref
        self.rho_ref_v = cavitation_fluid_setup.density_vapor_ref
        self.p_ref = cavitation_fluid_setup.pressure_ref
        self.c_l = cavitation_fluid_setup.speed_of_sound_liquid_ref
        self.c_v = cavitation_fluid_setup.speed_of_sound_vapor_ref
        self.c_m = cavitation_fluid_setup.speed_of_sound_mixture
        self.cp_ref_l = cavitation_fluid_setup.cp_liquid_ref
        self.cp_ref_v = cavitation_fluid_setup.cp_vapor_ref
        self.enthalpy_of_vaporization = cavitation_fluid_setup.enthalpy_of_evaporation_ref
        
        if self.liquid_phase_model == "TAIT":
            self.tait = Tait(unit_handler, material_setup)

        # TODO - clean solution
        self.R = 0.0
        self.pb = 0.0
        
        self.viscosity_l_sat = None
        self.viscosity_v_sat = None
        self.thermal_conductivity_l = None
        self.thermal_conductivity_v = None

        # Parameters for speed of sound relation in the two-phase (liquid/vapor) domain
        if self.mixture_phase_model == "FROZEN":
            C = 0.0
            self.B = -1.0 / (self.rho_ref_l - self.rho_ref_v) * (1.0 / (self.rho_ref_v * self.c_v**2) - 1.0 / (self.rho_ref_l * self.c_l**2))
            self.A = -self.rho_ref_l * self.B + 1/(self.rho_ref_l * self.c_l**2)

        elif self.mixture_phase_model == "EQUILIBRIUM_LIQUID":
            C = -(self.rho_ref_l * self.cp_ref_l * self.T_ref) / (self.rho_ref_v * self.enthalpy_of_vaporization)**2
            self.B = 1.0 / (self.rho_ref_v - self.rho_ref_l) * (1.0 / (self.rho_ref_v * self.c_v**2) - 1.0 / (self.rho_ref_l * self.c_l**2) + C)
            self.A = -self.rho_ref_l * self.B + 1.0 / (self.rho_ref_l * self.c_l**2) + (self.rho_ref_l * self.cp_ref_l * self.T_ref) / (self.rho_ref_v * self.enthalpy_of_vaporization)**2

        elif self.mixture_phase_model == "EQUILIBRIUM_FULL":
            C = (self.T_ref * (self.rho_ref_v * self.cp_ref_v - self.rho_ref_l * self.cp_ref_l)) / (self.rho_ref_v * self.enthalpy_of_vaporization)**2
            self.B = 1/(self.rho_ref_v - self.rho_ref_l) * (1/(self.rho_ref_v * self.c_v**2) - 1/(self.rho_ref_l * self.c_l**2) + C)
            self.A = -self.rho_ref_l * self.B + 1.0 / (self.rho_ref_l * self.c_l**2) + (self.rho_ref_l * self.cp_ref_l * self.T_ref) / (self.rho_ref_v * self.enthalpy_of_vaporization)**2

        elif self.mixture_phase_model == "LINEAR":
            pass
    
        else:
            raise NotImplementedError

    def _set_transport_properties(self) -> None:
        pass

    def get_specific_heat_capacity(self, T: Array) -> Union[float, Array]:
        raise NotImplementedError

    def get_psi(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / rho

    def get_grueneisen(self, rho: Array) -> Array:
        """See base class. """
        return None

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        """See base class. """
        if self.is_linear:
            c = jnp.where(rho > self.rho_ref_l, self.c_l, self.c_m)
        else:
            c_mixture = self.get_speed_of_sound_mixture(p, rho)
            c_liquid = self.get_speed_of_sound_liquid(p, rho)
            c = jnp.where(rho > self.rho_ref_l, c_liquid, c_mixture)

        return c

    def get_speed_of_sound_mixture(self, p: Array, rho: Array) -> Array:
        if self.mixture_phase_model == "LINEAR":
            c_mixture = self.c_m
        
        elif self.mixture_phase_model in ("FROZEN", "EQUILIBRIUM_LIQUID", "EQUILIBRIUM_FULL"):
            alpha = self.get_volume_fraction(rho)

            one_rho_cc = alpha / (self.rho_ref_v * self.c_v * self.c_v) \
                + (1.0 - alpha) / (self.rho_ref_l * self.c_l * self.c_l)
            
            if self.mixture_phase_model in ("EQUILIBRIUM_LIQUID", "EQUILIBRIUM_FULL"):
                temp = self.rho_ref_v * self.enthalpy_of_vaporization
                one_rho_cc += (1.0 - alpha) * self.rho_ref_l * self.cp_ref_l * self.T_ref \
                    / (temp * temp)
            
                if self.mixture_phase_model == "EQUILIBRIUM_FULL":
                    one_rho_cc += alpha * self.rho_ref_v * self.cp_ref_v * self.T_ref \
                        / (temp * temp)

            c_mixture = jnp.sqrt(1.0 / (rho * one_rho_cc))
        
        else:
            raise NotImplementedError
        
        return c_mixture
    
    def get_speed_of_sound_liquid(self, p: Array, rho: Array) -> Array:
        """Returns the speed of sound of the liquid phase
        at the given pressure and density.

        :param p: pressure buffer
        :type p: Array
        :param rho: density buffer
        :type rho: Array
        :return: speed of sound of liquid phase
        :rtype: Array
        """
        if self.liquid_phase_model == "LINEAR":
            c_liquid = self.c_l
        
        elif self.liquid_phase_model == "TAIT":
            c_liquid = self.tait.get_speed_of_sound(p, rho)
        
        else:
            raise NotImplementedError
         
        return c_liquid

    def get_pressure(self, e: Array, rho: Array) -> Array:
        """See base class. """
        if self.is_linear:
            c = jnp.where(rho > self.rho_ref_l, self.c_l, self.c_m)
            p = self.p_ref + (rho - self.rho_ref_l) * c * c

        else:
            p_liquid = self.get_pressure_liquid(rho)
            p_mixture = self.get_pressure_mixture(rho)
            p = jnp.where(rho > self.rho_ref_l, p_liquid, p_mixture)

        return p

    def get_pressure_liquid(self, rho: Array) -> Array:
        if self.liquid_phase_model == "LINEAR":
            p_liquid = self.p_ref + (rho - self.rho_ref_l) * self.c_l * self.c_l

        elif self.liquid_phase_model == "TAIT":
            p_liquid = self.tait.get_pressure(None, rho)

        else:
            raise NotImplementedError
        
        return p_liquid
    
    def get_pressure_mixture(self, rho: Array) -> Array:
        if self.mixture_phase_model == "LINEAR":
            p_mixture = self.p_ref + (rho - self.rho_ref_l) * self.c_m * self.c_m

        elif self.mixture_phase_model in ("FROZEN", "EQUILIBRIUM_LIQUID", "EQUILIBRIUM_FULL"):
            p_mixture = 1.0 / self.A * jnp.log(rho / (self.A + self.B * rho)) \
                - 1.0 / self.A * jnp.log(rho / (self.A + self.B * rho)) \
                + self.p_ref

        else:
            raise NotImplementedError
        
        return p_mixture

    def get_temperature(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / ( rho * self.R )
    
    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        """See base class. """
        # Specific internal energy
        return jnp.ones_like(p)

    def get_total_energy(self, p: Array, rho: Array, velocity_vec: Array) -> Array:
        """See base class. """
        # Total energy per unit volume
        return jnp.ones_like(p)

    def get_total_enthalpy(self, p: Array, rho: Array, velocity_vec: Array) -> Array:
        """See base class. """
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, velocity_vec) + p) / rho
    
    def get_volume_fraction(self, rho: Array) -> Array:
        alpha = jnp.where(rho > self.rho_ref_l, 0.0, (rho - self.rho_ref_l) / (self.rho_ref_v - self.rho_ref_l))
        return alpha
    
    def get_density(self, p: Array) -> Array:
        if self.is_linear:
            c = jnp.where(p > self.p_ref, self.c_l, self.c_m)
            rho = self.rho_ref_l + (p - self.p_ref) / (c * c)

        else:
            rho_mixture = self.get_density_mixture(p)
            rho_liquid = self.get_density_liquid(p)
            rho = jnp.where(p > self.p_ref, rho_liquid, rho_mixture)

        return rho

    def get_density_liquid(self, p: Array) -> Array:
        if self.liquid_phase_model == "LINEAR":
            rho_liquid = self.rho_ref_l + (p - self.p_ref) / (self.c_l * self.c_l)
        
        elif self.liquid_phase_model == "TAIT":
            rho_liquid = self.tait.get_density(p)
        
        else:
            raise NotImplementedError
        
        return rho_liquid

    def get_density_mixture(self, p: Array) -> Array:
        if self.mixture_phase_model == "LINEAR":
            rho_mixture = self.rho_ref_l + (p - self.p_ref) / (self.c_m * self.c_m)

        elif self.mixture_phase_model in ("FROZEN", "EQUILIBRIUM_LIQUID", "EQUILIBRIUM_FULL"):
            temp = jnp.exp(self.A * (p - self.p_ref) + jnp.log(self.rho_ref_l / (self.A + self.B * self.rho_ref_l)))
            rho_mixture = temp * self.A / (1.0 - temp * self.B)

        else:
            raise NotImplementedError

        return rho_mixture
    
    def get_mixture_density(self, alpha: Array) -> Array:
        rho = alpha * self.rho_ref_v + (1.0 - alpha) * self.rho_ref_l
        return rho

    def get_dynamic_viscosity(
            self,
            rho: Array,
            T: Array,
            **kwargs
            ) -> Array:
        # TODO rho input is not implemented?
        raise NotImplementedError
        alpha = self.get_volume_fraction(rho)
        return (1.0 - alpha) * (1.0 + 2.5 * alpha) * self.viscosity_l_sat + alpha * self.viscosity_v_sat
    
    def get_thermal_conductivity(
            self,
            rho: Array,
            T: Array,
            **kwargs
            ) -> Array:
        # TODO rho input is not implemented?
        raise NotImplementedError
        alpha = self.get_volume_fraction(rho)
        return (1.0 - alpha) * self.thermal_conductivity_l + alpha * self.thermal_conductivity_v
