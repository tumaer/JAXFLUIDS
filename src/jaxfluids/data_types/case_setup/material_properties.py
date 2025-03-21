from collections import namedtuple
from typing import NamedTuple, Tuple, Callable, Union, Dict, List

import jax.numpy as jnp
import numpy as np


class SutherlandParameters(NamedTuple):
    """Parameters of the Sutherland model for viscosity
    or thermal conductivity.

    value = value_ref * (T_ref + constant) / (T + constant) * (T / T_ref)**(3/2)

    value_ref: reference viscosity / thermal conductivity
    T_ref: reference temperature [K]
    constant: Sutherland constant [K]

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    value_ref: float
    T_ref: float
    constant: float

class DynamicViscositySetup(NamedTuple):
    model: str
    value: Callable = None
    sutherland_parameters: SutherlandParameters = None

class ThermalConductivitySetup(NamedTuple):
    model: str
    value: Callable = None
    prandtl_number: float = None
    sutherland_parameters: SutherlandParameters = None

class TransportPropertiesSetup(NamedTuple):
    dynamic_viscosity: DynamicViscositySetup
    bulk_viscosity: float
    thermal_conductivity: ThermalConductivitySetup

class IdealGasSetup(NamedTuple):
    specific_heat_ratio: float
    specific_gas_constant: float

class TaitSetup(NamedTuple):
    B: float
    N: float
    rho_ref: float
    p_ref: float

class StiffenedGasSetup(NamedTuple):
    specific_heat_ratio: float
    specific_gas_constant: float
    background_pressure: float
    energy_translation_factor: float
    thermal_energy_factor: float

class BarotropicCavitationFluidSetup(NamedTuple):
    mixture_phase_model: str
    liquid_phase_model: str
    temperature_ref: float
    density_liquid_ref: float
    density_vapor_ref: float
    pressure_ref: float
    speed_of_sound_liquid_ref: float
    speed_of_sound_vapor_ref: float
    speed_of_sound_mixture: float
    enthalpy_of_evaporation_ref: float
    cp_liquid_ref: float
    cp_vapor_ref: float

class FullThermodynamicCavitationFluidSetup(NamedTuple):
    pass

class StiffenedGasCompleteParameters(NamedTuple):
    specific_heat_ratio: float = None
    specific_gas_constant: float = None
    background_pressure: float = None
    energy_translation_factor: float = None
    thermal_energy_factor: float = None
    units: Dict[str, str] = {
        specific_heat_ratio: "none",
        specific_gas_constant: "none",
        background_pressure: "pressure",
        energy_translation_factor: "energy_translation_factor",
        thermal_energy_factor: "thermal_energy_factor",
    }

class StiffenedGasParameters(NamedTuple):
    specific_heat_ratio: float = None
    specific_gas_constant: float = None
    background_pressure: float = None

class IdealGasParameters(NamedTuple):
    specific_heat_ratio: float = None
    specific_gas_constant: float = None

class EquationOfStatePropertiesSetup(NamedTuple):
    model: str
    ideal_gas_setup: IdealGasSetup
    stiffened_gas_setup: StiffenedGasSetup
    tait_setup: TaitSetup
    barotropic_cavitation_fluid_setup: BarotropicCavitationFluidSetup

class MaterialPropertiesSetup(NamedTuple):
    """Specifies the properties of a single fluid material.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    eos: EquationOfStatePropertiesSetup
    transport: TransportPropertiesSetup

class MaterialPairingProperties(NamedTuple):
    surface_tension_coefficient: float

class LevelsetMixtureSetup(NamedTuple):
    """Describes the properties of a level-set material mixture,
    containing of a positive and a negative fluid.

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    positive_fluid: MaterialPropertiesSetup
    negative_fluid: MaterialPropertiesSetup
    pairing_properties: MaterialPairingProperties

def GetFluids(fluids_dict: Dict):
    fields = fluids_dict.keys()
    Fluids = namedtuple("Fluids", fields)
    fluids = Fluids(**fluids_dict)
    return fluids

class DiffuseMixtureSetup(NamedTuple):
    """Describes the material properties of a diffuse mixture
    consisting of N phases (materials).
    
    fluids: NamedTuple
    pairing: int

    :param NamedTuple: _description_
    :type NamedTuple: _type_
    """
    fluids: NamedTuple
    pairing_properties: MaterialPairingProperties

class MaterialManagerSetup(NamedTuple):
    single_material: MaterialPropertiesSetup = None
    levelset_mixture: LevelsetMixtureSetup = None
    diffuse_mixture: DiffuseMixtureSetup = None
