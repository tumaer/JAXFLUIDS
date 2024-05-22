from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup import ActivePhysicsSetup, ActiveForcingsSetup

TUPLE_PRIMES = ("rho", "u", "v", "w", "p")

TUPLE_CHEMICAL_COMPONENTS = (
    "H2", "O2", "H2O", "N2", "Ar",
    "H2O2", "OH_r", "H_r", "O_r", "HO2_r")

AVAILABLE_QUANTITIES = {
    "primitives" : ("density", "velocity", "pressure", "temperature"),
    "conservatives": ("mass", "momentum", "energy"),
    "levelset": ("levelset", "volume_fraction", "interface_pressure", "interface_velocity"),
    "real_fluid": ("density", "velocity", "pressure", "temperature",
                    "mass", "momentum", "energy"),
    "miscellaneous": ("mach_number", "schlieren", "absolute_velocity",
                        "vorticity", "absolute_vorticity", "qcriterion", "dilatation",
                        "volume_fraction"),
    "forcings": ("mass_flow")
    }

def get_reduced_primes_tuple_for_diffuse_interface(primes_tuple: Tuple) -> Tuple:
    primes_tuple_ = tuple()
    for prime in primes_tuple:
        if prime.startswith("alpharho_"):
            prime = prime[5:]
        primes_tuple_ += (prime,)
    return primes_tuple_

class EquationInformation:
    """ The EquationInformation class holds information on the
    system of equations that is being solved. It provides
    information on which variables are stored in the primitives
    and conservatives buffers.

    Currently the following equation types are available:
    1) SINGLE-PHASE
    2) SINGLE-PHASE-SOLID-LS
    3) TWO-PHASE-LS
    5) DIFFUSE-INTERFACE-5EQM
    """

    def __init__(
            self,
            primes_tuple: Tuple,
            fluid_names: Tuple,
            levelset_model: str,
            diffuse_interface_model: str,
            active_physics: ActivePhysicsSetup,
            active_forcings: ActiveForcingsSetup,
            ) -> None:
        
        self.primes_tuple = primes_tuple
        self.no_primes = len(primes_tuple)

        self.fluid_names = fluid_names
        self.no_fluids = len(fluid_names)

        self.active_physics = active_physics
        self.active_forcings = active_forcings

        # LEVEL SET
        self.levelset_model = levelset_model
        if levelset_model in [
            "FLUID-FLUID",
            "FLUID-SOLID-DYNAMIC",
            "FLUID-SOLID-DYNAMIC-COUPLED"
        ]:
            self.is_moving_levelset = True
        else:
            self.is_moving_levelset = False
        if levelset_model in [
            "FLUID-SOLID-STATIC", 
            "FLUID-SOLID-DYNAMIC", 
            "FLUID-SOLID-DYNAMIC-COUPLED"
        ]:
            self.is_solid_levelset = True
        else:
            self.is_solid_levelset = False

        # DIFFUSE INTERFACE
        self.diffuse_interface_model = diffuse_interface_model

        # SINGLE PHASE OR LEVELSET TWO-PHASE
        self.equation_type = "SINGLE-PHASE"
        self.no_equations = 5
        self.shape_equations = (self.no_equations,)
        self.mass_ids = 0
        self.mass_slices = jnp.s_[0:1] 
        self.velocity_ids = (1, 2, 3)
        self.velocity_slices = jnp.s_[1:4]
        self.momentum_xi_slices = (jnp.s_[1:2], jnp.s_[2:3], jnp.s_[3:4])
        self.energy_ids = 4
        self.energy_slices = jnp.s_[4:5]
        self.mass_and_energy_ids = (self.mass_ids, self.energy_ids)
        self.mass_and_vf_ids = None
        self.momentum_and_energy_ids = self.velocity_ids + (self.energy_ids,)
        self.momentum_and_energy_slices = jnp.s_[1:5]
        self.vf_ids = None
        self.vf_slices = None
        self.species_ids = None
        self.species_slices = None
        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.conservatives_slices = jnp.s_[:]

        self.material_field_slices = {
            "primitives": {
                "density": self.mass_slices,
                "velocity": self.velocity_slices,
                "pressure": self.energy_slices,
            },
            "conservatives": {
                "mass": self.mass_slices,
                "momentum": self.velocity_slices,
                "energy": self.energy_slices,
            },
        }

        self.primitive_quantities = ("density",) + 3 * ("velocity",) + ("pressure",)

        # LEVEL-SET SOLID
        if self.is_solid_levelset:
            self.equation_type = "SINGLE-PHASE-SOLID-LS"

        # LEVEL-SET TWO-PHASE
        if self.levelset_model == "FLUID-FLUID":
            self.equation_type = "TWO-PHASE-LS"
            self.shape_equations = (self.no_equations, 2)

        # DIFFUSE INTERFACE MODEL - 5 EQUATION MODEL
        if self.diffuse_interface_model == "5EQM":
            self.initialize_diffuse_interface_5eqm()

    def initialize_diffuse_interface_5eqm(self) -> None:
        """ Initializes equation informations for the 5 equation
        diffuse interface model.
        """

        no_fluids = self.no_fluids

        self.equation_type = "DIFFUSE-INTERFACE-5EQM"
        self.no_equations = 2 * no_fluids + 3
        self.shape_equations = (self.no_equations,)
        self.conservatives_slices = jnp.s_[:-(no_fluids-1)]
        self.mass_ids = tuple(range(no_fluids))
        self.mass_slices = jnp.s_[0:no_fluids]
        self.velocity_ids = (no_fluids, no_fluids+1, no_fluids+2)
        self.velocity_slices = jnp.s_[no_fluids:no_fluids+3]
        self.momentum_xi_slices = (
            jnp.s_[no_fluids:no_fluids+1],
            jnp.s_[no_fluids+1:no_fluids+2],
            jnp.s_[no_fluids+2:no_fluids+3])
        self.energy_ids = no_fluids + 3
        self.energy_slices = jnp.s_[no_fluids+3:no_fluids+4]
        self.mass_and_energy_ids = self.mass_ids + (self.energy_ids,)
        self.vf_ids = tuple(range(no_fluids+4,(2*no_fluids+4)-1))
        self.vf_slices = jnp.s_[no_fluids+4:(2*no_fluids+4)-1]
        self.mass_and_vf_ids = self.mass_ids + self.vf_ids
        self.momentum_and_energy_ids = self.velocity_ids + (self.energy_ids,)
        self.momentum_and_energy_slices = jnp.s_[no_fluids:no_fluids+4]
        self.velocity_minor_axes = (
            (no_fluids+1, no_fluids+2),
            (no_fluids+2, no_fluids  ),
            (no_fluids  , no_fluids+1),
        )

        self.primes_tuple_ = get_reduced_primes_tuple_for_diffuse_interface(self.primes_tuple)
        self.alpharho_i_dict = {f"alpharho_{i:d}": i for i in range(self.no_fluids)}
        self.alpha_i_dict = {f"alpha_{i:d}": i+self.no_fluids+4 for i in range(self.no_fluids-1)}

        self.material_field_slices = {
            "primitives": {
                "density": self.mass_slices,
                "velocity": self.velocity_slices,
                "pressure": self.energy_slices,
                "volume_fraction": self.vf_slices,
            },
            "conservatives": {
                "mass": self.mass_slices,
                "momentum": self.velocity_slices,
                "energy": self.energy_slices,
                "volume_fraction": self.vf_slices,
            },
        }

        self.primitive_quantities = self.no_fluids*("density",) + 3 * ("velocity",)\
            + ("pressure",) + (self.no_fluids -1) * ("volume_fraction",)

    def get_material_field_indices(self, active_axes_indices):
        """Dictioanry assigning material field quantity
        to buffer index.

        :param active_axes_indices: _description_
        :type active_axes_indices: _type_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: _type_
        """

        if self.equation_type in ("SINGLE-PHASE",
                                  "TWO-PHASE-LS",
                                  "SINGLE-PHASE-SOLID-LS"):
            material_field_indices = {
                "primitives": {
                    "density": 0,
                    "velocity": jnp.array(active_axes_indices)+1,
                    "pressure": 4,
                },
                "conservatives": {
                    "mass": 0,
                    "momentum": jnp.array(active_axes_indices)+1,
                    "energy": 4
                }
            }

        elif self.diffuse_interface_model == "5EQM":
            material_field_indices = {
                "primitives": {
                    **self.alpharho_i_dict,
                    **{"velocity": jnp.array([self.no_fluids + i for i in active_axes_indices]),
                    "pressure": self.no_fluids+3},
                    **self.alpha_i_dict,
                },
                "conservatives": {
                    **self.alpharho_i_dict,
                    **{"momentum": jnp.array([self.no_fluids + i for i in active_axes_indices]),
                    "energy": self.no_fluids+3},
                    **self.alpha_i_dict,
                }
            }

        else:
            raise NotImplementedError

        return material_field_indices
