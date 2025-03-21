from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

import jax.numpy as jnp
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup import ActivePhysicsSetup, ActiveForcingsSetup, SolidCouplingSetup

TUPLE_PRIMES = ("rho", "u", "v", "w", "p")

AVAILABLE_QUANTITIES = {
    "primitives" : ("density", "velocity", "pressure", "temperature"),
    "conservatives": ("mass", "momentum", "energy"),
    "levelset": ("levelset", "volume_fraction", "interface_pressure", "interface_velocity"),
    "solids": ("velocity", "temperature", "energy"),
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
    2) TWO-PHASE-LS
    3) DIFFUSE-INTERFACE-4EQM
    4) DIFFUSE-INTERFACE-5EQM

    TODO: should ids_mass and ids_energy be tuples for all equation_types???
    """

    def __init__(
            self,
            primes_tuple: Tuple,
            fluid_names: Tuple,
            levelset_model: str,
            diffuse_interface_model: str,
            cavitation_model: str,
            active_physics: ActivePhysicsSetup,
            active_forcings: ActiveForcingsSetup,
            solid_coupling: SolidCouplingSetup
            ) -> None:
        
        self.primes_tuple = primes_tuple
        self.no_primes = len(primes_tuple)

        self.fluid_names = fluid_names
        self.no_fluids = len(fluid_names)

        self.active_physics = active_physics
        self.active_forcings = active_forcings

        # LEVEL SET
        self.levelset_model = levelset_model
        self.solid_coupling = solid_coupling
        if levelset_model == "FLUID-FLUID":
            self.is_moving_levelset = True
            self.is_solid_levelset = False
        elif levelset_model == "FLUID-SOLID":
            self.is_solid_levelset = True
            # BUG ??? is_moving_levelset should be boolean
            self.is_moving_levelset = solid_coupling.dynamic in ("ONE-WAY", "TWO-WAY")
        else:
            self.is_solid_levelset = False
            self.is_moving_levelset = False

        # DIFFUSE INTERFACE
        self.diffuse_interface_model = diffuse_interface_model

        # CAVITATION
        self.cavitation_model = cavitation_model

        is_viscous = active_physics.is_viscous_flux
        is_heat_flux = active_physics.is_heat_flux
        self.is_compute_temperature = any((is_viscous, is_heat_flux))

        # SINGLE PHASE OR LEVELSET TWO-PHASE
        self.equation_type = "SINGLE-PHASE"
        self.no_equations = 5
        self.shape_equations = (self.no_equations,)
        self.ids_mass = 0
        self.s_mass = jnp.s_[0:1] 
        self.ids_velocity = (1, 2, 3)
        self.s_velocity = jnp.s_[1:4]
        self.s_momentum_xi = (jnp.s_[1:2], jnp.s_[2:3], jnp.s_[3:4])
        self.ids_energy = 4
        self.s_energy = jnp.s_[4:5]
        self.ids_mass_and_energy = (self.ids_mass, self.ids_energy)
        self.ids_mass_and_volume_fraction = None
        self.ids_momentum_and_energy = self.ids_velocity + (self.ids_energy,)
        self.s_momentum_and_energy = jnp.s_[1:5]
        self.ids_volume_fraction = None
        self.s_volume_fraction = None
        self.ids_species = None
        self.s_species = None
        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.conservatives_slices = jnp.s_[:]

        # Multi-component NSE
        self.species_tuple = None
        self.partial_density_ids_dict = None
        self.mass_fraction_ids_dict = None
        self.mole_fraction_ids_dict = None
        self.mass_fraction_tuple = None
        self.mole_fraction_tuple = None

        self.material_field_slices = {
            "primitives": {
                "density": self.s_mass,
                "velocity": self.s_velocity,
                "pressure": self.s_energy,
            },
            "conservatives": {
                "mass": self.s_mass,
                "momentum": self.s_velocity,
                "energy": self.s_energy,
            },
        }

        self.primitive_quantities = ("density",) + 3 * ("velocity",) + ("pressure",)

        # LEVEL-SET TWO-PHASE
        if self.levelset_model == "FLUID-FLUID":
            self.equation_type = "TWO-PHASE-LS"
            self.shape_equations = (self.no_equations, 2)

        # DIFFUSE INTERFACE MODEL - 4 EQUATION MODEL
        if self.diffuse_interface_model == "4EQM":
            self.initialize_diffuse_interface_4eqm()

        # DIFFUSE INTERFACE MODEL - 5 EQUATION MODEL
        if self.diffuse_interface_model == "5EQM":
            self.initialize_diffuse_interface_5eqm()
            
        # DIFFUSE INTERFACE MODEL - 6 EQUATION MODEL
        if self.diffuse_interface_model == "6EQM":
            # TODO 6EQM
            self.initialize_diffuse_interface_6eqm()

        # DIFFUSE INTERFACE MODEL - 7 EQUATION MODEL
        if self.diffuse_interface_model == "7EQM":
            # TODO 7EQM
            self.initialize_diffuse_interface_7eqm()
    
    def initialize_diffuse_interface_4eqm(self) -> None:
        """Initializes equation informations for the 4 equation
        diffuse interface model.
        """

        no_fluids = self.no_fluids

        self.equation_type = "DIFFUSE-INTERFACE-4EQM"
        self.no_equations = no_fluids + 4
        self.shape_equations = (self.no_equations,)
        self.conservatives_slices = jnp.s_[:-(no_fluids-1)]
        self.ids_mass = tuple(range(no_fluids))
        self.s_mass = jnp.s_[0:no_fluids]
        self.ids_velocity = (no_fluids, no_fluids+1, no_fluids+2)
        self.s_velocity = jnp.s_[no_fluids:no_fluids+3]
        self.s_momentum_xi = (
            jnp.s_[no_fluids:no_fluids+1],
            jnp.s_[no_fluids+1:no_fluids+2],
            jnp.s_[no_fluids+2:no_fluids+3])
        self.ids_energy = no_fluids + 3
        self.s_energy = jnp.s_[no_fluids+3:no_fluids+4]
        self.ids_mass_and_energy = self.ids_mass + (self.ids_energy,)
        self.ids_mass_and_volume_fraction = None
        self.ids_momentum_and_energy = self.ids_velocity + (self.ids_energy,)
        self.s_momentum_and_energy = jnp.s_[no_fluids:no_fluids+4]
        self.ids_volume_fraction = None
        self.s_volume_fraction = None
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
                "density": self.s_mass,
                "velocity": self.s_velocity,
                "pressure": self.s_energy,
            },
            "conservatives": {
                "mass": self.s_mass,
                "momentum": self.s_velocity,
                "energy": self.s_energy,
            },
        }

        self.primitive_quantities = self.no_fluids*("density",) \
            + 3 * ("velocity",) + ("pressure",)

    def initialize_diffuse_interface_5eqm(self) -> None:
        """ Initializes equation informations for the 5 equation
        diffuse interface model.
        """

        no_fluids = self.no_fluids

        self.equation_type = "DIFFUSE-INTERFACE-5EQM"
        self.no_equations = 2 * no_fluids + 3
        self.shape_equations = (self.no_equations,)
        self.conservatives_slices = jnp.s_[:-(no_fluids-1)]
        self.ids_mass = tuple(range(no_fluids))
        self.s_mass = jnp.s_[0:no_fluids]
        self.ids_velocity = (no_fluids, no_fluids+1, no_fluids+2)
        self.s_velocity = jnp.s_[no_fluids:no_fluids+3]
        self.s_momentum_xi = (
            jnp.s_[no_fluids:no_fluids+1],
            jnp.s_[no_fluids+1:no_fluids+2],
            jnp.s_[no_fluids+2:no_fluids+3])
        self.ids_energy = no_fluids + 3
        self.s_energy = jnp.s_[no_fluids+3:no_fluids+4]
        self.ids_mass_and_energy = self.ids_mass + (self.ids_energy,)
        self.ids_volume_fraction = tuple(range(no_fluids+4,(2*no_fluids+4)-1))
        self.s_volume_fraction = jnp.s_[no_fluids+4:(2*no_fluids+4)-1]
        self.ids_mass_and_volume_fraction = self.ids_mass + self.ids_volume_fraction
        self.ids_momentum_and_energy = self.ids_velocity + (self.ids_energy,)
        self.s_momentum_and_energy = jnp.s_[no_fluids:no_fluids+4]
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
                "density": self.s_mass,
                "velocity": self.s_velocity,
                "pressure": self.s_energy,
                "volume_fraction": self.s_volume_fraction,
            },
            "conservatives": {
                "mass": self.s_mass,
                "momentum": self.s_velocity,
                "energy": self.s_energy,
                "volume_fraction": self.s_volume_fraction,
            },
        }

        self.primitive_quantities = self.no_fluids*("density",) + 3 * ("velocity",)\
            + ("pressure",) + (self.no_fluids -1) * ("volume_fraction",)

    def initialize_diffuse_interface_6eqm(self) -> None:
        raise NotImplementedError

    def initialize_diffuse_interface_7eqm(self) -> None:
        raise NotImplementedError
    
    def get_material_field_indices(self, active_axes_indices):
        """Dictioanry assigning material field quantity
        to buffer index.

        :param active_axes_indices: _description_
        :type active_axes_indices: _type_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: _type_
        """

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS",):
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

        elif self.diffuse_interface_model == "4EQM":
            material_field_indices = {
                "primitives": {
                    **self.alpharho_i_dict,
                    **{"velocity": jnp.array([self.no_fluids + i for i in active_axes_indices]),
                    "pressure": self.no_fluids+3},
                },
                "conservatives": {
                    **self.alpharho_i_dict,
                    **{"momentum": jnp.array([self.no_fluids + i for i in active_axes_indices]),
                    "energy": self.no_fluids+3},
                }
            }

        else:
            raise NotImplementedError

        return material_field_indices
