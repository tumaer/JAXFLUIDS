PRIMITIVES_TYPES = ("ZEROGRADIENT", "SYMMETRY", "PERIODIC", "INACTIVE", "LINEAREXTRAPOLATION",
                    "WALL", "ISOTHERMALWALL", "MASSTRANSFERWALL", "MASSTRANSFERWALL_PARAMETERIZED",
                    "ISOTHERMALMASSTRANSFERWALL", "DIRICHLET", "NEUMANN",
                    "SIMPLE_INFLOW", "SIMPLE_OUTFLOW",
                    "DIRICHLET_PARAMETERIZED")

SOLIDS_THERMAL_TYPES = ("ZEROGRADIENT", "ADIABATIC", "PERIODIC", "ISOTHERMAL",
                        "INACTIVE", "SYMMETRY")

LEVELSET_TYPES = ("ZEROGRADIENT", "SYMMETRY", "PERIODIC", "INACTIVE",
                  "DIRICHLET", "REINITIALIZATION", "LINEAREXTRAPOLATION")

EDGE_TYPES = (
    "PERIODIC_ANY", "ANY_PERIODIC",
    "SYMMETRY_ANY", "ANY_SYMMETRY")

VERTEX_TYPES = (
    "PERIODIC_ANY_ANY", "ANY_PERIODIC_ANY", "ANY_ANY_PERIODIC",
    "SYMMETRY_ANY_ANY", "ANY_SYMMETRY_ANY", "ANY_ANY_SYMMETRY")