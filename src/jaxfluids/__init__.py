"""
JAX-FLUIDS

A fully-differentiable computational fluid dynamics solver
for compressible two-phase flows. Solves the compressible 
single or two-phase Navier-Stokes equations in three spatial
dimensions.

Let's make Machine Learning + Computational Fluid Dynamics great!

Contact:
deniz.bezgin@tum.de / aaron.buhendwa@tum.de

Documentation
-------------

Documentation available at: https://github.com/tumaer/JAXFLUIDS

Examples
--------

The examples folder includes example CFD simulations in
one, two, and three dimensions.

For example in examples/examples_1D/02_sod

    >>> python run_sod.py

Available subpackages
---------------------
forcing
    Implements external forces for the Navier-Stokes equations
iles
    Implicit Large-Eddy Simulation models
io_utils
    Tools for logging and writing output
levelset
    Comprehensive level-set implementation for two-phase flows
materials
    Materials and material managers
post_process
    Tools for post-processing CFD simulations
shock_sensor
    Shock-sensors
stencils
    Spatial reconstruction and derivative stencils
time_integration
    Explicit time integration schemes up to third-order
turb
    Tools for turbulent initial conditions and statistics

"""

from jaxfluids.initialization.initialization_manager import InitializationManager
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager

__version__ = "0.2.0"
__author__ = "Deniz Bezgin, Aaron Buhendwa"


__all__ = (
    "InitializationManager", 
    "InputManager",
    "SimulationManager", 
)
