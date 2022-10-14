Features
========

JAX-Fluids comes with the following features:

* Fully-differentiable computational fluid dynamics code for compressible two-phase flows
* Modular object-oriented implementation 
* Explicit time stepping (Euler, RK2, RK3)
* Adaptive high-order reconstruction (WENO-3/5/7, WENO-CU6, WENO-3NN, TENO)
* Riemann solvers (Lax-Friedrichs, Rusanov, HLL, HLLC, Roe)
* Implicit turbulence SGS model ALDM
* Two-phase simulations via the Level-set method (arbitrary solid boundaries)
* Cartesian grid
* Forcings for temperature/mass flow rate/turbulence
* Input file and numerical setup via JSON files
* H5 and XDMF output
* CPU/GPU capability