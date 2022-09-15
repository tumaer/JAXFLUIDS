# JAX-Fluids: A Differentiable Fluid Dynamics Package

JAX-Fluids is a fully-differentiable CFD solver for 3D, compressible two-phase flows.
We developed this package with the intention to push and facilitate research at the intersection
of ML and CFD. It is easy to use - running a simulation only requires a couple 
lines of code. Written entirely in JAX, the solver runs on CPU/GPU/TPU and 
enables automatic differentiation for end-to-end optimization 
of numerical models.

To learn more about implementation details and details on numerical methods provided 
by JAX-Fluids, read [our paper](https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466)

Authors:

- [Deniz A. Bezgin](https://www.epc.ed.tum.de/en/aer/mitarbeiter-innen/cv-2/a-d/m-sc-deniz-bezgin/)
- [Aaron B. Buhendwa](https://www.epc.ed.tum.de/en/aer/mitarbeiter-innen/cv-2/a-d/m-sc-aaron-buhendwa/)
- [Nikolaus A. Adams](https://www.epc.ed.tum.de/en/aer/members/cv/prof-adams/)

Correspondence via [mail](mailto:aaron.buhendwa@tum.de,mailto:deniz.bezgin@tum.de).

## Physical models and numerical methods

JAX-Fluids solves the Navier-Stokes-equations using the finite-volume-method on a Cartesian grid. 
The current version provides the following features:
- Explicit time stepping (Euler, RK2, RK3)
- High-order adaptive spatial reconstruction (WENO-3/5/7, WENO-CU6, WENO-3NN, TENO)
- Riemann solvers (Lax-Friedrichs, Rusanov, HLL, HLLC, Roe)
- Implicit turbulence sub-grid scale model ALDM
- Two-phase simulations via level-set method
- Immersed solid boundaries via level-set method
- Forcings for temperature, mass flow rate and kinetic energy spectrum
- Boundary conditions: Symmetry, Periodic, Wall, Dirichlet, Neumann
- CPU/GPU/TPU capability

## Example simulations
<img src="/docs/images/fullimplosion.png" alt="fullimplosion" height="250"/>
<img src="/docs/images/shockbubble_2d.png" alt="air helium shockbubble 2D" height="250"/>
<img src="/docs/images/shockbubble_3d.png" alt="air helium shockbubble 3D" height="250"/>
<img src="/docs/images/shuttle.png" alt="space shuttle at mach 2" height="250"/>

## Pip Installation
Before installing JAX-Fluids, please ensure that you have
an updated and upgraded pip version.
### CPU-only support
To install the CPU-only version of JAX-Fluids, you can run
```bash
git clone https://github.com/tumaer/JAXFLUIDS.git .
cd JAXFLUIDS
pip install .
```
Note that if you want to install JAX-Fluids in editable mode,
e.g., for code development on your local machine, run
```bash
pip install --editable .
```
### GPU and CPU support
If you want to install JAX-Fluids with CPU and GPU support, you must
first install [CUDA](https://developer.nvidia.com/cuda-downloads) -
we have tested JAX-Fluids with CUDA 11.1 or newer.
After installing CUDA, run the following
```bash
git clone https://github.com/tumaer/JAXFLUIDS.git .
cd JAXFLUIDS
pip install .[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For more information
on JAX on GPU please refer to the [github of JAX](https://github.com/google/jax)

## Quickstart
This github contains five jupyter-notebooks which will get you started quickly.
They demonstrate how to run simple simulations like a 1D sod shock tube or 
a 2D supersonic cylinder flow. Furthermore, they show how you can easily
switch up the numerical and/or case setup in order to, e.g., increase the order
of the spatial reconstruction stencil or decrease the resolution of the simulation.

## Upcoming features 
- 5-Equation diffuse interface model for multiphase flows 
- CPU/GPU/TPU parallelization based on homogenous domain decomposition

## Documentation
Will be available soon.

## Citation
https://doi.org/10.1016/j.cpc.2022.108527

```
@article{BEZGIN2022108527,
   title = {JAX-Fluids: A fully-differentiable high-order computational fluid dynamics solver for compressible two-phase flows},
   journal = {Computer Physics Communications},
   pages = {108527},
   year = {2022},
   issn = {0010-4655},
   doi = {https://doi.org/10.1016/j.cpc.2022.108527},
   url = {https://www.sciencedirect.com/science/article/pii/S0010465522002466},
   author = {Deniz A. Bezgin and Aaron B. Buhendwa and Nikolaus A. Adams},
   keywords = {Computational fluid dynamics, Machine learning, Differential programming, Navier-Stokes equations, Level-set, Turbulence, Two-phase flows}
} 
```
## License
This project is licensed under the GNU General Public License v3 - see 
the [LICENSE](LICENSE) file or for details https://www.gnu.org/licenses/.
