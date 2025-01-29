# JAX-Fluids: A Differentiable Fluid Dynamics Package

JAX-Fluids is a fully-differentiable CFD solver for 3D, compressible single-phase and two-phase flows.
We developed this package with the intention to facilitate research at the intersection
of ML and CFD. It is easy to use - running a simulation only requires a couple 
lines of code. Written entirely in JAX, the solver runs on CPU/GPU/TPU and 
enables automatic differentiation for end-to-end optimization 
of numerical models. JAX-Fluids is parallelized using JAX primitives and 
scales efficiently on state-of-the-art HPC clusters (tested on up to 512 NVIDIA A100 GPUs
and on up to 2048 TPU-v3 cores).

To learn more about implementation details and details on numerical methods provided 
by JAX-Fluids, feel free to read our papers [here](https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466)
and [here](https://arxiv.org/abs/2402.05193).
And also check out the [documentation](https://jax-fluids.readthedocs.io/en/latest/index.html) of JAX-Fluids.

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
- Two-phase simulations via level-set method and diffuse-interface method
- Immersed solid boundaries via level-set method
- Positivity-preserving techniques
- Forcings for temperature, mass flow rate and kinetic energy spectrum
- Boundary conditions: Symmetry, Periodic, Wall, Dirichlet, Neumann
- CPU/GPU/TPU capability
- Parallel simulations on GPU & TPU

## Example simulations
Space shuttle at Mach 2 - Immersed solid boundary method via level-set

<img src="/docs/images/shuttle.png" alt="space shuttle at mach 2" height="300"/>

Shock-bubble interaction with diffuse-interface method - approx. 800M cells on TPUv3-64

<img src="/docs/images/diffuse_bubble_array.png" alt="diffuse-interface bubble array" height="300"/>

Shock-bubble interaction with level-set method - approx. 2B cells on TPUv3-256

<img src="/docs/images/levelset_bubble_array.png" alt="level-set bubble array" height="300"/>

Shock-induced collapse of air bubbles in water (click link for video)

https://www.youtube.com/watch?v=mt8HjZhm60U

## Pip Installation
Before installing JAX-Fluids, please ensure that you have
an up-to-date version of pip.
```bash
pip install --upgrade pip
```

### CPU-only support
To install the CPU-only version of JAX-Fluids, you can run
```bash
pip install --upgrade "jax[cpu]"
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS
pip install .
```
Note: if you want to install JAX-Fluids in editable mode,
e.g., for code development on your local machine, run
```bash
pip install -e .
```

Note: if you want to use jaxlib on a Mac with M1 chip, check the discussion [here](https://github.com/google/jax/issues/5501).

### GPU and CPU support
If you want to install JAX-Fluids with CPU AND GPU support, you must
first install JAX with GPU support. There are two ways to do this:
1) installing CUDA & cuDNN via pip,
2) installing CUDA & cuDNN by yourself.

See [JAX installation](https://jax.readthedocs.io/en/latest/installation.html) for details.

We recommend installing CUDA & cuDNN using pip wheels:
```bash
pip install --upgrade "jax[cuda12]"
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS
pip install -e .
```
For more information
on JAX on GPU please refer to the [github of JAX](https://github.com/google/jax)

## Quickstart
This github contains five [jupyter-notebooks](https://github.com/tumaer/JAXFLUIDS/tree/main/notebooks) which will get you started quickly.
They demonstrate how to run simple simulations like a 1D sod shock tube or 
a 2D air-helium shock-bubble interaction. Furthermore, they show how you can easily
switch the numerical and/or case setup in order to, e.g., increase the order
of the spatial reconstruction stencil or decrease the resolution of the simulation.

## Documentation
Check out the [documentation](https://jax-fluids.readthedocs.io/en/latest/index.html) of JAX-Fluids.

## Acknowledgements
We gratefully acknowledge access to TPU compute resources granted by Google's TRC program.

## Citation
JAX-Fluids 2.0: Towards HPC for differentiable CFD of compressible two-phase flows
https://doi.org/10.1016/j.cpc.2024.109433
```
@article{Bezgin2025,
   author = {Deniz A. Bezgin and Aaron B. Buhendwa and Nikolaus A. Adams},
   doi = {10.1016/j.cpc.2024.109433},
   issn = {00104655},
   journal = {Computer Physics Communications},
   month = {3},
   pages = {109433},
   title = {JAX-Fluids 2.0: Towards HPC for differentiable CFD of compressible two-phase flows},
   volume = {308},
   url = {https://linkinghub.elsevier.com/retrieve/pii/S0010465524003564},
   year = {2025},
}
```

JAX-Fluids: A fully-differentiable high-order computational fluid dynamics solver for compressible two-phase flows
https://doi.org/10.1016/j.cpc.2022.108527

```
@article{Bezgin2023,
   author = {Deniz A. Bezgin and Aaron B. Buhendwa and Nikolaus A. Adams},
   doi = {10.1016/j.cpc.2022.108527},
   issn = {00104655},
   journal = {Computer Physics Communications},
   month = {1},
   pages = {108527},
   title = {JAX-Fluids: A fully-differentiable high-order computational fluid dynamics solver for compressible two-phase flows},
   volume = {282},
   url = {https://linkinghub.elsevier.com/retrieve/pii/S0010465522002466},
   year = {2023},
}
```
## License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file or for details https://en.wikipedia.org/wiki/MIT_License.
