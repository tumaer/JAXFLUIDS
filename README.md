# JAXFLUIDS - A Differentiable Fluid Dynamics Package
Physical systems are governed by partial differential equations (PDEs).
While PDEs are typically solved with numerical methods, the recent success of machine learning (ML)
has shown that ML methods can provide novel avenues of finding solutions to PDEs.
The Navier-Stokes equations describe fluid flows and are representative of nonlinear physical systems 
with complex spatio-temporal interactions.
Fluid flows are omnipresent in nature and engineering applications, and their accurate simulation is essential
for providing insights into these processes.
ML is becoming more and more present in computational fluid dynamics (CFD).
However, up to this date, there does not exist a general-purpose ML-CFD package which provides

1) powerful state-of-the-art numerical methods, 
2) seamless hybridization of ML with CFD, 
3) end-to-end automatic differentiation capabilities.

Automatic differentiation in particular is essential to ML-CFD research as it provides gradient information and enables 
optimization of preexisting and novel CFD models. 
In this work, we propose JAX-FLUIDS: a comprehensive fully-differentiable compressible 
two-phase CFD Python package.
JAX-FLUIDS allows the simulation of complex fluid dynamics with phenomena like three-dimensional turbulence, 
compressibility effects, and two-phase flows.
Written entirely in JAX, it is straightforward to include existing ML models into the proposed framework.
Furthermore, JAX-FLUIDS enables end-to-end optimization.
I.e., ML models are optimized with gradients that are backpropagted through the entire CFD algorithm, and therefore
contain not only information of the underlying PDE but also of the applied numerical methods.
We believe that a Python package like JAX-FLUIDS is crucial to facilitate research at the intersection of ML and CFD,
and may pave the way for an era of differentiable fluid dynamics.

# Installation
This github is currently under construction. The source code will be available upon publication of the article.
# Authors

[Deniz A. Bezgin](https://www.epc.ed.tum.de/en/aer/mitarbeiter-innen/cv-2/a-d/m-sc-deniz-bezgin/)
[Aaron B. Buhendwa](https://www.epc.ed.tum.de/en/aer/mitarbeiter-innen/cv-2/a-d/m-sc-aaron-buhendwa/)
[Nikolaus A. Adams](https://www.epc.ed.tum.de/en/aer/members/cv/prof-adams/)

Correspondence via [mail](mailto:aaron.buhendwa@tum.de,mailto:deniz.bezgin@tum.de).

# License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file for details

# Citation
https://arxiv.org/abs/2203.13760
