Welcome to the documentation of JAX-Fluids!
======================================

.. image:: ../images/header.jpg
      :width: 700

`JAX-Fluids <https://github.com/tumaer/JAXFLUIDS>`_ is a fully-differentiable CFD solver for 3D, compressible two-phase flows. 
We developed this package with the intention to push and facilitate research at the intersection of ML and CFD. 
It is easy to use - running a simulation only requires a couple lines of code. 
Written entirely in JAX, the solver runs on CPU/GPU/TPU and enables automatic differentiation for end-to-end optimization of numerical models.

To learn more about implementation details and details on numerical methods provided 
by JAX-Fluids, feel free to read `our paper <https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466>`_.

This documentation is work in progress.

.. raw:: html

	<iframe width="640" height="360" src="https://www.youtube.com/watch?v=mRuJF4qZX-Y"
	 title="Supersonic Turbulent Channel Flow"
	frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
	allowfullscreen></iframe>

Quick Installation
------------------
This is a quick installation guide to get you set up with JAX-Fluids. Please check out our detailed :doc:`installation guide <installation>` for more information!
Install ``jaxfluids`` to your Python environment.

.. code:: console

      $ git clone https://github.com/tumaer/JAXFLUIDS.git jaxfluids
      $ cd jaxfluids 
      $ pip install .

Let's run a quick simulation to check if everything is up and working. 

.. code:: console

      $ cd examples/examples_1D/02_sod
      $ python run_sod.py

.. toctree::
      :maxdepth: 1
      :caption: JAX-Fluids: First steps

      installation
      runsimulation
      tutorials

.. toctree::
      :maxdepth: 1
      :caption: Under the hood

      features
      available_modules
      change_log

.. toctree::
      :maxdepth: 2
      :caption: JAX-Fluids API

      jaxfluids

.. toctree::
      :maxdepth: 1
      :caption: Contact

      authors

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
