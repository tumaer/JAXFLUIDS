.. highlight:: shell

============
Installation
============


* It is recommended to use a conda virtual environment for this project. Conda can be installed following these instructions_. After successfully installing conda, the environment for this project can be created and activated:

.. _instructions: https://docs.conda.io/en/latest/miniconda.html

.. code-block:: console

    $ conda create --name venv_jaxfluids python=3.8
    $ conda activate venv_jaxfluids


CPU-only support
----------------

* To install the CPU-only version of JAX-Fluids, you can run

.. code-block:: console

    $ git clone https://github.com/tumaer/JAXFLUIDS.git # For ssh
    $ git clone https://github.com/tumaer/JAXFLUIDS.git # For https
    $ cd JAXFLUIDS
    $ pip install .

* This clones the JAX-Fluids repository and installs the package via pip. If you want to install JAX-Fluids in editable mode, e.g., for code development on your local machine, run

.. code-block:: console

    $ pip install --editable .

.. note::
    If you want to use ``jaxlib`` on a Mac with M1 chip, check the discussion here_.

.. _here: https://github.com/google/jax/issues/5501

GPU and CPU support
-------------------

* If you want to install JAX-Fluids with CPU and GPU support, you must first install CUDA_ - we have tested JAX-Fluids with CUDA 11.1 or newer. After installing CUDA, run the following

.. _CUDA: https://developer.nvidia.com/cuda-downloads

.. code-block:: console

    $ git clone https://github.com/tumaer/JAXFLUIDS.git
    $ cd JAXFLUIDS
    $ pip install .[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

* In case a GPU is available, ``jaxfluids`` uses it automatically. Nevertheless, it is possible to switch to CPU mode, or to select a specific GPU. This can be done using the following commands:

.. code-block:: console

    $ export CUDA_VISIBLE_DEVICES="" # For CPU mode
    $ export CUDA_VISIBLE_DEVICES="number" # For GPU mode, where number is the ID of the GPU that should be used

.. note::
    ``jax-fluids`` will throw a warning if it falls back to GPU usage if no GPU is found.
    If you intend to run on a GPU, you can check the GPU utilization by running ``$ nvidia-smi``
    in your terminal.

* With the successfully installed package, the [example cases](cases) are a good starting point.