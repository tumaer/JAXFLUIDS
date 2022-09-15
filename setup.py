import os
from setuptools import find_packages, setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_version():
    with open(os.path.join(_CURRENT_DIR, "src", "jaxfluids", "__init__.py")) as file:
        for line in file:
            if line.startswith("__version__"):
                return line[line.find("=") + 1:].strip(' \'"\n')

__version__ = get_version()

if __name__=='__main__':
    setup(
        name="jaxfluids",
        version=__version__,
        description="Fully-differentiable CFD solver for compressible two-phase flows.",
        author="Deniz Bezgin, Aaron Buhendwa",
        author_email="deniz.bezgin@tum.de, aaron.buhendwa@tum.de",
        long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
        long_description_content_type='text/markdown',
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=">=3.6",
        install_requires=[
            "dm-haiku",
            "h5py",
            "jax",
            "jaxlib",
            "matplotlib",
            "numpy",
            "optax",
        ],
        extras_require={
            # Use cuda to install CUDA version, use as follows:
            # $ pip install .[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
            "cuda": ["jaxlib"],
        },
        url="https://github.com/tumaer/JAXFLUIDS",
        license="GNU GPLv3",
        classifiers=[
            "Programming Language :: Python :: 3"
            "License :: OSI Approved :: GNU GPLv3 License"
            "Operating System :: OS Independent"
        ]
    )
