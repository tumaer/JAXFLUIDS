from jaxfluids_postprocess.matplotlib_utils import (
    create_1D_animation,
    create_1D_figure,
    create_2D_animation,
    create_2D_figure,
    create_3D_animation,
    create_3D_figure,
)

from jaxfluids_postprocess.post_process_utils import (
    create_xdmf_from_h5,
    generate_paraview_pngs,
    generate_paraview_pngs_2D,
    load_data,
    load_statistics,
    reassemble_parallel_data
)

from jaxfluids_postprocess.h5py_utils import (
    save_dict_to_h5,
    load_dict_from_h5,
)

__version__ = "0.1.0"
__author__ = "Deniz Bezgin, Aaron Buhendwa"

__all__ = (
    "create_1D_animation",
    "create_1D_figure",
    "create_2D_animation",
    "create_2D_figure",
    "create_3D_animation",
    "create_3D_figure",
    "create_xdmf_from_h5",
    "load_data",
    "load_statistics",
    "save_dict_to_h5",
    "load_dict_from_h5",
)