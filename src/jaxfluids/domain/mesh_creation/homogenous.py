from typing import List, Tuple

import numpy as np

from jaxfluids.data_types.case_setup.domain import MeshStretchingSetup

def homogeneous(
        axis: int,
        nxi: int,
        domain_size_xi: List,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_sizes_xi = (domain_size_xi[1] - domain_size_xi[0]) / nxi
    cell_centers_xi = np.linspace(domain_size_xi[0] + cell_sizes_xi/2, domain_size_xi[1] - cell_sizes_xi/2, nxi)
    cell_faces_xi = np.linspace(domain_size_xi[0], domain_size_xi[1], nxi+1)
    
    shape = np.roll(np.s_[-1,1,1], axis)
    cell_centers_xi = cell_centers_xi.reshape(shape)
    cell_faces_xi = cell_faces_xi.reshape(shape)
    cell_sizes_xi = cell_sizes_xi.reshape(1,1,1)
    return cell_centers_xi, cell_faces_xi, cell_sizes_xi