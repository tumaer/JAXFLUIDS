from typing import List, Tuple

import numpy as np

from jaxfluids.data_types.case_setup.domain import MeshStretchingSetup

def channel(
        axis: int,
        nxi: int,
        domain_size_xi: List,
        stretching_setup: MeshStretchingSetup,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = stretching_setup.tanh_value
    center_line = 0.5 * (domain_size_xi[1] + domain_size_xi[0])
    delta = 0.5 * (domain_size_xi[1] - domain_size_xi[0])
    cell_face_ids = np.arange(nxi + 1) / nxi
    
    cell_faces_xi = delta * np.tanh(C * (2 * cell_face_ids - 1)) / np.tanh(C) + center_line
    cell_centers_xi = 0.5 * (cell_faces_xi[1:] + cell_faces_xi[:-1])
    cell_sizes_xi = cell_faces_xi[1:] - cell_faces_xi[:-1]
    
    shape = np.roll(np.s_[-1,1,1], axis)
    cell_centers_xi = cell_centers_xi.reshape(shape)
    cell_faces_xi = cell_faces_xi.reshape(shape)
    cell_sizes_xi = np.array([cell_sizes_xi]).reshape(shape)

    return cell_centers_xi, cell_faces_xi, cell_sizes_xi