from typing import List, Tuple

import numpy as np

from jaxfluids.data_types.case_setup.domain import MeshStretchingSetup

def boundary_layer(
        axis: int,
        nxi: int,
        domain_size_xi: List,
        stretching_setup: MeshStretchingSetup,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = stretching_setup.tanh_value
    cell_face_ids = np.arange(nxi + 1) / nxi
    cell_faces_xi = (domain_size_xi[1] - domain_size_xi[0]) * np.tanh(C * (cell_face_ids - 1)) / np.tanh(C) + \
                    domain_size_xi[1]
    cell_faces_xi[0] = domain_size_xi[0]
    cell_faces_xi[-1] = domain_size_xi[-1]
    cell_centers_xi = 0.5 * (cell_faces_xi[1:] + cell_faces_xi[:-1])
    cell_sizes_xi = cell_faces_xi[1:] - cell_faces_xi[:-1]
    cell_sizes_xi = cell_sizes_xi

    shape = np.roll(np.s_[-1,1,1], axis)
    cell_centers_xi = cell_centers_xi.reshape(shape)
    cell_faces_xi = cell_faces_xi.reshape(shape)
    cell_sizes_xi = np.array([cell_sizes_xi]).reshape(shape)

    return cell_centers_xi, cell_faces_xi, cell_sizes_xi