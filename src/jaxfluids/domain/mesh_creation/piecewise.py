from typing import List, Tuple

import numpy as np

from jaxfluids.data_types.case_setup.domain import MeshStretchingSetup

def fun(r, C, N):
    return C - r * (r**N - 1) / (r - 1)

def dfun(r, N):
    return -1.0 / (r - 1)**2 * (N * r**(N+1) - (N+1) * r**N + 1)

def piecewise(
        axis: int,
        stretching_setup: MeshStretchingSetup,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    piecewise_parameters = stretching_setup.piecewise_parameters

    cell_faces_list = []
    dx_const_list = []
    
    # CALCULATE ALL CONSTANT PIECES
    for i, piecewise in enumerate(piecewise_parameters):
        piecewise_type = piecewise.type
        
        if piecewise_type in ("INCREASING", "DECREASING"):
            cell_faces_list.append(None)
            dx_const_list.append(None)
        
        elif piecewise_type == "CONSTANT":
            lower_bound = piecewise.lower_bound
            upper_bound = piecewise.upper_bound
            L_const = upper_bound - lower_bound
            nx_const = piecewise.cells
            dx_const = L_const / nx_const
            
            cell_faces_const = compute_cell_faces_constant_region(
                upper_bound, lower_bound, nx_const)
            
            cell_faces_list.append(cell_faces_const)
            dx_const_list.append(dx_const)

        else:
            raise NotImplementedError
    
    # CALCULATE ALL NON-CONSTANT PIECES
    for i, piecewise in enumerate(piecewise_parameters):
        piecewise_type = piecewise.type
        
        if piecewise_type in ("INCREASING", "DECREASING"):
            lower_bound = piecewise.lower_bound
            upper_bound = piecewise.upper_bound
            nx = piecewise.cells

            if piecewise_type == "INCREASING":
                dx_const = dx_const_list[i-1]
            elif piecewise_type == "DECREASING":
                dx_const = dx_const_list[i+1]
            else:
                raise NotImplementedError

            cell_faces_stretched = compute_cell_faces_stretched_region(
                piecewise_type, dx_const, nx, lower_bound, upper_bound)

            cell_faces_list[i] = cell_faces_stretched
        
        elif piecewise_type == "CONSTANT":
            pass

        else:
            raise NotImplementedError

    cell_faces_list_new = []
    for i, cell_faces in enumerate(cell_faces_list):
        if i != len(cell_faces_list)-1:
            cell_face_slice = cell_faces[:-1]
        else:
            cell_face_slice = cell_faces
        cell_faces_list_new.append(cell_face_slice)
    cell_faces = np.concatenate(cell_faces_list_new)
    cell_centers = 0.5 * (cell_faces[1:] + cell_faces[:-1])
    cell_sizes = cell_faces[1:] - cell_faces[:-1]

    shape = np.roll(np.s_[-1,1,1], axis)
    cell_centers = cell_centers.reshape(shape)
    cell_faces = cell_faces.reshape(shape)
    cell_sizes = np.array([cell_sizes]).reshape(shape)
    return cell_centers, cell_faces, cell_sizes

def compute_cell_faces_constant_region(
        upper_bound: float,
        lower_bound: float,
        nx: int
        ):
    cell_faces_xi = np.linspace(lower_bound, upper_bound, nx+1)
    return cell_faces_xi

def compute_cell_faces_stretched_region(
        type: str,
        dxi_fine: float,
        nxi_coarse: int,
        lower_bound: float,
        upper_bound: float,
        ):

    L_coarse = upper_bound - lower_bound
    coarsening_parameter = 1.1
    C = L_coarse / dxi_fine
    f = fun(coarsening_parameter, C, nxi_coarse)
    res = np.abs(f)
    tol = 1e-9
    maxiter = 100
    iter = 0
    while res > tol and iter < maxiter:
        coarsening_parameter = coarsening_parameter - f / dfun(coarsening_parameter, nxi_coarse)
        f = fun(coarsening_parameter, C, nxi_coarse)
        res = np.abs(f)
        iter = iter + 1
    assert res <= tol, f"Mesh generation has not converged in {iter} iterations. Residual error is {res:3.2e}."
    cell_sizes = dxi_fine * coarsening_parameter**np.arange(1, nxi_coarse+1)



    if type == "DECREASING":
        cell_faces = np.flip(upper_bound - np.cumsum(cell_sizes))
        cell_faces[0] = lower_bound
        cell_faces = np.concatenate([cell_faces, np.array([upper_bound])])
    elif type == "INCREASING":
        cell_faces = lower_bound + np.cumsum(cell_sizes)
        cell_faces[-1] = upper_bound
        cell_faces = np.concatenate([np.array([lower_bound]), cell_faces])

    return cell_faces