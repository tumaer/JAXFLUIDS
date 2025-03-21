from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

def create_slice(indices: Array, axis: int) -> Tuple:
    """Creates a slice tuple where indices are placed
    in axis direction. E.g., for axis = 0, the slice is
    (..., indices, :, :).

    :param indices: _description_
    :type indices: Array
    :param axis: _description_
    :type axis: int
    :return: _description_
    :rtype: Tuple
    """
    s_ = []
    for axis_index in range(3):
        if axis_index == axis:
            s_.append(np.s_[indices])
        else:
            s_.append(np.s_[:])
    s_ = (...,) + tuple(s_)
    return s_


def generate_permutations(no_subdomains: int) -> List[List[Tuple[int,int]]]:
    """Generates the all to all permutations.

    Generates for each send a list of permutations
    in form source-target tuples. E.g., for no_subdomains = 3:
    [
    send0:  [(0,1), (1,2), (2,0)],
    send1:  [(0,2), (1,0), (2,1)],
    ]

    :param no_subdomains: _description_
    :type no_subdomains: int
    :return: _description_
    :rtype: List
    """
    no_sends = no_subdomains - 1
    subdomain_ids = np.arange(no_subdomains)
    permutations_list = []
    for i in range(no_sends):
        permutations = []
        for j in range(no_subdomains):
            source = j
            target = np.roll(subdomain_ids, -(i+1))[j]
            permutations_j = (source, target)
            permutations.append(permutations_j)
        permutations_list.append(permutations)
    return permutations_list

def generate_indices(
        cells_xi: int,
        no_subdomains: int
        ) -> Tuple[Array, Array, Array]:
    """Generates the indices that are required
    for communication and reassembly.

    :param cells_xi: _description_
    :type cells_xi: int
    :param no_subdomains: _description_
    :type no_subdomains: int
    :return: _description_
    :rtype: _type_
    """

    cell_ids = jnp.arange(cells_xi)
    cell_ids_split = jnp.split(cell_ids, no_subdomains)
    subdomain_ids = np.arange(no_subdomains)
    no_sends = no_subdomains - 1

    send_indices = []
    for i in range(no_sends):
        send_slices_devices = []
        for j in range(no_subdomains):
            indices = np.roll(subdomain_ids, -(i+1))[j]
            send_slices_device_i = cell_ids_split[indices]
            send_slices_devices.append(send_slices_device_i)
        send_slices_devices = jnp.stack(send_slices_devices,axis=0)
        send_indices.append(send_slices_devices)
    send_indices = jnp.stack(send_indices, axis=0)

    reassemble_indices = []
    for i in range(no_subdomains):
        reassemble_slices_i = []
        for j in range(no_subdomains):
            subdomain_id = np.roll(subdomain_ids, j)[i]
            reassemble_slices_i.append(subdomain_id)
        reassemble_slices_i = jnp.stack(reassemble_slices_i)
        reassemble_indices.append(reassemble_slices_i)
    reassemble_indices = jnp.stack(reassemble_indices)

    keep_indices = jnp.stack([
        cell_ids_split[i] for i in range(no_subdomains)
        ], axis=0)

    return send_indices, reassemble_indices, keep_indices