# type: ignore
import numpy as np

from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import TypedDict, nbintp, npintp


@njit(boundscheck=True)
def GenerateTensorNetwork(network, Nodel_int):  # noqa: C901
    networkdict = TypedDict.empty(nbintp, nbintp)
    nodes = len(Nodel_int)
    for k in range(nodes):
        networkdict[Nodel_int[k]] = k

    num_lines = network.shape[0]
    network_mask = np.zeros(num_lines, dtype=np.bool_)
    valid_count = 0

    for i in range(num_lines):
        start_node = network[i, 0]
        end_node = network[i, 1]

        # Check if both start and end are in our valid nodes dict
        if start_node in networkdict and end_node in networkdict:
            network_mask[i] = True
            valid_count += 1

    # Create valid_network and remap indices
    valid_network = np.empty((valid_count, 2), dtype=npintp)
    idx = 0
    for i in range(num_lines):
        if network_mask[i]:
            valid_network[idx, 0] = networkdict[network[i, 0]]
            valid_network[idx, 1] = networkdict[network[i, 1]]
            idx += 1

    # Build cache
    total_neighbours = 0
    neigh_count = np.zeros(nodes, dtype=npintp)
    for n in range(nodes):
        for line in range(valid_count):
            if valid_network[line, 0] == n or valid_network[line, 1] == n:
                neigh_count[n] += 1
                total_neighbours += 1

    neigh_offsets = np.zeros(nodes + 1, dtype=npintp)
    neigh_neighbors = np.empty(total_neighbours, dtype=npintp)
    neigh_lines_arr = np.empty(total_neighbours, dtype=npintp)

    for n in range(nodes):
        neigh_offsets[n + 1] = neigh_offsets[n] + neigh_count[n]

    fill_pos = neigh_offsets.copy()
    for n in range(nodes):
        for line in range(valid_count):
            if valid_network[line, 0] == n:
                p = fill_pos[n]
                neigh_neighbors[p] = valid_network[line, 1]
                neigh_lines_arr[p] = line
                fill_pos[n] += 1
            elif valid_network[line, 1] == n:
                p = fill_pos[n]
                neigh_neighbors[p] = valid_network[line, 0]
                neigh_lines_arr[p] = line
                fill_pos[n] += 1

    return valid_network, network_mask, neigh_neighbors, neigh_lines_arr, neigh_offsets
