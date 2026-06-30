import numpy as np
from numba import njit, int64
from numba.typed.typeddict import Dict as TypedDict


@njit
def generate_network(network, Nodel_int):  # noqa: C901
    networkdict = TypedDict.empty(int64, int64)
    for k in range(len(Nodel_int)):
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
    valid_network = np.empty((valid_count, 2), dtype=np.int64)
    idx = 0
    for i in range(num_lines):
        if network_mask[i]:
            valid_network[idx, 0] = networkdict[network[i, 0]]
            valid_network[idx, 1] = networkdict[network[i, 1]]
            idx += 1

    # Build cache
    cache_0_donors = TypedDict.empty(int64, int64[:, :])
    nodes = len(Nodel_int)

    for n in range(nodes):
        # count the number of connections to pre-allocate arrays
        count = 0
        for line in range(valid_count):
            if valid_network[line, 0] == n or valid_network[line, 1] == n:
                count += 1

        # allocate and fill
        if count > 0:
            res_matrix = np.empty((2, count), dtype=np.int64)
            c = 0
            for line in range(valid_count):
                if valid_network[line, 0] == n:
                    res_matrix[0, c] = valid_network[line, 1]
                    res_matrix[1, c] = line
                    c += 1
                elif valid_network[line, 1] == n:
                    res_matrix[0, c] = valid_network[line, 0]
                    res_matrix[1, c] = line
                    c += 1

            cache_0_donors[n] = res_matrix
        else:
            cache_0_donors[n] = np.empty((2, 0), dtype=np.int64)

    return valid_network, network_mask, cache_0_donors
