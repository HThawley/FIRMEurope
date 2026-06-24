import numpy as np
from numba import int64
from numba.typed.typeddict import Dict as TypedDict


def generate_network(network, Nodel_int):
    network_mask = np.isin(network[:, 0], Nodel_int) & np.isin(network[:, 1], Nodel_int)
    valid_network = network[network_mask, :]

    networkdict = {v: k for k, v in enumerate(Nodel_int)}

    valid_network = np.array([networkdict[n] for n in valid_network.flatten()], np.int64).reshape(valid_network.shape)

    trans_mask = np.zeros((len(Nodel_int), len(valid_network)), np.bool_)
    for line, row in enumerate(valid_network):
        trans_mask[row[0], line] = True

    cache_0_donors = TypedDict.empty(int64, int64[:, :])

    for n in range(len(Nodel_int)):
        donors = []
        for line, row in enumerate(valid_network):
            if row[0] == n:
                donors.append((row[1], line))
            elif row[1] == n:
                donors.append((row[0], line))

        if donors:
            pdonors = np.array([d[0] for d in donors], dtype=np.int64)
            plines = np.array([d[1] for d in donors], dtype=np.int64)
            cache_0_donors[n] = np.vstack((pdonors, plines))
        else:
            cache_0_donors[n] = np.empty((2, 0), dtype=np.int64)

    return valid_network, network_mask, trans_mask, cache_0_donors
