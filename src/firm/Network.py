import numpy as np
from numba import njit, int64, uint64, prange
from numba.types import UniTuple
from numba.typed.typeddict import Dict as TypedDict
from datetime import datetime as dt


@njit
def network_neighbours(directconns, n, conn_cache):
    cache_result = conn_cache.get(n, None)
    if cache_result is not None:
        return cache_result
    result = np.stack((
        np.where(directconns[n]!=-1)[0], 
        directconns[n][np.where(directconns[n]!=-1)[0]]
    ))
    conn_cache[n] = result
    return result


@njit
def _nthary_network(network_piece, cache_0_donors):
    joins_start = cache_0_donors[network_piece[0]][0]
    joins_end = cache_0_donors[network_piece[-1]][0]
    
    joins_start = np.array([n for n in joins_start if n not in network_piece])
    num_srows = len(joins_start)
    joins_end = np.array([n for n in joins_end if n not in network_piece])
    num_erows = len(joins_end)
    
    _networkn = np.empty((num_srows+num_erows, len(network_piece) + 1), dtype=np.int64)
    
    for i in range(num_srows):
        _networkn[i, 0] = joins_start[i]
        _networkn[i, 1:] = network_piece
    for i in range(num_erows):
        _networkn[num_srows + i, -1] = joins_end[i]
        _networkn[num_srows + i, :-1] = network_piece
    
    return np.atleast_2d(_networkn)
    
@njit
def canonical_row(row):
    """Return the lexicographically smaller of the row or its reverse."""
    m = len(row)
    rev = np.empty(m, dtype=row.dtype)
    for i in range(m):
        rev[i] = row[m - 1 - i]

    for i in range(m):
        if row[i] < rev[i]:
            return row
        elif row[i] > rev[i]:
            return rev
    return row  # Equal

@njit
def dedup_networkn(networkn):
    n, m = networkn.shape
    keys = np.empty((n, m), dtype=networkn.dtype)

    for i in range(n):
        keys[i] = canonical_row(networkn[i])

    # Proxy lexsort using stable key — workaround for no lexsort in njit
    sort_idx = np.argsort(np.sum(keys * (np.arange(m)[::-1] + 1), axis=1))
    keep = np.ones(n, dtype=np.bool_)

    for i in range(1, n):
        same = True
        a = keys[sort_idx[i]]
        b = keys[sort_idx[i - 1]]
        for j in range(m):
            if a[j] != b[j]:
                same = False
                break
        if same:
            keep[sort_idx[i]] = False

    return networkn[keep]

@njit
def nthary_network(network_1, cache_0_donors):
    """primary, secondary, tertiary, ..., nthary"""
    """supply (n-1)thary to generate nthary etc."""
    network_cache = TypedDict.empty(int64, int64[:, :])
    num_rows = 0
    for i in range(len(network_1)):
        network_cache[i] = _nthary_network(network_1[i], cache_0_donors)
        num_rows += len(network_cache[i])        
    
    networkn = np.empty((num_rows, network_1.shape[1] + 1), dtype=np.int64)
    
    row = 0
    for i in range(len(network_1)): 
        nrows = len(network_cache[i])
        networkn[row : row + nrows, :] = network_cache[i]
        row += nrows
    
    networkn = dedup_networkn(networkn)
        
    return networkn

def generate_network(network, Nodel_int, networksteps):
    # direct network connections
    network_mask = np.array([(network == j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0) == 2
    network = network[network_mask, :]
    networkdict = {v: k for k, v in enumerate(Nodel_int)}
    # translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)
    
    trans_mask = np.zeros((len(Nodel_int), len(network)), np.bool_)
    for line, row in enumerate(network):
        trans_mask[row[0], line] = True

    directconns = -1 * np.ones((len(Nodel_int) + 1, len(Nodel_int) + 1), np.int64)
    for n, row in enumerate(network):
        directconns[*row] = n
        directconns[*row[::-1]] = n
    
    # build cache in advance of parallelisation
    cache_0_donors = TypedDict.empty(int64, int64[:, :])
    cache_n_donors = TypedDict.empty(UniTuple(int64, 2), int64[:, :, :])
    
    for n in Nodel_int:
        network_neighbours(directconns, n, cache_0_donors)
  
    nthary = network.copy()
    for leg in range(1, networksteps):
        nthary = nthary_network(nthary, cache_0_donors)
        for n in Nodel_int: 
            forward = np.where(n==nthary[:, 0])[0] 
            reverse = np.where(n==nthary[:,-1])[0]
            
            nodes = np.vstack((nthary[forward, 1:], nthary[reverse, :-1][:, ::-1]))
            lines = np.empty_like(nodes)
            
            for i in range(nodes.shape[0]):
                lines[i, 0] = directconns[n, nodes[i, 0]]
                for j in range(1, nodes.shape[1]):
                    lines[i, j] = directconns[nodes[i, j-1], nodes[i, j]]
            
            cache_n_donors[(n, leg)] = np.dstack((nodes, lines)).T
        
    return network, network_mask, trans_mask, cache_0_donors, cache_n_donors


