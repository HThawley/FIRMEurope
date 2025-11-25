# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:51:51 2024

@author: u6942852
"""


import numpy as np
from numba import njit  # type: ignore

from firm.Utils import cclock, array_min, array_max_2d_axis1, array_sum_2d_axis0, zero_safe_division  # type: ignore


@njit
def Interconnection(solution, Fillt, Surplust, Importt, Exportt):  # noqa: C901
    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    _transmission = np.zeros(solution.static.nodes, np.float64)
    leg = 0
    # loop through nodes with deficits
    for n in range(solution.static.nodes):
        if Fillt[n] < 1e-6:
            continue
        # appropriate slice of network array
        # pdonors is equivalent to donors later on but has different ndim so needs to
        #   be a different variable name for static typing
        pdonors, pdonor_lines = solution.static.cache_0_donors[n]
        _usage = 0.0  # badly named but avoids creating more variables
        for d in pdonors:
            _usage += Surplust[d]

        if _usage < 1e-6:
            # continue if no surplus to be traded
            continue

        for d, l in zip(pdonors, pdonor_lines):
            _usage = 0.0
            for m in range(solution.static.nodes):
                _usage += Importt[m, l]
            # maximum exportable
            _transmission[d] = min(
                Surplust[d],  # power resource constraint
                solution.assets.Clines[l] - _usage,  # line capacity constraint
            )

        # scale down to fill requirement
        _usage = 0.0
        for m in range(solution.static.nodes):
            _usage += _transmission[m]
        if _usage > Fillt[n]:
            _scale = Fillt[n] / _usage
            _transmission *= _scale
            _usage *= _scale
        if _usage < 1e-6:
            continue

        for i in range(len(pdonors)):
            # record transmission
            Importt[n, pdonor_lines[i]] += _transmission[pdonors[i]]
            Exportt[pdonors[i], pdonor_lines[i]] -= _transmission[pdonors[i]]
            # adjust deficit/surpluses
            Surplust[pdonors[i]] -= _transmission[pdonors[i]]
            _transmission[pdonors[i]] = 0

        Fillt[n] -= _usage

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.interc0 += 1
        solution.profile.times.interc0 += cclock() - start

    # Continue with nthary transmission
    # Note: This code block works for primary transmission too, but is slower
    if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
        _import = np.zeros(Importt.shape, np.float64)
        _capacity = np.zeros(solution.static.nhvi, np.float64)
        # loop through secondary, tertiary, ..., nthary connections
        for leg in range(1, solution.static.networksteps):
            if solution.static.profiling:
                start = cclock()

            # loop through nodes with deficits
            for n in range(solution.static.nodes):
                if Fillt[n] < 1e-6:
                    continue

                donors, donor_lines = solution.static.cache_n_donors[(n, leg)]

                if donors.shape[1] == 0:
                    break  # break if no valid donors

                _usage = 0.0  # badly named variable but avoids extra variables
                for d in donors[-1]:
                    _usage += Surplust[d]

                if _usage < 1e-6:
                    continue

                _capacity[:] = solution.assets.Clines - array_sum_2d_axis0(Importt)
                for d, dl in zip(donors[-1], donor_lines.T):  # print(d,dl)
                    # power use of each line, clipped to maximum capacity of lowest leg
                    _import[d, dl] = min(array_min(_capacity[dl]), Surplust[d])

                for line in range(solution.static.nhvi):
                    # total usage of the line across all import paths
                    _usage = 0.0
                    for m in range(solution.static.nodes):
                        _usage += _import[m, line]
                    # if usage exceeds capacity
                    if _usage > _capacity[line]:
                        # unclear why this raises zero division error from time to time
                        _scale = zero_safe_division(_capacity[line], _usage)
                        for m in range(solution.static.nodes):
                            # clip all legs
                            if _import[m, line] > 1e-6:
                                for o in range(solution.static.nhvi):
                                    _import[m, o] *= _scale

                # intermediate calculation array
                _transmission = array_max_2d_axis1(_import)

                # scale down to fill requirement
                _usage = 0.0
                for m in range(solution.static.nodes):
                    _usage += _transmission[m]
                if _usage > Fillt[n]:
                    _scale = Fillt[n] / _usage
                    _transmission *= _scale
                    _usage *= _scale
                if _usage < 1e-6:
                    continue

                for nd, d, dl in zip(range(donors.shape[1]), donors[-1], donor_lines.T):  # print(nd, d, dl)
                    Importt[n, dl[0]] += _transmission[d]
                    Exportt[donors[0, nd], dl[0]] -= _transmission[d]
                    for step in range(leg):
                        Importt[donors[step, nd], dl[step + 1]] += _transmission[d]
                        Exportt[donors[step + 1, nd], dl[step + 1]] -= _transmission[d]

                # Adjust fill and surplus
                Fillt[n] -= _usage
                Surplust -= _transmission

                _import[:] = 0.0
                _capacity[:] = 0.0

                if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                    break

            if solution.static.profiling == 3 or solution.static.profiling == -1:
                if leg == 1:
                    solution.profile.calls.interc1 += 1
                    solution.profile.times.interc1 += cclock() - start
                elif leg == 2:
                    solution.profile.calls.interc1 += 1
                    solution.profile.times.interc1 += cclock() - start
                elif leg == 3:
                    solution.profile.calls.interc3 += 1
                    solution.profile.times.interc3 += cclock() - start

            if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                break

    return Importt, Exportt
