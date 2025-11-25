import numpy as np
from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
from firm.Utils import (
    cclock,
    array_sum_2d_axis1,
)  # type: ignore


@njit
def Simulate(solution):  # noqa: C901
    if solution.static.profiling == -1:
        solution.profile.open_adj_simulation()
        start = cclock()
    if solution.static.profiling == 1:
        start = cclock()

    # allocate some memory
    working_buffer = np.zeros(solution.static.nodes, np.float64)

    for t in range(solution.static.intervals):
        # storage operation
        UpdateBalancingt(solution, t)

        # fill deficits from spilled power
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            if (solution.operations.Mspillage[t] > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    solution.operations.Mspillage[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t],
                )
                # update storage behaviour
                UpdateBalancingt(solution, t)

        # fill deficits by drawing down neighbours' storage reserves
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            # Msurplust = working_buffer
            GetSurplust(solution, t, working_buffer)

            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Timport[t],
                    solution.operations.Texport[t],
                )
                # update storage behaviour
                UpdateBalancingt(solution, t)

        UpdateSOCt(solution, t)

    # fill = working_buffer
    working_buffer[:] = 0.0
    # timestep backwards
    for t in range(solution.static.intervals - 1, -1, -1):
        if (working_buffer > 1e-6).any():
            if (solution.operations.Mspillage[t] > 1e-6).any():
                # cap fill by storage capacity
                for n in range(solution.static.nodes):
                    working_buffer[n] = min(
                        working_buffer[n],
                        (solution.assets.CphE[n] - solution.operations.Mphstorage[t - 1, n])
                        / solution.static.resolution
                        / solution.static.ph_charge_eff,
                    )
                # meet fill with neighbours' spillage - don't draw down power as this affects future SOC
                Interconnection(
                    solution,
                    working_buffer,
                    solution.operations.Mspillage[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t],
                )
                # fill adjusted in-place
        for n in range(solution.static.nodes):
            working_buffer[n] += solution.operations.Mdeficit[t, n] / solution.static.ph_charge_eff

    # fix storage traces
    BasicSimulate(solution)

    # meet deficits in place
    for t in range(solution.static.intervals):
        for n in range(solution.static.nodes):
            solution.operations.Mflexible[t, n] = min(solution.operations.Mdeficit[t, n], solution.assets.Cpeak[n])
        UpdateBalancingt(solution, t)

    # fill = working_buffer
    working_buffer[:] = 0.0
    for t in range(solution.static.intervals - 1, -1, -1):
        # timestep backwards
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            # recalculate transmission to meet deficits at point in time
            # remaining deficits are accumulated into `fill`
            working_buffer += PrechargeWithTranst(solution, t, solution.operations.Mdeficit[t])

        if (working_buffer > 1e-6).any():
            # clip fill if no precharge capacity (imperfect assumptions)
            ClipFillByStoraget(solution, t, working_buffer)

            if (working_buffer > 1e-6).any():
                # recalculate transmission to facilitate precharge
                # remaining deficits are forgotten about
                # fill adjusted in-place
                PrechargeWithTranst(solution, t, working_buffer)

    if solution.static.profiling == 1:
        solution.profile.calls.simulation += 1
        solution.profile.times.simulation += cclock() - start
    if solution.static.profiling == -1:
        solution.profile.calls.simulation += 1
        solution.profile.times.simulation += cclock() - start
        solution.profile.close_adj_simulation()

    BasicSimulate(solution)


@njit
def BasicSimulate(solution):
    if solution.static.profiling == 2 or solution.static.profiling == -1:
        start = cclock()

    for t in range(solution.static.intervals):
        UpdateUnbalancedt(solution, t)
        UpdateStoraget(solution, t)
        UpdateSOCt(solution, t)
        UpdateSpillDeft(solution, t)

    if solution.static.profiling == 2 or solution.static.profiling == -1:
        solution.profile.times.basic += cclock() - start
        solution.profile.calls.basic += 1


@njit
def GetSurplust(solution, t, Msurplust):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    for n in range(solution.static.nodes):
        Msurplust[n] = max(
            0,
            solution.operations.Mspillage[t, n]
            + solution.operations.Mphcharge[t, n]
            + min(solution.assets.CphP[n], solution.operations.Mphstorage[t - 1, n] / solution.static.resolution)
            - solution.operations.Mphdischarge[t, n],
        )
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.times.get_surplus += cclock() - start
        solution.profile.calls.get_surplus += 1


@njit
def ClipFillByStoraget(solution, t, fillt):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    # clip fill by storage capacity
    for n in range(solution.static.nodes):
        fillt[n] = min(
            fillt[n],
            (solution.assets.CphE[n] - solution.operations.Mphstorage[t - 1, n])
            / solution.static.resolution
            / solution.static.ph_discha_eff,
        )
        flex = min(
            fillt[n],
            solution.assets.Cpeak[n] - solution.operations.Mflexible[t, n],
            solution.assets.CphP[n]
            - solution.operations.Mphcharge[t, n]
            + solution.operations.Mphdischarge[t, n],
        )
        fillt[n] -= flex
        solution.operations.Mflexible[t, n] += flex

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.clip_fill += 1
        solution.profile.times.clip_fill += cclock() - start


@njit
def PrechargeWithTranst(solution, t, fillt):
    if solution.static.profiling == 2 or solution.static.profiling == -1:
        start = cclock()
    # original import/export
    _import = array_sum_2d_axis1(solution.operations.Timport[t])
    _export = array_sum_2d_axis1(solution.operations.Texport[t])
    # meet deficits just-in-time by importing flex from neighbours
    Interconnection(
        solution,
        fillt,
        solution.assets.Cpeak - solution.operations.Mflexible[t],
        solution.operations.Timport[t],
        solution.operations.Texport[t],
    )
    # flexible += iexports from neighbours
    solution.operations.Mflexible[t] += np.maximum(
        _import
        + _export
        - array_sum_2d_axis1(solution.operations.Timport[t] + solution.operations.Texport[t]),
        0
    )
    if solution.static.profiling == 2 or solution.static.profiling == -1:
        solution.profile.calls.trans_precharge += 1
        solution.profile.times.trans_precharge += cclock() - start
    # accumulate remaing deficits
    return solution.operations.Mdeficit[t] / solution.static.ph_charge_eff


@njit
def UpdateUnbalancedt(solution, t):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    for n in range(solution.static.nodes):
        _Timport = 0.0
        for m in range(solution.static.nhvi):
            _Timport += solution.operations.Timport[t, n, m]
            _Timport += solution.operations.Texport[t, n, m]
        solution.operations.Munbalanced[t, n] = (
            solution.operations.Mnetload[t, n] - solution.operations.Mflexible[t, n] - _Timport
        )
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.unbalancedt += 1
        solution.profile.times.unbalancedt += cclock() - start


@njit
def UpdateUnbalanced(solution):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()

    for t in range(solution.static.intervals):
        for n in range(solution.static.nodes):
            _Timport = 0.0
            for m in range(solution.static.nhvi):
                _Timport += solution.operations.Timport[t, n, m]
                _Timport += solution.operations.Texport[t, n, m]
            solution.operations.Munbalanced[t, n] = (
                solution.operations.Mnetload[t, n] - solution.operations.Mflexible[t, n] - _Timport
            )

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.unbalanced += 1
        solution.profile.times.unbalanced += cclock() - start


@njit
def UpdateStoraget(solution, t):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    for n in range(solution.static.nodes):
        solution.operations.Mphcharge[t, n] = min(
            -min(0, solution.operations.Munbalanced[t, n]),
            solution.assets.CphP[n],
            (solution.assets.CphE[n] - solution.operations.Mphstorage[t - 1, n])
            / solution.static.ph_charge_eff
            / solution.static.resolution
        )
        solution.operations.Mphdischarge[t, n] = min(
            max(0, solution.operations.Munbalanced[t, n]),
            solution.assets.CphP[n],
            solution.operations.Mphstorage[t - 1, n]
            / solution.static.resolution
        )

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.storage_behaviort += 1
        solution.profile.times.storage_behaviort += cclock() - start


@njit
def UpdateStorage(solution):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()

    for t in range(solution.static.intervals):
        for n in range(solution.static.nodes):
            solution.operations.Mphcharge[t, n] = min(
                -min(solution.operations.Munbalanced[t, n], 0), solution.assets.CphP[n])
            solution.operations.Mphdischarge[t, n] = min(
                max(solution.operations.Munbalanced[t, n], 0), solution.assets.CphP[n])

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.storage_behavior += 1
        solution.profile.times.storage_behavior += cclock() - start


@njit(fastmath=True)
def UpdateSOCt(solution, t):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    for n in range(solution.static.nodes):
        solution.operations.Mphstorage[t, n] = (
            solution.operations.Mphstorage[t - 1, n]
            + solution.static.resolution
            * (
                solution.operations.Mphcharge[t, n]
                * solution.static.ph_charge_eff
                - solution.operations.Mphdischarge[t, n]
            )
        )
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.update_soct += 1
        solution.profile.times.update_soct += cclock() - start


@njit(fastmath=True)
def UpdateSOC(solution):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()
    solution.operations.Mphstorage[-1] = 0.5 * solution.assets.CphE
    for t in range(solution.static.intervals):
        for n in range(solution.static.nodes):
            solution.operations.Mphcharge[t, n] = min(
                solution.operations.Mphcharge[t, n],
                (solution.assets.CphE[n] - solution.operations.Mphstorage[t - 1, n])
                / solution.static.ph_charge_eff
                / solution.static.resolution,
            )
            solution.operations.Mphdischarge[t, n] = min(
               solution.operations.Mphdischarge[t, n],
               solution.operations.Mphstorage[t - 1, n] / solution.static.resolution
            )
            solution.operations.Mphstorage[t, n] = (
                solution.operations.Mphstorage[t - 1, n]
                + solution.static.resolution
                * (
                    solution.operations.Mphcharge[t, n]
                    * solution.static.ph_charge_eff
                    - solution.operations.Mphdischarge[t, n]
                )
            )
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.update_soc += 1
        solution.profile.times.update_soc += cclock() - start


@njit
def UpdateSpillDeft(solution, t):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()

    for n in range(solution.static.nodes):
        _inter = (
            solution.operations.Munbalanced[t, n]
            + solution.operations.Mphcharge[t, n]
            - solution.operations.Mphdischarge[t, n]
        )
        solution.operations.Mdeficit[t, n] = max(0, _inter)
        solution.operations.Mspillage[t, n] = -min(0, _inter)

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.spilldeft += 1
        solution.profile.times.spilldeft += cclock() - start


@njit
def UpdateSpillDef(solution):
    if solution.static.profiling == 3 or solution.static.profiling == -1:
        start = cclock()

    for t in range(solution.static.intervals):
        for n in range(solution.static.nodes):
            _inter = (
                solution.operations.Munbalanced[t, n]
                + solution.operations.Mphcharge[t, n]
                - solution.operations.Mphdischarge[t, n]
            )
            solution.operations.Mdeficit[t, n] = max(_inter, 0)
            solution.operations.Mspillage[t, n] = -min(_inter, 0)

    if solution.static.profiling == 3 or solution.static.profiling == -1:
        solution.profile.calls.spilldef += 1
        solution.profile.times.spilldef += cclock() - start


@njit
def UpdateBalancingt(solution, t):
    # Convenience function as these three often go together
    # Not profiled
    UpdateUnbalancedt(solution, t)
    UpdateStoraget(solution, t)
    UpdateSpillDeft(solution, t)
