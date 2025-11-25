import csv
from datetime import datetime as dt

import numpy as np
from numba import njit, prange
from scipy.optimize import differential_evolution

from firm.Input import Evaluate, Solution, StaticData, Nodel
from firm.Costs import RawCosts
from firm.Parameters import Parameters, DE_Hyperparameters
from firm.Fileprinter import Fileprinter


def ObjectiveWrapper(xs, static_data, cost_model, fileprinter):
    result = ObjectiveParallel(xs.T, static_data, cost_model)
    if np.isnan(result).any():
        fileprinter(np.vstack((np.atleast_2d(result), xs)).T)
        fileprinter.Terminate()
        print(xs)
        print(np.where(np.isnan(result)))
        print(result)
        raise Exception
    fileprinter(np.vstack((np.atleast_2d(result), xs)).T)
    return result


@njit(parallel=True)
def ObjectiveParallel(xs, static_data, cost_model):
    result = np.empty((len(xs), 2), dtype=np.float64)
    for i in prange(len(xs)):
        result[i] = Objective(xs[i], static_data, cost_model)
    return result


@njit
def Objective(x, static_data, cost_model):
    """This is the objective function"""
    S = Solution(x, static_data)
    Evaluate(S, cost_model)
    return np.array([S.Lcoe, S.Penalties])


class CallbackClass:
    def __init__(self, display: int = 50, stagnation: int = 100, stag_rate: float = 1e-6):
        """
        This object is called after each iteration.
        display    - how often (# iterations) to print intermediate results to console
        stagnation - # of iterations with no improvement to best objective after which to terminate
        """
        self.it = 0
        self.display = display
        self.stagnation = stagnation
        self.stag_counter = 0
        self.stag_rate = stag_rate
        self.start = dt.now()
        self.elite = np.inf

    def __call__(self, intermediate_result):
        if self.display > 0:
            if self.it % self.display == 0:
                print(f"Iteration: {self.it}. Time taken: {dt.now()-self.start}. Best value: {intermediate_result.fun}")
        if self.stag_rate > 0 and self.stagnation > 1:
            if self.elite - intermediate_result.fun < self.stag_rate:
                self.stag_counter += 1
            else:
                self.elite = intermediate_result.fun
                self.stag_counter = 0
            if self.stag_counter == self.stagnation:
                if self.display > 0:
                    print(
                        f"Iteration: {self.it}. Time taken: {dt.now()-self.start}."
                        f"Best value: {intermediate_result.fun}"
                    )
                return True
        self.it += 1
        return False


def Optimise(static_data, hyperparameters):
    fileprinter = Fileprinter(
        f"Results/History{static_data.scenario}.csv",
        hyperparameters.f,
        header=["Cost", "Penalties"]
        + [f"pv{n}" for n in Nodel]
        + [f"rpv{n}" for n in Nodel]
        + [f"onsw{n}" for n in Nodel]
        + [f"offw{n}" for n in Nodel]
        + [f"nuke{n}" for n in Nodel]
        + [f"ccgt{n}" for n in Nodel]
        + [f"php{n}" for n in Nodel]
        + [f"phe{n}" for n in Nodel]
        + [f"bessp{n}" for n in Nodel]
        + [f"besse{n}" for n in Nodel]
        + [f"HVI{n}" for n in range(static_data.nhvi)],
        resume=False,
    )

    cost_model = RawCosts(static_data).CostFactors()

    starttime = dt.now()
    print("Optimisation starts at", starttime)
    result = differential_evolution(
        func=ObjectiveWrapper,
        args=(
            static_data,
            cost_model,
            fileprinter,
        ),
        bounds=list(zip(static_data.lb, static_data.ub)),
        tol=0,
        maxiter=hyperparameters.i,
        popsize=hyperparameters.p,
        mutation=hyperparameters.m,
        recombination=hyperparameters.r,
        disp=False,
        callback=CallbackClass(
            hyperparameters.v,
            *hyperparameters.s,
        ),
        polish=False,
        updating="deferred",
        x0=static_data.x0,
        vectorized=True,
    )
    fileprinter.Terminate()
    endtime = dt.now()
    timetaken = endtime - starttime
    print("Optimisation took", timetaken)

    with open(f"Results/Optimisation_resultx{static_data.scenario}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    return result, timetaken


@njit(parallel=True)
def _round_x(x0, static_data, cost_model):
    # step through 0.001, 0.01, 0.1, 1.0
    for i in range(3, 0, -1):
        # re-evaluate elite
        elite = Objective(x0, static_data, cost_model)
        # copy to prevent issues with parallelisation
        _x0 = x0.copy()
        for j in prange(len(x0)):
            # pick items below (0.001, 0.01, 0.1)
            if x0[j] < 0.1 ** i and x0[j] > 0:
                # copy to prevent issues with parallelisation
                _x = _x0.copy()
                # set to 0
                _x[j] = 0
                # evaluate
                re = Objective(_x, static_data, cost_model)
                if re <= elite:
                    # if no penalties, update x0
                    x0[j] = 0
    return x0


def Polish(
    x0,
    static_data,
    hyperparameters,
):

    fileprinter = Fileprinter(
        f"Results/History{static_data.scenario}.csv",
        hyperparameters.f,
        header=["Obj"]
        + [f"PV{n}" for n in range(static_data.pzones)]
        + [f"Wind{n}" for n in range(static_data.wzones)]
        + [f"PHP{n}" for n in range(static_data.nodes)]
        + [f"PHE{n}" for n in range(static_data.nodes)]
        + [f"HVI{n}" for n in range(static_data.nhvi)],
        resume=True,
    )

    cost_model = RawCosts(static_data).CostFactors()

    x0 = _round_x(x0, static_data, cost_model)

    lb_p, ub_p = static_data.lb.copy(), static_data.ub.copy()
    lb_p[np.where(x0 == 0)[0]] = 0
    ub_p[np.where(x0 == 0)[0]] = 0

    starttime = dt.now()
    print("Polishing starts at", starttime)
    result = differential_evolution(
        func=ObjectiveWrapper,
        args=(
            static_data,
            cost_model,
            fileprinter,
        ),
        bounds=list(zip(lb_p, ub_p)),
        tol=0,
        maxiter=hyperparameters.i,
        popsize=hyperparameters.p,
        mutation=hyperparameters.m,
        recombination=hyperparameters.r,
        disp=False,
        callback=CallbackClass(
            hyperparameters.v,
            *hyperparameters.s,
        ),
        polish=False,
        updating="deferred",
        x0=x0,
        vectorized=True,
    )
    fileprinter.Terminate()
    endtime = dt.now()
    timetaken = endtime - starttime
    print("Optimisation took", timetaken)

    with open(f"Results/Optimisation_resultx{static_data.scenario}.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    return result, timetaken


if __name__ == "__main__":
    parameters = Parameters(s=21, y=1, p=0, n=3)
    hyperparameters = DE_Hyperparameters(
        i=10,
        p=10,
        m=(0.5, 1.0),
        r=0.4,
        v=5,
        s=(10, 1),
        f=1,
    )
    polish_hparameters = DE_Hyperparameters(
        i=10,
        p=10,
        m=(0.5, 1.0),
        r=0.5,
        v=5,
        s=(10, 1),
        f=1,
    )

    static_data = StaticData(*parameters)
    cost_model = RawCosts(static_data).CostFactors()

    print(Objective(static_data.x0, static_data, cost_model))

    # result, time = Optimise(static_data, hyperparameters)
    # result, time = Polish(result.x, static_data, polish_hparameters)
