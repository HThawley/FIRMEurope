# import numpy as np
from numba import njit  # type: ignore

from firm.ForwardPass import ForwardPassRenewables, ForwardPassGas
from firm.ReversePass import ReversePassGas, ReversePassHydro, UpdateDynamics
# from firm.Utils import array_sum_2d_axis1  # type: ignore


@njit
def Simulate(solution):
    # Base Forward Pass
    ForwardPassRenewables(solution)

    # Sweep A: Hydro & Storage Trickling
    ReversePassHydro(solution)

    # Commit Trickling: Re-run balancing so trickle charged batteries actually discharge
    UpdateDynamics(solution)

    if (solution.operations.Mdeficit > 1e-6).any():
        # Dispatch Gas against the remaining deficits
        ForwardPassGas(solution)

    # Sweep B: Flexible Gas Trickling
    # Only run if deficits STILL exist after Sweep A exhausted free/stored energy
    total_deficit = solution.operations.Mdeficit.sum()
    if total_deficit > 1e-6:
        total_gas_headroom = (solution.assets.Cgas.sum() * solution.static.intervals
                              - solution.operations.Mgas.sum())  # avoid temp array

        if total_deficit > total_gas_headroom * 0.9:  # 0.9 is generous estimate of round trip efficiency
            solution.Feasible = False
            # 0.8 is approx round trip efficiency
            solution.estimated_deficit = (total_deficit - total_gas_headroom * 0.8)

        else:
            _feasible = ReversePassGas(solution)
            # Final Forward Pass (Lock in Gas actions)
            UpdateDynamics(solution)

            solution.Feasible = _feasible

    else:
        solution.Feasible = True

    solution.simulated = True
