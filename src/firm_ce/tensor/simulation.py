# type: ignore

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, TOLERANCE
from firm_ce.common.jit_overload import njit
from firm_ce.tensor.dynamics import GetTotalPeakHeadroomBound, ResetAnnualBudgets, any_gt_2d
from firm_ce.tensor.forward_pass import ForwardPassPeak, ForwardPassRenewables
from firm_ce.tensor.reverse_pass import ReversePassHydro, ReversePassPeak, UpdateDynamics


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def Simulate(solution):
    ResetAnnualBudgets(solution)

    ForwardPassRenewables(solution)  # Renewables and storage first
    ReversePassHydro(solution)  # trickle charge batteries
    UpdateDynamics(solution)  # update bookkeeping

    if feasible_early_exit(solution, not any_gt_2d(solution.operations.Mdeficit, TOLERANCE)):
        return

    ForwardPassPeak(solution)  # Dispatch Biomass, Biogas, Gas against the remaining deficits
    res = solution.static.resolution
    total_deficit = solution.operations.Mdeficit.sum() * res

    if feasible_early_exit(solution, total_deficit <= solution.static.allowance):
        return

    total_peak_headroom = GetTotalPeakHeadroomBound(solution)
    if total_deficit > total_peak_headroom * 0.9:
        solution.feasible = False
        solution.estimated_deficit = total_deficit - total_peak_headroom * 0.8
        solution.simulated = True
        return

    ReversePassPeak(solution)
    UpdateDynamics(solution)

    final_deficit = solution.operations.Mdeficit.sum() * res
    solution.estimated_deficit = final_deficit
    solution.feasible = final_deficit <= solution.static.allowance
    solution.simulated = True
    return


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def feasible_early_exit(solution, condition):
    if condition:
        solution.feasible = True
        solution.estimated_deficit = 0.0
        solution.simulated = True
        return True
    return False
