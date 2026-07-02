# type: ignore

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, TOLERANCE
from firm_ce.common.jit_overload import njit
from firm_ce.backend.tensor.dynamics import ResetAnnualBudgets, GetTotalPeakHeadroomBound
from firm_ce.backend.tensor.forward_pass import ForwardPassRenewables, ForwardPassPeak
from firm_ce.backend.tensor.reverse_pass import ReversePassPeak, ReversePassHydro, UpdateDynamics


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def Simulate(solution):
    ResetAnnualBudgets(solution)

    ForwardPassRenewables(solution)  # Renewables and storage first
    ReversePassHydro(solution)  # trickle charge batteries
    UpdateDynamics(solution)  # update bookkeeping

    if feasible_early_exit(solution, not (solution.operations.Mdeficit > TOLERANCE).any()):
        return

    ForwardPassPeak(solution)  # Dispatch Biomass, Biogas, Gas against the remaining deficits
    total_deficit = solution.operations.Mdeficit.sum()

    if feasible_early_exit(solution, not total_deficit <= TOLERANCE):
        return

    # remaining peak can only be used to precharge, not to directly meet load
    # 0.9 is generous estimate of [storage + network] efficiency
    total_peak_headroom = GetTotalPeakHeadroomBound(solution)
    if total_deficit > total_peak_headroom * 0.9:
        solution.feasible = False
        # 0.8 is closer approx of [storage + network] efficiency
        solution.estimated_deficit = total_deficit - total_peak_headroom * 0.8
        solution.simulated = True
        return

    ReversePassPeak(solution)
    UpdateDynamics(solution)

    final_deficit = solution.operations.Mdeficit.sum()
    solution.estimated_deficit = final_deficit
    solution.feasible = final_deficit <= TOLERANCE
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
