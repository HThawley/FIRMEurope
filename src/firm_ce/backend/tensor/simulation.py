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

    if (solution.operations.Mdeficit > TOLERANCE).any():
        ForwardPassPeak(solution)  # Dispatch Biomass, Biogas, Gas against the remaining deficits

    total_deficit = solution.operations.Mdeficit.sum()
    if total_deficit > TOLERANCE:
        total_gas_headroom = GetTotalPeakHeadroomBound(solution)

        # 0.9 is generous estimate of round trip + network efficiency
        if total_deficit > total_gas_headroom * 0.9:
            solution.feasible = False
            # 0.8 is approx (but still generous) round trip + network efficiency
            solution.estimated_deficit = total_deficit - total_gas_headroom * 0.8

        else:
            ReversePassPeak(solution)
            UpdateDynamics(solution)
            solution.estimated_deficit = solution.operations.deficit.sum()
            solution.feasible = not solution.operations.deficit.sum() > TOLERANCE

    else:
        solution.feasible = True
        solution.estimated_deficit = 0.0

    solution.simulated = True
