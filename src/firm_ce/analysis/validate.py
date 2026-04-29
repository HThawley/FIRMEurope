# type: ignore
import os
from datetime import datetime
import numpy as np
from warnings import warn

from firm_ce.analysis.accessor import Accessor
from firm_ce.common.constants import VALIDATION_TOL
from firm_ce.optimisation.single_time import evaluate


class ValidationWarning(UserWarning):
    pass


class Validation:
    def __init__(self, solution, results_dir):

        self.solution = solution
        if getattr(self.solution, "evaluated", False) is False:
            evaluate(solution)

        self.results_dir = results_dir

        self.accessor = Accessor(self.solution, 1.0)
        self.verbose = True
        self.logs = {}
        self._current_category = "General"

    def validate(self, verbose: bool = True) -> bool:
        self.verbose = verbose
        self.logs.clear()

        checks = {
            "Decision Variables Bounds": self.check_build_bounds,
            "Generator Limits": self.check_generator_limits,
            "Transmission Limits": self.check_transmission_limits,
            "Energy Balance & Flows": self.check_energy_balance_and_flows,
            "Storage Constraints": self.check_storage_limits,
            "Storage Accrual": self.check_storage_accrual,
            "Fuel Limits": self.check_fuel_limits,
        }

        for check_name, check_func in checks.items():
            self._current_category = check_name
            self.logs[check_name] = []
            check_func()

        failed_checks = [name for name, issues in self.logs.items() if issues]

        if failed_checks:
            if self.verbose:
                warn(f"Validation FAILED for: {', '.join(failed_checks)}", ValidationWarning)
            return False

        print("Validation PASSED: No issues found on any check.")
        return True

    def _log(self, msg: str):
        """Records the log under the current check category."""
        self.logs[self._current_category].append(msg)
        if self.verbose:
            warn(f"[{self._current_category}] {msg}", ValidationWarning)

    def dump_logs(self, filename: str = "validation_report.txt") -> None:
        """Dumps formatted logs to a human-readable text file."""
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(" VALIDATION REPORT\n")
            f.write(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            total_issues = sum(len(issues) for issues in self.logs.values())

            if total_issues == 0:
                f.write("STATUS: PASSED\n")
                f.write("No boundary, operational, or energy balance violations found.\n")
                return

            f.write(f"STATUS: FAILED ({total_issues} total issues found)\n\n")

            for category, issues in self.logs.items():
                if issues:
                    f.write(f"--- {category.upper()} ---\n")
                    for i, msg in enumerate(issues, 1):
                        f.write(f"  {i}. {msg}\n")
                    f.write("\n")

    def check_build_bounds(self) -> bool:
        """Check that all decision variables (new builds) are within allowed limits."""
        passed = True
        asset_groups = [self.solution.fleet.generators, self.solution.fleet.storages, self.solution.network.major_lines]

        for group in asset_groups:
            for asset in group.values():
                init_p, new_p, min_p, max_p = self.accessor.get_build_power(asset)

                if new_p < min_p - VALIDATION_TOL or new_p > max_p + VALIDATION_TOL:
                    self._log(
                        f"Bounds Violation: {asset.name} power new_build ({new_p:.3f}) outside [{min_p}, {max_p}]."
                    )
                    passed = False

                try:
                    init_e, new_e, min_e, max_e = self.accessor.get_build_energy(asset)
                    if new_e < min_e - VALIDATION_TOL or new_e > max_e + VALIDATION_TOL:
                        self._log(
                            f"Bounds Violation: {asset.name} power new_build ({new_e:.3f}) outside [{min_e}, {max_e}]."
                        )
                        passed = False

                except ValueError as e:
                    if asset.object_class == 'storage':
                        raise e

        return passed

    def check_generator_limits(self) -> bool:
        """Check that all generators are within their nominal operational limits."""
        passed = True
        for asset in self.solution.fleet.generators.values():
            capacity = self.accessor.get_power_capacity(asset)
            power = self.accessor.get_power_trace(asset)

            if np.any(power < -VALIDATION_TOL):
                exceedance = np.maximum(-power - VALIDATION_TOL, 0)
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Generator Violation: {asset.name} dispatch drops below 0 by up to {max_violation}."
                    f"Found: {count} exceedances. First as t={t1}. Largest at t={t2}"
                )
                passed = False

            if np.any(power > capacity + VALIDATION_TOL):
                exceedance = np.maximum(power - capacity - VALIDATION_TOL, 0)
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Generator Violation: {asset.name} dispatch exceeds capacity by up to {max_violation:.4f}."
                    f"Found: {count} exceedances. First as t={t1}. Largest at t={t2}"
                )
                passed = False

        return passed

    def check_transmission_limits(self) -> bool:
        """Check that all transmission lines are within their nominal limits."""
        passed = True
        for asset in self.solution.network.major_lines.values():
            capacity = self.accessor.get_power_capacity(asset)
            flows = self.accessor.get_power_trace(asset)

            if np.any(np.abs(flows) > capacity + VALIDATION_TOL):
                exceedance = np.maximum(np.max(np.abs(flows)) - capacity - VALIDATION_TOL, 0)
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Transmission Violation: {asset.name} flow magnitude exceeds capacity by up to {max_violation:.4f}."
                    f"Found: {count} exceedances. First as t={t1}. Largest at t={t2}"
                )
                passed = False
        return passed

    def check_energy_balance_and_flows(self) -> bool:
        """Check that transmission flows sum as expected, efficiency is applied, and nodes balance."""
        passed = True
        nodes = self.solution.network.nodes

        # Pre-allocate zero arrays for rapid vectorized nodal summation
        node_balances = {n.id: np.zeros_like(n.data) for n in nodes.values()}

        for node in nodes.values():
            node_balances[node.id] -= self.accessor.get_power_trace(node)
            # protect against possible future changes to sign convention
            node_balances[node.id] += np.abs(self.accessor.get_deficit_trace(node))
            node_balances[node.id] -= np.abs(self.accessor.get_spillage_trace(node))

        for gen in self.solution.fleet.generators.values():
            node_balances[gen.node.id] += self.accessor.get_power_trace(gen)

        for stor in self.solution.fleet.storages.values():
            # positive is generation, negative is charge
            node_balances[stor.node.id] += self.accessor.get_power_trace(stor)

        # -- Transmission (Flows and Efficiency) --
        for line in self.solution.network.major_lines.values():
            f = self.accessor.get_transmission_trace(line)
            eff = self.accessor.get_efficiency(line)

            # Split directional flows for efficiency application
            f_pos = np.where(f > 0, f, 0)
            f_neg = np.where(f < 0, -f, 0)

            # Initial node exports positive flows, imports negative flows * eff
            node_balances[line.node_start.id] -= f_pos
            node_balances[line.node_start.id] += f_neg * eff

            # Terminal node imports positive flows * eff, exports negative flows
            node_balances[line.node_end.id] += f_pos * eff
            node_balances[line.node_end.id] -= f_neg

        # -- Evaluate Nodal Mismatch --
        for node in nodes.values():
            exceedance = np.maximum(np.abs(node_balances[node.id]) - VALIDATION_TOL, 0)
            if np.any(exceedance > 0):
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                max_v_p, c_p, t1_p, t2_p = get_exceedance_stats(np.maximum(node_balances[node.id] - VALIDATION_TOL, 0))
                max_v_n, c_n, t1_n, t2_n = get_exceedance_stats(np.minimum(node_balances[node.id] - VALIDATION_TOL, 0))

                self._log(
                    f"Energy Balance Violation: Node {node.name} failed by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                    f"\n\tPositive exceedances - Found: {c_p}. First at t={t1_p}. Largest ({max_v_p}) at t={t2_p}"
                    f"\n\tNegative exceedances - Found: {c_n}. First at t={t1_n}. Largest ({max_v_n}) at t={t2_n}"
                )
                passed = False

        return passed

    def check_storage_limits(self) -> bool:
        """Ensure stored energy stays between 0 and energy capacity."""
        passed = True
        for asset in self.solution.fleet.storages.values():
            se = self.accessor.get_storage_level_trace(asset)
            cap = self.accessor.get_energy_capacity(asset)

            if np.any(se < -VALIDATION_TOL):
                exceedance = np.maximum(-se - VALIDATION_TOL, 0)
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Storage Violation: {asset.name} energy dropped below 0 by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                )
                passed = False

            if np.any(se > cap + VALIDATION_TOL):
                exceedance = np.maximum(se - cap - VALIDATION_TOL, 0)
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Storage Violation: {asset.name} energy exceeded capacity by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                )
                passed = False

        return passed

    def check_storage_accrual(self) -> bool:
        """Check that storage energy accrues according to charge/discharge and efficiency."""
        passed = True
        res = self.solution.static.resolution

        for asset in self.solution.fleet.storages.values():
            se = self.accessor.get_storage_level_trace(asset)
            cap = self.accessor.get_energy_capacity(asset)

            # both abs, protect against future sign convention change
            charge = np.abs(self.accessor.get_charge_trace(asset))
            discharge = np.abs(self.accessor.get_discharge_trace(asset))

            eff_c = self.accessor.get_charge_efficiency(asset)
            eff_d = self.accessor.get_discharge_efficiency(asset)

            if self.accessor.has_inflows(asset):
                inflows = self.accessor.get_inflow_trace(asset)
            else:
                inflows = np.zeros_like(se)

            expected_energy_delta = (charge * eff_c - discharge / eff_d) * res + inflows

            # Apply physical boundary limits to calculate the expected next state
            raw_next_se = se[:-1] + expected_energy_delta[1:]
            expected_next_se = np.clip(raw_next_se, 0.0, cap)

            # Compare actual next state to bounded expected next state
            mismatch = np.abs(se[1:] - expected_next_se)
            exceedance = np.maximum(mismatch - VALIDATION_TOL, 0)

            if np.any(exceedance > 0):
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                # t1, t2 shifted by 1 due to difference array slicing
                self._log(
                    f"Storage Accrual Violation: {asset.name} mismatched by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1+1}. Largest at t={t2+1}"
                )
                passed = False

        return passed

    def check_fuel_limits(self) -> bool:
        """Check that fuel energy never drops below 0 and depletes correctly based on generator dispatch."""
        passed = True
        res = self.solution.static.resolution

        # Group generators by their associated fuel
        fuel_gens = {fuel.id: [] for fuel in self.solution.fleet.fuels.values()}
        for gen in self.solution.fleet.generators.values():
            fuel_gens[gen.fuel.id].append(gen)

        for fuel in self.solution.fleet.fuels.values():
            rem_energy = self.accessor.get_remaining_energy_trace(fuel)

            if np.isinf(rem_energy[0]):
                continue

            # Bounds Check: Fuel >= 0
            if np.any(rem_energy < -VALIDATION_TOL):
                exceedance = np.maximum(-rem_energy - VALIDATION_TOL, 0)
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Fuel Limit Violation: {fuel.name} remaining energy dropped below 0 by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                )
                passed = False

            # Accrual/Depletion Check
            gens = fuel_gens.get(fuel.name, [])
            if not gens:
                continue

            total_dispatch_power = np.zeros_like(rem_energy)
            for gen in gens:
                total_dispatch_power += self.accessor.get_power_trace(gen)

            expected_change = -total_dispatch_power * res
            actual_change = rem_energy[1:] - rem_energy[:-1]

            mismatch = np.abs(actual_change - expected_change[1:])
            exceedance = np.maximum(mismatch - VALIDATION_TOL, 0)

            if np.any(exceedance > 0):
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                # t1, t2 shifted by 1 due to difference array slicing
                self._log(
                    f"Fuel Depletion Violation: {fuel.name} mismatched dispatch by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1+1}. Largest at t={t2+1}"
                )
                passed = False

        return passed


def get_exceedance_stats(exceedance):
    max_violation = np.max(exceedance)
    count = (exceedance > 0).sum()
    try:
        t1 = np.where(exceedance > 0)[0][0]
        t2 = np.argmax(exceedance)
    except IndexError:
        t1, t2 = np.nan, np.nan
    return max_violation, count, t1, t2
