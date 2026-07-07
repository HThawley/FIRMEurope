# type: ignore
import os
from datetime import datetime
import numpy as np
from warnings import warn

from firm_ce.analysis.accessor import Accessor
from firm_ce.common.constants import VALIDATION_TOL
from firm_ce.backend.scalar.solution import Solution_InstanceType, evaluate
from firm_ce.backend.tensor.solution import SolutionTensorType, EvaluateTensor
from firm_ce.system.scenario import Scenario


class ValidationWarning(UserWarning):
    pass


class Validation:
    def __init__(
        self,
        solution: Solution_InstanceType,
        results_dir: str,
    ):
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
                exceedance = np.maximum(np.abs(flows) - capacity - VALIDATION_TOL, 0)
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


def get_exceedance_stats(exceedance: np.ndarray):
    max_violation = np.max(exceedance)
    count = (exceedance > 0).sum()

    if count == 0:
        return max_violation, count, np.nan, np.nan

    flat_idx1 = np.argmax(exceedance > 0)
    flat_idx2 = np.argmax(exceedance)

    t1 = np.unravel_index(flat_idx1, exceedance.shape)
    t2 = np.unravel_index(flat_idx2, exceedance.shape)

    # Revert to scalar for 1D arrays to match legacy string formatting
    if len(t1) == 1:
        t1, t2 = t1[0], t2[0]

    return max_violation, count, t1, t2


class ValidationTensor:
    def __init__(
        self,
        solution: SolutionTensorType,
        results_dir: str,
        scenario: Scenario,
    ):
        self.solution = solution
        self.scenario = scenario

        if not getattr(self.solution, "evaluated", False):
            # Fallback to evaluate if it hasn't been run
            EvaluateTensor(solution)

        self.results_dir = results_dir
        self.verbose = True
        self.logs = {}
        self._current_category = "General"

    def validate(self, verbose: bool = True) -> bool:
        self.verbose = verbose
        self.logs.clear()

        checks = {
            "Decision Variables Bounds": self.check_build_bounds,
            "Generator Limits": self.check_dispatch_limits,
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

    def dump_logs(self, filename: str = "tensor_validation_report.txt") -> None:
        """Dumps formatted logs to a human-readable text file."""
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(" TENSOR VALIDATION REPORT\n")
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
        """Check that tensor decision variables are strictly non-negative."""
        passed = True

        x = self.solution.x
        if self.scenario.config.parameterisation == "relative":
            x = self.scenario.convert_x_to_abs(x)

        lb = self.scenario.lower_bounds_abs
        ub = self.scenario.upper_bounds_abs

        lb_exceedance = np.maximum(lb - x - VALIDATION_TOL, 0)
        if np.any(lb_exceedance > 0):
            max_violation, count, idx1, idx2 = get_exceedance_stats(lb_exceedance)
            self._log(
                f"Bounds Violation: x-vector contains {count} values below lower bound "
                f"by up to {max_violation:.4f}. Largest violation at index {idx2}."
            )
            passed = False

        ub_exceedance = np.maximum(x - ub - VALIDATION_TOL, 0)
        if np.any(ub_exceedance > 0):
            max_violation, count, idx1, idx2 = get_exceedance_stats(ub_exceedance)
            self._log(
                f"Bounds Violation: x-vector contains {count} values above upper bound "
                f"by up to {max_violation:.4f}. Largest violation at index {idx2}."
            )
            passed = False
        return passed

    def _check_1d_trace(self, trace_1d: np.ndarray, capacity: float, label: str) -> bool:
        """Check a 1D nodal trace is within [0, capacity]. Returns False if any violation found."""
        passed = True
        if np.any(trace_1d < -VALIDATION_TOL):
            exc = np.maximum(-trace_1d - VALIDATION_TOL, 0)
            max_v, count, t1, t2 = get_exceedance_stats(exc)
            self._log(
                f"{label} drops below 0 by up to {max_v:.4f}. "
                f"Found: {count}. First at t={t1}. Largest at t={t2}."
            )
            passed = False
        if np.any(trace_1d > capacity + VALIDATION_TOL):
            exc = np.maximum(trace_1d - capacity - VALIDATION_TOL, 0)
            max_v, count, t1, t2 = get_exceedance_stats(exc)
            self._log(
                f"{label} exceeds capacity by up to {max_v:.4f}. "
                f"Found: {count}. First at t={t1}. Largest at t={t2}."
            )
            passed = False
        return passed

    def check_dispatch_limits(self) -> bool:
        """Check that dispatch values fall between 0 and installed capacity."""
        passed = True
        o = self.solution.operations
        a = self.solution.assets
        static = self.solution.static
        nodes = static.nodes
        nodel = self.scenario.Nodel

        traces_2d = {
            "pfix": (o.Mpfix, a.Cpfix),
            "psat": (o.Mpsat, a.Cpsat),
            "offw": (o.Moffw, a.Coffw),
            "onsw": (o.Monsw, a.Consw),
            "nuke": (o.Mnuke, a.Cnuke),
        }
        for name, (trace, capacity) in traces_2d.items():
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    trace[:, n], capacity[n], f"Generator ({name}) at {nodel[n]}"
                )

        # Storage energy: Mdis/charge (intervals, nodes, nstor) vs CstorageP (nodes, nstor)
        stor_names = {0: "phes", 1: "bess4h", 2: "bess2h"}
        for s in range(static.nstor):
            sname = stor_names.get(s, f"stor_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mdischarge[:, n, s], a.CstorageP[n, s], f"Storage discharge ({sname}) at {nodel[n]}"
                )

                passed &= self._check_1d_trace(
                    o.Mcharge[:, n, s], a.CstorageP[n, s], f"Storage charge ({sname}) at {nodel[n]}"
                )

        # Hydro power dispatch: Mhydro (intervals, nodes, nhyd) vs ChydP (nodes, nhyd)
        hyd_names = {0: "pondage", 1: "reservoir"}
        for s in range(static.nhyd):
            hname = hyd_names.get(s, f"hyd_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mhydro[:, n, s], a.ChydP[n, s], f"Hydro ({hname}) at {nodel[n]}"
                )

        # Peak dispatch: Mpeak (intervals, nodes, npeak) vs Cpeak (nodes, npeak)
        # Ordering confirmed from StaticTensor: 0=biomass, 1=biogas, 2=ccgt
        peak_names = {0: "biomass", 1: "biogas", 2: "ccgt"}
        for s in range(static.npeak):
            pname = peak_names.get(s, f"peak_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mpeak[:, n, s], a.Cpeak[n, s], f"Generator ({pname}) at {nodel[n]}"
                )

        return passed

    def check_transmission_limits(self) -> bool:
        """Check that net line flows stay within installed capacity in both directions.

        Tnetflow is signed (positive = start→end, negative = end→start). Interconnection
        enforces cap_fwd = Clines - netflow ≥ 0 and cap_rev = Clines + netflow ≥ 0 at
        every step, so |Tnetflow| ≤ Clines is a structural invariant. Violations here
        indicate likely indicate a bug in flow accounting rather than a capacity breach
        during dispatch.
        """
        passed = True
        o = self.solution.operations
        a = self.solution.assets
        s = self.solution.static
        nodel = self.scenario.Nodel

        for i in range(s.nhvi):
            start = s.network[i, 0]
            end = s.network[i, 1]
            abs_flow = np.abs(o.Tnetflow[:, i])
            if np.any(abs_flow > a.Clines[i] + VALIDATION_TOL):
                exc = np.maximum(abs_flow - a.Clines[i] - VALIDATION_TOL, 0)
                max_v, count, t1, t2 = get_exceedance_stats(exc)
                direction = "→" if o.Tnetflow[t2, i] > 0 else "←"
                label = f"Line {nodel[start]} {direction} {nodel[end]}"
                self._log(
                    f"{label} net flow exceeds capacity by up to {max_v:.4f}. "
                    f"Found: {count}. First at t={t1}. Largest at t={t2}."
                )
                passed = False

        return passed

    def check_energy_balance_and_flows(self) -> bool:
        """Verify nodal energy balance across all intervals."""
        passed = True
        s = self.solution.static
        o = self.solution.operations
        nodel = self.scenario.Nodel

        # Nodal energy balance: Gen + NetImports = Load + Curtailment - Deficit
        total_gen = (o.Mpfix + o.Mpsat + o.Moffw + o.Monsw + o.Mnuke
                     + np.sum(o.Mpeak, axis=2) + np.sum(o.Mdischarge, axis=2)
                     + np.sum(o.Mhydro, axis=2) + s.Mror)

        total_load = s.Mload + np.sum(o.Mcharge, axis=2)

        nodal_balance = total_gen + o.Mimport - total_load
        expected_balance = o.Mcurtail - o.Mdeficit

        for n in range(s.nodes):
            exc = np.maximum(np.abs(nodal_balance[:, n] - expected_balance[:, n]) - VALIDATION_TOL, 0)
            if np.any(exc > 0):
                max_v, count, t1, t2 = get_exceedance_stats(exc)
                self._log(
                    f"Energy balance at {nodel[n]} violated by up to {max_v:.4f}. "
                    f"Found: {count}. First at t={t1}. Largest at t={t2}."
                )
                passed = False

        return passed

    def check_storage_limits(self) -> bool:
        """Ensure stored energy strictly obeys bounding constraints."""
        passed = True
        o = self.solution.operations
        a = self.solution.assets
        static = self.solution.static
        nodes = static.nodes
        nodel = self.scenario.Nodel

        # Storage energy: Mstorage (intervals, nodes, nstor) vs CstorageE (nodes, nstor)
        # Ordering from StaticTensor: 0=phes, 1=bess4h, 2=bess2h
        stor_names = {0: "phes", 1: "bess4h", 2: "bess2h"}
        for s in range(static.nstor):
            sname = stor_names.get(s, f"stor_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mstorage[:, n, s], a.CstorageE[n, s], f"Storage SOC ({sname}) at {nodel[n]}"
                )

        # Hydro power dispatch: Mreservoir (intervals, nodes, nhyd) vs ChydE (nodes, nhyd)
        hyd_names = {0: "pondage", 1: "reservoir"}
        for s in range(static.nhyd):
            hname = hyd_names.get(s, f"hyd_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mreservoir[:, n, s], a.ChydE[n, s], f"Hydro SOC ({hname}) at {nodel[n]}"
                )

        return passed

    def check_storage_accrual(self) -> bool:
        """Verify state-of-charge transitions obey charge/discharge efficiencies and inflows.

        Checks transitions from t=1 onward; the t=0 state (Mstorage_init → Mstorage[0])
        is not covered here. PHES inflow is combined with delta before clipping, which
        differs from GetForwardStorageHeadroom where inflow is clipped independently first —
        this can produce a false mismatch when the PHES reservoir is near-full and inflow
        alone would overflow. If that edge case fires frequently, align the clip order with
        UpdateSOCt.
        """
        passed = True
        s = self.solution.static
        o = self.solution.operations
        a = self.solution.assets
        nodel = self.scenario.Nodel

        # Compute full expected tensor first, then report per (storage type, node)
        # eff_c, eff_d: (nstor,) — broadcast over (intervals-1, nodes, nstor)
        eff_c = s.storage_charge_eff
        eff_d = s.storage_discha_eff

        expected_delta = (o.Mcharge[1:] * eff_c) - (o.Mdischarge[1:] / eff_d)

        # PHES natural inflows injected at nstor == 0 (confirmed index from StaticTensor)
        phes_inflows = np.zeros_like(expected_delta)
        phes_inflows[:, :, 0] = s.TSphes_inflow[1:]

        expected_energy = o.Mstorage[:-1] + (expected_delta * s.resolution) + phes_inflows
        Cstor_expanded = np.expand_dims(a.CstorageE, axis=0)
        expected_energy = np.clip(expected_energy, 0.0, Cstor_expanded)

        stor_names = {0: "phes", 1: "bess4h", 2: "bess2h"}
        for stor_idx in range(s.nstor):
            sname = stor_names.get(stor_idx, f"stor_{stor_idx}")
            for n in range(s.nodes):
                mismatch = np.abs(o.Mstorage[1:, n, stor_idx] - expected_energy[:, n, stor_idx])
                exc = np.maximum(mismatch - VALIDATION_TOL, 0)
                if np.any(exc > 0):
                    max_v, count, t1, t2 = get_exceedance_stats(exc)
                    # t indices are offset by 1 because the difference array starts at t=1
                    self._log(
                        f"Storage accrual ({sname}) at {nodel[n]} mismatched by up to {max_v:.4f}. "
                        f"Found: {count}. Largest at t={t2 + 1}."
                    )
                    passed = False

        return passed

    def check_fuel_limits(self) -> bool:
        """Validate energy budgets mapped to peak generators."""
        passed = True
        o = self.solution.operations
        s = self.solution.static

        if np.any(o.remaining_peak_budget < -VALIDATION_TOL):
            max_v, count, t1, t2 = get_exceedance_stats(np.maximum(-o.remaining_peak_budget - VALIDATION_TOL, 0))
            self._log(f"Fuel limits dropped below 0 by up to {max_v:.4f}. Found: {count}.")
            passed = False

        # Budget depletion check relies on checking year-end totals against static.Bpeak.
        # (Assuming intervals map to single/multiple years based on static parameters)
        total_peak_consumption = np.sum(o.Mpeak, axis=(0, 1)) * s.resolution

        # Only check the bounds that actually have budget data mapped inside static (npeak - 1 dimension)
        # If total consumption across the model violates the total Bpeak bounds
        if np.any(total_peak_consumption[:s.npeak-1] > np.sum(s.Bpeak, axis=0) + VALIDATION_TOL):
            exceedance = np.maximum(total_peak_consumption[:s.npeak-1] - np.sum(s.Bpeak, axis=0) - VALIDATION_TOL, 0)
            max_v, count, t1, t2 = get_exceedance_stats(exceedance)
            self._log(
                f"Total fuel budget depleted below minimum thresholds by up to {max_v:.4f}."
                f"Found: {count}. Largest at t={t2}."
            )
            passed = False

        return passed
