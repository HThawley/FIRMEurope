# type: ignore
from datetime import datetime
import os
from warnings import warn
import numpy as np

from firm_ce.analysis.accessor import Accessor
from firm_ce.common.constants import VALIDATION_TOL
from firm_ce.scalar.solution import Solution_InstanceType, evaluate
from firm_ce.system.scenario import Scenario
from firm_ce.tensor.solution import EvaluateTensor, SolutionTensorType

TENSOR_STOR_NAMES = {0: "olphes", 1: "clphes_new_phes", 2: "bess4h", 3: "bess2h"}
TENSOR_HYD_NAMES = {0: "pondage", 1: "reservoir"}
TENSOR_PEAK_NAMES = {0: "biomass", 1: "biogas", 2: "ocgt"}


class ValidationWarning(UserWarning):
    pass


def get_exceedance_stats(exceedance: np.ndarray):
    max_violation = np.max(exceedance)
    count = (exceedance > 0).sum()
    if count == 0:
        return max_violation, count, np.nan, np.nan

    t1 = np.unravel_index(np.argmax(exceedance > 0), exceedance.shape)
    t2 = np.unravel_index(np.argmax(exceedance), exceedance.shape)
    if len(t1) == 1:
        t1, t2 = t1[0], t2[0]
    return max_violation, count, t1, t2


class _BaseValidation:
    report_title: str = "VALIDATION REPORT"
    default_filename: str = "validation_report.txt"

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.verbose = True
        self.logs: dict[str, list[str]] = {}
        self._current_category = "General"

    def _get_checks(self) -> dict:
        raise NotImplementedError

    def validate(self, verbose: bool = True) -> bool:
        self.verbose = verbose
        self.logs.clear()

        for check_name, check_func in self._get_checks().items():
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
        self.logs[self._current_category].append(msg)
        if self.verbose:
            warn(f"[{self._current_category}] {msg}", ValidationWarning)

    def _check_1d_trace(self, trace_1d: np.ndarray, capacity: float, label: str) -> bool:
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

    def dump_logs(self, filename: str | None = None) -> None:
        filepath = os.path.join(self.results_dir, filename or self.default_filename)
        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f" {self.report_title}\n")
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


class Validation(_BaseValidation):
    report_title = "VALIDATION REPORT"
    default_filename = "validation_report.txt"

    def __init__(self, solution: Solution_InstanceType, results_dir: str):
        super().__init__(results_dir)
        self.solution = solution
        if not getattr(self.solution, "evaluated", False):
            evaluate(solution)
        self.accessor = Accessor(self.solution, 1.0)

    def _get_checks(self) -> dict:
        return {
            "Decision Variables Bounds": self.check_build_bounds,
            "Generator Limits": self.check_generator_limits,
            "Transmission Limits": self.check_transmission_limits,
            "Energy Balance & Flows": self.check_energy_balance_and_flows,
            "Storage Constraints": self.check_storage_limits,
            "Storage Accrual": self.check_storage_accrual,
            "Fuel Limits": self.check_fuel_limits,
        }

    def check_build_bounds(self) -> bool:
        passed = True
        asset_groups = [
            self.solution.fleet.generators,
            self.solution.fleet.storages,
            self.solution.network.major_lines,
        ]
        for group in asset_groups:
            for asset in group.values():
                _, new_p, min_p, max_p = self.accessor.get_build_power(asset)
                if new_p < min_p - VALIDATION_TOL or new_p > max_p + VALIDATION_TOL:
                    self._log(
                        f"Bounds Violation: {asset.name} power new_build ({new_p:.3f}) outside [{min_p}, {max_p}]."
                    )
                    passed = False

                if asset.object_class == "storage":
                    _, new_e, min_e, max_e = self.accessor.get_build_energy(asset)
                    if new_e < min_e - VALIDATION_TOL or new_e > max_e + VALIDATION_TOL:
                        self._log(
                            f"Bounds Violation: {asset.name} energy new_build ({new_e:.3f}) outside [{min_e}, {max_e}]."
                        )
                        passed = False
        return passed

    def check_generator_limits(self) -> bool:
        passed = True
        for asset in self.solution.fleet.generators.values():
            passed &= self._check_1d_trace(
                self.accessor.get_power_trace(asset),
                self.accessor.get_power_capacity(asset),
                f"Generator Violation: {asset.name} dispatch",
            )
        return passed

    def check_transmission_limits(self) -> bool:
        passed = True
        for asset in self.solution.network.major_lines.values():
            capacity = self.accessor.get_power_capacity(asset)
            flows = np.abs(self.accessor.get_power_trace(asset))
            if np.any(flows > capacity + VALIDATION_TOL):
                exc = np.maximum(flows - capacity - VALIDATION_TOL, 0)
                max_v, count, t1, t2 = get_exceedance_stats(exc)
                self._log(
                    f"Transmission Violation: {asset.name} flow magnitude exceeds capacity by up to {max_v:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                )
                passed = False
        return passed

    def check_energy_balance_and_flows(self) -> bool:
        passed = True
        nodes = self.solution.network.nodes
        node_balances = {n.id: np.zeros_like(n.data) for n in nodes.values()}

        for node in nodes.values():
            node_balances[node.id] -= self.accessor.get_demand_trace(node)
            node_balances[node.id] += np.abs(self.accessor.get_deficit_trace(node))
            node_balances[node.id] -= np.abs(self.accessor.get_curtail_trace(node))

        for gen in self.solution.fleet.generators.values():
            node_balances[gen.node.id] += self.accessor.get_power_trace(gen)

        for stor in self.solution.fleet.storages.values():
            node_balances[stor.node.id] += self.accessor.get_power_trace(stor)

        for line in self.solution.network.major_lines.values():
            f = self.accessor.get_transmission_trace(line)
            eff = self.accessor.get_efficiency(line)
            f_pos = np.where(f > 0, f, 0.0)
            f_neg = np.where(f < 0, -f, 0.0)

            node_balances[line.node_start.id] += f_neg * eff - f_pos
            node_balances[line.node_end.id] += f_pos * eff - f_neg

        for node in nodes.values():
            exceedance = np.maximum(np.abs(node_balances[node.id]) - VALIDATION_TOL, 0)
            if np.any(exceedance > 0):
                max_violation, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Energy Balance Violation: Node {node.name} failed by up to {max_violation:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                )
                passed = False
        return passed

    def check_storage_limits(self) -> bool:
        passed = True
        for asset in self.solution.fleet.storages.values():
            passed &= self._check_1d_trace(
                self.accessor.get_storage_level_trace(asset),
                self.accessor.get_energy_capacity(asset),
                f"Storage Violation: {asset.name} energy",
            )
        return passed

    def check_storage_accrual(self) -> bool:
        passed = True
        res = self.solution.static.resolution

        for asset in self.solution.fleet.storages.values():
            se = self.accessor.get_storage_level_trace(asset)
            cap = self.accessor.get_energy_capacity(asset)
            charge = np.abs(self.accessor.get_charge_trace(asset))
            discharge = np.abs(self.accessor.get_discharge_trace(asset))
            eff_c = self.accessor.get_charge_efficiency(asset)
            eff_d = self.accessor.get_discharge_efficiency(asset)
            inflows = self.accessor.get_inflow_trace(asset) * res if self.accessor.has_inflows(asset) else 0.0

            expected_delta = (charge * eff_c - discharge / eff_d) * res + inflows
            expected_next_se = np.clip(
                se[:-1] + (expected_delta[1:] if np.ndim(expected_delta) > 0 else 0.0), 0.0, cap
            )

            mismatch = np.abs(se[1:] - expected_next_se)
            exceedance = np.maximum(mismatch - VALIDATION_TOL, 0)
            if np.any(exceedance > 0):
                max_v, count, t1, t2 = get_exceedance_stats(exceedance)
                self._log(
                    f"Storage Accrual Violation: {asset.name} mismatched by up to {max_v:.4f}. "
                    f"Found: {count} exceedances. First at t={t1 + 1}. Largest at t={t2 + 1}"
                )
                passed = False
        return passed

    def check_fuel_limits(self) -> bool:
        passed = True
        res = self.solution.static.resolution
        fuel_gens = {fuel.id: [] for fuel in self.solution.fleet.fuels.values()}
        for gen in self.solution.fleet.generators.values():
            fuel_gens[gen.fuel.id].append(gen)

        for fuel in self.solution.fleet.fuels.values():
            rem_energy = self.accessor.get_remaining_energy_trace(fuel)
            if len(rem_energy) == 0 or np.isinf(rem_energy[0]):
                continue

            if np.any(rem_energy < -VALIDATION_TOL):
                exc = np.maximum(-rem_energy - VALIDATION_TOL, 0)
                max_v, count, t1, t2 = get_exceedance_stats(exc)
                self._log(
                    f"Fuel Limit Violation: {fuel.name} remaining energy dropped below 0 by up to {max_v:.4f}. "
                    f"Found: {count} exceedances. First at t={t1}. Largest at t={t2}"
                )
                passed = False

            gens = fuel_gens.get(fuel.id, [])
            if not gens:
                continue

            total_dispatch_power = sum(
                (self.accessor.get_power_trace(g) for g in gens), start=np.zeros_like(rem_energy)
            )
            expected_change = -total_dispatch_power * res
            actual_change = rem_energy[1:] - rem_energy[:-1]
            exc = np.maximum(np.abs(actual_change - expected_change[1:]) - VALIDATION_TOL, 0)
            if np.any(exc > 0):
                max_v, count, t1, t2 = get_exceedance_stats(exc)
                self._log(
                    f"Fuel Depletion Violation: {fuel.name} mismatched dispatch by up to {max_v:.4f}. "
                    f"Found: {count} exceedances. First at t={t1 + 1}. Largest at t={t2 + 1}"
                )
                passed = False
        return passed


class ValidationTensor(_BaseValidation):
    report_title = "TENSOR VALIDATION REPORT"
    default_filename = "tensor_validation_report.txt"

    def __init__(self, solution: SolutionTensorType, results_dir: str, scenario: Scenario):
        super().__init__(results_dir)
        self.solution = solution
        self.scenario = scenario
        if not getattr(self.solution, "evaluated", False):
            EvaluateTensor(solution)

    def _get_checks(self) -> dict:
        return {
            "Decision Variables Bounds": self.check_build_bounds,
            "Generator Limits": self.check_dispatch_limits,
            "Transmission Limits": self.check_transmission_limits,
            "Energy Balance & Flows": self.check_energy_balance_and_flows,
            "Storage Constraints": self.check_storage_limits,
            "Storage Accrual": self.check_storage_accrual,
            "Fuel Limits": self.check_fuel_limits,
        }

    def check_build_bounds(self) -> bool:
        passed = True
        x = (
            self.scenario.convert_x_to_abs(self.solution.x)
            if self.scenario.config.parameterisation == "relative"
            else self.solution.x
        )
        lb, ub = self.scenario.lower_bounds_abs, self.scenario.upper_bounds_abs

        lb_exc = np.maximum(lb - x - VALIDATION_TOL, 0)
        if np.any(lb_exc > 0):
            max_v, count, _, idx2 = get_exceedance_stats(lb_exc)
            self._log(
                f"Bounds Violation: x-vector contains {count} values below lower bound "
                f"by up to {max_v:.4f}. Largest violation at index {idx2}."
            )
            passed = False

        ub_exc = np.maximum(x - ub - VALIDATION_TOL, 0)
        if np.any(ub_exc > 0):
            max_v, count, _, idx2 = get_exceedance_stats(ub_exc)
            self._log(
                f"Bounds Violation: x-vector contains {count} values above upper bound "
                f"by up to {max_v:.4f}. Largest violation at index {idx2}."
            )
            passed = False
        return passed

    def check_dispatch_limits(self) -> bool:
        passed = True
        o, a, static = self.solution.operations, self.solution.assets, self.solution.static
        nodes, nodel = static.nodes, self.scenario.Nodel

        traces_2d = {
            "pfix": (o.Mpfix, a.Cpfix),
            "psat": (o.Mpsat, a.Cpsat),
            "offw": (o.Moffw, a.Coffw),
            "onsw": (o.Monsw, a.Consw),
            "nuke": (o.Mnuke, a.Cnuke),
        }
        for name, (trace, capacity) in traces_2d.items():
            for n in range(nodes):
                passed &= self._check_1d_trace(trace[:, n], capacity[n], f"Generator ({name}) at {nodel[n]}")

        for s in range(static.nstor):
            sname = TENSOR_STOR_NAMES.get(s, f"stor_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mdischarge[:, n, s], a.CstorageP[n, s], f"Storage discharge ({sname}) at {nodel[n]}"
                )
                passed &= self._check_1d_trace(
                    o.Mcharge[:, n, s], a.CstorageP[n, s], f"Storage charge ({sname}) at {nodel[n]}"
                )

        for s in range(static.nhyd):
            hname = TENSOR_HYD_NAMES.get(s, f"hyd_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(o.Mhydro[:, n, s], a.ChydP[n, s], f"Hydro ({hname}) at {nodel[n]}")

        for s in range(static.npeak):
            pname = TENSOR_PEAK_NAMES.get(s, f"peak_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(o.Mpeak[:, n, s], a.Cpeak[n, s], f"Generator ({pname}) at {nodel[n]}")

        return passed

    def check_transmission_limits(self) -> bool:
        passed = True
        o, a, s = self.solution.operations, self.solution.assets, self.solution.static
        nodel = self.scenario.Nodel

        for i in range(s.nhvi):
            start, end = s.network[i, 0], s.network[i, 1]
            abs_flow = np.abs(o.Tnetflow[:, i])
            if np.any(abs_flow > a.Clines[i] + VALIDATION_TOL):
                exc = np.maximum(abs_flow - a.Clines[i] - VALIDATION_TOL, 0)
                max_v, count, t1, t2 = get_exceedance_stats(exc)
                direction = "→" if o.Tnetflow[t2, i] > 0 else "←"
                self._log(
                    f"Line {nodel[start]} {direction} {nodel[end]} net flow exceeds capacity by up to {max_v:.4f}. "
                    f"Found: {count}. First at t={t1}. Largest at t={t2}."
                )
                passed = False
        return passed

    def check_energy_balance_and_flows(self) -> bool:
        passed = True
        s, o = self.solution.static, self.solution.operations
        nodel = self.scenario.Nodel

        total_gen = (
            o.Mpfix
            + o.Mpsat
            + o.Moffw
            + o.Monsw
            + o.Mnuke
            + np.sum(o.Mpeak, axis=2)
            + np.sum(o.Mdischarge, axis=2)
            + np.sum(o.Mhydro, axis=2)
            + s.Mror
        )
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
        passed = True
        o, a, static = self.solution.operations, self.solution.assets, self.solution.static
        nodes, nodel = static.nodes, self.scenario.Nodel

        for s in range(static.nstor):
            sname = TENSOR_STOR_NAMES.get(s, f"stor_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mstorage[:, n, s], a.CstorageE[n, s], f"Storage SOC ({sname}) at {nodel[n]}"
                )

        for s in range(static.nhyd):
            hname = TENSOR_HYD_NAMES.get(s, f"hyd_{s}")
            for n in range(nodes):
                passed &= self._check_1d_trace(
                    o.Mreservoir[:, n, s], a.ChydE[n, s], f"Hydro SOC ({hname}) at {nodel[n]}"
                )
        return passed

    def check_storage_accrual(self) -> bool:
        passed = True
        s, o, a = self.solution.static, self.solution.operations, self.solution.assets
        nodel = self.scenario.Nodel

        expected_delta = (o.Mcharge[1:] * s.storage_charge_eff) - (o.Mdischarge[1:] / s.storage_discha_eff)
        phes_inflows = np.zeros_like(expected_delta)
        phes_inflows[:, :, 0] = s.TSphes_inflow[1:]

        expected_energy = np.clip(
            o.Mstorage[:-1] + (expected_delta * s.resolution) + phes_inflows,
            0.0,
            np.expand_dims(a.CstorageE, axis=0),
        )

        for stor_idx in range(s.nstor):
            sname = TENSOR_STOR_NAMES.get(stor_idx, f"stor_{stor_idx}")
            for n in range(s.nodes):
                exc = np.maximum(
                    np.abs(o.Mstorage[1:, n, stor_idx] - expected_energy[:, n, stor_idx]) - VALIDATION_TOL, 0
                )
                if np.any(exc > 0):
                    max_v, count, _, t2 = get_exceedance_stats(exc)
                    self._log(
                        f"Storage accrual ({sname}) at {nodel[n]} mismatched by up to {max_v:.4f}. "
                        f"Found: {count}. Largest at t={t2 + 1}."
                    )
                    passed = False
        return passed

    def check_fuel_limits(self) -> bool:
        passed = True
        o, s = self.solution.operations, self.solution.static

        if np.any(o.remaining_peak_budget < -VALIDATION_TOL):
            max_v, count, _, _ = get_exceedance_stats(np.maximum(-o.remaining_peak_budget - VALIDATION_TOL, 0))
            self._log(f"Fuel limits dropped below 0 by up to {max_v:.4f}. Found: {count}.")
            passed = False

        total_peak_consumption = np.sum(o.Mpeak, axis=(0, 1)) * s.resolution
        if np.any(total_peak_consumption[: s.npeak - 1] > np.sum(s.Bpeak, axis=0) + VALIDATION_TOL):
            exc = np.maximum(total_peak_consumption[: s.npeak - 1] - np.sum(s.Bpeak, axis=0) - VALIDATION_TOL, 0)
            max_v, count, _, t2 = get_exceedance_stats(exc)
            self._log(
                f"Total fuel budget depleted below minimum thresholds by up to {max_v:.4f}. "
                f"Found: {count}. Largest at t={t2}."
            )
            passed = False
        return passed
