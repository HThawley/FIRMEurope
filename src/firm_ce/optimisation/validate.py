# type: ignore
import os

import pandas as pd
from typing import Tuple
from warnings import warn

from firm_ce.system.scenario import Scenario
from firm_ce.io.file_manager import ResultFile
from firm_ce.common.constants import TOLERANCE


class ValidationWarning(UserWarning):
    pass


class Validation:
    def __init__(
        self,
        scenario: Scenario,
        statistics_attr: str = "statistics"
    ):
        if not hasattr(scenario, statistics_attr):
            raise RuntimeError("Run statistics before validating!")
        self.statistics = getattr(scenario, statistics_attr)

        if not self.statistics.statistics_generated:
            raise RuntimeError("Run statistics before validating!")

        self.solution = self.statistics.solution

        self.validation_directory = self.create_validation_directory(
            self.statistics.results_directory
        )

        self.full_intervals_count = self.solution.static.block_lengths.sum()

        self.validation_generated = False

    def create_validation_directory(self, statistics_directory: str) -> str:
        validation_dir = os.path.join(statistics_directory, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        return validation_dir

    def validate(self) -> None:
        if not self.solution.evaluated:
            raise RuntimeError("Solution must be evaluated before validation")

        self.result_files = {
            "energy_balance_ASSETS": self.validate_energy_balance("assets"),
            "energy_balance_NODES": self.validate_energy_balance("nodes"),
            "energy_balance_NETWORK": self.validate_energy_balance("network"),
            "capacities": self.validate_capacities_internal(),
            "operational_capacities": self.validate_capacities_operational(),
        }

        self.validation_generated = True

        return None

    def write_results(self) -> None:
        if not self.validation_generated:
            raise RuntimeError("Validation must be generated before writing results")

        for result_file in self.result_files.values():
            result_file.write()

        return None

    def validate_energy_balance(self, aggregation_type: str) -> ResultFile:
        def get_import_columns(balance: pd.DataFrame, node: str) -> pd.MultiIndex:
            major_lines = balance.columns[balance.columns.get_level_values("Asset Type") == "Major Line"]
            imports = major_lines[major_lines.get_level_values("Asset Name").str.split("-").str.get(-1) == node]
            return imports

        def get_export_columns(balance: pd.DataFrame, node: str) -> pd.MultiIndex:
            major_lines = balance.columns[balance.columns.get_level_values("Asset Type") == "Major Line"]
            exports = major_lines[major_lines.get_level_values("Asset Name").str.split("-").str.get(0) == node]
            return exports

        def get_dispatch_columns(balance: pd.DataFrame, node: str) -> pd.MultiIndex:
            cols = balance.columns[
                balance.columns.get_level_values("Asset Name").str.contains(node)
                & (
                    balance.columns.get_level_values("Column Name").str.contains("Dispatch")
                    | balance.columns.get_level_values("Column Name").isin(("Solar", "Wind", "Run-of-river", "Ror", "Baseload"))
                )
            ]
            return cols

        def compute_energy_delta(df: pd.DataFrame, balance: pd.DataFrame, node: str) -> pd.DataFrame:
            df[node] -= balance[node, "Node", "Demand"]
            df[node] += balance[node, "Node", "Spillage"]  # spillage is negative-valued
            df[node] += balance[node, "Node", "Deficit"]
            return df

        match aggregation_type:
            case "assets" | "nodes":
                balance = pd.read_csv(
                    fr"{self.statistics.results_directory}/energy_balance_{aggregation_type.upper()}.csv",
                    header=[0, 1, 2],
                    skiprows=[2, 4],
                    index_col=0,
                )

                nodes = [col[0] for col in balance.columns if col[2] == "Demand"]
                df = pd.DataFrame(0, index=pd.RangeIndex(self.full_intervals_count), columns=nodes)

                for n in nodes:
                    compute_energy_delta(df, balance, n)
                    cols = get_dispatch_columns(balance, n)
                    df[n] += balance[cols].sum(axis=1)

                    exports = get_export_columns(balance, n)
                    imports = get_import_columns(balance, n)
                    df[n] += balance[imports].sum(axis=1)
                    df[n] -= balance[exports].sum(axis=1)

                check_pass = (df.abs() <= TOLERANCE).all()

                for check, node in zip(check_pass, nodes):
                    if not check:
                        warn(f"Warning: Node '{node}' failed check: 'energy balance'.", ValidationWarning)

                total_row = pd.DataFrame(df.abs().sum(axis=0).to_list(), columns=["Total"], index=nodes).T
                df = pd.concat((total_row, df))

            case "network":
                balance = pd.read_csv(
                    fr"{self.statistics.results_directory}/energy_balance_{aggregation_type.upper()}.csv",
                    header=[0],
                    skiprows=[0, 1, 2, 4],
                    index_col=0,
                )

                df = pd.DataFrame(0, index=pd.RangeIndex(self.full_intervals_count), columns=["Network"])
                df["Network"] -= balance["Demand"]
                df["Network"] += balance["Spillage"]  # spillage is negative-valued
                df["Network"] += balance["Deficit"]
                cols = balance.columns[
                    balance.columns.str.contains("Dispatch")
                    | balance.columns.isin(("Solar", "Wind", "Run-of-river", "Ror", "Baseload"))
                ]
                df["Network"] += balance[cols].sum(axis=1)

                check = (df.abs() <= TOLERANCE).all().all()
                if not check:
                    warn("Warning: 'Network' failed check: 'energy balance'.", ValidationWarning)

                total_row = pd.DataFrame(df.abs().sum(axis=0).to_list(), columns=["Total"], index=["Network"]).T
                df = pd.concat((total_row, df))

        result_file = ResultFile(f"energy_balance_{aggregation_type.upper()}", self.validation_directory, df)

        return result_file

    def validate_capacities_internal(self):
        capacities = pd.read_csv(
            fr"{self.statistics.results_directory}/capacities.csv",
            header=[0, 1, 2, 3],
            skiprows=[4],
            index_col=0,
        )

        def append_check(df: pd.DataFrame, item: Tuple, name: str, check: bool) -> pd.DataFrame:
            if not check:
                warn(f"Warning: Asset '{item[0]}' failed check: '{name}'.", ValidationWarning)
            res = pd.DataFrame([[*item, name, check]], columns=df.columns)
            df = pd.concat((df, res), axis=0, ignore_index=True)
            return df

        ids = capacities.columns.get_level_values("Asset ID").unique().astype(int)

        df = pd.DataFrame(columns=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Check", "Pass"])
        for item in capacities.columns:
            df = append_check(df, item, "Upper Build Limit", capacities.loc["New Build Capacity", item] - capacities.loc["Max Build", item] <= TOLERANCE)
            df = append_check(df, item, "Lower Build Limit", capacities.loc["New Build Capacity", item] - capacities.loc["Min Build", item] >= -TOLERANCE)

            if item[1] == "Minor Line":
                expected = 0
                for asset in self.solution.fleet.generators.values():
                    if asset.line.name == item[1] and asset.id in ids:
                        col = capacities.columns[(capacities.columns.get_level_values("Asset Type") == "Generator")
                                                 & (capacities.columns.get_level_values("Asset ID") == asset.id)
                                                 & (capacities.columns.get_level_values("Column Name") == "Power Capacity")]
                        expected += capacities.loc["Total Capacity", col]
                for asset in self.solution.fleet.reservoirs.values():
                    if asset.line.name == item[1] and asset.id in ids:
                        col = capacities.columns[(capacities.columns.get_level_values("Asset Type") == "Reservoir")
                                                 & (capacities.columns.get_level_values("Asset ID") == asset.id)
                                                 & (capacities.columns.get_level_values("Column Name") == "Power Capacity")]
                        expected += capacities.loc["Total Capacity", col]
                for asset in self.solution.fleet.storages.values():
                    if asset.line.name == item[1] and asset.id in ids:
                        col = capacities.columns[(capacities.columns.get_level_values("Asset Type") == "Storage")
                                                 & (capacities.columns.get_level_values("Asset ID") == asset.id)
                                                 & (capacities.columns.get_level_values("Column Name") == "Power Capacity")]
                        expected += capacities.loc["Total Capacity", col]
                df = append_check(df, item, "Minor Line Capacity", abs(expected - capacities.loc["Total Capacity", item]) <= TOLERANCE)

        result_file = ResultFile("val_capacities", self.validation_directory, df)

        return result_file

    def validate_capacities_operational(self):
        balance = pd.read_csv(
            fr"{self.statistics.results_directory}/energy_balance_ASSETS.csv",
            header=[0, 1, 2, 3],
            skiprows=[4],
            index_col=0,
        )
        nodes = balance.columns[balance.columns.get_level_values("Asset Type") == "Node"].get_level_values("Asset Name").unique().to_list()
        balance = balance[balance.columns[balance.columns.get_level_values("Asset Type") != "Node"]]
        capacities = pd.read_csv(
            fr"{self.statistics.results_directory}/capacities.csv",
            header=[0, 1, 2, 3],
            skiprows=[4],
            index_col=0,
        )
        capacities = capacities[capacities.columns[capacities.columns.get_level_values("Asset Type") != "Minor Line"]]

        def is_within_max(observed, theoretic):
            # theoretic >= observed
            # theoretic + TOLERANCE >= observed
            return abs(theoretic) - abs(observed) >= -TOLERANCE

        def is_within_min(observed, theoretic):
            # observed >= theoretic
            # observed >= theoretic - TOLERANCE
            return abs(observed) - abs(theoretic) >= -TOLERANCE

        def append_check(df: pd.DataFrame, item: Tuple, name: str, check: bool) -> pd.DataFrame:
            # TODO: add magnitude
            if not check:
                warn(f"Warning: Asset '{item[0]}' failed check: '{name}'.", ValidationWarning)
            res = pd.DataFrame([[*item, name, check]], columns=df.columns)
            df = pd.concat((df, res), axis=0, ignore_index=True)
            return df

        def match_column(balance: pd.DataFrame, match_term: str, asset_name: str, asset_type: str, asset_id: int, *args) -> Tuple:
            # *args is ignored, but allows us to * item in the for loop. purely for convenience.
            cols = balance.columns[
                (balance.columns.get_level_values("Asset Name") == asset_name)
                & (balance.columns.get_level_values("Asset Type") == asset_type)
                & (balance.columns.get_level_values("Asset ID") == asset_id)
                & balance.columns.get_level_values("Column Name").str.contains(match_term)
            ]
            assert len(cols) == 1
            return cols[0]

        def match_transm_column(balance: pd.DataFrame, item: Tuple) -> Tuple:
            return match_column(balance, "Flow", *item)

        def match_dispatch_column(balance: pd.DataFrame, item: Tuple) -> Tuple:
            return match_column(balance, "Dispatch", *item)

        def match_energy_column(balance: pd.DataFrame, item: Tuple) -> Tuple:
            return match_column(balance, "Energy", *item)

        def check_max_dispatch(df: pd.DataFrame, balance: pd.DataFrame, capacities: pd.DataFrame, item: Tuple) -> pd.DataFrame:
            observed_max = balance[match_dispatch_column(balance, item)].max() / 1000.  # MW to GW
            theoretic_max = capacities.loc["Total Capacity", item]
            df = append_check(df, item, "Max Dispatch", is_within_max(observed_max, theoretic_max))
            return df

        def check_min_dispatch(df: pd.DataFrame, balance: pd.DataFrame, capacities: pd.DataFrame, item: Tuple, zero: bool) -> pd.DataFrame:
            # TODO: fix
            observed_min = balance[match_dispatch_column(balance, item)].min() / 1000.  # MW to GW
            if zero:
                theoretic_min = 0
                df = append_check(df, item, "Min Dispatch", is_within_min(observed_min, theoretic_min))
            else:
                theoretic_min = - capacities.loc["Total Capacity", item]
                df = append_check(df, item, "Max Charge", is_within_min(observed_min, theoretic_min))
            return df

        def check_storage_limits(df: pd.DataFrame, balance: pd.DataFrame, capacities: pd.DataFrame, item: Tuple) -> pd.DataFrame:
            column = match_energy_column(balance, item)
            observed_max = balance[column].max() / 1000.  # MWh -> GWh
            theoretic_max = capacities.loc["Total Capacity", item]
            df = append_check(df, item, "Max Storage", is_within_max(observed_max, theoretic_max))
            observed_min = balance[column].min() / 1000.  # MWh -> GWh
            df = append_check(df, item, "Min Storage", is_within_min(observed_min, 0))
            return df

        def check_transm_limits(df: pd.DataFrame, balance: pd.DataFrame, capacties: pd.DataFrame, item: Tuple) -> pd.DataFrame:
            column = match_transm_column(balance, item)
            observed_max = balance[column].max() / 1000.  # MW -> GW
            theoretic_max = capacities.loc["Total Capacity", item]
            df = append_check(df, item, "Max Transm", is_within_max(observed_max, theoretic_max))
            observed_min = balance[column].min() / 1000.  # MW -> GW
            df = append_check(df, item, "Min Transm", is_within_min(observed_min, theoretic_max))
            return df

        df = pd.DataFrame(columns=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Check", "Pass"])
        for item in capacities.columns:
            asset_name, asset_type, asset_id, attribute = item
            match asset_type:
                case "Generator":
                    assert attribute == "Power Capacity"
                    df = check_max_dispatch(df, balance, capacities, item)
                    df = check_min_dispatch(df, balance, capacities, item, True)
                case "Reservoir":
                    match attribute:
                        case "Power Capacity":
                            df = check_max_dispatch(df, balance, capacities, item)
                            df = check_min_dispatch(df, balance, capacities, item, True)
                        case "Energy Capacity":
                            df = check_storage_limits(df, balance, capacities, item)
                case "Storage":
                    match attribute:
                        case "Power Capacity":
                            df = check_max_dispatch(df, balance, capacities, item)
                            df = check_min_dispatch(df, balance, capacities, item, False)
                        case "Energy Capacity":
                            df = check_storage_limits(df, balance, capacities, item)
                case "Major Line":
                    assert attribute == "Power Capacity"
                    check_transm_limits(df, balance, capacities, item)

        def get_import_columns(balance: pd.DataFrame, node: str) -> pd.MultiIndex:
            major_lines = balance.columns[balance.columns.get_level_values("Asset Type") == "Major Line"]
            imports = major_lines[major_lines.get_level_values("Asset Name").str.split("-").str.get(-1) == node]
            return imports

        def get_export_columns(balance: pd.DataFrame, node: str) -> pd.MultiIndex:
            major_lines = balance.columns[balance.columns.get_level_values("Asset Type") == "Major Line"]
            exports = major_lines[major_lines.get_level_values("Asset Name").str.split("-").str.get(0) == node]
            return exports

        def get_sign(x):
            if x > TOLERANCE:
                return 1
            elif x < -TOLERANCE:
                return -1
            return 0

        def is_consistent_sign(row: pd.Series):
            if row.isin((0, 1)).all():
                return True
            elif row.isin((0, -1)).all():
                return True
            return False

        for node in nodes:
            ports = pd.concat((
                balance[get_import_columns(balance, node)],
                -balance[get_export_columns(balance, node)]
            ), axis=1)
            ports = ports.map(get_sign)
            consistency = ports.apply(is_consistent_sign, axis=1)
            df = append_check(df, (f"{node=}", "Major Line", "-", "Flow"), "Simultaneous import/export", consistency.all())

        # TODO: inflows
        # TODO: charge/discharge -> storage level

        result_file = ResultFile("operational_capacities", self.validation_directory, df)

        return result_file
