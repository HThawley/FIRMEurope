import pandas as pd
import numpy as np
import os
from numba import njit

DISCOUNT_RATE = 0.07
LINES_DISCOUNT_RATE = 0.05
nodes = pd.read_csv(r"inputs/config/nodes.csv").iloc[:, 1].to_list()
startyear = None
endyear = None

suffix_dict = {
    "hydro-reservoir-inflow": "_hydro",
    "hydro-ror": "_ror",
    "open-field-pv": "_pv",
    "rooftop-pv": "_roof",
    "wind-offshore": "_offw",
    "wind-onshore": "_onsw",
    "pumped_hydro": "_phes",
    "battery": "_bess",
}

tech_fuels = {
    "hydro-reservoir-inflow": "water",
    "hydro-ror": "water",
    "open-field-pv": "solar",
    "rooftop-pv": "solar",
    "wind-offshore": "wind",
    "wind-onshore": "wind",
}

tech_to_flex = {
    "hydro-reservoir-inflow": "hydro",
    "hydro-ror": "ror",
    "open-field-pv": "solar",
    "rooftop-pv": "solar",
    "wind-offshore": "wind",
    "wind-onshore": "wind",
}

tech_to_conn = {
    "hydro-reservoir-inflow": "other_connections",
    "hydro-ror": "other_connections",
    "open-field-pv": "pv_connection",
    "rooftop-pv": "pv_connection",
    "wind-offshore": "offw_connection",
    "wind-onshore": "onsw_connection",
    "ccgt": "other_connections",
    "biogas": "other_connections",
    "nuclear": "other_connections",
}

tech_to_cost_file_name = {
    "hydro-reservoir-inflow": "hydro_reservoir",
    "hydro-ror": "hydro_run_of_river",
    "open-field-pv": "open_field_pv",
    "rooftop-pv": "roof_mounted_pv",
    "wind-offshore": "wind_offshore",
    "wind-onshore": "wind_onshore_monopoly",
}


@njit
def generate_interval_no(interval, day):
    for i in range(1, len(interval)):
        if day[i] == day[i - 1]:
            interval[i] = interval[i - 1] + 1
        else:
            interval[i] = 1
    return interval


def read_cost_file(fname):
    df = pd.read_csv(f"unprocessed_inputs/techno_economic/{fname}.csv", index_col="techs").fillna(0)
    # next(iter(...)) to get first item in dictionary without knowing key
    return next(iter(df.astype(float).to_dict().values()))


def read_and_process_timeseries(filename, suffixes=True, write=True):
    df = pd.read_csv(filename)
    timecol = df.columns[0]
    df[timecol] = pd.to_datetime(df[timecol])
    df["Year"] = df[timecol].dt.year
    df["Month"] = df[timecol].dt.month
    df["Day"] = df[timecol].dt.day
    df["Interval"] = 1
    df["Interval"] = generate_interval_no(df["Interval"].to_numpy(), df["Day"].to_numpy())

    if suffixes:
        for k, v in suffix_dict.items():
            if k in filename:
                suffix = v
                new_fname = k

        ts_cols = {
            col: col + suffix
            for col in df.columns
            if col
            not in (
                timecol,
                "Year",
                "Month",
                "Day",
                "Interval",
            )
        }
        df = df.rename(columns=ts_cols)

        columns = ["Year", "Month", "Day", "Interval"] + list(ts_cols.values())
    else:
        new_fname = filename.split("/")[-1].removesuffix(".csv")
        columns = ["Year", "Month", "Day", "Interval"] + list(df.columns[1:-4])

    df = df[columns]
    global startyear, endyear
    if startyear is None:
        startyear = df["Year"].min()
    else:
        startyear = min(startyear, df["Year"].min())
    if endyear is None:
        endyear = df["Year"].max()
    else:
        endyear = min(startyear, df["Year"].max())

    if write:
        df.to_csv(f"inputs/data/{new_fname}.csv", index=False)
    return df


def build_flexible_annual_biogas_csv(years="infer", write=True):

    if years == "infer":
        global startyear, endyear
        if startyear is None or endyear is None:
            raise RuntimeError("read and process timeseries first Or explicitly provide to this function")
    else:
        startyear, endyear = years

    resource = pd.read_csv(
        "unprocessed_inputs/existing_and_potential_capacity/biogas_yearly_resource.csv",
    )
    resource["locs"] += "_biogas"
    flexible_annual = pd.DataFrame(columns=resource["locs"], dtype=float)
    flexible_annual = flexible_annual.rename(columns={"locs": "Year"})

    resource.index = resource.locs
    resource = resource.drop(columns="locs").to_dict()["biogas_to_electricity_supply"]
    for year in range(startyear, endyear + 1):
        resource["Year"] = year
        flexible_annual = pd.concat((flexible_annual, pd.DataFrame(resource, index=[0]))).reset_index(drop=True)
    flexible_annual["Year"] = flexible_annual["Year"].astype(int)
    yearcolumn = flexible_annual.pop("Year")
    flexible_annual.insert(0, "Year", yearcolumn)

    if write:
        flexible_annual.to_csv("inputs/data/flexible_annual_biogas.csv", index=False)
    return flexible_annual


def build_generators_csv(write=True):
    columns = {
        "name": str,
        "fuel": str,
        "capex": float,
        "fom": float,
        "vom": float,
        "lifetime": int,
        "discount_rate": float,
        "heat_rate_base": float,
        "heat_rate_incr": float,
        "unit_size": float,
        "node": str,
        "max_build": float,
        "min_build": float,
        "initial_capacity": float,
        "line": str,
        "unit_type": str,
        "scenarios": str,
        "near_optimum": str,
        "range_group": bool,
    }
    generators = pd.DataFrame(columns=columns.keys()).astype(columns)

    cost_energy_cap = read_cost_file("cost_energy_cap")
    cost_om_annual = read_cost_file("cost_om_annual")
    cost_om_con = read_cost_file("cost_om_con")
    cost_om_prod = read_cost_file("cost_om_prod")
    lifetime = read_cost_file("lifetime")

    def build_re_row(name, node, tech, fuel, cost_name):
        row = {
            "name": name,
            "fuel": fuel,
            "capex": cost_energy_cap[cost_name],
            "fom": cost_om_annual[cost_name],
            "vom": cost_om_con[cost_name] + cost_om_prod[cost_name],
            "lifetime": lifetime[cost_name],
            "discount_rate": DISCOUNT_RATE,
            "heat_rate_base": 0,
            "heat_rate_incr": 0,
            "unit_size": 0.001,
            "node": node,
            "max_build": 0,
            "min_build": 0,
            "initial_capacity": 0,
            "line": tech_to_conn[tech],
            "unit_type": tech_to_flex[tech],
            "scenarios": "all",
            "near_optimum": False,
            "range_group": "",
        }
        return row

    excl_build_limits = pd.read_csv(
        "unprocessed_inputs/existing_and_potential_capacity/renewables_capacity_potential_exclusive.csv",
        index_col="locs",
    )
    comp_build_limits = pd.read_csv(
        "unprocessed_inputs/existing_and_potential_capacity/renewables_capacity_potential_competing.csv",
        index_col="locs",
    )

    for tech, fuel in tech_fuels.items():
        if "reservoir" in tech:
            continue
        elif "hydro" in tech:  # run-of-river
            initial_capacity = pd.read_csv(
                "unprocessed_inputs/existing_and_potential_capacity/hydro_capacity_current.csv",
                usecols=["locs", tech_to_cost_file_name[tech]],
                index_col="locs",
            ).to_dict()[tech_to_cost_file_name[tech]]
            min_build = 0
            max_build = 0
        else:
            initial_capacity = 0
            min_build = 0
            max_build = excl_build_limits.loc[excl_build_limits["techs"] == tech_to_cost_file_name[tech], "energy_cap_max"]

            if tech == "open-field-pv":
                col = [col for col in comp_build_limits.columns if "pv" in col.lower()][0]
                max_build = pd.merge(max_build, comp_build_limits.loc[:, col], on="locs", how="outer").fillna(0).sum(axis=1)
            elif tech == "wind-onshore":
                col = [col for col in comp_build_limits.columns if "wind" in col.lower()][0]
                max_build = pd.merge(max_build, comp_build_limits.loc[:, col], on="locs", how="outer").fillna(0).sum(axis=1)

        # time series should be processed first. Only nodes with time series data should be considered
        time_series = pd.read_csv(f"inputs/data/{tech}.csv", header=0, nrows=0)
        for node in time_series.columns:
            if node in ["Year", "Month", "Day", "Interval"]:
                continue
            row = build_re_row(node, node[:3], tech, fuel, tech_to_cost_file_name[tech])

            node = node[:3]
            if hasattr(max_build, "__getitem__"):
                _max_build = max_build[node] if node in max_build.keys() else 0.0
            else:
                _max_build = max_build
            row["max_build"] = _max_build
            row["min_build"] = min_build

            if hasattr(initial_capacity, "__getitem__"):
                _initial = initial_capacity[node] if node in initial_capacity.keys() else 0.0
            else:
                _initial = initial_capacity
            row["initial_capacity"] = _initial

            generators = pd.concat((generators, pd.DataFrame(row, index=[0]))).reset_index(drop=True)

    nuclear = pd.read_csv(
        r"unprocessed_inputs/existing_and_potential_capacity/nuclear_capacity_current.csv",
        usecols=["locs", "existing", "under_construction"],
        index_col="locs",
    ).to_dict()
    biogas_nodes = pd.read_csv(
        r"unprocessed_inputs/existing_and_potential_capacity/biogas_yearly_resource.csv",
    )["locs"].to_list()
    for node in nodes:
        if node in nuclear["existing"].keys():
            row = {
                "name": node + "_nuclear",
                "fuel": "nuclear",
                "capex": cost_energy_cap["nuclear"],
                "fom": cost_om_annual["nuclear"],
                "vom": cost_om_con["nuclear"] + cost_om_prod["nuclear"],
                "lifetime": lifetime["nuclear"],
                "discount_rate": DISCOUNT_RATE,
                "heat_rate_base": 0,
                "heat_rate_incr": 34,
                "unit_size": 1.0,
                "node": node,
                "max_build": 50,
                "min_build": nuclear["under_construction"][node],
                "initial_capacity": nuclear["existing"][node],
                "line": "other_connections",
                "unit_type": "baseload",
                "scenarios": "Base,ZeroCarbon",
                "near_optimum": False,
                "range_group": "",
            }
            generators = pd.concat((generators, pd.DataFrame(row, index=[0]))).reset_index(drop=True)

        if node in biogas_nodes:
            row = {
                "name": node + "_biogas",
                "fuel": "biogas",
                "capex": cost_energy_cap["biogas_to_electricity_supply"],
                "fom": cost_om_annual["biogas_to_electricity_supply"],
                "vom": cost_om_con["biogas_to_electricity_supply"] + cost_om_prod["biogas_to_electricity_supply"],
                "lifetime": lifetime["biogas_to_electricity_supply"],
                "discount_rate": DISCOUNT_RATE,
                "heat_rate_base": 0,
                "heat_rate_incr": 6.2,  # CCGT rate
                "unit_size": 0.5,
                "node": node,
                "max_build": 0,
                "min_build": 0,
                "initial_capacity": 0,
                "line": "other_connections",
                "unit_type": "flexible",
                "scenarios": "Base,Re100",
                "near_optimum": False,
                "range_group": "",
            }
            generators = pd.concat((generators, pd.DataFrame(row, index=[0]))).reset_index(drop=True)

        row = {
            "name": node + "_ccgt",
            "fuel": "gas",
            "capex": cost_energy_cap["ccgt"],
            "fom": cost_om_annual["ccgt"],
            "vom": cost_om_con["ccgt"] + cost_om_prod["ccgt"],
            "lifetime": lifetime["ccgt"],
            "discount_rate": DISCOUNT_RATE,
            "heat_rate_base": 0,
            "heat_rate_incr": 6.2,
            "unit_size": 0.5,
            "node": node,
            "max_build": 0,
            "min_build": 0,
            "initial_capacity": 0,
            "line": "other_connections",
            "unit_type": "flexible",
            "scenarios": "Base",
            "near_optimum": False,
            "range_group": "",
        }
        generators = pd.concat((generators, pd.DataFrame(row, index=[0]))).reset_index(drop=True)

    generators = generators.reset_index().rename(columns={"index": "id"})
    if write:
        generators.to_csv("inputs/config/generators.csv", index=False)
    return generators


def build_reservoirs_csv(write=True):
    columns = {
        "name": str,
        "fuel": str,
        "capex_p": float,
        "capex_e": float,
        "fom": float,
        "vom": float,
        "lifetime": int,
        "discount_rate": float,
        "heat_rate_base": float,
        "heat_rate_incr": float,
        "unit_size": float,
        "node": str,
        "discharge_efficiency": float,
        "max_build_p": float,
        "max_build_e": float,
        "min_build_p": float,
        "min_build_e": float,
        "initial_power_capacity": float,
        "initial_energy_capacity": float,
        "duration": float,
        "line": str,
        "unit_type": str,
        "scenarios": str,
        "near_optimum": bool,
        "range_group": str,
    }

    def build_row(name, node, tech, fuel, cost_name):
        row = {
            "name": name,
            "fuel": fuel,
            "capex_p": cost_energy_cap[cost_name],
            "capex_e": 0,
            "fom": cost_om_annual[cost_name],
            "vom": cost_om_con[cost_name] + cost_om_prod[cost_name],
            "lifetime": lifetime[cost_name],
            "discount_rate": DISCOUNT_RATE,
            "heat_rate_base": 0,
            "heat_rate_incr": 0,
            "unit_size": 0.001,
            "node": node,
            "discharge_efficiency": 0.9,  # TODO: revise this
            "max_build_p": 0,
            "max_build_e": 0,
            "min_build_p": 0,
            "min_build_e": 0,
            "initial_power_capacity": power_capacity[node],
            "initial_energy_capacity": energy_capacity[node],
            "duration": 0,
            "line": tech_to_conn[tech],
            "unit_type": tech_to_flex[tech],
            "scenarios": "all",
            "near_optimum": False,
            "range_group": "",
        }
        return row

    reservoirs = pd.DataFrame(columns=columns.keys()).astype(columns)

    cost_energy_cap = read_cost_file("cost_energy_cap")  # actually power
    cost_om_annual = read_cost_file("cost_om_annual")
    cost_om_con = read_cost_file("cost_om_con")
    cost_om_prod = read_cost_file("cost_om_prod")
    lifetime = read_cost_file("lifetime")
    tech = "hydro-reservoir-inflow"
    fuel = tech_fuels[tech]

    power_capacity = pd.read_csv(
        "unprocessed_inputs/existing_and_potential_capacity/hydro_capacity_current.csv",
        usecols=["locs", tech_to_cost_file_name[tech]],
        index_col="locs",
    ).to_dict()[tech_to_cost_file_name[tech]]
    energy_capacity = pd.read_csv(
        "unprocessed_inputs/existing_and_potential_capacity/hydro_storage_capacity_current.csv",
        usecols=["locs", tech_to_cost_file_name[tech]],
        index_col="locs",
    ).to_dict()[tech_to_cost_file_name[tech]]

    # time series should be processed first. Only nodes with time series data should be considered
    time_series = pd.read_csv(f"inputs/data/{tech}.csv", header=0, nrows=0)
    for node in time_series.columns:
        if node in ["Year", "Month", "Day", "Interval"]:
            continue
        row = build_row(node, node[:3], tech, fuel, tech_to_cost_file_name[tech])

        node = node[:3]

        reservoirs = pd.concat((reservoirs, pd.DataFrame(row, index=[0]))).reset_index(drop=True)
    reservoirs = reservoirs.reset_index().rename(columns={"index": "id"})
    if write:
        reservoirs.to_csv("inputs/config/reservoirs.csv", index=False)
    return reservoirs


def build_storages_csv(write=True):
    columns = {
        "name": str,
        "initial_power_capacity": float,
        "initial_energy_capacity": float,
        "duration": float,
        "capex_p": float,
        "capex_e": float,
        "fom": float,
        "vom": float,
        "lifetime": int,
        "discount_rate": float,
        "node": str,
        "charge_efficiency": float,
        "discharge_efficiency": float,
        "max_build_p": float,
        "max_build_e": float,
        "min_build_p": float,
        "min_build_e": float,
        "line": str,
        "unit_type": str,
        "scenarios": str,
        "near_optimum": bool,
        "range_group": str,
    }
    storages = pd.DataFrame(columns=columns.keys()).astype(columns)

    existing_power = pd.read_csv(
        r"unprocessed_inputs/existing_and_potential_capacity/hydro_capacity_current.csv",
        usecols=["locs", "pumped_hydro"],
        index_col="locs",
    )
    existing_energy = pd.read_csv(
        r"unprocessed_inputs/existing_and_potential_capacity/hydro_storage_capacity_current.csv",
        usecols=["locs", "pumped_hydro"],
        index_col="locs",
    )

    existing = pd.merge(existing_power, existing_energy, on="locs", how="outer", suffixes=("_p", "_e"))
    existing = existing.fillna(0)

    capex_p = read_cost_file("cost_energy_cap")  # power - file is badly named
    capex_e = read_cost_file("cost_storage_cap")  # energy
    lifetime = read_cost_file("lifetime")
    efficiency = read_cost_file("energy_eff")

    def build_row(node, tech, p, e):
        row = {
            "name": node + suffix_dict[tech],
            "initial_power_capacity": p,
            "initial_energy_capacity": e,
            "duration": 0,  # overridden by energy capacity
            "capex_p": capex_p[tech],
            "capex_e": capex_e[tech],
            "fom": 0,
            "vom": 0,
            "lifetime": lifetime[tech],
            "discount_rate": DISCOUNT_RATE,
            "node": node,
            "charge_efficiency": efficiency[tech],
            "discharge_efficiency": 1.0,  # losses aggregated into charge direction
            "max_build_p": 1000,
            "max_build_e": 1000,
            "min_build_p": 0,
            "min_build_e": 0,
            "line": suffix_dict[tech].split("_")[-1] + "_connection",
            "unit_type": suffix_dict[tech].split("_")[-1],
            "scenarios": "all",
            "near_optimum": False,
            "range_group": "",
        }
        return row

    for node in nodes:
        if node not in existing.index:
            continue
        p, e = existing.loc[existing.index == node, ["pumped_hydro_p", "pumped_hydro_e"]].values.flatten()
        row = build_row(node, "pumped_hydro", p, e)
        storages = pd.concat((storages, pd.DataFrame(row, index=[0]))).reset_index(drop=True)
    for node in nodes:
        row = build_row(node, "battery", 0, 0)
        storages = pd.concat((storages, pd.DataFrame(row, index=[0]))).reset_index(drop=True)
    storages = storages.reset_index().rename(columns={"index": "id"})
    if write:
        storages.to_csv("inputs/config/storages.csv", index=False)
    return storages


def build_lines_csv(write=True):
    columns = {
        "name": str,
        "length": float,
        "capex": float,
        "transformer_capex": float,
        "fom": float,
        "vom": float,
        "lifetime": int,
        "discount_rate": float,
        "node_start": str,
        "node_end": str,
        "loss_factor": float,
        "initial_capacity": float,
        "max_build": float,
        "min_build": float,
        "unit_type": str,
        "scenarios": str,
        "near_optimum": bool,
        "range_group": str,
    }
    lines = pd.DataFrame(
        columns=columns.keys(),
    ).astype(columns)

    cost_energy_cap = read_cost_file("cost_energy_cap")
    cost_om_annual = read_cost_file("cost_om_annual")

    cost_om_con = read_cost_file("cost_om_con")
    cost_om_prod = read_cost_file("cost_om_prod")

    lifetime = read_cost_file("lifetime")
    efficiency = read_cost_file("energy_eff")

    potential = pd.read_csv(
        "unprocessed_inputs/existing_and_potential_capacity/transmission_capacity_current_potential.csv",
        index_col="locs",
    )
    potential[["techs", "end"]] = potential["techs"].str.split(":", expand=True)

    def build_row(start, end, existing, max_build, line_type):
        row = {
            "name": start + "-" + end,
            "length": 1,
            "capex": cost_energy_cap[line_type],
            "transformer_capex": 0,
            "fom": cost_om_annual[line_type],
            "vom": cost_om_con[line_type] + cost_om_prod[line_type],
            "lifetime": lifetime[line_type],
            "discount_rate": LINES_DISCOUNT_RATE,
            "node_start": start,
            "node_end": end,
            "loss_factor": 1 - efficiency[line_type],
            "initial_capacity": existing,
            "max_build": max_build,
            "min_build": 0,
            "unit_type": line_type,
            "scenarios": "all",
            "near_optimum": False,
            "range_group": "",
        }
        return row

    for start_node, line in potential.iterrows():
        row = build_row(
            start_node,
            line["end"],
            line["current"],
            line["potential"],
            line["techs"],
        )
        lines = pd.concat((lines, pd.DataFrame(row, index=[0]))).reset_index(drop=True)

    nodes = lines[["node_start", "node_end"]]
    nodes = np.sort(nodes.values, axis=1)
    lines_copy = lines.copy()
    lines_copy["canonical_start"] = nodes[:, 0]
    lines_copy["canonical_end"] = nodes[:, 1]

    grouped = lines_copy.groupby(["canonical_start", "canonical_end", "unit_type"])
    for name, group in grouped:
        for j, col in enumerate(lines.columns):
            if col in ("name", "node_start", "node_end"):
                continue
            is_equal = group.iloc[0, j] == group.iloc[1, j]
            is_nan = pd.isna(group.iloc[0, j]) and pd.isna(group.iloc[1, j])
            assert is_equal or is_nan, f"Duplicate lines are asymmetric: {name=}, {col=}"
    lines = lines_copy.drop_duplicates(
        subset=["canonical_start", "canonical_end", "unit_type"],
        keep="first",
    )
    lines = lines.drop(columns=["canonical_start", "canonical_end"])

    for name in (
        "other_connections",
        "pv_connection",
        "onsw_connection",
        "offw_connection",
        "phes_connection",
        "bess_connection",
    ):
        row = {
            "name": name,
            "length": 0,
            "capex": 0,
            "transformer_capex": 0,
            "fom": 0,
            "vom": 0,
            "lifetime": 0,
            "discount_rate": LINES_DISCOUNT_RATE,
            "node_start": "",
            "node_end": "",
            "loss_factor": 0,
            "initial_capacity": 0,
            "max_build": 0,
            "min_build": 0,
            "unit_type": "",
            "scenarios": "all",
            "near_optimum": False,
            "range_group": "",
        }
        lines = pd.concat((lines, pd.DataFrame(row, index=[0]))).reset_index(drop=True)

    lines = lines.reset_index().rename(columns={"index": "id"})
    if write:
        lines.to_csv("inputs/config/lines.csv", index=False)
    return lines


if __name__ == "__main__":

    for file in os.listdir("unprocessed_inputs/timeseries"):
        print(file, end=": ", flush=True)
        if "capacityfactors" in file:
            read_and_process_timeseries(f"unprocessed_inputs/timeseries/{file}", True)
        if "demand" in file:
            read_and_process_timeseries(f"unprocessed_inputs/timeseries/{file}", False)
        print("done")
    print("flexible_annual_biogas: ", end="")
    build_flexible_annual_biogas_csv()
    print("done")
    print("generators: ", end="")
    build_generators_csv()
    print("done")
    print("reservoirs: ", end="")
    build_reservoirs_csv()
    print("done")
    print("storage: ", end="")
    build_storages_csv()
    print("done")
    print("lines: ", end="")
    build_lines_csv()
    print("done")
