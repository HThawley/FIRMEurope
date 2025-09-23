import pandas as pd 
import os 
from numba import njit

DISCOUNT_RATE = 0.07

suffix_dict = {
    "hydro-reservoir-inflow":"_hydro",
    "hydro-ror":"_ror",
    "open-field-pv":"_pv",
    "rooftop-pv":"_roof",
    "wind-offshore":"_offw",
    "wind-onshore":"_onsw",
}

tech_fuels = {
    "hydro-reservoir-inflow":"water",
    "hydro-ror":"water",
    "open-field-pv":"solar",
    "rooftop-pv":"solar",
    "wind-offshore":"wind",
    "wind-onshore":"wind",
}

tech_to_flex = {
    "hydro-reservoir-inflow":"flexible",
    "hydro-ror":"ror",
    "open-field-pv":"solar",
    "rooftop-pv":"solar",
    "wind-offshore":"wind",
    "wind-onshore":"wind",
}

tech_to_conn = {
    "hydro-reservoir-inflow":"other_connections",
    "hydro-ror":"other_connections",
    "open-field-pv":"pv_connection",
    "rooftop-pv":"pv_connection",
    "wind-offshore":"off-wind_connection",
    "wind-onshore":"ons-wind_connection",
}

tech_to_cost_file_name = {
    "hydro-reservoir-inflow":"hydro_reservoir",
    "hydro-ror":"hydro_run_of_river",
    "open-field-pv":"open_field_pv",
    "rooftop-pv":"roof_mounted_pv",
    "wind-offshore":"wind_offshore",
    "wind-onshore":"wind_onshore_monopoly",
}

@njit
def generate_interval_no(interval, day):
    for i in range(1, len(interval)):
        if day[i] == day[i-1]:
            interval[i] = interval[i-1] + 1
        else:
            interval[i] = 1
    return interval

def read_and_process_timeseries(filename, suffixes=True):
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

        ts_cols = {col:col+suffix for col in df.columns if col not in (
            timecol, "Year", "Month", "Day", "Interval",
        )}
        df = df.rename(columns=ts_cols)

        columns = ["Year", "Month", "Day", "Interval"] + list(ts_cols.values())
    else: 
        new_fname = filename.split("/")[-1].removesuffix(".csv")
        columns = ["Year", "Month", "Day", "Interval"] + list(df.columns[1:])

    df = df[columns]
    
    df.to_csv(f"inputs/data/{new_fname}.csv", index=False)

def read_cost_file(fname, skip_units=True):
    df = pd.read_csv(f"unprocessed_inputs/techno_economic/{fname}.csv", index_col="techs", skipfooter=int(skip_units)).fillna(0)
    return df.to_dict()["0"]

def build_generators_csv():
    columns = ["name","fuel","capex","fom","vom","lifetime","discount_rate",
               "heat_rate_base","heat_rate_incr","unit_size","node","max_build",
               "min_build","initial_capacity","line","unit_type","scenarios",
               "near_optimum","range_group"]
    dtypes=[str, str, float, float, float, int, float, float, float, float, str, float, float, float, str, str, str, str, bool, str]

    generators = pd.DataFrame(
        columns=columns,
    ).astype(dict(zip(columns, dtypes)))
    
    cost_energy_cap = read_cost_file("cost_energy_cap")
    cost_om_annual = read_cost_file("cost_om_annual")

    cost_om_con = read_cost_file("cost_om_con")
    cost_om_prod = read_cost_file("cost_om_prod")

    lifetime = read_cost_file("lifetime", False)
    
    def build_row(tech, fuel, cost_name):
        row = {
            "name":"",
            "fuel": fuel,
            "capex": cost_energy_cap[cost_name],
            "fom": cost_om_annual[cost_name], 
            "vom": cost_om_con[cost_name] + cost_om_prod[cost_name], 
            "lifetime": lifetime[cost_name],
            "discount_rate": DISCOUNT_RATE,
            "heat_rate_base": 0, 
            "heat_rate_incr": 0, 
            "unit_size": 0.001, 
            "node": "",
            "max_build": 0,
            "min_build": 0,
            "initial_capacity": 0,
            "line": tech_to_conn[tech],
            "unit_type": tech_to_flex[tech],
            "scenarios": "base",
            "near_optimum": False, 
            "range_group":"",
        }
        return row
    
    for tech, fuel in tech_fuels.items():
        df = pd.read_csv(f"inputs/data/{tech}.csv", header=0, nrows=0)
        
        if "hydro" in tech:
            initial_capacity = pd.read_csv(
                "unprocessed_inputs/existing_and_potential_capacity/hydro_capacity_current.csv",
                usecols=["locs", tech_to_cost_file_name[tech]],
                index_col="locs",
            ).to_dict()[tech_to_cost_file_name[tech]]
            min_build=0
            max_build=0
        else:
            initial_capacity=0
            min_build=0
            max_build = pd.read_csv(
                "unprocessed_inputs/existing_and_potential_capacity/renewables_capacity_potential_exclusive.csv",
                index_col="locs",
            )
            max_build = max_build.loc[max_build["techs"] == tech_to_cost_file_name[tech], "energy_cap_max"]
        
            if tech == "open-field-pv":
                competing = pd.read_csv(
                    "unprocessed_inputs/existing_and_potential_capacity/renewables_capacity_potential_competing.csv",
                    index_col="locs",
                )
                col = [col for col in competing.columns if "pv" in col.lower()][0]
                competing = competing.loc[:, col]
                max_build = pd.merge(max_build, competing, on="locs", how="outer").fillna(0).sum(axis=1)
            elif tech == "wind-onshore":
                competing = pd.read_csv(
                    "unprocessed_inputs/existing_and_potential_capacity/renewables_capacity_potential_competing.csv",
                    index_col="locs",
                )
                col = [col for col in competing.columns if "wind" in col.lower()][0]
                competing = competing.loc[:, col]
                max_build = pd.merge(max_build, competing, on="locs", how="outer").fillna(0).sum(axis=1)

        for node in df.columns:
            if node in ["Year", "Month", "Day", "Interval"]:
                continue
            row = build_row(tech, fuel, tech_to_cost_file_name[tech])

            name, node = node, node[:3]
            # update row elements
            row["name"] = name
            row["node"] = node
            if hasattr(max_build, "__getitem__"):
                _max_build = max_build[node] if node in max_build.keys() else 0.
            else: 
                _max_build = max_build
            row["max_build"] = _max_build
            
            row["min_build"] = min_build
            
            if hasattr(initial_capacity, "__getitem__"):
                _initial = initial_capacity[node] if node in initial_capacity.keys() else 0.
            else: 
                _initial = initial_capacity
            row["initial_capacity"] = _initial
            
            generators = pd.concat((generators, pd.DataFrame(row, index=[0]))).reset_index(drop=True)
    generators = generators.reset_index().rename(columns={"index":"id"})
    generators.to_csv("inputs/config/generators.csv", index=False)




            



if __name__ == "__main__":
    
    for file in os.listdir("unprocessed_inputs/timeseries"):
        if "capacityfactors" in file:
            read_and_process_timeseries(f"unprocessed_inputs/timeseries/{file}", True)
        if "demand" in file:
            read_and_process_timeseries(f"unprocessed_inputs/timeseries/{file}", False)
    df = build_generators_csv()