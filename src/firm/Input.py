import numpy as np
import pandas as pd
from numba import boolean, float64, int64, njit, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore

from firm.Simulation import Simulate
from firm.Utils import array_max, trim_leap_days
from firm.Network import generate_network
from firm.Costs import CostsType
from firm.Profile import ProfileData, ProfileDataType

Nodel = pd.read_csv("data/nodes.csv", header=None)[0].to_numpy()
Nodel_int = np.arange(len(Nodel))
Node_dict = dict(zip(Nodel, Nodel_int))

Mload = pd.read_csv("data/electricity-demand.csv")
Mload = trim_leap_days(Mload)
# MLoad /= 1000  # MW to GW

resolution = 24 / Mload["Interval"].max()
first_year = Mload.iloc[0, 0]
final_year = Mload.iloc[-1, 0]
assert (Mload.iloc[-1, [1, 2, 3]] == [12, 31, Mload["Interval"].max()]).all(), "time series does not end at end of year"
maxyears = final_year - first_year + 1
phes_efficiency = 0.9, 0.9
bess_efficiency = 0.95, 0.95

TSpv = trim_leap_days(pd.read_csv("data/open-field-pv.csv"))
TSrpv = trim_leap_days(pd.read_csv("data/rooftop-pv.csv"))
TSonsw = trim_leap_days(pd.read_csv("data/wind-onshore.csv"))
TSoffw = trim_leap_days(pd.read_csv("data/wind-offshore.csv"))
TSror = trim_leap_days(pd.read_csv("data/hydro-ror.csv"))
TSbio = pd.read_csv("data/flexible_annual_biogas.csv")


for df in (TSpv, TSrpv, TSonsw, TSoffw, TSror):
    assert df.shape == Mload.shape, "time series shapes do not match"
    assert (
        df.loc[0, ["Year", "Month", "Day", "Interval"]] == Mload.loc[0, ["Year", "Month", "Day", "Interval"]]
    ).all(), "time series do not align"
    assert (
        df.loc[len(Mload) - 1, ["Year", "Month", "Day", "Interval"]]
        == Mload.loc[len(Mload) - 1, ["Year", "Month", "Day", "Interval"]]
    ).all(), "time series do not align"
    # sort columns and store as numpy

TSpv = TSpv.loc[:, Nodel].to_numpy()
TSrpv = TSrpv.loc[:, Nodel].to_numpy()
TSonsw = TSonsw.loc[:, Nodel].to_numpy()
TSoffw = TSoffw.loc[:, Nodel].to_numpy()
TSror = TSror.loc[:, Nodel].to_numpy()

Mload = Mload.to_numpy() / 1000  # MW to GW

existing = pd.read_csv("data/existing_capacity.csv", index_col="locs")

EhydP = existing.loc[Nodel, "hydro_reservoir_power"].to_numpy()
EhydE = existing.loc[Nodel, "hydro_reservoir_energy"].to_numpy()
Eror = existing.loc[Nodel, "hydro_run_of_river"].to_numpy()
EphP = existing.loc[Nodel, "pumped_hydro_power"].to_numpy()
EphE = existing.loc[Nodel, "pumped_hydro_energy"].to_numpy()
Enuke = existing.loc[Nodel, ["existing_nuclear", "under_construction_nuclear"]].sum(axis=1).to_numpy()

# TODO: transmission line efficiency
lines = pd.read_csv("data/lines.csv")
Elines = lines.loc[:, "existing_capacity"].to_numpy()
Rlines = lines.loc[:, "max_build"].to_numpy()
basic_network = lines.loc[:, ["start", "end"]].map(lambda x: Node_dict[x]).to_numpy()

resource = pd.read_csv("data/resource_potential.csv", index_col="locs")
Rpv_base = resource.loc[Nodel, "pv"].to_numpy()
Rrpv_base = resource.loc[Nodel, "rooftop"].to_numpy()
Ronsw_base = resource.loc[Nodel, "onsw"].to_numpy()
Roffw_base = resource.loc[Nodel, "offw"].to_numpy()

Rpv_st = resource.loc[Nodel, "pv_sociotech"].to_numpy()
Rrpv_st = resource.loc[Nodel, "rooftop_sociotech"].to_numpy()
Ronsw_st = resource.loc[Nodel, "onsw_sociotech"].to_numpy()
Roffw_st = resource.loc[Nodel, "offw_sociotech"].to_numpy()
Rror_st = resource.loc[Nodel, "run_of_river"].to_numpy()

Rnuke_mask = resource.loc[Nodel, "nuclear build"].to_numpy()
Rnuke_const = 50  # GW / node

data_spec = [
    ("scenario", int64),
    ("profiling", int64),
    ("Nodel_int", int64[:]),
    ("resolution", float64),
    ("ph_charge_eff", float64),
    ("ph_discha_eff", float64),
    ("bs_charge_eff", float64),
    ("bs_discha_eff", float64),
    ("years", int64),
    ("intervals", int64),
    ("Mload", float64[:, :]),
    ("TSpv", float64[:, :]),
    ("TSrpv", float64[:, :]),
    ("TSonsw", float64[:, :]),
    ("TSoffw", float64[:, :]),
    ("TSror", float64[:, :]),
    ("EhydP", float64[:]),
    ("EhydE", float64[:]),
    ("Eror", float64[:]),
    ("EphP", float64[:]),
    ("EphE", float64[:]),
    ("Enuke", float64[:]),
    ("Ebio", float64[:]),
    ("Elines", float64[:]),
    ("flexible_resource", float64[:, :]),
    ("basic_network", int64[:, :]),
    ("network", int64[:, :, :, :]),
    ("network_mask", boolean[:]),
    ("directconns", int64[:, :]),
    ("trans_mask", boolean[:, :]),
    ("triangulars", int64[:]),
    ("Rnuke_mask", boolean[:]),
    ("nhvi", int64),
    ("nodes", int64),
    ("energy", float64),
    # Topology
    ("network_mask", boolean[:]),
    ("network", int64[:, :]),
    ("networksteps", int64),
    ("cache_0_donors", types.DictType(int64, int64[:, :])),
    ("cache_n_donors", types.DictType(types.UniTuple(int64, 2), int64[:, :, :])),
    ("lb", float64[:]),
    ("ub", float64[:]),
    ("x0", float64[:]),
]


@jitclass(data_spec)
class StaticData:
    def __init__(
        self,
        scenario: int,
        years: int,
        profiling: int,
        networksteps: int,
    ):
        self.scenario = scenario
        self.profiling = profiling
        self.networksteps = networksteps
        self.resolution = resolution

        # TODO: these are not accounted for correctly in Simulation.py
        self.ph_charge_eff, self.ph_discha_eff = phes_efficiency
        self.bs_charge_eff, self.bs_discha_eff = bess_efficiency

        if years == -1:
            self.years = maxyears
        elif years <= maxyears:
            self.years = years
        else:
            raise Exception
        self.intervals = int(self.years * 8760 / self.resolution)

        # PLACE HOLDER
        # scenario mask
        scenario_mask = np.ones(len(Nodel_int), np.bool_)
        self.Nodel_int = Nodel_int[scenario_mask]

        self.Mload = np.atleast_2d(Mload[: self.intervals, scenario_mask])
        self.TSpv = np.atleast_2d(TSpv[: self.intervals, scenario_mask])
        self.TSrpv = np.atleast_2d(TSrpv[: self.intervals, scenario_mask])
        self.TSonsw = np.atleast_2d(TSonsw[: self.intervals, scenario_mask])
        self.TSoffw = np.atleast_2d(TSoffw[: self.intervals, scenario_mask])
        self.TSror = np.atleast_2d(TSror[: self.intervals, scenario_mask])

        self.EhydP = EhydP[scenario_mask]
        self.EhydE = EhydE[scenario_mask]
        self.Eror = Eror[scenario_mask]
        self.EphP = EphP[scenario_mask]
        self.EphE = EphE[scenario_mask]
        self.Enuke = Enuke[scenario_mask]
        self.Ebio = np.zeros(scenario_mask.sum(), dtype=np.float64)

        with objmode():
            (
                self.network,
                self.network_mask,
                self.trans_mask,
                self.cache_0_donors,
                self.cache_n_donors,
            ) = generate_network(basic_network, self.Nodel_int, self.networksteps)

        self.Elines = Elines[self.network_mask]
        self.Rnuke_mask = Rnuke_mask[scenario_mask]

        self.nhvi = self.network_mask.sum()
        self.nodes = len(self.Nodel_int)

        self.energy = self.Mload.sum() * 1000 * self.resolution / self.years  # MWh p.a.

        self.lb = np.concat((
            np.zeros(self.nodes),  # pv
            np.zeros(self.nodes),  # rpv
            np.zeros(self.nodes),  # onsw
            np.zeros(self.nodes),  # offw
            np.zeros(self.nodes),  # ror
            np.zeros(self.nodes),  # nuke
            np.zeros(self.nodes),  # gas
            np.zeros(self.nodes),  # php
            np.zeros(self.nodes),  # phe
            np.zeros(self.nodes),  # bessp
            np.zeros(self.nodes),  # besse
            np.zeros(self.nhvi),  # lines
        ))
        self.ub = np.concat((
            Rpv_st[scenario_mask],  # pv
            Rrpv_st[scenario_mask],  # rpv
            Ronsw_st[scenario_mask],  # onsw
            Roffw_st[scenario_mask],  # offw
            Rror_st[scenario_mask],  # ror
            np.array([Rnuke_const if allowed else 0.0 for allowed in Rnuke_mask]),  # nuke
            np.zeros(self.nodes),  # gas
            np.full(self.nodes, 10),  # php
            np.full(self.nodes, 480),  # phe
            np.zeros(self.nodes),  # bessp
            np.zeros(self.nodes),  # besse
            Rlines[self.network_mask],  # lines
        ))

        mloadmax = np.array([array_max(col) for col in self.Mload.T])
        self.x0 = np.concatenate(
            (
                self.Mload.sum() / self.intervals * 0.75 / self.nodes / np.array([col.mean() for col in self.TSpv.T]),
                self.Mload.sum() / self.intervals * 0.0 / self.nodes / np.array([col.mean() for col in self.TSrpv.T]),
                self.Mload.sum() / self.intervals * 0.5 / self.nodes / np.array([col.mean() for col in self.TSonsw.T]),
                self.Mload.sum() / self.intervals * 0.25 / self.nodes / np.array([col.mean() for col in self.TSoffw.T]),
                self.Mload.sum() / self.intervals * 0.0 / self.nodes / np.array([col.mean() for col in self.TSror.T]),
                np.zeros(self.nodes),
                np.zeros(self.nodes),
                mloadmax * 0.5,
                mloadmax * 36,
                mloadmax * 0.5,
                mloadmax * 4,
                np.repeat(array_max(mloadmax) * 0.8, self.nhvi),
            )
        )
        self.x0 = np.minimum(self.ub, self.x0)


StaticDataType = StaticData.class_type.instance_type


@njit
def get_next_slice(x, length, idx_start):
    return x[idx_start: idx_start + length].copy(), idx_start + length


asset_spec = [
    # Capacities in GW/GWh
    ("Cpv", float64[:]),
    ("Crpv", float64[:]),
    ("Consw", float64[:]),
    ("Coffw", float64[:]),
    ("Cror", float64[:]),
    ("Cnuke", float64[:]),
    ("Cgas", float64[:]),
    ("CphP", float64[:]),
    ("CphE", float64[:]),
    ("CbsP", float64[:]),
    ("CbsE", float64[:]),
    ("ChydP", float64[:]),
    ("ChydE", float64[:]),
    ("Cbio", float64[:]),
    ("Clines", float64[:]),
    ("Cpeak", float64[:]),
]


@jitclass(asset_spec)
class AssetData:
    def __init__(
        self,
        static: StaticDataType,  # type: ignore
        x: np.ndarray[np.float64],
    ):
        idx_start = 0
        self.Cpv, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Crpv, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Consw, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Coffw, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cror, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cnuke, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cgas, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.CphP, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.CphE, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.CbsP, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.CbsE, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Clines, idx_start = get_next_slice(x, static.nhvi, idx_start)

        self.Cbio = static.Ebio
        self.Cror += static.Eror
        self.CphP += static.EphP
        self.CphP += static.EphP
        self.CphE += static.EphE
        self.Cnuke += static.Enuke
        self.ChydP = static.EhydP
        self.ChydE = static.EhydE

        self.Cpeak = self.ChydP + self.CphP + self.CbsP


AssetDataType = AssetData.class_type.instance_type


operational_spec = [
    # Nodally diaggregated operations in GW/GWh
    ("Mflexible", float64[:, :]),
    ("Mphdischarge", float64[:, :]),
    ("Mbsdischarge", float64[:, :]),
    ("Mphcharge", float64[:, :]),
    ("Mbscharge", float64[:, :]),
    ("Mphstorage", float64[:, :]),
    ("Mbsstorage", float64[:, :]),
    ("Mdeficit", float64[:, :]),
    ("Mspillage", float64[:, :]),
    ("Mnetload", float64[:, :]),
    ("Mimport", float64[:, :]),
    ("Mpv", float64[:, :]),
    ("Mrpv", float64[:, :]),
    ("Monsw", float64[:, :]),
    ("Moffw", float64[:, :]),
    ("Mload", float64[:, :]),
    ("Mreservoir", float64[:, :]),
    ("Mhydro", float64[:, :]),
    ("Mbio", float64[:, :]),
    ("Mgas", float64[:, :]),
    ("Munbalanced", float64[:, :]),
    ("Timport", float64[:, :, :]),
    ("Texport", float64[:, :, :]),
]


@jitclass(operational_spec)
class OperationalData:
    def __init__(
        self,
        static: StaticDataType,  # type: ignore
        assets: AssetDataType,  # type: ignore
    ):
        self.Mnetload = (
            static.Mload
            - assets.Cpv * static.TSpv
            - assets.Crpv * static.TSrpv
            - assets.Consw * static.TSonsw
            - assets.Coffw * static.TSoffw
            - assets.Cror * static.TSror
        )
        self.Munbalanced = self.Mnetload.copy()
        self.Mdeficit = np.maximum(0, self.Mnetload)
        self.Mspillage = -np.minimum(0, self.Mnetload)

        self.Mhydro = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mgas = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mflexible = np.zeros((static.intervals, static.nodes), dtype=np.float64)

        self.Mphdischarge = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mphcharge = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mphstorage = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mphstorage[-1] = 0.5 * assets.CphE

        self.Mbsdischarge = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mbscharge = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mbsstorage = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mbsstorage[-1] = 0.5 * assets.CbsE

        self.Timport = np.zeros((static.intervals, static.nodes, static.nhvi), dtype=np.float64)
        self.Texport = np.zeros((static.intervals, static.nodes, static.nhvi), dtype=np.float64)


OperationalDataType = OperationalData.class_type.instance_type


# Specify the types for jitclass
solution_spec = [
    ("x", float64[:]),
    ("static", StaticDataType),
    ("costs", CostsType),
    ("assets", AssetDataType),
    ("operations", OperationalDataType),
    ("profile", ProfileDataType),
    # Objectives
    ("Penalties", float64),
    ("Feasible", boolean),
    ("Lcoe", float64),
    ("Lcog", float64),
    ("Lcob", float64),
    ("Lcosp", float64),
    ("Lcosb", float64),
    ("Lcobs", float64),
    ("Lcobt", float64),
    ("Lcobl", float64),
    ("Capex", float64),
    ("Opex", float64),
]


@jitclass(solution_spec)
class Solution:
    def __init__(
        self,
        x: np.ndarray,
        static: StaticDataType,  # type: ignore
    ):
        self.x = x
        self.static = static
        assert len(x) == len(self.static.lb)
        self.assets = AssetData(self.static, x)
        self.operations = OperationalData(self.static, self.assets)
        if self.static.profiling != 0:
            self.profile = ProfileData(self.static.profiling, 0.0)


SolutionType = Solution.class_type.instance_type


@njit
def Evaluate(S: SolutionType, costs: CostsType):  # type: ignore
    Simulate(S)

    S.Penalties = np.maximum(0, S.operations.Mdeficit.sum()) * 1000  # MWh/resolution

    CHVI = np.zeros(len(S.static.network_mask), dtype=np.float64)
    CHVI[S.static.network_mask] = S.assets.Clines

    cost = np.array(
        [
            # generation capex
            S.assets.Cpv.sum() * costs.pv[0],
            S.assets.Consw.sum() * costs.onsw[0],
            S.assets.Coffw.sum() * costs.offw[0],
            0,  # S.CGas.sum()  * costs.gas[0],
            (S.assets.ChydP.sum()) * costs.hydro[0],
            # generation fom
            S.assets.Cpv.sum() * costs.pv[1],
            S.assets.Consw.sum() * costs.onsw[1],
            0,  # S.CGas.sum()  * costs.gas[1],
            (S.assets.ChydP.sum()) * costs.hydro[1],
            # generation vom
            # pv, onsw, battery are 0
            0,  # S.GGas.sum() * S.static.resolution / S.static.years * costs.gas[2],
            S.operations.Mflexible.sum() * S.static.intervals
            * S.static.resolution / S.static.years * costs.hydro[2],
            # storage
            S.assets.CphP.sum() * costs.phes[0],
            S.assets.CphE.sum() * costs.phes[1],
            S.assets.CphP.sum() * costs.phes[2],
            S.operations.Mphdischarge.sum() * S.static.resolution / S.static.years * costs.phes[3],
            costs.phes[4],
        ]
        +
        # transmission network
        list(
            (
                S.assets.Cpv.sum()
                # rpv not costed - in distribution
                + S.assets.Consw.sum()
                + S.assets.Coffw.sum()
                # + S.CGas.sum()
                + S.assets.ChydP.sum()
                + S.assets.Cbio.sum()
                + S.assets.CphP.sum()
                # batteries not costed - locationally unconstrained
            )
            * costs.ac
        )
        + list((CHVI * costs.hvi).sum(axis=1))
    )

    # Levelised Costs of:
    # Electricity
    S.Lcoe = cost.sum() / S.static.energy
    # Generation
    S.Lcog = cost[:10].sum() / (
        1000
        * S.static.resolution
        / S.static.years
        * (
            S.operations.Mpv.sum()
            + S.operations.Mrpv.sum()
            + S.operations.Monsw.sum()
            + S.operations.Moffw.sum()
            # +S.operations.Mgas.sum()
            + S.operations.Mflexible.sum()
        )
    )
    # Storage
    # S.LCOSP = zero_safe_division(cost[10:15].sum(), S.MDischarge.sum()*S.static.resolution/S.static.years)
    # Balancing - Storage
    S.Lcobs = cost[10:15].sum() / S.static.energy
    # Balancing - Transmission
    S.Lcobt = cost[15:].sum() / S.static.energy
    # Balancing - Spillage
    S.Lcobl = S.Lcoe - S.Lcog - S.Lcobs - S.Lcobt
    S.Lcob = S.Lcobs + S.Lcobt + S.Lcobl

    S.Capex = sum([cost[i] for i in [0, 1, 2, 3, 10, 11, 15, 18]]) / S.static.energy
    S.Opex = S.Lcoe - S.Capex
    S.Feasible = S.Penalties < 1e-6

    return S.Lcoe, S.Penalties


if __name__ == "__main__":
    from firm.Parameters import Parameters
    from firm.Costs import RawCosts

    parameters = Parameters(s=21, y=1, p=0, n=3)
    static = StaticData(*parameters)
    costs = RawCosts(static).CostFactors()
    x0 = static.x0
    S = Solution(x0, static)
    Evaluate(S, costs)

    print("Lcoe:", S.Lcoe)
    print("Feasible:", S.Feasible)
    print("Penalties:", S.Penalties)
