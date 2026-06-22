import numpy as np
import pandas as pd
from numba import boolean, float64, int64, njit, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore

from firm.Simulation import Simulate
from firm.Utils import array_max
from firm.Network import generate_network
from firm.Costs import CostsType
from firm.Profile import ProfileData, ProfileDataType

Nodel = pd.read_csv("data/nodes.csv", header=None)[0].to_numpy()
Nodel_int = np.arange(len(Nodel))
Node_dict = dict(zip(Nodel, Nodel_int))

Mload = pd.read_csv("data/load.csv")

resolution = 24 / Mload["Interval"].max()
first_year = Mload.iloc[0, 0]
final_year = Mload.iloc[-1, 0]
maxyears = final_year - first_year + 1
assert (Mload.iloc[-1, [1, 2, 3]] == [12, 31, Mload["Interval"].max()]).all(), "time series does not end at end of year"

phes_efficiency = 0.9, 0.9
bess_efficiency = 0.95, 0.95


def read_and_trim(name):
    df = pd.read_csv(F"data/{name}.csv").fillna(0)
    assert all(
        (col in df.columns) for col in ("Year", "Month", "Day", "Interval")
    ), f"time series does not have time index in requried format. {name=}"

    df = df[(df.Year >= first_year) & (df.Year <= final_year)]
    df = df[~((df.Month == 2) & (df.Day == 29))]
    df = df.reset_index(drop=True)

    assert (
        df.loc[0, ["Year", "Month", "Day", "Interval"]] == Mload.loc[0, ["Year", "Month", "Day", "Interval"]]
    ).all(), f"time series does not align with MLoad. {name=}"

    assert df.shape == Mload.shape, f"time series shapes do not match. {name=}"
    # sort columns and store as numpy
    df = df.loc[4:, Nodel].to_numpy()

    return df


TSpfix = read_and_trim("solar_fixed")
TSpsat = read_and_trim("solar_sat")
TSonsw = read_and_trim("wind_onshore")
TSoffw = read_and_trim("wind_offshore")
TSror = read_and_trim("run_of_river_cf")
TSpond_inflow = read_and_trim("pondage_inflows")
TSphes_inflow = read_and_trim("ol-phes_inflows")
TShyd_inflow = read_and_trim("reservoir_inflows")
TSbio = pd.read_csv("data/flexible_annual_biogas.csv")

Mload = Mload.iloc[:, 4:].to_numpy()

existing = pd.read_csv("data/existing_capacity.csv", index_col="locs")

Eror = existing.loc[Nodel, "hydro_run_of_river"].to_numpy()
EpondP = existing.loc[Nodel, "pondage_power"].to_numpy()
EpondE = existing.loc[Nodel, "pondage_energy"].to_numpy()
EphP = existing.loc[Nodel, "pumped_power"].to_numpy()
EphE = existing.loc[Nodel, "pumped_energy"].to_numpy()
EhydP = existing.loc[Nodel, "reservoir_power"].to_numpy()
EhydE = existing.loc[Nodel, "reservoir_energy"].to_numpy()
Enuke = existing.loc[Nodel, "nuclear"].to_numpy()

# TODO: transmission line efficiency
lines = pd.read_csv("data/lines.csv")
Elines = lines.loc[:, "existing_capacity"].to_numpy()
Rlines = lines.loc[:, "max_build"].to_numpy()
basic_network = lines.loc[:, ["start", "end"]].map(lambda x: Node_dict[x]).to_numpy()

resource = pd.read_csv("data/resource_potential.csv", index_col="locs")
pv_base = resource.loc[Nodel, "pv"].to_numpy()
Rpsat_base = pv_base / 2
Rpfix_base = pv_base / 2
Ronsw_base = resource.loc[Nodel, "onsw"].to_numpy()
Roffw_base = resource.loc[Nodel, "offw"].to_numpy()
Rnlte = resource.loc[Nodel, "nuclear_LTE"].to_numpy()
Rphe = resource.loc[Nodel, "pumped_energy"].to_numpy()

pv_st = resource.loc[Nodel, "pv_sociotech"].to_numpy()
Rpfix_st = pv_st / 2
Rpsat_st = pv_st / 2
Ronsw_st = resource.loc[Nodel, "onsw_sociotech"].to_numpy()
Roffw_st = resource.loc[Nodel, "offw_sociotech"].to_numpy()

assert ((Enuke == -1) == (Rnlte == -1)).all(), "nuclear build allowance inconsistent"
Rnuke_mask = ~(Enuke == -1)
Rnlte_mask = Rnlte > 0
Rnuke_const = 200  # GW / node
Rgas_const = 200  # GW / node
Rphh_min = 4  # 4 hour storage minimum considered for PHES

data_spec = [
    ("scenario", int64),
    ("profiling", int64),
    ("Nodel_int", int64[:]),
    ("resolution", float64),
    ("storage_charge_eff", float64[:]),
    ("storage_discha_eff", float64[:]),
    ("years", int64),
    ("intervals", int64),
    ("Mload", float64[:, :]),
    ("TSpfix", float64[:, :]),
    ("TSpsat", float64[:, :]),
    ("TSonsw", float64[:, :]),
    ("TSoffw", float64[:, :]),
    ("TSror", float64[:, :]),
    ("TSpond_inflow", float64[:, :]),
    ("TSphes_inflow", float64[:, :]),
    ("TShyd_inflow", float64[:, :]),
    ("EhydP", float64[:]),
    ("EhydE", float64[:]),
    ("EpondP", float64[:]),
    ("EpondE", float64[:]),
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
    ("Rnlte_mask", boolean[:]),
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

        self.storage_charge_eff = np.array(
            [phes_efficiency[0], bess_efficiency[0], bess_efficiency[0], bess_efficiency[0]]
        )
        self.storage_discha_eff = np.array(
            [phes_efficiency[1], bess_efficiency[1], bess_efficiency[1], bess_efficiency[1]]
        )

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
        self.TSpfix = np.atleast_2d(TSpfix[: self.intervals, scenario_mask])
        self.TSpsat = np.atleast_2d(TSpsat[: self.intervals, scenario_mask])
        self.TSonsw = np.atleast_2d(TSonsw[: self.intervals, scenario_mask])
        self.TSoffw = np.atleast_2d(TSoffw[: self.intervals, scenario_mask])
        self.TSror = np.atleast_2d(TSror[: self.intervals, scenario_mask])
        self.TSpond_inflow = np.atleast_2d(TSpond_inflow[: self.intervals, scenario_mask])
        self.TSphes_inflow = np.atleast_2d(TSphes_inflow[: self.intervals, scenario_mask])
        self.TShyd_inflow = np.atleast_2d(TShyd_inflow[: self.intervals, scenario_mask])

        self.EhydP = EhydP[scenario_mask]
        self.EhydE = EhydE[scenario_mask]
        self.Eror = Eror[scenario_mask]
        self.EphP = EphP[scenario_mask]
        self.EphE = EphE[scenario_mask]
        self.Enuke = Enuke[scenario_mask]
        self.Ebio = np.zeros(scenario_mask.sum(), dtype=np.float64)
        self.EpondP = EpondP[scenario_mask]
        self.EpondE = EpondE[scenario_mask]

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
        self.Rnlte_mask = Rnlte_mask[scenario_mask]

        self.nhvi = self.network_mask.sum()
        self.nodes = len(self.Nodel_int)

        self.energy = self.Mload.sum() * 1000 * self.resolution / self.years  # MWh p.a.

        self.lb = np.concat((
            np.zeros(self.nodes)[scenario_mask],  # pfix
            np.zeros(self.nodes)[scenario_mask],  # psat
            np.zeros(self.nodes)[scenario_mask],  # onsw
            np.zeros(self.nodes)[scenario_mask],  # offw
            np.zeros(self.Rnuke_mask.sum()),  # nuke
            np.zeros(self.Rnlte_mask.sum()),  # nuke LTE
            np.zeros(self.nodes)[scenario_mask],  # gas
            np.zeros(self.nodes)[scenario_mask],  # php
            np.zeros(self.nodes)[scenario_mask],  # phe
            np.zeros(self.nodes)[scenario_mask],  # b1p
            np.zeros(self.nodes)[scenario_mask],  # b2p
            np.zeros(self.nodes)[scenario_mask],  # b4p
            np.zeros(self.nhvi),  # lines
        ))
        self.ub = np.concat((
            Rpfix_st[scenario_mask],  # pfix
            Rpsat_st[scenario_mask],  # psat
            Ronsw_st[scenario_mask],  # onsw
            Roffw_st[scenario_mask],  # offw
            np.full(self.Rnuke_mask.sum(), Rnuke_const),  # nuke
            Rnlte[scenario_mask][self.Rnlte_mask],  # nuke LTE
            np.full(self.nodes, Rgas_const)[scenario_mask],  # gas
            Rphe[scenario_mask] / Rphh_min,  # php
            Rphe[scenario_mask],  # phe
            np.zeros(self.nodes)[scenario_mask],  # b1p
            np.zeros(self.nodes)[scenario_mask],  # b2p
            np.zeros(self.nodes)[scenario_mask],  # b4p
            Rlines[self.network_mask],  # lines
        ))

        mloadmax = np.array([array_max(col) for col in self.Mload.T])
        ave_demand = np.zeros(self.nodes)
        for i in range(self.nodes):
            ave_demand[i] = self.Mload[:, i].mean()

        self.x0 = np.concatenate(
            (
                ave_demand * 0.4 / np.array([col.mean() if col.mean() > 0 else 0 for col in self.TSpfix.T]),  # pfix
                ave_demand * 0.1 / np.array([col.mean() if col.mean() > 0 else 0 for col in self.TSpsat.T]),  # psat
                ave_demand * 0.5 / np.array([col.mean() if col.mean() > 0 else 0 for col in self.TSonsw.T]),  # onsw
                ave_demand * 0.25 / np.array([col.mean() if col.mean() > 0 else 0 for col in self.TSoffw.T]),  # offw
                np.zeros(self.Rnuke_mask.sum()),  # nuke
                Rnlte[scenario_mask][self.Rnlte_mask],  # nuke LTE
                ave_demand * 0.05,  # gas
                mloadmax * 0.5,
                mloadmax * 24,
                mloadmax * 0.1,
                mloadmax * 0.2,
                mloadmax * 0.2,
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
    ("Cpfix", float64[:]),
    ("Cpsat", float64[:]),
    ("Consw", float64[:]),
    ("Coffw", float64[:]),
    ("Cror", float64[:]),
    ("CpondP", float64[:]),
    ("CpondE", float64[:]),
    ("Cnuke", float64[:]),
    ("Cgas", float64[:]),
    ("CphP", float64[:]),
    ("CphE", float64[:]),
    ("Cb1P", float64[:]),
    ("Cb1E", float64[:]),
    ("Cb2P", float64[:]),
    ("Cb2E", float64[:]),
    ("Cb4P", float64[:]),
    ("Cb4E", float64[:]),
    ("CstorageP", float64[:, :]),
    ("CstorageE", float64[:, :]),
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
        self.Cpfix, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cpsat, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Consw, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Coffw, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cror, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cnuke, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cgas, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.CphP, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.CphE, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cb1P, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cb2P, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Cb4P, idx_start = get_next_slice(x, static.nodes, idx_start)
        self.Clines, idx_start = get_next_slice(x, static.nhvi, idx_start)

        self.Cb1E = self.Cb1P * 1.0
        self.Cb2E = self.Cb2P * 2.0
        self.Cb4E = self.Cb4P * 4.0

        self.Cbio = static.Ebio
        self.Cror += static.Eror
        self.Cror += static.Eror
        self.CphP += static.EphP
        self.CphE += static.EphE
        self.Cnuke += static.Enuke
        self.ChydP = static.EhydP
        self.ChydE = static.EhydE
        self.CpondP = static.EpondP
        self.CpondE = static.EpondE

        self.Cpeak = self.ChydP + self.CpondP + self.CphP + self.Cb1P + self.Cb2P + self.Cb4P
        self.CstorageP = np.stack((self.CphP, self.Cb1P, self.Cb2P, self.Cb4P))
        self.CstorageE = np.stack((self.CphE, self.Cb1E, self.Cb2E, self.Cb4E))


AssetDataType = AssetData.class_type.instance_type


operational_spec = [
    # Nodally disaggregated operations in GW/GWh
    ("Mgas", float64[:, :]),
    ("Mdischarge", float64[:, :, :]),
    ("Mcharge", float64[:, :, :]),
    ("Mstorage", float64[:, :, :]),
    ("Mdeficit", float64[:, :]),
    ("Mspillage", float64[:, :]),
    ("Mnetload", float64[:, :]),
    ("Mimport", float64[:, :]),
    ("Mpfix", float64[:, :]),
    ("Mpsat", float64[:, :]),
    ("Monsw", float64[:, :]),
    ("Moffw", float64[:, :]),
    ("Mload", float64[:, :]),
    ("Mreservoir", float64[:, :, :]),
    ("Mhydro", float64[:, :, :]),
    ("Mbio", float64[:, :]),
    ("Mgas", float64[:, :]),
    ("Munbalanced", float64[:, :]),
    ("Timport", float64[:, :, :]),
    ("Texport", float64[:, :, :]),

    ("precharge_flag", boolean[:, :]),
    ("trickling_flag", boolean[:, :]),
    ("charge_max_t", float64[:, :, :]),
    ("discharge_max_t", float64[:, :, :]),
    ("hydro_min_future", float64[:, :]),
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
            - assets.Cpfix * static.TSpfix
            - assets.Cpsat * static.TSpsat
            - assets.Consw * static.TSonsw
            - assets.Coffw * static.TSoffw
            - assets.Cror * static.TSror
        )
        self.Munbalanced = self.Mnetload.copy()
        self.Mdeficit = np.maximum(0, self.Mnetload)
        self.Mspillage = -np.minimum(0, self.Mnetload)

        self.Mgas = np.zeros((static.intervals, static.nodes), dtype=np.float64)
        self.Mgas = np.zeros((static.intervals, static.nodes), dtype=np.float64)

        self.Mdischarge = np.zeros((4, static.intervals, static.nodes), dtype=np.float64)
        self.Mcharge = np.zeros((4, static.intervals, static.nodes), dtype=np.float64)
        self.Mstorage = np.zeros((4, static.intervals, static.nodes), dtype=np.float64)

        self.Mstorage[0, -1, :] = 0.5 * assets.CphE
        self.Mstorage[1, -1, :] = 0.5 * assets.Cb4E
        self.Mstorage[2, -1, :] = 0.5 * assets.Cb2E
        self.Mstorage[3, -1, :] = 0.5 * assets.Cb1E

        self.Mhydro = np.zeros((2, static.intervals, static.nodes), dtype=np.float64)
        self.Mreservoir = np.zeros((2, static.intervals, static.nodes), dtype=np.float64)

        self.Mreservoir[0, -1, :] = 0.5 * assets.CpondE
        self.Mreservoir[1, -1, :] = 0.5 * assets.ChydE

        self.Timport = np.zeros((static.intervals, static.nodes, static.nhvi), dtype=np.float64)
        self.Texport = np.zeros((static.intervals, static.nodes, static.nhvi), dtype=np.float64)

        self.precharge_flag = np.zeros((4, static.nodes), dtype=np.bool_)
        self.trickling_flag = np.zeros((4, static.nodes), dtype=np.bool_)
        self.charge_max_t = np.zeros((4, static.intervals, static.nodes), dtype=np.float64)
        self.discharge_max_t = np.zeros((4, static.intervals, static.nodes), dtype=np.float64)
        self.hydro_min_future = np.zeros((2, static.nodes), dtype=np.float64)


OperationalDataType = OperationalData.class_type.instance_type


# Specify the types for jitclass
solution_spec = [
    ("x", float64[:]),
    ("static", StaticDataType),
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
            self.profile = ProfileData()


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
            S.assets.Cpfix.sum() * costs.pfix[0],
            S.assets.Consw.sum() * costs.onsw[0],
            S.assets.Coffw.sum() * costs.offw[0],
            S.assets.Cgas.sum() * costs.gas[0],
            S.assets.ChydP.sum() * costs.hydro[0],
            # generation fom
            S.assets.Cpfix.sum() * costs.pfix[1],
            S.assets.Consw.sum() * costs.onsw[1],
            S.assets.Cgas.sum() * costs.gas[1],
            S.assets.ChydP.sum() * costs.hydro[1],
            # generation vom
            S.operations.Mgas.sum() * S.static.resolution / S.static.years * costs.gas[2],
            # PHES storage
            S.assets.CphP.sum() * costs.phes[0],
            S.assets.CphE.sum() * costs.phes[1],
            S.assets.CphP.sum() * costs.phes[2],
            S.operations.Mdischarge[0].sum() * S.static.resolution / S.static.years * costs.phes[3],
            S.assets.CphP.sum() * costs.phes[4],
            # Battery storage (B4, B2, B1 aggregated)
            (S.assets.Cb1P.sum() + S.assets.Cb2P.sum() + S.assets.Cb4P.sum()) * costs.battery[0],
            (S.assets.Cb1E.sum() + S.assets.Cb2E.sum() + S.assets.Cb4E.sum()) * costs.battery[1],
            (S.assets.Cb1P.sum() + S.assets.Cb2P.sum() + S.assets.Cb4P.sum()) * costs.battery[2],
            (S.operations.Mdischarge[1:].sum()) * S.static.resolution / S.static.years * costs.battery[3],
        ]
        +
        # transmission network
        list(
            (
                S.assets.Cpfix.sum()
                + S.assets.Consw.sum()
                + S.assets.Coffw.sum()
                + S.assets.Cgas.sum()
                + S.assets.ChydP.sum()
                + S.assets.Cbio.sum()
                + S.assets.CphP.sum()
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
            S.operations.Mpfix.sum()
            + S.operations.Mpsat.sum()
            + S.operations.Monsw.sum()
            + S.operations.Moffw.sum()
            + S.operations.Mgas.sum()
        )
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
            S.operations.Mpfix.sum()
            + S.operations.Mpsat.sum()
            + S.operations.Monsw.sum()
            + S.operations.Moffw.sum()
            + S.operations.Mgas.sum()
            + S.operations.Mgas.sum()
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

    parameters = Parameters(s=1, y=1, p=0, n=3)
    static = StaticData(*parameters)
    costs = RawCosts(static).CostFactors()
    x0 = static.x0
    S = Solution(x0, static)
    Evaluate(S, costs)

    print("Lcoe:", S.Lcoe)
    print("Feasible:", S.Feasible)
    print("Penalties:", S.Penalties)
