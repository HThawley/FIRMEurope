
import numpy as np
import pandas as pd
from numba import boolean, prange, float64, int64, njit, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba.typed.typeddict import Dict as TypedDict

from firm.Simulation import Simulate
from firm.Utils import zero_safe_division, array_max, cclock
from firm.Network import generate_network
from firm.Costs import phes, annualization_phes, USD_to_AUD, USD_to_EUR, discount_rate

#%% 

MLoad = pd.read_csv("Data/load.csv").iloc[:, 1:] / 1000
assert MLoad.columns.nunique() == len(MLoad.columns), "Duplicated buses in load"
ECapacity = pd.read_csv("Data/existing_capacity.csv")
assert np.isin(ECapacity["bus"], MLoad.columns).all(), "ECapacity has extra buses"
ECapacity.index = ECapacity["bus"]
costs = pd.read_csv("Data/costs.csv", index_col=0)
Ecapex = 0.0
links = pd.read_csv("Data/links.csv")
lines = pd.read_csv("Data/lines.csv")
assert np.isin(links["bus0"], MLoad.columns).all(), "links has extra buses"
assert np.isin(links["bus1"], MLoad.columns).all(), "links has extra buses"
assert np.isin(lines["bus0"], MLoad.columns).all(), "lines has extra buses"
assert np.isin(lines["bus1"], MLoad.columns).all(), "lines has extra buses"
FOffw = pd.read_csv("Data/Offwind_costs.csv")
FOffw.index = FOffw["Generator"].str.split(" ").str[:2].str.join(" ")
assert np.isin(FOffw.index, MLoad.columns).all(), "Offwind costs has extra buses"
FOffw = FOffw.loc[FOffw["Generator"].str.contains("ac"), "capital_cost"]
assert FOffw.index.duplicated().sum() == 0, "Duplicated cost data in offwind"
FOffw = np.array([FOffw[node] if node in FOffw.index else 0 for node in MLoad.columns]) * 1000. #/MW to /GW
FGas = pd.read_csv("Data/gas_costs.csv", index_col=0)
assert np.isin(FGas.index, MLoad.columns).all(), "Gas has extra buses"
_fgas = FGas.mean().values[0]
FGas = np.array([FGas.loc[bus, "marginal_cost"].mean() if bus in FGas.index else _fgas for bus in MLoad.columns])

#%%
NNode = MLoad.columns.to_numpy()
NiNode = np.arange(len(NNode), dtype=np.int64)
N_Ni = dict(zip(NNode, NiNode))

#%%

def read_cf_file(file_name):
    TS_ = pd.read_csv(f"Data/{file_name}.csv").iloc[:, 1:]
    assert TS_.shape[0] == MLoad.shape[0], f"{file_name} covers different length time interval"
    N_ = TS_.columns.str.split(" ").str[:2].str.join(" ")
    assert np.isin(N_, NNode).all(), f"{file_name} has nodes not in NNode"
    return TS_.to_numpy(), N_.to_numpy()

TSRor, NRor = read_cf_file("ror_cf")
TSPvFix, NPvFix = read_cf_file("fixsolar_cf")
TSPvSat, NPvSat = read_cf_file("satsolar_cf")
TSOnsw, NOnsw = read_cf_file("onwind_cf")
TSOffw, NOffw = read_cf_file("offwind_cf")

RHydro = pd.read_csv("Data/inflow.csv").iloc[:, 1:]
RHydro.columns = RHydro.columns.str.split(" ").str[:2].str.join(" ")
assert RHydro.shape[0] == MLoad.shape[0], "RHydro covers different length time interval"
RHydro = np.stack([RHydro[node] if node in RHydro.columns else np.zeros(len(RHydro)) 
                   for node in MLoad.columns]) 
RHydro /= 1000 # MW to GW
#TODO: Better inflow modelling
RHydro = RHydro.sum(axis=1)

#%%

NNiPvFix, NiPvSat, NiOnsw, NiOffw, NiRor = (np.array([N_Ni[node] for node in N_], dtype=np.int64) 
                                            for N_ in (NPvFix, NPvSat, NOnsw, NOffw, NRor))

#%%
ERor = ECapacity.loc[ECapacity["ror"] > 0, ["ror"]]

# We do this here as we do not consider capacity expansion of ROR
MRor = pd.DataFrame(0.0, index = MLoad.index, columns = MLoad.columns)
MRor.loc[:, ERor.index] = pd.DataFrame(ERor["ror"].to_numpy() * TSRor / 1000, # MW to GW
                                       columns = ERor.index)
ERor = np.array([ERor.loc[bus,"ror"] if bus in ERor.index else 0.0 for bus in MLoad.columns])

MLoad = MLoad.to_numpy()
MRor = MRor.to_numpy()

#%%

EPhse = ECapacity.loc[ECapacity["phsp"] > 0, ["phsp", "phse"]]
EPhsp = np.array([EPhse.loc[node, "phsp"] if node in EPhse.index else 0.0 for node in NNode]) / 1000 # MW to GW
EPhse = np.array([EPhse.loc[node, "phse"] if node in EPhse.index else 0.0 for node in NNode]) / 1000 # MW to GW

EHyde = ECapacity.loc[ECapacity["hydrop"] > 0, ["hydrop", "hydroe"]]
EHydp = np.array([EHyde.loc[node, "hydrop"] if node in EHyde.index else 0 for node in NNode]) / 1000 # MW to GW
EHyde = np.array([EHyde.loc[node, "hydroe"] if node in EHyde.index else 0 for node in NNode]) / 1000 # MW to GW

def get_E_(name):
    E_ = ECapacity.loc[ECapacity[name] > 0, [name]]
    E_ = np.array([E_.loc[node, name] if node in E_.index else 0 for node in NNode]) / 1000 # MW to GW
    return E_ 

ECcgt = get_E_("CCGT")
EOcgt = get_E_("OCGT")
ENuke = get_E_("nuclear")
EBio = get_E_("biomass")

#%%

lines[["bus0", "bus1"]] = lines[["bus0", "bus1"]].map(lambda bus: N_Ni[bus])
links[["bus0", "bus1"]] = links[["bus0", "bus1"]].map(lambda bus: N_Ni[bus])

Hvi = pd.concat((links, lines))
Hvi["p_nom_tot"] = Hvi["s_nom"] + Hvi["p_nom"]
hv_index = Hvi.groupby(["bus0", "bus1"])[["p_nom", "s_nom"]].sum().index
EHVDC = Hvi.groupby(["bus0", "bus1"])["p_nom"].sum()[hv_index].to_numpy() / 1000. # MW to GW
EHVAC = Hvi.groupby(["bus0", "bus1"])["s_nom"].sum()[hv_index].to_numpy() / 1000. # MW to GW
EHvi  = Hvi.groupby(["bus0", "bus1"])["p_nom_tot"].sum()[hv_index].to_numpy() / 1000. # MW to GW

FHvi = Hvi.groupby(["bus0", "bus1"])["capital_cost"].min()[hv_index].to_numpy() * 1000. #/MW to /GW # build cheaper of AC/DC

# Hvi["loss"] = Hvi["length"] * 0.05 * 0.001 #5% per 1000 km
## TODO: improve this assumption
# LHvi = Hvi.groupby(["bus0", "bus1"])["loss"].mean()[Hvi_index].to_numpy()
network = np.array([[*conn] for conn in hv_index], dtype=np.int64)

#%%

FECapex = np.array([costs.loc[tech, "capital_cost"] for tech in 
                    ("ror", "PHS", "hydro", "OCGT", "CCGT", "nuclear", 
                     "biomass", "DC_link", "AC_line")]) * 1000. # /MW to /GW

FEOpex = np.array([costs.loc[tech, "marginal_cost"] for tech in 
                   ("biomass", "nuclear")]) * 1000 #/MWh to /GWh

phes = tuple(p/USD_to_AUD*USD_to_EUR for p in phes[0:5]) + phes[5:]
FPhes = annualization_phes(*phes, discount_rate)

FCCapex = np.array([costs.loc[tech, "capital_cost"] for tech in 
                   ("fixsolar", "satsolar", "onwind", "OCGT")
                   ]) * 1000. #MW to /GW

#%%

# =============================================================================
# read costs and create cost structure
# read phes efficiency and hydro efficiency
# =============================================================================

# =============================================================================
# Calculate capex of all existing infrastructure
# 
# ror
# hydro 
# phes
# ocgt
# ccgt
# aclines
# dclinks
# =============================================================================

#%%

data_spec=[
    ("scenario", int64),
    ("profiling", boolean),
    ("resolution", float64),
    ("efficiency", float64),
    ("networksteps", int64),
    ("years", int64),

    ("intervals", int64),
    
    ("NiCoverage", int64[:]),
    ("NiNode", int64[:]),

    ("network", int64[:, :]),
    ("network_mask", boolean[:]),
    ("trans_mask", boolean[:, :]),
    ("cache_0_donors", types.DictType(int64, int64[:, :])),
    ("cache_n_donors", types.DictType(types.UniTuple(int64, 2), int64[:, :, :])),

    ("nodes", int64),
    ("nhvi", int64),
    
    ("MLoad", float64[:, :]),
    ("MRor", float64[:, :]),
    ("RHydro", float64[:]),
    # ("RHydro", float64[:, :]), # TODO: better inflow modelling

    ("TSPvFix", float64[:, :]),
    ("TSPvSat", float64[:, :]),
    ("TSOnsw", float64[:, :]),
    ("TSOffw", float64[:, :]),
    
    ("ERor", float64[:]),
    ("EPhsp", float64[:]),
    ("EPhse", float64[:]),
    ("EHydp", float64[:]),
    ("EHyde", float64[:]),
    ("ECcgt", float64[:]),
    ("EOcgt", float64[:]),
    ("EGas", float64[:]),
    ("ENuke", float64[:]),
    ("EBio", float64[:]),
    ("EFlex", float64[:]),
    ("EHvi", float64[:]),
    ("EHVDC", float64[:]),
    ("EHVAC", float64[:]),
    
    ("energy", float64),

    ("lb", float64[:]),
    ("ub", float64[:]),
    ("x0", float64[:]),
    
    ("FECapex", float64), 
    ("FEOpex", float64[:]), 
    ("FCCapex", float64[:]), 
    ("FPhes", float64[:]), 
    ("FOffw", float64[:]),
    ("FGas", float64[:]),
    ("FHvi", float64[:]),
    ]

@jitclass(data_spec)
class SolutionData:
    def __init__(
            self, 
            scenario: int, 
            years: int,
            profiling: bool,
            networksteps: int
            ):
        self.scenario = scenario
        self.profiling = profiling
        self.resolution = 1
        self.efficiency = 0.8
        self.networksteps = networksteps
        self.FECapex=0.0
        
        maxyears = int(self.resolution * len(MLoad) / 8760) 
        if years == -1:
            self.years = maxyears
        elif years <= maxyears:
            self.years = years
        else: 
            raise Exception
        self.intervals = int(self.years * 8760 / self.resolution)
        
        # placeholder for NiCoverage(scenario)
        self.NiNode = NiNode#[self.NiCoverage]

        with objmode():
            (self.network, 
             self.network_mask, 
             self.trans_mask, 
             self.cache_0_donors,
             self.cache_n_donors, 
            ) = generate_network(network, self.NiNode, self.networksteps)
            
        self.nodes = len(self.NiNode)
        self.nhvi = self.network_mask.sum()

        self.MLoad = MLoad[: self.intervals, self.NiNode]
        self.MRor  = MRor[ : self.intervals, self.NiNode]
        self.RHydro = RHydro[self.NiNode]        
        # self.RHydro = RHydro[: self.intervals, self.NiNode]        

        self.TSPvFix = TSPvFix[: self.intervals, self.NiNode]
        self.TSPvSat = TSPvSat[: self.intervals, self.NiNode]
        self.TSOnsw  = TSOnsw[ : self.intervals, self.NiNode]
        self.TSOffw  = TSOffw[ : self.intervals, self.NiNode]

        self.ERor  = ERor[self.NiNode]
        self.EPhsp = EPhsp[self.NiNode]
        self.EPhse = EPhse[self.NiNode]
        self.EHydp = EHydp[self.NiNode]
        self.EHyde = EHyde[self.NiNode]
        self.ECcgt = ECcgt[self.NiNode]
        self.EOcgt = EOcgt[self.NiNode]
        self.ENuke = ENuke[self.NiNode]
        self.EBio  = EBio[ self.NiNode]
        
        self.EGas = self.ECcgt + self.EOcgt
        self.EFlex = self.ECcgt + self.EOcgt + self.EBio

        self.EHvi  = EHvi[ self.network_mask]
        self.EHVDC = EHVDC[self.network_mask]
        self.EHVAC = EHVAC[self.network_mask]
        
        self.FECapex += self.ERor.sum()  * FECapex[0]
        self.FECapex += self.EPhsp.sum() * FECapex[1]
        self.FECapex += self.EHydp.sum() * FECapex[2]
        self.FECapex += self.EOcgt.sum() * FECapex[3]
        self.FECapex += self.ECcgt.sum() * FECapex[4]
        self.FECapex += self.ENuke.sum() * FECapex[5]
        self.FECapex += self.EBio.sum()  * FECapex[6]
        
        self.FECapex += self.EHVDC.sum() * FECapex[7]
        self.FECapex += self.EHVAC.sum() * FECapex[8]
        
        self.FEOpex = FEOpex
        self.FCCapex = FCCapex
        self.FPhes = FPhes
        self.FOffw = FOffw[self.NiNode]
        self.FGas = FGas[self.NiNode]
        self.FHvi = FHvi[self.network_mask]
        
        self.energy = self.MLoad.sum() * 1000 * self.resolution / self.years  # MWh p.a.
        
        self.lb = np.array(
            [0.0] * self.nodes + #pvfix
            [0.0] * self.nodes + #pvsat
            [0.0] * self.nodes + #onsw
            [0.0] * self.nodes + #offw
            [0.0] * self.nodes + #ocgt
            [0.0] * self.nodes + #phsp
            [0.0] * self.nodes + #phse
            [0.0] * self.nhvi
            )
        
        self.ub = np.array(
            [32.0] * self.nodes + #pvfix
            [32.0] * self.nodes + #pvsat
            [32.0] * self.nodes + #onsw
            [32.0] * self.nodes + #offw
            [32.0] * self.nodes + #ocgt
            [32.0] * self.nodes + #phsp
            [3200.0] * self.nodes + #phse
            [32.0]  * self.nhvi
            )
        
        _t0 = np.array([col.mean() for col in self.TSPvFix.T])
        _t1 = np.array([col.mean() for col in self.TSPvSat.T])
        _t2 = np.array([col.mean() for col in self.TSOnsw.T])
        _t3 = np.array([col.mean() for col in self.TSOffw.T])
        mloadmax = np.array([array_max(col) for col in self.MLoad.T])
        self.x0 = np.concatenate(
            (
                self.MLoad.sum() / self.intervals * 0.3 / self.nodes / _t0,
                self.MLoad.sum() / self.intervals * 0.3 / self.nodes / _t1,
                self.MLoad.sum() / self.intervals * 0.3 / self.nodes / _t2,
                self.MLoad.sum() / self.intervals * 0.3 / self.nodes / _t3,
                mloadmax * 0.25,
                mloadmax * 1,
                mloadmax * 48,
                np.repeat(array_max(mloadmax) * 0.4, self.nhvi),
            )
        )
        self.x0 = np.clip(self.x0, self.lb, self.ub)
        
    
#%%

# Specify the types for jitclass
solution_spec = [
    ("x", float64[:]),
    ("scenario", int64),
    ("intervals", int64),
    ("nodes", int64),
    ("nhvi", int64),
    ("resolution", float64),
    ("years", int64),
    ("efficiency", float64),
    ("energy", float64),
    ("NiNode", int64[:]),
    # Topology
    ("network_mask", boolean[:]),
    ("network", int64[:, :]),
    ("networksteps", int64),
    ("cache_0_donors", types.DictType(int64, int64[:, :])),
    ("cache_n_donors", types.DictType(types.UniTuple(int64, 2), int64[:, :, :])),
    # Capacity expansion in GW/GWh
    ("CPvFix", float64[:]),
    ("CPvSat", float64[:]),
    ("COnsw", float64[:]),
    ("COffw", float64[:]),
    ("COcgt", float64[:]),
    ("CPhsp", float64[:]),
    ("CPhse", float64[:]),
    ("CHvi", float64[:]),
    # Existing capacites in GW/GWh
    ("EPhsp", float64[:]),
    ("EPhse", float64[:]),
    ("EHydp", float64[:]),
    # ("EHyde", float64[:]), # TODO: better inflow modelling
    ("EFlex", float64[:]), # EOCgt + ECcgt + EBio
    ("EGas", float64[:]), # EOCgt + ECcgt + EBio
    ("EBio", float64[:]), # EOCgt + ECcgt + EBio
    ("ENuke", float64[:]),
    ("EHvi", float64[:]),
    # Cost factors
    ("FECapex", float64), 
    ("FEOpex", float64[:]), 
    ("FCCapex", float64[:]), 
    ("FPhes", float64[:]), 
    ("FOffw", float64[:]),
    ("FGas", float64[:]),
    ("FHvi", float64[:]),
    # Existing + Expanded capacity (only where both are used)
    ("GPhsp", float64[:]),
    ("GPhse", float64[:]),
    ("GFlex", float64[:]), # EOCgt + ECcgt + EBio + COcgt
    ("GHvi", float64[:]),
    # Nodally disaggregated Hydro resource     
    ("RHydro", float64[:]), # for fast maths, kept in units such that (MHydro.sum(axis=0)<=RHydro).all()
    # Nodally diaggregated operations in GW/GWh
    ("MLoad", float64[:, :]),
    ("MNetload", float64[:, :]),
    ("MUnbalanced", float64[:,:]),
    ("MDeficit", float64[:, :]),
    ("MSpillage", float64[:, :]),
    ("MImport", float64[:, :]),
    ("MRor", float64[:, :]),
    ("MHydro", float64[:, :]),
    ("MPvFix", float64[:, :]),
    ("MPvSat", float64[:, :]),
    ("MOnsw", float64[:, :]),
    ("MOffw", float64[:, :]),
    ("MFlex", float64[:, :]),
    ("MDischarge", float64[:, :]),
    ("MCharge", float64[:, :]),
    ("MStorage", float64[:, :]),
    # Transmission
    ("THvi", float64[:, :]),
    ("Topology", float64[:, :]),
    ("trans_mask", boolean[:, :]),
    ("TImport", float64[:, :, :]),
    ("TExport", float64[:, :, :]),
    #Objectives
    ("Penalties", float64),
    ("LCOE", float64),
    ("LCOG", float64),
    ("LCOB", float64),
    ("LCOSP", float64),
    ("LCOSB", float64),
    ("LCOBS", float64),
    ("LCOBT", float64),
    ("LCOBL", float64),
    ("CAPEX", float64),
    ("OPEX", float64),
    # Profiling
    ("profiling", boolean),
    ("profile_overhead", float64),
    # time profiling
    ("time_transmission", int64),
    ("time_backfill", int64),
    ("time_basic", int64),
    ("time_interconnection0", int64),
    ("time_interconnection1", int64),
    ("time_interconnection2", int64),
    ("time_interconnection3", int64),
    ("time_storage_behavior", int64),
    ("time_storage_behaviort", int64),
    ("time_spilldef", int64),
    ("time_spilldeft", int64),
    ("time_update_soc", int64),
    ("time_update_soct", int64),
    ("time_unbalancedt", int64),
    ("time_unbalanced", int64),
    
    ("calls_transmission", int64),
    ("calls_backfill", int64),
    ("calls_basic", int64),
    ("calls_interconnection0", int64),
    ("calls_interconnection1", int64),
    ("calls_interconnection2", int64),
    ("calls_interconnection3", int64),
    ("calls_storage_behavior", int64),
    ("calls_storage_behaviort", int64),
    ("calls_spilldef", int64),
    ("calls_spilldeft", int64),
    ("calls_update_soc", int64),
    ("calls_update_soct", int64),
    ("calls_unbalancedt", int64),
    ("calls_unbalanced", int64),
]
@jitclass(solution_spec)
class Solution:
    def __init__(
            self, 
            x: np.ndarray, 
            sd: SolutionData
            ):
        assert len(x) == len(sd.lb)

        self.x = x

        self.scenario = sd.scenario
        self.nodes = sd.nodes
        self.nhvi = sd.nhvi

        self.resolution = sd.resolution
        self.efficiency = sd.efficiency
        self.years = sd.years
        self.intervals = sd.intervals
        self.energy = sd.energy
        
        self.NiNode = sd.NiNode
        self.network_mask = sd.network_mask
        self.network = sd.network
        self.networksteps = sd.networksteps
        self.trans_mask = sd.trans_mask
        self.cache_0_donors = sd.cache_0_donors
        self.cache_n_donors = sd.cache_n_donors
        
        self.CPvFix = x[              : self.nodes]
        self.CPvSat = x[self.nodes * 1: self.nodes * 2]
        self.COnsw  = x[self.nodes * 2: self.nodes * 3]
        self.COffw  = x[self.nodes * 3: self.nodes * 4]
        self.COcgt  = x[self.nodes * 4: self.nodes * 5]
        self.CPhsp  = x[self.nodes * 5: self.nodes * 6]
        self.CPhse  = x[self.nodes * 6: self.nodes * 7]
        self.CHvi   = x[self.nodes * 7: ]
        
        self.EPhsp = sd.EPhsp
        self.EPhse = sd.EPhse
        self.EHydp = sd.EHydp
        # self.EHyde = sd.EHyde # TODO: better inflow modelling
        self.EFlex = sd.EFlex
        self.EGas = sd.EGas
        self.EBio = sd.EBio
        self.ENuke = sd.ENuke
        self.EHvi = sd.EHvi
        
        self.GFlex = self.COcgt + self.EFlex
        self.GHvi = self.CHvi + self.EHvi
        self.GPhsp = self.CPhsp + sd.EPhsp
        self.GPhse = self.CPhse + sd.EPhse
        
        self.FECapex = sd.FECapex
        self.FEOpex = sd.FEOpex
        self.FCCapex = sd.FCCapex
        self.FPhes = sd.FPhes
        self.FOffw = sd.FOffw
        self.FGas = sd.FGas
        self.FHvi = sd.FHvi

        self.RHydro = sd.RHydro * self.years # units such that (MHydro.sum(axis=0)<=RHydro).all()

        self.MLoad = sd.MLoad
        self.MRor = sd.MRor
        
        self.MPvFix = sd.TSPvFix * self.CPvFix
        self.MPvSat = sd.TSPvSat * self.CPvSat
        self.MOnsw  = sd.TSOnsw  * self.COnsw
        self.MOffw  = sd.TSOffw  * self.COffw

        self.profile_overhead=0.0
        self.profiling = sd.profiling
        if self.profiling:
            self.time_transmission = 0
            self.time_backfill = 0
            self.time_basic = 0
            self.time_interconnection0 = 0
            self.time_interconnection1 = 0
            self.time_interconnection2 = 0
            self.time_interconnection3 = 0
            self.time_storage_behavior = 0
            self.time_storage_behaviort = 0
            self.time_spilldef = 0
            self.time_spilldeft = 0
            self.time_update_soc = 0
            self.time_update_soct = 0
            self.time_unbalancedt = 0
            self.time_unbalanced = 0
            
            self.calls_transmission = 0
            self.calls_backfill = 0
            self.calls_basic = 0
            self.calls_interconnection0 = 0
            self.calls_interconnection1 = 0
            self.calls_interconnection2 = 0
            self.calls_interconnection3 = 0
            self.calls_storage_behavior = 0
            self.calls_storage_behaviort = 0
            self.calls_spilldef = 0
            self.calls_spilldeft = 0
            self.calls_update_soc = 0
            self.calls_update_soct = 0
            self.calls_unbalancedt = 0
            self.calls_unbalanced = 0

            self.profile_overhead = 0.0
            for _ in range(10000):
                start=cclock()
                self.calls_transmission += 1
                self.profile_overhead += cclock()-start
            self.calls_transmission += 1
            self.profile_overhead/=10000
        
#%% 

@njit
def Evaluate(solution):
    Simulate(solution)

    solution.Penalties = np.maximum(0, solution.MDeficit.sum())*1000  # MWh/resolution

    capex = 0.0
    capex += solution.CPvFix.sum() * solution.FCCapex[0]
    capex += solution.CPvSat.sum() * solution.FCCapex[1]
    capex += solution.COnsw.sum()  * solution.FCCapex[2]
    capex += (solution.COffw * solution.FOffw).sum()
    capex += solution.COcgt.sum()  * solution.FCCapex[3]
    capex += solution.CPhsp.sum() * solution.FPhes[0]
    capex += solution.CPhsp.sum() * solution.FPhes[2]
    capex += solution.CPhse.sum() * solution.FPhes[1]
    capex += (solution.CHvi * solution.FHvi).sum()
    capex += solution.FECapex
    
# =============================================================================
# Similar logic for existing CCGT/OCGT
# =============================================================================
    FuelOpex = 0.0    

    for t in range(solution.intervals):
        for n in range(solution.nodes):
            # use existing biomass plants first
            FuelOpex += min(solution.EBio[n], solution.MFlex[t, n]) * solution.FEOpex[0] 
            # use existing gas plants second
            FuelOpex += min(solution.EGas[n], max(0, solution.MFlex[t, n] - solution.EBio[n])) * solution.FGas[n] 
            # use new gas last
            FuelOpex += max(0, solution.MFlex[t, n] - solution.EGas[n] - EBio[n]) * solution.FGas[n]
    FuelOpex += solution.ENuke.sum() * solution.intervals * solution.FEOpex[1]
    FuelOpex *= solution.resolution / solution.years # -> GWh p.a.
    

    # Levelised Costs of:
    # Electricity
    solution.LCOE = (capex + FuelOpex)/ solution.energy
    
# =============================================================================
#     LCOx are low priority to fix
# =============================================================================
    # Generation
    # solution.LCOG = cost[:10].sum() / (
    #     1000 * solution.resolution / solution.years * (
    #         solution.MPv.sum()
    #         + solution.MWind.sum()
    #         + solution.MGas.sum()
    #         + solution.MHydro.sum()
    #     )
    # )
    # # Storage
    # # solution.LCOSP = zero_safe_division(cost[10:15].sum(), solution.MDischarge.sum()*solution.resolution/solution.years)
    # # Balancing - Storage
    # solution.LCOBS = cost[10:15].sum() / solution.energy
    # # Balancing - Transmission
    # solution.LCOBT = cost[15:].sum() / solution.energy
    # # Balancing - Spillage
    # solution.LCOBL = solution.LCOE - solution.LCOG - solution.LCOBS - solution.LCOBT
    # solution.LCOB = solution.LCOBS + solution.LCOBT + solution.LCOBL
    # solution.CAPEX = capex.sum()
    # solution.OPEX = opex.sum()

    return solution.LCOE, solution.Penalties

@njit
def EnergyMix(solution):
    PPvFix = solution.MPvFix.sum()
    PPvSat = solution.MPvSat.sum()
    PPv = PPvFix + PPvSat
    POnsw = solution.MOnsw.sum()
    POffw = solution.MOffw.sum()
    PWind = POnsw + POffw
    PRor = solution.MRor.sum()
    PHyd = solution.MHyd.sum()
    PNuke = solution.ENuke.sum() * solution.intervals 
    PBio = 0.0
    PEGas = 0.0 
    PCGas = 0.0 
    for t in range(solution.intervals):
        for n in range(solution.nodes):
            # use existing biomass plants first
            PBio += min(solution.EBio[n], solution.MFlex[t, n])
            # use existing gas plants second
            PEGas += min(solution.EGas[n], max(0, solution.MFlex[t, n] - solution.EBio[n])) 
            # use new gas last
            PCGas += max(0, solution.MFlex[t, n] - solution.EGas[n] - EBio[n])
    PGas = PEGas + PCGas
    
    Demand = MLoad.sum()
    
    
    
    