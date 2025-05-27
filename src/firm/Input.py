
import numpy as np
import pandas as pd
from numba import boolean, prange, float64, int64, njit, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba.typed.typeddict import Dict as TypedDict

from firm.Simulation import Simulate
from firm.Utils import zero_safe_division, array_max, cclock
from firm.Network import generate_network

Nodel = np.genfromtxt("Data/load.csv", delimiter=",", max_rows=1, dtype=str)[1:]
n_node = dict((name, i) for i, name in enumerate(Nodel))
MLoad = np.genfromtxt("Data/load.csv", delimiter=",", skip_header=1, usecols=range(1, 1+len(Nodel))) 
MLoad /= 1000  # MW to GW

PVl = np.genfromtxt("Data/solar_cf.csv", delimiter=",", max_rows=1, dtype=str)
PVl = np.array([' '.join(x.split(' ')[:2]) for x in PVl[1:]], dtype=str)
TSPV = np.genfromtxt("Data/solar_cf.csv", delimiter=",", skip_header=1, usecols=range(1, 1 + len(PVl)))

Windl = np.genfromtxt("Data/wind_cf.csv", delimiter=",", max_rows=1, dtype=str)
Windl = np.array([' '.join(x.split(' ')[:2]) for x in Windl[1:]], dtype=str)
TSWind = np.genfromtxt("Data/wind_cf.csv", delimiter=",", skip_header=1, usecols=range(1, 1 + len(Windl)))

CRor = dict(np.genfromtxt("Data/ror_plants.csv", delimiter=",", skip_header=1, dtype=None, usecols=[1,2]))
Rorl = np.genfromtxt("Data/ror_cf.csv", delimiter=",", max_rows=1, dtype=str)
Rorl = np.array([' '.join(x.split(' ')[:2]) for x in Rorl[1:]], dtype=str)
TSRor = np.genfromtxt("Data/ror_cf.csv", delimiter=",", skip_header=1, usecols=range(1, 1 + len(Rorl)))
MRor= np.zeros(MLoad.shape)
MRor[:, np.isin(Nodel, Rorl, True)] = TSRor * np.array([CRor[node] for node in Rorl])

Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

CHydro = dict(np.genfromtxt("Data/hydro_plants.csv", delimiter=",", skip_header=1, dtype=None, usecols=[1,2]))
CHydro = np.array([CHydro[node] if node in CHydro.keys() else 0 for node in Nodel]) / 1000.

RHydro = pd.read_csv("Data/inflow.csv", sep=',')
RHydro = RHydro.drop(columns=['snapshot'])
RHydro = RHydro.sum() 
RHydro.index = [' '.join(x.split(' ')[:2]) for x in RHydro.index]
RHydro = np.array([RHydro[node] if node in RHydro.index else 0 for node in Nodel]) / 1000

CGas = pd.read_csv("Data/gas.csv", sep=",", usecols=[1,2])
CGas = CGas.groupby('bus')['p_nom'].sum()
CGas = np.array([CGas[node] if node in CGas.keys() else 0 for node in Nodel]) / 1000. 

Net = pd.concat((
    pd.read_csv("Data/links.csv", usecols=range(1,6), delimiter=","), 
    pd.read_csv("Data/lines.csv", usecols=range(1,6), delimiter=",")
    ))
Net['p_nom'] = Net[['p_nom', 's_nom']].apply(lambda row: row['p_nom'] if not pd.isna(row['p_nom']) else row['s_nom'], axis=1)
Net[['bus0', 'bus1']] = Net[['bus0', 'bus1']].map(lambda bus: n_node[bus])
Net['loss'] = Net['length'] * 0.05 * 0.001 #5% per 1000 km
CNet = Net.groupby(['bus0', 'bus1'])['p_nom'].sum().to_numpy()
LNet = Net.groupby(['bus0', 'bus1'])['loss'].mean().to_numpy()
BNet = Net[['bus0', 'bus1']].drop_duplicates().to_numpy()
NetCost = Net.groupby(['bus0', 'bus1'])['capital_cost'].mean().to_numpy()
NetLength = Net.groupby(['bus0', 'bus1'])['length'].mean().to_numpy()
del Net

#%%

data_spec=[
    ("scenario", int64),
    ("profiling", boolean),
    ("resolution", float64),
    ("efficiency", float64),
    ("years", int64),
    ("intervals", int64),
    ("MLoad", float64[:, :]),
    ("MRor", float64[:, :]),
    ("RHydro", float64[:]),
    ("TSPV", float64[:, :]),
    ("TSWind", float64[:, :]),
    ("CHydro", float64[:]),
    ("CGas", float64[:]),
    ("CHVI", float64[:]),
    ("CPeak", float64[:]),
    ("coverage_int", int64[:]),
    ("Nodel_int", int64[:]),
    ("PVl_int", int64[:]),
    ("Windl_int", int64[:]),
    ("network", int64[:, :]),
    ("network_mask", boolean[:]),
    ("networksteps", int64),
    ("trans_mask", boolean[:, :]),
    
    ("cache_0_donors", types.DictType(int64, int64[:, :])),
    ("cache_n_donors", types.DictType(types.UniTuple(int64, 2), int64[:, :, :])),
    
    ("nhvi", int64),
    ("nodes", int64),
    ("pzones", int64),
    ("wzones", int64),
    ("pidx", int64),
    ("widx", int64),
    ("gidx", int64),
    ("spidx", int64),
    ("seidx", int64),
    ("energy", float64),
    ("lb", float64[:]),
    ("ub", float64[:]),
    ("x0", float64[:]),
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
        
        maxyears = int(self.resolution * len(MLoad) / 8760) 
        if years == -1:
            self.years = maxyears
        elif years <= maxyears:
            self.years = years
        else: 
            raise Exception
        self.intervals = int(self.years * 8760 / self.resolution)
        
        # Retain flexibility to define scenarios later
        self.coverage_int = Nodel_int.copy()
        
        self.MLoad =  MLoad[: self.intervals,  np.isin(Nodel_int, self.coverage_int)]
        self.MRor  =  MRor[: self.intervals,  np.isin(Nodel_int, self.coverage_int)]
        self.TSPV =   TSPV[: self.intervals,   np.isin(PVl_int,   self.coverage_int)]
        self.TSWind = TSWind[: self.intervals, np.isin(Windl_int, self.coverage_int)]
        
        self.CHydro =    CHydro[   np.isin(Nodel_int, self.coverage_int)]
        self.CGas =      CGas[     np.isin(Nodel_int, self.coverage_int)] 
        self.RHydro =    RHydro[   np.isin(Nodel_int, self.coverage_int)]

        self.Nodel_int = Nodel_int[np.isin(Nodel_int, self.coverage_int)]
        self.PVl_int =   PVl_int[  np.isin(PVl_int,   self.coverage_int)]
        self.Windl_int = Windl_int[np.isin(Windl_int, self.coverage_int)]

        with objmode():
            (self.network, 
             self.network_mask, 
             self.trans_mask, 
             self.cache_0_donors,
             self.cache_n_donors, 
            ) = generate_network(BNet, self.Nodel_int, self.networksteps)
            
        self.nhvi = self.network_mask.sum()
        self.CHVI = CNet[self.network_mask]
        
        self.nodes = len(self.Nodel_int)
        
        self.pzones = len(self.PVl_int)
        self.wzones = len(self.Windl_int)
        self.pidx = self.pzones
        self.widx = self.pidx + self.wzones
        self.gidx = self.widx + self.nodes
        self.spidx = self.gidx + self.nodes
        self.seidx = self.spidx + self.nodes
        
        self.energy = self.MLoad.sum() * 1000 * self.resolution / self.years  # MWh p.a.
        
        self.lb = np.array(
            [0.0] * self.pzones + 
            [0.0] * self.wzones + 
            list(self.CGas) + 
            [0.0] * self.nodes + 
            [0.0] * self.nodes + 
            list(self.CHVI)
            )
        self.ub = np.array(
            [24.0]  * self.pzones + 
            [24.0]  * self.wzones + 
            [24.0]  * self.nodes + 
            [24.0]  * self.nodes + 
            [600.0] * self.nodes + 
            [20.0]  * self.nhvi
            )
        
        tspvmean = np.array([col.mean() for col in self.TSPV.T])
        tswindmean = np.array([col.mean() for col in self.TSWind.T])
        mloadmax = np.array([array_max(col) for col in self.MLoad.T])
        self.x0 = np.concatenate(
            (
                self.MLoad.sum() / self.intervals * 0.70 / self.pzones / tspvmean,
                self.MLoad.sum() / self.intervals * 0.70 / self.wzones / tswindmean,
                mloadmax * 0.25,
                mloadmax * 1,
                mloadmax * 36,
                np.repeat(array_max(mloadmax) * 0.6, self.nhvi),
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
    ("Nodel_int", int64[:]),
    # ('PVl_int', int64[:]),
    # ('Windl_int', int64[:]),
    # Topology
    ("network_mask", boolean[:]),
    ("network", int64[:, :]),
    ("networksteps", int64),
    ("cache_0_donors", types.DictType(int64, int64[:, :])),
    ("cache_n_donors", types.DictType(types.UniTuple(int64, 2), int64[:, :, :])),
    # Capacities in GW/GWh
    ("CPV", float64[:]),
    ("CWind", float64[:]),
    ("CPHP", float64[:]),
    ("CPHS", float64[:]),
    ("CHVI", float64[:]),
    ("CPeak", float64[:]),
    ("CHydro", float64[:]),
    ("CGas", float64[:]),
    # Nodally disaggregated Hydro resource     
    ("RHydro", float64[:]), # for fast maths, kept in units such that (MHydro.sum(axis=0)<=RHydro).all()
    # Nodally diaggregated operations in GW/GWh
    ("MDischarge", float64[:, :]),
    ("MCharge", float64[:, :]),
    ("MStorage", float64[:, :]),
    ("MDeficit", float64[:, :]),
    ("MSpillage", float64[:, :]),
    ("MNetload", float64[:, :]),
    ("MImport", float64[:, :]),
    ("MPV", float64[:, :]),
    ("MWind", float64[:, :]),
    ("MLoad", float64[:, :]),
    ("MRor", float64[:, :]),
    ("MHydro", float64[:, :]),
    ("MGas", float64[:, :]),
    ("MUnbalanced", float64[:,:]),
    # Transmission
    ("THVI", float64[:, :]),
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
    ('profile_overhead', float64),
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
        self.resolution = sd.resolution
        self.efficiency = sd.efficiency
        self.years = sd.years
        self.intervals = sd.intervals
        self.energy = sd.energy
        
        self.Nodel_int = sd.Nodel_int
        # self.PVl_int, self.Windl_int = sd.PVl_int, sd.Windl_int
        self.network_mask = sd.network_mask
        self.network = sd.network
        self.networksteps = sd.networksteps
        self.trans_mask = sd.trans_mask

        self.nhvi = self.network_mask.sum()

        self.CPV =   x[        : sd.pidx]
        self.CWind = x[sd.pidx : sd.widx]
        self.CGas =  x[sd.widx : sd.gidx]
        self.CPHP =  x[sd.gidx : sd.spidx]
        self.CPHS =  x[sd.spidx: sd.seidx]
        self.CHVI =  x[sd.seidx: ]
        self.CHydro =    sd.CHydro

        self.RHydro = sd.RHydro * self.years # units such that (MHydro.sum(axis=0)<=RHydro).all()

        self.MLoad = sd.MLoad
        self.MRor = sd.MRor
        self.MPV = np.zeros((self.intervals, self.nodes))
        self.MWind = np.zeros((self.intervals, self.nodes))
        for i, n in enumerate(self.Nodel_int):
            self.MPV[:, i] += (sd.TSPV[:self.intervals, sd.PVl_int == n] * self.CPV[sd.PVl_int == n]).sum(axis=1)
            self.MWind[:, i] += (sd.TSWind[:self.intervals, sd.Windl_int == n] * self.CWind[sd.Windl_int == n]).sum(axis=1)

        self.cache_0_donors = sd.cache_0_donors
        self.cache_n_donors = sd.cache_n_donors
        
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
def Evaluate(S, costFactors):
    Simulate(S)

    S.Penalties = np.maximum(0, S.MDeficit.sum())*1000  # MWh/resolution

    CHVI = np.zeros(len(S.network_mask), dtype=np.float64)
    CHVI[S.network_mask] = S.CHVI

    cost = np.array(
        [
            # generation capex
            S.CPV.sum() * costFactors.pv[0],
            S.CWind.sum() * costFactors.onsw[0],
            S.CGas.sum()  * costFactors.gas[0],
            S.CHydro.sum() * costFactors.hydro[0],
            # generation fom
            S.CPV.sum() * costFactors.pv[1],
            S.CWind.sum() * costFactors.onsw[1],
            S.CGas.sum()  * costFactors.gas[1],
            S.CHydro.sum() * costFactors.hydro[1],
            # generation vom
            # pv, onsw, battery are 0
            S.MGas.sum() * S.resolution / S.years * costFactors.gas[2],
            S.MHydro.sum() * S.resolution / S.years * costFactors.hydro[2],
            # storage
            S.CPHP.sum() * costFactors.phes[0],
            S.CPHS.sum() * costFactors.phes[1],
            S.CPHP.sum() * costFactors.phes[2],
            S.MDischarge.sum() * S.resolution / S.years * costFactors.phes[3],
            costFactors.phes[4],
        ]
        +
        # transmission network
        list(
            (
                S.CPV.sum() +
                S.CWind.sum() +
                S.CGas.sum() +
                S.CHydro.sum()
            )
            * costFactors.ac
        )
        + list((CHVI * costFactors.hvi).sum(axis=1))
    )

    # Levelised Costs of:
    # Electricity
    S.LCOE = cost.sum() / S.energy
    # Generation
    S.LCOG = cost[:10].sum() / (
        1000 * S.resolution / S.years * (
            S.MPV.sum()
            + S.MWind.sum()
            + S.MGas.sum()
            + S.MHydro.sum()
        )
    )
    # Storage
    # S.LCOSP = zero_safe_division(cost[10:15].sum(), S.MDischarge.sum()*S.resolution/S.years)
    # Balancing - Storage
    S.LCOBS = cost[10:15].sum() / S.energy
    # Balancing - Transmission
    S.LCOBT = cost[15:].sum() / S.energy
    # Balancing - Spillage
    S.LCOBL = S.LCOE - S.LCOG - S.LCOBS - S.LCOBT
    S.LCOB = S.LCOBS + S.LCOBT + S.LCOBL

    S.CAPEX = sum([cost[i] for i in [0, 1, 2, 3, 10, 11, 15, 18]]) / S.energy
    S.OPEX = S.LCOE - S.CAPEX

    return S.LCOE, S.Penalties
