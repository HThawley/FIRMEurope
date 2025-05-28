
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

EHydro = pd.read_csv("Data/hydro_plants.csv", index_col=['bus'], usecols=[1, 2])['p_nom']
EHydro = np.array([EHydro[node] if node in EHydro.index else 0 for node in Nodel]) / 1000.

RHydro = pd.read_csv("Data/inflow.csv", sep=',')
RHydro = RHydro.drop(columns=['snapshot'])
RHydro = RHydro.sum() 
RHydro.index = [' '.join(x.split(' ')[:2]) for x in RHydro.index]
RHydro = np.array([RHydro[node] if node in RHydro.index else 0 for node in Nodel]) / 1000

Gas = pd.read_csv("Data/gas.csv", sep=",")
EGas = Gas.groupby('bus')['p_nom'].sum()
EGas = np.array([EGas[node] if node in EGas.index else 0 for node in Nodel]) / 1000. 
## TODO: improve this assumption
EGasMarginal = Gas.groupby('bus')['marginal_cost'].mean() 
## TODO: include in SD and S so can be [coverage_int]ed
EGasMarginal = np.array([EGasMarginal[node] if node in EGasMarginal.index else 0 for node in Nodel]) 
EGasCapexSum = (Gas['capital_cost'] * Gas['p_nom']).sum()
Gas['type'] = Gas['Generator'].str.split(' ').str[2]

## TODO: include in SD and S so can be [coverage_int]ed
CGasCapex, CGasMarginal = Gas.groupby('type')[['capital_cost', 'marginal_cost']].mean().loc['OCGT']
CGasCapex /= 1000 # per MW to per GW
del Gas

links = pd.read_csv("Data/links.csv", usecols=range(1,6), delimiter=",")
lines = pd.read_csv("Data/lines.csv", usecols=range(1,6), delimiter=",")
links[['bus0', 'bus1']] = links[['bus0', 'bus1']].map(lambda bus: n_node[bus])
lines[['bus0', 'bus1']] = lines[['bus0', 'bus1']].map(lambda bus: n_node[bus])
lines = lines.rename(columns={'s_nom':'p_nom'})
HVI = pd.concat((links, lines))
HVI['loss'] = HVI['length'] * 0.05 * 0.001 #5% per 1000 km
HVI_index = HVI.groupby(['bus0', 'bus1']).sum().index
EHVI = HVI.groupby(['bus0', 'bus1'])['p_nom'].sum()[HVI_index].to_numpy()
## TODO: improve this assumption
LHVI = HVI.groupby(['bus0', 'bus1'])['loss'].mean()[HVI_index].to_numpy()
network = HVI[['bus0', 'bus1']].drop_duplicates().to_numpy()

## TODO: include in SD and S so can be [network_mask]ed
CHVICapex = HVI.groupby(['bus0', 'bus1'])['capital_cost'].min()[HVI_index].to_numpy() / 1000 # per MW to per GW
EHVICapexSum = (HVI['p_nom'] * HVI['length'] * HVI['capital_cost']).sum() / 1000 

del HVI, links, lines

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
    ("EHydro", float64[:]),
    ("EGas", float64[:]),
    ("EHVI", float64[:]),
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
        
        self.EHydro =    EHydro[   np.isin(Nodel_int, self.coverage_int)]
        self.EGas =      EGas[     np.isin(Nodel_int, self.coverage_int)] 
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
            ) = generate_network(network, self.Nodel_int, self.networksteps)
            
        self.nhvi = self.network_mask.sum()
        self.EHVI = EHVI[self.network_mask]
        
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
            [0.0] * self.nodes + 
            [0.0] * self.nodes + 
            [0.0] * self.nodes + 
            [0.0] * self.nhvi
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
                np.repeat(array_max(mloadmax) * 0.1, self.nhvi),
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
    # Capacity expansion in GW/GWh
    ("CPV", float64[:]),
    ("CWind", float64[:]),
    ("CPHP", float64[:]),
    ("CPHS", float64[:]),
    ("CHVI", float64[:]),
    ("CGas", float64[:]),
    # Existing capacites in GW/GWh
    ("EHydro", float64[:]),
    ("EGas", float64[:]),
    ("EHVI", float64[:]),
    # Existing + Expanded capacity (only where both are used)
    ("GGas", float64[:]),
    ("GHVI", float64[:]),
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
        
        self.EHydro = sd.EHydro
        self.EGas = sd.EGas
        self.EHVI = sd.EHVI
        
        self.GGas = self.CGas + self.EGas
        self.GHVI = self.CHVI + self.EHVI

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
def Evaluate(solution, costFactors):
    Simulate(solution)

    solution.Penalties = np.maximum(0, solution.MDeficit.sum())*1000  # MWh/resolution

    CHVI = np.zeros(len(solution.network_mask), dtype=np.float64)
    CHVI[solution.network_mask] = solution.CHVI

    capex = np.array([
        solution.CPV.sum() * costFactors.pv[0],
        solution.CWind.sum() * costFactors.onsw[0],
        solution.EHydro.sum() * costFactors.hydro[0],
        solution.CPHP.sum() * costFactors.phes[0],
        solution.CPHS.sum() * costFactors.phes[1],
        costFactors.phes[4],
        ((solution.CPV.sum() + solution.CWind.sum() + solution.CGas.sum())*costFactors.ac[:2]).sum(), # new_build connection
        (solution.CHVI * CHVICapex).sum(),
        EHVICapexSum, 
        EGasCapexSum,
        ])

    EMGas = np.zeros(solution.nodes, np.float64)
    CMGas = np.zeros(solution.nodes, np.float64)
    
    for t in range(solution.intervals):
        for n in range(solution.nodes):
            EMGas[n] += min(solution.EGas[n], solution.MGas[t, n]) # use existing gas plants first
            CMGas[n] += max(0, solution.MGas[t, n] - solution.EGas[n]) # use new gas second
    GasOpex = 0.0
    for n in range(solution.nodes):
        GasOpex += EMGas[n] * EGasMarginal[n] 
        GasOpex += CMGas[n] * CGasMarginal
    GasOpex *= solution.resolution / solution.years / 1000
    
    opex = np.array([
        solution.CPV.sum() * costFactors.pv[1],
        solution.CWind.sum() * costFactors.onsw[1],
        solution.EHydro.sum() * costFactors.hydro[1], 
        solution.MHydro.sum() * solution.resolution / solution.years * costFactors.hydro[2],
        solution.CPHP.sum() * costFactors.phes[2],
        solution.MDischarge.sum() * solution.resolution / solution.years * costFactors.phes[3],
        GasOpex,
        ])
    

    # Levelised Costs of:
    # Electricity
    solution.LCOE = (capex.sum() + opex.sum()) / solution.energy
    
# =============================================================================
#     LCOx are low priority to fix
# =============================================================================
    # Generation
    # solution.LCOG = cost[:10].sum() / (
    #     1000 * solution.resolution / solution.years * (
    #         solution.MPV.sum()
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
