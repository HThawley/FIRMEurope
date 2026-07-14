# type: ignore
import numpy as np
from numpy.typing import NDArray

from firm_ce.system.scalar.parameters import ScenarioParameters_InstanceType
from firm_ce.system.scalar.components import Fleet_InstanceType
from firm_ce.system.scalar.topology import Network_InstanceType
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import DictType, TypedDict, boolean, nbfloat, npfloat, nbintp, npint, npintp, unicode_type
from firm_ce.common.helpers import safe_divide_2d_1d

from firm_ce.system.tensor.network import GenerateTensorNetwork
from firm_ce.system.tensor.costs import CostTensor, CostTensorType


if JIT_ENABLED:
    static_spec = [
        # -- Config --
        ("resolution", nbfloat),
        ("allowance", nbfloat),
        ("intervals", nbintp),
        ("asset_node_map", DictType(unicode_type, nbintp[:])),

        # -- Static Data --
        ("costs", CostTensorType),
        ("years", nbintp),
        ("years_float", nbfloat),
        ("year_of_interval", nbintp[:]),
        ("energy", nbfloat),
        ("legacy_costs", nbfloat),
        ("mean_annual_demand_mwh", nbfloat),

        # -- Object data --
        # (nstor,)
        ("storage_charge_eff", nbfloat[:]),
        ("storage_discha_eff", nbfloat[:]),
        # (nhvi,)
        ("line_efficiencies", nbfloat[:]),

        # -- Time Series --
        # (intervals, nodes)
        ("TSpfix", nbfloat[:, :]),
        ("TSpsat", nbfloat[:, :]),
        ("TSoffw", nbfloat[:, :]),
        ("TSonsw", nbfloat[:, :]),
        ("TSnuke", nbfloat[:, :]),
        ("TSnlte", nbfloat[:, :]),
        ("TSphes_inflow", nbfloat[:, :]),
        # (intervals, nodes, hydro_type)
        ("TShyd_inflow", nbfloat[:, :, :]),
        # (years, npeak)
        ("Bpeak", nbfloat[:, :]),  # annual fuel budget

        # -- Existing Capacity --
        # (nodes,)
        ("EphP", nbfloat[:]),
        ("EphE", nbfloat[:]),
        ("Eror", nbfloat[:]),
        ("Enuke", nbfloat[:]),
        # (nhvi,)
        ("Elines", nbfloat[:]),
        # (nodes, static.nhyd)
        ("EhydP", nbfloat[:, :]),
        ("EhydE", nbfloat[:, :]),
        # (nodes, static.nstor)
        ("EstorageP", nbfloat[:, :]),
        ("EstorageE", nbfloat[:, :]),

        # -- Power Traces (nodal) --
        # (intervals, nodes)
        ("Mload", nbfloat[:, :]),
        ("Mror", nbfloat[:, :]),
        ("Mnetload_mror", nbfloat[:, :]),

        # -- Topology --
        ("nodes", nbintp),
        ("nhvi", nbintp),
        ("nnuke", nbintp),
        ("nnlte", nbintp),
        ("nstor", nbintp),
        ("nhyd", nbintp),
        ("npeak", nbintp),
        # (nodes,)
        ("Nodel_int", nbintp[:]),
        ("scenario_mask", boolean[:]),
        ("Rnuke_mask", boolean[:]),
        ("Rnlte_mask", boolean[:]),
        ("internal_loss", nbfloat[:]),
        # (nnuke,)
        ("Rnuke_Rnlte_mask", boolean[:]),
        # (nhvi,)
        ("network_mask", boolean[:]),
        # (nhvi, 2)
        ("network", nbintp[:, :]),

        # -- Network Cache --
        ("neigh_neighbors", nbintp[:]),
        ("neigh_lines_arr", nbintp[:]),
        ("neigh_offsets", nbintp[:]),

        # -- x-vector mapping --
        ("abs_rel_scaler", nbfloat[:]),
        ("pfix_offset", nbintp),
        ("pfix_nodes", nbintp[:]),
        ("pfix_len", nbintp),
        ("psat_offset", nbintp),
        ("psat_nodes", nbintp[:]),
        ("psat_len", nbintp),
        ("offw_offset", nbintp),
        ("offw_nodes", nbintp[:]),
        ("offw_len", nbintp),
        ("onsw_offset", nbintp),
        ("onsw_nodes", nbintp[:]),
        ("onsw_len", nbintp),
        ("biog_offset", nbintp),
        ("biog_nodes", nbintp[:]),
        ("biog_len", nbintp),
        ("biom_offset", nbintp),
        ("biom_nodes", nbintp[:]),
        ("biom_len", nbintp),
        ("ccgt_offset", nbintp),
        ("ccgt_nodes", nbintp[:]),
        ("ccgt_len", nbintp),
        ("nuke_offset", nbintp),
        ("nuke_nodes", nbintp[:]),
        ("nuke_len", nbintp),
        ("nlte_offset", nbintp),
        ("nlte_nodes", nbintp[:]),
        ("nlte_len", nbintp),
        ("php_offset", nbintp),
        ("php_nodes", nbintp[:]),
        ("php_len", nbintp),
        ("b2p_offset", nbintp),
        ("b2p_nodes", nbintp[:]),
        ("b2p_len", nbintp),
        ("b4p_offset", nbintp),
        ("b4p_nodes", nbintp[:]),
        ("b4p_len", nbintp),
        ("phe_offset", nbintp),
        ("phe_nodes", nbintp[:]),
        ("phe_len", nbintp),
        ("lines_offset", nbintp),
    ]
else:
    static_spec = []


@jitclass(static_spec)
class StaticTensor:
    def __init__(
        self,
        scenario_parameters: ScenarioParameters_InstanceType,
        fleet: Fleet_InstanceType,
        network: Network_InstanceType,
        asset_node_map: TypedDict[unicode_type, nbintp[:]],
        abs_rel_scaler: NDArray[float],
    ):
        self.resolution = scenario_parameters.resolution
        self.allowance = scenario_parameters.allowance
        self.years = scenario_parameters.year_count
        self.years_float = scenario_parameters.year_float
        self.intervals = scenario_parameters.intervals_count
        self.year_of_interval = scenario_parameters.year_of_interval
        self.nodes = scenario_parameters.node_count
        self.nhvi = network.major_line_count
        self.energy = scenario_parameters.demand_sum_mwh
        self.mean_annual_demand_mwh = (self.energy / self.years_float)
        self.abs_rel_scaler = abs_rel_scaler.copy()

        self.legacy_costs = 0.0
        for gen in fleet.generators.values():
            if gen.unit_type == "ror":
                # vom is 0
                self.legacy_costs += gen.initial_capacity * 1e6 * gen.cost.fom
                self.legacy_costs += gen.initial_capacity * 1e6 * gen.cost.capex_p / gen.cost.annuity_factor
            # nuclear caught by Cnuke - legcay nuke has same costs as new
        for sto in fleet.storages.values():
            # legacy phes has different costs to new phes
            if sto.unit_type in ("clphes", "hydro", "olphes", "pond"):
                self.legacy_costs += (  # annualised build, power
                    sto.initial_power_capacity * 1e6 * sto.cost.capex_p / sto.cost.annuity_factor
                )
                self.legacy_costs += (  # annualised build, energy
                    sto.initial_energy_capacity * 1e6 * sto.cost.capex_e / sto.cost.annuity_factor
                )
                self.legacy_costs += (  # fom
                    sto.initial_power_capacity * 1e6 * sto.cost.fom
                )
                # vom is 0
        # legacy lines have same cost as new

        self.Nodel_int = np.arange(self.nodes, dtype=npintp)
        self.Mload = np.zeros((self.intervals, self.nodes), npfloat)
        self.internal_loss = np.zeros(self.nodes, dtype=npfloat)
        for node in network.nodes.values():
            self.Mload[:, node.order] += node.data
            self.internal_loss[node.order] = node.internal_loss
            if node.order not in self.Nodel_int:
                raise Exception

        # computationally efficient x-vector mapping
        empty_nodes = np.empty(0, dtype=npintp)

        _map = asset_node_map.get("pv_fixed", empty_nodes)
        self.pfix_offset = 0
        self.pfix_nodes = _map
        self.pfix_len = len(_map)

        _map = asset_node_map.get("pv_track", empty_nodes)
        self.psat_offset = self.pfix_len
        self.psat_nodes = _map
        self.psat_len = len(_map)

        _map = asset_node_map.get("offw", empty_nodes)
        self.offw_offset = self.psat_len + self.psat_offset
        self.offw_nodes = _map
        self.offw_len = len(_map)

        _map = asset_node_map.get("onsw", empty_nodes)
        self.onsw_offset = self.offw_len + self.offw_offset
        self.onsw_nodes = _map
        self.onsw_len = len(_map)

        _map = asset_node_map.get("biogas", empty_nodes)
        self.biog_offset = self.onsw_len + self.onsw_offset
        self.biog_nodes = _map
        self.biog_len = len(_map)

        _map = asset_node_map.get("biomass", empty_nodes)
        self.biom_offset = self.biog_len + self.biog_offset
        self.biom_nodes = _map
        self.biom_len = len(_map)

        _map = asset_node_map.get("ccgt", empty_nodes)
        self.ccgt_offset = self.biom_len + self.biom_offset
        self.ccgt_nodes = _map
        self.ccgt_len = len(_map)

        _map = asset_node_map.get("nuclear", empty_nodes)
        self.nuke_offset = self.ccgt_len + self.ccgt_offset
        self.nuke_nodes = _map
        self.nuke_len = len(_map)

        _map = asset_node_map.get("nuclear_lte", empty_nodes)
        self.nlte_offset = self.nuke_len + self.nuke_offset
        self.nlte_nodes = _map
        self.nlte_len = len(_map)

        _map = asset_node_map.get("nphes", empty_nodes)
        self.php_offset = self.nlte_len + self.nlte_offset
        self.php_nodes = _map
        self.php_len = len(_map)

        _map = asset_node_map.get("bess2h", empty_nodes)
        self.b2p_offset = self.php_len + self.php_offset
        self.b2p_nodes = _map
        self.b2p_len = len(_map)

        _map = asset_node_map.get("bess4h", empty_nodes)
        self.b4p_offset = self.b2p_len + self.b2p_offset
        self.b4p_nodes = _map
        self.b4p_len = len(_map)

        _map = asset_node_map.get("nphes", empty_nodes)
        self.phe_offset = self.b4p_len + self.b4p_offset
        self.phe_nodes = _map
        self.phe_len = len(_map)

        self.lines_offset = self.phe_len + self.phe_offset

        basic_network = np.zeros((self.nhvi, 2), npintp)
        self.Elines = np.zeros(self.nhvi, npfloat)
        self.line_efficiencies = np.zeros(self.nhvi, npfloat)
        for line in network.major_lines.values():
            line_order = line.order
            basic_network[line_order, 0] = line.node_start.order
            basic_network[line_order, 1] = line.node_end.order
            self.Elines[line_order] = line.initial_capacity
            self.line_efficiencies[line_order] = line.efficiency

        self.nstor = 3  # TODO: dynamic
        self.nhyd = 2  # TODO: dynamic
        self.npeak = 3  # TODO: dynamic

        self.Bpeak = np.zeros((self.years, self.npeak-1), npfloat)
        for fuel in fleet.fuels.values():
            if fuel.name == "biomass":
                self.Bpeak[:, 0] = fuel.annual_limit
            if fuel.name == "biogas":
                self.Bpeak[:, 1] = fuel.annual_limit

        self.storage_charge_eff = np.zeros(self.nstor, npfloat)
        self.storage_discha_eff = np.zeros(self.nstor, npfloat)
        storage_type_count = np.zeros(self.nstor, npint)

        # original assumption is flat efficiency across fleet
        # have implemented by taking the mean
        for sto in fleet.storages.values():
            if "phes" in sto.unit_type:
                s = 0  # PHES
            elif sto.unit_type == 'bess4h':
                s = 1  # 4-hour battery
            elif sto.unit_type == 'bess2h':
                s = 2  # 2-hour battery
            else:
                continue
            self.storage_charge_eff[s] += sto.charge_efficiency
            self.storage_discha_eff[s] += sto.discharge_efficiency
            storage_type_count[s] += 1
        self.storage_charge_eff /= storage_type_count
        self.storage_discha_eff /= storage_type_count

        self.Eror = np.zeros(self.nodes, npfloat)
        self.Enuke = np.zeros(self.nodes, npfloat)
        self.EstorageP = np.zeros((self.nodes, self.nstor), npfloat)
        self.EstorageE = np.zeros((self.nodes, self.nstor), npfloat)
        self.EhydP = np.zeros((self.nodes, self.nhyd), npfloat)
        self.EhydE = np.zeros((self.nodes, self.nhyd), npfloat)

        self.Rnuke_mask = np.zeros(self.nodes, np.bool_)
        self.Rnlte_mask = np.zeros(self.nodes, np.bool_)

        self.Eror = np.zeros(self.nodes, npfloat)
        self.Mror = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSpfix = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSpsat = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSonsw = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSoffw = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSnuke = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSnlte = np.zeros((self.intervals, self.nodes), npfloat)
        self.TSphes_inflow = np.zeros((self.intervals, self.nodes), npfloat)
        self.TShyd_inflow = np.zeros((self.intervals, self.nodes, self.nhyd), npfloat)

        counts = np.zeros((6, self.nodes), npint)
        for gen in fleet.generators.values():
            n = gen.node.order
            if gen.unit_type == "pv_fixed":
                self.TSpfix[:, n] += gen.data
                counts[0, n] += 1
            elif gen.unit_type == "pv_track":
                self.TSpsat[:, n] += gen.data
                counts[1, n] += 1
            elif gen.unit_type == "offw":
                self.TSoffw[:, n] += gen.data
                counts[2, n] += 1
            elif gen.unit_type == "onsw":
                self.TSonsw[:, n] += gen.data
                counts[3, n] += 1
            elif gen.unit_type == "ror":
                self.Eror[n] += gen.initial_capacity
                self.Mror[:, n] += (gen.initial_capacity * gen.data)
            elif gen.unit_type == "nuclear":
                self.Rnuke_mask[n] = True
                self.Enuke[n] += gen.initial_capacity
                self.TSnuke[:, n] += gen.data
                counts[4, n] += 1
            elif gen.unit_type == "nuclear_LTE":
                self.Rnlte_mask[n] = True
                self.TSnlte[:, n] += gen.data
                counts[5, n] += 1

        self.Rnuke_Rnlte_mask = self.Rnlte_mask[self.Rnuke_mask]
        # This ought to be full of 1s and 0s. Future dev
        safe_divide_2d_1d(self.TSpfix, counts[0], self.TSpfix)
        safe_divide_2d_1d(self.TSpsat, counts[1], self.TSpsat)
        safe_divide_2d_1d(self.TSoffw, counts[2], self.TSoffw)
        safe_divide_2d_1d(self.TSonsw, counts[3], self.TSonsw)
        safe_divide_2d_1d(self.TSnuke, counts[4], self.TSnuke)
        safe_divide_2d_1d(self.TSnlte, counts[5], self.TSnlte)

        for sto in fleet.storages.values():
            n = sto.node.order
            if "phes" in sto.unit_type:
                self.EstorageP[n, 0] += sto.initial_power_capacity
                self.EstorageE[n, 0] += sto.initial_energy_capacity
                if sto.inflows:
                    self.TSphes_inflow[:, n] += sto.data
            elif sto.unit_type == "pond":
                self.TShyd_inflow[:, n, 0] += sto.data
                self.EhydP[n, 0] += sto.initial_power_capacity
                self.EhydE[n, 0] += sto.initial_energy_capacity
            elif sto.unit_type == "hydro":
                self.TShyd_inflow[:, n, 1] += sto.data
                self.EhydP[n, 1] += sto.initial_power_capacity
                self.EhydE[n, 1] += sto.initial_energy_capacity

        self.Mnetload_mror = np.zeros((self.intervals, self.nodes), npfloat)
        self.Mnetload_mror = (self.Mload / (1 - self.internal_loss)) - self.Mror

        (self.network,
         self.network_mask,
         self.neigh_neighbors,
         self.neigh_lines_arr,
         self.neigh_offsets
         ) = GenerateTensorNetwork(basic_network, self.Nodel_int)

        self.nnuke = self.Rnuke_mask.sum()
        self.nnlte = self.Rnlte_mask.sum()

        self.costs = CostTensor(
            self.resolution,
            self.nodes,
            self.nhvi,
            self.years_float,
            fleet,
            network
        )


if JIT_ENABLED:
    StaticTensorType = StaticTensor.class_type.instance_type
else:
    StaticTensorType = StaticTensor
