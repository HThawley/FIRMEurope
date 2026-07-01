# type: ignore
import numpy as np

from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass, njit
from firm_ce.common.typing import DictType, TypedDict, boolean, nbfloat, npfloat, nbintp, npint, npintp, unicode_type, nbint
from firm_ce.common.helpers import safe_divide_array


@njit(boundscheck=True)
def GenerateTensorNetwork(network, Nodel_int):  # noqa: C901
    networkdict = TypedDict.empty(nbintp, nbintp)
    for k in range(len(Nodel_int)):
        networkdict[Nodel_int[k]] = k

    num_lines = network.shape[0]
    network_mask = np.zeros(num_lines, dtype=np.bool_)
    valid_count = 0

    for i in range(num_lines):
        start_node = network[i, 0]
        end_node = network[i, 1]

        # Check if both start and end are in our valid nodes dict
        if start_node in networkdict and end_node in networkdict:
            network_mask[i] = True
            valid_count += 1

    # Create valid_network and remap indices
    valid_network = np.empty((valid_count, 2), dtype=npintp)
    idx = 0
    for i in range(num_lines):
        if network_mask[i]:
            valid_network[idx, 0] = networkdict[network[i, 0]]
            valid_network[idx, 1] = networkdict[network[i, 1]]
            idx += 1

    # Build cache
    cache_0_donors = TypedDict.empty(nbintp, nbintp[:, :])
    nodes = len(Nodel_int)

    for n in range(nodes):
        # count the number of connections to pre-allocate arrays
        count = 0
        for line in range(valid_count):
            if valid_network[line, 0] == n or valid_network[line, 1] == n:
                count += 1

        # allocate and fill
        if count > 0:
            res_matrix = np.empty((2, count), dtype=npintp)
            c = 0
            for line in range(valid_count):
                if valid_network[line, 0] == n:
                    res_matrix[0, c] = valid_network[line, 1]
                    res_matrix[1, c] = line
                    c += 1
                elif valid_network[line, 1] == n:
                    res_matrix[0, c] = valid_network[line, 0]
                    res_matrix[1, c] = line
                    c += 1

            cache_0_donors[n] = res_matrix
        else:
            cache_0_donors[n] = np.empty((2, 0), dtype=npintp)

    return valid_network, network_mask, cache_0_donors


if JIT_ENABLED:
    static_spec = [
        # -- Config --
        ("resolution", nbfloat),
        ("allowance", nbfloat),
        ("intervals", nbintp),
        ("asset_node_map", DictType(unicode_type, nbintp[:])),

        # -- Static Data --
        ("years", nbintp),
        ("years_float", nbfloat),
        ("year_of_interval", nbintp[:]),
        ("energy", nbfloat),
        ("legacy_costs", nbfloat),

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
        # (nnuke,)
        ("Rnuke_Rnlte_mask", boolean[:]),
        # (nhvi,)
        ("network_mask", boolean[:]),
        # (nhvi, 2)
        ("network", nbintp[:, :]),
        # dict
        ("cache_0_donors", DictType(nbintp, nbintp[:, :])),

        # -- x-vector mapping --
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
        for node in network.nodes.values():
            self.Mload[:, node.order] += node.data
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

        _map = asset_node_map.get("2hr_bess", empty_nodes)
        self.b2p_offset = self.php_len + self.php_offset
        self.b2p_nodes = _map
        self.b2p_len = len(_map)

        _map = asset_node_map.get("4hr_bess", empty_nodes)
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
            elif np.isclose(sto.duration, 4.0):
                s = 1  # 4-hour battery
            elif np.isclose(sto.duration, 2.0):
                s = 2  # 2-hour battery
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
        self.TSpfix = safe_divide_array(self.TSpfix, counts[0])
        self.TSpsat = safe_divide_array(self.TSpsat, counts[1])
        self.TSoffw = safe_divide_array(self.TSoffw, counts[2])
        self.TSonsw = safe_divide_array(self.TSonsw, counts[3])
        self.TSnuke = safe_divide_array(self.TSnuke, counts[4])
        self.TSnlte = safe_divide_array(self.TSnlte, counts[5])

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
        self.Mnetload_mror = self.Mload - self.Mror

        self.network, self.network_mask, self.cache_0_donors = GenerateTensorNetwork(
            basic_network, self.Nodel_int
        )

        self.nnuke = self.Rnuke_mask.sum()
        self.nnlte = self.Rnlte_mask.sum()


if JIT_ENABLED:
    StaticTensorType = StaticTensor.class_type.instance_type
else:
    StaticTensorType = StaticTensor


if JIT_ENABLED:
    cost_spec = [
        # -- Fixed Costs --
        # (nodes,)
        ("Fpfix", nbfloat[:]),
        ("Fpsat", nbfloat[:]),
        ("Foffw", nbfloat[:]),
        ("Fonsw", nbfloat[:]),
        ("Fbiog", nbfloat[:]),
        ("Fbiom", nbfloat[:]),
        ("Fgas", nbfloat[:]),
        ("Fnuke", nbfloat[:]),
        ("Fnlte", nbfloat[:]),
        ("FphP", nbfloat[:]),
        ("FphE", nbfloat[:]),
        ("Fb4P", nbfloat[:]),
        ("Fb2P", nbfloat[:]),
        # (nhvi,)
        ("Flines", nbfloat[:]),

        # -- Variable Costs --
        # (nodes,)
        ("Vpfix", nbfloat[:]),
        ("Vpsat", nbfloat[:]),
        ("Voffw", nbfloat[:]),
        ("Vonsw", nbfloat[:]),
        ("Vbiog", nbfloat[:]),
        ("Vbiom", nbfloat[:]),
        ("Vgas", nbfloat[:]),
        ("Vnuke", nbfloat[:]),
        ("Vnlte", nbfloat[:]),
        ("Vph", nbfloat[:]),
        ("Vb4", nbfloat[:]),
        ("Vb2", nbfloat[:]),
        # (nhvi,)
        ("Vlines", nbfloat[:]),
    ]
else:
    cost_spec = []


@jitclass(cost_spec)
class CostTensor:
    def __init__(
        self,
        static: StaticTensorType,
        fleet: Fleet_InstanceType,
        network: Network_InstanceType,
    ):
        res = static.resolution
        nodes = static.nodes
        nhvi = static.nhvi
        years_float = static.years_float

        # Initialize arrays based on static node dimensions
        self.Fpfix = np.zeros(nodes, dtype=npfloat)
        self.Fpsat = np.zeros(nodes, dtype=npfloat)
        self.Foffw = np.zeros(nodes, dtype=npfloat)
        self.Fonsw = np.zeros(nodes, dtype=npfloat)
        self.Fbiog = np.zeros(nodes, dtype=npfloat)
        self.Fbiom = np.zeros(nodes, dtype=npfloat)
        self.Fgas = np.zeros(nodes, dtype=npfloat)
        self.Fnuke = np.zeros(nodes, dtype=npfloat)
        self.Fnlte = np.zeros(nodes, dtype=npfloat)
        self.FphP = np.zeros(nodes, dtype=npfloat)
        self.FphE = np.zeros(nodes, dtype=npfloat)
        self.Fb4P = np.zeros(nodes, dtype=npfloat)
        self.Fb2P = np.zeros(nodes, dtype=npfloat)
        #  Batteries have no capex_e
        self.Flines = np.zeros(nhvi, dtype=npfloat)

        self.Vpfix = np.zeros(nodes, dtype=npfloat)
        self.Vpsat = np.zeros(nodes, dtype=npfloat)
        self.Voffw = np.zeros(nodes, dtype=npfloat)
        self.Vonsw = np.zeros(nodes, dtype=npfloat)
        self.Vbiog = np.zeros(nodes, dtype=npfloat)
        self.Vbiom = np.zeros(nodes, dtype=npfloat)
        self.Vgas = np.zeros(nodes, dtype=npfloat)
        self.Vnuke = np.zeros(nodes, dtype=npfloat)
        self.Vnlte = np.zeros(nodes, dtype=npfloat)
        self.Vph = np.zeros(nodes, dtype=npfloat)
        self.Vb4 = np.zeros(nodes, dtype=npfloat)
        self.Vb2 = np.zeros(nodes, dtype=npfloat)
        self.Vlines = np.zeros(nhvi, dtype=npfloat)
        # TODO:  legacy hydro vom. Low priority as currently 0 $
        # self.Vlegacy_hydro

        for gen in fleet.generators.values():
            n = gen.node.order
            # Pre-calculate annualized fixed costs per unit capacity
            Fval = 1e6 * ((gen.cost.capex_p / gen.cost.annuity_factor) + gen.cost.fom)
            # fuel_cost_h currently 0 for all generators
            Vval = res * 1e3 * (gen.cost.vom + gen.cost.fuel_cost_mwh) / years_float
            if gen.unit_type == "pv_fixed":
                self.Fpfix[n] = Fval
                self.Vpfix[n] = Vval
            elif gen.unit_type == "pv_track":
                self.Fpsat[n] = Fval
                self.Vpsat[n] = Vval
            elif gen.unit_type == "offw":
                self.Foffw[n] = Fval
                self.Voffw[n] = Vval
            elif gen.unit_type == "onsw":
                self.Fonsw[n] = Fval
                self.Vonsw[n] = Vval
            elif gen.unit_type == "biogas":
                self.Fbiog[n] = Fval
                self.Vbiog[n] = Vval
            elif gen.unit_type == "biomass":
                self.Fbiom[n] = Fval
                self.Vbiom[n] = Vval
            elif gen.unit_type == "nuclear":
                self.Fnuke[n] = Fval
                self.Vnuke[n] = Vval
            elif gen.unit_type == "nuclear_lte":
                self.Fnlte[n] = Fval
                self.Vnlte[n] = Vval
            elif gen.unit_type == "ccgt":
                self.Fgas[n] = Fval
                self.Vgas[n] = Vval

        for sto in fleet.storages.values():
            n = sto.node.order
            FvalP = 1e6 * ((sto.cost.capex_p / sto.cost.annuity_factor) + sto.cost.fom)
            FvalE = 1e6 * (sto.cost.capex_e / sto.cost.annuity_factor)
            Vval = res * 1e3 * sto.cost.vom / years_float
            if sto.unit_type == "nphes":
                self.FphP[n] = FvalP
                self.FphE[n] = FvalE
                self.Vph[n] = Vval
            elif sto.unit_type == "4hr_battery":
                self.Fb4P[n] = FvalP
                self.Vb4[n] = Vval
            elif sto.unit_type == "2hr_battery":
                self.Fb2P[n] = FvalP
                self.Vb2[n] = Vval

        for line in network.major_lines.values():
            n = line.order
            Fval = 1e6 * (
                (line.length * line.cost.capex_p / line.cost.annuity_factor)
                + (line.cost.transformer_capex / line.cost.annuity_factor)
                + (line.length * line.cost.fom)
            )
            Vval = res * 1e3 * line.cost.vom / years_float
            self.Flines[n] = Fval
            self.Vlines[n] = Vval

        # minor_lines currently have no costs
        # for line in network.minor_lines.values():
        #     pass


if JIT_ENABLED:
    CostTensorType = CostTensor.class_type.instance_type
else:
    CostTensorType = CostTensor


if JIT_ENABLED:
    asset_spec = [
        # Capacities in GW/GWh
        ("Cpfix", nbfloat[:]),
        ("Cpsat", nbfloat[:]),
        ("Coffw", nbfloat[:]),
        ("Consw", nbfloat[:]),
        # ("Cbiog", nbfloat[:]),
        # ("Cbiom", nbfloat[:]),
        # ("Cgas", nbfloat[:]),
        ("Cpeak", nbfloat[:, :]),
        ("Cnuke", nbfloat[:]),
        ("Cnlte", nbfloat[:]),
        ("CnphP", nbfloat[:]),
        ("CnphE", nbfloat[:]),
        ("CstorageP", nbfloat[:, :]),
        ("CstorageE", nbfloat[:, :]),
        ("ChydP", nbfloat[:, :]),
        ("ChydE", nbfloat[:, :]),
        ("Clines", nbfloat[:]),
        ("Clongdur", nbfloat[:]),
        ("Cshortdur", nbfloat[:]),
    ]
else:
    asset_spec = []


@jitclass(asset_spec)
class AssetTensor:
    def __init__(
        self,
        x: np.ndarray[npfloat],
        static: StaticTensorType,
    ):
        nodes = static.nodes

        self.Cpfix = np.zeros(nodes, dtype=npfloat)
        self.Cpsat = np.zeros(nodes, dtype=npfloat)
        self.Coffw = np.zeros(nodes, dtype=npfloat)
        self.Consw = np.zeros(nodes, dtype=npfloat)
        # self.Cbiog = np.zeros(nodes, dtype=npfloat)
        # self.Cbiom = np.zeros(nodes, dtype=npfloat)
        # self.Cgas = np.zeros(nodes, dtype=npfloat)
        self.Cpeak = np.zeros((nodes, static.npeak), dtype=npfloat)
        self.Cnuke = static.Enuke.copy()
        self.Cnlte = np.zeros(nodes, dtype=npfloat)
        self.ChydP = static.EhydP.copy()
        self.ChydE = static.EhydE.copy()
        self.CnphP = np.zeros(nodes, dtype=npfloat)
        self.CnphE = np.zeros(nodes, dtype=npfloat)
        self.CstorageP = static.EstorageP.copy()
        self.CstorageE = static.EstorageE.copy()
        self.Clines = static.Elines.copy()

        for i in range(static.pfix_len):
            self.Cpfix[static.pfix_nodes[i]] += x[static.pfix_offset + i]

        for i in range(static.psat_len):
            self.Cpsat[static.psat_nodes[i]] += x[static.psat_offset + i]

        for i in range(static.offw_len):
            self.Coffw[static.offw_nodes[i]] += x[static.offw_offset + i]

        for i in range(static.onsw_len):
            self.Consw[static.onsw_nodes[i]] += x[static.onsw_offset + i]

        for i in range(static.biog_len):
            self.Cpeak[static.biog_nodes[i], 1] += x[static.biog_offset + i]

        for i in range(static.biom_len):
            self.Cpeak[static.biom_nodes[i], 0] += x[static.biom_offset + i]

        for i in range(static.ccgt_len):
            self.Cpeak[static.ccgt_nodes[i], 2] += x[static.ccgt_offset + i]

        for i in range(static.nuke_len):
            self.Cnuke[static.nuke_nodes[i]] += x[static.nuke_offset + i]
        for i in range(static.nlte_len):
            _cap = x[static.nlte_offset + i]
            self.Cnuke[static.nlte_nodes[i]] += _cap
            self.Cnlte[static.nlte_nodes[i]] += _cap

        for i in range(static.php_len):
            _cap = x[static.php_offset + i]
            self.CnphP[static.php_nodes[i]] += _cap
            self.CstorageP[static.php_nodes[i], 0] += _cap

        for i in range(static.b4p_len):
            _cap = x[static.b4p_offset + i]
            self.CstorageP[static.b4p_nodes[i], 1] += _cap
            self.CstorageE[static.b4p_nodes[i], 1] += 4.0 * _cap

        for i in range(static.b2p_len):
            _cap = x[static.b2p_offset + i]
            self.CstorageP[static.b2p_nodes[i], 2] += _cap
            self.CstorageE[static.b2p_nodes[i], 2] += 2.0 * _cap

        for i in range(static.phe_len):
            _cap = x[static.phe_offset + i]
            self.CnphE[static.php_nodes[i]] += _cap
            self.CstorageE[static.phe_nodes[i], 0] += _cap

        for i in range(static.nhvi):
            self.Clines[i] += x[static.lines_offset + i]

        self.Clongdur = self.ChydP[:, 0] + self.ChydP[:, 1] + self.CstorageP[:, 0]
        self.Cshortdur = self.CstorageP[:, 1] + self.CstorageP[:, 2]


if JIT_ENABLED:
    AssetTensorType = AssetTensor.class_type.instance_type
else:
    AssetTensorType = AssetTensor


if JIT_ENABLED:
    operation_spec = [
        # -- Nodal power flow accounting --
        # (intervals, nodes)
        ("Mreservoir_init", nbfloat[:, :]),
        ("Mstorage_init", nbfloat[:, :]),

        ("Mnetload", nbfloat[:, :]),
        ("Mdeficit", nbfloat[:, :]),
        ("Mcurtail", nbfloat[:, :]),
        ("Munbalanced", nbfloat[:, :]),
        ("Mimport", nbfloat[:, :]),
        ("Mexport", nbfloat[:, :]),
        ("Mphes_spill", nbfloat[:, :]),

        ("Mpfix", nbfloat[:, :]),
        ("Mpsat", nbfloat[:, :]),
        ("Moffw", nbfloat[:, :]),
        ("Monsw", nbfloat[:, :]),
        # ("Mbiog", nbfloat[:, :]),
        # ("Mbiom", nbfloat[:, :]),
        # ("Mgas", nbfloat[:, :]),
        ("Mnuke", nbfloat[:, :]),
        # (intervals, nhvi)
        ("Tnetflow", nbfloat[:, :]),
        # (intervals, nodes, nstor)
        ("Mdischarge", nbfloat[:, :, :]),
        ("Mcharge", nbfloat[:, :, :]),
        ("Mstorage", nbfloat[:, :, :]),
        # (intervals, nodes, nhydro)
        ("Mreservoir", nbfloat[:, :, :]),
        ("Mhyd_spill", nbfloat[:, :, :]),
        ("Mhydro", nbfloat[:, :, :]),
        # (intervals, nodes, npeak)
        ("Mpeak", nbfloat[:, :, :]),

        # -- Temporary memory buffers --
        # scalar
        ("has_deficit_t", boolean),
        ("has_curtail_t", boolean),
        # (nodes,)
        ("cap_fwd", nbfloat[:]),
        ("cap_rev", nbfloat[:]),
        ("eff_fwd", nbfloat[:]),
        ("eff_rev", nbfloat[:]),
        ("eff", nbfloat[:]),
        ("visited", boolean[:]),
        ("parent_node", nbintp[:]),
        ("parent_line", nbintp[:]),
        ("path_nodes", nbintp[:]),
        ("path_lines", nbintp[:]),
        ("rolling_deficits", nbfloat[:]),
        ("surplus_buffer", nbfloat[:]),
        ("surplus_orig", nbfloat[:]),
        ("fill_buffer", nbfloat[:]),
        ("fill_orig", nbfloat[:]),
        ("stall_checkpoint", nbfloat[:]),
        ("stall_counter", nbint[:]),
        # (intervals, nodes)
        ("precharge_flag", boolean[:, :]),
        ("trickling_flag", boolean[:, :]),
        ("hydro_min_future", nbfloat[:, :]),
        ("storage_min_future", nbfloat[:, :]),
        ("storage_max_future", nbfloat[:, :]),
        # (intervals, nodes, nstor)
        ("charge_max_t", nbfloat[:, :, :]),
        ("discharge_max_t", nbfloat[:, :, :]),
        # (nodes, nhydro)
        ("hydro_headroom", nbfloat[:, :]),
        # (years, npeak-1)
        ("remaining_peak_budget", nbfloat[:, :]),
    ]

else:
    operation_spec = []


@jitclass(operation_spec)
class OperationTensor:
    def __init__(
        self,
        static: StaticTensorType,
        assets: AssetTensorType,
    ):
        nodes = static.nodes
        intervals = static.intervals
        nhvi = static.nhvi
        nhyd = static.nhyd
        nstor = static.nstor

        self.Mpfix = assets.Cpfix * static.TSpfix
        self.Mpsat = assets.Cpsat * static.TSpsat
        self.Monsw = assets.Consw * static.TSonsw
        self.Moffw = assets.Coffw * static.TSoffw
        self.Mnuke = assets.Cnuke * static.TSnuke
        # self.Mbiog = np.zeros((intervals, nodes), dtype=npfloat)
        # self.Mbiom = np.zeros((intervals, nodes), dtype=npfloat)
        # self.Mgas = np.zeros((intervals, nodes), dtype=npfloat)
        self.Mpeak = np.zeros((intervals, nodes, static.npeak), dtype=npfloat)

        self.Mnetload = (
            static.Mnetload_mror
            - self.Mpfix
            - self.Mpsat
            - self.Moffw
            - self.Monsw
            - self.Mnuke
        )

        self.Munbalanced = self.Mnetload.copy()
        self.Mdeficit = np.maximum(npfloat(0.0), self.Mnetload)
        self.Mcurtail = -np.minimum(npfloat(0.0), self.Mnetload)

        self.Mimport = np.zeros((intervals, nodes), dtype=npfloat)
        self.Mexport = np.zeros((intervals, nodes), dtype=npfloat)
        self.Tnetflow = np.zeros((intervals, nhvi), dtype=npfloat)

        self.Mdischarge = np.zeros((intervals, nodes, nstor), dtype=npfloat)
        self.Mcharge = np.zeros((intervals, nodes, nstor), dtype=npfloat)
        self.Mstorage = np.zeros((intervals, nodes, nstor), dtype=npfloat)
        self.Mstorage_init = npfloat(0.5) * assets.CstorageE
        self.Mphes_spill = np.zeros((intervals, nodes), dtype=npfloat)

        self.Mhydro = np.zeros((intervals, nodes, nhyd), dtype=npfloat)
        self.Mreservoir = np.zeros((intervals, nodes, nhyd), dtype=npfloat)
        self.Mreservoir_init = npfloat(0.5) * assets.ChydE
        self.Mhyd_spill = np.zeros((intervals, nodes, nhyd), dtype=npfloat)

        self.has_deficit_t = False
        self.has_curtail_t = False

        self.cap_fwd = np.empty(nhvi, dtype=npfloat)
        self.cap_rev = np.empty(nhvi, dtype=npfloat)
        self.eff_fwd = np.empty(nhvi, dtype=npfloat)
        self.eff_rev = np.empty(nhvi, dtype=npfloat)
        self.eff = np.empty(nodes, dtype=npfloat)
        self.visited = np.empty(nodes, dtype=np.bool_)
        self.parent_node = np.empty(nodes, dtype=npintp)
        self.parent_line = np.empty(nodes, dtype=npintp)
        self.path_nodes = np.empty(nodes, dtype=npintp)
        self.path_lines = np.empty(nodes, dtype=npintp)
        self.rolling_deficits = np.empty(nodes, dtype=npfloat)
        self.surplus_buffer = np.empty(nodes, dtype=npfloat)
        self.surplus_orig = np.empty(nodes, dtype=npfloat)
        self.fill_buffer = np.empty(nodes, dtype=npfloat)
        self.fill_orig = np.empty(nodes, dtype=npfloat)
        self.stall_checkpoint = np.empty(nodes, dtype=npfloat)
        self.stall_counter = np.empty(nodes, dtype=npint)

        self.hydro_headroom = np.empty((nodes, nhyd), dtype=npfloat)
        self.hydro_min_future = np.zeros((nodes, nhyd), dtype=npfloat)

        self.precharge_flag = np.zeros((nodes, nstor), dtype=np.bool_)
        self.trickling_flag = np.zeros((nodes, nstor), dtype=np.bool_)
        self.storage_min_future = np.zeros((nodes, nstor), dtype=npfloat)
        self.storage_max_future = np.zeros((nodes, nstor), dtype=npfloat)

        self.charge_max_t = np.zeros((intervals, nodes, nstor), dtype=npfloat)
        self.discharge_max_t = np.zeros((intervals, nodes, nstor), dtype=npfloat)

        self.remaining_peak_budget = np.zeros((static.years, static.npeak-1), dtype=npfloat)


OperationTensorType = OperationTensor.class_type.instance_type
