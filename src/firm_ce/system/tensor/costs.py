# type: ignore
import numpy as np

from firm_ce.system.scalar.components import Fleet_InstanceType
from firm_ce.system.scalar.topology import Network_InstanceType
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, npfloat

from firm_ce.system.tensor.static import StaticTensorType


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
            elif sto.unit_type == "bess4h":
                self.Fb4P[n] = FvalP
                self.Vb4[n] = Vval
            elif sto.unit_type == "bess2h":
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
