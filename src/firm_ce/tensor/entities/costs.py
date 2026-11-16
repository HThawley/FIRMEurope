# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, npfloat
from firm_ce.scalar.entities.components import Fleet_InstanceType
from firm_ce.scalar.entities.topology import Network_InstanceType
from firm_ce.system.costs_common import (
    apportion_new_phes,
    get_generator_costs,
    get_line_costs,
    get_storage_costs,
)


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
        ("Faa_phP", nbfloat[:]),
        ("Faa_phE", nbfloat[:]),
        ("Fe_phP", nbfloat[:]),
        ("Fe_phE", nbfloat[:]),
        ("E_aa_max", nbfloat[:]),
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
        res: float,
        nodes: int,
        nhvi: int,
        years_float: float,
        fleet: Fleet_InstanceType,
        network: Network_InstanceType,
    ):
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
        self.Faa_phP = np.zeros(nodes, dtype=npfloat)
        self.Faa_phE = np.zeros(nodes, dtype=npfloat)
        self.Fe_phP = np.zeros(nodes, dtype=npfloat)
        self.Fe_phE = np.zeros(nodes, dtype=npfloat)
        self.Fb4P = np.zeros(nodes, dtype=npfloat)
        self.Fb2P = np.zeros(nodes, dtype=npfloat)
        self.Flines = np.zeros(nhvi, dtype=npfloat)

        self.E_aa_max = np.zeros(nodes, dtype=npfloat)

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

        for gen in fleet.generators.values():
            n = gen.node.order

            ann_build, fom, vom, fuel = get_generator_costs(gen, res, years_float)
            Fval = ann_build + fom
            Vval = vom + fuel

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
            elif gen.unit_type == "ocgt":
                self.Fgas[n] = Fval
                self.Vgas[n] = Vval

        for sto in fleet.storages.values():
            n = sto.node.order

            ann_build_p, ann_build_e, fom, vom = get_storage_costs(sto, res, years_float)
            FvalP = ann_build_p + fom
            FvalE = ann_build_e
            Vval = vom

            if sto.unit_type == "aa_phes":
                self.Faa_phP[n] = FvalP
                self.Faa_phE[n] = FvalE
                self.E_aa_max[n] = sto.max_build_e
                self.Vph[n] = Vval
            elif sto.unit_type == "e_phes":
                self.Fe_phP[n] = FvalP
                self.Fe_phE[n] = FvalE
                self.Vph[n] = Vval
            elif sto.unit_type == "bess4h":
                self.Fb4P[n] = FvalP
                self.Vb4[n] = Vval
            elif sto.unit_type == "bess2h":
                self.Fb2P[n] = FvalP
                self.Vb2[n] = Vval

        for line in network.major_lines.values():
            n = line.order

            ann_build_p, fom, vom = get_line_costs(line, res, years_float)
            Fval = ann_build_p + fom
            Vval = vom

            self.Flines[n] = Fval
            self.Vlines[n] = Vval


if JIT_ENABLED:
    CostTensorType = CostTensor.class_type.instance_type
else:
    CostTensorType = CostTensor


__all__ = [
    "CostTensor",
    "CostTensorType",
    "apportion_new_phes",
    "get_generator_costs",
    "get_line_costs",
    "get_storage_costs",
]
