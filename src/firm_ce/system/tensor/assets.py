# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, npfloat

from firm_ce.system.tensor.static import StaticTensorType

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
            x_idx = static.pfix_offset + i
            node_idx = static.pfix_nodes[i]
            self.Cpfix[node_idx] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.psat_len):
            x_idx = static.psat_offset + i
            node_idx = static.psat_nodes[i]
            self.Cpsat[node_idx] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.offw_len):
            x_idx = static.offw_offset + i
            node_idx = static.offw_nodes[i]
            self.Coffw[node_idx] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.onsw_len):
            x_idx = static.onsw_offset + i
            node_idx = static.onsw_nodes[i]
            self.Consw[node_idx] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.biog_len):
            x_idx = static.biog_offset + i
            node_idx = static.biog_nodes[i]
            self.Cpeak[node_idx, 1] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.biom_len):
            x_idx = static.biom_offset + i
            node_idx = static.biom_nodes[i]
            self.Cpeak[node_idx, 0] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.ccgt_len):
            x_idx = static.ccgt_offset + i
            node_idx = static.ccgt_nodes[i]
            self.Cpeak[node_idx, 2] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.nuke_len):
            x_idx = static.nuke_offset + i
            node_idx = static.nuke_nodes[i]
            self.Cnuke[node_idx] += x[x_idx] * static.abs_rel_scaler[x_idx]

        for i in range(static.nlte_len):
            x_idx = static.nlte_offset + i
            node_idx = static.nlte_nodes[i]
            _cap = x[x_idx] * static.abs_rel_scaler[x_idx]
            self.Cnuke[node_idx] += _cap
            self.Cnlte[node_idx] += _cap

        for i in range(static.php_len):
            x_idx = static.php_offset + i
            node_idx = static.php_nodes[i]
            _cap = x[x_idx] * static.abs_rel_scaler[x_idx]
            self.CnphP[node_idx] = _cap
            self.CstorageP[node_idx, 0] += _cap

        for i in range(static.b4p_len):
            x_idx = static.b4p_offset + i
            node_idx = static.b4p_nodes[i]
            _cap = x[x_idx] * static.abs_rel_scaler[x_idx]
            self.CstorageP[node_idx, 1] += _cap
            self.CstorageE[node_idx, 1] += 4.0 * _cap

        for i in range(static.b2p_len):
            x_idx = static.b2p_offset + i
            node_idx = static.b2p_nodes[i]
            _cap = x[x_idx] * static.abs_rel_scaler[x_idx]
            self.CstorageP[node_idx, 2] += _cap
            self.CstorageE[node_idx, 2] += 2.0 * _cap

        for i in range(static.phe_len):
            x_idx = static.phe_offset + i
            node_idx = static.phe_nodes[i]
            if static.relative_param:
                _cap = x[x_idx] * self.CnphP[node_idx]
                # enforce absolute energy bounds that aren't handled by relative parameterisation
                _cap = min(static.phe_max_e[node_idx], _cap)
                _cap = max(static.phe_min_e[node_idx], _cap)
            else:
                _cap = x[x_idx]
            self.CnphE[node_idx] += _cap
            self.CstorageE[node_idx, 0] += _cap

        for i in range(static.nhvi):
            x_idx = static.lines_offset + i
            self.Clines[i] += x[x_idx] * static.abs_rel_scaler[x_idx]

        # pondage, hydro, and pumped hydro
        self.Clongdur = self.ChydP[:, 0] + self.ChydP[:, 1] + self.CstorageP[:, 0]
        # 4 hour, 2 hour batteries
        self.Cshortdur = self.CstorageP[:, 1] + self.CstorageP[:, 2]


if JIT_ENABLED:
    AssetTensorType = AssetTensor.class_type.instance_type
else:
    AssetTensorType = AssetTensor
