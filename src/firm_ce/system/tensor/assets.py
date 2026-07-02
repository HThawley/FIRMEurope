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
