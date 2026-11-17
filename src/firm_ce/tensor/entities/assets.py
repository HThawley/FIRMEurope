# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit
from firm_ce.common.typing import nbfloat, npfloat
from firm_ce.tensor.entities.static import StaticTensorType

if JIT_ENABLED:
    asset_spec = [
        # Capacities in GW/GWh
        ("Cpfix", nbfloat[:]),
        ("Cpsat", nbfloat[:]),
        ("Coffw", nbfloat[:]),
        ("Consw", nbfloat[:]),
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_assets(
    assets,
    x: np.ndarray[npfloat],
    static: StaticTensorType,
) -> None:
    """
    Reinitialises a pre-allocated AssetTensor in-place for a new decision vector `x`.
    Zero heap allocations.
    """
    nodes = static.nodes

    assets.Cpfix.fill(npfloat(0.0))
    assets.Cpsat.fill(npfloat(0.0))
    assets.Coffw.fill(npfloat(0.0))
    assets.Consw.fill(npfloat(0.0))
    assets.Cpeak.fill(npfloat(0.0))
    assets.Cnuke[:] = static.Enuke
    assets.Cnlte.fill(npfloat(0.0))
    assets.ChydP[:, :] = static.EhydP
    assets.ChydE[:, :] = static.EhydE
    assets.CnphP.fill(npfloat(0.0))
    assets.CnphE.fill(npfloat(0.0))
    assets.CstorageP[:, :] = static.EstorageP
    assets.CstorageE[:, :] = static.EstorageE
    assets.Clines[:] = static.Elines

    rel = static.relative_param
    scaler = static.abs_rel_scaler

    for i in range(static.pfix_len):
        x_idx = static.pfix_offset + i
        node_idx = static.pfix_nodes[i]
        assets.Cpfix[node_idx] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.psat_len):
        x_idx = static.psat_offset + i
        node_idx = static.psat_nodes[i]
        assets.Cpsat[node_idx] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.offw_len):
        x_idx = static.offw_offset + i
        node_idx = static.offw_nodes[i]
        assets.Coffw[node_idx] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.onsw_len):
        x_idx = static.onsw_offset + i
        node_idx = static.onsw_nodes[i]
        assets.Consw[node_idx] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.biog_len):
        x_idx = static.biog_offset + i
        node_idx = static.biog_nodes[i]
        assets.Cpeak[node_idx, 1] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.biom_len):
        x_idx = static.biom_offset + i
        node_idx = static.biom_nodes[i]
        assets.Cpeak[node_idx, 0] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.ocgt_len):
        x_idx = static.ocgt_offset + i
        node_idx = static.ocgt_nodes[i]
        assets.Cpeak[node_idx, 2] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.nuke_len):
        x_idx = static.nuke_offset + i
        node_idx = static.nuke_nodes[i]
        assets.Cnuke[node_idx] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for i in range(static.nlte_len):
        x_idx = static.nlte_offset + i
        node_idx = static.nlte_nodes[i]
        _cap = x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))
        assets.Cnuke[node_idx] += _cap
        assets.Cnlte[node_idx] += _cap

    for i in range(static.php_len):
        x_idx = static.php_offset + i
        node_idx = static.php_nodes[i]
        _cap = x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))
        assets.CnphP[node_idx] = _cap
        assets.CstorageP[node_idx, 1] += _cap

    for i in range(static.b4p_len):
        x_idx = static.b4p_offset + i
        node_idx = static.b4p_nodes[i]
        _cap = x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))
        assets.CstorageP[node_idx, 2] += _cap
        assets.CstorageE[node_idx, 2] += 4.0 * _cap

    for i in range(static.b2p_len):
        x_idx = static.b2p_offset + i
        node_idx = static.b2p_nodes[i]
        _cap = x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))
        assets.CstorageP[node_idx, 3] += _cap
        assets.CstorageE[node_idx, 3] += 2.0 * _cap

    for i in range(static.phe_len):
        x_idx = static.phe_offset + i
        node_idx = static.phe_nodes[i]
        _cap = x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))
        assets.CnphE[node_idx] += _cap
        assets.CstorageE[node_idx, 1] += _cap

    for i in range(static.nhvi):
        x_idx = static.lines_offset + i
        assets.Clines[i] += x[x_idx] * (scaler[x_idx] if rel else npfloat(1.0))

    for n in range(nodes):
        # reservoir hydro (h=1), open-loop PHES (s=0), and closed-loop PHES (s=1)
        assets.Clongdur[n] = assets.ChydP[n, 1] + assets.CstorageP[n, 0] + assets.CstorageP[n, 1]
        # pondage (h=0), 4 hour (s=2), 2 hour (s=3) batteries
        assets.Cshortdur[n] = assets.ChydP[n, 0] + assets.CstorageP[n, 2] + assets.CstorageP[n, 3]


@jitclass(asset_spec)
class AssetTensor:
    def __init__(
        self,
        x: np.ndarray[npfloat],
        static: StaticTensorType,
    ):
        nodes = static.nodes

        self.Cpfix = np.empty(nodes, dtype=npfloat)
        self.Cpsat = np.empty(nodes, dtype=npfloat)
        self.Coffw = np.empty(nodes, dtype=npfloat)
        self.Consw = np.empty(nodes, dtype=npfloat)
        self.Cpeak = np.empty((nodes, static.npeak), dtype=npfloat)
        self.Cnuke = np.empty(nodes, dtype=npfloat)
        self.Cnlte = np.empty(nodes, dtype=npfloat)
        self.ChydP = np.empty((nodes, static.nhyd), dtype=npfloat)
        self.ChydE = np.empty((nodes, static.nhyd), dtype=npfloat)
        self.CnphP = np.empty(nodes, dtype=npfloat)
        self.CnphE = np.empty(nodes, dtype=npfloat)
        self.CstorageP = np.empty((nodes, static.nstor), dtype=npfloat)
        self.CstorageE = np.empty((nodes, static.nstor), dtype=npfloat)
        self.Clines = np.empty(static.nhvi, dtype=npfloat)
        self.Clongdur = np.empty(nodes, dtype=npfloat)
        self.Cshortdur = np.empty(nodes, dtype=npfloat)

        reset_assets(self, x, static)


if JIT_ENABLED:
    AssetTensorType = AssetTensor.class_type.instance_type
else:
    AssetTensorType = AssetTensor
