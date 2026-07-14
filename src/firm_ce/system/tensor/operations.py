# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import boolean, nbfloat, npfloat, nbintp, npint, npintp, nbint

from firm_ce.system.tensor.static import StaticTensorType
from firm_ce.system.tensor.assets import AssetTensorType

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
        self.Mdeficit = np.zeros((intervals, nodes), dtype=npfloat)
        self.Mcurtail = np.zeros((intervals, nodes), dtype=npfloat)

        self.Mimport = np.zeros((intervals, nodes), dtype=npfloat)
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


if JIT_ENABLED:
    OperationTensorType = OperationTensor.class_type.instance_type
else:
    OperationTensorType = OperationTensor
