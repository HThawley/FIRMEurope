from numba import int64  # type: ignore
from numba.experimental import jitclass  # type: ignore


profile_function_spec = [
    ("simulation", int64),
    ("adj_simulation", int64),
    ("basic", int64),
    ("interc0", int64),
    ("interc1", int64),
    ("interc2", int64),
    ("interc3", int64),
    ("storage_behavior", int64),
    ("storage_behaviort", int64),
    ("spilldef", int64),
    ("spilldeft", int64),
    ("update_soc", int64),
    ("update_soct", int64),
    ("unbalancedt", int64),
    ("unbalanced", int64),
    ("clip_fill", int64),
    ("trans_precharge", int64),
    ("get_surplus", int64),
    ("overhead", int64),
]


@jitclass(profile_function_spec)
class ProfileContainer:
    def __init__(self):
        self.simulation = 0
        self.basic = 0
        self.trans_precharge = 0
        self.interc0 = 0
        self.interc1 = 0
        self.interc2 = 0
        self.interc3 = 0
        self.storage_behavior = 0
        self.storage_behaviort = 0
        self.spilldef = 0
        self.spilldeft = 0
        self.update_soc = 0
        self.update_soct = 0
        self.unbalancedt = 0
        self.unbalanced = 0
        self.clip_fill = 0
        self.get_surplus = 0
        self.overhead = 0

    def get_adj_simulation(self):
        return (
            self.interc0
            + self.interc1
            + self.interc2
            + self.interc3
            + self.storage_behaviort
            + self.spilldeft
            + self.unbalancedt
            + self.update_soct
        )

    def get_total(self):
        return (
            self.simulation
            + self.basic
            + self.trans_precharge
            + self.interc0
            + self.interc1
            + self.interc2
            + self.interc3
            + self.storage_behavior
            + self.storage_behaviort
            + self.spilldef
            + self.spilldeft
            + self.update_soc
            + self.update_soct
            + self.unbalancedt
            + self.unbalanced
            + self.clip_fill
            + self.get_surplus
            # overhead not counted
        )

    def apply_overhead(self, overhead, calls):
        self.simulation -= overhead * calls.simulation
        self.basic -= overhead * calls.basic
        self.trans_precharge -= overhead * calls.trans_precharge
        self.interc0 -= overhead * calls.interc0
        self.interc1 -= overhead * calls.interc1
        self.interc2 -= overhead * calls.interc2
        self.interc3 -= overhead * calls.interc3
        self.storage_behavior -= overhead * calls.storage_behavior
        self.storage_behaviort -= overhead * calls.storage_behaviort
        self.spilldef -= overhead * calls.spilldef
        self.spilldeft -= overhead * calls.spilldeft
        self.update_soc -= overhead * calls.update_soc
        self.update_soct -= overhead * calls.update_soct
        self.unbalancedt -= overhead * calls.unbalancedt
        self.unbalanced -= overhead * calls.unbalanced
        self.clip_fill -= overhead * calls.clip_fill
        self.get_surplus -= overhead * calls.get_surplus


ProfileContainerType = ProfileContainer.class_type.instance_type


profile_spec = [
    ("times", ProfileContainerType),
    ("calls", ProfileContainerType),
    ("simulation_adj_t", int64),
    ("simulation_adj_c", int64),
]


@jitclass(profile_spec)
class ProfileData:
    def __init__(self):
        self.times = ProfileContainer()
        self.calls = ProfileContainer()
        self.simulation_adj_t = 0
        self.simulation_adj_c = 0

    def open_adj_simulation(self):
        self.times.adj_simulation = self.times.get_adj_simulation()
        self.calls.adj_simulation = self.calls.get_adj_simulation()

    def close_adj_simulation(self):
        adj_t = self.times.get_adj_simulation()
        adj_c = self.calls.get_adj_simulation()
        self.simulation_adj_t += (self.times.adj_simulation - adj_t)
        self.simulation_adj_t += (self.calls.adj_simulation - adj_c)

    def apply_overhead(self, overhead):
        self.times.apply_overhead(overhead, self.calls)


ProfileDataType = ProfileData.class_type.instance_type
