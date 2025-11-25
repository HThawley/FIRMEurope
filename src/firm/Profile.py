from numba import float64, int64  # type: ignore
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
]


@jitclass(profile_function_spec)
class ProfileContainer:
    def __init__(self, level):
        if level == 1 or level == -1:
            self.simulation = 0
        if level == 2 or level == -1:
            self.basic = 0
            self.trans_precharge = 0
        if level == 3 or level == -1:
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

    def _get_adj_simulation(self):
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


ProfileContainerType = ProfileContainer.class_type.instance_type


profile_spec = [
    ("level", int64),
    ("profile_overhead", float64),
    ("times", ProfileContainerType),
    ("calls", ProfileContainerType),
]


@jitclass(profile_spec)
class ProfileData:
    def __init__(self, level=0, overhead=0.0):
        self.profile_overhead = overhead
        self.level = level
        self.times = ProfileContainer(self.level)
        self.calls = ProfileContainer(self.level)

    def open_adj_simulation(self):
        self.times.adj_simulation = self.times._get_adj_simulation()
        self.calls.adj_simulation = self.calls._get_adj_simulation()

    def close_adj_simulation(self):
        adj_t = self.times._get_adj_simulation()
        adj_c = self.calls._get_adj_simulation()
        self.times.simulation += (self.times.adj_simulation - adj_t)
        self.calls.simulation += (self.calls.adj_simulation - adj_c) * self.profile_overhead


ProfileDataType = ProfileData.class_type.instance_type
