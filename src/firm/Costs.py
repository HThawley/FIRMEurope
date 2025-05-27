import numpy as np
from numba import boolean, float64, int64, njit  # type: ignore
from numba.experimental import jitclass  # type: ignore
from firm.Input import NetLength, NetCost

USD_to_AUD = 1 / 0.65  # AUD to USD where necessary
discount_rate = 0.0599  # Real discount rate - same as gencost
USD_inflation = 1.18  # 2020->2023
AUD_inflation = 1.16  # 2020->2023
MWh_per_GJ = 0.27778
carbon_price = 0  # AUD/tCO2e
tCO2e_per_GJ_gas = 0.05
tCO2e_per_GJ_coal = 0.1


# costs come from Apx Table B.9 of GenCost 2023-24
# year = 2023
# ==============================================================================
# utility solar
csiro_pv = (
    1526,  # capex AUD/kW
    17,  # fom   AUD/kW p.a.
    0,  # vom   AUD/MWh
    30,  # life  years
)

# onshore wind
csiro_onsw = (
    3038,  # capex AUD/kW
    25,  # fom   AUD/kW p.a.
    0,  # vom   AUD/MWh
    25,  # life  years
)

# offshore wind
csiro_offw = (
    5545,  # capex AUD/kW
    149.9,  # fom   AUD/kW
    0,  # vom   AUD/MWh
    25,  # life  years
)

# large open cycle gas
csiro_gas = (
    943,  # capex AUD/kW
    10.2,  # fom   AUD/kW p.a.
    7.3,  # vom   AUD/MWh
    16.5 / 0.33 / MWh_per_GJ,  # fuel AUD/MWh
    tCO2e_per_GJ_gas / MWh_per_GJ * carbon_price,  # carbon intensity tCO2e/MWh
    25,  # life  years
)

# black coal
csiro_coal = (
    5616,  # capex AUD/kW
    53.2,  # fom   AUD/kW p.a.
    4.2,  # vom   AUD/MWh
    7.8 / 0.42 / MWh_per_GJ,  # fuel AUD/MWh
    tCO2e_per_GJ_coal / MWh_per_GJ * carbon_price,  # carbon intensity tCO2e/MWh
    30,  # life  years
)

# costs adjusted for inflation but otherwise unchanged from Lu et al. 2021 https://doi.org/10.1016/j.energy.2020.119678
# ==============================================================================
# TODO: find costs for this
interconnector = (
    160 * AUD_inflation,  # capex AUD/kW
    1.6 * AUD_inflation,  # fom   AUD/kW p.a.
    0,  # vom   AUD/MWh
    30,  # life  years
)

# undersea costs includer converter
hv_undersea = (
    4000 * AUD_inflation,  # capex AUD/MW-km
    40 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,  # vom   AUD/MWh
    30,  # life  years
)

hvac = (
    1500 * AUD_inflation,  # capex AUD/MW-km
    15 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,  # vom   AUD/MWh
    50,  # life  years
)

# costs from re100 cost model - Class AA site
# 1 GW / 160 GWh
# 700 m head, 10 km separation
# 8.0 W/R
# ==============================================================================
phes = (
    1164 * USD_inflation * USD_to_AUD,  # capex AUD/kW
    15 * USD_inflation * USD_to_AUD,  # capex AUD/kWh
    8.21 * USD_inflation * USD_to_AUD,  # fom AUD/kW p.a.
    0.6 * USD_inflation * USD_to_AUD,  # vom AUD/MWh
    112000 * USD_inflation * USD_to_AUD,  # AUD per replace
    50,  # replace lifetime
    100,  # life years
)

# same O&M as PHES, but no capital
hydro = (
    0,  # capex (existing only)
    8.21 * USD_inflation * USD_to_AUD,  # fom AUD/kW p.a. #same as phes (approx)
    0.3 * USD_inflation * USD_to_AUD / 2,  # vom AUD/MWh #half phes (one-way trip) (approx)
    50,  # life years
)

# Battery costs
# Lazard LCOE+ 2 Hour utility battery
battery = (
    45 * USD_to_AUD,  # capex AUD/kW
    294 * USD_to_AUD,  # capex AUD/kWh
    5.35 * USD_to_AUD,  # fom AUD/kWh
    0,  # vom
    10,  # life years
)


@njit
def present_value(life, dr):
    return (1 - (1 + dr) ** (-1 * life)) / dr


@njit
def annualization(capex, fom, vom, life, dr):
    """
    Calculate annualized costs parametrically for power and energy
    Input:
        capex - $/kW
        fom   - $/kW p.a.
        vom   - $/MWh
        life  - years
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    return np.array(
        [
            1_000_000 * capex / present_value(life, dr),  # $ p.a./GW
            1_000_000 * fom,  # $ p.a./GW
            1000 * vom,  # $ p.a./GWh p.a.
        ],
        np.float64,
    )


@njit
def annualization_transmission(capex, fom, vom, life, d, dr):
    """
    Calculate annualized costs parametrically for power and energy, for transmission lines only
    Input:
        capex - $/MW-km
        fom   - $/MW-km p.a.
        vom   - $/MWh
        life  - years
        d     - km
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    return np.array(
        [
            d * capex * 1000 / present_value(life, dr),  # $ p.a./GW
            d * fom * 1000,  # $ p.a./GW
            vom * 1000,  # $ p.a./GWh p.a.
        ]
    )


@njit
def annualization_phes(capex_p, capex_e, fom, vom, replace_cost, replace_life, life, dr):
    """Calculate annualized costs parametrically for power and energy, for PHES only
    capex_p, fom: AUD/kW
    capex_e: AUD/kWh
    fom: AUD/kW p.a.
    vom: AUD/MWh p.a.
    replace: AUD per replace
    replace_life: years"""
    pv = present_value(life, dr)
    return np.array(
        [
            capex_p * 1_000_000 / pv,  # capex $ p.a./GW
            capex_e * 1_000_000 / pv,  # capex $ p.a./GWh
            fom * 1_000_000,  # fom $ p.a./GW
            vom * 1000,  # vom $ p.a./GWh p.a.
            # TODO: check this
            replace_cost
            * ((1 + dr) ** (-1 * replace_cost) + (1 + dr) ** (-1 * replace_life * 2))
            / pv,  # replace capex $p.a.
        ]
    )


@njit
def annualization_battery(capex_p, capex_e, fom, vom, life, dr):
    """Calculate annualized costs parametrically for power and energy, for batteries only
    capex_p, fom: AUD/kW
    capex_e: AUD/kWh
    fom: AUD/kWh p.a.
    vom: AUD/MWh
    life: years"""
    pv = present_value(life, dr)
    return np.array(
        [
            capex_p * 1_000_000 / pv,  # capex $ p.a./GW
            capex_e * 1_000_000 / pv,  # capex $ p.a./GWh
            fom * 1_000_000,  # fom $ p.a./GW
            vom * 1000,  # vom $ p.a./GWh p.a.
        ]
    )


@njit
def annualization_fossils(capex, fom, vom, fuel, carbon, life, dr):
    """
    Calculate annualized costs parametrically for power and energy
    Input:
        capex - $/kW
        fom   - $/kW p.a.
        vom   - $/MWh
        fuel  - $/MWh
        life  - years
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    return np.array(
        [
            1_000_000 * capex / present_value(life, dr),  # $ p.a./GW
            1_000_000 * fom,  # $ p.a./GW
            1000 * vom + 1000 * fuel + 1000 * carbon,  # $ p.a./GWh p.a.
        ],
        np.float64,
    )


@jitclass(
    [
        ("carbon_price", float64),
        ("dr", float64),
        ("pv", float64[:]),
        ("onsw", float64[:]),
        ("offw", float64[:]),
        ("gas", float64[:]),
        ("coal", float64[:]),
        ("hydro", float64[:]),
        ("phes", float64[:]),
        ("battery", float64[:]),
        ("hvac", float64[:]),
        ("hvi", float64[:]),
        ("hvu", float64[:]),
        ("scenario", int64),
        ("network_mask", boolean[:]),
    ]
)
class RawCosts:
    def __init__(
        self,
        solution_data
    ):
        self.scenario = solution_data.scenario
        self.network_mask = solution_data.network_mask

        self.pv = np.array(csiro_pv, np.float64)
        self.onsw = np.array(csiro_onsw, np.float64)
        self.offw = np.array(csiro_offw, np.float64)
        self.gas = np.array(csiro_gas, np.float64)
        self.coal = np.array(csiro_coal, np.float64)
        self.hydro = np.array(hydro, np.float64)
        self.phes = np.array(phes, np.float64)
        self.battery = np.array(battery, np.float64)
        self.hvac = np.array(hvac, np.float64)
        # self.hvi = np.array(interconnector, np.float64)
        self.hvu = np.array(hv_undersea, np.float64)

        self.dr = discount_rate
        self.UpdateCarbonPrice(carbon_price)

    def UpdateCarbonPrice(self, price):
        self.gas[4] = tCO2e_per_GJ_gas / MWh_per_GJ * price
        self.coal[4] = tCO2e_per_GJ_coal / MWh_per_GJ * price

    def GetCostFactors(self):
        return CostFactors(self)


@jitclass(
    [
        ("pv", float64[:]),
        ("onsw", float64[:]),
        ("offw", float64[:]),
        ("gas", float64[:]),
        ("coal", float64[:]),
        ("hydro", float64[:]),
        ("phes", float64[:]),
        ("battery", float64[:]),
        ("ac", float64[:]),
        ("hvi", float64[:, :]),
    ]
)
class CostFactors:
    def __init__(self, rc):
        self.pv = annualization(rc.pv[0], rc.pv[1], rc.pv[2], rc.pv[3], rc.dr)
        self.onsw = annualization(
            rc.onsw[0], rc.onsw[1], rc.onsw[2], rc.onsw[3], rc.dr
        )
        self.offw = annualization(
            rc.offw[0], rc.offw[1], rc.offw[2], rc.offw[3], rc.dr
        )

        self.gas = annualization_fossils(
            rc.gas[0],
            rc.gas[1],
            rc.gas[2],
            rc.gas[3],
            rc.gas[4],
            rc.gas[5],
            rc.dr,
        )
        self.coal = annualization_fossils(
            rc.coal[0],
            rc.coal[1],
            rc.coal[2],
            rc.coal[3],
            rc.coal[4],
            rc.coal[5],
            rc.dr,
        )

        self.hydro = annualization(
            rc.hydro[0], rc.hydro[1], rc.hydro[2], rc.hydro[3], rc.dr
        )
        self.phes = annualization_phes(
            rc.phes[0],
            rc.phes[1],
            rc.phes[2],
            rc.phes[3],
            rc.phes[4],
            rc.phes[5],
            rc.phes[6],
            rc.dr,
        )

        self.battery = annualization_battery(
            rc.battery[0],
            rc.battery[1],
            rc.battery[2],
            rc.battery[3],
            rc.battery[4],
            rc.dr,
        )

        self.ac = annualization_transmission(
            rc.hvac[0], rc.hvac[1], rc.hvac[2], rc.hvac[3], 20, rc.dr
        )

        self.hvi = np.zeros((len(rc.network_mask), 3), np.float64)
        for i in range(len(rc.network_mask)):
            self.hvi[i] = annualization_transmission(
                NetCost[i], 0, 0, 30, NetLength[i], rc.dr
            )
        self.hvi = self.hvi.T

if __name__=='__main__':
    from firm.Parameters import Parameters
    from firm.Input import SolutionData
    parameters = Parameters(21, 1, False, 4)
    sd = SolutionData(*parameters)
    costFactors = RawCosts(sd).GetCostFactors()
    