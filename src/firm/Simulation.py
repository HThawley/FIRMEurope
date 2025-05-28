# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
from firm.Utils import (
    cclock, 
    array_sum_2d_axis1, 
    zero_safe_division,
)  #type: ignore 

@njit
def Instantiate(solution):
    solution.MNetload = solution.MLoad - solution.MPV - solution.MWind - solution.MRor
    solution.MUnbalanced = solution.MNetload.copy()
    solution.MDeficit, solution.MSpillage = np.maximum(0, solution.MNetload), -np.minimum(0, solution.MNetload)

    solution.MHydro = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MGas = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)

    solution.MDischarge = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MCharge = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MStorage = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MStorage[-1] = 0.5 * solution.CPHS

    solution.TImport = np.zeros((solution.intervals, solution.nodes, solution.nhvi), dtype=np.float64)
    solution.TExport = np.zeros((solution.intervals, solution.nodes, solution.nhvi), dtype=np.float64)
    # solution.THVI = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)


@njit
def Simulate(solution):
    Instantiate(solution)
    if solution.profiling:
        time_adj = (solution.time_interconnection0+
                    solution.time_interconnection1+
                    solution.time_interconnection2+
                    solution.time_interconnection3+
                    solution.time_spilldeft+
                    solution.time_unbalancedt)
        calls_adj = (solution.calls_interconnection0+
                     solution.calls_interconnection1+
                     solution.calls_interconnection2+
                     solution.calls_interconnection3+
                     solution.calls_spilldeft+
                     solution.calls_unbalancedt)
        start = cclock()
        
    ### Maximum Dispatch of Gas
    solution.MGas = solution.GGas * np.ones((solution.intervals, solution.nodes), np.float64)
    TransmissionSimulate(solution)
    
    ### Dispatch Hydro to fill deficits
    FillSimulate(solution, solution.EHydro, solution.MHydro)
    
    # Energy units by which a node exceeds in Hydro resource 
    flex_exceedance = np.zeros(solution.nodes, np.float64)
    factor = np.zeros(solution.nodes, np.float64)
    for n in range(solution.nodes):
        for t in range(solution.intervals):
            factor[n] += solution.MHydro[t, n]
        flex_exceedance[n] = factor[n] - solution.RHydro[n]
    for n in range(solution.nodes):
        factor[n] = min(1, zero_safe_division(factor[n] - flex_exceedance[n], factor[n]))
    for t in range(solution.intervals):
        for n in range(solution.nodes):
            solution.MHydro[t, n] *= factor[n]
    for n in range(solution.nodes):
        _mhydro = 0.0
        for t in range(solution.intervals):
            _mhydro += solution.MHydro[t, n]
        flex_exceedance[n] = _mhydro - solution.RHydro[n]

    solution.MGas[:] = 0
    solution.TImport[:] = 0
    solution.TExport[:] = 0
    
    UpdateUnbalanced(solution)
    UpdateSpillDef(solution)
    TransmissionSimulate(solution)
    
    # Disptach gas 
    FillSimulate(solution, solution.GGas, solution.MGas)

    ### Redipatch as much as gas as possible to hydro
    _mhydro = np.zeros(solution.intervals, np.float64)
    for n in range(solution.nodes):
        if flex_exceedance[n] < -10:
            _mhfactor = 0.0
            for t in range(solution.intervals):
                _mhydro[t] = min(solution.MGas[t, n], solution.EHydro[n] - solution.MHydro[t, n])
                _mhfactor += _mhydro[t]
            _mhfactor = max(1, zero_safe_division(flex_exceedance[n], _mhfactor))
            for t in range(solution.intervals):
                _redispatch = _mhydro[t] * _mhfactor
                solution.MHydro[t, n] += _redispatch
                solution.MGas[t, n] -= _redispatch
                
    if solution.profiling:
        solution.calls_backfill +=1 
        solution.time_backfill += cclock() - start
        time_adj -= (solution.time_interconnection0+
                     solution.time_interconnection1+
                     solution.time_interconnection2+
                     solution.time_interconnection3+
                     solution.time_spilldeft+
                     solution.time_unbalancedt)
        calls_adj -= (solution.calls_interconnection0+
                      solution.calls_interconnection1+
                      solution.calls_interconnection2+
                      solution.calls_interconnection3+
                      solution.calls_spilldeft+
                      solution.calls_unbalancedt)
        solution.time_backfill += time_adj
        solution.time_backfill += calls_adj * solution.profile_overhead

    UpdateUnbalanced(solution)
    UpdateSpillDef(solution)
    BasicSimulate(solution)

@njit 
def FillSimulate(solution, CFlex, MFlex):
    
    for t in range(solution.intervals):
        for n in range(solution.nodes):
            MFlex[t, n] = min(solution.MDeficit[t, n], CFlex[n])
        UpdateUnbalancedt(solution, t)
        UpdateSpillDeft(solution, t)

    for t in range(solution.intervals - 1, -1, -1):
        # timestep backwards
        if solution.MDeficit[t].sum() > 1e-6:
            # original import/export
            _import = array_sum_2d_axis1(solution.TImport[t])
            _export = array_sum_2d_axis1(solution.TExport[t])
            # meet deficits just-in-time by importiing flex from neighbours
            Interconnection(
                solution,
                solution.MDeficit[t],
                CFlex - MFlex[t],
                solution.TImport[t],
                solution.TExport[t],
            )
            # flexible += im/exports from neighbours
            MFlex[t] += np.maximum(
                (_import + _export - array_sum_2d_axis1(solution.TImport[t] + solution.TExport[t])), 0
                )
   
    fill = np.zeros(solution.nodes, np.float64)
    flex = 0.0
    for t in range(solution.intervals - 1, -1, -1):
        # timestep backwards
        if solution.MDeficit[t].sum() > 1e-6:
            # original import/export
            _import = array_sum_2d_axis1(solution.TImport[t])
            _export = array_sum_2d_axis1(solution.TExport[t])
            # meet deficits just-in-time by importiing flex from neighbours
            Interconnection(
                solution,
                solution.MDeficit[t],
                CFlex - MFlex[t],
                solution.TImport[t],
                solution.TExport[t],
            )
            # flexible += iexports from neighbours
            MFlex[t] += np.maximum(
                (_import + _export - array_sum_2d_axis1(solution.TImport[t] + solution.TExport[t])), 0
            )
            # accumulate remaing deficits
            fill += solution.MDeficit[t] / solution.efficiency
        if fill.sum() > 1e-6:
            # clip fill by storage capacity
            for n in range(solution.nodes):
                fill[n] = min(fill[n], (solution.CPHS[n] - solution.MStorage[t - 1, n]) / solution.resolution / solution.efficiency)
                flex = min(fill[n], CFlex[n] - MFlex[t, n], solution.CPHP[n] - solution.MCharge[t, n] + solution.MDischarge[t, n])
                fill[n] -= flex
                MFlex[t, n] += flex

            if fill.sum() > 1e-6:
                _import = array_sum_2d_axis1(solution.TImport[t])
                _export = array_sum_2d_axis1(solution.TExport[t])
                Interconnection(
                    solution, fill, CFlex - MFlex[t], solution.TImport[t], solution.TExport[t]
                )
                MFlex[t] += np.maximum(
                    (_import + _export - array_sum_2d_axis1(solution.TImport[t] + solution.TExport[t])), 0
                )
    

@njit
def TransmissionSimulate(solution):
    if solution.profiling:
        time_adj = (solution.time_interconnection0+
                    solution.time_interconnection1+
                    solution.time_interconnection2+
                    solution.time_interconnection3+
                    solution.time_storage_behaviort+
                    solution.time_spilldeft+
                    solution.time_unbalancedt+
                    solution.time_update_soct)
        calls_adj = (solution.calls_interconnection0+
                     solution.calls_interconnection1+
                     solution.calls_interconnection2+
                     solution.calls_interconnection3+
                     solution.calls_storage_behaviort+
                     solution.calls_spilldeft+
                     solution.calls_unbalancedt+
                     solution.calls_update_soct)
        start = cclock()
    Surplust = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals):
        # storage operation
        UpdateStoraget(solution, t)
        UpdateSpillDeft(solution, t)

        # fill deficits from spilled power
        if (solution.MDeficit[t] > 1e-6).any():
            if (solution.MSpillage[t] > 1e-6).any():
                Interconnection(
                    solution, solution.MDeficit[t], solution.MSpillage[t], solution.TImport[t], solution.TExport[t]
                )
                # update storage behaviour
                UpdateUnbalancedt(solution, t)
                UpdateStoraget(solution, t)
                # UpdateSpillDeft(solution, t) # updated inplace by Interconnection
 
        # fill deficits by drawing down neighbours' storage reserves
        if (solution.MDeficit[t] > 1e-6).any():
            for n in range(solution.nodes):
                Surplust[n] = max(0, 
                    solution.MSpillage[t, n]
                    + solution.MCharge[t, n]
                    + min(solution.CPHP[n], solution.MStorage[t - 1, n] / solution.resolution)
                    - solution.MDischarge[t, n])
            if (Surplust > 1e-6).any():
                Interconnection(solution, solution.MDeficit[t], Surplust, solution.TImport[t], solution.TExport[t])
                # update storage behaviour
                UpdateUnbalancedt(solution, t)
                UpdateStoraget(solution, t)
                UpdateSpillDeft(solution, t)

        UpdateSOCt(solution, t)

    # precharge batteries with spillage only 
    fill = np.zeros(solution.nodes, dtype=np.float64)
    # timestep backwards
    for t in range(solution.intervals - 1, -1, -1):
        if (fill > 1e-6).any():
            if (solution.MSpillage[t] > 1e-6).any():
                # cap fill by storage capacity
                for n in range(solution.nodes):
                    fill[n] = min(fill[n], (solution.CPHS[n] - solution.MStorage[t - 1, n]) / solution.resolution / solution.efficiency)
                # meet fill with neighbours' spillage - don't draw down power as this affects future SOC
                Interconnection(solution, fill, solution.MSpillage[t], solution.TImport[t], solution.TExport[t])
                # fill adjusted in-place
        for n in range(solution.nodes):
            fill[n] += solution.MDeficit[t, n] / solution.efficiency

    if solution.profiling:
        solution.calls_transmission +=1 
        solution.time_transmission += cclock() - start 
        time_adj -= (solution.time_interconnection0+
                     solution.time_interconnection1+
                     solution.time_interconnection2+
                     solution.time_interconnection3+
                     solution.time_storage_behaviort+
                     solution.time_spilldeft+
                     solution.time_unbalancedt+
                     solution.time_update_soct)
        calls_adj -= (solution.calls_interconnection0+
                      solution.calls_interconnection1+
                      solution.calls_interconnection2+
                      solution.calls_interconnection3+
                      solution.calls_storage_behaviort+
                      solution.calls_spilldeft+
                      solution.calls_unbalancedt+
                      solution.calls_update_soct)
        solution.time_transmission += time_adj
        solution.time_transmission += calls_adj * solution.profile_overhead
        

    # fix storage traces
    BasicSimulate(solution)


@njit
def BasicSimulate(solution):
    # if solution.profiling:
    #     start_basic = cclock()
    for t in range(solution.intervals):
        UpdateUnbalancedt(solution, t)
        UpdateStoraget(solution, t)
        UpdateSOCt(solution, t)
        UpdateSpillDeft(solution, t)
    # UpdateUnbalanced(solution)
    # UpdateStorage(solution)
    # UpdateSOC(solution)
    # UpdateSpillDef(solution)
    # if solution.profiling:
    #     solution.time_basic += cclock() - start_basic
    #     solution.calls_basic +=1 


@njit
def UpdateUnbalancedt(solution, t):
    if solution.profiling:
        start = cclock()
    for n in range(solution.nodes):
        _timport = 0.0
        for m in range(solution.nhvi):
            _timport += solution.TImport[t,n,m] 
            _timport += solution.TExport[t,n,m]
        solution.MUnbalanced[t,n] = solution.MNetload[t,n] - solution.MHydro[t,n] - solution.MGas[t,n] - _timport
    if solution.profiling:
        solution.calls_unbalancedt +=1 
        solution.time_unbalancedt += cclock() - start
        


@njit
def UpdateUnbalanced(solution):
    if solution.profiling:
        start = cclock()
    
    for t in range(solution.intervals):
        for n in range(solution.nodes):
            _timport = 0.0
            for m in range(solution.nhvi):
                _timport += solution.TImport[t, n, m]
                _timport += solution.TExport[t, n, m]
            solution.MUnbalanced[t, n] = solution.MNetload[t, n] - solution.MHydro[t,n] - solution.MGas[t,n] - _timport

    if solution.profiling:
        solution.calls_unbalanced +=1 
        solution.time_unbalanced += cclock() - start
        


@njit
def UpdateStoraget(solution, t):
    if solution.profiling:
        start = cclock()
    for n in range(solution.nodes):
        solution.MCharge[t, n] = min(-min(0,solution.MUnbalanced[t, n]), solution.CPHP[n], (solution.CPHS[n] - solution.MStorage[t - 1, n]) / solution.efficiency / solution.resolution)
        solution.MDischarge[t, n] = min(max(0, solution.MUnbalanced[t, n]), solution.CPHP[n], solution.MStorage[t - 1, n] / solution.resolution)

    if solution.profiling:
        solution.calls_storage_behaviort +=1 
        solution.time_storage_behaviort += cclock() - start
        


@njit
def UpdateStorage(solution):
    if solution.profiling:
        start = cclock()

    for t in range(solution.intervals):
        for n in range(solution.nodes):
            solution.MCharge[t, n] = min(-min(solution.MUnbalanced[t, n], 0), solution.CPHP[n])
            solution.MDischarge[t, n] = min(max(solution.MUnbalanced[t, n], 0), solution.CPHP[n])

    if solution.profiling:
        solution.calls_storage_behavior +=1 
        solution.time_storage_behavior += cclock() - start
        

    
@njit(fastmath=True)
def UpdateSOCt(solution, t):
    if solution.profiling:
       start = cclock()
    for n in range(solution.nodes):
        solution.MStorage[t, n] = solution.MStorage[t - 1, n] + solution.resolution * (solution.MCharge[t, n] * solution.efficiency - solution.MDischarge[t, n])
    if solution.profiling:
        solution.calls_update_soct +=1 
        solution.time_update_soct += cclock() - start
        

@njit(fastmath=True)
def UpdateSOC(solution):
    if solution.profiling:
        start = cclock()
    solution.MStorage[-1] = 0.5 * solution.CPHS 
    for t in range(solution.intervals):
        for n in range(solution.nodes):
            solution.MCharge[t, n] = min(solution.MCharge[t, n], (solution.CPHS[n] - solution.MStorage[t - 1, n]) / solution.efficiency / solution.resolution)
            solution.MDischarge[t, n] = min(solution.MDischarge[t, n], solution.MStorage[t - 1, n] / solution.resolution)
            solution.MStorage[t, n] = solution.MStorage[t - 1, n] + solution.resolution * (solution.MCharge[t, n] * solution.efficiency - solution.MDischarge[t, n])
    if solution.profiling:
        solution.calls_update_soc +=1 
        solution.time_update_soc += cclock() - start
        

@njit
def UpdateSpillDeft(solution, t):
    if solution.profiling:
        start = cclock()

    for n in range(solution.nodes):
        _inter = (solution.MUnbalanced[t, n] + solution.MCharge[t, n] - solution.MDischarge[t, n])
        solution.MDeficit[t, n] = max(0, _inter)
        solution.MSpillage[t,n] = -min(0, _inter)
        
    if solution.profiling:
        solution.calls_spilldeft +=1 
        solution.time_spilldeft += cclock() - start
        

@njit
def UpdateSpillDef(solution):
    if solution.profiling:
        start = cclock()

    for t in range(solution.intervals):
        for n in range(solution.nodes):
            _inter = solution.MUnbalanced[t, n] + solution.MCharge[t, n] - solution.MDischarge[t, n]
            solution.MDeficit[t, n] = max(_inter, 0)
            solution.MSpillage[t, n] = -min(_inter, 0)

    if solution.profiling:
        solution.calls_spilldef +=1 
        solution.time_spilldef += cclock() - start
        
