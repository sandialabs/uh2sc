# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:14 2024

@author: dlvilla
"""

import os
from warnings import warn

import numpy as np

from CoolProp import CoolProp as CP

def filter_cpu_count(cpu_count):
    """
    Filters the number of cpu's to use for parallel runs such that
    the maximum number of cpu's is properly constrained.

    Parameters
    ----------
    cpu_count : int or None :
        int : number of proposed cpu's to use in parallel runs
        None : choose the number of cpu's based on os.cpu_count()
    Returns
    -------
    int
        Number of CPU's to use in parallel runs.

    """

    if isinstance(cpu_count,(type(None), int)):

        max_cpu = os.cpu_count()

        if cpu_count is None:
            if max_cpu == 1:
                return 1
            else:
                return max_cpu -1

        elif max_cpu <= cpu_count:
            warn("The requested cpu count is greater than the number of "
                 +"cpu available. The count has been reduced to the maximum "
                 +"number of cpu's ({0:d}) minus 1 (unless max cpu's = 1)".format(max_cpu))
            if max_cpu == 1:
                return 1
            else:
                return max_cpu - 1
        else:
            return cpu_count

def process_CP_gas_string(matstr):
    # Detects if a multi component fluid is specified using & for separation of components
    if "&" in matstr:
        comp_frac_pair = [str.replace("["," ").replace("]","").split(" ") for str in  matstr.split("&")]
        comp0 = [pair[0] for pair in comp_frac_pair]
        compSRK0 = [pair[0]+"-SRK" for pair in comp_frac_pair]
        molefracs0 = np.asarray([float(pair[1]) for pair in comp_frac_pair])
        molefracs = molefracs0 / sum(molefracs0)

        sep = "&"
        comp = sep.join(comp0)
        compSRK = sep.join(compSRK0)
    # Normally single component fluid is specified
    else:
        comp = matstr
        molefracs = [1.0]
        compSRK = matstr

    return comp, molefracs, compSRK

def cavern_initial_mass_flows(inputs,time):
        # calculate the molefractions
    mass_flows = np.array([[inputs["wells"][wname]["valves"][vname]["mdot"][0]
                       for vname, valve in well["valves"].items()]
                     for wname, well in inputs["wells"].items()])
    total_mass_flow = mass_flows.sum()
    
    
    fluids = {}
    
    for wname, well in inputs["wells"].items():
        for vname, valve in well["valves"].items():
            cp_mat_str = inputs["wells"][wname]["valves"][vname]["reservoir"]["fluid"]
            pressure = inputs["wells"][wname]["valves"][vname]["reservoir"]["pressure"]
            temperature = inputs["wells"][wname]["valves"][vname]["reservoir"]["temperature"]
            
            chem_comp, molefracs, compSRK = process_CP_gas_string(cp_mat_str)
            
            fluid = CP.AbstractState("HEOS", chem_comp)
            fluid.specify_phase(CP.iphase_gas)
            fluid.set_mole_fractions(molefracs)
            fluid.update(CP.PT_INPUTS, pressure, temperature)
            
            MW = fluid.molar_mass()
            
            
    
    mass_frac = mass_flows/total_mass_flow
    
    
    
    
    return []

