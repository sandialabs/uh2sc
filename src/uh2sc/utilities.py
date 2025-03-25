# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:14 2024

@author: dlvilla
"""

import os
from warnings import warn

import numpy as np

from CoolProp import CoolProp as CP

from uh2sc.errors import (FluidStrBracketError, 
                          FluidStrNumberError, 
                          FluidStrNumbersDoNotAddToOneError)

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

def _find_all_char_in_str(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def _is_valid_matstr(matstr):
    """
    Assure that matstr follows the exact form required for CoolProps to 
    read it.
    
    """
    _valid = True
    bad_float_exception = None
    bad_fluid_str = None
    bad_brackets = None
    bad_numbers = None
    
    COOL_PROP_WEB = r"http://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids"
    
    def _test_if_cool_prop_fluid(fluidstr,valid):
        
        bad_fluidstr = None
        try:
            _fluid = CP.AbstractState("HEOS",fluidstr)
        except ValueError as excep:
            valid = False
            bad_fluidstr = excep
        return bad_fluidstr, valid
        
    gas_molefracs = []
    if "&" in matstr:
    
        for gas in matstr.split("&"):
            num_opening_brackets = len(_find_all_char_in_str(gas,"["))
            num_closing_brackets = len(_find_all_char_in_str(gas,"]"))
            # only one pair of brackets 
            if (num_opening_brackets != 1 
             or num_closing_brackets != 1):
                _valid = False
                bad_brackets = FluidStrBracketError(f"The gas `{gas}` must have a "
                                         +"single pair of brackets "
                                         +"surrounding a single number")
                
            # brackets enclose a float
            try:
                gas_molefracs.append(float(gas.split("[")[-1].split("]")[0]))
            except ValueError:
                _valid = False
                bad_float_exception = FluidStrNumberError(f"The gas `{gas}` must "
                                                 +"have a number between "
                                                 +"the brackets!")
                
                
        # gas string before brackets is 
        fluidstr = gas.split("[")[0]
        bad_fluidstr, _valid = _test_if_cool_prop_fluid(fluidstr, _valid)
        
        if len(gas_molefracs) == len(matstr.split("&")):
            must_be_one = np.array(gas_molefracs).sum()
            
            if must_be_one < 0.9999 or must_be_one > 1.0001:
                _valid = False
                bad_numbers = FluidStrNumbersDoNotAddToOneError(
                    f"The molefractions for gas mixture `{matstr}` must sum to"
                    +" one to 4th decimal precision!")
                
    else:
        bad_fluidstr,_valid = _test_if_cool_prop_fluid(matstr, _valid)

    if not _valid:
        exceptions = []
        if bad_fluidstr is not None:
            exceptions.append(bad_fluidstr)
        if bad_float_exception is not None:
            exceptions.append(bad_float_exception)
        if bad_brackets is not None:
            exceptions.append(bad_brackets)
        if bad_numbers is not None:
            exceptions.append(bad_numbers)
        top_exception = ValueError(f"The fluid mixture string `{matstr}` is invalid."
                         +" The string must either be a single fluid "
                         +"string that is a valid input string for cool "
                         +f"props ({COOL_PROP_WEB}) or it must be a list "
                         +"of valid cool prop fluid strings delimited by"
                         +" ampersands (&) and after each valid cool prop"
                         +" fluid a set of brackets `[]` that contains a"
                         +" number. All the numbers must add exactly to "
                         +"one!") 
        exceptions.append(top_exception)
        
        raise ValueError(*exceptions)

def process_CP_gas_string(matstr):
    """
    Detects if a multi component fluid is specified using & for separation of components
    
    To know how to construct a valid fluid mixture string, go to http://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids
    
    The input matstr
    
    
    """

    _is_valid_matstr(matstr)
    
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
        
    fluid = CP.AbstractState("HEOS",comp)
    fluid.set_mole_fractions(molefracs)
    fluid.specify_phase(CP.iphase_gas)

    return comp, molefracs, compSRK, fluid

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
            
            chem_comp, molefracs, compSRK, fluid = process_CP_gas_string(cp_mat_str)
            

            fluid.update(CP.PT_INPUTS, pressure, temperature)
            
            MW = fluid.molar_mass()
            
            
    
    mass_frac = mass_flows/total_mass_flow
    
    
    
    
    return []

