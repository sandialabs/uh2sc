# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:14 2024

@author: dlvilla
"""

import os
from warnings import warn
import re

import numpy as np

from CoolProp import CoolProp as CP

from uh2sc.errors import (FluidStrBracketError, 
                          FluidStrNumberError, 
                          FluidStrNumbersDoNotAddToOneError)
from uh2sc.constants import Constants
from uh2sc.thermodynamics import (solubility_of_nacl_in_h2o, 
                                  brine_saturated_pressure, 
                                  density_of_brine_water)

const = Constants()


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
        massfracs0 = np.asarray([float(pair[1]) for pair in comp_frac_pair])
        massfracs = massfracs0 / sum(massfracs0)

        sep = "&"
        comp = sep.join(comp0)
        compSRK = sep.join(compSRK0)
    # Normally single component fluid is specified
    else:
        comp = matstr
        massfracs = [1.0]
        compSRK = matstr
        
    fluid = CP.AbstractState("HEOS",comp)
    fluid.set_mass_fractions(massfracs)

    return comp, massfracs, compSRK, fluid

def reservoir_mass_flows(model,time):
    """
    Interpolates the components of mass flow coming from all valves and 
    captures this into a single mass in/mass out for the cavern if mass were
    being immediately removed (i.e. no well pressure losses effects assumed).
    If pressure losses matter, it also returns 
    all well valve flow conditions in terms of mass flow per fluid for the 
    global fluid_components vector so that the well equations can be solved
    and travel time and pressure differences can be accounted for in the
    well model before mass reaches the cavern.
    
    # NOTE: this function assumes the underlying model fluids have already been updated
    # with the current pressure and temperature!
    """
    
    inputs = model.inputs
    
    mdot_cavern = np.zeros(len(model.fluid_components))
    mdot_valves = {}
    
    for wname, well_fluids in model.fluids.items():
        if wname != 'cavern':
            mdot_valves[wname] = {}
            for vname, valve_fluid in well_fluids.items():
                mdot_valves[wname][vname] = np.zeros(len(model.fluid_components))
                
                # get stored information
                mdot_arr = model.mdots[wname][vname]
                mass_fracs = valve_fluid.get_mass_fractions()
                comp_names = valve_fluid.fluid_names()
                
                main_ind = [model.fluid_components.index(comp) for comp in comp_names]
                # see find_all_fluids, time is the second row, mdot is the first
                # row.
                mdot = np.interp(time, mdot_arr[1,:], mdot_arr[0,:])
                
                if mdot < 0.0:
                    # flow is coming out of the cavern and we must change the
                    # mass fractions to those of what is in the cavern
                    # this only applies when a well assumes ideal pipes that
                    # produce no losses.
                    mass_fracs = model.fluids['cavern'].get_mass_fractions()
                    comp_names = model.fluids['cavern'].fluid_names()
  
                mdot_comp = mdot * np.array(mass_fracs)
                mdot_valves[wname][vname][main_ind] += mdot_comp
                mdot_cavern[main_ind] += mdot_comp

    return mdot_valves, mdot_cavern
            
            
    
    
    
    
    return []

def find_all_fluids(model):
    """
    Finds all of the fluids named in the input and sets up every fluid to
    include all terms even if they are zero for a reservoir.
    
    
    """
    fluid_components = []
    fluid_mapping = {}
    
    # first get all fluid names from reservoirs and store them in "fluid_components" list
    for wname, well in model.inputs["wells"].items():
        fluid_mapping[wname] = {}
        for vname, valve in well["valves"].items():
            if valve["type"] != "mdot":
                raise NotImplementedError("Only mdot valves have been implemented!")

            if "reservoir" in valve:
                
                reservoir = valve["reservoir"]

                vfluids = valve["reservoir"]["fluid"]
                comps, molefracs, compSRK, fluid = process_CP_gas_string(vfluids)
                #Translate to exact CoolProp name for the fluid in the DB
                for pure_fluid in fluid.fluid_names():
                    if pure_fluid not in fluid_components:
                        fluid_components.append(pure_fluid)
                fluid_mapping[wname][vname] = fluid
    
    # add the initial fluid in the cavern to fluid_components
    icomps, imolefracs, icompSRK, ifluid = process_CP_gas_string(model.inputs["initial"]["fluid"])
    for ipure_fluid in ifluid.fluid_names():
        if ipure_fluid not in fluid_components:
            fluid_components.append(ipure_fluid)
    fluid_mapping['cavern'] = ifluid
    
    
    # now characterize fluids that contain all pure_fluids in the entire analysis
    # for each well's valves.
    fluids = {}
    mdots = {}
    for wname, well in model.inputs["wells"].items():
        fluids[wname] = {}
        mdots[wname] = {}
        for vname, valve in well["valves"].items():
            if valve["type"] != "mdot":
                raise NotImplementedError("Only mdot valves have "
                                          +"been implemented!")

            if "reservoir" in valve:
                reservoir = valve["reservoir"]
                
                fluid_str = _construct_ordered_fluid_str(fluid_components,
                                                         fluid_mapping,
                                                         (wname,vname))
                comps, massfracs, compSRK, fluid = process_CP_gas_string(fluid_str)
                mdots[wname][vname] = np.stack((np.array(valve["mdot"]),np.array(valve["time"])))
                fluids[wname][vname] = fluid
    
    # create a fluid for the cavern as well with all components present.
    cavern_fluid_str = _construct_ordered_fluid_str(fluid_components,
                                                    fluid_mapping)    
    icomps, imolefracs, icompSRK, ifluid = process_CP_gas_string(cavern_fluid_str)
    fluids["cavern"] = ifluid
    
    pres = model.inputs['initial']['pressure']
    temp = model.inputs['initial']['temperature']
    # set initial fluid properties
    for wname, fluiddict in fluids.items():
        if wname != 'cavern':
            for vname, fluid in fluiddict.items():
                fluid.update(CP.PT_INPUTS,pres,temp)
    fluids['cavern'].update(CP.PT_INPUTS,pres,temp)
    
    model.fluids = fluids
    model.fluid_components = fluid_components
    model.mdots = mdots
    
    molar_masses = {}
    for fcomp in fluid_components:
        tempfluid = CP.AbstractState("HEOS",fcomp)
        molar_masses[fcomp] = tempfluid.molar_mass()
    
    # in kg/mol
    model.molar_masses = molar_masses
        
            
    return fluid_components, fluids, mdots, molar_masses


def _construct_ordered_fluid_str(fluid_components,fluid_mapping,names=None):
    if len(fluid_components) == 1:
        # a single fluid, no mole fractions needed.
        return fluid_components[0]
    else:
        if isinstance(names, tuple):
            wname,vname = names
            cp_fluid = fluid_mapping[wname][vname]
        else:
            cp_fluid = fluid_mapping["cavern"]
            
        fluid_str = ""
        prestr = ""
        
        cp_fluid_names = cp_fluid.fluid_names()
        cp_mole_fracs = cp_fluid.get_mole_fractions()
        for pure_fluid in fluid_components:
            if pure_fluid in cp_fluid_names:
                cp_ind = cp_fluid_names.index(pure_fluid)
                molfrac = cp_mole_fracs[cp_ind]
            else:
                molfrac = 0.0
            if len(fluid_str) > 0:
                prestr = "&"
                
            fluid_str += f"{prestr}{pure_fluid}[{molfrac:.8e}]"
        return fluid_str
    
    
def calculate_pressure(fluid,rho_cavern,t_cavern,tol=1e-6, max_iter=100):
        # P = n * rho * R * T
          
        ideal_pressure = fluid.gas_constant()/fluid.molar_mass() * rho_cavern * t_cavern
        
        pressure = ideal_pressure
        pressure0 = 0
        _iter = 0
        
        while (np.abs((pressure - pressure0)/pressure) > tol) and _iter < max_iter:
            pressure0 = pressure
            fluid.update(CP.PT_INPUTS,pressure0,t_cavern)
        
            Z = fluid.compressibility_factor()
        
            pressure = Z * fluid.gas_constant()/fluid.molar_mass() * rho_cavern * t_cavern
            _iter += 1
            
        if _iter == max_iter:
            raise ValueError("No convergence on pressure calculation!")
    
        return pressure
    
def calculate_component_masses(fluid,mass):
    return np.array(fluid.get_mass_fractions()) * mass
    
    # if fluid.phase() == CP.iphase_gas or fluid.phase() == CP.iphase_supercritical:
    #     if liquid_mass != 0.0:
    #         raise ValueError("You cannot have a liquid_mass when the fluid"
    #                          +" in the cavern is only gas!")
            
    #     return {'gas':gas_mass * np.array(fluid.get_mass_fractions()),
    #             'liquid':np.zeros(len(fluid.fluid_names()))}
        
    # elif fluid.phase() == CP.iphase_liquid:
    #     if gas_mass != 0:
    #         breakpoint()
    #         raise ValueError("You canot have a gas_mass when the fluid"
    #                          +" in the cavern is only liquid!")
    #     return {'gas':np.zeros(len(fluid.fluid_names())),
    #             'liquid':liquid_mass * np.array(fluid.get_mass_fractions())}            
        
    # elif fluid.phase() == CP.iphase_twophase:
    
        
    #     gmolfracs = {fluid_name:gmol for fluid_name,gmol in zip(fluid.fluid_names(),fluid.mole_fractions_vapor())}
    #     lmolfracs = {fluid_name:lmol for fluid_name,lmol in zip(fluid.fluid_names(),fluid.mole_fractions_liquid())}

    #     gsum = np.array([gmol * molar_masses[name] for name, gmol in gmolfracs.items()]).sum()
    #     lsum = np.array([lmol * molar_masses[name] for name, lmol in lmolfracs.items()]).sum()
        
    #     gmass = gas_mass * np.array([gmol * molar_masses[name]/gsum for name, gmol in gmolfracs.items()])
    #     lmass = liquid_mass * np.array([lmol * molar_masses[name]/lsum for name, lmol in lmolfracs.items()])

    #     return {'gas':gmass, 'liquid':lmass}
        
    # else:
    #     breakpoint()
    #     raise NotImplementedError("Only gas, liquid, or gas and liquid mixtures are modeled!")
        
def brine_average_pressure(fluid,water,height_total,height_brine,t_brine):
    """
    Approximate the average pressure for the brine
    
    """
    # TODO: create a test for this!
    pres_g = fluid.p()
    rho_g = fluid.rhomass()
    # assume density is constant
    height_gas_o_2 = (height_total - height_brine)/2
    pres_g_surf = pres_g + rho_g * const.g['value'] * height_gas_o_2
    rho_pure_water = water.rhomass()
    height_brine_o_2 = height_brine / 2
    solubility_brine = solubility_of_nacl_in_h2o(t_brine)
    rho_brine = density_of_brine_water(t_brine, pres_g_surf, solubility_brine, rho_pure_water)
            
    return (pres_g_surf + height_brine_o_2 * const.g['value'] * rho_brine,
           solubility_brine,
           rho_brine)
    