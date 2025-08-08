# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:14 2024

@author: dlvilla
"""

import os
from warnings import warn

import numpy as np
from scipy.optimize import fsolve, root_scalar


from CoolProp import CoolProp as CP

from uh2sc.errors import (FluidStrBracketError,
                          FluidStrNumberError,
                          FluidStrNumbersDoNotAddToOneError,
                          FluidMixtureStateInfeasibleInCoolProp,
                          FluidMixturePresTempNotValid,
                          FluidDoesNotExistInCoolProp,
                          DeveloperError)
from uh2sc.constants import Constants
from uh2sc.thermodynamics import (solubility_of_nacl_in_h2o,
                                  density_of_brine_water)
from uh2sc.fluid import FluidWithFitOption
import logging

const = Constants()

def _create_fluid_from_tup(tup):
    fluid = CP.AbstractState(tup[0],tup[1])
    fluid.set_mass_fractions(tup[2])
    return fluid


def _update_fluid(fluid_tup,pres,temp):
    try:
        fluid = _create_fluid_from_tup(fluid_tup)
        fluid.update(CP.PT_INPUTS,pres,temp)
    except ValueError as exc:
        if "not good" in str(exc):
            raise FluidMixtureStateInfeasibleInCoolProp("The fluid mixture"
            +f" {fluid.fluid_names()} is infeasible at mass "
            +f"fractions equal to {fluid.get_mass_fractions()}' please adjust"
            +" your mass fractions to ratios that work!. The mixture model"
            +" became unstable:") from exc
        elif "p is not a valid number" in str(exc):
            raise FluidMixturePresTempNotValid("The fluid mixture"
            +f" {fluid.fluid_names()} temperature {temp} and pressure {pres} "
            +"are not valid inputs at mass for mass"
            +f"fractions equal to {fluid.get_mass_fractions()}' please adjust"
            +" your mass fractions to ratios that work!. The mixture model"
            +" listed pressure as an invalid value!\n\n") from exc

        else:
            raise exc
            
    return fluid


def integrate_piecewise_linear(t0, t1, t_samples, y_samples, epsilon=1e-6):
    t_samples = np.asarray(t_samples, dtype=float)
    y_samples = np.asarray(y_samples, dtype=float)

    if t_samples.ndim != 1 or y_samples.ndim != 1:
        raise ValueError("t_samples and y_samples must be 1-D arrays.")
    if len(t_samples) != len(y_samples):
        raise ValueError("t_samples and y_samples must have the same length.")
    if len(t_samples) < 2:
        raise ValueError("Need at least two sample points.")
    if not np.all(np.diff(t_samples) > 0):
        raise ValueError("t_samples must be strictly increasing.")

    t_min, t_max = t_samples[0], t_samples[-1]

    # Handle numeric noise tolerance
    if t0 < t_min - epsilon or t1 > t_max + epsilon:
        raise ValueError(f"Integration bounds [{t0}, {t1}] are outside sample range [{t_min}, {t_max}] beyond tolerance.")
    if t0 < t_min:
        t0 = t_min
    if t1 > t_max:
        t1 = t_max

    if t0 > t1:
        raise ValueError("t0 must be <= t1.")

    # Slice the relevant segment for integration
    mask = (t_samples >= t0) & (t_samples <= t1)
    t_seg = t_samples[mask]
    y_seg = y_samples[mask]

    # Ensure t0 and t1 are included exactly
    if t0 not in t_seg:
        y0 = np.interp(t0, t_samples, y_samples)
        t_seg = np.insert(t_seg, 0, t0)
        y_seg = np.insert(y_seg, 0, y0)
    if t1 not in t_seg:
        y1 = np.interp(t1, t_samples, y_samples)
        t_seg = np.append(t_seg, t1)
        y_seg = np.append(y_seg, y1)

    return np.trapezoid(y_seg, t_seg)


# convenience wrapper for your use:
def average_mdot_for_step(model, pipe):
    t0 = model.time
    t1 = model.time + model.time_step
    mass_total =  integrate_piecewise_linear(t0, t1, pipe.valve['time'], pipe.valve['mdot'])
    if t1 == t0:
        return 0.0
    else:    
        return mass_total/(t1-t0)


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
    bad_brackets = None
    bad_numbers = None

    COOL_PROP_WEB = r"http://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids"

    def _test_if_cool_prop_fluid(fluidstr,valid):

        bad_fluidstr = None
        try:
            CP.AbstractState("HEOS",fluidstr)
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

def create_CP_gas_string(fluid):
    names = fluid.fluid_names()
    mol_fractions = fluid.get_mole_fractions()

    return "HEOS::" + "&".join([name +f"[{molefrac}]" for
                     molefrac, name in zip(mol_fractions, names)])

# originally written by AI, adjusted as needed with help from ChatGPT later.
def verify_mixture_bips(fluid_tup):
    """
    Verifies the presence of binary interaction parameters (BIPs)
    for all unique pairs in a CoolProp mixture model.

    Args:
        fluid_tup: 3-tuple: [0] fluid backend, [1] fuid components [2] fluid mass fractions
             Everything needed to create an AbstractState instance
                                        representing the mixture.

    Returns:
        tuple: A tuple containing:
            - bool: True if all unique binary pairs have non-zero BIPs, False otherwise.
            - dict: A dictionary where keys are fluid pairs (as tuples)
                    and values are the retrieved BIP (k_ij) values.
    """

    bip_status = {}
    one_bip_present = False

    # Thank you ChatGPT, the other AI did not know this!
    var_to_try = ["kij","vij","gammaT","betaT"]

    try:
        fluid = _create_fluid_from_tup(fluid_tup)
        fluid_names = fluid.fluid_names()
        num_fluids = len(fluid_names)

        if num_fluids < 2:
            return True, bip_status

        # Iterate through all unique pairs of fluids
        for i in range(num_fluids):
            for j in range(i + 1, num_fluids):  # Start from i + 1 to avoid checking pairs twice and self-interaction
                fluid1_name = fluid_names[i]
                fluid2_name = fluid_names[j]

                for var in var_to_try:
                    try:
                        # Attempt to retrieve the k_ij BIP
                        # {Link: get_binary_interaction_double() https://coolprop.org/fluid_properties/Mixtures.html} requires indices and the parameter name
                        var_value = fluid.get_binary_interaction_double(i, j, var)
                        bip_status[(fluid1_name, fluid2_name, var)] = var_value

                        if var_value != 0.0:
                            one_bip_present = True

                    except Exception:
                        bip_status[(fluid1_name, fluid2_name)] = "Error"

    except Exception as e:
        logging.warning(f"An error occurred while processing the fluid mixture: {e}")
        return False, {}

    return one_bip_present, bip_status


def pickup_relevant_ml_models(model,cpfluid):
    mldir = os.path.join(os.path.dirname(__file__),"..","ml_fluids")
    file_ending = ".joblib"
    ml_models = [file for file in os.listdir(mldir) if file.endswith(file_ending)]
    
    for ml_model in ml_models:
        # choose the best model if more than one exists?
        pass
    
    return []
    

def process_CP_gas_string(matstr,backend,model):
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

    cpfluid_tup = (backend,comp,massfracs) 
    cpfluid = _create_fluid_from_tup(cpfluid_tup)

    # make sure that the fluid setup is covered by CoolProps!
    one_bip_present, bip_status = verify_mixture_bips(cpfluid_tup)

    if not one_bip_present:
        raise FluidDoesNotExistInCoolProp(f"The proposed mixture of fluids {comp} does not "
        +"have Binary Interaction Parameters (BIPs) in CoolProps UH2SC"
        +" cannot simulate this!")
    if backend != "REFPROP" and "Hydrogen" in cpfluid.fluid_names() and len(cpfluid.fluid_names()) > 1:
        model.logging.warning(f"Backend {backend} is unlikely to accurately"+
                              " simulate hydrogen mixtures! Please consider"
                              +" switching to REFPROP backend!")
        
    # now see if you can pick up a machine learning training set that covers
    # this analysis
    ml_models = pickup_relevant_ml_models(model,cpfluid)
    if len(ml_models)==0:
        ml_model = None
    elif len(ml_models)>1:
        raise DeveloperError(f"{len(ml_models)} machine learning models for"
                             +f" {comp} were found! Only one is allowed"
                             +" currently! Please remove one from src/ml_fluids!")
    else:
        ml_model = ml_models[0]
    
    # This now intelligently blends ML and CoolProp such that ML is used if
    # the requested temperature and pressure are within the training interval.
    # you can use
    fluid = FluidWithFitOption(cpfluid_tup, 
                               ml_model, 
                               model.logging, 
                               model.inputs["calculation"]["machine_learning_acceptable_percent_error"] 
                               )
    

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

    PT = [inputs['initial']['pressure'],inputs['initial']['temperature']]
    
    model.fluids['cavern'].set_state(CP.AbstractState,PT)

    mdot_cavern = np.zeros(len(model.fluid_components))
    mdot_valves = {}

    for wname, well_fluids in model.fluids.items():
        if wname != 'cavern':
            mdot_valves[wname] = {}
            for vname, valve_fluid in well_fluids.items():
                
                
                valve_fluid.set_state(CP.AbstractState,PT=PT)
                
                
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
                
                valve_fluid.del_state()
    model.fluids['cavern'].del_state()

    return mdot_valves, mdot_cavern


def find_all_fluids(model):
    """
    Finds all of the fluids named in the input and sets up every fluid to
    include all terms even if they are zero for a reservoir.


    """
    fluid_components = []
    fluid_mapping = {}
    backend = model.inputs["calculation"]["cool_prop_backend"]

    # first get all fluid names from reservoirs and store them in "fluid_components" list
    for wname, well in model.inputs["wells"].items():
        fluid_mapping[wname] = {}
        for vname, valve in well["valves"].items():
            if valve["type"] != "mdot":
                raise NotImplementedError("Only mdot valves have been implemented!")

            if "reservoir" in valve:

                vfluids = valve["reservoir"]["fluid"]
                comps, molefracs, compSRK, fluid = process_CP_gas_string(vfluids, backend, model)
                #Translate to exact CoolProp name for the fluid in the DB
                for pure_fluid in fluid.fluid_names():
                    if pure_fluid not in fluid_components:
                        fluid_components.append(pure_fluid)
                fluid_mapping[wname][vname] = fluid

    # add the initial fluid in the cavern to fluid_components
    icomps, imolefracs, icompSRK, ifluid = process_CP_gas_string(
        model.inputs["initial"]["fluid"], backend, model)
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

                fluid_str = _construct_ordered_fluid_str(fluid_components,
                                                         fluid_mapping,
                                                         (wname,vname))
                comps, massfracs, compSRK, fluid = process_CP_gas_string(
                    fluid_str, backend, model)
                mdots[wname][vname] = np.stack((np.array(valve["mdot"]),np.array(valve["time"])))
                fluids[wname][vname] = fluid

    # create a fluid for the cavern as well with all components present.
    cavern_fluid_str = _construct_ordered_fluid_str(fluid_components,
                                                    fluid_mapping)
    icomps, imolefracs, icompSRK, ifluid = process_CP_gas_string(
        cavern_fluid_str, backend, model)
    fluids["cavern"] = ifluid


    model.fluids = fluids
    model.fluid_components = fluid_components
    model.mdots = mdots


    return fluid_components, fluids, mdots


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
        
        cp_fluid.set_state(CP.AbstractState)
        cp_fluid_names = cp_fluid.fluid_names()
        cp_mass_fracs = cp_fluid.get_mass_fractions()
        cp_fluid.del_state()
        
        for pure_fluid in fluid_components:
            if pure_fluid in cp_fluid_names:
                cp_ind = cp_fluid_names.index(pure_fluid)
                massfrac = cp_mass_fracs[cp_ind]
            else:
                massfrac = 0.0
            if len(fluid_str) > 0:
                prestr = "&"

            fluid_str += f"{prestr}{pure_fluid}[{massfrac:.8e}]"
        return fluid_str



def brine_average_pressure(fluid,water,height_total,height_brine,rho_g=None,pres_g=None):
    """
    Approximate the average pressure for the brine

    """
    t_brine = water.T()
    # TODO: create a test for this!
    if rho_g is None:
        rho_g = fluid.rhomass()
    if pres_g is None:
        pres_g = fluid.p()

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




def evaporation_energy(water,
                        t_cavern,
                        t_brine,
                        vol_cavern):
    # find vapor mass change due to condensation (and settling to the brine)
    # or evaporation
    restore_p_brine = water.p()
    restore_t_brine = water.T()

    # get density of saturated vapor at the temperature and saturated
    # vapor pressure
    water.update(CP.QT_INPUTS,1.0,t_cavern)
    rho_vapor = water.rhomass()
    p_vapor = water.p()

    # get the heat of vaporization at the average temperature during the time
    # step
    water.update(CP.QT_INPUTS,0.0,t_cavern)
    h_vapor_0 = water.hmass()
    water.update(CP.QT_INPUTS,1.0,t_cavern)
    h_vapor_1 = water.hmass()
    h_evaporate = h_vapor_1 - h_vapor_0

    # restore the water CoolProp fluids to their original state.
    water.update(CP.PT_INPUTS,restore_p_brine, restore_t_brine)

    mass_vapor = rho_vapor * vol_cavern


    return (mass_vapor, rho_vapor, h_vapor_1, p_vapor, h_evaporate)



def _find_pressure_from_rho_T(fluid0, rho_target, T):

    def objective(P):
        try:
            fluid0.update(CP.PT_INPUTS, P, T)
            return fluid0.rhomass() - rho_target
        except:
            return 1e6  # big error if flash fails

    res = root_scalar(objective, bracket=[1e4, 5e8], method='brentq')
    if not res.converged:
        raise ValueError("Could not find pressure for given density and T")

    P_solution = res.root
    fluid0.update(CP.PT_INPUTS, P_solution, T)


def conservation_of_volume(vol_cavern, volume_total, area, water, t_cavern, 
                           t_brine, m_cavern, m_brine, fluid, 
                           return_vapor_variables=False):

    volume_liquid_brine = volume_total - vol_cavern
    height_brine = volume_liquid_brine / area
    height_total = volume_total / area

    (mass_vapor, rho_vapor, h_vapor_1, p_vapor, h_evaporate) = (
        evaporation_energy(water,
                           t_cavern,
                           t_brine,
                           vol_cavern))


    rho_gas_no_vapor = m_cavern.sum() / vol_cavern
    rho_brine = m_brine / volume_liquid_brine

    if len(fluid.fluid_names()) == 1:
        pressure_gas = CP.PropsSI('P','D',rho_gas_no_vapor,
                                      'T', t_cavern,create_CP_gas_string(fluid))
    else:
        _find_pressure_from_rho_T(fluid, rho_gas_no_vapor, t_cavern)
        pressure_gas = fluid.p()

    (pressure_brine,
     solubility_brine,
     rho_brine_with_salt) = brine_average_pressure(fluid,water,
                                                   height_total,
                                                   height_brine,
                                                   t_brine)

    fluid.update(CP.PT_INPUTS, pressure_gas, t_cavern)
    if isinstance(pressure_brine, np.ndarray):
        pressure_brine = pressure_brine[0]
    water.update(CP.PT_INPUTS, pressure_brine, t_brine)

    if return_vapor_variables:
        return mass_vapor, rho_vapor, h_vapor_1, p_vapor, h_evaporate

    else:
        return water.rhomass() - rho_brine


def calculate_cavern_pressure(fluid,
                              m_cavern,
                              t_cavern,
                              water,
                              m_brine,
                              t_brine,
                              volume_total,
                              area,
                              volume_cavern_estimate):

    cavern_gas_volume = fsolve(conservation_of_volume, volume_cavern_estimate, 
                           args=(volume_total, area, water, t_cavern, 
                               t_brine, m_cavern, m_brine, fluid))

    pressure_gas_novapor = fluid.p()


    return pressure_gas_novapor, cavern_gas_volume

def calculate_component_masses(fluid,mass):
    return np.array(fluid.get_mass_fractions()) * mass

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

def init_water(water_tup):
    water = CP.AbstractState(water_tup[0],water_tup[1])
    water.set_mass_fractions(water_tup[2])
    water.update(CP.PT_INPUTS,water_tup[3][0],water_tup[3][1])
    return water
