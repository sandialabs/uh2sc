# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license

import math
from CoolProp.CoolProp import PropsSI
from warnings import warn
import numpy as np


def Gr(L, Tfluid, Tvessel, P, species):
    """
    Calculation of Grasshof number. See eq. 4.7-4 in
    C. J. Geankoplis Transport Processes and Unit Operations, International Edition,
    Prentice-Hall, 1993

    Parameters
    ----------
    L : float
        Vessel length
    Tfluid : float
        Temperature of the bulk fluid inventory
    Tvessel : float 
        Temperature of the vessel wall (bulk) 
    P : float 
        Pressure of fluid inventory

    Returns
    ----------
    Gr : float
        Grasshof number
    """
    # Estimating the temperature at the fluid film interface
    T = (Tfluid + Tvessel) / 2
    beta = PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "T|gas", T, "P", P, species)
    nu = PropsSI("V", "T|gas", T, "P", P, 'HEOS::'+species.split('::')[1]) / PropsSI("D", "T|gas", T, "P", P, species)
    Gr = 9.81 * beta * abs(Tvessel - Tfluid) * L ** 3 / nu ** 2
    return Gr


def Pr(T, P, species):
    """
    Calculation of Prandtl number, eq. 4.5-6 in
    C. J. Geankoplis Transport Processes and Unit Operations, International Edition,
    Prentice-Hall, 1993

    Parameters
    ----------
    T : float
        Temperature of the fluid film interface 
    P : float 
        Pressure of fluid inventory

    Returns
    ----------
    Pr : float
        Prantdl number
    """
    C = PropsSI("C", "T|gas", T, "P", P, species)
    V = PropsSI("V", "T|gas", T, "P", P, 'HEOS::'+species.split('::')[1])
    L = PropsSI("L", "T|gas", T, "P", P, 'HEOS::'+species.split('::')[1])
    Pr = C * V / L

    return Pr


def Nu(Ra, Pr):
    """
    Calculation of Nusselt number for natural convection. See eq. 4.7-4  and Table 4.7-1 in
    C. J. Geankoplis Transport Processes and Unit Operations, International Edition,
    Prentice-Hall, 1993

    Parameters
    ----------
    Ra : float
        Raleigh number
    Pr : float 
        Prandtl number

    Returns
    ----------
    Nu : float
        Nusselt numebr
    """
    if Ra >= 1e9:
        NNu = 0.13 * Ra ** 0.333
    elif Ra < 1e9 and Ra > 1e4:
        NNu = 0.59 * Ra ** 0.25
    else:
        NNu = 1.36 * Ra ** 0.20
    return NNu

def h_inside(L, Tvessel, Tfluid, fluid):
    """
    Calculation of internal natural convective heat transfer coefficient from Nusselt number
    and using the coolprop low level interface.

    Parameters
    ----------
    L : float
        Vessel length
    Tfluid : float
        Temperature of the bulk fluid inventory
    Tvessel : float 
        Temperature of the vessel wall (bulk) 
    fluid : obj
        Coolprop fluid object 
    
    Returns
    ----------
    h_inner : float
        Heat transfer coefficient
    """
    cond = fluid.conductivity()
    visc = fluid.viscosity()
    cp = fluid.cpmass()
    Pr = cp * visc / cond
    beta = fluid.isobaric_expansion_coefficient()
    nu = visc / fluid.rhomass()
    Gr = 9.81 * beta * abs(Tvessel - Tfluid) * L ** 3 / nu ** 2
    Ra = Pr * Gr
    NNu = Nu(Ra, Pr)
    h_inner = NNu * cond / L
    return h_inner

def Pr_fluid(fluid):
    cond = fluid.conductivity()
    visc = fluid.viscosity()
    cp = fluid.cpmass()
    Pr = cp * visc / cond
    
    return Pr, visc, cond


def circular_nusselt(mdot, _L_, _D_,fluid):
    """
    Nusselt number calculation for a circular pipe
    
    From Equation 8-16 page 469 of:
        
    Thomas, Lindon C. 2000. "Heat Transfer" 2nd Ed. 
       
    Inputs
    ------
    
    mdot : float : 
        mass flow rate in kg/s 
        
    _L_ : float :
        Length of the annular total length of pipe (not the control volume pipe length!)
        
    _D_ : float :
        Diameter of the pipe in meters
    
    fluid : CoolProp.CoolProp.AbstractState :
        A CoolProp fluid object that is set to Tb and Pb so that 
        thermodynamnic properties can be calculated
    
    Returns
    -------
    
    Nu : float :
        Nusselt dimensionless number
        
    Raises
    ------
    
    ValueError : 
        If any of the ranges of validity for the relationship are violated
        this function raises a ValueError Exception.
    
    """
    
    Pr, visc, cond = Pr_fluid(fluid) 
    
    if Pr < 0.5 or Pr > 200:
        raise ValueError("Prandlt number {0:6.4e} is outside the valid range".format(Pr) + 
                         "The valid range for Prandtl's number for the "+
                         "Gnielinski relationship is 5 to 200.0.")
    
    # using mdot form of Reynold's number
    Re = 4.0 * mdot /(visc * np.pi * _D_)
    
    if Re <= 4000: # laminar
        #_f_ = 16/Re
        Nu = 4.36 # Table 8-1 of Lindon, 2000. uniform wall heat flux
        
    else:
        # turbluent
        if _L_ / _D_ < 1.36 * Re ** 0.25:
            raise ValueError("Two short of a pipe! The length of pipe is small"+
                             " enough to make entrance effects important")
            
        # if Re < 4000.0:
        #     raise ValueError("Reynolds number {0:6.4e} is outside the valid range".format(Re) + 
        #                      "The valid range for Reynold's number for the "+
        #                      "circular pipe relationship is anything greater than 4000.0")
        
        _f_ = (1.58 * np.log(Re) - 3.28) ** (-2.0)
        
        _f_o_2 = _f_ / 2.0
        
        Nu = (_f_o_2) * Re * Pr / (1.07 + 12.7 * np.sqrt(_f_o_2) * (Pr ** (2/3) - 1))
        
    
    _h_ = Nu * cond / _L_
    
    
    return Nu, _h_


def annular_nusselt(mdot, _L_, Di,Do,Tb,Tw,is_inner,fluid,is_test=False):
    """
    Nusselt number calculation for an annular pipe
    
    Returns either the inner or outer surface Nusselt number for equation 5 
    of:
        
    Gnielinski, Volker. 2015. "Turbulent Heat Transfer in Annular Spaces--
       A New Comprehensive Correlation" Heat Transfer Engineering 36:787-789
       DOI https://doi.org/10.1080/01457632.2015.962953
       
    It is assumed that the superposition of the inner with outer surface insulated and 
    outer with inner surface insulated relationships are a good approximation
    for the full solution where neither surface is insulated.
    
    Laminar flow comes from Table 8-1 of Lindon, 2000 (see above for circular)
    
    Inputs
    ------
    
    mdot : float : 
        mass flow rate in kg/s 
        
    _L_ : float :
        Length of the annular total length of pipe (not the control volume pipe length!)
        
    Di : float :
        Inner diameter of the pipe in meters
        
    Do : float : 
        Outer diameter of the pipe in meters
        
    Tb : float :
        Bulk (average) temperature of the fluid flowing in the pipe in Kelvin
        
    Tw : float :
        Wall temperature of the fluid flowing in the pipe in Kelvin
        
    is_inner : bool :
        True = calculate Nu for the inner surface
        False = calculate Nu for the outer surface
    
    fluid : CoolProp.CoolProp.AbstractState :
        A CoolProp fluid object that is set to Tb and Pb so that 
        thermodynamnic properties can be calculated
        
    is_test : bool : Optional : Default = False
        Adjusts behavior to set F_ann = 1 since this is the test
        data Gnielinski provided for comparison
    
    Returns
    -------
    
    Nu : float :
        Nusselt dimensionless number
        
    Raises
    ------
    
    ValueError : 
        If any of the ranges of validity for the relationship are violated
        this function raises a ValueError Exception.
    
    """
    
    
    _a_ = Di/Do
    

    
    # TO DO many of these calculations only need to be computed once
    # within a pipe object. For now, leave them as is. but 
    # for speed eventually pass a VerticalPipe object into this function
    # with all static parameters calculated.
    
    Pr, visc, cond = Pr_fluid(fluid) 
    
    #area = (np.pi / 4.0) * (Do ** 2.0 - Di ** 2.0)
    #hydraulic factor - D_hydraulic cancels out in the Area term.
    D_factor = (np.pi / 4.0) * (Do + Di)
    
    Re = mdot / (visc * D_factor)
    
    if Re < 4000.0:
        # Laminar flow comes from Table 8-1 of Lindon, 2000 (see above for circular)
        # TODO - NOT SURE WHAT THE PRANDTL limitations are for lamiar Flow
        # none listed in Lindon.
        if _a_ < 0.05:
            warn("The annular diameter ratio Di/Do is less than 0.05 and"+
                 " flow is laminar. Accuracy may not be correct for convective heat transfer")
        
        _a_table = np.array([0.0,0.05,0.1,0.2,0.4,0.6,0.8,1.0])
        
        if is_inner:
            #actual theoretical value at 0.0 is infinity I have given a value of 100.
            Nu_table = np.array([1e2,17.81,11.91,8.499,6.593,5.912,5.58,5.385])
        else:
            Nu_table = np.array([4.364,4.792,4.834,4.833,4.979,5.099,5.24,5.385])
            
        Nu = np.interp(_a_,_a_table,Nu_table)
    
    else:
        # turbulent
        
        if Pr < 0.1 or Pr > 1000.0:
            raise ValueError("Prandlt number {0:6.4e} is outside the valid range".format(Pr) + 
                             "The valid range for Prandtl's number for the "+
                             "Gnielinski relationship is 0.1 to 1000.0.")
        
        # avoid repeat calculation of the same quanity
        D_hydraulic = Do - Di
        
        _a_sq = _a_ ** 2.0
        log_a_ = np.log(_a_)
        T_ratio = Tb / Tw
        
        if T_ratio > 1:
            _n_ = 0.0
        elif T_ratio < 1.0 and T_ratio > 0.5:
            _n_ = 0.45
        else:
            raise ValueError("The T_ratio < 0.5 and Gnielinski's Nusselt number "+
                             "relationship is not known to be valid!")
        
        # if Re < 4000.0:
        #     raise ValueError("Reynolds number {0:6.4e} is outside the valid range".format(Re) + 
        #                      "The valid range for Reynold's number for the "+
        #                      "Gnielinski relationship is anything greater than 4000.0")
        
        if is_test:
            _K_ = 1
        else:
            _K_ = T_ratio ** _n_
        
        Re_star = Re * ((1 + _a_sq) * log_a_ + (1 - _a_sq))/(log_a_*(1.0 - _a_)**2.0)
        
        f_ann = (1.8 * np.log10(Re_star) - 1.5) ** (-2.0)
        
        if is_test:
            F_ann = 1.0
        else:
            if is_inner:
                F_ann = 0.75 * _a_ ** (-0.17)
            else:
                F_ann = (0.9 - 0.15 * _a_ ** 0.6)
        
        f_ann_o_8 = f_ann / 8
        
        Nu = ((f_ann_o_8 * (Re - 1000) * Pr) / (
              1 + 12.7 * np.sqrt(f_ann_o_8) * (Pr ** (2/3) -1)) * (     
                  1 + (D_hydraulic/_L_)**(2/3)) * F_ann * _K_)
        # eqn 1
        # k1 = 1.07 + 900/Re - 0.63/(1 + 10*Pr)
        # Nu = ((f_ann_o_8 * Re * Pr) / (
        #       k1 + 12.7 * np.sqrt(f_ann_o_8) * (Pr ** (2/3) -1)) * (     
        #           1 + (D_hydraulic/_L_)**(2/3)) * F_ann * _K_)
    
    _h_ = Nu * cond / _L_
    
    
    return Nu, _h_
    
    

def h_inner_annulus(L, T, Twall, fluid, mdot, Di, Do):
    D_hydraulic = Do-Di

def h_outer_annulus(L, T, Twall, fluid, mdot, Di, Do):
    D_hydraulic = Do-Di

def h_circular_pipe(L, T, Twall, fluid, mdot, Diam):
    pass

def h_inside_mixed(L, Tvessel, Tfluid, fluid, mdot, D):
    """
    Calculation of internal mixed natural/forced convective heat transfer coefficient from Nusselt number
    and using the coolprop low level interface.

    Parameters
    ----------
    L : float
        Vessel length
    Tfluid : float
        Temperature of the bulk fluid inventory
    Tvessel : float 
        Temperature of the vessel wall (bulk) 
    fluid : obj
        Coolprop fluid object 
    mdot : float 
        Mass flow
    D : float 
        Characteristic diameter for Reynolds number estimation

    Returns
    ----------
    h_inner : float
        Heat transfer coefficient
    """
    cond = fluid.conductivity()
    visc = fluid.viscosity()
    cp = fluid.cpmass()
    Pr = cp * visc / cond

    T = (Tfluid + Tvessel) / 2
    beta = fluid.isobaric_expansion_coefficient()
    nu = visc / fluid.rhomass()
    Gr = 9.81 * beta * abs(Tvessel - Tfluid) * L ** 3 / nu ** 2
    Ra = Pr * Gr

    NNu_free = Nu(Ra,Pr)  # 0.13 * NRa**0.333

    Re = 4 * abs(mdot) / (visc * math.pi * D)
    NNu_forced = 0.56 * Re ** 0.67
    return (NNu_free + NNu_forced) * cond  / L


def gas_release_rate(P1, P2, rho, k, CD, area):
    """
    Gas massflow (kg/s) trough a hole at critical (sonic) or subcritical
    flow conditions. The formula is based on Yellow Book equation 2.22.

    Methods for the calculation of physical effects, CPR 14E, van den Bosch and Weterings (Eds.), 1996

    Parameters
    ----------
    P1 : float 
        Upstream pressure
    P2 : float 
        Downstream pressure
    rho : float 
        Fluid density
    k : float 
        Ideal gas k (Cp/Cv) 
    CD : float
        Coefficient of discharge
    are : float
        Orifice area

    Returns
    ----------
        : float
        Gas release rate / mass flow of discharge
    """
    if P1 > P2:
        if P1 / P2 > ((k + 1) / 2) ** ((k) / (k - 1)):
            flow_coef = 1
        else:
            flow_coef = (
                2
                / (k - 1)
                * (((k + 1) / 2) ** ((k + 1) / (k - 1)))
                * ((P2 / P1) ** (2 / k))
                * (1 - (P2 / P1) ** ((k - 1) / k))
            )

        return (
            math.sqrt(flow_coef)
            * CD
            * area
            * math.sqrt(rho * P1 * k * (2 / (k + 1)) ** ((k + 1) / (k - 1)))
        )
    else:
        return 0


def relief_valve(P1, Pback, Pset, blowdown, k, CD, T1, Z, MW, area):
    """
    Pop action relief valve model including hysteresis.
    The pressure shall rise above P_set to open and
    decrease below P_reseat (P_set*(1-blowdown)) to close

    Parameters
    ----------
    P1 : float
        Upstream pressure
    Pback : float 
        Downstream / backpressure
    Pset : float
        Set pressure of the PSV / relief valve
    blowdown : float 
        The percentage of the set pressure at which the valve reseats
    k : float 
        Ideal gas k (Cp/Cv) 
    CD : float
        Coefficient of discharge
    T1 : float
        Upstream temperature
    Z : float 
        Compressibility
    MW : float 
        Molecular weight of the gas relieved
    area : float 
        PSV orifice area
    
    Returns
    ----------
        : float 
        Relief rate / mass flow
    """

    global psv_state
    if P1 > Pset:
        eff_area = area
        psv_state = "open"
    elif P1 < Pset * (1 - blowdown):
        eff_area = 0
        psv_state = "closed"
    else:
        if psv_state == "open":
            eff_area = area
        elif psv_state == "closed":
            eff_area = 0
        else:
            raise ValueError("Unknown PSV open/close state.")

    if eff_area > 0:
        return api_psv_release_rate(P1, Pback, k, CD, T1, Z, MW, area)
    else:
        return 0.0


def api_psv_release_rate(P1, Pback, k, CD, T1, Z, MW, area):
    """
    PSV vapour relief rate calculated according to API 520 Part I 2014
    Eq. 5, 9, 15, 18

    Parameters
    ----------
    P1 : float
        Upstream pressure
    Pback : float 
        Downstream / backpressure
    k : float 
        Ideal gas k (Cp/Cv) 
    CD : float
        Coefficient of discharge
    T1 : float
        Upstream temperature
    Z : float 
        Compressibility
    MW : float 
        Molecular weight of the gas relieved
    area : float 
        PSV orifice area

    Returns
    ----------
        : float 
        Relief rate / mass flow
    """


    P1 = P1 / 1000
    Pback = Pback / 1000
    area = area * 1e6 
    MW = MW * 1000
    C = 0.03948 * (k * (2 / (k + 1)) ** ((k + 1) / (k - 1))) ** 0.5
    if P1 / Pback > ((k + 1) / 2) ** ((k) / (k - 1)):
        w = CD * area * C * P1 / math.sqrt(T1 * Z / MW)
    else:
        r = Pback / P1
        f2 = ((k / (k - 1)) * r ** (2 / k) * (1 - r**((k - 1) / k)) / (1 - r))**0.5
        w = CD * area * f2 / (T1 * Z / (MW * P1 * (P1 - Pback)))**0.5 / 17.9
    return w/3600

def cv_vs_time(Cv_max,t,time_constant=0,characteristic="linear"):
    """
    Control valve flow coefficient vs time / actuator postion
    assuming a linear rate of actuator for the three archetypes of 
    characteristics: linear, equal percentage and fast/quick opening. 
    
    Parameters
    ----------
    Cv_max : float 
        Valve flow coefficient at full open position
    t : float
        Time 
    time_constant : float (optional)
        The time required for the actuator to fully open. 
        Default to instant open
    characteristic : string (optional)
        Valve characteristic
        Default to linear.
    """

    if time_constant == 0:
        return Cv_max
    else:
        if characteristic=="linear":
            return Cv_max * min(t/time_constant,1)
        elif characteristic=="eq":
            # https://www.spiraxsarco.com/learn-about-steam/control-hardware-electric-pneumatic-actuation/control-valve-characteristics
            tau=50
            travel=min(t/time_constant,1)
            return Cv_max * math.exp( travel * math.log(tau)) / tau
        elif characteristic=="fast":
            # square root function used
            return Cv_max * min(t/time_constant,1)**(0.5)
        else:
            return Cv_max

def control_valve(P1, P2, T, Z, MW, gamma, Cv, xT=0.75, FP=1):
    """
    Flow calculated from ANSI/ISA control valve equations for single phase gas flow.
    Equation 19 pp. 132 in
    Control Valves / Guy Borden, editor; Paul Friedmann, style editor

    Parameters
    ----------
    P1 : float
        Upstream pressure
    P2 : float 
        Downstream / backpressure
    T : float
        Upstream temperature
    Z : float 
        Upstream compressibility
    MW : float 
        Molecular weight of the gas relieved
    gamma : float 
        Upstream Ideal gas k (Cp/Cv) 
    Cv : float
        Valve coefficient 
    xT : float
        Value of xT for valve fitting assembly, default value
    FP : float
        Piping geometry factor

    Returns
    ----------
        : float
        Mass flow
    """

    P1 = P1 / 1e5
    P2 = P2 / 1e5
    MW = MW * 1000
    N8 = 94.8
    Fk = gamma / 1.4
    x = (P1 - P2) / P1
    if x < 0:
        x = 0
    Y = 1.0 - min(x, Fk * xT) / (3.0 * Fk * xT)
    mass_flow = N8 * FP * Cv * P1 * Y * (MW * min(x, xT * Fk) / T / Z) ** 0.5
    return mass_flow / 3600  # kg/s


def h_inside_pipe(density, velocity, hydraulic_diameter, dynamic_viscosity,
                  specific_heat, thermal_conductivity, surface_diameter):
    """
    Nuf = 0.15 × Ref0.33 × Pr0.43 - at laminar flow;

    Nuf = 0.021 × Ref0.8 × Pr0.43 - at turbulent flow
    
    https://calcdevice.com/forced-convection-of-inner-pipe-surface-id130.html
    
    """
    
    reynolds_number = density * velocity * hydraulic_diameter / dynamic_viscosity
    prandtl_number = specific_heat / (dynamic_viscosity * thermal_conductivity)
    
    if reynolds_number < 2000:
        coef = [0.15, 0.33, 0.43]
    elif reynolds_number > 2000 and reynolds_number < 10000:
        warn("The flow is in the transition region and convective coefficient"+
             " will not be accurate",UserWarning)
        coef = [0.021, 0.8, 0.43]
    else:
        coef = [0.021, 0.8, 0.43]
        
        
    nusselt_number = coef[0] * (reynolds_number ** coef[1] 
                             * prandtl_number ** coef[2])
        
    return nusselt_number * thermal_conductivity / surface_diameter
    
    
    