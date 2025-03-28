#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:30:13 2025

@author: dlvilla
"""
import numpy as np

from uh2sc.constants import Constants
from CoolProp import CoolProp as CP

con = Constants()


def density_of_brine_water(temperature_K: float, pressure_Pa: float, 
                           salt_weight_percent: float,
                           density_pure_water: float,
                           pub="Numbere_etal_1977"): 
    
    if pub == "Numbere_etal_1977":
    # D. Numbere, W.E. Brigham, and M.B. Standing. 1977. "Correlations for
    # Physical Properties of Petroleum Reservoir Brines" Stanford University
    # Petroleum Research Institute" November Standford, CA. 
    # https://www.osti.gov/servlets/purl/6733264
    
        temperature_F = temperature_K / con.rankine_2_kelvin['value'] - con.f_r_offset['value']
        pressure_psi = con.pa_2_psi['value'] * pressure_Pa
        
        if temperature_F < 0 or temperature_F > 400:
            raise ValueError("The valid temperature range is 0 to 400F")
        elif pressure_psi < 14 or pressure_psi > 1.0e4:
            raise ValueError("The valid pressure range is 14 (1atm) to 10,000psi!")
        elif salt_weight_percent < 0 or salt_weight_percent > 30: # allow maximum solubility values!
            raise ValueError("The valid salt weight percent range is 0 to 25%")
            
        cs = salt_weight_percent
        temp = temperature_F
        pres = pressure_psi
        
        
        return density_pure_water * (cs * (7.65e-3 -1.09e-7 * pres + 
            cs *(2.16e-5 +1.74e-9 * pres)-(1.07e-5 - 3.24e-10*pres)*temp + 
            (3.76e-8-1.0e-12*pres)*temp**2) + 1)

def solubility_of_nacl_in_h2o(temperature_K: float):
   """
   Data source 1 (higher less precise)
   # https://www.sigmaaldrich.com/US/en/support/calculators-and-apps/solubility-table-c
   # ompounds-water-temperature?srsltid=AfmBOoojvIACeaEc-6cpTNmu96FdF37mv-ti2F9JOH72bmfOfZbmSTAm
   # 0.0     20      40      60      80     100    Content at 20C in %  solubility in g/100g  density
   #35.6	35.8	36.42	37.05	38.05	39.2	26.4  1.201
   
   Data source 2
   #
   
   http://www.chlorates.exrockets.com/nacl.html						
Temperature Ñ”F	Temperature Ñ”C	%Salt	log T(K)			Error (%)
-6	-21.11	23.31	2.401210926	23.31	27.85833514	19.51237725
0	-17.78	23.83	2.406914704	23.83	27.26444018	14.41225421
10	-12.22	24.7	2.416274281	24.7	26.67876112	8.011178642
20	-6.67	25.53	2.425420089	25.53	26.41328681	3.459799472
30	-1.1	26.16	2.434409208	26.16	26.32234728	0.620593591
32	0	26.29	2.436162647	26.29	26.31679308	0.101913595
32.2	0.1	26.31	2.4363217	26.31	26.31643355	0.006433549
40	4.44	26.33	2.443169076	26.33	26.31918299	-0.041082446
50	10	26.36	2.451786436	26.36	26.35456526	-0.020617364
60	15.56	26.395	2.460236129	26.395	26.40441859	0.035683227
70	21.11	26.45	2.468509791	26.45	26.45953333	0.03604285
80	26.67	26.52	2.476643268	26.52	26.51927626	-0.00272903
100	37.78	26.68	2.492453062	26.68	26.66407549	-0.059687055
125	51.67	26.92	2.511442161	26.92	26.91789019	-0.007837329
150	65.56	27.21	2.529635646	27.21	27.24105895	0.114145335
175	79.44	27.62	2.547085193	27.62	27.58315718	-0.133391811
200	93.33	27.91	2.563872486	27.91	27.92974627	0.070749804
212	100	28.12	2.571708832	28.12	28.1238066	0.013536969
220	104.44	28.29	2.576847924	28.29	28.2798239	-0.035970664
227.5	108.7	28.46	2.58172216	28.46	28.46190226	0.00668398

Fit is just for > 0C data
R² = 9.995934087841450E-01
and largest error is 0.114%
   
   
   """
   if temperature_K < 273 or temperature_K > 381.85:
       raise ValueError("Temperature must be between 273K and 381.85K")
   
   coef = np.array([3.172251803710940E+06,
                    -4.764736792489000E+07,
                    2.981692711541820E+08,
                    -9.950606956585000E+08,
                    1.867767632899880E+09,
                    -1.869644298615200E+09,
                    7.797370187705370E+08],dtype=np.float64)
   
   return np.polyval(coef,np.log10(temperature_K))
   
   
def brine_saturated_pressure(temperature_K, salt_weight_percent:float, water: float):
    """
    Geral L. Dittman 1977. "Calculation of Brine Properties." February Lawrence
    Livermore National Laboratories. https://www.osti.gov/servlets/purl/7111583

    Parameters
    ----------
    temperature_K : float
        temperature of the water in kelvin
    salt_weight_percent : float
        DESCRIPTION.
    water : CoolProp AbstractState object
        

    Returns
    -------
    float
        Saturated vapor pressure of saline water with "salt_weight_percent" and
        temperature_K 

    """
    Tcurrent = water.T()
    Pcurrent = water.p()
    water.update(CP.QT_INPUTS, 1.0, Tcurrent)
    psat = water.p()
    
    # return to the state before calling this function.
    water.update(CP.PT_INPUTS, Pcurrent, Tcurrent)
    
    # the last point is extrapolated. the data goes to 25%
    a1_data = np.array([0.969,0.934,0.894,0.847,0.794,0.794+(0.749-0.847)/5])
    xs_data = np.array([5,10,15,20,25,30])
    
    a1 = np.interp(salt_weight_percent, xs_data, a1_data)
    
    return a1 * psat