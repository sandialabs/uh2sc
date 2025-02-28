# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:16:01 2024

@author: dlvilla
"""

class Constants:
    Rg = {'value':8.314,
          'units':'J/mol/K',
          'notes':'universal gas constant'}
    g = {'value':9.81,
         'units':'m/s2',
         'notes':"graviational acceleration at earth's surface"}
    num_conv = {'value':1e-12,
                'unit':'N/A',
                'notes':'numeric convergence to near zero criterion throughout the code'}
    K_C_offset = {'value':273.15,
                  'unit':'Kelvin',
                  'notes':''}
    inches_2_meters = {'value':0.0254,
                       'units':'in/m',
                       'notes':''}
    mmscf_p_day_2_m3_p_s = {'value':28316.846592/86400,
                            'unit':'mmscf/day / m3/s',
                            'notes':'mmscf_p_day stands for million standard'+
                                    ' cubic feet per day. m3_p_s stands for'+
                                    'meters cubed per second'}
    psi_2_pascal = {'value':6894.76,
                    'unit':'psi/Pa',
                    'notes':'psi stands for pounds per square inch'}
    F_R_offset = {'value':458.67,
                  'unit':'Rankine',
                  'notes':'Offset between degrees Fahrenheit and degrees Rankine'}
    Rankine_2_Kelvin = {'value':5/9,
                        'unit':"Rankine/Kelvin",
                        'notes':''}
    feet_2_meters = {'value':0.3048,
                     'unit':'foot/meter',
                     'notes':''}
    two = {'value': 2.0,
           'unit':'non-dimensional'}