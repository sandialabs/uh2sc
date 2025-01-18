# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:05:03 2024

@author: dlvilla
"""

import unittest

import numpy as np
from CoolProp import CoolProp as CP
from uh2sc.transport import annular_nusselt,circular_nusselt
import pandas as pd
import os
from matplotlib import pyplot as plt
import warnings

def prepare_csv_data(filename):
    df = pd.read_csv(filename,names=["Re","Nu"],dtype=float)
    return df
    

class Test_Transport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_all_tests = True
        cls.plot_results = True
        cls.verify_data_path = os.path.join(os.path.dirname(__file__),"test_data","Gnielinski_Figue1_Pr07_powers_of_10.csv")

    
    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_nusselt(self):
        # compare to Figure 1 calculations for Prandlt = 0.7
            
            comp = "H2&CH4"
            # This molefrac combination produces a Prandlt number of 0.7
            # more viscous flows are not of great interest to this work
            molefracs = [0.033001,0.966999]
            
            
            fluid = CP.AbstractState("HEOS", comp)
            fluid.specify_phase(CP.iphase_gas)
    
            fluid.set_mole_fractions(molefracs)
    
            initial_pressure = 6e6
            initial_temperature = 373 #K
            
            fluid.update(CP.PT_INPUTS, initial_pressure,  
                                            initial_temperature)
            
            # Use this to hone in on a Prandtl number.
            #Pr = fluid.viscosity() * fluid.cpmass() /fluid.conductivity()
            #print(Pr)
            
            # NOT SURE WHY THIS IS THE SOLUTION, WHY DID GNIELINSKI not specify
            # the 'a' ratio?
            Di = 1e-40
            Do = 0.02
            
            L = (Do - Di)/0.001
            
            
            Tb = initial_temperature
            Tw = initial_temperature - 1.0
            
            # this does not matter for this comparison since F_ann 
            is_inner = True
            D_factor = (np.pi / 4.0) * (Do + Di)
            visc = fluid.viscosity()
            #Re = mdot / (visc * D_factor)
            
            mdot1 = 4000.0 * visc * D_factor 
            mdot2 = 1e6 * visc * D_factor
            
            Nu_comp = prepare_csv_data(self.verify_data_path)
            
            mdot_arr = np.arange(mdot1,mdot2,0.005)
            Nu_arr = np.zeros(len(mdot_arr))
            error_arr = np.zeros(len(mdot_arr))
            Nu_verify = np.zeros(len(mdot_arr))
            Re_arr = np.zeros(len(mdot_arr))
            
            for idx,mdot in enumerate(mdot_arr):
                

                Nu_arr[idx], h = annular_nusselt(mdot, L, Di, Do, Tb, Tw, is_inner, 
                                          fluid, is_test=True)
                
                Re_arr[idx] = mdot / (visc * D_factor)
                
                
                
                Nu_verify[idx] = 10**np.interp(np.log10(Re_arr[idx]),Nu_comp["Re"].values, Nu_comp["Nu"].values)
                
                error_arr[idx] = 100*(Nu_arr[idx] - Nu_verify[idx])/Nu_verify[idx]
                
                self.assertTrue(-2.0 < error_arr[idx] and error_arr[idx] < 2.0)
                
                
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            
            ax.loglog(Re_arr,Nu_verify,marker="x")
            ax.loglog(Re_arr,Nu_arr,marker="o")
            ax.grid("on")
                
            #print(error_arr)
            #print("L = " + str(L))
            
            # test laminar
            mdot = mdot1 / 2
            
            # this case will warn that _a_ < 0.05 we don't want this to display 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Nu_test = annular_nusselt(mdot, L, Di, Do, Tb, Tw, is_inner, 
                                      fluid, is_test=True)
            
            self.assertAlmostEqual(Nu_test[0], 100.0)
            
            Di = 0.01
            Do = 0.02
            
            L = (Do - Di)/0.001
            is_inner = False
            D_factor = (np.pi / 4.0) * (Do + Di)
            visc = fluid.viscosity()
            #Re = mdot / (visc * D_factor)
            
            mdot1 = 1000.0 * visc * D_factor 
            
            Nu_test = annular_nusselt(mdot, L, Di, Do, Tb, Tw, is_inner, 
                                      fluid, is_test=True)
            
            self.assertAlmostEqual((5.099 + 4.979)/2, Nu_test[0])
            
    def test_circular_nusselt(self):
        comp = "Air"
        # This molefrac combination produces a Prandlt number of 0.7
        # more viscous flows are not of great interest to this work
        molefracs = [1.0]
        
        
        fluid = CP.AbstractState("HEOS", comp)
        fluid.specify_phase(CP.iphase_gas)

        fluid.set_mole_fractions(molefracs)

        initial_pressure = 6e6
        initial_temperature = 373 #K
        
        fluid.update(CP.PT_INPUTS, initial_pressure,  
                                        initial_temperature)
        
        D = 0.02
        visc = fluid.viscosity()
        #Re = 4.0 * mdot /(visc * np.pi * _D_)
        mdot = 10000.0 * visc * np.pi * D / 4.0
        
        L = 20.0
        
        Nu, h = circular_nusselt(mdot, L, D, fluid)
        
        self.assertAlmostEqual(31.111083536870957, Nu)
        
        mdot = 1000.0 * visc * np.pi * D / 4.0
        
        Nu, h = circular_nusselt(mdot, L, D, fluid)
        
        self.assertAlmostEqual(Nu, 4.36)
            
            
if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_Transport())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('utilities_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())      