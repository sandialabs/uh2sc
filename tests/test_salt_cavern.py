# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:21:12 2023

@author: dlvilla
"""

import unittest
from uh2sc import SaltCavern
import os

class Test_SaltCavern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot_results = False
        
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    @unittest.skip("The pressure relief valve type psv must be" 
                 +"added again to uh2sc so that the current case can run. This case must" 
                 +"be fixed and is the first verification case that will be used to" 
                 +"show that the Well class with a single double vertical pipe combination (1 active one inactive)" 
                 +"is working")
    def test_salt_cavern(self):
        """
        Perform a single cycle 
        """

        inp = os.path.join(os.path.dirname(__file__),"test_data","salt_cavern_test.yml")

        sc = SaltCavern(inp)
        """
        RUN
        
        """
        Tvessel = sc.T_cavern
        Tfluid = sc.T_cavern

        Tvessel, Tfluid = sc.step()
        
  
        for i in range(1):
    
            new_inp = sc.input
            new_inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["mdot"] = [-13.0196,-13.0196]
            
            sc.input = new_inp
            sc.validate_input()

            Tvessel, Tfluid  = sc.step()
            new_inp = sc.input
            new_inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["mdot"] = [13.0196,13.0196]
            
            Tvessel, Tfluid = sc.step()
        
        if self.plot_results:
            from matplotlib import pyplot as plt
            plt.plot(sc.cavern_results['Time (s)'],sc.cavern_results['Gas temperature (K)'])
            plt.grid("on")
            plt.xlabel("Time (s)")
            plt.ylabel("Hydrogen gas temperature in cavern (K)")

        self.assertTrue(sc.T_cavern[-1] > 380.0 and sc.T_cavern[-1] < 381.0 )
        self.assertTrue(sc.P_cavern[-1] > 57000000.0 and sc.P_cavern[-1] < 58000000.0)
        minP = sc.P_cavern.min()
        minT = sc.T_cavern.min()
        self.assertTrue(minP > 7400000 and minP < 7410000)
        self.assertTrue(minT > 266.0 and minT < 267.0)
        

if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_SaltCavern())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('utilities_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())            