# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:21:12 2023

@author: dlvilla
"""
import os
import unittest

import numpy as np
from matplotlib import pyplot as plt
from uh2sc.model import Model, ADJ_COMP_TESTING_NAME
from CoolProp import CoolProp as CP


def initialize_model():
    inp = {"cavern":{"depth":1000.0,
                     "overburden_pressure":19829198.656747766,
                     "height":304.8,
                     "diameter":25.76033180350307,
                     "emissivity":0.99,
                     "ghe_name":ADJ_COMP_TESTING_NAME},
           "initial":{"temperature":326.5,
                      "pressure":9000000.0,
                      "fluid":"H2",
                      "start_date":"2023-01-01",
                      "liquid_height":1.0,
                      "liquid_temperature": 326.5},
           "calculation":{"time_step": 3000.0,
                          "end_time":1e6, #2.592e6,
                          "run_parallel":False},
           "heat_transfer":{"h_inner":"calc"},
           "wells":{},
           "ghes":{}}
    
    model = Model(inp,
                  single_component_test=True,
                  mdot=[-1,-1],
                  time=[0,inp["calculation"]["end_time"]],
                  type="CAVERN",
                  r_out=inp['cavern']['diameter'],
                  salt_therm_cond=5.190311418685122,
                  farfield_temp=326.5,
                  solver_options={"TOL":1.0e-2})
    return model

class TestSaltCavern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot_results = False

        


    @classmethod
    def tearDownClass(cls):
        pass
    
    def test_ht_coef(self):
        """
        Example 1 from https://www.sfu.ca/~mbahrami/ENSC%20388/Notes/Natural%20Convection.pdf
        
        4 m high maintained at 60C to atmospheric air at 10C 
        10 m wide
        
        
        """
        model = initialize_model()
        cavern = model.components['cavern']
        
        air = CP.AbstractState("HEOS","Air")
        air.update(CP.PT_INPUTS,101325,35+273.15 )
        cavern._p_cavern = 101325
        
        cavern._t_cavern = 60+273.15
        cavern._t_cavern_wall = 10 +273.15
        ht = cavern._wall_ht_coef(4.0,air)
        
        self.assertTrue(ht < 6.0 and ht > 4.0)
    
    def test_evapor_coef(self):
        # values tested come from 
        # https://steamtables.online/
        
        model = initialize_model()
        
        water_m1 = CP.AbstractState("HEOS","Water")
        water_m1.set_mass_fractions([1.0])
        water_m1.update(CP.PT_INPUTS,101325,35+372.15 )
        
        water = CP.AbstractState("HEOS","Water")
        water.set_mass_fractions([1.0])
        water.update(CP.PT_INPUTS,101325,35+374.15)
        
        cavern = model.components['cavern']
        
        # 100 % vaporization
        (mass_vapor, mass_change_vapor, e_vapor_brine, e_vapor_cavern) = (
            cavern._evaporation_energy(water, water_m1, 
            374.15, 372.15, 374.15, 372.15, 1.0, 1.0))
        
        self.assertTrue(mass_vapor > 0.6 and mass_vapor < 0.62)
        self.assertTrue(mass_change_vapor < 0.041 and mass_change_vapor > 0.039)
        self.assertTrue(np.abs(e_vapor_brine) > 90100 and
                        np.abs(e_vapor_brine) < 90200)
    
    
    def test_salt_cavern(self):
        """
        Perform decompressing of the Cavern.
        
        This is a no-flux condition on the cavern that tests whether
        it properly models adiabatic compression and expansion.
        
        """
        model = initialize_model()
        
        model.run()
        
        figd,axd = model.plot_solution([0,1,2,3,4,5])
        
        pass


if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestSaltCavern())

    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('utilities_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())
