#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 21:57:33 2025

This is a place holder for comparing literature based data of gas mixtures 
vs CoolProps output since gas mixtures can have inaccurate outputs.

@author: dlvilla
"""
import unittest

from CoolProp import CoolProp as CP
from uh2sc.thermodynamics import (density_of_brine_water, 
                                  solubility_of_nacl_in_h2o,
                                  brine_saturated_pressure)


class TestThermodynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.print_figures = True
        cls.print_msg = True
        cls.run_all = True
        water = CP.AbstractState("HEOS","H2O")
        water.set_mass_fractions([1.0])
        water.update(CP.PT_INPUTS, 16e6, 373)
        cls.water = water


    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_saline_density(self):

        
        pure_water_density = self.water.rhomass()
        
        saline_density = density_of_brine_water(373, 16e6, 25, pure_water_density)
        
        ratio = saline_density/pure_water_density
        self.assertAlmostEqual(1.1879567860089, ratio)
        
    def test_salt_solubility(self):
        
        sol = solubility_of_nacl_in_h2o(104.44+273)
        self.assertAlmostEqual(sol, 28.279825925827026)
        
    def test_brine_saturated_pressure(self):
        pb_sat = brine_saturated_pressure(293,25,self.water)
        self.assertAlmostEqual(pb_sat, 1840.229586272883)
        
        
        
        
    
if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestThermodynamics())


    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('cool_prop_accuracy_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())