#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:10:40 2025

@author: dlvilla
"""
import unittest
import os
import warnings

from uh2sc.utilities import process_CP_gas_string, filter_cpu_count


class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.print_figures = True
        cls.print_msg = True
        cls.run_all = True


    @classmethod
    def tearDownClass(cls):
        pass
    
    def test_process_CP_gas_strings(self):
        """
        Go to http://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids
        
        for the kinds of fluids you can enter.
        
        """
        
        
        str1 = "H2"
        
        comp, molefracs, compSRK, fluid = process_CP_gas_string(str1)
        
        str2 = "H2[0.8]&Methane[0.16]&Ethane[0.02]&Propane[0.01]&Butane[0.005]&CarbonDioxide[0.005]"
        
        comp, molefracs, compSRK, fluid = process_CP_gas_string(str2)
        
        # test an incorrect string:
        with self.assertRaises(ValueError):
            str3 = "H2&Methane[0.1]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str3)
        # bad fluid string
        with self.assertRaises(ValueError):
            str4 = "sldkfhjaglk;jhsalfkj"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str4)
        with self.assertRaises(ValueError):
            str5 = "H2[0.9]&Methane[0.1dk]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str5)
        # non-float in brackets
        with self.assertRaises(ValueError):
            str6 = "H2[not a number]&Methane[0.1]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str6)  
        # mole fractions do not add to 1.
        with self.assertRaises(ValueError):
            str6 = "H2[0.85]&Methane[0.1]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str6)   
            
    def test_filter_cpu_count(self):
        
        max_cpu = os.cpu_count()
        
        
        less_than = max_cpu - 2
        
        more_than = max_cpu + 1
        
        warnings.filterwarnings('ignore')
        cpu_count = filter_cpu_count(more_than)
        self.assertEqual(cpu_count,max_cpu-1)
        warnings.filterwarnings('always')
        
        if less_than > 0:
            cpu_count = filter_cpu_count(less_than)
            self.assertEqual(less_than, cpu_count)
        
        cpu_count = filter_cpu_count(None)
        self.assertEqual(cpu_count,max_cpu - 1)
        
        cpu_count = filter_cpu_count(1)
        self.assertEqual(cpu_count,1)

        
    

if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestUtilities())


    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('utilities_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())