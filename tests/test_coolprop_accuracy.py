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

class TestCoolPropAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.print_figures = True
        cls.print_msg = True
        cls.run_all = True


    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_place_holder(self):
        pass
    
    
if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestCoolPropAccuracy())


    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('cool_prop_accuracy_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())