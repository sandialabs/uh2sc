# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:21:12 2023

@author: dlvilla
"""
import os
import unittest
from matplotlib import pyplot as plt
from uh2sc import SaltCavern

class TestSaltCavern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot_results = False


    @classmethod
    def tearDownClass(cls):
        pass

    def test_salt_cavern(self):
        """
        Perform a single cycle
        
        This is a no-flux condition on the cavern that tests whether
        it properly models adiabatic compression and expansion.
        
        """
        
        
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
