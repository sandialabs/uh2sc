# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:53:41 2024

@author: dlvilla
"""

import unittest
# import numpy as np
# from uh2sc.well import Well
# from matplotlib import pyplot as plt
# import pandas as pd
import os
import yaml
from uh2sc.errors import InputFileError
from uh2sc.model import Model

class Test_Well(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        
    @classmethod
    def tearDownClass(cls):
        pass

    
    
    def test_invalid_input_with_2_wells(self):
        inp_path = os.path.join(os.path.dirname(__file__), "test_data", 
                                "nieland_verification_methane_1_cycles.yml")

        # Loop over variable inputs
        with open(inp_path, 'r', encoding='utf-8') as infile:
            inp = yaml.load(infile, Loader=yaml.FullLoader)
            
        inp["ghes"]["my_big_ghe"] = inp["ghes"]["nielson_ghe"]
        
        with self.assertRaises(InputFileError):
            Model(inp)
    
    
if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_Well())
    
    #obj = Test_Well()
    
    #obj.test_well()
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('well_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())  
        