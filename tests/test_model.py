#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:22:27 2025

@author: dlvilla
"""

import unittest
import yaml
import os


from uh2sc.model import Model

class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.filedir = os.path.dirname(__file__)
        inp_path = os.path.join(cls.filedir,"test_data",
                                "nielson_verification_h2_12_cycles.yml")
        # load constants for the Nieland paper.
        with open(inp_path, 'r', encoding='utf-8') as infile:
            cls.inp = yaml.load(infile, Loader=yaml.FullLoader)
            

    @classmethod
    def tearDownClass(cls):
        pass



    def test_write_model(self):
        # short and sweet.
        self.inp['calculation']['end_time'] = 2 * 24 * 3600
        
        self.inp["initial"]["pressure"] = 20511902.93775
        self.inp["initial"]["fluid"] = "H2"
        self.inp['initial']['temperature'] = 326.5
        
        model = Model(self.inp,solver_options={"TOL": 1.0e-2})
        
        model.run()
        
        solution = model.solutions
        
        print(model.xg_description[0:6])
        print(solution[172800.0][0:6])
        
        model.write_results()
        



if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestModel())


    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('model_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())
        
        
        