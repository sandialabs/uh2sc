#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:10:40 2025

@author: dlvilla
"""
import unittest
import os
import warnings
import logging
import yaml

import numpy as np

from uh2sc.model import Model
from uh2sc.utilities import (process_CP_gas_string, filter_cpu_count,
                             find_all_fluids, reservoir_mass_flows,
                             calculate_component_masses)
from CoolProp import CoolProp as CP

class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.print_figures = True
        cls.print_msg = True
        cls.run_all = True
        inp_path = os.path.join(os.path.dirname(__file__),"test_data","nieland_verification_h2_4_cycles.yml")
        with open(inp_path, 'r', encoding='utf-8') as infile:
            inp = yaml.load(infile, Loader=yaml.FullLoader)
            
            
        #logging.warning("Please ignore all warnings between this and another string saying to start paying attention!")
        inp["calculation"]["run_parallel"] = True
        cls.model = Model(inp=inp)
        cls.model.logging.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_process_CP_gas_strings(self):
        """
        Go to http://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids

        for the kinds of fluids you can enter.

        """
        model = self.model
        
        str1 = "H2"

        comp, molefracs, compSRK, fluid = process_CP_gas_string(str1,"HEOS", model)

        str1 = "H2[0.9]&Methane[0.1]"

        comp, molefracs, compSRK, fluid = process_CP_gas_string(str1,"HEOS", model)

        str1 = "H2[0.9]&CO2[0.1]"

        comp, molefracs, compSRK, fluid = process_CP_gas_string(str1,"HEOS", model)

        # This is an infeasible mixture!

        str2 = "H2[0.8]&Methane[0.16]&Ethane[0.02]&Propane[0.01]&Butane[0.005]&CarbonDioxide[0.005]"
        comp, molefracs, compSRK, fluid = process_CP_gas_string(str2,"HEOS", model)

        # test an incorrect string:
        with self.assertRaises(ValueError):
            str3 = "H2&Methane[0.1]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str3,"HEOS", model)
        # bad fluid string
        with self.assertRaises(ValueError):
            str4 = "sldkfhjaglk;jhsalfkj"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str4,"HEOS", model)
        with self.assertRaises(ValueError):
            str5 = "H2[0.9]&Methane[0.1dk]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str5,"HEOS", model)
        # non-float in brackets
        with self.assertRaises(ValueError):
            str6 = "H2[not a number]&Methane[0.1]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str6,"HEOS", model)
        # mole fractions do not add to 1.
        with self.assertRaises(ValueError):
            str6 = "H2[0.85]&Methane[0.1]"
            comp, molefracs, compSRK, fluid = process_CP_gas_string(str6,"HEOS", model)

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


    def test_find_all_fluids_and_mass_flow(self):

        model = self.model
        # include a valve that has no effect
        inputs = {'calculation':{"min_time_step": 1200,
                                      "max_time_step": 172800,
                                      "end_time": 31104000,
                                      "run_parallel": True,
                                      "cool_prop_backend":"HEOS",
                                      "machine_learning_acceptable_percent_error":0.1},
            
                       'initial':{'fluid':'Ethane[0.1]&Methane[0.9]',
                                  'pressure':101325.0,
                                  'temperature':323.0},
                       'wells':{'cavern_well':{'valves':{'inflow_mdot':
                            {'type':'mdot','reservoir':{'fluid':'Ethane[0.1]&Methane[0.9]'},'mdot':[0,10],'time':[0,100]},
                                            'mdot2_valve':{'type':'mdot','reservoir':{'fluid':'Ethane[0.05]&Methane[0.9]&Butane[0.05]'},'mdot':[0,-10,10],'time':[0,50,100]}}}}}
        model.inputs = inputs


        fluid_components3, fluids3, mdots3 = find_all_fluids(model)


        self.assertListEqual(fluid_components3, ['Ethane', 'Methane', "Butane"])

        mdot_valves, mdot_cavern = reservoir_mass_flows(model,5.0)
        self.assertAlmostEqual(-1.0, mdot_valves['cavern_well']['mdot2_valve'].sum())
        self.assertAlmostEqual(0.5, mdot_valves['cavern_well']['inflow_mdot'].sum())
        self.assertAlmostEqual(-0.5, mdot_cavern.sum())



        inputs2 = {'initial':{'fluid':'H2',
                                  'pressure':101325.0,
                                  'temperature':323.0},
                       'wells':{'cavern_well':{'valves':{'inflow_mdot':
                            {'type':'mdot','reservoir':{'fluid':'H2'},'mdot':[0,10],'time':[0,100]}}}}}
        inputs2["calculation"] = inputs["calculation"]
        model.inputs = inputs2
            

        fluid_components, fluids, mdots = find_all_fluids(model)
        self.fluid_components = fluid_components
        self.fluids = fluids

        mdot_valves, mdot_cavern = reservoir_mass_flows(model,0.0)

        self.assertAlmostEqual(mdot_cavern[0], 0.0)


        self.assertEqual(fluid_components[0], "H2")

        inputs3 = {'initial':{'fluid':'Methane[0.749]&Ethane[0.25]&CO2[0.001]',
                                  'pressure':101325.0,
                                  'temperature':323.0},
                       'wells':{'cavern_well':{'valves':{'inflow_mdot':
                            {'type':'mdot','reservoir':{'fluid':'Methane[0.75]&Ethane[0.15]&CO2[0.1]'},'mdot':[0,10],'time':[0,100]}}}}}
        inputs3["calculation"] = inputs["calculation"]
        
        model.inputs = inputs3

        fluid_components2, fluids2, mdots2 = find_all_fluids(model)

        self.assertListEqual(fluid_components2, ['Methane', 'Ethane', 'CO2'])


    def test_calculate_component_masses(self):

        test_mass_fractions = [0.05,0.45,0.5]
        fluid = CP.AbstractState("HEOS", "H2&Methane&H2O")
        fluid.set_mass_fractions(test_mass_fractions)
        fluid.update(CP.PT_INPUTS,1e5,323)

        masses = calculate_component_masses(fluid,1.0)

        self.assertLess(np.abs((masses-np.array(test_mass_fractions)).sum()), 1e-8)


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
