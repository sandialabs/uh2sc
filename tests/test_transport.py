# -*- coding: utf-8 -*-
# pylint: disable=c-extension-no-member
"""
Created on Wed Mar 20 15:05:03 2024

@author: dlvilla
"""

import unittest
import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from CoolProp import CoolProp as CP
from uh2sc.transport import annular_nusselt,circular_nusselt

def prepare_csv_data(filename):
    df = pd.read_csv(filename,names=["Re","Nu"],dtype=float)
    return df


class TestTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_all_tests = True
        cls.plot_results = True
        cls.verify_data_path = os.path.join(os.path.dirname(__file__),
            "test_data","Gnielinski_Figue1_Pr07_powers_of_10.csv")


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
        d_i = 1e-40
        d_o = 0.02

        length = (d_o - d_i)/0.001


        temp_b = initial_temperature
        temp_w = initial_temperature - 1.0

        # this does not matter for this comparison since F_ann
        is_inner = True
        d_factor = (np.pi / 4.0) * (d_o + d_i)
        visc = fluid.viscosity()
        #Re = mdot / (visc * D_factor)

        mdot1 = 4000.0 * visc * d_factor
        mdot2 = 1e6 * visc * d_factor

        nu_comp = prepare_csv_data(self.verify_data_path)

        mdot_arr = np.arange(mdot1,mdot2,0.005)
        nu_arr = np.zeros(len(mdot_arr))
        error_arr = np.zeros(len(mdot_arr))
        nu_verify = np.zeros(len(mdot_arr))
        re_arr = np.zeros(len(mdot_arr))

        for idx,mdot in enumerate(mdot_arr):


            nu_arr[idx], _h = annular_nusselt(mdot, length, d_i, d_o, temp_b, temp_w, is_inner,
                                        fluid, is_test=True)

            re_arr[idx] = mdot / (visc * d_factor)

            nu_verify[idx] = 10**np.interp(np.log10(re_arr[idx]),
                   nu_comp["Re"].values, nu_comp["Nu"].values)

            error_arr[idx] = 100*(nu_arr[idx] - nu_verify[idx])/nu_verify[idx]

            self.assertTrue(-2.0 < error_arr[idx] and error_arr[idx] < 2.0)


        _fig,ax = plt.subplots(1,1,figsize=(10,10))

        ax.loglog(re_arr,nu_verify,marker="x")
        ax.loglog(re_arr,nu_arr,marker="o")
        ax.grid("on")

        #print(error_arr)
        #print("L = " + str(L))

        # test laminar
        mdot = mdot1 / 2

        # this case will warn that _a_ < 0.05 we don't want this to display
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nu_test = annular_nusselt(mdot, length, d_i, d_o, temp_b, temp_w, is_inner,
                                    fluid, is_test=True)

        self.assertAlmostEqual(nu_test[0], 100.0)

        d_i = 0.01
        d_o = 0.02

        length = (d_o - d_i)/0.001
        is_inner = False
        d_factor = (np.pi / 4.0) * (d_o + d_i)
        visc = fluid.viscosity()

        mdot1 = 1000.0 * visc * d_factor

        nu_test = annular_nusselt(mdot, length, d_i, d_o, temp_b, temp_w, is_inner,
                                    fluid, is_test=True)

        self.assertAlmostEqual((5.099 + 4.979)/2, nu_test[0])

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

        diam = 0.02
        visc = fluid.viscosity()
        #Re = 4.0 * mdot /(visc * np.pi * _D_)
        mdot = 10000.0 * visc * np.pi * diam / 4.0

        length = 20.0

        nu, _h = circular_nusselt(mdot, length, diam, fluid)

        self.assertAlmostEqual(31.111083536870957, nu)

        mdot = 1000.0 * visc * np.pi * diam / 4.0

        nu, _h = circular_nusselt(mdot, length, diam, fluid)

        self.assertAlmostEqual(nu, 4.36)


if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestTransport())

    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('utilities_test_profile.txt', 'w+',encoding='utf-8') as f:
            f.write(s.getvalue())
