# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license

# pylint: disable=invalid-name,c-extension-no-member

import unittest
import os
import cProfile
import pstats
import io

import pytest
import yaml

from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP

from uh2sc import transport as tp
from uh2sc import validator


def get_example_input(fname):

    fname = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..","examples",fname)
    with open(fname,'r',encoding='utf-8') as infile:
        input_ = yaml.load(infile, Loader=yaml.FullLoader)

    return input_

class TestBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot_results = False

    @classmethod
    def tearDownClass(cls):
        pass

    def test_orifice(self):
        P1 = 10.0e5
        P2 = 5.5e5
        D = PropsSI("D", "P", P1, "T", 298.15, "HEOS::N2")
        cpcv = PropsSI("CP0MOLAR", "T", 298.15, "P", P1, "HEOS::N2") / PropsSI(
            "CVMOLAR", "T", 298.15, "P", P1, "HEOS::N2"
        )
        assert tp.gas_release_rate(
            P1, P2, D, cpcv, 0.85, 0.01 ** 2 / 4 * 3.1415
        ) == pytest.approx(9.2 / 60, rel=0.001)


    def test_orifice1(self):
        P1 = 10.0e5
        P2 = 6.5e5
        D = PropsSI("D", "P", P1, "T", 298.15, "HEOS::N2")
        cpcv = PropsSI("CP0MOLAR", "T", 298.15, "P", P1, "HEOS::N2") / PropsSI(
            "CVMOLAR", "T", 298.15, "P", P1, "HEOS::N2"
        )
        assert tp.gas_release_rate(
            P1, P2, D, cpcv, 0.85, 0.01 ** 2 / 4 * 3.1415
        ) == pytest.approx(9.2 / 60, rel=0.2)


    def test_controlvalve(self):
        P1 = 10.0e5
        P2 = 5.5e5
        T1 = 20.0 + 273.15
        MW = PropsSI("M", "P", P1, "T", T1, "HEOS::N2")
        Z1 = PropsSI("Z", "P", P1, "T", T1, "HEOS::N2")
        gamma = PropsSI("CP0MOLAR", "T", T1, "P", P1, "HEOS::N2") / PropsSI(
            "CVMOLAR", "T", T1, "P", P1, "HEOS::N2"
        )
        assert tp.control_valve(P1, P2, T1, Z1, MW, gamma, 500) == pytest.approx(
            21.92, rel=0.05
        )

    def test_cv_vs_time(self):
        assert tp.cv_vs_time(1,0.5,time_constant=1,characteristic="linear") == 0.5
        assert tp.cv_vs_time(1,0.5,time_constant=1,characteristic="eq"
                             ) == pytest.approx(0.14, abs=0.002)
        assert tp.cv_vs_time(1,0.5,time_constant=1,characteristic="fast"
                             ) == pytest.approx(0.707, abs=0.002)
        assert tp.cv_vs_time(1,0.5,time_constant=0) == 1.0

    def test_psv3(self):
        Pback = 1e5
        Pset = 18.2e5
        blowdown = 0.1
        P1 = 0.99 * Pset * (1 - blowdown)
        T1 = 100.0 + 273.15
        Z = PropsSI("Z", "P", P1, "T", T1, "HEOS::N2")
        MW = PropsSI("M",  "HEOS::N2")
        gamma = PropsSI("CP0MOLAR", "T", T1, "P", P1, "HEOS::N2") / PropsSI(
            "CVMOLAR", "T", T1, "P", P1, "HEOS::N2"
        )
        CD = 0.975
        area = 71e-6
        assert tp.relief_valve(
            P1, Pback, Pset, blowdown, gamma, CD, T1, Z, MW, area
        ) == 0
        assert tp.relief_valve(
            P1*1.01, Pback, Pset, blowdown, gamma, CD, T1, Z, MW, area
        ) == 0


    def test_psv2(self):
        P1 = 21.0e5
        Pback = 1e5
        Pset = 20.99e5
        blowdown = 0.1
        T1 = 100.0 + 273.15
        Z = PropsSI("Z", "P", P1, "T", T1, "HEOS::N2")
        MW = PropsSI("M",  "HEOS::N2")
        gamma = PropsSI("CP0MOLAR", "T", T1, "P", P1, "HEOS::N2") / PropsSI(
            "CVMOLAR", "T", T1, "P", P1, "HEOS::N2"
        )
        CD = 0.975
        area = 71e-6
        assert tp.relief_valve(
            P1, Pback, Pset, blowdown, gamma, CD, T1, Z, MW, area
        ) == pytest.approx(1046 / 3600, rel=0.03)
        assert tp.relief_valve(
            Pset*0.99, Pback, Pset, blowdown, gamma, CD, T1, Z, MW, area
        ) == pytest.approx(1046 / 3600, rel=0.03)


    def test_psv(self):
        P1 = 100e5
        Pback = 1e5
        Pset = 99.2e5
        blowdown = 0.1
        T1 = 25.0 + 273.15
        Z = PropsSI("Z", "P", P1, "T", T1, "HEOS::N2")
        MW = PropsSI("M",  "HEOS::N2")
        gamma = PropsSI("CP0MOLAR", "T", T1, "P", P1, "HEOS::N2") / PropsSI(
            "CVMOLAR", "T", T1, "P", P1, "HEOS::N2"
        )
        CD = 0.975
        area = 71e-6
        assert tp.relief_valve(
            P1, Pback, Pset, blowdown, gamma, CD, T1, Z, MW, area
        ) == pytest.approx(1.57, rel=0.02)


    def test_api_psv_relief(self):
        assert tp.api_psv_release_rate(121.9e5, 71e5, 1.39, 0.975, 298.15, 1.01, 2/1e3, 71e-6
                                       ) == pytest.approx(1846/3600, rel=0.01)
        assert tp.api_psv_release_rate(121.9e5, 1e5, 1.39, 0.975, 298.15, 1.01, 2/1e3, 71e-6
                                       ) == pytest.approx(1860/3600, rel=0.01)


    def test_hinside(self):
        fluid = CP.AbstractState("HEOS","air")
        Tboundary = (311 + 505.4) / 2
        fluid.update(CP.PT_INPUTS, 1e5, Tboundary)
        h = tp.h_inside(0.305,311,505.4,fluid)
        assert h == pytest.approx(7, abs=0.1)


    def test_hinner_mixed(self):
        # Value changed from 7 to 7.6 because of annular mode of heat transport.
        # A more
        mdot = 1e-10
        D = 0.010
        fluid = CP.AbstractState("HEOS","Air")
        fluid.update(CP.PT_INPUTS, 101325, 311)
        h_ = tp.h_inside_mixed(0.305, 311, 505.4, fluid, mdot, D)
        assert h_ == pytest.approx(7.6, abs=0.05)


    def test_nnu(self):
        NGr = tp.Gr(0.305, 311, 505.4, 1e5, "HEOS::air")
        NPr = tp.Pr((311 + 505.4) / 2, 1e5, "HEOS::air")
        NRa = NGr * NPr
        NNu = tp.Nu(NRa, NPr)
        assert NNu == pytest.approx(62.2, abs=1.0)


    def test_nra(self):
        NGr = tp.Gr(0.305, 311, 505.4, 1e5, "HEOS::air")
        NPr = tp.Pr((311 + 505.4) / 2, 1e5, "HEOS::air")
        assert (NGr * NPr) == pytest.approx(1.3e8, abs=0.1e8)


    def test_ngr(self):
        assert tp.Gr(0.305, 311, 505.4, 1e5, "HEOS::air") == pytest.approx(1.8e8, abs=0.1e8)


    def test_npr(self):
        assert tp.Pr((311 + 505.4) / 2, 1e5, "HEOS::air") == pytest.approx(0.7, rel=0.01)

    def test_validator(self):

        dir_ = os.path.join(os.path.abspath(os.path.dirname(__file__)),"test_data")

        for fname in os.listdir(dir_):
            if fname[-4:] == ".yml":
                with open(os.path.join(dir_,fname), encoding='utf-8') as infile:
                    input_ = yaml.load(infile, Loader=yaml.FullLoader)
                assert validator.validation(input_)


if __name__ == "__main__":
    profile = False

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestBasics())

    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('testall_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())
