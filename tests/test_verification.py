# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 2023 3:34 PM CST

This test takes ~250 s to execute and is longer than all of the rest.

@author: dlvilla
"""

import os
from datetime import datetime, timedelta
import unittest

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import yaml

from CoolProp.CoolProp import PropsSI
from uh2sc import SaltCavern
from uh2sc.model import Model

def prepare_csv_data(nonleapyear,filename):
    df = pd.read_csv(filename,names=["day of year","degrees fahrenheit"],dtype=float)
    # this is a dummy year
    epoch = datetime(nonleapyear,1,1)
    result = [pd.Timestamp(epoch + timedelta(days=dayofyear)) for dayofyear in df["day of year"]]

    df.index = result

    return df

def create_df_from_sim_output(nonleapyear,results_dict):

    df = pd.DataFrame(results_dict)
    epoch = datetime(nonleapyear,1,1)
    result = [pd.Timestamp(epoch + timedelta(seconds=second_of_year))
              for second_of_year in df["Time (s)"]]

    df.index = result

    return df


def fahrenheit_to_kelvin(deg_fahrenheit):
    return 5/9 * (deg_fahrenheit - 32) + 273.15

class Constants:
    # physical constants
    gravitational_constant = 9.81 # m/s2

    # unit conversion
    feet_to_meters = 12*2.54/100
    seconds_per_hour = 3600
    hours_per_year = 8760
    hours_per_day = 24

    lbs_ft3_2_kg_m3 = 16.01846337396
    btu_p_hrftF_2_W_p_m_K = 1/0.578
    btu_p_lbF_2_J_p_kgK = 4186
    tstep = seconds_per_hour/12 # 2.5 min time step
    scale_factor_between_kelvin_and_rankin = 1.8
    psi_to_Pa = 6894.75729
    nonleapyear = 2009
    second_in_day = 24*3600
    days_per_year = 360  # per Nielson


class Nielson2008CavernCase(object):

    def __init__(self):

        """

        SETUP AND CONSTANTS, NONE OF THESE CHANGE RUN-TO-RUN

        """
        con = Constants()

        inp = {}
        subcat = ["vessel","initial","calculation","valve","heat_transfer","validation","reservoir"]
        for cat in subcat:
            inp[cat] = {}



        # CAVERN GEOMETRY
        self.cavern_height = 1000 * con.feet_to_meters
        self.cavern_volume = 5.61e6 * con.feet_to_meters **3
        self.cavern_diameter = np.sqrt(self.cavern_volume / (self.cavern_height * np.pi * 0.25))
        self.cavern_top_depth = 3000 * con.feet_to_meters


        # CAVERN CHARACTERISTICS
        _overburden_pressure = con.gravitational_constant * (1000 * con.feet_to_meters
                                                        * 144 * con.lbs_ft3_2_kg_m3
                                                    + 2000 * con.feet_to_meters
                                                        * 135 * con.lbs_ft3_2_kg_m3)
                                                        #Pa  ~ pressure at 1000 m

        # values from Nieland2008
        max_casing_seat_pressure = 2465 * con.psi_to_Pa
        max_pressure_gradient = 0.85 * con.psi_to_Pa / con.feet_to_meters
        min_casing_seat_pressure = 870 * con.psi_to_Pa
        min_pressure_gradient = 0.30 * con.psi_to_Pa / con.feet_to_meters

        self.max_avg_pressure = max_casing_seat_pressure + max_pressure_gradient * (
            self.cavern_height/2 + 100 * con.feet_to_meters)
        self.min_avg_pressure = min_casing_seat_pressure + min_pressure_gradient * (
            self.cavern_height/2 + 100 * con.feet_to_meters)

        # axisymmetric model inputs
        axisym_elements_per_length = 4
        distance_to_insulated = 100
        distance_to_ground_temp = 600

        # salt characteristics
        salt_thermal_conductivity = 3 * con.btu_p_hrftF_2_W_p_m_K  #W/m-K
        salt_specific_heat = 0.2 * con.btu_p_lbF_2_J_p_kgK #J/kgK
        salt_density = 135 * con.lbs_ft3_2_kg_m3 # kg/m3

        surface_temperature = fahrenheit_to_kelvin(70)
        temp_increase_with_depth = (0.012
                                    / con.scale_factor_between_kelvin_and_rankin
                                    / con.feet_to_meters)
        avg_ground_temperature = (surface_temperature
                                  + temp_increase_with_depth
                                  * (self.cavern_top_depth
                                     + self.cavern_height / 2))

        # calculation inputs
        inp["calculation"]["time_step"] = con.tstep
        inp["calculation"]["type"] = "energybalance"

        # give direct control without an input file
        inp["vessel"]["length"] = self.cavern_height  #meters
        inp["vessel"]["diameter"] = self.cavern_diameter # meters
        inp["vessel"]["thickness"] = distance_to_insulated
        inp["vessel"]["density"] = salt_density  #kg/m3
        inp["vessel"]["heat_capacity"] = salt_specific_heat
        inp["vessel"]["orientation"] = "vertical"


        inp["valve"]["flow"] = "discharge"
        inp["valve"]["back_pressure"] = 8e6 # does not affect the calculation as configured
        inp["valve"]["type"] = "mdot"


        # This input has to be tricked into coming from the ground model.
        inp["heat_transfer"]["type"] = "salt_cavern"
        inp["heat_transfer"]["temp_ambient"] = avg_ground_temperature
        inp["heat_transfer"]["h_inner"] = "calc"  # we need to Figure out what this should be
        inp["heat_transfer"]["h_outer"] = 5  # NOT USED FOR SALT CAVERNS

        inp["reservoir"]["temperature"] = fahrenheit_to_kelvin(100)
        #inp["reservoir"]["pressure"] = 1e6


        # NOW SETUP INPUTS FOR THE CAVERN in a dictionary

        inpc = {}

        inpc["thermal_conductivity"] = salt_thermal_conductivity
        inpc["number_element"] = 4 #axisym_elements_per_length * distance_to_insulated
        inpc["distance_to_ground_temp"] = distance_to_ground_temp #m
        inpc["start_date"] = pd.date_range("1/1/2023",'1/1/2023',1)[0]


        """
        Variable inputs to allow 6 comparisons

        """
        # study parameters
        self.study_input = {'H2':{'days_per_cycle':[30,90,360],
                             'error':{30:{'max':0.7,
                                          'min':2.6,
                                          'mean':1.0},
                                      90:{'max':2.5,
                                          'min':0.1,
                                          'mean':1.3},
                                     360:{'max':2.4,
                                          'min':0.1,
                                          'mean':1.5}},
                             'file':{30:"2008_Nieland_Fig15_12_cycles_per_year.csv",
                                     90:"2008_Nieland_Fig15_4_cycles_per_year.csv",
                                    360:"2008_Nieland_Fig15_1_cycle_per_year.csv"},
                             'initial_temperatures':{30:326.5,
                                                     90:320.8,
                                                    360:318.0}
                             },
                       'Methane':{'days_per_cycle':[30,90,360],
                            'error':{30:{'max':1.0,
                                         'min':3.5,
                                         'mean':1.2},
                                     90:{'max':2.4,
                                         'min':1.1,
                                         'mean':0.7},
                                    360:{'max':2.4,
                                         'min':0.7,
                                         'mean':1.4}},
                            'file':{30:"2008_Nieland_Fig13_12_cycles_per_year.csv",
                                    90:"2008_Nieland_Fig13_4_cycles_per_year.csv",
                                   360:"2008_Nieland_Fig13_1_cycle_per_year.csv"},
                            'initial_temperatures':{30:330.8,
                                                    90:324.4,
                                                   360:319.8}
                            }
                       }
        self.inp = inp
        self.inpc = inpc

def cycle_flow_commands(sc):
    new_inp = sc.input
    cur_time = new_inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["time"]
    tstep = cur_time[1] - cur_time[0]
    cur_mdot = new_inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["mdot"][0]
    nstep = len(cur_time)


    if cur_mdot <= 0.0:

        new_inp["calculation"]["end_time"] = 2 * new_inp["calculation"]["end_time"]
        new_nstep = int(2 * nstep)
        new_mdot = -0.5 * cur_mdot

    elif cur_mdot > 0.0:

        new_inp["calculation"]["end_time"] = 0.5 * new_inp["calculation"]["end_time"]
        new_nstep = int(0.5 * nstep)
        new_mdot = -2.0 * cur_mdot

    else:
        raise ValueError("the inp['valve']['flow'] can only be = ['filling','discharge']")

    new_inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["time"] = [
        tstep*idx for idx in range(new_nstep+1)]
    new_inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["mdot"] = [
        new_mdot for idx in range(new_nstep+1)]

    sc.input = new_inp
    sc.validate_input()


class TestSaltCavernVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load constants for the Nielson paper.
        cls.nielson_obj = Nielson2008CavernCase()

        cls.print_figures = True
        cls.print_msg = True
        cls.run_all = True


    @classmethod
    def tearDownClass(cls):
        pass


    def test_verification_(self):

        """
        This test takes ~200-300 seconds and is a validation test that compares
        to Nielson's model results for a salt cavern. It is the case where
        the well heat transfer and dynamics are considered to be negligible.

        """

        if self.run_all:
            

            con = Constants()

            study_input = self.nielson_obj.study_input
            inp = self.nielson_obj.inp
            # inpc is no longer used.
            #inpc = self.nielson_obj.inpc

            inp_path = os.path.join(os.path.dirname(__file__),
               "test_data","nielson_verification.yml")

            #Loop over variable inputs
            with open(inp_path,'r',encoding='utf-8') as infile:
                inp = yaml.load(infile, Loader=yaml.FullLoader)

            for gas_type,subd in study_input.items():

                days_per_cycle_list = subd["days_per_cycle"]

                for days_per_cycle in days_per_cycle_list:

                    # THIS IS NO LONGER USED BUT I HAVE LEFT IT HERE 
                    # SO THAT YOU CAN SEE HOW I DETERMINED THE MASS RATE
                    # NEEDED.
                    #Prep inputs

                    # # establish simulation time parameters
                    # end_time = (1/3) * con.seconds_per_hour * con.hours_per_day * days_per_cycle
                    # nstep = end_time / con.tstep
                    # if np.floor(nstep) != nstep:
                    #     raise ValueError("You must make the cavern_time_step"
                    #                      +" an integer multiple of the tstep!")
                    # else:
                    #     nstep = int(nstep)
                    # inp["calculation"]["end_time"] = end_time

                    # read verification dataset from Nielson
                    filename = subd['file'][days_per_cycle]
                    verify_obj = prepare_csv_data(con.nonleapyear,os.path.join(
                        os.path.dirname(__file__),"test_data",filename))

                    # temp_max_pressure = fahrenheit_to_kelvin(verify_obj["degrees fahrenheit"].max())
                    # temp_min_pressure = fahrenheit_to_kelvin(verify_obj["degrees fahrenheit"].min())

                    # rho_max_pressure = PropsSI('D','T',temp_max_pressure,
                    #                            'P',self.nielson_obj.max_avg_pressure,gas_type)
                    # rho_min_pressure = PropsSI('D','T',temp_min_pressure,
                    #                            'P',self.nielson_obj.min_avg_pressure,gas_type)

                    # mass_max = rho_max_pressure * self.nielson_obj.cavern_volume
                    # mass_min = rho_min_pressure * self.nielson_obj.cavern_volume
                    
                    # breakpoint()
                    # # HERE IS WHERE I LEFT OFF. THIS routine is specifying mass flow in
                    # # in a way that should probably just be put in the input file
                    # # We want to move away from a lot of custom stuff happening 
                    # # here and to have correct values in the actual input file

                    # # calculate mass flow needed.
                    # mdot = -(mass_max - mass_min) / (end_time)
                    # # 30e6 Pa is approximately 1000 m deep overburden pressure
                    # inp["initial"]["pressure"] = self.nielson_obj.max_avg_pressure
                    # inp["initial"]["fluid"] = gas_type
                    # inp['initial']['temperature'] = subd['initial_temperatures'][days_per_cycle]

                    # # this just takes single steps
                    # inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["time"] = [
                    #     con.tstep*idx for idx in range(nstep+1)]
                    # inp["wells"]["cavern_well"]["valves"]["inflow_mdot"]["mdot"] = [
                    #     mdot for idx in range(nstep+1)]

                    # create model object
                    model = Model(inp)

                    # create salt caver object
                    # sc = SaltCavern(inp)
                    # """
                    # RUN
                    # """

                    for i in range(int(con.days_per_year/days_per_cycle)):
                        if self.print_msg:
                            print(i)
                        
                        model.run()
                        cycle_flow_commands(model)

                        model.run()
                        cycle_flow_commands(model)

                    sc = model.components["salt_cavern"]
                    v_df = create_df_from_sim_output(con.nonleapyear, sc.cavern_results)

                    s_kelvin = fahrenheit_to_kelvin(verify_obj['degrees fahrenheit'])
                    if self.print_figures:
                        fig,ax = plt.subplots(1,1,figsize=(20,10))
                        v_df['Gas temperature (K)'].plot(ax=ax,label="Gas")
                        v_df['Wall temperature (K)'].plot(ax=ax,label="Wall")

                        s_kelvin.plot(ax=ax,label="Gas Nielson",
                                      linestyle="None",marker="x",markersize=10)
                        ax.legend(["Gas","Wall","Gas Nielson"])
                        ax.grid("on")


                        fig.savefig('comparison'+ filename[:-3]+'png',dpi=300)

                    time_s_nielson = verify_obj['day of year'].values * con.second_in_day
                    y_k_nielson = s_kelvin.values
                    ybar_nielson = np.interp(time_s_nielson,v_df['Time (s)'].values,v_df[
                        'Gas temperature (K)'].values)
                    error = ybar_nielson - y_k_nielson
                    uerror = error[int(len(error)/2):]
                    max_error = np.abs(uerror.max())
                    min_error = np.abs(uerror.min())
                    mean_error = np.abs(uerror.mean())
                    msg = ("The verification match between Nielson 2008 and this"
                          +" model has degraded in comparison to the 12/8/2023 original check.")

                    #
                    if self.print_msg:
                        print("++++++++++++++++++++++++++++++++++")
                        print(gas_type + ":" + str(days_per_cycle) + ":\n\n")
                        print("max:" + str(max_error) + " < " + str(
                            subd['error'][days_per_cycle]['max']))
                        print("min:" + str(min_error) + " < " + str(
                            subd['error'][days_per_cycle]['min']))
                        print("mean:" + str(mean_error) + " < " + str(
                            subd['error'][days_per_cycle]['mean']))
                        print("------------------------------------------")
                    self.assertLess(max_error, subd['error'][days_per_cycle]['max'], msg)
                    self.assertLess(min_error, subd['error'][days_per_cycle]['min'], msg)
                    self.assertLess(mean_error, subd['error'][days_per_cycle]['mean'], msg)


if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestSaltCavernVerification())


    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('utilities_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())
