# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 2023 3:34 PM CST

This test takes ~250 s to execute and is longer than all of the rest.

@author: dlvilla
"""

import os
from datetime import datetime, timedelta
import unittest
from copy import deepcopy

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
              for second_of_year in df["Time (sec)"]]

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
    tstep = seconds_per_hour * 6 # 2.5 min time step
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

        # CAVERN GEOMETRY
        self.cavern_height = 1000 * con.feet_to_meters
        self.cavern_volume = 5.61e6 * con.feet_to_meters **3
        self.cavern_diameter = np.sqrt(self.cavern_volume / (self.cavern_height * np.pi * 0.25))
        self.cavern_top_depth = 3000 * con.feet_to_meters


        # values from Nieland2008
        max_casing_seat_pressure = 2465 * con.psi_to_Pa
        max_pressure_gradient = 0.85 * con.psi_to_Pa / con.feet_to_meters
        min_casing_seat_pressure = 870 * con.psi_to_Pa
        min_pressure_gradient = 0.30 * con.psi_to_Pa / con.feet_to_meters

        self.max_avg_pressure = max_casing_seat_pressure + max_pressure_gradient * (
            self.cavern_height/2 + 100 * con.feet_to_meters)
        self.min_avg_pressure = min_casing_seat_pressure + min_pressure_gradient * (
            self.cavern_height/2 + 100 * con.feet_to_meters)


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


def cycle_flow_commands(model):

    inp = model.inputs
    new_inp = deepcopy(inp)
    
    model_time = model.time

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

    return new_inp


class TestSaltCavernVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load constants for the Nielson paper.
        cls.nielson_obj = Nielson2008CavernCase()

        # one parameter that moves this to a long run -time 
        # verification case with plots.
        cls.run_verification = True

    @classmethod
    def tearDownClass(cls):
        pass


    def test_verification(self):

        """
        This test takes ~200-300 seconds and is a validation test that compares
        to Nielson's model results for a salt cavern. It is the case where
        the well heat transfer and dynamics are considered to be negligible.

        """

        con = Constants()

        study_input = self.nielson_obj.study_input

        inp_path = os.path.join(os.path.dirname(__file__),
           "test_data","nielson_verification.yml")

        #Loop over variable inputs
        with open(inp_path,'r',encoding='utf-8') as infile:
            inp = yaml.load(infile, Loader=yaml.FullLoader)
            
        if not self.run_verification:
            # reduce to a single gas.
            study_input = {"H2":study_input.pop("H2")}
            
            
        for gas_type,subd in study_input.items():
            
            if self.run_verification:
               days_per_cycle_list = subd["days_per_cycle"]
            else:
                # cut it short
               days_per_cycle_list = [subd["days_per_cycle"][0]]

            for days_per_cycle in days_per_cycle_list:

                # THIS IS NO LONGER USED BUT I HAVE LEFT IT HERE 
                # SO THAT YOU CAN SEE HOW I DETERMINED THE MASS RATE
                # NEEDED.
                #Prep inputs

                # establish simulation time parameters
                end_time = (1/3) * con.seconds_per_hour * con.hours_per_day * days_per_cycle

                # read verification dataset from Nielson
                filename = subd['file'][days_per_cycle]
                verify_obj = prepare_csv_data(con.nonleapyear,os.path.join(
                    os.path.dirname(__file__),"test_data",filename))

                temp_max_pressure = fahrenheit_to_kelvin(
                    verify_obj["degrees fahrenheit"].max())
                temp_min_pressure = fahrenheit_to_kelvin(
                    verify_obj["degrees fahrenheit"].min())

                rho_max_pressure = PropsSI('D','T',temp_max_pressure,
                                           'P',self.nielson_obj.max_avg_pressure,gas_type)
                rho_min_pressure = PropsSI('D','T',temp_min_pressure,
                                           'P',self.nielson_obj.min_avg_pressure,gas_type)

                mass_max = rho_max_pressure * self.nielson_obj.cavern_volume
                mass_min = rho_min_pressure * self.nielson_obj.cavern_volume
                
                # # calculate mass flow needed.
                mdot = -(mass_max - mass_min) / (end_time)
                # 30e6 Pa is approximately 1000 m deep overburden pressure
                inp["initial"]["pressure"] = self.nielson_obj.max_avg_pressure
                inp["initial"]["fluid"] = gas_type
                inp['initial']['temperature'] = subd['initial_temperatures'][days_per_cycle]
                
                if not self.run_verification:
                    # just run for 2 days and stop
                    inp['calculation']['end_time'] = 2 * 24 * 3600

                # create model object
                model = Model(inp,solver_options={"TOL":1.0e-2})
                
                if self.run_verification:
                    model.components['cavern'].troubleshooting = True

                # """
                # RUN
                # """
                model.run()

                cav_res = model.components['cavern'].results
                
                plt.plot(cav_res['Time (sec)'],cav_res['Cavern energy (J)'],
                         cav_res['Time (sec)'],cav_res['Brine energy (J)'])

                model.plot_solution(model.xg_descriptions)

                sc = model.components["cavern"]
                v_df = create_df_from_sim_output(con.nonleapyear, sc.results)

                s_kelvin = fahrenheit_to_kelvin(verify_obj['degrees fahrenheit'])
                if self.run_verification:
                    fig,ax = plt.subplots(1,1,figsize=(20,10))
                    v_df['Cavern temperature (K)'].plot(ax=ax,label="Gas")
                    v_df['Cavern wall temperature (K)'].plot(ax=ax,label="Wall")

                    s_kelvin.plot(ax=ax,label="Gas Nielson",
                                  linestyle="None",marker="x",markersize=10)
                    ax.legend(["Gas","Wall","Gas Nielson"])
                    ax.grid("on")


                    fig.savefig('comparison'+ filename[:-3]+'png',dpi=300)

                time_s_nielson = verify_obj['day of year'].values * con.second_in_day
                y_k_nielson = s_kelvin.values
                time_s_nielson_cutoff = time_s_nielson[time_s_nielson <= inp['calculation']['end_time']]
                
                ybar_nielson = np.interp(time_s_nielson_cutoff,v_df['Time (sec)'].values,v_df[
                    'Cavern temperature (K)'].values)
                
                
                error = ybar_nielson - y_k_nielson[0:len(ybar_nielson)]
                uerror = error[int(len(error)/2):]
                error_extremes = np.array([np.abs(uerror.max()), np.abs(uerror.min())])
                max_error = np.max(error_extremes)
                min_error = np.min(error_extremes)
                mean_error = np.abs(uerror.mean())

                #
                if self.run_verification:
                    msg = ("The verification match between Nielson 2008 and this"
                          +" model has degraded in comparison to the 12/8/2023 original check.")
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
                else:
                    # don't let the error shift without 
                    max_error_threshold = 7.24
                    min_error_threshold = 6.09
                    mean_error_bounds = [6.6,6.7]
                    if (max_error < max_error_threshold 
                        or min_error > min_error_threshold
                        or (mean_error > mean_error_bounds[0]
                        and mean_error < mean_error_bounds[1])):
                        print("The solution has shifted since unit testing was"
                              +" created 7-24-2025. You need to run the full "
                              +"verification again by setting "
                              +"self.run_verification=True in test_verification.py."
                              +" If the graphical comparisons are good enough by"
                              +" your judgement, you can change the unit test"
                              +" comparison thresholds")
                    self.assertTrue(max_error < max_error_threshold)
                    self.assertTrue(min_error > min_error_threshold)
                    self.assertTrue(mean_error < mean_error_bounds[1] 
                                    and mean_error > mean_error_bounds[0])



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
