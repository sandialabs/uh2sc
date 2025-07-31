# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 2023 3:34 PM CST

This test takes ~250 s to execute and is longer than all of the rest.

@author: dlvilla
"""

import os
from datetime import datetime, timedelta
import unittest
import logging

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import yaml

from uh2sc.model import Model
from uh2sc.errors import CavernStateOutOfOperationalBounds
from joblib import Parallel, delayed

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
    days_per_year = 360  # per nieland

def run_gas_type(gas_type, subd, run_verification, nieland_obj, con):
    if run_verification:
        days_per_cycle_list = subd["days_per_cycle"]
    else:
        days_per_cycle_list = [subd["days_per_cycle"][0]]

    for days_per_cycle in days_per_cycle_list:
        # Prep inputs
        inp_path = os.path.join(os.path.dirname(__file__), "test_data", subd["inputs"][days_per_cycle])

        # Loop over variable inputs
        with open(inp_path, 'r', encoding='utf-8') as infile:
            inp = yaml.load(infile, Loader=yaml.FullLoader)

        # establish simulation time parameters
        end_time = (1/3) * con.seconds_per_hour * con.hours_per_day * days_per_cycle

        # read verification dataset from nieland
        filename = subd['file'][days_per_cycle]
        verify_obj = prepare_csv_data(con.nonleapyear, os.path.join(os.path.dirname(__file__), "test_data", filename))

        if not run_verification:
            # just run for 2 days and stop   TODO
            inp['calculation']['end_time'] = 65 * 24 * 3600

        # create model object
        model = Model(inp, solver_options={"TOL": 1.0e-2})

        if run_verification:
            model.components['cavern'].troubleshooting = True

        """
        RUN
        """
        model.run()

        cav_res = model.components['cavern'].results

        plt.plot(cav_res['Time (sec)'], cav_res['Cavern energy (J)'],
                 cav_res['Time (sec)'], cav_res['Brine energy (J)'])

        model.plot_solution(model.xg_descriptions)

        sc = model.components["cavern"]
        v_df = create_df_from_sim_output(con.nonleapyear, sc.results)

        
        s_kelvin = fahrenheit_to_kelvin(verify_obj['degrees fahrenheit'])
        #TODO
        if True: #run_verification:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            v_df['Cavern temperature (K)'].plot(ax=ax, label="Gas")
            v_df['Cavern wall temperature (K)'].plot(ax=ax, label="Wall")

            s_kelvin.plot(ax=ax, label="Gas nieland", linestyle="None", marker="x", markersize=10)
            ax.legend(["Gas", "Wall", "Gas nieland"])
            ax.grid("on")

            fig.savefig('comparison' + filename[:-3] + 'png', dpi=300)

        time_s_nieland = verify_obj['day of year'].values * con.second_in_day
        y_k_nieland = s_kelvin.values
        time_s_nieland_cutoff = time_s_nieland[time_s_nieland <= inp['calculation']['end_time']]

        ybar_nieland = np.interp(time_s_nieland_cutoff, v_df['Time (sec)'].values, v_df['Cavern temperature (K)'].values)

        error = ybar_nieland - y_k_nieland[0:len(ybar_nieland)]
        uerror = error[int(len(error) / 2):]
        error_extremes = np.array([np.abs(uerror.max()), np.abs(uerror.min())])
        max_error = np.max(error_extremes)
        min_error = np.min(error_extremes)
        mean_error = np.abs(uerror.mean())
        
        model.write_results(filename[:-3]+"_results.csv")

        if run_verification:
            
            with open(f"{gas_type}_{days_per_cycle}_output.txt", "w") as f:
                f.write(f"{gas_type} {days_per_cycle}:\n\n")
                f.write(f"max: {max_error} < {subd['error'][days_per_cycle]['max']}\n")
                f.write(f"min: {min_error} < {subd['error'][days_per_cycle]['min']}\n")
                f.write(f"mean: {mean_error} < {subd['error'][days_per_cycle]['mean']}\n")
                f.write("------------------------------------------\n")
            
            # test_obj.assertLess(max_error, subd['error'][days_per_cycle]['max'], msg)
            # test_obj.assertLess(min_error, subd['error'][days_per_cycle]['min'], msg)
            # test_obj.assertLess(mean_error, subd['error'][days_per_cycle]['mean'], msg)
        
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
            breakpoint()
            
            #assert (max_error < max_error_threshold)
            #assert (min_error > min_error_threshold)
            #assert (mean_error < mean_error_bounds[1] 
            #                and mean_error > mean_error_bounds[0])


class Nieland2008CavernCase(object):

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
                                                    360:318.0},
                             'inputs':{30:"nieland_verification_h2_12_cycles.yml",
                                       90:"nieland_verification_h2_4_cycles.yml",
                                      360:"nieland_verification_h2_1_cycles.yml"}
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
                                                   360:319.8},
                            'inputs':{30:"nieland_verification_methane_12_cycles.yml",
                                      90:"nieland_verification_methane_4_cycles.yml",
                                     360:"nieland_verification_methane_1_cycles.yml"}
                            }
                       }


class TestSaltCavernVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load constants for the nieland paper.
        cls.nieland_obj = Nieland2008CavernCase()

        # one parameter that moves this to a long run -time 
        # verification case with plots.
        cls.run_verification = True
        
        cls.run_parallel = True
        
        cls.filedir = os.path.dirname(__file__)
        
        cls.run_all = False

    @classmethod
    def tearDownClass(cls):
        pass



    def test_nieland_verification(self):
        """
        This test takes ~200-300 seconds and is a validation test that compares
        to nieland's model results for a salt cavern. It is the case where
        the well heat transfer and dynamics are considered to be negligible.
        
        It compares the uh2sc model to a simpler model. 
        
        The comparison is of limited value though because uh2sc has a more
        complex heat transfer mechanism at the cavern wall. It was found that
        temperature values (and the shape of the trajectory) are extremely
        sensitive to the number of element in the GHE modeled. As a result,
        this case should not serve as a measure of whether uh2sc is an accurate
        model because Nieland's results are purely model-based.

        """
        if self.run_all:
            con = Constants()
    
            study_input = self.nieland_obj.study_input
    
            if not self.run_verification:
                # reduce to a single gas.
                study_input = {"H2": study_input.pop("H2")}
    
            if self.run_parallel:
                print("Launching parallel jobs!")
                Parallel(n_jobs=-1)(delayed(run_gas_type)(gas_type, 
                                                          subd, 
                                                          self.run_verification, 
                                                          self.nieland_obj, 
                                                          con) for gas_type, subd in study_input.items())
            else:
                for gas_type, subd in study_input.items():
                    run_gas_type(gas_type, subd, self.run_verification, self.nieland_obj, con)
                
                
    def test_gas_mixture_mass_balance(self):
        """
        Verify that gas mixtures conserve mass correctly and that you can
        add one gas to another. In the input file without errors.
        
        The limitations of CoolProp must be considered.
        
        Here we take a 0.9 Methane, 0.1 Ethane mixture at 320 K and 8 MPa
        
        and Add 1e6 kg of pure Ethane
        
        
        """
        
        if self.run_all:
            
            infile = os.path.join(self.filedir,"test_data","gas_mixture_test_too_fast_of_flow.yml")
            with open(infile, 'r', encoding='utf-8') as infile:
                inp = yaml.load(infile, Loader=yaml.FullLoader)
        

            # Turn off UserWarning
            logging.getLogger().setLevel(logging.ERROR)

            # test analytics errors
            with self.assertRaises(CavernStateOutOfOperationalBounds):
                model = Model(inp)            
                model.run()            
    
            # temperature
            inp['initial']['temperature'] = 380
            with self.assertRaises(CavernStateOutOfOperationalBounds):
                model = Model(inp)            
                model.run()     
                
            # pressure
            inp['initial']['temperature'] = 313.15
            inp['initial']['pressure'] = 7e6
            with self.assertRaises(CavernStateOutOfOperationalBounds):
                model = Model(inp)            
                model.run()    
            
            # Turn UserWarning back on
            logging.getLogger().setLevel(logging.WARNING)
            
            # now run the actual case and cross zero over the time so 
            # that the total added mass equals the total taken away
            # this should prove that conservation of mass is working.
            upper_limit_flow = 0.1620941707317945
            
            inp['wells']['cavern_well']['valves']['inflow_mdot']['mdot'] = (
                [-upper_limit_flow * 0.95,
                 upper_limit_flow * 0.95])
            inp['initial']['pressure'] = 5e6
            
            model = Model(inp)
            
            model.run()

            mass_ethane0 = 34678.648 #kg
            mass_methane0 = 312107.834 #kg
            mass_change_in_one_half_day = 3326.172383 * 2 #kg
            # only ehtane is coming in
            
            mass_ethane1 = mass_ethane0 + mass_change_in_one_half_day
            
            # at 1/2 day the flow reverses and gradually rises to the original
            # magnitude but in the opposite direction. The same amount of mass
            # in is mass out.
            new_mass = (mass_ethane1 + mass_methane0)
            new_mass_fraction_ethane = mass_ethane1 / new_mass
            new_mass_fraction_methane = mass_methane0 / new_mass
            
            mass_methane2 = mass_methane0 - mass_change_in_one_half_day * new_mass_fraction_methane
            mass_ethane2 = mass_ethane1 - mass_change_in_one_half_day * new_mass_fraction_ethane
            
            self.assertTrue(np.abs(mass_ethane2 - 
                                   model.components['cavern']._m_cavern[0]) 
                                   < 200)
            self.assertTrue(np.abs(mass_methane2 - 
                                   model.components['cavern']._m_cavern[1]) 
                                   < 200)
            
            

    
    def test_hanEtAl_air_with_actual_test_data(self):
        
        if True: #self.run_all:
        
            #load input file
            infile = os.path.join(self.filedir,"test_data","HanEtAl_2022_24hr_ops_validation_air.yml")
            #with open(infile, 'r', encoding='utf-8') as infile:
            #    inp = yaml.load(infile, Loader=yaml.FullLoader)
                
            model = Model(infile)
            
            model.run()
            
            df = model.dataframe
            
            breakpoint()
            pass

        


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
