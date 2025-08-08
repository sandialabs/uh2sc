# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:37:16 2023

@author: dlvilla
"""
#pylint: disable=W0631
import unittest
import pde
import numpy as np
import os
import yaml


from matplotlib import pyplot as plt
from uh2sc.model import Model

from uh2sc.errors import InputFileError


class TestGHE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_ghe(self):
        """
        This test confirms that the salt_cavern radially symmetric
        solver gets similar answers to a PDE solution of the polar heat
        equation using finite difference elements. A problem that runs
        for 1 month on a daily time step gets a ravern wall temperature difference
        of 2K where the PDE solution heats up less because it uses a linear
        spacing whereas the axisym solver uses a log spacing for
        more accurate surface temperature.

        """
        if True:
                    # SETUP YOUR METHOD OF SOLUTION
            t_end = 2592000
            t_step_pde = 300
            t_step = 86400  # a daily time step is fine
            implicit_mult = 1  # maximal time step that only had a slight error
            # a 1 day time step is ok for the implicit solver. ( in comparison to
            # 600 seconds for the original explicity solver.)
            # 
    
            # salt characteristics
            salt_thermal_conductivity = 0.6 #W/m-K
            salt_specific_heat = 880 #J/kg-K
            salt_density = 2200 # kg/m3
    
            r_cavern = 10
            kg = salt_thermal_conductivity
            rhog = salt_density
            cpg = salt_specific_heat
            h_cavern = 100 #m
            number_element = 150
            number_pde_elem = 150
            dist_next_cavern = 40.5
            t_g = 323
            dist_to_tg_reservoir = 1e10 # make transfer to ground negligible
    
            q_in = 8000  # for polar coordinates this is W/m
    
            q_in_cavern = q_in * h_cavern
            bc_cavern = {"Q0":q_in_cavern, "Qend": 0.0}
    
    
            alpha = kg/(rhog * cpg)
    
    
            inp = {}
            inp["name"] = "test_ghe"
            inp["distance_to_farfield_temp"] = 1e10
            inp["density"] = 2200
            inp["farfield_temperature"] = 323
            inp["heat_capacity"] = 880
            inp["thermal_conductivity"] = 0.6
            inp["modeled_radial_thickness"] = 40.5
            inp["initial_conditions"] = {}
            inp["initial_conditions"]["Q0"] = q_in_cavern
            inp["initial_conditions"]["Qend"] = 0.0
            inp["number_elements"] = 150
    
            ghe_inp = {}
            ghe_inp["ghes"] = {}
            ghe_inp["ghes"]["ghe_test"] = inp
    
            model = Model(ghe_inp,single_component_test=True,
                           dt=t_step*implicit_mult,end_time=t_end,type="GHE",
                           inner_radius=r_cavern,Q0=q_in_cavern,Qend=0.0,
                           height=h_cavern)
            model.residual_normalization = {'cavern_gas_energy': 3591431272594.772, 
                                            'cavern_gas_mass': 726543.991369199, 
                                            'cavern_pressure': 20027490.643315244, 
                                            'temperature_norm': 110, 'heat_flux_norm': 20783745.790479004, 
                                            'mass_flow_norm': 4.204536987090272, 
                                            'brine_mass': 726543.991369199, 
                                            'brine_energy': 3591431272594.772}
    
            model.run()
    
            comparison_data = self.comparable_solution_by_different_method(alpha, q_in, kg, t_end,
                                                        r1=r_cavern,
                                                        r2=dist_next_cavern/2.0,
                                                        num_elem=number_pde_elem+1,
                                                        t_step=t_step_pde)
    
            r_step = (dist_next_cavern/2 - r_cavern)/(number_pde_elem+1)
            linear_grid = np.arange(r_cavern,dist_next_cavern/2,r_step)
    
            sol = model.solutions
            avg_error = {}
            max_error = {}
    
            idx = 0
            for tup in sol.items():
                crow = comparison_data[-1,:]
                time = tup[0]
                row = sol[time]
                interp_crow = np.interp(model.components["ghe_test"].grid,linear_grid,crow)
                avg_error[time] = (row[1:-1] - interp_crow).sum()/number_element
                max_error[time] = (row[1:-1] - interp_crow).max()
                idx+=1
    
            if len(linear_grid) > len(crow):
                linear_grid = linear_grid[:len(crow)-len(linear_grid)]
    
            fig, ax = plt.subplots(1,1)
            ax.plot(model.components["ghe_test"].grid,row[1:-1],linear_grid,crow)
            ax.legend(["uh2sc","symbolic pde solver"])
            ax.set_xlabel("Radial distance (m)")
            ax.set_ylabel("Temperature (K)")
            ax.set_title("Ground heat exchanger verification")
            plt.xlim(10,14)
            ax.grid('on')
            
    
    
            self.assertFalse(max_error[time] > 2.1) # Kelvin
            self.assertFalse(avg_error[time] > 0.75) # Kelvin
        

    def test_invalid_input_with_2_ghes(self):
        inp_path = os.path.join(os.path.dirname(__file__), "test_data", 
                                "nieland_verification_methane_1_cycles.yml")

        # Loop over variable inputs
        with open(inp_path, 'r', encoding='utf-8') as infile:
            inp = yaml.load(infile, Loader=yaml.FullLoader)
            
        inp["wells"]["my_big_well"] = inp["wells"]["cavern_well"]
        
        with self.assertRaises(InputFileError):
            model = Model(inp)


    def comparable_solution_by_different_method(self,alpha,q_in,kg,t_end,r1,r2,num_elem,t_step):

        grid = pde.PolarSymGrid((r1,r2),num_elem)
        eq = pde.PDE({'T':f"{alpha:10.6e}*(d2_dr2(T) + d_dr(T)/r)"},
                     bc={'r-':{'derivative':q_in/(kg*2*np.pi*r1)},
                         'r+':{"derivative":0.0}})

        state = pde.ScalarField.from_expression(grid,"323")
        storage = pde.MemoryStorage()

        eq.solve(state, t_range=t_end, dt=100, tracker=storage.tracker(t_step))

        #pde.plot_kymograph(storage)
        return np.array(storage.data)


if __name__ == "__main__":
    PROFILE = False

    if PROFILE:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestGHE())

    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('utilities_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())
