# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:37:16 2023

@author: dlvilla
"""

import pde
import numpy as np
import unittest

from matplotlib import pyplot as plt

from uh2sc.hdclass import ExplicitAxisymmetricRadialHeatTransfer

class Test_GHE(unittest.TestCase):
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
        
        # SETUP YOUR METHOD OF SOLUTION
        t_end = 2592000
        t_step = 600  # a daily time step is fine
        
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
        Tg = 323
        dist_to_Tg_reservoir = 1e10 # make transfer to ground negligible
        save_results=False
        
        Qin = 8000  # for polar coordinates this is W/m
        
        Qin_cavern = Qin * h_cavern
        
        alpha = kg/(rhog * cpg)
        
        
        axsym = ExplicitAxisymmetricRadialHeatTransfer(r_cavern,kg,rhog,cpg,h_cavern,number_element,dist_next_cavern,Tg,
                     dist_to_Tg_reservoir)
        
        for step in range(int(t_end/t_step)):
        
            Tg_t = axsym.step(dt=t_step, Qsalt0=Qin_cavern)
            
            
        
        
        
        comparison_data = self.comparable_solution_by_different_method(alpha, Qin, kg, t_end, 
                                                    r1=r_cavern, 
                                                    r2=dist_next_cavern/2.0, 
                                                    num_elem=number_pde_elem+1,
                                                    t_step=t_step)
        
        r_step = (dist_next_cavern/2 - r_cavern)/(number_pde_elem+1)
        linear_grid = np.arange(r_cavern,dist_next_cavern/2,r_step)
        
        sol = axsym.solutions
        avg_error = {}
        max_error = {}
        for tup,crow in zip(axsym.solutions,comparison_data):
            row = sol[tup]['Tg']
            interp_crow = np.interp(axsym.grid,linear_grid,crow)
            avg_error[tup] = (row - interp_crow).sum()/number_element
            max_error[tup] = (row - interp_crow).max()

        if len(linear_grid) > len(crow):
            linear_grid = linear_grid[:len(crow)-len(linear_grid)]
        
        plt.plot(axsym.grid,row,linear_grid,crow)
        plt.legend(["mine","pde"])


        self.assertFalse(max_error[tup] > 2.1) # Kelvin
        self.assertFalse(avg_error[tup] > 0.75) # Kelvin
        



        # GET THE COMPARABLE SOLUTION

    def comparable_solution_by_different_method(self,alpha,Qin,kg,t_end,r1,r2,num_elem,t_step):
    
        
        grid = pde.PolarSymGrid((r1,r2),num_elem)
        eq = pde.PDE({'T':"{0:10.6e}*(d2_dr2(T) + d_dr(T)/r)".format(alpha)},
                     bc={'r-':{'derivative':Qin/(kg*2*np.pi*r1)},
                         'r+':{"derivative":0.0}})
        
        state = pde.ScalarField.from_expression(grid,"323")
        storage = pde.MemoryStorage()
        
        eq.solve(state, t_range=t_end, dt=100, tracker=storage.tracker(t_step))

        #pde.plot_kymograph(storage)
        return np.array(storage.data)
        
    
if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_GHE())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('utilities_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())          
