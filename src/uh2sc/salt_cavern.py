# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:20:21 2023

@author: dlvilla
"""
import pandas as pd
import yaml

from uh2sc.abstract import AbstractComponent
from uh2sc.hdclass import HydDown


class SaltCavern(AbstractComponent, HydDown):

    """
    This uses a much abridged version of HyDown functionality and adds radial
    transient transport through salt. The outer boundary condition
    is perfectly insulated - this would be realistic for a salt cavern
    that is surrounded by other salt caverns that are being similarly cycled.

    It also strings together Hydown and ExplicitAxisymmetricRadialHeatTransfer
    results into a single set of results and initial conditions so that it is
    straightforward to cycle various conditions for the input such as mass flow
    or

    TODO - make cavern size a function of pressure (elastic strain) and creep
           (plastic strain).

    Variables:
    ==========
    
    t_gas = The equilibrium gas temperature of a continuously varying
               mixture of gasses.
    t_gas_wall = The temperature at the salt cavern wall (connected to the
               axisymmetric heat transfer)
    p_gas = the average gas pressure in the cavern 
    
    mass_gas = The total mass of gas in the cavern
    
    molefrac[ngas] = The molefractions of the gas in the cavern (gas species include 
               all types included in the input file)
    
    mass_flux_gas[nwell][ngas] = for each well, the gas flux of each gas species 
               in kg/s (wells below liquid can only pull and push liquid)
    
    t_cavern_liquid = The equilibrium liquid temperature at the bottom of the 
               salt cavern
    t_cavern_liquid_wall = The equilibrium salt wall temperature at the bottom 
               of the salt cavern
               
    NOTE: for now, the fluid salinity is the only attribute that is allowed to
          vary. The size of the cavern doesn't change (i.e., solution mining
          is not modeled)
    
    mass_liquid - the mass of the fluid present in the cavern
    
    molefrac_liquid[2] - the molefractions of fluid contituents (water, salt in solution)
    
    mass_flux_liquid[nwell][ngas] = for each well, the liquid flux of each liquid species
              in kg/s (wells above liquid cna only pull and push gas)
              
    Equations
    
    Aximsymmetric heat transfer (1) - flux from cavern = flux into ground
    conservation of energy ()
    conservation of mass for each gas species (ngas)
    conservation of liquid mass (nliquid)
    conservation of liquid energy
    conservation of cavern volume (i.e. liquid interface height)
    
    residual[0] = Q_axisym - Q_inner
    residual[1] = -Cg * dT/dt + sum(mdot_in cp_in t_in, wells) - sum (mdot_out cp_out, t_out, wells)
            + Q_inner + Q_liquid
    residual[2..ngas+2] = -dm/dt + mdot_in - mdot_out
    residual[ngas+2...ngas+2+nliquid] = 
            
    dEliquid/dt =         
    
    How to handle
    
    1) Pressure of cavern exceeds well reservoir pressure - flow can no longer
    happen - a check valve is assumed and pressure in the well reaches the 
    cavern pressure minus bouyancy effects up the well.
    
    
    
    
    

    
    

    """

    _result_name_map = {"time_array":"Time (s)",
                    "T_cavern_wall":"Wall temperature (K)",
                    "T_cavern":"Gas temperature (K)",
                    "mass_fluid":"Total gas mass (kg)",
                    "total_mass_rate":"Mass Flux (+ into cavern) (kg/s)",
                    "P_cavern":"Gas pressure (Pa)",
                    "Q_inner":"Cavern wall heat flux (W)",
                    "mole_frac":"Gas mole fractions for gas in cavern",
                    "mole_frac_in":"Gas mole fractions for gas coming in"}

    def __str__(self):
        return "Salt Cavern Object"

    def __init__(self,input_dict,global_indices):
        # this includes validation of the input dictionary which must follow
        # the main schema in validatory.py
        # do stuff relevant to model
        self._gindices = global_indices
        
        super().__init__(input_dict)

        self.first_step = True
        self.step_num = 0.0

        self.cavern_results = {val: [] for val in self._result_name_map.values()}


    def _process_new_input(self,new_input):
        """
        This doesn't guarantee proper control of the HydDown sub-object but
        it does eliminate many very hard to understand input errors.

        """
        if not new_input is None:

            for key,val in new_input:
                if key in self.input:
                    self.input[key] = new_input[key]
                else:
                    raise KeyError("The key '{0}' is not a valid input to HydDown".format(key))

            # revalidate the input.
            self.validate_input()



    def step(self,new_input=None):

        self._process_new_input(new_input)

        # run HydDown object with heat_transfer = salt_cavern
        self.run()

        # Trick HydDown to
        # transfer the new state so that it sticks for the next time step
        self.m0 = self.mass_fluid[-1]
        self.p0 = self.P_cavern[-1]
        self.T0 = self.T_cavern[-1]
        self.Tv0 = self.T_cavern_wall[-1]
        self.rho0 = self.rho[-1]

        # Store the new results into a concatenated set of lists so that self.initialize
        # will not erase previous results
        for key,val in self._result_name_map.items():

            cav_res = self.cavern_results[val]
            if key == "time_array" and not self.first_step:
                # time must grow from previous steps
                # TODO HydDown places a zero time value at the end of every array
                # you must get rid of this.
                res = [cav_res[-1]+elem for elem in getattr(self,key)]
            else:
                res = getattr(self,key)

            # numpy array concatenation is really slow. Eventually HydDown should
            # be converted to storing results in lists.
            self.cavern_results[val] = self.cavern_results[val] + list(res)


        if self.first_step:
            self.first_step = False
        self.step_num += 1

        return self.T_cavern_wall[-1], self.T_cavern[-1]
    
    
    def output_df(self):
        return pd.DataFrame(self.cavern_results)


    # METHODS NEEDING DEFINITIONS FROM THE AbstractComponent super-class
    @property
    def global_indices(self):
        """
        This property must give the indices that give the begin and end location
        in the global variable vector (xg)
        """
        return self._gindices

    @property
    def previous_adjacent_components(self):
        """
        Interface variable indices for the previous component
        """
        pass

    @property
    def next_adjacent_components(self):
        """
        interface variable indices for the next component
        """
        
    @property
    def component_type(self):
        """
        A string that allows the user to identify what kind of component 
        this is so that specific properties and methods can be invoked

        """


    def evaluate_residuals(self,x=None):
        """
        Must first evaluate all interface equations for indices produced by interface_var_ind_prev_comp

        Then must evaluate all internal component equations
        
        Equations for the salt cavern
        
        Energy flow


        Args:
            x numpy.array : global x vector for the entire
                             differiental/algebraic system
                             
                
        This is a dynamic ODE implicit solution via euler integration




        Inputs:
        =======
            Variables:
            Local Index 0 = Q0 - flux into the system
            Local Inices 1 to num_element + 2 Tgvec:
                vector of ground temperatures spaced radially from the GSHX
            Local Index Qend - flux out of the system (ussually fixed to 0.0)

        Parameters:

            Tgvec_m1 - Tgvec for the previous time step

        Returns
        =======
            residuals - error level from solving the implicit equations
                        np.array of length number_elements + 1
                
        """
        
        
        breakpoint()


    def get_x(self):
        pass


    def load_var_values_from_x(self,xg):
        pass


