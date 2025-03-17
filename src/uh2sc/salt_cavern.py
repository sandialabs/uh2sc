# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:20:21 2023

@author: dlvilla
"""
import pandas as pd
import yaml

from .abstract import AbstractComponent
from .hdclass import HydDown


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

    """

    _result_name_map = {"time_array":"Time (s)",
                    "T_cavern_wall":"Wall temperature (K)",
                    "T_cavern":"Gas temperature (K)",
                    "mass_fluid":"Total gas mass (kg)",
                    "total_mass_rate":"Mass Flux (+ into cavern) (kg/s)",
                    "P_cavern":"Gas pressure (Pa)",
                    "Q_inner":"Cavern wall heat flux (W)"}

    def __str__(self):
        return "Salt Cavern Object"

    def __init__(self,inp):

        # enable reading a file or accepting a valid dictionary
        if isinstance(inp,str):
            with open(inp,'r',encoding='utf-8') as infile:
                input_dict = yaml.load(infile, Loader=yaml.FullLoader)
        elif isinstance(inp,dict):
            input_dict = inp
        else:
            raise TypeError("The input object 'inp' must be a string that "+
                            "gives a path to a uh2sc yaml input file or must"+
                            " be a dictionary.")

        # this includes validation of the input dictionary which must follow
        # the main schema in validatory.py
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
