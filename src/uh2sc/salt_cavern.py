# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:20:21 2023

@author: dlvilla
"""
import pandas as pd
import numpy as np
import yaml

from CoolProp import CoolProp as CP

from uh2sc.abstract import AbstractComponent
from uh2sc.hdclass import HydDown
from uh2sc.utilities import (reservoir_mass_flows, 
                             calculate_component_masses,
                             brine_average_pressure)
from uh2sc.constants import Constants
from uh2sc.thermodynamics import (density_of_brine_water, 
                                  brine_saturated_pressure, 
                                  solubility_of_nacl_in_h2o)
from uh2sc.hdclass import ImplicitEulerAxisymmetricRadialHeatTransfer
from uh2sc.well import Well
from uh2sc.transport import natural_convection_nu

const = Constants()



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
    
    molefrac[ngas] = The vapor molefractions of the gas in the cavern (gas species include 
               all types included in the input file)
    
    liquid_molefrac[ngas] = the liquid molefractions of the liquid in the cavern ...
    
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

    def __init__(self,input_dict,global_indices,model):
        # this includes validation of the input dictionary which must follow
        # the main schema in validatory.py
        # do stuff relevant to model
        T0 = input_dict['initial']['temperature']
        T0brine = input_dict['initial']['liquid_temperature']
        p0 = input_dict['initial']['pressure']
        
        #pure water for brine calculations
        water = CP.AbstractState("HEOS","Water")
        water.update(CP.PT_INPUTS,p0,T0)
        water.set_mole_fractions([1.0])
        self._water = water
        
        # time and indices
        self._time = 0
        self._dt = input_dict['calculation']['time_step']
        self._first_step = True
        self._step_num = 0.0
        self._gindices = global_indices
        self._fluid = model.fluids['cavern']
        self._fluid.update(CP.PT_INPUTS, input_dict['initial']['pressure'],
                                         input_dict['initial']['temperature'])
        self._number_fluids = len(model.fluids['cavern'].fluid_names())
        
        #initial geometry
        self._area_horizontal = np.pi * input_dict['cavern']['diameter']**2/4
        self._area_vertical = np.pi * input_dict['cavern']['diameter'] * input_dict["cavern"]["height"]
        self._height_total = input_dict["cavern"]["height"]
        self._height_brine = input_dict['initial']['liquid_height']
        self._vol_brine = self._height_brine * self._area_horizontal
        self._vol_cavern = (self._height_total - self._height_brine) * self._area_horizontal
        
        # temperature and pressure states
        self._t_cavern_m1 = T0
        self._p_cavern_m1 = p0
        self._t_cavern_wall_m1 = T0
        self._t_brine_m1 = T0brine
        self._t_brine_wall_m1 = T0brine
        self._t_cavern = T0
        self._p_cavern = p0
        self._t_cavern_wall = T0
        self._t_brine = T0brine
        self._t_brine_wall = T0brine
        (self._p_brine, solubility_brine,
         rho_brine) = brine_average_pressure(self._fluid,self._water, 
                                             self._height_total, 
                                             self._height_brine, self._t_brine)
        

        self._m_brine = self._vol_brine * rho_brine
        self._m_brine_m1 = self._m_brine
        
        self._m_cavern = calculate_component_masses(self._fluid,
                                model.molar_masses,
                                self._fluid.rhomass() * self._vol_cavern,
                                liquid_mass=0.0)['gas']
        self._m_cavern_m1 = self._m_cavern
        
        # heat transfer coefficients
        self._ht_coef = input_dict['heat_transfer']['h_inner']
        
        
        self._q_cavern_wall = 0.0 #starting with t_cavern_wall = t_liquid_wall
        self._q_brine_wall = 0.0
        
        # solution
        self._NUM_EQN = self._number_fluids + 5 # see get_x for the 5 first variables.


        self._model = model

        self.cavern_results = {val: [] for val in self._result_name_map.values()}


    def _brine_average_pressure(self):
        """
        Approximate the average pressure for the brine
        
        """
        # TODO: create a test for this!
        pres_g = self._fluid.p()
        rho_g = self._fluid.rhomass()
        # assume density is constant
        height_gas_o_2 = (self._height_total - self._height_brine)/2
        pres_g_surf = pres_g + rho_g * const.g['value'] * height_gas_o_2
        rho_pure_water = self._water.rhomass()
        height_brine_o_2 = self._height_brine / 2
        solubility_brine = solubility_of_nacl_in_h2o(self._t_brine)
        rho_brine = density_of_brine_water(self._t_brine, pres_g_surf, solubility_brine, rho_pure_water)
                
        return (pres_g_surf + height_brine_o_2 * const.g['value'] * rho_brine,
               solubility_brine,
               rho_brine)
        
        


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
        prev_comp = {}
        for wname, well in self._prev_components:
            prev_comp[wname] = self._model.components[wname]
        return prev_comp
            

    @property
    def next_adjacent_components(self):
        """
        interface variable indices for the next component which can only be 
        1 GHE for a cavern (even though this is written as a loop)
        """
        next_comp = {}
        for ghe_name, ghe in self._next_components.items():
            next_comp[ghe_name] = self._model.components[ghe_name]
        return next_comp
        
        
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
        
        # The equations are
        breakpoint()
        # THIS IS WHERE YOU STOPPED, BELOW YOU HAVE AN INCOMPLETE AND VERY
        # LONG SET OF EQUATIONS TO MATCH AND CALCULATE RESIDUALS FOR
        # MASS IS COMPLETE, GHE WALL FLUX IS COMPLETE, BRINE MASS IS COMPLETE
        # BUT DOES NOT HAVE ANY BRINE WELL (A FEATURE I AM GOING TO WAIT ON
        # ENERGY AND THE TEMPERATURE TIE TO GHE ARE NOT FINISHED.)
        
        # number_fluids - mass balance of each species
        # 1. wall heat flux match to GHE(s)
        # 1. wall temperature tie to GHE first temperature
        # 1. energy balance for cavern
        # 1. energy balance for brine
        # 1. mass balance for brine

        
        # IMPORTANT! - "m1" indicates a variable's value in the previous time 
        #              step. It is for -1 like (n-1) in an index.
        
        
        
        # HERE ARE THE VARIABLES YOU HAVE
        # self._m_cavern (self._number_fluids size array)
        # self._t_cavern
        # self._t_cavern_wall
        # self._m_brine
        # self._t_brine
        # self._t_brine_wall

        
        # other components
        # self._wells_mdot = {}
        # self._wells_temp = {}
        # self._wells_pres = {}
        
        residuals = np.zeros(self._NUM_EQN)
        
        # Inititialise / setting initial values for t=0
        
        if x is None:
            self.load_var_values_from_x(self._model.xg)
        else:
            self.load_var_values_from_x(x)


        fluid_total_mass_flow = np.zeros(self._number_fluids)
        fluid_total_energy_flow = np.zeros(self._number_fluids)
        # equation 0-num_fluid-1 conservation of each component's mass
        for wname, well_mdot in self._wells_mdot.items():
            for vname, v_mdot in well_mdot.items():
                fluid = self._model.fluids[wname][vname]
                fluid_total_mass_flow += v_mdot
                fluid.update(CP.PT_INPUTS,
                             self._wells_pres[wname][vname],
                             self._wells_temp[wname][vname])
                hmass = fluid.hmass()
                fluid_total_energy_flow += hmass * v_mdot
        
        ### --- CONSERVATION OF MASS --- ###
        # conservation of mass for each fluid
        residuals[0:self._number_fluids] = (fluid_total_mass_flow * self._dt 
                                            + self._m_cavern_m1 
                                            - self._m_cavern)
        cfluid = self._model.fluids['cavern']
        cfluid.update(CP.PT_INPUTS,self._p_cavern_m1,self._t_cavern_m1)
        hmass_cavern_m1 = cfluid.hmass()
        cfluid.update(CP.PT_INPUTS,self._p_cavern,self._t_cavern)
        hmass_cavern = cfluid.hmass()
        
        ### -----   WALL HEAT FLUX MATCH TO GHE's ----- ###
        q_axi_total = np.array(self._q_axisym).sum()
        
        ht_coef_wall = self._wall_ht_coef(self.height_total-self.height_brine,cfluid)
        # TODO, you are just using water, you need it to be brine! maybe you should
        # write a class that inherits from the CoolProp Abstract state but incorporates
        # the brine properties when needed.
        ht_coef_brine_wall = self._wall_ht_coef(self.height_brine,self._water)

        # same as above
        q_cavern_wall = ((self._area_horizontal + self._area_vertical * 
                          (1 - (self._height_total-self._height_brine)/
                           self._height_total))
            * ht_coef_wall
            * (self._t_cavern_wall - self._t_cavern))
        
        q_cavern_brine_wall = ((self._area_horizontal + self._area_vertical * 
                               self._height_brine / self._height_total)
                               * ht_coef_brine_wall
                               * (self._t_brine_wall - self._t_brine))
        
        # ----  HEAT TO GHE ---- #
        # heat flux balancer between axisymmetric heat transfer (perhpas more than one
        # along the cavern)
        residuals[self._number_fluids] = -q_cavern_wall - q_cavern_brine_wall + q_axi_total
            
        
        # vapor pressure of brine in cavern
        p_brine_m1, solubility_brine_m1, rho_brine_m1 = brine_average_pressure(cfluid,self._water,self._height_total,self._height_brine,self._t_brine)
        self._water.update(CP.PT_INPUTS,p_brine_m1,self._t_brine)
        #evaluate again to get a better answer (pressure dependence is low enough to not warrant convergence)
        p_brine, solubility_brine, rho_brine = brine_average_pressure(cfluid,self._water,self._height_total,self._height_brine,self._t_brine)
        self._water.update(CP.PT_INPUTS,p_brine,self._t_brine)
        

        
        # TODO calculate change in cavern volume as brine is added or
        # subtracted, this is not yet done but is easy to add.
        
        # change in vapor pressure - is compensated by either condensation 
        # (negative vapor pressure change) which adds heat to the cavern gas
        # or evaporation (positive vapor pressure change) whichs takes heat away
        # from the brine
        # vp_brine = brine_saturated_pressure(self._t_brine,solubility_brine,self._water)
        # self._water.update(CP.PT_INPUTS,p_brine_m1,self._t_brine_m1)
        # vp_brine_m1 = brine_saturated_pressure(self._t_brine_m1,solubility_brine_m1,self._water)
        
        # find vapor mass change 
        self._water.update(CP.QT_INPUTS,1.0,self._t_cavern)
        rho_vapor = self._water.rhomass()
        self._water.update(CP.QT_INPUTS,1.0,self._t_cavern_m1)
        rho_vapor_m1 = self._water.rhomass()
        
        mass_change_vapor = (rho_vapor - rho_vapor_m1) * self._vol_cavern
        
        # CALCULATE VAPOR ENERGY EFFECTS
        
        # get the heat of vaporization
        self._water.update(CP.QT_INPUTS,0.0,(self._t_cavern+self._t_cavern_m1)/2)
        h_vapor_0 = self._water.hmass()
        self._water.update(CP.QT_INPUTS,1.0,(self._t_cavern+self._t_cavern_m1)/2)
        h_vapor_1 = self._water.hmass()
        h_evaporate = h_vapor_1 - h_vapor_0
        
        if mass_change_vapor > 0.0:
            E_vapor_brine = -h_evaporate * mass_change_vapor
            E_vapor_cavern = 0.0
        else:
            E_vapor_brine = 0.0
            # comes out positive since mass_change_vapor is negative
            E_vapor_cavern = -h_evaporate * mass_change_vapor
        
        #TODO
        # brine-cavern transfers (conduction, convection, radiation)
        
        ### -----   ENERGY    ----  ###
        #cavern
        residuals[self._number_fluids+1] = self._dt * (fluid_total_energy_flow + 
                                                       q_cavern_wall +
                                                       q_cavern_brine_wall +
                                                       
        #brine
        

            

        
        breakpoint()

        
        cfluid = self._model.fluids['cavern']
        
        ### ----- MASS OF BRINE   ---- ###
        residuals[self._number_fluids + 3] = self._m_brine - self._m_brine_m1 + mass_change_vapor
        


# EVERYTHING BELOW HERE IS JUNK FROM THE OLD CODE ERASE IT WHEN RESIDUALS IS WORKING!

        
        self.initialize()
        inp = self.inputs

        # mass balance 
        residuals[0] = 0

        self._mass_rate(0)
        self.total_mass_rate[0] = self._total_mass_rate(0)

        # Run actual integration by updating values by numerical integration/time stepping
        # Mass of fluid is calculated from previous time step mass and mass flow rate
        for i in range(1, len(self.time_array)):

            # if i == 100:
            #     breakpoint()
            #     from matplotlib import pyplot as plt
            #     plt.plot(self.time_array[:i],self.T_cavern[:i])
            #     plt.show()

            # must sum mass rate
            total_mass_rate = self._total_mass_rate(i)
            self.total_mass_rate[i] = total_mass_rate

            # filling is positive total mass rate, discharge is negative
            # the perspective is from the cavern.
            self.mass_fluid[i] = (
                self.mass_fluid[i - 1] + total_mass_rate * self.tstep
            )

            self.rho[i] = self.mass_fluid[i] / self.vol

            hi = self._calc_h_in(i)
            self.h_inside[i] = hi

            # same as above
            self.Q_inner[i] = (
                self.surf_area_inner
                * hi
                * (self.T_cavern_wall[i - 1] - self.T_cavern[i - 1])
            )

            # Axisymmetric, time-transient heat transfer
            # the Q_inner is negative when leaving the gas into
            # the salt but positive coming into the salt (thus)
            # the negative sign
            Tgvec = self.axsym.step(self.tstep,-self.Q_inner[i])

            self.T_cavern_wall[i] = Tgvec[0]

            # Run the explicit models of pipe losses for each well.
            for wname, well in self.wells.items():
                if not well.input["ideal_pipes"]:
                    well.step(i,self.P_cavern[i-1],self,wname)


            #NMOL = self.mass_fluid[i - 1] / self.MW
            #NMOL_ADD = (self.mass_fluid[i] - self.mass_fluid[i - 1]) / self.MW
            # New
            U_start = self.U_mass[i - 1] * self.mass_fluid[i - 1]

            # Finding the inlet/outlet enthalpy rate for the energy balance
            # HERE IS WHERE I STOPPED, YOU NEED TO WRITE A FUNCTION OVER
            # WELLS and PIPES that finds the average h_in

            h_in = self._enthalpy_rate(i)

            if i > 1:
                P1 = self.P_cavern[i - 2]
            else:
                P1 = self.P_cavern[i - 1]

            U_end = (
                U_start
                + self.tstep * total_mass_rate * h_in
                + self.tstep * self.Q_inner[i]
            )
            self.U_mass[i] = U_end / self.mass_fluid[i]

            P1, T1, self.U_res[i] = self.UDproblem(U_end/ self.mass_fluid[i],self.rho[i],self.P_cavern[i-1],self.T_cavern[i-1])


            self.P_cavern[i] = P1
            self.T_cavern[i] = T1
            self.fluid.update(CP.PT_INPUTS, self.P_cavern[i],  self.T_cavern[i])


            # Updating H,S,U states
            self.H_mass[i] = self.fluid.hmass()
            self.S_mass[i] = self.fluid.smass()
            self.U_mass[i] = self.fluid.umass()

            cpcv = self.fluid.cp0molar() / (self.fluid.cp0molar() - Constants.Rg['value'])

            for wname,well in inp["wells"].items():
                self._mass_rate(i)

            # dlvilla diagnostics to determine if the model
            if self.include_fault_analytics:
                self._model_fault_analytics(i)




        self.isrun = True
        
        
        breakpoint()


    def get_x(self):
        return np.array([[self._t_cavern] +
                         [self._t_cavern_wall] +
                         [self._m_brine] +
                         [self._t_brine] +
                         [self._t_brine_wall] +
                         list(self._m_cavern)])


    def load_var_values_from_x(self,xg):
        """
        This loads everything from get_x above coming from the solver.
        It also is the place that you need to load all variables 
        from previous or next components that interface with the salt_cavern
        
        """
        
        xloc = xg[self._gindices[0]:self._gindices[1]+1]
        self._t_cavern = xloc[0]
        self._t_cavern_wall = xloc[1]
        self._m_brine = xloc[2]
        self._t_brine = xloc[3]
        self._t_brine_wall = xloc[4]
        self._m_cavern = np.array([mass for mass in xloc[5:5+self._number_fluids]])
        
        # see model._build_ghes for the variable structure of each 
        # axisymmetric ground heat exchanger.
        self._q_axisym = []
        for cname, comp in self._next_components.items():
            if isinstance(comp, ImplicitEulerAxisymmetricRadialHeatTransfer):
                self._q_axisym.append(xg[comp.global_indices[0]])
            else:
                raise NotImplementedError("Only Axisymmetric heat transfer "
                                          +"can be a next component for salt"
                                          +" caverns currently!")
        
        # see model._build_well for the variable structure of each well
        self._wells_mdot = {}
        self._wells_temp = {}
        self._wells_pres = {}
        numf = self._number_fluids
        for cname, comp in self._prev_components.items():
            if isinstance(comp, Well):
                vnum = 1
                self._wells_mdot[cname] = {}
                self._wells_pres[cname] = {}
                self._wells_temp[cname] = {}
                gind = comp.global_indices
                for pname, pipe in comp.pipes.items():
                # flux into axisymetric
                    vname = pipe.valve_name
                    self._wells_mdot[cname][vname] = xg[gind[0]*vnum:gind[0]*vnum+numf]
                    self._wells_temp[cname][vname] = xg[gind[0]*vnum+numf+2]
                    self._wells_pres[cname][vname] = xg[gind[0]*vnum+numf+3]

                    
                
            else:
                raise NotImplementedError("Only Wells "
                                          +"can be a previous component for "
                                          +"salt caverns currently!")
                
    
    
    def _wall_ht_coef(self, length,cfluid):
        """
        Calculation of internal natural convective heat transfer coefficient from Nusselt number
        and using the coolprop low level interface.
        """
        
        if self._ht_coef == "calc":
            t_film = (self._t_cavern+self._t_cavern_wall)/2
            cfluid.update(CP.PT_INPUTS, self._p_cavern, t_film)
            cond = cfluid.conductivity()
            visc = cfluid.viscosity()
            cp = cfluid.cpmass()
            prandtl = cp * visc / cond
            beta = cfluid.isobaric_expansion_coefficient()
            nu = visc / cfluid.rhomass()
            grashoff = const['g']['value'] * beta * abs(self._t_cavern_wall - self._t_cavern) * length ** 3 / nu ** 2
            rayleigh = prandtl * grashoff
            nusselt = natural_convection_nu(rayleigh, prandtl)
            ht_coef = nusselt * cond / length
        else:
            ht_coef = self._ht_coef
        return ht_coef

