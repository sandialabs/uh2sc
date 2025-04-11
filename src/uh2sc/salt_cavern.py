# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:20:21 2023

@author: dlvilla
"""

import pandas as pd
import numpy as np

from CoolProp import CoolProp as CP

from uh2sc.abstract import AbstractComponent
from uh2sc.utilities import calculate_component_masses
from uh2sc.constants import Constants
from uh2sc.thermodynamics import (density_of_brine_water, 
                                  solubility_of_nacl_in_h2o)
from uh2sc.ghe import ImplicitEulerAxisymmetricRadialHeatTransfer
from uh2sc.well import Well
from uh2sc.transport import natural_convection_nu_vertical
from matplotlib import pyplot as plt


const = Constants()

class SaltCavern(AbstractComponent):

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
        self._STEFAN_BOLTZMANN = const.stefan_boltzman['value']
        
        #pure water for brine calculations
        water = CP.AbstractState("HEOS","Water")
        water.update(CP.PT_INPUTS,p0,T0brine)
        water.set_mass_fractions([1.0])
        self._height_brine = input_dict['initial']['liquid_height']
        self._height_total = input_dict["cavern"]["height"]
        self._fluid = model.fluids['cavern']
        self._fluid.update(CP.PT_INPUTS, input_dict['initial']['pressure'],
                                         input_dict['initial']['temperature'])
        
        delp_brine = 0.5 * const.g['value'] * (water.rhomass() * self._height_brine  + (self._height_total - self._height_brine) * self._fluid.rhomass())
        water.update(CP.PT_INPUTS,p0 + delp_brine,T0brine)

        self._water = water
        water_m1 = CP.AbstractState("HEOS","Water")
        water_m1.set_mass_fractions([1.0])
        water_m1.update(CP.PT_INPUTS,p0 + delp_brine,T0brine)
        self._water_m1 = water_m1
        
        # time and indices
        self._time = 0
        self._dt = input_dict['calculation']['time_step']
        self._first_step = True
        self._step_num = 0.0
        self._gindices = global_indices

        _fluid_m1 = CP.AbstractState("HEOS","&".join(self._fluid.fluid_names()))
        _fluid_m1.set_mass_fractions(self._fluid.get_mass_fractions())
        _fluid_m1.update(CP.PT_INPUTS, input_dict['initial']['pressure'],
                                       input_dict['initial']['temperature'])

        self._fluid_m1 = _fluid_m1
        self._number_fluids = len(model.fluids['cavern'].fluid_names())
        
        #initial geometry
        self._diameter = input_dict['cavern']['diameter']
        self._area_horizontal = np.pi * self._diameter**2/4
        self._area_vertical = np.pi * input_dict['cavern']['diameter'] * input_dict["cavern"]["height"]
        
        
        self._vol_brine = self._height_brine * self._area_horizontal
        self._vol_brine_m1 = self._vol_brine
        self._vol_cavern = (self._height_total - self._height_brine) * self._area_horizontal
        self._vol_cavern_m1 = self._vol_cavern
        self._VOL_TOTAL = self._height_total * self._area_horizontal
        
        
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
        self._emissivity = input_dict['cavern']['emissivity']
        (self._p_brine, solubility_brine,
         rho_brine) = self._brine_average_pressure(self._fluid,self._water)
        
        
        

        self._m_brine = self._vol_brine * rho_brine
        self._m_brine_m1 = self._m_brine
        
        self._m_cavern = calculate_component_masses(self._fluid,
                                model.molar_masses,
                                self._fluid.rhomass() * self._vol_cavern,
                                liquid_mass=0.0)['gas']
        self._m_cavern_m1 = self._m_cavern
        
        # heat transfer coefficients
        self._ht_coef = input_dict['heat_transfer']['h_inner']
        
        (self._m_vapor, 
         mass_vapor_change, 
         e_vapor_brine, 
         e_vapor_cavern) = (
            self._evaporation_energy(self._water, self._water_m1, 
                                T0, T0, 
                                T0brine, T0brine, 
                                self._vol_cavern, self._vol_cavern_m1))
        self._m_vapor_m1 = self._m_vapor
        
        
        # solution
        self._NUM_EQN = self._number_fluids + 5 # see get_x for the 5 first variables.


        self._model = model

        self.cavern_results = {val: [] for val in self._result_name_map.values()}


    def _brine_average_pressure(self,fluid,water):
        """
        Approximate the average pressure for the brine
        
        """
        # TODO: create a test for this!
        pres_g = fluid.p()
        rho_g = fluid.rhomass()
        # assume density is constant
        height_gas_o_2 = (self._height_total - self._height_brine)/2
        pres_g_surf = pres_g + rho_g * const.g['value'] * height_gas_o_2
        rho_pure_water = water.rhomass()
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
        return "cavern"
    
    def equations_list(self):
        # THIS MUST BE UPDATED WHENEVER THE RESIDUALS EQUATION ORDER
        # IS CHANGED!
        e_list = []
        for fluid_name in self._fluid.fluid_names():
            e_list += [f"Cavern conservation of mass for {fluid_name}"]
            
        e_list += ["Heat flux continuity between cavern and axisymmetric heat transfer"]
        
        if self._model.is_test_mode:
            e_list += [f"Temperature continuity between cavern and axisymmetric heat transfer 1"]
        else:
            for idx, t_axisym in enumerate(self._t_axisym):
                e_list += [f"Temperature continuity between cavern and axisymmetric heat transfer {idx}"]
            
        e_list += ["Cavern energy balance"]
        e_list += ["Brine energy balance"]
        e_list += ["Brine mass balance"]
        
        return e_list
            

    def evaluate_residuals(self,x=None,get_independent_vars=False):
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
                
        
        The equations can be discovered via 
        
        self._model.equations_list()[self.global_indices[0]:self.global_indices[1]]
        
        The variables can be discovered via 
        
        self._model.xg_descriptions[self.global_indices[0]:self.global_indices[1]]
        
        """
        

        # IMPORTANT! - "m1" indicates a variable's value in the previous time 
        #              step. It is for -1 like (n-1) in an index.
        
        # CONVENTION - flux is negative if it is leaving the cavern
        
        # Inititialize / setting initial values for t=0        
        if x is not None:
            self.load_var_values_from_x(x)
        

        _eqn = 0
        
        # geometry
        height_total = self._height_total
        height_brine = self._height_brine
        height_cavern = (height_total-height_brine)
        diameter_cavern = self._diameter
        brine_area = self._area_horizontal
        
        # time step
        dt = self._dt
        
        #-----------------
        # INDEPENDENT VARIABLES 
        #-----------------
        
        # unpack variables and previous time step's variable values (constant now)
        # _m1 is for previous time step.
        t_cavern = self._t_cavern
        t_cavern_m1 = self._t_cavern_m1
        t_cavern_wall = self._t_cavern_wall
        #t_cavern_wall_m1 = self._t_cavern_wall_m1
        t_brine = self._t_brine
        t_brine_m1 = self._t_brine_m1
        t_brine_wall = self._t_brine_wall
        #t_brine_wall_m1 = self._t_brine_wall_m1
        m_cavern = self._m_cavern
        m_cavern_m1 = self._m_cavern_m1
        m_brine = self._m_brine
        m_brine_m1 = self._m_brine_m1

        #----------------
        # DEPENDENT VARIABLES -- AVOID STORING IN self since these must
        # be derived from the independent variables above
        #----------------


        m_vapor = self._m_vapor
        m_vapor_m1 = self._m_vapor_m1        
        # calculate densities 


        # CALCULATE VAPOR ENERGY EFFECTS
        (mass_vapor, mass_change_vapor, e_vapor_brine, e_vapor_cavern) = (
            self._evaporation_energy(self._water, self._water_m1, 
                                    self._t_cavern, self._t_cavern_m1, 
                                    self._t_brine, self._t_brine_m1, 
                                    self._vol_cavern, self._vol_cavern_m1))

        rho_brine = (m_brine-m_vapor)/self._vol_brine
        
        # Change in brine volume
        del_vol_brine = -mass_change_vapor/rho_brine
        del_height_brine = del_vol_brine / self._area_horizontal
        vol_cavern = self._vol_cavern - del_vol_brine
        vol_cavern_m1 = self._vol_cavern_m1
        vol_brine = self._vol_brine + del_vol_brine
        vol_brine_m1 = self._vol_brine_m1
        
        rho_cavern = (m_cavern.sum()+m_vapor)/vol_cavern
        rho_cavern_m1 = (m_cavern_m1.sum() + m_vapor_m1)/vol_cavern_m1

        rho_brine_m1 = (m_brine_m1 - m_vapor)/vol_brine_m1
        
        # update fluids _m1 is updated in shift_solution as well
        # this has to be updated here because new values for the
        # variables are continuously tried by the NewtonSolver
        fluid = self._fluid
        fluid_m1 = self._fluid_m1
        water = self._water
        water_m1 = self._water_m1
        fluid.update(CP.DmassT_INPUTS,rho_cavern,t_cavern)
        water.update(CP.DmassT_INPUTS,rho_brine,t_brine)
        fluid_m1.update(CP.DmassT_INPUTS, rho_cavern_m1, t_cavern_m1)
        water_m1.update(CP.DmassT_INPUTS,rho_brine_m1, t_brine_m1)
        


        # pressure
        p_cavern = fluid.p()
        #p_cavern_m1 = fluid_m1.p()
        # brine pressure
        (p_brine_m1, solubility_brine_m1, rho_brine_m1) = (
            self._brine_average_pressure(fluid_m1,water_m1))
        (p_brine, solubility_brine, rho_brine) = (
            self._brine_average_pressure(fluid, water))
        
        #enthalpies
        hmass_cavern_m1 = fluid_m1.hmass()
        hmass_cavern = fluid.hmass()
        hmass_brine_m1 = water_m1.hmass()
        hmass_brine = water.hmass()
        
        # total energy of cavern and brine
        e_cavern_m1 = hmass_cavern_m1 * m_cavern_m1
        e_cavern = hmass_cavern * m_cavern
        e_brine_m1 = hmass_brine_m1 * m_brine_m1
        e_brine = hmass_brine * m_brine
        
        ### --- GAS MASS FLUX AND GAS MASS ENERGY FLUX --- ###
        # equation 0 to num_fluid-1 conservation of each component's mass
        (cavern_total_mass_flow, 
         cavern_mass_energy_flow) = self._mass_flux_and_mass_energy_flux(fluid,
                                                                         hmass_cavern,
                                                                         p_cavern,
                                                                         t_cavern)
        q_axi_total, q_axi_brine, q_axi_cavern = self._axisymmetric_heat_flux(
            t_cavern_wall,t_brine_wall,height_cavern)

        ht_coef_wall = self._wall_ht_coef(height_cavern,fluid)
        # TODO, you are just using water, you need it to be brine! maybe you should
        # write a class that inherits from the CoolProp Abstract state but incorporates
        # the brine properties when needed.
        #breakpoint()
        ht_coef_brine_wall = self._wall_ht_coef(height_brine,water)

        # same as above
        q_cavern_wall = ((self._area_horizontal + self._area_vertical * 
                          (1 - height_cavern/
                           height_total))
            * ht_coef_wall
            * (t_cavern_wall - t_cavern))
        
        q_brine_wall = ((self._area_horizontal + self._area_vertical * 
                               height_brine / height_total)
                               * ht_coef_brine_wall
                               * (t_brine_wall - t_brine))


        
        
        

        
        # RADIATION ENERGY TRANSFER FROM BRINE TO CAVERN WALLS
        f12 = self._radiation_vf(diameter_cavern,height_cavern)
        q_rad = (f12 
                 * self._emissivity 
                 * self._STEFAN_BOLTZMANN 
                 * self._area_horizontal 
                 * (self._t_cavern ** 4 
                 - self._t_brine ** 4))
        
        #TODO make brine have input/output of mass via wells 
        brine_mass_energy_flow = 0.0
        
        if get_independent_vars:
            # PUT ANY VARIABLES YOU WANT TO WRAP INTO shift_solution, or
            # into results timeseries here!
            return (del_vol_brine,
                    mass_vapor,
                    vol_cavern,
                    vol_brine,
                    p_brine,
                    )
            
        
        
        # ------------------------------
        # END OF DEPENDENT VARIABLES
        # ------------------------------
        
        ##
        ###
        ####  EQUATIONS - WRITTEN IN RESIDUAL FORM f(x)=0
        ###
        ##
        residuals = np.zeros(self._NUM_EQN)

        ### --- CONSERVATION OF MASS --- ###
        # conservation of mass for each fluid
        residuals[0:self._number_fluids] = (cavern_total_mass_flow * dt 
                                            + m_cavern_m1 
                                            - m_cavern)
        _eqn = self._number_fluids

        # ----  HEAT TO GHE ---- #
        # heat flux balance between axisymmetric heat transfer (perhpas more than one
        # along the cavern)
        # TODO - this is only going to work with one ground heat exchanger. If you want to
        # stack ground heat exchangers (to capture temperature gradient), then you 
        # need to update this.
        
        residuals[_eqn] = -q_cavern_wall - q_brine_wall + q_axi_total
        _eqn += 1
        
        # ----  GHE temperature ---- #
        # The weighted average of the cavern and brine wall temperatures is equal to the GHE wall temperature
        # This is written as would be expected but doesn't consider different depths 
        # for GHE's
        
        if self._model.is_test_mode:
            # cavern flux continuity
            residuals[_eqn] = q_cavern_wall - q_axi_cavern
            _eqn += 1
        else:
            for t_axisym in self._t_axisym:
                residuals[_eqn] = ((-t_cavern_wall 
                                         * height_cavern - del_height_brine * 0.5 
                                         + t_brine_wall 
                                         * height_brine + del_height_brine * 0.5)
                                        /height_total 
                                        - t_axisym)
            _eqn += 1
        


        

        ### -----   CAVERN ENERGY    ----  ###
        #cavern

        residuals[_eqn] = (dt 
                           * (cavern_mass_energy_flow 
                           + q_cavern_wall
                           - q_rad) 
                           + e_vapor_cavern
                           - e_cavern.sum()
                           + e_cavern_m1.sum())
        if self._model.time > 80000 and self._model.converged_solution_point:
            breakpoint()
            plt.bar(height=np.array([dt*cavern_mass_energy_flow,q_cavern_wall*dt,-q_rad*dt,e_vapor_cavern,e_cavern_m1.sum()-e_cavern.sum()]),
                     x=['mass e','q_cavern_wall','q_rad','e_vapor','del_e_cavern'])
            #breakpoint()
        
        _eqn += 1   
                                         
        ### ----  BRINE ENERGY ---- ###

        residuals[_eqn] = (dt 
                          * (brine_mass_energy_flow 
                          + q_brine_wall
                          + q_rad) 
                          + e_vapor_brine
                          - e_brine
                          + e_brine_m1)
        if hasattr(self,"troubleshooting"):
            plt.bar(height=np.array([dt*brine_mass_energy_flow,q_brine_wall*dt,q_rad*dt,e_vapor_brine,e_brine_m1-e_brine]),
                     x=['mass e','q_brine_wall','q_rad','e_vapor','del_e_brine'])
        
        _eqn += 1


        ### ----- Brine Mass ---- ####
        
        residuals[_eqn] = -m_brine + m_brine_m1 - mass_change_vapor
        
        
        if _eqn + 1 != self._NUM_EQN:
            raise ValueError("The number of residuals calculated is not equal"
                             +" to the number of equations/variables! This is"
                             +" a developer bug!")

        #TODO, add flag to store results

        return residuals
        
        

    def get_x(self):
        return np.array([[self._t_cavern] +
                         [self._t_cavern_wall] +
                         [self._m_brine] +
                         [self._t_brine] +
                         [self._t_brine_wall] +
                         list(self._m_cavern)])
    
    
    def shift_solution(self):
        # shift independent variables
        self._t_cavern_m1 = self._t_cavern
        self._t_cavern_wall_m1 = self._t_cavern_wall
        self._m_brine_m1 = self._m_brine
        self._t_brine_m1 = self._t_brine
        self._t_brine_wall_m1 = self._t_brine_wall
        self._m_cavern_m1 = self._m_cavern
        
        (del_vol_brine,mass_vapor,
                vol_cavern,
                vol_brine,
                p_brine) = self.evaluate_residuals(get_independent_vars=True)
        
        # shift the composition of the gas fluid
        # calculate 
        total_mass = self._m_cavern.sum()
        self._fluid_m1.set_mass_fractions(self._m_cavern/total_mass)
        rho_cavern = total_mass / self._vol_cavern
        self._fluid_m1.update(CP.DmassT_INPUTS,rho_cavern, self._t_cavern)
        self._p_cavern_m1 = self._fluid.p()
        self._water_m1.update(CP.PT_INPUTS,p_brine,self._t_brine)
        
        self._vol_cavern_m1 = self._VOL_TOTAL - vol_brine - del_vol_brine
        self._vol_brine_m1 = self._VOL_TOTAL - self._vol_cavern_m1
        


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
        if hasattr(self,"_next_components"):
            self._q_axisym = []
            self._t_axisym = []

            for cname, comp in self._next_components.items():
                if isinstance(comp, ImplicitEulerAxisymmetricRadialHeatTransfer):
                    self._q_axisym.append(xg[comp.global_indices[0]]) # first place is the wall fliux
                    self._t_axisym.append(xg[comp.global_indices[1]]) # second place is the first temperature
                else:
                    raise NotImplementedError("Only Axisymmetric heat transfer "
                                              +"can be a next component for salt"
                                              +" caverns currently!")
        
        if hasattr(self,"_prev_components"):
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
            t_restore = cfluid.T()
            p_restore = cfluid.p()
            cfluid.update(CP.PT_INPUTS, self._p_cavern, t_film)
            cond = cfluid.conductivity()
            visc = cfluid.viscosity()
            cp = cfluid.cpmass()
            prandtl = cp * visc / cond
            beta = cfluid.isobaric_expansion_coefficient()
            nu = visc / cfluid.rhomass()
            grashoff = const.g['value'] * beta * abs(self._t_cavern_wall - self._t_cavern) * length ** 3 / nu ** 2
            rayleigh = prandtl * grashoff
            nusselt = natural_convection_nu_vertical(rayleigh, prandtl)
            ht_coef = nusselt * cond / length
            
            cfluid.update(CP.PT_INPUTS,p_restore,t_restore)
        else:
            ht_coef = self._ht_coef
        return ht_coef
    
    
    def _brine_cavern_ht_coef(self, radius, cfluid):
        # TODO - put actual values!!
        return 5.0
    
    
    def _radiation_vf(self, cavern_diameter, cavern_height):
        _r = cavern_diameter / 2.0
        _rho = (np.sqrt(4.0 * _r**2 + 1) - 1)/_r
        return _rho / (2.0 * _r)
        
    
    
    def _evaporation_energy(self,water, water_m1, 
                            t_cavern, t_cavern_m1, 
                            t_brine, t_brine_m1, 
                            vol_cavern, vol_cavern_m1):
        # find vapor mass change due to condensation (and settling to the brine)
        # or evaporation
        restore_p_brine = water.p()
        restore_t_brine = water.T()
        restore_p_brine_m1 = water_m1.p()
        restore_t_brine_m1 = water_m1.T()
        
        # get density of saturated vapor at the temperature and saturated 
        # vapor pressure
        water.update(CP.QT_INPUTS,1.0,t_cavern)
        rho_vapor = water.rhomass()
        water_m1.update(CP.QT_INPUTS,1.0,t_cavern_m1)
        rho_vapor_m1 = water_m1.rhomass()
        
        # get the heat of vaporization at the average temperature during the time 
        # step
        water.update(CP.QT_INPUTS,0.0,(t_cavern+t_cavern_m1)/2)
        h_vapor_0 = water.hmass()
        water.update(CP.QT_INPUTS,1.0,(t_cavern+t_cavern_m1)/2)
        h_vapor_1 = water.hmass()
        h_evaporate = h_vapor_1 - h_vapor_0
        
        # restore the water CoolProp fluids to their original state.
        water.update(CP.PT_INPUTS,restore_p_brine, restore_t_brine)
        water_m1.update(CP.PT_INPUTS,restore_p_brine_m1, restore_t_brine_m1)
        
        
        mass_vapor_m1 = rho_vapor_m1 * vol_cavern_m1
        mass_vapor = rho_vapor * vol_cavern

        mass_change_vapor = mass_vapor - mass_vapor_m1
        
        if mass_change_vapor > 0.0:
            e_vapor_brine = -h_evaporate * mass_change_vapor
            e_vapor_cavern = 0.0
        else:
            e_vapor_brine = 0.0
            # comes out positive since mass_change_vapor is negative
            e_vapor_cavern = -h_evaporate * mass_change_vapor
        
        return mass_vapor, mass_change_vapor, e_vapor_brine, e_vapor_cavern
        
        
        pass
    
    def _mass_flux_and_mass_energy_flux(self,fluid,hmass,p_cavern,t_cavern):
        
        cavern_total_mass_flow = np.zeros(self._number_fluids)
        cavern_mass_energy_flow = np.zeros(self._number_fluids)
        
        if self._model.is_test_mode:
            v_mdot = np.interp(self._model.time, 
                               self._model.test_inputs["time"],
                               self._model.test_inputs["mdot"])
            fluid.update(CP.PT_INPUTS,
                         p_cavern,
                         t_cavern)
            
            cavern_total_mass_flow = v_mdot
            cavern_mass_energy_flow = hmass * v_mdot

        else:
            for wname, well_mdot in self._wells_mdot.items():
                for vname, v_mdot in well_mdot.items():
                    wfluid = self._model.fluids[wname][vname]
                    cavern_total_mass_flow += v_mdot
                    wfluid.update(CP.PT_INPUTS,
                                 self._wells_pres[wname][vname],
                                 self._wells_temp[wname][vname])
                    hmass_arr = np.array([wfluid.hmass() if v_mdot[idx] > 0.0
                                          else hmass 
                                          for idx in range(self._number_fluids)])
                    cavern_mass_energy_flow += hmass_arr * v_mdot
                    
        return cavern_total_mass_flow, cavern_mass_energy_flow
    

    def _axisymmetric_heat_flux(self,t_cavern_wall,t_brine_wall,height_cavern):
        
        if self._model.is_test_mode:
            q_axi_cavern = ((-t_cavern_wall 
                           + self._model.test_inputs["farfield_temp"])
                          /self._model.test_inputs["r_radial"])
            q_axi_brine = ((-t_brine_wall 
                             + self._model.test_inputs["farfield_temp"])
                            /self._model.test_inputs["r_radial"])
            q_axi_total = q_axi_cavern + q_axi_brine
        else:
            ### -----   WALL HEAT FLUX MATCH TO GHE's ----- ###
            q_axi_total = np.array(self._q_axisym).sum()
            q_axi_cavern = q_axi_total * height_cavern / self._height_total
            q_axi_brine = q_axi_total * self._height_brine / self._height_total
            
        return q_axi_total, q_axi_brine, q_axi_cavern
            
        
        
        