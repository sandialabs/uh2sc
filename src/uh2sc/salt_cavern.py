# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:20:21 2023

@author: dlvilla
"""

import pandas as pd
import numpy as np

from CoolProp import CoolProp as CP

from uh2sc.abstract import AbstractComponent
from uh2sc.utilities import (calculate_component_masses,
                             calculate_cavern_pressure,
                             brine_average_pressure,
                             conservation_of_volume,
                             init_water)
from uh2sc.constants import Constants
from uh2sc.ghe import ImplicitEulerAxisymmetricRadialHeatTransfer
from uh2sc.well import Well
from uh2sc.transport import natural_convection_nu_vertical
from uh2sc.errors import CavernStateOutOfOperationalBounds
from uh2sc.fluid import FluidWithFitOption


const = Constants()

class AnalyticsDB():
    variables = ["temperature (K)",
                "pressure (MPa)",
                "mass flow (kg/s)"]

    beyond_points = ["0-5%",
                    "5-10%",
                    ">10%"]

    changes = ["cavern.min_operational_temperature and cavern.max_"
                      +"operational_temperature",
                      ("cavern.overburden_pressure, cavern.min_operational"
                      +"_pressure_ratio and cavern.max_operational_press"
                      +"ure_ratio"),
                      "cavern.max_volume_change_per_day"]

    consequences = [("reaching temperature extremes can lead to rock "
                    +"strength decreasing (or brittle fracture for cold),"
                    +" accelerate equipment degredation, stop creep which"
                    +" keeps self-healing of stress from occuring, and"
                    +" increase condensation and/or ice formation leading"
                    +" to blockage and damage to sensors among other things."),
                    ("reaching close to the overburden pressure or low "
                     +"pressure can cause tensile failures at the roof"
                     +" leading to sudden changes in caver geometry and "
                     +"even cavern collapse. For low pressure, collapse"
                     +" and increased creep will occur. Also loss of "+
                     "cavern seal (leaks) and well integrity failure can"
                     +" occur."),
                    ("cycling salt"
                    +" caverns too quickly wastes energy, Thermal"
                    +" gradients and stresses in the Halite (salt)"
                    +" walls, Spalling of salt (flaking off), and"
                    +" stress build up due to inability of the "
                    +"salt to creep leading to potential "
                    +"nonuniform collapse.")]

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
        """
        
        """
        if model is None:
            self._use_relative_convergence = False
            self.residual_normalization = None
        else:
            self._use_relative_convergence = model._use_relative_convergence
            self.residual_normalization = model.residual_normalization
        
        # this includes validation of the input dictionary which must follow
        # the main schema in validatory.py
        # do stuff relevant to model
        T0 = input_dict['initial']['temperature']
        T0brine = input_dict['initial']['liquid_temperature']
        p0 = input_dict['initial']['pressure']
        self._STEFAN_BOLTZMANN = const.stefan_boltzman['value']

        #initial geometry
        self._diameter = input_dict['cavern']['diameter']
        self._area_horizontal = np.pi * self._diameter**2/4
        self._area_vertical = np.pi * input_dict['cavern']['diameter'] * input_dict["cavern"]["height"]

        #pure water for brine calculations
        water = CP.AbstractState("HEOS","Water")
        water.update(CP.PT_INPUTS,p0,T0brine)
        water.set_mass_fractions([1.0])
        self._initial_volume_brine = input_dict['initial']['liquid_height'] * self._area_horizontal
        self._height_brine = self._initial_volume_brine / self._area_horizontal
        self._height_brine_m1 = self._height_brine
        self._height_total = input_dict["cavern"]["height"]
        self._fluid = model.fluids['cavern']
        self._fluid.set_state(CP.AbstractState, [input_dict['initial']['pressure'],
                                         input_dict['initial']['temperature']])

        delp_brine = 0.5 * const.g['value'] * (water.rhomass() * self._height_brine  + (self._height_total - self._height_brine) * self._fluid.rhomass())
        water.update(CP.PT_INPUTS,p0 + delp_brine,T0brine)

        self._water = ("HEOS","Water",[1.0],[p0 + delp_brine, T0brine])

        self._water_m1 = ("HEOS","Water",[1.0],[p0 + delp_brine, T0brine])

        # time and indices
        self._gindices = global_indices

        _fluid_m1 = (self._fluid.backend,
                     "&".join(self._fluid.fluid_names()),
                     self._fluid.get_mass_fractions(),
                     )


        self._fluid_m1 = FluidWithFitOption(_fluid_m1[0:3], 
                                            self._fluid._fit, 
                                            self._fluid._logging, 
                                            self._fluid._acceptable_error,
                                            [self._fluid.p(),self._fluid.T()])
        
        self._number_fluids = len(model.fluids['cavern'].fluid_names())

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
         rho_brine) = brine_average_pressure(self._fluid,
                                             water,
                                             self._height_total,
                                             self._height_brine,
                                             self._t_brine)
        self._p_brine_m1 = self._p_brine
        self._m_brine = self._vol_brine * rho_brine
        self._m_brine_m1 = self._m_brine
        self._m_cavern = calculate_component_masses(self._fluid,
                                self._fluid.rhomass() * self._vol_cavern)
        self._m_cavern_m1 = self._m_cavern

        # heat transfer coefficients
        self._ht_coef = input_dict['heat_transfer']['h_inner']
        self._ht_cavern_brine = input_dict['heat_transfer']['h_cavern_brine']


        # solution
        self._NUM_EQN = self._number_fluids + 4 # see get_x for the 5 first variables.

        self._model = model

        self._residual_iter = 0

            
        self._fluid.del_state()



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

        # time step
        dt = self._model.time_step

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


        m_cavern = self._m_cavern     #mass of dry gasses in the cavern (excluding water vapor)
        m_cavern_m1 = self._m_cavern_m1
        m_brine = self._m_brine
        m_brine_m1 = self._m_brine_m1

        # update composition of the current and previous time step fluids.
        fluid = self._fluid
        fluid.set_state(CP.AbstractState,[self._p_cavern,t_cavern])
        
        fluid_m1 = self._fluid_m1
        fluid_m1.set_state(CP.AbstractState,[self._p_cavern_m1, self._t_cavern_m1])
        
        # initialize water Abstract State objects
        water = init_water(self._water)
        water_m1 = init_water(self._water_m1)

        fluid.set_mass_fractions(m_cavern/m_cavern.sum())
        fluid_m1.set_mass_fractions(m_cavern_m1/m_cavern_m1.sum())

        cavern_volume_estimate = self._VOL_TOTAL - self._initial_volume_brine

        #----------------
        # DEPENDENT VARIABLES -- AVOID STORING IN self since these must
        # be derived from the independent variables above
        #----------------
        (p_cavern_novapor,
         vol_cavern) = calculate_cavern_pressure(fluid,
                                      m_cavern,
                                      t_cavern,
                                      water,
                                      m_brine,
                                      t_brine,
                                      self._VOL_TOTAL,
                                      self._area_horizontal,
                                      cavern_volume_estimate)

        rho_cavern_novapor = fluid.rhomass()
        p_brine = water.p()
        rho_brine = water.rhomass()

        (mass_vapor, rho_vapor, h_vapor,
         p_vapor, h_evaporate) = conservation_of_volume(vol_cavern,
             self._VOL_TOTAL, self._area_horizontal, water, t_cavern,
             t_brine, m_cavern, m_brine, fluid,True)

        # unpack previous time step. (don't run calculate cavern_pressure again
        # because it is expensive!)
        p_cavern_no_vapor_m1 = self._p_cavern_m1
        p_brine_m1 = self._p_brine_m1
        vol_cavern_m1 = self._vol_cavern_m1

        fluid_m1.update(CP.PT_INPUTS,p_cavern_no_vapor_m1,t_cavern_m1)

        #rho_cavern_novapor_m1 = fluid_m1.rhomass()

        water_m1.update(CP.PT_INPUTS,p_brine_m1,t_brine_m1)
        rho_brine_m1 = water_m1.rhomass()

        (mass_vapor_m1, rho_vapor_m1, h_vapor_m1,
         p_vapor_m1, h_evaporate_m1) = conservation_of_volume(vol_cavern_m1,
             self._VOL_TOTAL, self._area_horizontal, water_m1, t_cavern_m1,
             t_brine_m1, m_cavern_m1, m_brine_m1, fluid_m1,True)

        mass_change_vapor = mass_vapor - mass_vapor_m1
        avg_h_evap = (h_evaporate + h_evaporate_m1)/2

        volume_liquid_brine = m_brine/rho_brine
        #volume_liquid_brine_m1 = m_brine_m1/rho_brine_m1

        # checking for consistency from fsolve!
        vol_cavern_check = np.abs((self._VOL_TOTAL - vol_cavern)/volume_liquid_brine - 1)
        if vol_cavern_check > 0.00001:
            raise ValueError("The function 'calculate_cavern_pressure' has "
                             +"not found a state that balances mass, volume, "
                             +"and pressure!")

        height_total = self._height_total
        height_cavern = vol_cavern / self._area_horizontal
        height_brine = volume_liquid_brine / self._area_horizontal
        diameter_cavern = self._diameter

        # evaporization/condensation respiration
        e_vapor = -avg_h_evap * mass_change_vapor #
        # if mass change vapor > 0 then we are evaporaing which gives negative heat
        e_vapor_brine = 0.0 # e_vapor * height_brine / height_total
        e_vapor_cavern = e_vapor


        # update fluids _m1 is updated in shift_solution as well
        # this has to be updated here because new values for the
        # variables are continuously tried by the NewtonSolver
        water.update(CP.DmassT_INPUTS,rho_brine,t_brine)
        #fluid_m1.update(CP.DmassT_INPUTS, rho_cavern_m1, t_cavern_m1)
        water_m1.update(CP.DmassT_INPUTS,rho_brine_m1, t_brine_m1)

        # Change in brine volume
        del_vol_brine = -mass_change_vapor/rho_brine


        #enthalpies
        hmass_cavern_m1 = fluid_m1.hmass()
        hmass_cavern = fluid.hmass()
        hmass_brine_m1 = water_m1.hmass()
        hmass_brine = water.hmass()


        # total energy of cavern and brine
        e_cavern_m1 = hmass_cavern_m1 * m_cavern_m1.sum()
        e_cavern = hmass_cavern * m_cavern.sum()
        e_brine_m1 = hmass_brine_m1 * m_brine_m1
        e_brine = hmass_brine * m_brine

        ### --- GAS MASS FLUX AND GAS MASS ENERGY FLUX --- ###
        # equation 0 to num_fluid-1 conservation of each component's mass

        (cavern_gas_mass_flow,
         cavern_mass_energy_flow,
         cavern_vapor_mass_flow,
         cavern_vapor_energy_flow) = self._mass_flux_and_mass_energy_flux(fluid,
                                                                         hmass_cavern,
                                                                         p_cavern_novapor,
                                                                         t_cavern,
                                                                         rho_vapor,
                                                                         rho_cavern_novapor,
                                                                         h_vapor)

        q_cavern_wall, q_brine_wall = self.cavern_wall_heat_flux(t_cavern,
                                  t_cavern_wall,t_brine,
                                  fluid,water,height_cavern,height_brine,
                                  height_total)

        q_cavern_brine = (self._ht_cavern_brine
                          * self._area_horizontal
                          * (t_cavern - t_brine))

        # RADIATION ENERGY TRANSFER FROM BRINE TO CAVERN WALLS
        f12 = self._radiation_vf(diameter_cavern,height_cavern)
        q_rad = (f12
                 * self._emissivity
                 * self._STEFAN_BOLTZMANN
                 * self._area_horizontal
                 * (self._t_cavern ** 4
                 - self._t_brine ** 4))

        # WATER CANNOT BE ADDED BUT IT CAN BE EXTRACTED!
        brine_mass_energy_flow = cavern_vapor_energy_flow

        # Expansion/Compression term
        drhodt = cavern_gas_mass_flow.sum() / vol_cavern
        dPdT = rho_cavern_novapor * fluid.compressibility_factor() * fluid.gas_constant()/fluid.molar_mass()

        #dPdT = CP.PropsSI("d(P)/d(T)|Cpmass",'P',p_cavern_novapor,'T',t_cavern,"Hydrogen")

        vt_o_d = vol_cavern * t_cavern / rho_cavern_novapor
        #vt_o_d = 0.0

        #expansion_term = vt_o_d * dPdT * drhodt

        if get_independent_vars:

            time = self._model.time
            self._model.logging.info(f"Results for salt cavern at time {time} gathered!")



            # update global values.
            self._p_cavern = p_cavern_novapor
            self._mdot_cavern = cavern_gas_mass_flow
            self._mdot_vapor = cavern_vapor_mass_flow



            # PUT ANY VARIABLES YOU WANT TO WRAP INTO shift_solution, or
            # into results timeseries here!
            # THIS IS A MESS WITH ARRAYS VS FLOATS, Everything needs to
            # be floats
            
            
            
            return (float(self._sum_or_float(del_vol_brine)),
                    float(self._sum_or_float(mass_vapor)),
                    float(self._sum_or_float(vol_cavern)),
                    float(volume_liquid_brine),
                    float(p_brine),
                    float(cavern_vapor_mass_flow),
                    float(p_cavern_novapor + p_vapor),
                    float(e_brine),
                    float(e_cavern),
                    float(p_cavern_novapor))



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
        residuals[0:self._number_fluids] = (cavern_gas_mass_flow * dt
                                            + m_cavern_m1
                                            - m_cavern)
        if self._use_relative_convergence:
            residuals[0:self._number_fluids] = (
                residuals[0:self._number_fluids]
                /self.residual_normalization["cavern_gas_mass"]
                )
        
        _eqn = self._number_fluids


        # ----  GHE temperature ---- #
        # The weighted average of the cavern and brine wall temperatures is equal to the GHE wall temperature
        # This is written as would be expected but doesn't consider different depths
        # for GHE's

        if self._model.is_test_mode:
            q_axi_cavern = self._simplified_axisymmetric_heat_flux(
                t_cavern_wall,height_cavern)
            # cavern flux continuity
            residuals[_eqn] = q_cavern_wall - q_axi_cavern
            if self._use_relative_convergence:
                residuals[_eqn] = (
                    residuals[_eqn] 
                    / self.residual_normalization["heat_flux_norm"]
                    )
            _eqn += 1
        else:
            residuals[_eqn] = t_cavern_wall-self._t_axisym[0]
            if self._use_relative_convergence:
                residuals[_eqn] = (
                    residuals[_eqn] 
                    / self.residual_normalization["temperature_norm"]
                    )
            _eqn += 1

        ### -----   CAVERN ENERGY    ----  ###
        #cavern
        residuals[_eqn] = ((dt
                           * (self._sum_or_float(cavern_mass_energy_flow)
                           + self._sum_or_float(q_cavern_wall)
                           - q_rad
                           - q_cavern_brine
                           + self._sum_or_float(vt_o_d * drhodt * dPdT)))     # expansion term)
                           + self._sum_or_float(e_vapor_cavern)
                           - (e_cavern.sum()
                           - e_cavern_m1.sum()))
        if self._use_relative_convergence:
            residuals[_eqn] = (
                residuals[_eqn] 
                / self.residual_normalization["cavern_gas_energy"]
                )
        _eqn += 1

        ### ----  BRINE ENERGY ---- ###

        residuals[_eqn] = (dt
                          * (brine_mass_energy_flow
                          + q_brine_wall
                          + q_rad
                          + q_cavern_brine
                          + e_vapor_brine)
                          - e_brine
                          + e_brine_m1)
        if self._use_relative_convergence:
            residuals[_eqn] = (
                residuals[_eqn] 
                / self.residual_normalization["brine_energy"]
                )

        _eqn += 1

        ### ----- Brine Mass ---- ####

        residuals[_eqn] = (-m_brine 
                           + m_brine_m1 
                           - self._sum_or_float(mass_change_vapor) 
                           + cavern_vapor_mass_flow * dt)
        
        if self._use_relative_convergence:
            residuals[_eqn] = (
                residuals[_eqn] 
                / self.residual_normalization["brine_mass"]
                )

        if _eqn + 1 != self._NUM_EQN:
            raise ValueError("The number of residuals calculated is not equal"
                             +" to the number of equations/variables! This is"
                             +" a developer bug!")

        self._residual_iter += 1

        if np.isnan(residuals).sum() > 0.0:
            ind_nan = np.where(np.isnan(residuals) == True)[0]
            raise ValueError(f"The following salt cavern equation number (starting from 0) produced NaN! {ind_nan}")
            
        # deinitialize states.
        fluid.del_state()
        fluid_m1.del_state()

        return residuals

    def _analytics(self):
        """
        Make sure results make sense for a salt cavern

        """

        if isinstance(self._mdot_cavern,float):
            gas_mass_flow = self._mdot_cavern
        else:
            gas_mass_flow = self._mdot_cavern.sum()

        # total mass flow
        total_mass_flow = gas_mass_flow + self._mdot_vapor


        check_values = [self._t_cavern,
                        self._p_cavern,
                        total_mass_flow.sum()]

        for (bdict,check_value, variable,
             beyond_point, change, consequence) in zip(self._model.bounds_checks,
                                                       check_values,
                                                       AnalyticsDB.variables,
                                                       AnalyticsDB.beyond_points,
                                                       AnalyticsDB.changes,
                                                       AnalyticsDB.consequences):

            for warn_level, bounds in bdict.items():
                if warn_level == "error":
                    raise_error = True
                else:
                    raise_error = False

                if isinstance(bounds,float):

                    if check_value > bounds:
                        warn_str = (f"{warn_level}: The cavern {variable} (={check_value}) is {beyond_point}"
                        +f" beyond its operational upper limit of ({bounds})"
                        +f". You can change the {change}"
                        +" to ammend this condition but do so carefully."
                        +" Many failures can occur that UH2SC "
                        +f"will not capture including: {consequence}")
                        if raise_error:
                            raise CavernStateOutOfOperationalBounds(warn_str)
                        else:
                            self._model.logging.warning(warn_str)
                else:
                    if check_value < bounds[0] or check_value > bounds[1]:
                        warn_str = (f"{warn_level}: The cavern {variable} (={check_value}) is {beyond_point}"
                        +f" beyond its operational limits of ({bounds})"
                        +f". You can change the {change}"
                        +" to ammend this condition but do so carefully."
                        +" Many failures can occur that UH2SC "
                        +f"will not capture including: {consequence}")
                        if raise_error:
                            raise CavernStateOutOfOperationalBounds(warn_str)
                        else:
                            self._model.logging.warning(warn_str)

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
        Interface variable indices for the previous component (wells in this case)
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
    def independent_vars_descriptions(self):
        """
        You must keep this ordered list the same as what evaluate_residuals
        returns when  get_independent_vars=True

        """
        return ["Brine volume change (m3)",
                "Water vapor mass in cavern (kg)",
                "Cavern volume (m3)",
                "Brine volume (m3)",
                "Brine average pressure (Pa)",
                "Water vapor mass flow in (+) out (-) of cavern (kg/s)",
                "Total cavern pressure (gasses + water vapor) (Pa)",
                "Energy of brine (J)",
                "Energy of cavern (J)",
                "Cavern average gas pressure excluding water vapor (Pa)"]


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

        if self._model.is_test_mode:
            e_list += ["Heat flux continuity between cavern and axisymmetric heat transfer"]


        if not self._model.is_test_mode:
            for idx, t_axisym in enumerate(self._t_axisym):
                e_list += [f"Temperature continuity between cavern and axisymmetric heat transfer {idx}"]

        e_list += ["Cavern energy balance"]
        e_list += ["Brine energy balance"]
        e_list += ["Brine mass balance"]

        if len(e_list) > self._NUM_EQN:
            raise ValueError("Developer Error, the number of equation "
                             +"descriptions must be equal to the number of "
                             +"equations!")

        return e_list







    def get_x(self):
        return np.array([[self._t_cavern] +
                         [self._t_cavern_wall] +
                         [self._m_brine] +
                         [self._t_brine] +
                         list(self._m_cavern)])


    def shift_solution(self):
        # shift independent variables
        self._t_cavern_m1 = self._t_cavern
        self._t_cavern_wall_m1 = self._t_cavern_wall
        self._m_brine_m1 = self._m_brine
        self._t_brine_m1 = self._t_brine
        self._m_cavern_m1 = self._m_cavern



        (del_vol_brine,mass_vapor,
                vol_cavern,
                vol_brine,
                p_brine,
                cavern_vapor_mass_flow,
                p_cavern,
                e_brine,
                e_cavern,
                p_cavern_novapor) = self.evaluate_residuals(get_independent_vars=True)

        self._p_brine_m1 = p_brine

        # shift the composition of the gas fluid
        total_mass = self._m_cavern.sum()
        self._fluid_m1.set_mass_fractions(self._m_cavern/total_mass)
        rho_cavern = total_mass / self._vol_cavern

        self._p_cavern_m1 = p_cavern
        
        
        #self._water_m1.update(CP.PT_INPUTS,p_brine,self._t_brine)

        self._vol_cavern_m1 = self._VOL_TOTAL - vol_brine - del_vol_brine
        self._vol_brine_m1 = self._VOL_TOTAL - self._vol_cavern_m1
        self._e_cavern = e_cavern
        self._e_brine = e_brine

        # make sure pressure, temperature, and mass flow are within operational
        # limits set by the user or by default reasonable values.
        self._analytics()



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
        #self._t_brine_wall = xloc[4]
        self._m_cavern = np.array([mass for mass in xloc[4:4+self._number_fluids]])


        # see model._build_ghes for the variable structure of each
        # axisymmetric ground heat exchanger.
        if hasattr(self,"_next_components"):
            self._q_axisym = []
            self._t_axisym = []

            for cname, comp in self._next_components.items():
                if isinstance(comp, ImplicitEulerAxisymmetricRadialHeatTransfer):
                    self._q_axisym.append(xg[comp.global_indices[0]]) # first place is the wall fliux
                    self._t_axisym.append(xg[comp.global_indices[0]+1]) # second place is the first temperature
                else:
                    raise NotImplementedError("Only Axisymmetric heat transfer "
                                              +"can be a next component for salt"
                                              +" caverns currently!")


        self._set_wells_vars(xg)


    def _set_wells_vars(self,xg):
        """


        """

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
                        if vnum > 1:
                            raise NotImplementedError("The current code only supports single pipe wells!")
                    # flux into axisymetric
                        vname = pipe.valve_name
                        self._wells_mdot[cname][vname] = xg[gind[0]*vnum:gind[0]*vnum+numf]
                        self._wells_temp[cname][vname] = xg[gind[0]*vnum+numf+2]
                        self._wells_pres[cname][vname] = xg[gind[0]*vnum+numf+3]
                        vnum+=1

                else:
                    raise NotImplementedError("Only Wells "
                                              +"can be a previous component for "
                                              +"salt caverns currently!")


    def cavern_wall_heat_flux(self,t_cavern,t_cavern_wall,t_brine,
                              fluid,water,height_cavern,height_brine,height_total):
        ht_coef_wall = self._wall_ht_coef(height_cavern,fluid,t_cavern,t_cavern_wall,self._p_cavern)
        # TODO, you are just using water, you need it to be brine! maybe you should
        # write a class that inherits from the CoolProp Abstract state but incorporates
        # the brine properties when needed.
        ht_coef_brine_wall = self._wall_ht_coef(height_brine,water,t_brine,t_cavern_wall,self._p_brine)

        # same as above
        q_cavern_wall = ((self._area_horizontal + self._area_vertical *
                          (height_cavern/
                           height_total))
            * ht_coef_wall
            * (t_cavern_wall - t_cavern))



        q_brine_wall = ((self._area_horizontal + self._area_vertical *
                               height_brine / height_total)
                               * ht_coef_brine_wall
                               * (t_cavern_wall - t_brine))



        return q_cavern_wall, q_brine_wall

    def _wall_ht_coef(self, length,cfluid,temp,temp_wall,pressure):
        """
        Calculation of internal natural convective heat transfer coefficient from Nusselt number
        and using the coolprop low level interface.
        """

        if self._ht_coef == "calc":
            t_film = (temp+temp_wall)/2
            t_restore = cfluid.T()
            p_restore = cfluid.p()
            cfluid.update(CP.PT_INPUTS, pressure, t_film)
            cond = cfluid.conductivity()
            visc = cfluid.viscosity()
            cp = cfluid.cpmass()
            prandtl = cp * visc / cond
            beta = cfluid.isobaric_expansion_coefficient()
            nu = visc / cfluid.rhomass()
            grashoff = const.g['value'] * beta * abs(temp_wall - temp) * length ** 3 / nu ** 2
            rayleigh = prandtl * grashoff
            nusselt = natural_convection_nu_vertical(rayleigh, prandtl)
            ht_coef = nusselt * cond / length


            cfluid.update(CP.PT_INPUTS,p_restore,t_restore)

        else:
            ht_coef = self._ht_coef

        # # for numerical conditioning purposes
        # if ht_coef < 0.0001:
        #     ht_coef = 0.0001

        return ht_coef


    def _brine_cavern_ht_coef(self, radius, cfluid):
        # TODO - put actual values!!
        return 5.0


    def _radiation_vf(self, cavern_diameter, cavern_height):
        _r = cavern_diameter / 2.0
        _rho = (np.sqrt(4.0 * _r**2 + 1) - 1)/_r
        return _rho / (2.0 * _r)


    def _mass_flux_and_mass_energy_flux(self,fluid,hmass,p_cavern,t_cavern,rho_vapor,rho_cavern,h_vapor):
        # rho_cavern includes rho_vapor
        """
        inputs:
            
            fluid : is the cavern gas fluid
            
            hmass : is the cavern enthalpy
            
            p_cavern : cavern pressure
            
            t_cavern : cavern temperature
            
            rho_vapor : density of non-interacting water vapor (i.e., saturated vapor state)
            
            rho_cavern : density of the gas in the cavern
            
            h_vapor : energy content of the water vapor.
        
        """

        gas_mass_fraction = rho_cavern/(rho_cavern + rho_vapor)
        water_vapor_mass_fraction = rho_vapor / (rho_cavern + rho_vapor)

        if self._model.is_test_mode:
            v_mdot = np.interp(self._model.time,
                               self._model.test_inputs["time"],
                               self._model.test_inputs["mdot"])
            fluid.update(CP.PT_INPUTS,
                         p_cavern,
                         t_cavern)

            if len( fluid.fluid_names()) > 1:
                
                # WHEN MULTIPLE WELLS ARE CREATED YOU HAVE TO UPDATE THIS
                # HERE IS THE MARKER "MULTI_WELL_CHANGE_NEEDED"
                num_well = 0
                for fluid_name, wfluid in self._model.fluids.items():
                    if fluid_name != "cavern":
                        # only other fluid is the well fluid
                        rfluid = wfluid
                        num_well += 1
                if num_well > 1:
                    raise NotImplementedError("The input must have two wells in it."
                                         +" The UH2SC code is not yet set up "+
                                         "to handle this")

                rfluid.set_state(CP.AbstractState,[p_cavern,t_cavern])
                rhmass = rfluid.hmass()
            else:
                rfluid = fluid
                rhmass = hmass

            if v_mdot < 0.0:
                cavern_gas_mass_flow = v_mdot * gas_mass_fraction * np.array(fluid.get_mass_fractions())
                cavern_vapor_mass_flow = v_mdot * water_vapor_mass_fraction
            else:
                cavern_gas_mass_flow = v_mdot * np.array(rfluid.get_mass_fractions())
                cavern_vapor_mass_flow = 0.0

            if v_mdot < 0.0:
                cavern_mass_energy_flow = hmass * v_mdot
                cavern_vapor_energy_flow = h_vapor * cavern_vapor_mass_flow
            else:
                cavern_mass_energy_flow = rhmass * v_mdot
                cavern_vapor_energy_flow = 0.0
                
            if rfluid != fluid:
                rfluid.del_state()

        else:
            cavern_gas_mass_flow = np.zeros(self._number_fluids)
            cavern_mass_energy_flow = 0.0
            cavern_vapor_mass_flow = 0.0
            cavern_vapor_energy_flow = 0.0
            for wname, well_mdot in self._wells_mdot.items():
                for vname, v_mdot in well_mdot.items():

                    wfluid = self._model.fluids[wname][vname]
                    wfluid.set_state(CP.AbstractState)

                    if v_mdot.sum() > 0:
                        cavern_gas_mass_flow += v_mdot
                        cavern_vapor_mass_flow = 0.0
                    else:
                        v_mdot_out = v_mdot.sum() * np.array(fluid.get_mass_fractions())
                        # but water vapor is going to go out too!
                        cavern_gas_mass_flow += v_mdot_out * gas_mass_fraction
                        cavern_vapor_mass_flow += v_mdot_out.sum() * water_vapor_mass_fraction

                    wfluid.update(CP.PT_INPUTS,
                                 self._wells_pres[wname][vname],
                                 self._wells_temp[wname][vname])


                    if v_mdot.sum() > 0.0:
                        cavern_mass_energy_flow += v_mdot.sum() * wfluid.hmass()
                        cavern_vapor_energy_flow += 0.0   #TODO - allow water vapor below saturation pressure to be present.
                    else:
                        cavern_mass_energy_flow += cavern_gas_mass_flow * hmass
                        cavern_vapor_energy_flow += cavern_vapor_mass_flow * h_vapor
                        
                    wfluid.del_state()

        return (cavern_gas_mass_flow, cavern_mass_energy_flow,
                cavern_vapor_mass_flow, cavern_vapor_energy_flow)


    def _simplified_axisymmetric_heat_flux(self,t_cavern_wall,height_cavern):

        if self._model.is_test_mode:
            #q_axi_cavern = 0.0
            q_axi_cavern = ((-t_cavern_wall
                            + self._model.test_inputs["farfield_temp"])
                           /self._model.test_inputs["r_radial"])
        else:
            raise ValueError("The _simplified_axisymetric_heat_flux is only "
                             +"meant for test mode!")

        return q_axi_cavern
    
    def _sum_or_float(self,val):
        if isinstance(val,np.ndarray):
            return val.sum()
        else:
            return val
