# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:24:33 2024

@author: dlvilla
"""

import numpy as np
from CoolProp import CoolProp as CP
from uh2sc.constants import Constants as const
from uh2sc.transport import annular_nusselt, circular_nusselt
from warnings import warn

class Well(object):
    
    def __init__(self,well_dict,P0,T0,molefracs,comp):
        
        """
        CP - cool props object
        """
        self.input = well_dict
        
        if not "ideal_pipes" in well_dict:
            self.input["ideal_pipes"] = False
        
        pipe_materials = [PipeMaterial(surf_rough_h,therm_cond) 
                          for surf_rough_h, therm_cond 
                          in zip(well_dict["pipe_roughness_ratios"],
                                 well_dict["pipe_thermal_conductivities"])]
        
        num_cv = [np.ceil(length/well_dict["control_volume_length"]) 
                       for length in well_dict["pipe_lengths"]]
        pidx = range(len(pipe_materials))
        
        Pres_dict = {}
        Tres_dict = {}
        molefracs_dict = {}
        comp_dict = {}
        for vname, valve in well_dict["valves"].items():
            if "reservoir" in valve:
                Pres_dict[vname] = valve["reservoir"]["pressure"]
                Tres_dict[vname] = valve["reservoir"]["temperature"]
                
                comp_f, molefracs_f, compSRK_f = process_CP_gas_string(valve["reservoir"]["fluid"])
                
            else:
                Pres_dict[vname] = P0
                Tres_dict[vname] = T0
                comp_f = comp
                molefracs_f = molefracs
                
                
            molefracs_dict[vname] = molefracs_f
            comp_dict[vname] = comp_f
        
        # establish individual pipes
        # TODO - make the height and length be able to be different 
        # (for non-straight or angled pipes)
        pipes = {valve_name:VerticalPipe(
                              molefracs,
                              comp,
                              length,
                              length,
                              pipe_mat,
                              well_dict["valves"][valve_name],
                              valve_name,
                              P0,
                              T0,
                              well_dict["pipe_diameters"][3*idx],
                              well_dict["pipe_diameters"][3*idx+1],
                              well_dict["pipe_diameters"][3*idx+2],
                              well_dict["pipe_diameters"][3*idx+3],
                              num_cv,
                              min_loss
                              ) for length, pipe_mat, num_cv, idx, min_loss, valve_name 
                                 in zip(well_dict["pipe_lengths"],
                                   pipe_materials,
                                   num_cv,
                                   pidx,
                                   well_dict["pipe_total_minor_loss_coefficients"],
                                   well_dict["valves"].keys())}
        
        # establish connectivity between pipes.
        if len(pipes)==1:
            self.pipes = pipes
        else:    
            for idx,pipe in enumerate(pipes):
                if idx==0:
                    if pipe.diameters[0] != 0.0 or pipe.diameters[1] != 0.0:
                        raise NotImplementedError("The first pipe must be "+
                                                  "circular with inner "+
                                                  "outside and inner inside "+
                                                  "diameters of zero!")
                    pipe.set_adjacent_pipes(adj_inner_pipe=None,adj_outer_pipe=pipes[idx+1])
                elif idx==len(pipes):
                    pipe.set_adjacent_pipes(adj_inner_pipe=pipes[idx-1],adj_outer_pipe=None)
                else:
                    pipe.set_adjacent_pipes(adj_inner_pipe=pipes[idx-1],adj_outer_pipe=pipes[idx+1])
            

    def step(self,i,P_cavern,cavern_obj,wname):
        
        # the fluid arrives at the exact pressure the cavern is at
        # for ideal pipes. Energy changes are not quantified.
        if self.input["ideal_pipes"]:
            for pipe in self.pipes.items():
               # follow the exact pressure 
               pipe.P_fluid[-1] = P_cavern
        else:
            for pname, pipe in self.pipes.items():
                pipe.step(cavern_obj,wname,i)
        
        
    
class VerticalPipe(object):
    
    """
    convention is position 0 is the node at the surface (i.e. connected to the valve)
    and that -1 (the last node) is the node connected to the salt cavern.
    
    TODO - Significant work in validation for actual salt caverns is greatly needed.
    The complexity of the presence of liquid water (as droplets and running up/down pipe walls)
    Is neglected here. Also particulate is neglected. (see Ganat et. al., 2018 
                                                       Gas-Liquid Two-Phase Upward Flow
                                                       Through a Vertical Pipe... Energies 
                                                       https://doi.org/10.3390/en11112937 )
    
    The multi-species gas constitutive equations also may be oversimplified.
    
    
    
    """
    
    def __init__(self,
                 molefracs,
                 comp,
                 length,
                 height_change,
                 pipe_material,  
                 valve,
                 valve_name,
                 initial_pressure,
                 initial_temperature,  
                 outside_inner_diameter,  
                 inside_inner_diameter,                 
                 inside_outer_diameter,
                 outside_outer_diameter,
                 number_control_volumes,
                 total_minor_losses_coefficient,
                 ):
        """
        Inputs
        ------
        mass_rate0 : float : 
            The mass rate (kg/s) at which fluid is flowing into the cavern (+)
            or out of the cavern (-). If flowing into, then intial_temperature
            indicates the temperature at the surface, if flowing out of, the 
            initial_temperature indicates the cavern temperature.
        
        """
        # input checking.
        if height_change > length:
            raise ValueError("The height change must be equal to or less than the pipe length")
        if not isinstance(comp, str):
            raise TypeError("The 'comp' input must be a string of valid"+
                            " CoolProp fluid names with '&' between them!")
        
        
        if len(comp.split("&")) != len(molefracs):
            raise ValueError("The 'molefracs' input must have the same "+
                             "number of entries as indicated in the "+
                             "'comp' input string!")
        
        self.height_change = height_change
        self.length = length
        self.len_p_cv = length / number_control_volumes
        self.material = pipe_material
        self.diameters = [outside_inner_diameter,
                          inside_inner_diameter,
                          inside_outer_diameter,
                          outside_outer_diameter]
        self.num_cv = int(number_control_volumes)
        self.min_loss = total_minor_losses_coefficient
        self.D_h, self.area = self._hydraulic_diameter()
        self.fluid = CP.AbstractState("HEOS", comp)
        self.fluid.specify_phase(CP.iphase_gas)

        self.fluid.set_mole_fractions(molefracs)

        
        self.fluid.update(CP.PT_INPUTS, initial_pressure,  
                                        initial_temperature)
            
        self.molefracs = molefracs
        self.comp = comp
        self.valve = valve
        self.valve_name = valve_name
        self.L_cv = length / number_control_volumes
        self.L_cv_o2 = self.L_cv / 2
        
        # more input checking
        if valve['type'] != "mdot":
            raise NotImplementedError("Only mdot valves are currently developed")
        elif valve['time'][0] != 0.0:
            raise ValueError("mdot valves must have a time value of 0.0 for all start times!")
        else:
            mass_rate0 = valve['mdot'][0]
        
        # state variables
        num_cv_p_1 = int(number_control_volumes + 1)
        self.num_cv = int(number_control_volumes)
        self.T_fluid, self.P_fluid = self._initial_adiabatic_static_column(initial_temperature,
                                                            initial_pressure,
                                                            mass_rate0)
        self.mass_rate = mass_rate0 * np.ones(num_cv_p_1)
        
        # TODO, mass_loss is a place holder. it is not developed.
        self.mass_loss = np.zeros(int(number_control_volumes))
        
        self.depths = [idx*self.L_cv for idx in range(num_cv_p_1)]
        self.adjacent_pipes = {"inner":None,"outer":None}
        self.results = {"T_fluid":[],
                        "P_fluid":[],
                        "T_wall":[],
                        "mass_loss":[],
                        "molefracs":[],
                        "mass_rate":[]}
        # If this is not an annulus, but just a simple pipe, then R_pipe_inner is not needed 
        if self.diameters[0] == 0.0:
            self.R_pipe_innter = np.nan
        else:
            self.R_pipe_inner = np.log(self.diameters[1]/self.diameters[0])/(2.0*np.pi 
                            * self.L_cv * self.material.kp)
        self.R_pipe_outer = np.log(self.diameters[3]/self.diameters[2])/(2.0*np.pi 
                            * self.L_cv * self.material.kp)  
        
    def _verify_adjacent_pipes_aligned(self,adj_inner_pipe,adj_outer_pipe): 
        outer_aligned = True
        inner_aligned = True        
        if adj_inner_pipe is None:
            outer_aligned = adj_outer_pipe.diameters[0] != self.diameters[-1]
        if adj_outer_pipe is None:
            inner_aligned = adj_inner_pipe.diameters[-1] !=  self.diameters[0]
        if not (outer_aligned and inner_aligned):
            raise ValueError("Adjacent pipes must have equal outermost and innermost diameters!")
    
    def set_adjacent_pipes(self,adj_inner_pipe=None,adj_outer_pipe=None):
        
        """
        Sets adjacent vertical pipes (if any) within a well so that there
        is thermal interaction established between the pipes.
        
        adj_inner_pipe should be None for a circular pipe that is innermost
        
        adj_outer_pipe should be None for a pipe whose outer surface is in
        contact with the ground.
        
        Inputs
        ------
        
        adj_inner_pipe : Optional : VerticalPipe : None
            Must be a VerticalPipe with outside outer diameter equal to this
            Vertical pipe's outside inner diameter.
            
        adj_outer_pipe : Optional : VerticalPipe : None
            Must be a VerticalPipe with outside inner diameter equal to this
            Vertical pipe's outside outer diameter.
            
        Sets
        ----
        
        self.adjacent_pipes: dict : 
            Dictionary with key =
            "inner" - the pipe that is adjacent to the inner outside diameter of 
                      this pipe.
            "outer" - the pipe that is adjacent to the outer outside diameter of
                      this pipe.
        
        """
        
        if adj_inner_pipe is None and adj_outer_pipe is None:
            return
        if (not adj_inner_pipe is None) and (not isinstance(adj_inner_pipe,VeriticalPipe)):
            raise TypeError("adj_inner_pipe must be a VerticalPipe ojbect!")
        if (not adj_outer_pipe is None) and (not isinstance(adj_outer_pipe,VeriticalPipe)):
            raise TypeError("adj_outer_pipe must be a VerticalPipe ojbect!")    
        
        # assure alignment
        self._verify_adjacent_pipes_aligned(adj_inner_pipe, adj_outer_pipe)
        
        #set the needed objects
        self.adjacent_pipes = {"inner": adj_inner_pipe,"outer":adj_outer_pipe}
            
            
            
        
        
    def _hydraulic_diameter(self):
        
        if self.diameters[0] == 0.0 and self.diameters[1] == 0.0:
            area_times_4 = np.pi * self.diameters[2]**2
            perimeter = np.pi * self.diameters[2]
        else:
            area_times_4 = np.pi * (self.diameters[2] ** 2 - self.diameters[1] ** 2)
            perimeter = np.pi * (self.diameters[2] + self.diameters[1])
        
        return area_times_4 / perimeter, area_times_4/4.0

    def step(self,cavern,wname,i):
        """
        The pipe equations can be solved by substition. if "flow_from_surface_to_cavern"
        is true, then the first temperature and pressure are the reservoir temperature 
        and pressure 
        """
        if self.valve['type'] != "mdot":
            raise NotImplementedError("Only mdot valves have been implemented in this version")

        # boundary condition for mass flow rate through valve.
        time = cavern.tstep * i
        
        mass_rate0 = np.interp(time,self.valve['time'],self.valve['mdot'])
        
        
        if mass_rate0 > 0:  # inflow into the cavern for this pipe
            flow_from_surface_to_cavern = True
        else:
            flow_from_surface_to_cavern = False
        
        # defines self.mass_rate based on the current defined rate of mass flow
        self._mass_balance(mass_rate0)
        
        # calculate all of the mass rates TODO - this assumes that the pipe
        # has an mdot valve (i.e. mass_rate directly defined, in the future,
        # this will not necessarily be the case, each valve type will produce
        # a mass rate based on the P1,T1)
        if flow_from_surface_to_cavern:
            self.P_fluid[0] = self.valve['reservoir']['pressure']
            self.T_fluid[0] = self.valve['reservoir']['temperature']
            # these factors (fac0, fac1) enable the code to either start from
            # the surface and work down (injection) or start from the cavern
            # and work up (production)
            fac0 = 0
            fac1 = 1 
        else:    
            self.P_fluid[-1] = cavern.P_cavern[i]
            self.T_fluid[-1] = cavern.T_cavern[i]
            fac0 = 1 
            fac1 = -1
            
        for cvn in range(self.num_cv):
            # either count up or count down depending on mass flow direction
            adj_cvn = int(fac0 * self.num_cv + fac1 * cvn)
            adj_cvn_pm1 = int(adj_cvn+fac1)
            
            # pressure loss and adiabatic expansion or contraction
            state1 = (self.P_fluid[adj_cvn],self.T_fluid[adj_cvn],self.mass_rate[adj_cvn],self.depths[adj_cvn])
            state2 = (self.P_fluid[adj_cvn_pm1],self.T_fluid[adj_cvn_pm1],self.mass_rate[adj_cvn_pm1],self.depths[adj_cvn_pm1])
            delP, delT = self._pressure_loss_and_adiabatic_expcomp(state1,state2,self.mass_loss[adj_cvn_pm1])
            
            self.P_fluid[adj_cvn_pm1] = self.P_fluid[adj_cvn] + delP
            self.T_fluid[adj_cvn_pm1] = self.T_fluid[adj_cvn] + delT
            

            # this is where we have left off.
            #next heat gains/losses to the surroundings.
            
            #delT2 = self._heat_loss()
            
            
            # now that pressures are updated, solve the energy balance so that 
            
            
        
    def _mass_balance(self,mass_rate0):
        
        if self.valve['type'] != 'mdot':
            raise NotImplementedError("The current code only handles 'mdot' valves")
        else:
            self.mass_rate[0] = mass_rate0
            
            for cvn in range(1,self.num_cv+1):
                self.mass_rate[cvn] = self.mass_rate[cvn-1] - self.mass_loss[cvn-1]
                
    def _heat_loss(self,):
        pass
    
    
        #Tbar_next_pipe_inner = self.
    
    
    def _energy_balance(self,state1,state2,mdot_loss):
        """
        
        
        """
        P1, T1, mdot1, height1 = state1
        P2, T2, mdot2, height2 = state2
        
        _, rho1, h1, v1, mu1, Re1, cp1 = self._get_variables(Pbar, Tbar, mdotbar)
        
        (friction_factor, density, enthalpy, velocity, viscosity, 
                reynolds_number, specific_heat)
         
    def _initial_adiabatic_static_column(self,initial_temperature,initial_pressure,mass_rate0,return_rho=False):
        """
        Provides an initial condition of a static adiabatic column for the Vertical
        Pipe which is a fairly good approximation of the process involved.
        
        It is assumed that there is only a single fluid type in the pipe 
        at this time.
        
        Returns
        -------
        None.

        """
        T_fluid = np.zeros(self.num_cv+1)
        P_fluid = np.zeros(self.num_cv+1)

        if return_rho:
            rho_fluid = np.zeros(self.num_cv+1)
    
        delh = self.len_p_cv
        grav = const.g['value']
    
        if mass_rate0 > 0:
            flow_into_cavern = True
            cv_n = 0
            sign = 1 
        else:
            flow_into_cavern = False
            cv_n = self.num_cv
            sign = -1
        
        T_fluid[cv_n] = initial_temperature
        P_fluid[cv_n] = initial_pressure
        
        fluid = self.fluid
        
        for idx in range(self.num_cv):
            
            if flow_into_cavern:
                # counting up if we started at the surface 
                cv_n_r1 = cv_n
                cv_n += 1
                
            else:
                # counting down if we started at the cavern
                cv_n_r1 = cv_n
                cv_n -= 1
            
            fluid.update(CP.PT_INPUTS, P_fluid[cv_n_r1],
                                             T_fluid[cv_n_r1])

            if return_rho:
                rho_fluid[cv_n] = fluid.rhomass()
            R_gas = fluid.gas_constant() / fluid.molar_mass()
            Z = fluid.compressibility_factor()
            gamma = fluid.cpmass()/fluid.cvmass()
            
            
            delP = P_fluid[cv_n_r1] * (1 - np.exp(-grav*delh/(R_gas * T_fluid[cv_n_r1] * Z)))
            
            delT = (1 - 1/gamma)* grav/(Z * R_gas) * delh
            

                
            # must increase or decrease depending on solution method.
            P_fluid[cv_n] = P_fluid[cv_n_r1] + sign * delP
            T_fluid[cv_n] = T_fluid[cv_n_r1] + sign * delT
        
        if return_rho:
            fluid.update(CP.PT_INPUTS, P_fluid[cv_n],
                                         T_fluid[cv_n])
            rho_fluid[cv_n] = fluid.rhomass()
            
            return T_fluid,P_fluid,rho_fluid
        else:
            return T_fluid,P_fluid
        
        


    def _average_velocity(self,mdot,density):
        return mdot / density / self.area
    
    def _viscous_pressue_loss(self,friction_factor,length,velocity, min_loss):
        return ((friction_factor * length / self.D_h + min_loss) * velocity ** 2 
                / (2 * const.g["value"]))
    
    def _pressure_loss_and_adiabatic_expcomp(self,state1,state2,mdot_loss):
        """
        Each control volume evaluates at two points in anticipation of 
        including mass loss due to leaking at the "mid point" of the control 
        volume.
        
        state 1 is the entering point (upper point if injection, 
                                       lower point if production)
        
        """
        P1, T1, mdot1, height1 = state1
        P2, T2, mdot2, height2 = state2

        if mdot1 >= 0:
            bouyant_sign = 1.0
        else:
            bouyant_sign = -1.0
        
        # this check assures the function is not used out of context 
        # mass must be conserved
        # 
        if np.abs(mdot1 - mdot_loss - mdot2) >= const.num_conv['value']:
            raise ValueError("Mass must be conserved!")
        
        Pbar = (P1 + P2)/2
        Tbar = (T1 + T2)/2
        mdotbar = (mdot1 + mdot2)/2
        ffbar, rhobar, hbar, vbar, _, _, _ = self._get_variables(Pbar, Tbar, mdotbar)  
       
        #(friction_factor, density, enthalpy, velocity, viscosity, 
        #        reynolds_number, specific_heat)
        if mdot_loss != 0:
            # THIS IS NOT YET IMPLEMENTED
            raise NotImplementedError("This feature has not yet beeen implemented"+
                                 " and the following code must be tested/"+
                                 "validated")
            P1bar = (P1 + Pbar)/2
            T1bar = (T1 + Tbar)/2
            ff1bar, rho1bar, h1bar, v1, _, _, _ = self._get_variables(P1bar, T1bar, mdot1)
            
            P2bar = (P2 + Pbar)/2
            T2bar = (T2 + Tbar)/2
            ff2bar, rho2bar, h2bar, v2, _, _, _ = self._get_variables(P2bar, T2bar, mdot2)

            
            delP1bar_viscous = self._viscous_pressue_loss(ff1bar, self.L_cv_o2,
                                                          v1, 0.5 * self.min_loss) 
            delP2bar_viscous = self._viscous_pressue_loss(ff2bar, self.L_cv_o2,
                                                          v2, 0.5 * self.min_loss)
            delP1bar_kinetic = ((vbar + v1)/2)*(v1 - vbar) * rho1bar
            delP2bar_kinetic = ((vbar + v2)/2)*(vbar - v2) * rho2bar
            delP1bar_bouyant = bouyant_sign * const.g["value"] * rho1bar * self.L_cv_o2
            delP2bar_bouyant = bouyant_sign * const.g["value"] * rho2bar * self.L_cv_o2
            
            delP = (delP1bar_viscous + delP2bar_viscous + 
                    delP1bar_kinetic + delP2bar_kinetic +
                    delP1bar_bouyant + delP2bar_bouyant)
        else: # no changes to mass flow so the equation is simpler with
              # no kinetic term.

              
            delPbar_viscous = self._viscous_pressue_loss(ffbar, self.L_cv, vbar, self.min_loss)
            delPbar_bouyant = bouyant_sign * const.g["value"] * rhobar * self.L_cv
            
            delP = delPbar_viscous + delPbar_bouyant
            
        # now quantify adiabatic compression/expansion heating
        gamma = self.fluid.cpmass()/self.fluid.cvmass()
        
        #TODO, evaluate how much difference it makes to reevaluate properties at P1 + 0.5 * delP
        
        # Polytropic, adiabatic, reversible expansion - TODO - does cooprops have ways
        # to capture irreversibility of expansion compression cycles??? PErhaps a calculation
        # at the beginning of the simulation to quantify n over the range of pressures and volumes 
        # of interest would help??
        T2new = T1 * ((P1 + delP)/P1) ** ((gamma - 1)/gamma)
        delT = T2new - T1
        
        
        return delP, delT
    
    def _heat_loss(self,state1,state2,):
        
        
        Rp_out = self.R_pipe_outer
        
        
        # THIS IS MY LAST POINT I LEFT OFF ON. THE HEAT TRANSFER
        # COEFFICIENTS ARE VERIFIED TO BE ACCURATE!
        if ((self.diameter[0] == 0) and (self.diameter[1] == 0)):
            # heat transer unidirectional (out)
            Nu, h_out = circular_nusselt(mdot, _L_, _D_, fluid)
            
        else:
            Rp_in = self.R_pipe_inner
            Nu, h_in = annular_nusselt(mdot, _L_, Di, Do, Tb, Tw, True, fluid)
            Nu, h_out = annular_nusselt(mdot, _L_, Di, Do, Tb, Tw, False, fluid)
            
            pass
            # heat transfer bi-directional (in and out)
        #TODO - these are static values MOVE THEM to conductive axi-symmetric heat resistivities
      
        

        
        # h_film_outer = h_outside_pipe(density, 
        #                              velocity, 
        #                              hydraulic_diameter, 
        #                              dynamic_viscosity, 
        #                              specific_heat,
        #                              thermal_conductivity)
        
            
            
            
            

    
        
        
    def _get_variables(self,pressure,temperature,mdot):

        self.fluid.update(CP.PT_INPUTS,pressure,temperature)

        viscosity = self.fluid.viscosity()
        density = self.fluid.rhomass()
        enthalpy = self.fluid.hmass()
        specific_heat = self.fluid.cpmass()
        # average velocity in the pipe
        velocity = self._average_velocity(mdot,density)
        
        reynolds_number = density * velocity * self.D_h / viscosity
        
        friction_factor = self._darcy_weisback_friction_factor(reynolds_number)
        
        return (friction_factor, density, enthalpy, velocity, viscosity, 
                reynolds_number, specific_heat)
        
        
    
    
    def _darcy_weisback_friction_factor(self,Re):
        """
        Per equations 30-32 of ASHRAE fundamentals 2021 for Chapter 3 Fluid flow
        
        """
        if Re < 2000:
            # viscous flow
            return 64.0/Re
        elif Re >= 2000 and Re <= 10000:
            # transition region, behavior is unpredictable
            warn("The flow in pipe '" + self.valve_name + 
                 "' is in the transition region where Re >= 2000"+
                 " but Re <= 10000. Predictions will not be accurate! "+
                 "A value of 0.03 has been given for the friction coefficient",
                 UserWarning)
            
            return 0.03
        elif Re > 1e4:
            # turbulent
            AA = (2.457 * np.log(1/((7/Re) ** 0.9 +
                                (0.27 * self.relative_roughness)))) ** 16
            BB = (37530/Re) ** 16
            ff = 8 * ((8 / Re) ** 12 + 1 / (AA + BB) ** 1.5) ** (1/12)
            
            return ff


class PipeMaterial(object):
    
    def __init__(self, 
                 surface_roughness_height_ratio,
                 thermal_conductivity):
        self.kp = thermal_conductivity
        self.relative_rougness = surface_roughness_height_ratio
        
        pass
    
def process_CP_gas_string(matstr):
    # Detects if a multi component fluid is specified using & for separation of components
    if "&" in matstr:
        comp_frac_pair = [str.replace("["," ").replace("]","").split(" ") for str in  matstr.split("&")] 
        comp0 = [pair[0] for pair in comp_frac_pair]
        compSRK0 = [pair[0]+"-SRK" for pair in comp_frac_pair]
        molefracs0 = np.asarray([float(pair[1]) for pair in comp_frac_pair])
        molefracs = molefracs0 / sum(molefracs0)
        
        sep = "&"
        comp = sep.join(comp0)
        compSRK = sep.join(compSRK0)
    # Normally single component fluid is specified
    else:
        comp = matstr
        molefracs = [1.0]
        compSRK = matstr
        
    return comp, molefracs, compSRK
        