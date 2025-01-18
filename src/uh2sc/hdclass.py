# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license

import math
import numpy as np
from scipy.optimize import minimize
from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
from uh2sc import transport as tp
from uh2sc import validator
from uh2sc.well import Well 
from uh2sc.well import process_CP_gas_string
from uh2sc.errors import NumericAnomaly,MassTooLow,InputFileError
from uh2sc.constants import Constants
from uh2sc.utilities import filter_cpu_count


class ExplicitAxisymmetricRadialHeatTransfer(object):
    
    def __init__(self,r_cavern,kg,rhog,cpg,h_cavern,number_element,dist_nexT_cavern_wall,Tg,
                 dist_to_Tg_reservoir,Tgvec0=None):
        """
        Inputs: (all are float or int)
        -------
            r_cavern             = cavern radius
            kg                   = ground thermal conductivity
            rhog                 = ground density
            cpg                  = ground specific heat
            h_cavern             = height of the cavern
            number_element       = number of elements in the radial heat transfer model
            dist_nexT_cavern_wall     = axis distance between caverns
            Tg                   = average reservoir ground temperature far from
                                   the salt cavern (constant).
            dist_to_Tg_reservoir = distance from the heat transfer upper and 
                                   lower surfaces to a reservoir at Tg
            Tgvec0               = Initial condition temperatures. If not provided,
                                   then the entire ground is assumed to be at Tg.

        """

        self.number_element = number_element

        self.r_cavern = r_cavern

        self.h_cavern = h_cavern 
        
        self.solutions = {}
        self.time_step_num = 0
        self.time = 0.0
        # this is just for not having to reinitialize over and over.

        self.Tg = Tg
        
        self.Q = np.zeros(number_element+1)
        
        self.Qg = np.zeros(number_element)
        
        # initial conditions
        if Tgvec0 is None:
            self.Tgvec_m1 = Tg * np.ones(number_element+1)
        else:
            self.Tgvec_m1 = Tgvec0
        
        # Space finite volume elements on a log scale to allow a smooth capture of the
        # surface temperature gradient.
        rr = np.logspace(np.log10(r_cavern),np.log10(dist_nexT_cavern_wall/2.0),num=number_element+1)
        
        rrCg = [(r2+r1)/2 for r2,r1 in zip(rr[1:],rr[:-1])]
        rrCg.append(rr[-1])
        rrCg.insert(0,rr[0])
        
        # Radial Heat capacitance of rings of volume of ground
        self.Cg = [rhog * np.pi * (r2**2.0 - r1**2.0) * self.h_cavern * cpg for r2,r1 in zip(rrCg[1:],rrCg[:-1])]
        # Radial thermal resistivity of rings of volume of ground
        self.Rg = [np.log(rout/rin)/(2.0*np.pi * self.h_cavern * kg) for rout,rin in zip(rr[1:],rr[:-1])]
        self.Rg.append(self.Rg[-1]*1e4)  # This produces a zero flux boundary condition
        
        if isinstance(dist_to_Tg_reservoir,(float,int)):
            self.RTg = [dist_to_Tg_reservoir/(np.pi/4.0 * (rout**2 - rin**2) * kg) for rout,rin in zip(rrCg[1:],rrCg[:-1])]
        else:  # it is assumed that dist_to_Tg_reservoir is a list or array
            try:
                if len(dist_to_Tg_reservoir) != number_element:
                    raise ValueError("The dist_to_Tg_reservoir must be an array or list of equal length as the number of elements")
            except:
                raise TypeError("dist_to_Tg_reservoir must be a float, array, or list!")
                    
            self.RTg = [dist/(np.pi/4.0 * (rout**2 - rin**2) * kg) for rout,rin,dist in zip(
                rr[1:],rr[:-1],dist_to_Tg_reservoir)]
            
        self.grid = rr
        
        

    def _Qfunc(self,Tgvec_m1,idx):
        if idx >= self.number_element:
            return 0.0 # zero flux boundary condition
        else:
            return (-Tgvec_m1[idx+1] + Tgvec_m1[idx])/self.Rg[idx]

        
    def euler_equn(self,dt,Qsalt0):
        """
        Inputs:
            Variables: 
                
            Tgvec - vector of ground temperatures spaced radially from the GSHX

            Tgvec_m1 - Tgvec for the previous time step
        """
        self.time_step_num += 1 
        self.time = self.time + dt
        
        # initialize vectors
        Tgvec_m1 = self.Tgvec_m1
        Q = np.zeros(self.number_element+2)
        Tgvec = np.zeros(self.number_element+1)
        Qg = np.zeros(self.number_element+1)
        Q[0] = Qsalt0
        
        for idx in range(0,self.number_element+1):

            Qg[idx] = (Tgvec_m1[idx] - self.Tg)/self.RTg[idx]
            
            Q[idx+1] = self._Qfunc(Tgvec_m1,idx)
                
            #Euler integration of the ODE

            Tgvec[idx] = (Q[idx]-(Q[idx+1]+Qg[idx-1])) * dt / self.Cg[idx-1] + Tgvec_m1[idx] 
                                

            # keep track of total energy lost to ground temperature potential

        self.Q = Q
        self.Qg = Qg

        return Tgvec
    
    def step(self,dt,Qsalt0,reset=False):
            
        Tgvec = self.euler_equn(dt, Qsalt0)
        
        if not reset:
            self.solutions[self.time] = {"Q":self.Q,
                                                  "Tg":Tgvec,
                                                  "Qg":self.Qg}
            # use current condition for the next step.
            self.Tgvec_m1 = Tgvec
        
        # return the total heat flux out which is passed to hyddown.
        return Tgvec

class HydDown:
    """
    Main class to to hold problem definition, running problem, storing results etc.
    """
    def __init__(self, inp):
        """
        Parameters
        ----------
        inp : dict
            Dict holding problem definition
        """
        
        #TODO - move fault analytics option to the input structure
        self.include_fault_analytics = True
        self.anomaly_factor = 20
        self.input = inp
        self.verbose = 0
        self.isrun = False
        self.validate_input()
        self.read_input()
        self.initialize()
       
    def validate_input(self):
        """
        Validating the provided problem definition dict

        Raises
        ------
        ValueError
            If missing input is detected.
        """
        valid_tup = validator.validation(self.input)
        vobjs = valid_tup[1]
        valid_test = valid_tup[0]
        invalid = False
        error_string = ""
        for key,val in valid_test.items():            
            valid = val
            vobj = vobjs[key]
            if valid is False:
                #TODO - process this better so that the user knows the exact
                #       field (or first error) that needs correcting.
                error_string = error_string + "\n\n" + key + "\n\n" + str(vobj.errors)
                invalid = True
                
        if invalid:
            raise InputFileError("Error in input file:\n\n" + error_string)

    def read_input(self):
        """
        Reading in input/ problem definition dict and assigning to classs
        attributes.
        """
        self.length = self.input["cavern"]["height"]
        self.diameter = self.input["cavern"]["diameter"]

        self.p0 = self.input["initial"]["pressure"]
        self.T0 = self.input["initial"]["temperature"]
        self.Tv0 = self.input["cavern"]["salt_farfield_temperature"]
        self.species = "HEOS::" + self.input["initial"]["fluid"]

        # Detects if a multi component fluid is specified using & for separation of components
        comp, molefracs, compSRK = process_CP_gas_string(self.input["initial"]["fluid"])
        self.comp = comp
        self.molefracs = molefracs
        self.compSRK = compSRK
        
        
        self.tstep = self.input["calculation"]["time_step"]
        self.time_tot = self.input["calculation"]["end_time"]
        self.salt_thickness = self.input["cavern"]["salt_thickness"]
        self.h_in = self.input["heat_transfer"]["h_inner"]
        
        # Reading well-specific data
        self.wells = {}
        for wname, well in self.input["wells"].items():            
            self.wells[wname] = Well(well, self.p0, self.T0, self.molefracs, self.comp)
            
        self._prepare_for_parallel()
    
    def _prepare_for_parallel(self):
        
        if "run_parallel" in self.input["calculation"]:
            self._run_parallel = self.input["calculation"]["run_parallel"]
            if self._run_parallel:
                import multiprocessing as mp
                # MANUAL
                # we want 1 processor for each well 1 processor for radial heat transfer 
                # and 1 processor for calculating enthalpy
                num_static = 2 # 1 radial HT, 2) calculating enthalpy
                num_dynamic = len(self.wells)
                self._num_cpu = filter_cpu_count(num_static + num_dynamic)
                self._pool = mp.Pool(self._num_cpu)
                
                
                
                
        else:
            # we may change this default in the future
            self._run_parallel = False
                
    def initialize(self):
        """
        Preparing for running problem by creating the fluid objects required
        instantiating arrays for storing time-dependent results, setting additional 
        required class attributes.
        """
        inp = self.input
        
        self.time_tot = inp["calculation"]["end_time"]
        if not self.isrun:
            self.vol = self.diameter ** 2 / 4 * math.pi * self.length  # m3
            self.vol_tot = (
                (self.diameter + 2 * self.salt_thickness) ** 2
                / 4
                * math.pi
                * (self.length + 2 * self.salt_thickness)
            )  # m3
            self.vol_solid = self.vol_tot - self.vol
            self.surf_area_outer = (
                self.diameter + 2 * self.salt_thickness
            ) ** 2 / 4 * math.pi * 2 + (self.diameter + 2 * self.salt_thickness) * math.pi * (
                self.length + 2 * self.salt_thickness
            )
            self.surf_area_inner = (self.diameter) ** 2 / 4 * math.pi * 2 + (
                self.diameter
            ) * math.pi * self.length
    
            self.fluid = CP.AbstractState("HEOS", self.comp)
            self.fluid.specify_phase(CP.iphase_gas)
            self.fluid.set_mole_fractions(self.molefracs)
            self.fluid.update(CP.PT_INPUTS, self.p0,  self.T0)
            
            # cavern fluid with properties evaluated at the film temperature
            # (i.e. (T_cavern + T_cavern_wall)/2)
            self.transport_fluid = CP.AbstractState("HEOS", self.compSRK)
            self.transport_fluid.specify_phase(CP.iphase_gas)
            self.transport_fluid.set_mole_fractions(self.molefracs)
            
            # minimum mass
            referencePT_rho_H2 = 0.08527
            self.min_mass = self.vol * referencePT_rho_H2

            # add on for salT_cavern_wall
            self.axsym = ExplicitAxisymmetricRadialHeatTransfer(r_cavern=self.diameter/2.0,
                                                   kg=inp["cavern"]["salt_thermal_conductivity"],
                                                   rhog=inp["cavern"]["salt_density"],
                                                   cpg=inp["cavern"]["salt_heat_capacity"],
                                                   h_cavern=self.length,
                                                   number_element=inp["heat_transfer"]["number_radial_elements"],
                                                   dist_nexT_cavern_wall=self.diameter/2.0 + self.salt_thickness,
                                                   Tg=inp["cavern"]["salt_farfield_temperature"],
                                                   dist_to_Tg_reservoir=inp["cavern"]["distance_to_farfield_temp"])
            
            self.rho0 = self.fluid.rhomass() #PropsSI("D", "T", self.T0, "P", self.p0, self.species)
            self.m0 = self.rho0 * self.vol
            self.MW = self.fluid.molar_mass() #PropsSI("M", self.species)

        
        # data storage
        data_len = int(self.time_tot / self.tstep)
        self.rho = np.zeros(data_len)
        self.T_cavern = np.zeros(data_len)
        self.T_vent = np.zeros(data_len)
        self.T_cavern_wall = np.zeros(data_len)
        self.Q_outer = np.zeros(data_len)
        self.Q_inner = np.zeros(data_len)
        self.h_inside = np.zeros(data_len)
        self.H_mass = np.zeros(data_len)
        self.S_mass = np.zeros(data_len)
        self.U_mass = np.zeros(data_len)
        self.U_tot = np.zeros(data_len)
        self.U_res = np.zeros(data_len)
        self.P_cavern = np.zeros(data_len)
        self.mass_fluid = np.zeros(data_len)
        self.total_mass_rate = np.zeros(data_len)
        
        # this is mass rates coming directly into and out of the cavern
        # pipe leak rates may make the mass rate different at the valve.
        self.mass_rate = {}
        self.res_fluid = {}
        for wname, well in inp["wells"].items():
            
            # establish reservoir fluid properties
            self.res_fluid[wname] = {}
            self.mass_rate[wname] = {}
            
            for vname, valve in well["valves"].items():
                
                self.mass_rate[wname][vname] = np.zeros(data_len,float)
                
                if "reservoir" in valve:
                    T_res = valve["reservoir"]["temperature"]
                    P_res = valve["reservoir"]["pressure"]
                    self.res_fluid[wname][vname] = CP.AbstractState("HEOS", self.comp)
                    self.res_fluid[wname][vname].specify_phase(CP.iphase_gas)
                    self.res_fluid[wname][vname].set_mole_fractions(self.molefracs)
                    self.res_fluid[wname][vname].update(CP.PT_INPUTS, P_res,  T_res) 
        #initialize cavern variables
        self.H_mass[0] = self.fluid.hmass() 
        self.S_mass[0] = self.fluid.smass() 
        self.U_mass[0] = self.fluid.umass() 
        self.U_tot[0] = self.fluid.umass() * self.m0
        self.P_cavern[0] = self.p0
        self.mass_fluid[0] = self.m0
        self.time_array = [idx * self.tstep for idx in range(data_len)]


        




    def PHres(self, T, P, H):
        """
        Residual enthalpy function to be minimised during a PH-problem
        
        Parameters
        ----------
        H : float 
            Enthalpy at initial/final conditions
        P : float
            Pressure at final conditions. 
        T : float 
            Updated estimate for the final temperature at P,H
        """
        self.vent_fluid.update(CP.PT_INPUTS, P, T)
        return ((H-self.vent_fluid.hmass())/H)**2 

    def PHproblem(self, H, P, Tguess):
        """
        Defining a constant pressure, constant enthalpy problem i.e. typical adiabatic 
        problem like e.g. valve flow for the vented flow (during discharge). 
        For multicomponent mixture the final temperature is changed/optimised until the residual 
        enthalpy is near zero in an optimisation step. For single component fluid the coolprop 
        built in methods are used for speed. 
        
        Parameters
        ----------
        H : float 
            Enthalpy at initial/final conditions
        P : float
            Pressure at final conditions. 
        Tguess : float 
            Initial guess for the final temperature at P,H
        """

        # Multicomponent case
        if "&" in self.species:                     
            x0 = Tguess
            res = minimize(self.PHres, x0, args=(P, H), method='Nelder-Mead', options={'xatol':0.1,'fatol':0.001})
            T1 = res.x[0]
        # single component fluid case
        else:
            T1 = PropsSI(
                "T", "P", P, "H", H, self.species
               )  
        return T1

    def UDres(self, x, U, rho):
        """
        Residual U-rho to be minimised during a U-rho/UV-problem
        
        Parameters
        ----------
        U : float 
            Internal energy at final conditions
        rho : float
            Density at final conditions
        """
        self.fluid.update(CP.PT_INPUTS, x[0], x[1])
        return ((U-self.fluid.umass())/U)**2 + ((rho-self.fluid.rhomass())/rho)**2

    def UDproblem(self, U, rho, Pguess, Tguess):
        """
        Defining a constant UV problem i.e. constant internal energy and density/volume 
        problem relevant for the 1. law of thermodynamics. 
        For multicomponent mixture the final temperature/pressure is changed/optimised until the 
        residual U/rho is near zero. For single component fluid the coolprop 
        built in methods are used for speed. 
        
        Parameters
        ----------
        U : float 
            Internal energy at final conditions
        rho : float
            Density at final conditions. 
        Pguess : float 
            Initial guess for the final pressure at U, rho
        Tguess : float 
            Initial guess for the final temperature at U, rho
        """
        if "&" in self.species:                    
            x0 = [Pguess, Tguess]
            res = minimize(self.UDres, x0, args=(U, rho), method='Nelder-Mead', options={'xatol':0.1,'fatol':0.001})
            P1 = res.x[0]
            T1 = res.x[1]
            Ures = U-self.fluid.umass()
        else:
            P1 = PropsSI(
                "P", "D", rho, "U", U, self.species
                )
            T1 = PropsSI(
                "T", "D", rho, "U", U, self.species
               )
            Ures = 0 
        return P1, T1, Ures
    
    def _calc_h_in(self,i):
        if self.h_in == "calc":
            
            L = self.length
                
            T_film = (self.T_cavern[i - 1]+self.T_cavern_wall[i - 1])/2
            self.transport_fluid.update(CP.PT_INPUTS, self.P_cavern[i-1], T_film)
            
            if self.total_mass_rate[i] > 0.0: #if self.input["valve"]["flow"] == "filling":
               # TODO - REmOVE : this should be the same but I have it this way for direct verification 
               # that I have not made any changes.
               hi = tp.h_inside_mixed(L, self.T_cavern_wall[i-1], self.T_cavern[i-1], self.transport_fluid, self.total_mass_rate[i-1], self.diameter)
            else:
               # outflow does not cause significant forced mixing on the level of caverns
               hi = tp.h_inside(L, self.T_cavern_wall[i-1], self.T_cavern[i-1], self.transport_fluid)
        else:
            hi = self.h_in
        return hi

    def _total_mass_rate(self,i):
        val = np.array([[self.mass_rate[wname][vname][i] for vname, valve in well["valves"].items()] 
                for wname, well in self.input["wells"].items()]).sum()
        if isinstance(val,np.ndarray):
            val = val.sum()
        return val
        

    def run(self, disable_pbar=True,Tv0=None,Tf0=None):
        """
        Routine for running the actual problem defined i.e. integrating the mass and energy balances
        """
        # Inititialise / setting initial values for t=0

        self.initialize()
        inp = self.input
        
        # Enable setting initial conditions
        if not Tf0 is None:
            self.T_cavern[0] = Tf0
            self.fluid.update(CP.PT_INPUTS, self.p0,  self.Tf0)
            self.rho[0] = self.fluid.rhomass()
        else:
            self.T_cavern[0] = self.T0
            self.rho[0] = self.rho0
            
        if not Tv0 is None:
            self.T_cavern_wall[0] = Tv0
        else:
            self.T_cavern_wall[0] = self.Tv0

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
    
    def _model_fault_analytics(self,i):
        self._detect_numeric_anomaly(i)
        self._detect_unrealistic_mass(i)
    
    
    def _detect_unrealistic_mass(self,i):
        # the mass should never fall to the amount of atmospheric substance
      
        if self.mass_fluid[i] < self.min_mass:
            breakpoint()
            raise MassTooLow("The mass has fallen below what atmospheric H2"+
                             " would provide. Operating at such low pressure"+
                             " is not permitted in H2 Salt Cavern operations."+
                             " Please consider revising the inputs to avoidS"+
                             " this.")
        

        
    def _detect_numeric_anomaly(self,i):
        
        if i > 10:
            im = i - 10

            
            avg_derivative = np.mean(np.abs(np.diff(self.T_cavern[im:i-1])))
            cur_derivative = np.abs(self.T_cavern[i-1] - self.T_cavern[i])
            fact = self.anomaly_factor
            
            if cur_derivative > fact * avg_derivative:
                breakpoint()
                raise NumericAnomaly("The current rate of change is "+
                                     "{0:5.2f}".format(fact)+ 
                                     " times greater than the average rate of"+
                                     " change for 10 time steps")

    def _enthalpy_rate(self,i):
        
        h_in_times_msra = 0.0
        for wname, well in self.wells.items():
            for vname, pipe in well.pipes.items():
                
                msra = self.mass_rate[wname][vname][i]
                
                if msra > 0.0:
                    rfluid = self.res_fluid[wname][vname]

                    if well.input["ideal_pipes"]:
                        # HANDLE IDEAL CASE WHERE PIPES ARE NOT MODELED AND THE 
                        # RESERVOIR FLUID IS APPROXIMATED TO ENTER THE CAVERN AT 
                        # CAVERN PRESSURE WITH NO LOSS IN TEMPERATURE FROM THE RESERVOIR
                        rfluid.update(CP.PT_INPUTS, self.P_cavern[i-1], pipe.valve['reservoir']['temperature'])
                    else:
                        # NON-IDEAL CASE WHERE THE PIPES MAY LOSE MORE OR LESS ENERGY DEPENDING
                        # ON THE STATE OF THE GROUND AND RESERVOIR TEMPERATURE
                        rfluid.update(CP.PT_INPUTS, pipe.P_fluid[-1],  pipe.T_fluid[-1])
                    h_in_times_msra += msra * rfluid.hmass()
                else:
                # The last element interacts with the cavern CV 
                    pipe.fluid.update(CP.PT_INPUTS, self.P_cavern[i-1],  self.T_cavern[i-1])
                    h_in_times_msra += msra * pipe.fluid.hmass()
        # total mass rate is calculated elsewhere
        h_net_in = h_in_times_msra / self._total_mass_rate(i)
        return h_net_in 
        
        if self.mass_rate[i-1] > 0.0:
            h_in = self.res_fluid.hmass()
        else:
            h_in = self.fluid.hmass() 
            
        return h_in
        
        
    def _mass_rate(self,i):
        inp = self.input

        # specific heats ratio for mass rates.
        cpcv = self.fluid.cp0molar() / (self.fluid.cp0molar() - Constants.Rg['value'])
        # Calculating initial mass rate for t=0 depending on mass flow device
        # and filling/discharge mode
        for wname, well in inp["wells"].items():
            for vname, valve in well["valves"].items():
                
                if valve["type"] == "mdot" and i != 0:
                    # no need to go through the logic, the entire 
                    # mass_rate history is a boundary condition.
                    continue
                
                #
                msra = self.mass_rate[wname][vname]
                
                # establish reservoir conditions
                if "reservoir" in valve:    
                    T_res = valve["reservoir"]["temperature"]
                    P_res = valve["reservoir"]["pressure"]
                    rfluid = self.res_fluid[wname][vname]
                else:
                    # T_res not needed because flow only happens out 
                    # of the cavern for 'psv'
                    P_res = valve["back_pressure"]
                
            
                if valve["type"] == "orifice":
                    if P_res >= self.P_cavern[i]: # flow from the reservoir fluid will 
                                                # fill the cavern
                        k = rfluid.cp0molar() / (rfluid.cp0molar() - Constants.Rg['value'])
                        msra[i] = -tp.gas_release_rate(
                            P_res,
                            self.P_cavern[i],
                            rfluid.rhomass(),
                            k,
                            valve["discharge_coef"],
                            valve["diameter"] ** 2 / 4 * math.pi,
                        )
                    else:
                        msra[i] = tp.gas_release_rate(
                            self.P_cavern[i],
                            P_res,
                            self.rho[i],
                            cpcv,
                            valve["discharge_coef"],
                            valve["diameter"] ** 2 / 4 * math.pi,
                        )
                elif valve["type"] == "mdot" and i == 0:
                    # mass flow is not a function of the pressures and the entire
                    # history is already determined. 
                    if "mdot" in valve.keys() and "time" in valve.keys():
                        mdot = np.asarray(valve["mdot"])
                        time = np.asarray(valve["time"])
                        max_i = int(time[-1] / self.tstep)+1
                        interp_time = self.time_array
                        # unlike the original version of HyDown, + means filling, - means discharging
                        # no need for the "filling" and "discharge" input parameter.
                        msra[:max_i] = np.interp(interp_time, time, mdot)
                    else:
                        msra[:] = np.array(valve["mass_flow"],float)
        
                elif valve["type"] == "controlvalve":
                    Cv = tp.cv_vs_time(valve["Cv"], self.time_array[i], valve["time_constant"], valve["characteristic"])
                    if P_res >= self.P_cavern[i]:
                        # reservoir will push more fluid into the cavern
                        Z = rfluid.compressibility_factor() 
                        MW = self.MW
                        k = rfluid.cp0molar() / (rfluid.cp0molar()-Constants.Rg['value'])
                        
                        msra[i] = -tp.control_valve(
                            P_res, self.P_cavern[i], T_res, Z, MW, k, Cv
                        )
                    else: # cavern will discharge fluid to reservoir
                        Z = self.fluid.compressibility_factor() 
                        MW = self.MW 
                        k = cpcv
                        msra[i] = tp.control_valve(
                            self.P_cavern[i], P_res, self.T_cavern[i], Z, MW, k, Cv
                        )
                elif valve["type"] == "psv":
                    # only discharge for pressure safety valve
                    has_end_pres = 'end_pressure' in valve
                    if has_end_pres:
                        pres_gt_end_pres = self.P_cavern[i] > valve['end_pressure']
                    else:
                        pres_gt_end_pres = True
                    
                    # the valve closes below end_pressure
                    if has_end_pres and not pres_gt_end_pres:
                        msra[i]=0
                    else:
                        msra[i] = tp.relief_valve(
                            self.P_cavern[i],
                            valve["back_pressure"],
                            valve["set_pressure"],
                            valve["blowdown"],
                            cpcv,
                            valve["discharge_coef"],
                            self.T_cavern[i],
                            self.fluid.compressibility_factor(),
                            self.MW,
                            valve["diameter"] ** 2 / 4 * math.pi,
                        )
                    
                        
                else:
                    # validation should keep this from ever happening.
                    raise ValueError("input valve types allowed are: ['psv','controlvalve','mdot','orifice']")
                
                


