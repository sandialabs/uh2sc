# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:24:33 2024

@author: dlvilla
"""
from warnings import warn
import numpy as np
from CoolProp import CoolProp as CP
from uh2sc.constants import Constants as const
from uh2sc.transport import annular_nusselt, circular_nusselt
from uh2sc.utilities import (calculate_component_masses,
                             calculate_cavern_pressure)
from uh2sc.abstract import AbstractComponent
from uh2sc.constants import Constants

const = Constants()

#from uh2sc.hdclass import ImplicitEulerAxisymmetricRadialHeatTransfer


class Well(AbstractComponent):

    """
    A series of Vertical pipes of increasing or equal lengths starting with the longest
    pipe always being the innermost circular (regular) pipe. The lengths must reach
    the SaltCavern object.

    Each pipe is controlled by two boundary conditions:

    1) The surface mass flow in (toward the salt cavern) or out (away from the salt cavern)
    2) The temperature of either a) for flow in, a specified boundary condition for
                                    temperature at the surface or:
                                 b) for flow out, the cavern_temperature at the
                                    lowest part of the pipe.

    The pressure is always assumed to be the cavern pressure at the elevation
    at which a pipe flow exits into the salt cavern


    THIS ONLY SUPPORT ONE WELL CURRENTLY! I AM MARKING PARTS OF THE CODE
    THAT WILL NEED TO CHANGE WHEN MULTIPLE WELLS IS IMPLEMENTED WITH THIS STRING
    "MULTI_WELL_CHANGE_NEEDED"

    """

    def __init__(self,well_name,well_dict,model,global_indices):

        """
        CP - cool props object
        """
        # manage convergence type 
        if model is None:
            self._use_relative_convergence = False
            self.residual_normalization = None
        else:
            self._use_relative_convergence = model._use_relative_convergence
            self.residual_normalization = model.residual_normalization    
            
        self._NUM_EQN = 4 + model.number_fluids
        self._gindices = global_indices
        self.input = well_dict
        self._model = model
        self._name = well_name
        

        
        
        P0 = model.inputs['initial']['pressure']
        T0 = model.inputs['initial']['temperature']
      

        if not "ideal_pipes" in well_dict:
            self.input["ideal_pipes"] = False

        pipe_materials = [PipeMaterial(surf_rough_h,therm_cond)
                          for surf_rough_h, therm_cond
                          in zip(well_dict["pipe_roughness_ratios"],
                                 well_dict["pipe_thermal_conductivities"])]

        num_cv = [np.ceil(length/well_dict["control_volume_length"])
                       for length in well_dict["pipe_lengths"]]
        pidx = range(len(pipe_materials))


        # establish individual pipes
        # TODO - make the height and length be able to be different
        # (for non-straight or angled pipes)
        pipes = {valve_name:VerticalPipe(self,
                              model.fluids[well_name][valve_name],
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

        self._number_fluids = 0
        for pipe_name, pipe in pipes.items():
            self._number_fluids += pipe._number_fluids
            
            
    
        if len(pipes) != 1:
            raise NotImplementedError("The number of pipes per well is limited to 1 for the present version of uh2sc!")

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
        currently wells do not connect to elements before them.
        """

        return None
            

    @property
    def next_adjacent_components(self):
        """
        interface variable indices for the next component which can only be 
        1 GHE for a cavern (even though this is written as a loop)
        """
        if self._model.is_test_mode:
            return self._model.test_inputs["cavern"]
        else:
            return {'cavern':self._model.components['cavern']}
        
        
    @property
    def component_type(self):
        """
        A string that allows the user to identify what kind of component 
        this is so that specific properties and methods can be invoked

        """
        return "Well"


    def evaluate_residuals(self,x=None,get_independent_vars=False):
        """
        Must first evaluate all interface equations for indices produced by interface_var_ind_prev_comp

        Then must evaluate all internal component equations
        
        Equations for the salt cavern
        
        Energy flow


        Args:
            x numpy.array : global x vector for the entire
                             differiental/algebraic system
            get_independent_vars bool : enables collecting variables that
                             are otherwise unexposed. (NOT CURRENTLY USED)
                             
                
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
        if x is not None:
            self.load_var_values_from_x(x)

        residuals = np.zeros(self._NUM_EQN)
        _eqn = 0
        # mass flow continuity
        for comp_name, comp in self.next_adjacent_components.items():
            for pname, pipe in self.pipes.items():
                
                
                mdot = np.interp(self._model.time,pipe.valve['time'],pipe.valve['mdot'])
                total_flow = pipe.mass_rates[0,:].sum()
                
                if mdot > 0: # flow is into the cavern
                    fluid = pipe.fluid
                    t_in = pipe.valve['reservoir']['temperature']
                    if pipe.valve['reservoir']['pressure'] == "follow cavern":
                        p_in = comp._p_cavern
                        t_exit_inverted, p_exit_inverted = pipe.initial_adiabatic_static_column(
                                t_in, p_in, -mdot)
                        p_in = p_exit_inverted[0]
                    else:
                        p_in = pipe.valve['reservoir']['pressure']
                        
                    t_exit, p_exit = pipe.initial_adiabatic_static_column(
                            t_in, p_in, mdot)
                    if p_exit[-1] < comp._p_cavern*0.99:
                        # insufficient potential kinetic energy to flow!
                        warn("Flow is not occuring because reservoir pressure"
                             +" plus adiabatic column pressure change is less"
                             +" than the cavern pressure!")
                        mdot = 0
                else:
                    fluid = comp._fluid
                    t_in = comp._t_cavern
                    p_in = comp._p_cavern
                    t_exit, p_exit = pipe.initial_adiabatic_static_column(
                            t_in, p_in, mdot)
                    if p_exit[0] < const.atmospheric_pressure['value']:
                        # insufficient potential kinetic energy to flow!
                        warn("Flow is not occuring because cavern pressure"
                             +" minus adiabatic column pressure change is less"
                             +" than atmospheric pressure. Caverns should not"
                             +" be run at low pressure!")
                        mdot = 0
                        
                fluid.set_state(CP.AbstractState)
                gmdots = calculate_component_masses(fluid, mdot)
                fluid.del_state()
                # mass balance of each gas component
                
                # factor 1000 to make mass flow match more important!
                residuals[0:self._number_fluids] = 1000 * (pipe.mass_rates[0,:] - gmdots)
                _eqn += self._number_fluids
                if self._use_relative_convergence:
                    residuals[0:self._number_fluids] = (
                        residuals[0:self._number_fluids]
                        /self.residual_normalization["mass_flow_norm"]
                        )
                    
                # enforce adiabatic change in pressure and temperature.
                residuals[_eqn] = pipe.temp_fluid[-1] - t_exit[-1]
                if self._use_relative_convergence:
                    residuals[_eqn] = residuals[_eqn]/self.residual_normalization["temperature_norm"]
                _eqn += 1
                
                residuals[_eqn] = pipe.pres_fluid[-1] - p_exit[-1]
                if self._use_relative_convergence:
                    residuals[_eqn] = residuals[_eqn]/self.residual_normalization["cavern_pressure"]
                _eqn += 1
                
                residuals[_eqn] = pipe.temp_fluid[0] - t_exit[0]
                if self._use_relative_convergence:
                    residuals[_eqn] = residuals[_eqn]/self.residual_normalization["temperature_norm"]
                _eqn += 1
                
                residuals[_eqn] = pipe.pres_fluid[0] - p_exit[0]
                if self._use_relative_convergence:
                    residuals[_eqn] = residuals[_eqn]/self.residual_normalization["cavern_pressure"]
                
        if get_independent_vars:
            return () # nothing needed here until the well becomes more complex
        
        return residuals
    
    @property
    def independent_vars_descriptions(self):
        return []

    def equations_list(self):
        e_list = []
        for pname, pipe in self.pipes.items():
            for fluid_name in pipe.fluid.fluid_names():
                e_list += [f"Well {self._name}, Pipe {pname}, {fluid_name} mass balance"]
            
            e_list += [f"Well {self._name}, Pipe {pname}, exit temperature adiabatic continuity"]
            e_list += [f"Well {self._name}, Pipe {pname}, exit pressure adiabatic continuity"]
            e_list += [f"Well {self._name}, Pipe {pname}, entrance temperature adiabatic continuity"]
            e_list += [f"Well {self._name}, Pipe {pname}, entrance pressure adiabatic continuity"]
            
        return e_list
            
         

    
    def get_x(self):
        # THIS HAS TO BE UPDATED IF YOU IMPLEMENT CONTROL VOLUMES!!!
        x = []
        for pname, pipe in self.pipes.items():
            for idx in range(self._number_fluids):
                x += [pipe.mass_rates[0,idx]]
            x += [pipe.temp_fluid[0]]
            x += [pipe.pres_fluid[0]]
            x += [pipe.temp_fluid[1]]
            x += [pipe.pres_fluid[1]]
        return np.array(x)


    def load_var_values_from_x(self,xg):
        # THIS HAS TO BE UPDATED IF YOU IMPLEMENT more than one CONTROL VOLUMES!!!
        nfl = self._number_fluids
        xloc = xg[self.global_indices[0]:self.global_indices[1]+1]
        for name, pipe in self.pipes.items():
            pipe.mass_rates = np.array([xloc[0:nfl]])
            pipe.temp_fluid = np.array([xloc[nfl],xloc[nfl+2]])
            pipe.pres_fluid = np.array([xloc[nfl+1],xloc[nfl+3]])
    
            
            
    def shift_solution(self):
        """
        Currently nothing has to happen hear because the pipe 
        is a purely algebraic function whose boundary conditions do 
        not depend on the previous time step.
        """
        pass




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

    Geometrical Explanation
    =======================

    This class can either be a fully anular pipe as seen below or a regular pipe with just
    the outside_outer_diameter and inside_outer_diameter defined (the "inner" diameter are NaN)
    in this second case. As seen below vertical pipes interface with each other w/r to conduction/convection
    heat transfer or else they are a simple pipe and are the inner flow or else they interface
    to the ground or grout material on the outside of the well (i.e. see the
    ExplicitAxisymmetricRadialHeatTransfer class)


          ZZZZZZZZZZZZZZZZZZZZZZ
        Z _____________________ Z
       Z /+ + + + + + + +  +   \ Z
      Z /+ + ______________ + + \ Z
     Z /+ + / ************  \+ + \ Z     + = steel or other pipe material that is not flowing
    Z /+ + /** __________ ** \+ + \ Z    * = analyzed annualar section for this pipe
    Z |+ +/** /+ + + + + \ ** \+ +| Z    X = previous annular or regular pipe
    Z |+ +|**/+ + /---\+ +\** |+ +| Z        of the 1-D well model this vertical pipe is part of
    Z |+ +|**|+ +| X X|+ +| **|+ +| Z    Z = next annular pipe is located in the
                     ^   ^   ^   ^           1-D well model this vertical pipe is part of
                     |   |   |   |
                     ^   ^   ^   ^ outside_outer_diameter
                     |   |   ^ inside_outer_diameter
                     |   ^ inside_inner_diameter
                     ^ outside_inner_diameter

    The length of the pipe is equal to the height change if the pipe is exactly vertical. Otherwise,
    The pipe is not vertical (this only serves to increase the frictional losses in the pipe)
    It is assumed that the pipes are nearly vertical (don't use this class to simulate a horizontal pipe!)
     _______
    \      ^
   | \     |
   |  \    | height_change
   |   \   |
   |    \  |     length = sqrt(height_change**2 + horizontal_change**2)
   |     \ v
   |      \_
   | <---->|
     horizontal_change
     _____________ surface
           |  0
           |  1     The fluid flow control volumes (CVs)are
           |  2     numbered from 0 at the surface to
           |  .     the "number_of_control_volumes"
           |  .     which exits into the cavern.
           | -3
         __|_-2 cavern entrance
        /  |-1\ = number_control_volumes
       /       \  The well can penetrate the cavern
       | cavern \ and deliver gas at whatever depth
                  its exit is at!

    Hope that helps you understand how the geometry of this class' inputs!

    """

    def __init__(self,
                 well,
                 fluid,
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
        well : uh2sc.well.Well - the well this pipe is in.
        
        
        fluid : fluid.FluidWithFitOption - the fluid mixture (or pure_fluid)
                being moved through the pipe. This fluid is created at the model
                level so that it contains every pure_fluid throughout the entire
                model even if the molefractions are zero.

        length : float :
            Length of the vertical pipe

        height_change : float :
            Total vertical distance the length covers. If height_change < length
            then the pipe is diagonal and not purely vertical. This simulation only
            approximates pipes with no more than 10 deg off from vertical.

        pipe_material : uh2sc.well.PipeMaterial :
            class carrying all of the parameters needed to analyze the pipe material

        valve : dict :
            Dictionary containing all the inputs needed for the valve at the surface
            end of the pipe

        valve_name : str : identifying name of the valve dictionary

        initial_pressure : float :
            The initial pressure throughout the pipe (single value)

        initial_temperature : float :
            The initial temperature throughout the pipe (single value)

        outside_inner_diameter : float :
            The smallest diameter of the annular section facing the previous annular pipe
            in the Well if another one exists. The value will be NaN if this is a regular
            pipe.

        inside_inner_diameter : float :
            The lesser diameter in the annular flow region. If this is a regular
            pipe, this value is NaN

        inside_outer_diameter : float :
            The greater diameter in the annular flow region.

        outside_outer_diameter : float :
            The largest diameter of the annular section facing the next annualar pipe
            in the Well if another exists. Otherwise, it interfaces to the
            ExplicitAxisymmetricRadialHeatTransfer object of the well.

        number_control_volumes : int :
            The number of control volumes (constrained to be equal to all adjacent)
            VerticalPipe objects.

        total_minor_losses_coefficient : float :
            Total minor loss coefficients for bends, flanges, etc..
            see https://en.wikipedia.org/wiki/Minor_losses_in_pipe_flow


        mass_rate0 : float :
            The mass rate (kg/s) at which fluid is flowing into the cavern (+)
            or out of the cavern (-). If flowing into, then intial_temperature
            indicates the temperature at the surface, if flowing out of, the
            initial_temperature indicates the cavern temperature.

        Returns
        -------
        VerticalPipe object

        Raises
        ------
            ValueError - on input values that are incorrect
            TypeError - on input types that are incorrect
        """
        # input checking. NOTE: needs to be in the input validation instead of
        # here!
        self._well = well

        
        if height_change > length:
            raise ValueError("The height change must be equal to or less than the pipe length")


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
        fluid.set_state(CP.AbstractState,[initial_pressure,initial_temperature])
        self.fluid = fluid
        self.fluid.specify_phase(CP.iphase_gas)

        self.fluid.update(CP.PT_INPUTS, initial_pressure,
                                        initial_temperature)
        self._number_fluids = len(self.fluid.fluid_names())
        self.valve = valve
        self.valve_name = valve_name
        self.L_cv = length / number_control_volumes
        self.L_cv_o2 = self.L_cv / 2
        num_cv_p_1 = int(number_control_volumes + 1)
        self.num_cv = int(number_control_volumes)
        
        
        # more input checking
        if valve['type'] != "mdot":
            raise NotImplementedError("Only mdot valves are currently developed")
        elif valve['time'][0] != 0.0:
            raise ValueError("mdot valves must have a time value of 0.0 for all start times!")
        else:
            mass_rate0 = valve['mdot'][0]
            # mass balance for each gaseous fluid.
            if well is not None: # None in model call that just needs initial_adiabatic_column function
                mass_rates = calculate_component_masses(self.fluid,
                                                       mass_rate0)
            else:
                mass_rates = np.zeros((num_cv_p_1,self._number_fluids))
                
                
        self.fluid.del_state()
        # state variables
        self.temp_fluid, self.pres_fluid = self.initial_adiabatic_static_column(initial_temperature,
                                                            initial_pressure,
                                                            mass_rate0)
        

        self.mass_rates = mass_rates * np.ones([num_cv_p_1,self._number_fluids])

        # TODO, mass_loss is a place holder. it is not developed.
        self.mass_loss = np.zeros(int(number_control_volumes))

        self.depths = [idx*self.L_cv for idx in range(num_cv_p_1)]
        self.adjacent_pipes = {"inner":None,"outer":None}
        self.results = {"temp_fluid":[],
                        "pres_fluid":[],
                        "temp_wall":[],
                        "mass_loss":[],
                        "molefracs":[],
                        "mass_rate":[]}
        # If this is not an annulus, but just a simple pipe, then r_pipe_inner is not needed
        if self.diameters[0] == 0.0:
            self.r_pipe_innter = np.nan
        else:
            self.r_pipe_inner = np.log(self.diameters[1]/self.diameters[0])/(2.0*np.pi
                            * self.L_cv * self.material.kp)
        self.r_pipe_outer = np.log(self.diameters[3]/self.diameters[2])/(2.0*np.pi
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
        if (not adj_inner_pipe is None) and (not isinstance(adj_inner_pipe,VerticalPipe)):
            raise TypeError("adj_inner_pipe must be a VerticalPipe ojbect!")
        if (not adj_outer_pipe is None) and (not isinstance(adj_outer_pipe,VerticalPipe)):
            raise TypeError("adj_outer_pipe must be a VerticalPipe ojbect!")

        # assure alignment
        self._verify_adjacent_pipes_aligned(adj_inner_pipe, adj_outer_pipe)

        #set the needed objects
        self.adjacent_pipes = {"inner": adj_inner_pipe,"outer":adj_outer_pipe}




    # NOT USED
    def _hydraulic_diameter(self):

        if self.diameters[0] == 0.0 and self.diameters[1] == 0.0:
            area_times_4 = np.pi * self.diameters[2]**2
            perimeter = np.pi * self.diameters[2]
        else:
            area_times_4 = np.pi * (self.diameters[2] ** 2 - self.diameters[1] ** 2)
            perimeter = np.pi * (self.diameters[2] + self.diameters[1])

        return area_times_4 / perimeter, area_times_4/4.0


    def initial_adiabatic_static_column(self,initial_temperature,initial_pressure,mass_rate0,return_rho=False):
        """
        Provides an initial condition of a static adiabatic column for the Vertical
        Pipe which is a fairly good approximation of the process involved.

        It is assumed that there is only a single fluid type in the pipe
        at this time.

        Returns
        -------
        None.

        """
        self.fluid.set_state(CP.AbstractState,[initial_pressure,initial_temperature])
        
        temp_fluid = np.zeros(self.num_cv+1)
        pres_fluid = np.zeros(self.num_cv+1)

        if return_rho:
            rho_fluid = np.zeros(self.num_cv+1)

        delh = self.len_p_cv
        grav = const.g['value']

        if isinstance(mass_rate0,(float,int)):
            inflow = mass_rate0 > 0
        elif isinstance(mass_rate0,np.ndarray):
            inflow = mass_rate0.sum() > 0
        else:
            raise ValueError("mdot must be an array or numeric!")
            
        if inflow > 0:
            flow_into_cavern = True
            cv_n = 0
            sign = 1
        else:
            flow_into_cavern = False
            cv_n = self.num_cv
            sign = -1

        temp_fluid[cv_n] = initial_temperature
        pres_fluid[cv_n] = initial_pressure

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

            fluid.update(CP.PT_INPUTS, pres_fluid[cv_n_r1],
                                             temp_fluid[cv_n_r1])

            if return_rho:
                rho_fluid[cv_n] = fluid.rhomass()
            R_gas = fluid.gas_constant() / fluid.molar_mass()
            z_compr = fluid.compressibility_factor()
            gamma = fluid.cpmass()/fluid.cvmass()


            del_pres = pres_fluid[cv_n_r1] * (1 - np.exp(-grav*delh/(R_gas * temp_fluid[cv_n_r1] * z_compr)))

            del_temp = (1 - 1/gamma)* grav/(z_compr * R_gas) * delh



            # must increase or decrease depending on solution method.
            pres_fluid[cv_n] = pres_fluid[cv_n_r1] + sign * del_pres
            temp_fluid[cv_n] = temp_fluid[cv_n_r1] + sign * del_temp

        if return_rho:
            fluid.update(CP.PT_INPUTS, pres_fluid[cv_n],
                                         temp_fluid[cv_n])
            rho_fluid[cv_n] = fluid.rhomass()
            
            fluid.del_state()

            return temp_fluid,pres_fluid,rho_fluid
        else:
            
            fluid.del_state()
            return temp_fluid,pres_fluid



    # NOT USED
    def _average_velocity(self,mdot,density):
        return mdot / density / self.area
    
    # NOT USED
    def _viscous_pressue_loss(self,friction_factor,length,velocity, min_loss):
        return ((friction_factor * length / self.D_h + min_loss) * velocity ** 2
                / (2 * const.g["value"]))
    # NOT USED
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

    # NOT USED - DEVELOPMENT STOPPED
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

            q_out = 0


            # heat transfer bi-directional (in and out)
        #TODO - these are static values MOVE THEM to conductive axi-symmetric heat resistivities




        # h_film_outer = h_outside_pipe(density,
        #                              velocity,
        #                              hydraulic_diameter,
        #                              dynamic_viscosity,
        #                              specific_heat,
        #                              thermal_conductivity)

    # NOT USED
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



    # NOT USED
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
