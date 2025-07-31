

from copy import deepcopy
from warnings import warn
import numpy as np
import logging
import yaml

from matplotlib import pyplot as plt
import pandas as pd

import CoolProp.CoolProp as CP

from uh2sc import validator
from uh2sc.errors import InputFileError, NewtonSolverError
from uh2sc.solvers import NewtonSolver
from uh2sc.abstract import AbstractComponent, ComponentTypes
from uh2sc.ghe import ImplicitEulerAxisymmetricRadialHeatTransfer
from uh2sc.utilities import (process_CP_gas_string, 
                             reservoir_mass_flows, 
                             find_all_fluids,
                             calculate_component_masses,
                             brine_average_pressure)
from uh2sc.salt_cavern import SaltCavern
from uh2sc.well import Well, VerticalPipe, PipeMaterial
from uh2sc.constants import Constants
from uh2sc.thermodynamics import (density_of_brine_water, 
                                  brine_saturated_pressure, 
                                  solubility_of_nacl_in_h2o)




ADJ_COMP_TESTING_NAME = "testing"

class Model(AbstractComponent):
    """
    Formulate a model that consistst of components. Each component has an abstract
     class as its base in uh2sc.abstract.AbstractComponent. Interface and boundary
     conditions are handled as additional equations. Each component has a method
     called "evaluate_residuals." Current valid components are:

     1. Cavern
     2. Arbitrary number of wells
     3. Arbitrary number of 1-D axisymmetric heat transfer through the ground

     Future versions may include multiple caverns that are connected via
     surface pipes and may include pumps/compressors.
    """

    def __init__(self,inp,single_component_test=False,solver_options=None,**kwargs):

        """
        Construct a combined model of 1) a salt cavern, 2) an arbitrary number
        of wells and 3) Radial heat transfer away from the salt cavern and wells.



        Inputs
        ======

        inp : str or dict : str = filepath to a yaml file that conforms to the
                                  uh2sc schema or:
                            dict = dict that conforms to the uh2sc schema

        """
        if solver_options is None:
            solver_options = {"TOL": 1.0e-2}
            
        self.converged_solution_point = False
        self.inputs = self._read_input(inp)
        if isinstance(inp,str):
            self.input_file = inp
        else:
            self.input_file = None
        time_step = self.inputs["calculation"]["time_step"]
        
        nstep = int(self.inputs["calculation"]["end_time"]
                    / time_step
                    + 1)
        self._end_time = self.inputs["calculation"]["end_time"]
        self._max_time_step = time_step
        self._min_time_step = 100
        self.time = 0.0
        self.time_step = self._max_time_step
        
        self.test_inputs = kwargs
        if len(kwargs) != 0:
            self.is_test_mode = True
        else:
            self.is_test_mode = False

        if not single_component_test:
            if len(kwargs) != 0:
                raise ValueError("kwargs are only allowed in single component"
                                +" testing! Do not call Model with kwargs!")
            self._validate()
            num_caverns = 1
            num_ghes = len(self.inputs["ghes"])
            num_wells = len(self.inputs["wells"])
        else:
            num_caverns = 0
            num_ghes = 0
            num_wells = 0
            # each component type has a single component test mode to assure
            # it can solve a simple case that does not include the other parts
            if kwargs["type"] == ComponentTypes(2).name:
                num_ghes = 1
            elif kwargs["type"] == ComponentTypes(4).name:
                num_caverns = 1
            elif kwargs["type"] == ComponentTypes(3).name:
                num_wells = 1
            else:
                raise ValueError(f"Only valid kwargs['type']:" 
                        +f"{' '.join([ct.name for ct in ComponentTypes])}")


        self._build(num_caverns,num_wells,num_ghes)

        self.solver = NewtonSolver(solver_options)


        self._solutions = {}
        
        self.evaluate_residuals(get_independent_vars=True)
        
        # used for analytics to assure variables in the model are realistic
        # with respect to actual salt cavern gas dynamics.
        self.components['cavern']._analytics()

    def _form_array(self):
        """
        Create a numpy array and column names
        
        """
        keys = np.array(list(self._solutions.keys()))
        values = np.array(list(self._solutions.values()))
        column_names = self.xg_descriptions
        column_names = ["Time (s)"] + column_names
        out_arr = np.concat([keys.reshape([len(keys),1]),values],axis=1)
        return column_names, out_arr
    
    def write_results(self,filename="uh2sc_results.csv"):
        """
        
        Write out results as a CSV
        
        """
        column_names, out_arr = self._form_array()
        np.savetxt(filename, out_arr, header=",".join(column_names), delimiter=',')
    
    @property
    def dataframe(self, relative_time=False):
        """
        Provide results as a dataframe
        
        Inputs:
        =======
        
        relative_time: bool: optional : Default =False
              If True, then time is just a value equal to
              the number of seconds since the simulation began
              If False, then the index is based on date-time stamps.
        
        """
        # get the results
        column_names, out_arr = self._form_array()
        start_date_str = self.inputs['initial']['start_date']
        time_step = self.inputs['calculation']['time_step']
        end_time = self.inputs['calculation']['end_time']

        # process for dataframe considerations        
        if relative_time:
            index = range(0, end_time + time_step, time_step)
        else:
            # date considered.
            start_date = pd.to_datetime(start_date_str)
            # Calculate the number of time steps
            num_steps = int(end_time / time_step) + 1
            
            # Create a datetime index with the specified time step
            index = pd.date_range(start=start_date, periods=num_steps, freq=f'{time_step}S')
        
        df = pd.DataFrame(out_arr,index=index,columns=column_names)
        
        return df
        
    
    
    @property
    def solutions(self):
        return self._solutions
    
    @property
    def component_type(self):
        return "model"

    def run(self,new_inp=None):
        """
        inputs:
            new_inp : dict - an input object that, if present, resets 
                      all of the inputs but does not reinitialize the
                      model. This way, the model can be commanded to change 
                      mass flows and such on the fly while maintaining the
                      current state of the cavern.
        
        """
        if new_inp is not None:
            raise NotImplemented("The ability to add new input "
                                 +"has not been completed!")
            self.inputs = self._read_input(new_inp)
            self._validate()
            time_step = self.inputs["calculation"]["time_step"]

            nstep = int(self.inputs["calculation"]["end_time"]

                        / time_step
                        + 1)

            for component_name, component in self.components.items():
                if component.component_type == 'Well':
                    new_well = Well(component_name,new_inp['wells'][component_name],self, component.global_indices)
                    self.components[component_name] = new_well
        
        
        # record initial state
        self._solutions[self.time] = deepcopy(self.get_x())
        
        while self.time < self._end_time:
            x_org = self.get_x()

            logging.info(f"UH2SC model beginning calculations for time={self.time}.")
            # solve the current time step
            
            tup = self.solver.solve(self)
            
            
            solver_converged = bool(tup[0])
            if solver_converged:
                # shift the time
                
                
                if 'cavern' in self.components:
                    if hasattr(self.components['cavern'], "troubleshooting"):
                        print(self.time)
                
                # update the state of the model
                self.shift_solution()
                
                # shift time
                self.time += self.time_step
                
                # gather the results
                self._solutions[self.time] = deepcopy(self.get_x())
                
                
                
                # increase the time step if it has shrunk
                if self.time_step < self._max_time_step:
                    proposed_time_step = 1.5 * self.time_step
                    if proposed_time_step < self._max_time_step:
                        self.time_step = proposed_time_step
                    else:
                        self.time_step = self._max_time_step
            
            else:
                if self.time_step > self._min_time_step:
                    # try resetting and trying a smaller time step
                    self.load_var_values_from_x(x_org)
                    self.time_step = 0.5*self.time_step
                    

                    
                else:
                    msg = tup[1]
                    raise NewtonSolverError("The Newton solver returned an" 
                            +f" error for time {self.time} and the time step has"
                            +" reached the minimum allowed value. The solver"
                            +f" message is: {msg}")


    @property
    def next_adjacent_components(self):
        """_summary_
        One day we may make this code be able to
        connect two models but we have not yet
        Done this
        """
        return []

    @property
    def previous_adjacent_components(self):
        """_summary_
        One day we may make this code be able to
        connect two models but we have not yet
        Done this
        """
        return []

    @property
    def global_indices(self):
        """
        This property must give the indices that give the begin and end location
        in the global variable vector (xg). Since this is the entire model
        it will return the entire range.
        """
        return (0, len(self.xg))

    def get_x(self):
        xg = self.xg
        for cname, component in self.components.items():
            bind, eind = component.global_indices

            xg[bind:eind+1] = component.get_x()

        return xg

    def evaluate_residuals(self,x=None,get_independent_vars=False):
        residuals = []
        if x is None:
            for _cname, component in self.components.items():

                # this is the single x behavior used by
                # local evaluations of residuals

                residuals += list(component.evaluate_residuals())
                if get_independent_vars:
                    self.independent_vars = component.evaluate_residuals(
                        get_independent_vars=get_independent_vars)

                
            return np.array(residuals)
        else:
            # this is for scipy where a matrix of all the different
            # vectors needed to evaluate the jacobian is fed in
            if x.ndim == 2:
                for xv in x:
                    v_residuals = []
                    for _cname, component in self.components.items():
                        v_residuals += list(component.evaluate_residuals(xv))
                        if get_independent_vars:
                            self.independent_vars = component.evaluate_residuals(
                                x=xv,
                                get_independent_vars=get_independent_vars)
                    residuals.append(v_residuals)
            elif x.ndim == 1:
                for _cname, component in self.components.items():

                    # this is the single x behavior used by
                    # local evaluations of residuals
                    residuals += list(component.evaluate_residuals(x))
                    if get_independent_vars:
                        self.independent_vars = component.evaluate_residuals(
                            x=x,
                            get_independent_vars=get_independent_vars)



            return np.array(residuals)


    def load_var_values_from_x(self, xg_new):
        for cname, component in self.components.items():
            component.load_var_values_from_x(xg_new)
            
    def shift_solution(self):
        for cname, component in self.components.items():
            component.shift_solution()
            
    def equations_list(self):
        e_list = []
        for cname, component in self.components.items():
            e_list += component.equations_list()
        return e_list

    def plot_solution(self,variables):
        """
        Inputs:
            variables: list: of either integers or strings, strings must
                 be in self.xg_descriptions
        
        
        """
        xgd = self.xg_descriptions
        num_var = len(xgd)
        
        axd = {}
        figd = {}
        for variable in variables:
            if isinstance(variable, int):
                if variable < len(xgd) and variable > -1:
                    variable_str = self.xg_descriptions[variable]
                else:
                    raise ValueError(f"Only integers less than the number "
                                     +f"of variables {num_var} are allowed,"
                                     +f" you entered {variable}!")
            else:
                if variable in self.xg_descriptions:
                    variable_str = variable
                else:
                    raise ValueError("You must enter a valid variable name"
                                     +f" string, you entered {variable} which"
                                     +" is not in the variable list "
                                     +"model.xg_descriptions")
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            
            idx = self.xg_descriptions.index(variable_str)
            
            values = np.array([[time,solution[idx]] for time,solution 
                               in self._solutions.items()])
            
            ax.plot(values[:,0],values[:,1],label=variable_str)
            ax.grid("on")
            ax.set_xlabel("time (s)")
            ax.set_ylabel(variable_str)
            
            
            axd[variable_str] = ax
            figd[variable_str] = fig
        
            plt.show()
            
        return figd, axd
        
        pass    

    def _build(self,num_caverns,num_wells,num_ghes):
        """
        Assemble the order of the global variable vector the ordering is as follows:

        Though it appears that gas/liquid can be modeled interchangeably. The current
        version of uh2sc only allows a pipe to carry either a gas or a liquid permanently
        in a simulation. There is no ability to switch or mix gas and liquid in the pipes
        Such mixtures are much more difficult to model and are beyond the scope of the
        simple model uh2sc uses. The purpose of adding liquid can be to control gas pressure
        uh2sc does not model chemistry inside the liquid.

        THIS NEEDS UPDATING (BELOW)
        Component    Num var        Description
        Cavern            1.             Cavern Gas Temperature
                          1.             Cavern Wall Temperature
                          1.             Cavern Liquid Temperature
                          1.             Cavern Liquid Wall Temperature
                          1.             Cavern Pressure at Liquid Interface
                          1.             Liquid interface distance from cavern top
                          num_gas        Gas fluxes at the liquid interface
                          num_gas        Gas/Liquid fluxes from well 1   pipe 1
           .              num_gas        Gas/Liquid fluxes from well 1   pipe 2
           .              .
           .              .
           .              num_gas        Gas/Liquid fluxes from well 1   pipe np_w1
           .              num_gas        Gas/Liquid fluxes from well 2   pipe 1
           .              .              .
           .              .              .
                          num_gas        Gas/Liquid fluxes from well nw  pipe np_wnw
        Wells
                          1.             Gas/Liquid temperature in well 1 pipe 1 element 1
                          1 or 2         In/Out pipe wall temps in well 1 pipe 1 element 1
                          NOTE: is 1 for a simple pipe, is 2 for a concentric pipe
                          1.             Gas/Liquid pressure    in well 1 pipe 1 element 1
                          num_gas        Gas/Liquid flow rates  in well 1 pipe 1 element 1
                          NOTE: These could be exiting or entering depending on the flow conditions!
                          np_w1          Gas/Liquid temperature of fluid flows element 1
                          np_w1          Gas/Liquid pressure of gas/liquid element 1
        GHEs              1.             Temperature element 1
                          NOTE: The surface temperature of the outermost pipe or salt cavern
                                wall is the same as the first temperature for the GHE.
                          1.             Temperature element 2
                          .
                          .              Temperature element n

        """

        xg_descriptions = []
        self.xg = []
        self.xg_descriptions = []
        xg = []
        num_var = {}
        components = {}
        
        # assigns self.fluids for reservoirs from mdot valves and the initial
        # cavern state (which changes unless every gas is the same)
        # also assigns self.fluid_components which is the list of all pure fluids
        # that exist in the model and is used to define equations
        # sets self.molar_masses and others!
        if not self.is_test_mode:
            # if not, then we are in a test mode.
            find_all_fluids(self)
            self.number_fluids = len(self.molar_masses)

        if num_caverns != 0:
            self._bounds_characteristics()
            self._build_cavern(xg_descriptions,xg,components)
            num_var["caverns"] = len(xg_descriptions)
            
        else:
            num_var["caverns"] = 0
        if num_wells != 0:
            self._build_wells(xg_descriptions,xg,components)
            num_var["wells"] = (len(xg_descriptions)
                                             - num_var["caverns"])
        else:
            num_var["wells"] = 0

        if num_ghes != 0:
            self._build_ghes(xg_descriptions,xg,components)
            num_var["ghes"] = (len(xg_descriptions)
                                             - num_var["caverns"]
                                             - num_var["wells"])
        else:
            num_var["ghes"] = 0

        self.xg_descriptions = xg_descriptions
        self.xg = np.array(xg)
        self.components = components

        if not self.is_test_mode:
            self._connect_components()
            
        self.load_var_values_from_x(self.xg)


    def _build_ghes(self,x_desc,xg,components):
        """
        Build all GHE's and link them to the
        specified well or the cavern

        Args:
            x_desc list: _description_
            xg list: _description_
            components (_type_): _description_
        """

        ghes = self.inputs["ghes"]

        for name,ghe in ghes.items():
            if self.is_test_mode:
                # This is testing mode and you have to add stuff that
                # would normally come from the main ("schema_general.yml") schema.
                t_step = self.test_inputs["dt"]
                height = self.test_inputs["height"]
                inner_radius = self.test_inputs["inner_radius"]
                adjacent_comps = {}
                adjcomp_name = ADJ_COMP_TESTING_NAME
                # backfit missing stuff to get the solution to work.
                self.inputs["calculation"] = {}
                self.inputs["calculation"]["end_time"] = self.test_inputs["end_time"]
                self.inputs["calculation"]["time_step"] = self.test_inputs["dt"]
            else:
                # find the adjacent components to this GHE!
                adjacent_comps = self._find_ghe_adj_comp(name)

                # set the inner_radius and height either based on user input
                # or based on the maximum value from adjacent copmonents for height
                # and diameter.
                if "inner_radius" in ghe:
                    inner_radius = ghe["inner_radius"]
                else:
                    if len(adjacent_comps) == 0:
                        raise ValueError("Axi-symmetric heat transfer must have "
                                         +"at least one adjacent component if the"
                                         +" inner_radius is not defined in the input!")

                    inner_radius = self._assign_max_from_adj_comp(adjacent_comps,
                                                                  "diameter",
                                                                  0.5,
                                                                  -1)
                if "height" in ghe:
                    height = ghe["height"]
                else:
                    if len(adjacent_comps) == 0:
                        raise ValueError("Axi-symmetric heat transfer must have "
                                         +"at least one adjacent component if the"
                                         +" height is not defined in the input!")
                    height = self._assign_max_from_adj_comp(adjacent_comps,
                                                            "height",
                                                            1.0)
                # set the constant time step

                t_step = self.inputs["calculation"]["time_step"]

            # begin global indice
            beg_idx = len(xg)

            xg += [ghe["initial_conditions"]["Q0"]]
            x_desc += [f"GHE name `{name}` heat flux at inner_radius (W)"]

            # add all new terms that belong to the descriptions and to the
            xg += [ghe["farfield_temperature"] for _idx in range(ghe["number_elements"]+1)]
            x_desc += [f"GHE name `{name}` element {idx} outer temperature "
                       +f"(inner temperature element {idx-1})"
                       for idx in range(ghe["number_elements"]+1)]

            xg += [ghe["initial_conditions"]["Qend"]]
            x_desc += [f"GHE name `{name}` heat flux at outer_radius (W)"]

            #end of global indices for this component
            end_idx = len(xg)-1

            
            self.xg = xg
            self.xg_descriptions = x_desc

            components[name] = ImplicitEulerAxisymmetricRadialHeatTransfer(inner_radius,
                  ghe["thermal_conductivity"],
                  ghe["density"],
                  ghe["heat_capacity"],
                  height,
                  ghe["number_elements"],
                  ghe["modeled_radial_thickness"],
                  ghe["farfield_temperature"],
                  ghe["distance_to_farfield_temp"],
                  bc=ghe["initial_conditions"],
                  dt0=t_step,
                  adj_comps=adjacent_comps,
                  global_indices=(beg_idx,end_idx),
                  model=self)

    

    def _build_cavern(self,x_desc,xg,components):
        """
        Build the cavern (no multi-cavern capability yet!)
        
        """
        if self.is_test_mode:
            fluid_name = self.inputs["initial"]["fluid"]
            self.fluids = {}

            
            comp, massfracs, compSRK, fluid = process_CP_gas_string(fluid_name)
            
            fluid.update(CP.PT_INPUTS,self.inputs['initial']['pressure'],self.inputs['initial']['temperature'])
            
            self.fluids["cavern"] = fluid
            self.molar_masses = {}
            for pfluid in fluid.fluid_names():
                tempfluid = CP.AbstractState("HEOS",pfluid)
                tempfluid.set_mass_fractions([1.0])

                self.molar_masses[pfluid] = tempfluid.molar_mass()
                
            self.test_inputs["r_radial"] = (np.log(self.test_inputs['r_out']
                                                   /(self.inputs['cavern']['diameter']/2.0))
                                            / (2 * np.pi * self.inputs["cavern"]["height"] 
                                               * self.test_inputs['salt_therm_cond']))
            if '&' in fluid_name:
                fluid_str = "H2[1.0]&Methane[0.0]"  # fill methane cavern with new hydrogen
                rcomp, rmassfracs, rcompSRK, rfluid = process_CP_gas_string(fluid_str)
                rfluid.specify_phase(CP.iphase_gas)
                rfluid.update(CP.PT_INPUTS, 20e6, 310)
                
                self.fluids['cavern_well'] = rfluid

        else:
            prev_components = self.inputs['wells']
            ghe_name = self.inputs['cavern']['ghe_name']
            next_components = {ghe_name:self.inputs["ghes"][ghe_name]}
        
        cavern = self.inputs["cavern"]
        beg_idx = len(xg)
        

        #pure water for brine calculations
        water = CP.AbstractState("HEOS","Water")
        water.update(CP.PT_INPUTS,self.inputs['initial']['pressure'],self.inputs['initial']['liquid_temperature'])
        water.set_mole_fractions([1.0])
        self.water = water
        
        #geometry
        area_horizontal = np.pi * self.inputs['cavern']['diameter']**2/4
        area_vertical = np.pi * self.inputs['cavern']['diameter'] * self.inputs["cavern"]["height"]
        height_total = self.inputs["cavern"]["height"]
        height_brine = self.inputs['initial']['liquid_height']
        vol_brine = height_brine * area_horizontal
        height_total = self.inputs["cavern"]["height"]
        height_brine = self.inputs['initial']['liquid_height']
        t_brine = self.inputs['initial']['liquid_temperature']
        
        # repeat this to reach the correct brine density and pressure.
        (p_brine, solubility_brine,
         rho_brine) = brine_average_pressure(self.fluids['cavern'],water,height_total,height_brine,t_brine)
        water.update(CP.PT_INPUTS,p_brine,t_brine)
        (p_brine, solubility_brine,
         rho_brine) = brine_average_pressure(self.fluids['cavern'],water,height_total,height_brine,t_brine)
        
        vol_cavern = (height_total - height_brine) * area_horizontal
        m_cavern = vol_cavern * self.fluids['cavern'].rhomass()
        
        
        # add all cavern variables
        x_desc += ["Cavern gas temperature (K)"]
        xg += [self.inputs["initial"]["temperature"]]
        
        x_desc += ["Cavern wall temperature (K)"]
        xg += [self.inputs["initial"]["temperature"]]
        
        #x_desc += ["Cavern pressure at average height from cavern top to liquid surface (Pa)"]
        #xg += [self.inputs["initial"]["pressure"]]
        
        x_desc += ["Brine mass (kg)"]
        xg += [rho_brine * vol_brine]
        
        x_desc += ["Brine temperature (K)"]
        xg += [self.inputs["initial"]["temperature"]]
        
        #x_desc += ["Brine wall temperature (K)"]
        #xg += [self.inputs["initial"]["temperature"]]
        
        # mass balance for each gaseous fluid.
        mass_dict = calculate_component_masses(self.fluids['cavern'],
                                               mass=m_cavern,
                                               )
            
        for m_pure, fname in zip(mass_dict,self.fluids['cavern'].fluid_names()):
            xg += [m_pure]
            x_desc += [f"Cavern {fname} mass (kg)"]
        
        end_idx = len(xg)-1  # minus one because of 0 indexing!
        
        self.xg = xg
        self.xg_description = x_desc
        
        components["cavern"] = SaltCavern(self.inputs,global_indices=
                                          (beg_idx,end_idx),
                                          model=self)
        

    def _build_wells(self,x_desc,xg,components):
        """
        Warning! this function has been written without
        the number of control volumes in the well being considered!
        It only assigns input and output values as variables equivalent
        to a single control volume
        
        """
        
        if len(self.test_inputs) != 0:
            find_all_fluids(self)
            
            self.test_inputs["cavern"] = {"cavern":SaltCavern(self.inputs,global_indices=
                                              (-2,-1),
                                              model=self)}
            
        else:
            pass
            # this is undeveloped
        
        wells = self.inputs["wells"]
        
        
        
        for wname, well in wells.items():
            numcv = float(self.inputs['wells'][wname]['pipe_lengths'][0] / 
                          self.inputs['wells'][wname]['control_volume_length'])
            if numcv != 1.0:
                raise NotImplementedError("Only one control volume is allowed!"
                                          +" The well functionality is limited "
                                          +"to an adiabatic vertical column. A"
                                          +" future version will include "
                                          +"multiple control volumes along "
                                          +"the pipes with heat transfer losses/gains")
            
            len_pipe_diameters = len(well["pipe_diameters"])
            beg_idx = len(xg)
            
            if (well["ideal_pipes"] 
                # IDEAL WELL VARIABLE SETUP!
                and (len_pipe_diameters==4 
                and well["pipe_diameters"][0]==0.0 
                and well["pipe_diameters"][1] == 0.0)):
                
                for vname, valve in well["valves"].items():

                    #NOTE: The current implementation does not include
                    # any non-vertical pipe capability!
                    # we create a loop but there will only be one valve!

                    valve_inflow, cavern_inflow = reservoir_mass_flows(self, 0.0)
                    mdot = valve_inflow[wname][vname]
                    
                    molefracs = self.fluids[wname][vname].get_mole_fractions()
                    pure_fluid_names = self.fluids[wname][vname].fluid_names()
                    massflows = calculate_component_masses(
                        self.fluids[wname][vname],
                        mdot.sum())
                    
                    for massflow, pname in zip(massflows, pure_fluid_names):
                        xg += [massflow]
                        x_desc += [f"Well `{wname}` valve `{vname}` mass flow for {pname}"]
                    
                    if mdot.sum() > 0.0:
                        initial_temperature = valve["reservoir"]["temperature"]
                        initial_pressure = valve["reservoir"]["pressure"]
                    else:
                        initial_temperature = self.inputs["initial"]["temperature"]
                        initial_pressure = self.inputs["initial"]["pressure"]
                    
                    
                    temp_mat = PipeMaterial(well["pipe_roughness_ratios"][0],
                                            well["pipe_thermal_conductivities"][0])
                    
                    # only creating this here so that the adiabatic static
                    # column function can be used.
                    temp_vp = VerticalPipe(None,
                                           self.fluids[wname][vname],
                                           well["pipe_lengths"][0],
                                           well["pipe_lengths"][0],
                                           temp_mat,
                                           valve,
                                           vname,
                                           initial_pressure,
                                           initial_temperature,
                                           well["pipe_diameters"][0],
                                           well["pipe_diameters"][1],
                                           well["pipe_diameters"][2],
                                           well["pipe_diameters"][3],
                                           1,
                                           well["pipe_total_minor_loss_coefficients"])
                    
                    temp_fluid,pres_fluid = temp_vp.initial_adiabatic_static_column(
                        initial_temperature, initial_pressure, mdot)
                    
                    if isinstance(mdot,(float,int)):
                        inflow = mdot > 0.0
                    elif isinstance(mdot, np.ndarray):
                        inflow = mdot.sum() > 0.0
                    else:
                        raise ValueError("mdot must be an array or numeric!")
                    
                    if inflow:
                    
                        xg += [valve["reservoir"]["temperature"]] 
                        x_desc += ["Valve reservoir temperature (K)"]
                        xg += [valve["reservoir"]["pressure"]]
                        x_desc += ["Valve reservoir pressure (Pa)"]
                    
                        xg += [temp_fluid[-1]]
                        x_desc += ["Pipe entrance temperature to cavern (K)"]
                        xg += [pres_fluid[-1]]
                        x_desc += ["Pipe entrance pressure to cavern (K)"]
                        
                    else:
                        
                        xg += [temp_fluid[0]] 
                        x_desc += ["Valve reservoir temperature (K)"]
                        xg += [pres_fluid[0]]
                        x_desc += ["Valve reservoir pressure (Pa)"]
                    
                        xg += [self.inputs['initial']['temperature']]
                        x_desc += ["Pipe entrance temperature to cavern (K)"]
                        xg += [self.inputs['initial']['pressure']]
                        x_desc += ["Pipe entrance pressure to cavern (K)"]
                        

                
            else:
                raise NotImplementedError("We only have implemented ideal"
                                          +" wells with 1 pipe!")
            end_idx = len(xg)-1
            
            self.xg = xg
            self.xg_descriptions = x_desc
            
            components[wname] = Well(wname, well, self, (beg_idx,end_idx))
            
    def _connect_components(self):
        
        # cavern-ghe
        cghe = self.inputs['cavern']['ghe_name']
        self.components['cavern']._next_components = {}
        self.components['cavern']._next_components[cghe] = self.components[cghe]
        self.components[cghe]._prev_components = {}
        self.components[cghe]._next_components = {} # there are no next for GHE!
        self.components[cghe]._prev_components['cavern'] = self.components['cavern']
        
        # well-cavern
        self.components['cavern']._prev_components = {}
        for comp_name, comp in self.components.items():
            if isinstance(comp,Well):
                self.components['cavern']._prev_components[comp_name] = comp
                self.components[comp_name]._next_components = {}
                self.components[comp_name]._next_components['cavern'] = self.components['cavern']
                # Currently there are no prev components, in the future, maybe 
                # we could make this a pipe that leads to a compressor or leads
                # to a network of sub-surface storage salt caverns.
                self.components[comp_name]._prev_components = {}
                
        
    def _assign_max_from_adj_comp(self,adj_comps,name,multfact,arrind=None):
        val = 0.0

        for tup in adj_comps:
            adjacent_comp = tup[1]
            if "pipe_diameters" in adjacent_comp:
                if name == "height":
                    name = "pipe_lengths"
                if name == "diameter":
                    name = "pipe_diameters"
                if arrind is None:
                    val = np.max(np.array([val,adjacent_comp[name]*multfact]))
                else:
                    val = np.max(np.array([val, adjacent_comp[name][arrind]*multfact]))

            else:
                val = np.max(np.array([val,adjacent_comp[name]*multfact]))

        return val
    

    def _validate(self):
        """
        Validating the provided problem definition dict

        (originally from HydDown class)

        Raises
        ------
        ValueError
            If missing input is detected.
        """
        valid_tup = validator.validation(self.inputs)
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
            raise InputFileError(f"Error in input file {self.input_file}:\n\n" + error_string)

    def _read_input(self,inp):
            # enable reading a file or accepting a valid dictionary
        if isinstance(inp,str):
            with open(inp,'r',encoding='utf-8') as infile:
                input_dict = yaml.load(infile, Loader=yaml.FullLoader)
        elif isinstance(inp,dict):
            input_dict = inp
        else:
            raise TypeError("The input object 'inp' must be a string that "+
                            "gives a path to a uh2sc yaml input file or must"+
                            " be a dictionary with the proper format for uh2sc.")
        return input_dict

    def _find_ghe_adj_comp(self,ghe_name):
        """
        Find out how many adjacent components the axisymmetric
        ground heat exchange (GHE) has. Every component contributes
        a heat flux to the GHE
        """
        adj_comp = []
        for wname, well in self.inputs["wells"].items():
            if "ghe_name" in well:
                if well["ghe_name"] == ghe_name:
                    adj_comp.append((wname,well))
        if ghe_name == self.inputs["cavern"]["ghe_name"]:
            adj_comp.append(("cavern",self.inputs["cavern"]))
        return adj_comp
    
    def _bounds_characteristics(self):
        """
        These characteristic pressures and temperatures (which can) be
        controlled via user inputs provide a way for warnings and then
        an error to be thrown if the cavern simulation is outside of operational
        limits.
        
        """
        self._temperature_bounds = {"minor_warning":[None,None],
                                    "major_warning":[None,None],
                                    "error":[None,None]}
        self._pressure_bounds = {"minor_warning":[None,None],
                                    "major_warning":[None,None],
                                    "error":[None,None]}
        self._mass_flow_upper_limit = {"minor_warning":None,
                                    "major_warning":None,
                                    "error":None}

        opres = self.inputs['cavern']['overburden_pressure']
        
        if "min_operational_temperature" in self.inputs["cavern"]:
            self._temperature_bounds["minor_warning"][0] = self.inputs["cavern"]["min_operational_temperature"]
        else:
            self._temperature_bounds["minor_warning"][0] = 295
        if "max_operational_temperature" in self.inputs["cavern"]:
            self._temperature_bounds["minor_warning"][1] = self.inputs["cavern"]["min_operational_temperature"]
        else:
            self._temperature_bounds["minor_warning"][1] = 330
            
        if "min_operational_pressure_ratio" in self.inputs["cavern"]:
            self._pressure_bounds["minor_warning"][0] = self.inputs["cavern"]["min_operational_pressure_ratio"]*opres
        else:
            self._pressure_bounds["minor_warning"][0] = 0.5*opres
        if "max_operational_pressure_ratio" in self.inputs["cavern"]:
            self._pressure_bounds["minor_warning"][1] = self.inputs["cavern"]["max_operational_pressure_ratio"] *opres
        else:
            self._pressure_bounds["minor_warning"][1] = 0.8 * opres
            
        self._pressure_bounds["major_warning"] = [0.95*self._pressure_bounds["minor_warning"][0],
                                         1.05 * self._pressure_bounds["minor_warning"][1]]
        self._pressure_bounds["error"] = [0.9*self._pressure_bounds["minor_warning"][0],
                                         1.1 * self._pressure_bounds["minor_warning"][1]]

        
        self._temperature_bounds["major_warning"] = [self._temperature_bounds["minor_warning"][0]-10,
                                                     self._temperature_bounds["minor_warning"][1]+10]
        self._temperature_bounds["error"] = [self._temperature_bounds["minor_warning"][0]-20,
                                                     self._temperature_bounds["minor_warning"][1]+20]
        
        # flow limits
        if "max_volume_change_per_day" in self.inputs["cavern"]:
            max_vol_fraction = self.inputs["cavern"]["max_volume_change_per_day"]
        else:
            max_vol_fraction = 0.1
        
        cfl = self.fluids['cavern']
        
        temp = cfl.T()
        pres = cfl.p()
        
        
        cfl.update(CP.PT_INPUTS,self._pressure_bounds["minor_warning"][0],
                                self._temperature_bounds["minor_warning"][0])
        
        # calculate total volume
        total_volume = 0.25 * np.pi * ((self.inputs['cavern']['diameter']) ** 2
                                   ) * self.inputs['cavern']['height']
        
        mass_min = total_volume * cfl.rhomass()
        
        cfl.update(CP.PT_INPUTS,self._pressure_bounds["minor_warning"][1],
                                self._temperature_bounds["minor_warning"][1])
        
        mass_max = total_volume * cfl.rhomass()
        
        self.cavern_working_mass = mass_max - mass_min
        self.cavern_max_mass = mass_max
        self.cavern_min_mass = mass_min
        
        self._mass_flow_upper_limit["minor_warning"] = self.cavern_working_mass * max_vol_fraction / (3600 * 24)
        self._mass_flow_upper_limit["major_warning"] = 1.05 * self._mass_flow_upper_limit["minor_warning"]
        self._mass_flow_upper_limit["error"] = 1.1 * self._mass_flow_upper_limit["minor_warning"]
        
        cfl.update(CP.PT_INPUTS, pres, temp)
        
        self.bounds_checks = [self._temperature_bounds, 
                              self._pressure_bounds, 
                              self._mass_flow_upper_limit]
        
        
        
        
        
        
        
    


