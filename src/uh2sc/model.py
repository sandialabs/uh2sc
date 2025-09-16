

from copy import deepcopy
import numpy as np
import logging
import yaml
import uuid
import time

from matplotlib import pyplot as plt
import pandas as pd

import pickle

import CoolProp.CoolProp as CP

from uh2sc import validator
from uh2sc.errors import (InputFileError, NewtonSolverError, DeveloperError,
                          CavernStateOutOfOperationalBounds)
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

class PickleHelp:
    @staticmethod
    def find_unpickleable_attrs(obj, path="Model", visited=None):
        
        no_unpickleable_found = True
        
        if visited is None:
            visited = set()
        if id(obj) in visited:
            return
        visited.add(id(obj))

        if isinstance(obj, (str, int, float, bool, type(None))):
            return  # primitive types are always pickleable

        try:
            pickle.dumps(obj)
        except Exception as e:
            no_unpickleable_found = False
            print(f"Unpickleable at {path}: {type(obj)} — {e}")

        if isinstance(obj, dict):
            for k, v in obj.items():
                PickleHelp.find_unpickleable_attrs(k, path=f"{path}[{repr(k)}]", visited=visited)
                PickleHelp.find_unpickleable_attrs(v, path=f"{path}[{repr(k)}]", visited=visited)
        elif hasattr(obj, "__dict__"):
            for attr_name, attr_val in vars(obj).items():
                PickleHelp.find_unpickleable_attrs(attr_val, path=f"{path}.{attr_name}", visited=visited)
        elif isinstance(obj, (list, tuple, set)):
            for i, item in enumerate(obj):
                PickleHelp.find_unpickleable_attrs(item, path=f"{path}[{i}]", visited=visited)
        
        return no_unpickleable_found



class TimeStepAdvisor:
    # THIS IS A GOOD IDEA THAT HAS NOT BEEN IMPLEMENTED YET AND NEEDS WORK
    # AND TESTING.
    
    def __init__(self,end_time,min_time_step,min_data_needed=20):
        self.w_log = None
        self.b_log = None
        self.coeffs_time = None
        self.coeffs_iter = None
        self.time_mean = None
        self.time_std = None
        self.iter_mean = None
        self.iter_std = None
        self.is_trained = False
        self.multipliers = np.linspace(0.5, 2.0, 16)
        self.data = np.zeros((int(end_time/min_time_step),7))
        self.data_column_names = ["step_num", 
                                  "sim_time",
                                  "time_step", 
                                  "num_iter", 
                                  "real_time", 
                                  "converged", 
                                  "input_rate_change"]
        
        self.end_time = end_time
        self._min_data_needed = int(min_data_needed)

    def _sigmoid(self, z):
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    def _fit_logistic_regression(self, X, y, lr=0.01, n_iter=500):
        m, n = X.shape
        w = np.zeros(n)
        b = 0.0

        for _ in range(n_iter):
            z = X @ w + b
            p = self._sigmoid(z)
            error = p - y
            w -= lr * (X.T @ error) / m
            b -= lr * np.mean(error)

        return w, b

    def _predict_logistic_proba(self, X):
        return self._sigmoid(X @ self.w_log + self.b_log)

    def _fit_linear_regression(self, X, y):
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        return np.linalg.lstsq(X_aug, y, rcond=None)[0]

    def _predict_linear(self, X, coeffs):
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        return X_aug @ coeffs

    def fit_and_predict(self, step_num, default_value=0.5):
        """
        Fit a simple model and predict what the best time step multiplication
        factor is 
        
        Inputs
        ======
        
        default_value : float : A value between 0.5 and 2.0 that is the
                   fall back multiplier if the algorithm fails to fit or make
                   a prediction.
                   
        Returns
        =======
        
        multiplier : float : The value predicted to be the best time step
                  multiplier for the given situation.
        
        """
        
        fit_worked = self._fit(step_num)
        if fit_worked:
            return self._predict(default_value)
        else:
            return default_value
        

    def _fit(self,step_num):
        """
        THIS IS NOT WORKING YET. EVERYTHING IS IN PLACE BUT THE MODEL
        IS TOO SIMPLE.
        
        Fit models using simulation data up to `end_time`.
        `data` must be an Nx7 array with columns:
        step_num, sim_time, time_step, num_iter, real_time, converged, input_rate_change
        """
        # I DO NOT HAVE THIS WORKING YET AND DO NOT HAVE TIME 
        return False
        
        # data = self.data[0:step_num,:]
        # end_time = self.end_time
        
        # if data.shape[0] < self._min_data_needed:
        #     return False


        # _, sim_time, time_step, num_iter, real_time, converged, input_rate_change = data.T
        # X = np.column_stack([time_step, input_rate_change, sim_time])
        # y_conv = converged
        # y_time = real_time
        # y_iter = num_iter

        # # Split 80/20
        # n = len(X)
        # split = int(0.8 * n)
        # X_train = X[:split]
        # y_conv_train = y_conv[:split]
        # y_time_train = y_time[:split]
        # y_iter_train = y_iter[:split]

        # # Train models
        # self.w_log, self.b_log = self._fit_logistic_regression(X_train, y_conv_train)
        # self.coeffs_time = self._fit_linear_regression(X_train, y_time_train)
        # self.coeffs_iter = self._fit_linear_regression(X_train, y_iter_train)

        # # Store normalization stats
        # self.time_mean = y_time_train.mean()
        # self.time_std = y_time_train.std() + 1e-6
        # self.iter_mean = y_iter_train.mean()
        # self.iter_std = y_iter_train.std() + 1e-6
        # self.X_test = X[split:]
        # self.is_trained = True

        # return True

    def _predict(self, default_value):
        """
        Predict optimal time step multiplier using test set from training.
        Returns a value between 0.5 and 2.0 (or `default_value` if not trained).
        """
        if not self.is_trained or self.X_test.shape[0] == 0:
            return default_value

        try:
            X_test = self.X_test
            num_multipliers = len(self.multipliers)
            num_samples = X_test.shape[0]

            # Broadcasted modifications
            X_mod = np.broadcast_to(X_test, (num_multipliers, num_samples, 3)).copy()
            X_mod[:, :, 0] *= self.multipliers[:, None]  # scale time_step

            # Flatten
            X_flat = X_mod.reshape(-1, 3)

            # Predictions
            p_conv = self._predict_logistic_proba(X_flat)
            pred_time = self._predict_linear(X_flat, self.coeffs_time)
            pred_iter = self._predict_linear(X_flat, self.coeffs_iter)

            # Normalize
            pred_time = (pred_time - self.time_mean) / self.time_std
            pred_iter = (pred_iter - self.iter_mean) / self.iter_std
            denom = pred_time + pred_iter + 1e-6

            utility = p_conv / denom
            utility_matrix = utility.reshape(num_multipliers, num_samples)
            mean_utility = utility_matrix.mean(axis=1)

            best_idx = np.argmax(mean_utility)
            return float(np.clip(self.multipliers[best_idx], 0.5, 2.0))

        except Exception as e:
            self.logging.warning(f"[WARN] Prediction failed: {e}")
            return default_value


ADJ_COMP_TESTING_NAME = "testing"

class Model(AbstractComponent):
    """
    Formulate a model that consistst of components. Each component has an abstract
     class as its base in uh2sc.abstract.AbstractComponent. Interface and boundary
     conditions are handled as additional equations. Each component has a method
     called "evaluate_residuals." Current valid components are:

     1. Cavern
     2. Arbitrary number of wells (FOR NOW ITS JUST 1 single pipe WELL!)
     3. Arbitrary number of 1-D axisymmetric heat transfer through the ground

     Future versions may include multiple caverns that are connected via
     surface pipes and may include pumps/compressors.
    """

    def __init__(self,inp,single_component_test=False,solver_options=None,
                 get_independent_vars=True,**kwargs):

        """
        Construct a combined model of 1) a salt cavern, 2) an arbitrary number
        of wells and 3) Radial heat transfer away from the salt cavern and wells.



        Inputs
        ======

        inp : str or dict : str = filepath to a yaml file that conforms to the
                                  uh2sc schema or:
                            dict = dict that conforms to the uh2sc schema
        
        single_component_test : bool (Default = False):
           Is for internal testing only. This gives the capacity to run 
           a model as a single component.

        solver_options : dict : Default = None,
           Provides a mechanism for changing solver options for the NewtonSolver
           class. See the "options" parameter for NewtonSolver.

        get_independent_vars : bool : Default = True
           Makes UH2SC also output independent variables such as pressure, evaporation rates, etc. Otherwise
           only the state variables (mass, mass_flow, temperature, heat flow) are output.

        """
        self.is_single_component_test = single_component_test
        self.logging = logging.getLogger(__name__)
        
        if isinstance(inp,str):
            self.input_file = inp
        else:
            self.input_file = None
        
        self.inputs = self._read_input(inp)
        
        if not single_component_test:
            # Deal with default values for optional inputs.
            if "cool_prop_backend" not in self.inputs["calculation"]:
                self.inputs["calculation"]["cool_prop_backend"] = "HEOS"
                
            if "machine_learning_acceptable_percent_error" not in self.inputs["calculation"]:
                self.inputs["calculation"]["machine_learning_acceptable_percent_error"] = 0.1
                
            if "run_parallel" not in self.inputs["calculation"]:
                self.inputs["calculation"]["run_parallel"] = False
            
            if not self.inputs["calculation"]["run_parallel"]:
                self.logging.warning("This simulation is running in serial which"
                    +" may be slower than running in parallel and usually should be"
                    +" confined to debugging! Ignore this message if you are"
                    +" running lots of uh2sc runs in parallel. uh2sc will fail if you set"
                    +" input['calculations']['run_parallel']=True in the input file"
                    +" and also try to run several simulation in parallel using joblib!")
            
            if not "num_processors" in self.inputs["calculation"]:
                self.inputs["calculation"]["num_processors"] = 1
                
            self.PT = [self.inputs["initial"]["pressure"],self.inputs["initial"]["temperature"]]
            
            self.time_step = self.inputs["initial"]["time_step"]
            
            self._end_time = self.inputs["calculation"]["end_time"]
            self._max_time_step = self.inputs["calculation"]["max_time_step"]
            self._min_time_step = self.inputs["calculation"]["min_time_step"]
            self._run_parallel = self.inputs["calculation"]["run_parallel"]
        
        if solver_options is None:
            #TOL will be replaced later on if use_relative_convergence=True
            solver_options = {"TOL": 1.0e-2,"LOG_PROGRESS":True,
                              "LOG_LEVEL":logging.WARNING, "MAXITER":100,
                              "USE_RELATIVE_CONVERGENCE":True}
            
        
        # logging will be setup differently
        # once self.run is invoked.
        
        self.converged_solution_point = False
        
        self.time = 0.0
        
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
                _str0 = ' '.join([ct.name for ct in ComponentTypes])
                raise ValueError("Only valid kwargs['type']:"
                        +_str0)
                
        if "USE_RELATIVE_CONVERGENCE" in solver_options:
            
            self._use_relative_convergence = solver_options["USE_RELATIVE_CONVERGENCE"]
            
        else:
            self._use_relative_convergence = False
            self.logging.info("Relative convergence factors not added to each"
            +" equation. This works but may take longer! You can set the "
            +"USE_RELATIVE_CONVERGENCE=True in the solver_options input to"
            +" change this.")       


        self._build(num_caverns,num_wells,num_ghes)
            
            
        if self._use_relative_convergence and not single_component_test:
            solver_options["TOL"] = self.solution_tolerance
            
 
        self.solver = NewtonSolver(solver_options)

        self._solutions = {}
        
        if not single_component_test:
            self._time_advice = TimeStepAdvisor(self.inputs["calculation"]["end_time"],
                                                self._min_time_step)
            # this drives the input_change_rate variable in time_advice. This will
            # need to change if another way of forcing change in the model (than
            # changing mass flow in/out is ever added (i.e. compressor pressure level etc...))
            well_ind = [idx for idx,desc in enumerate(self.xg_descriptions) 
                        if "Well" in desc and "mass flow for" in desc]
            if len(well_ind) != self.number_fluids:
                raise DeveloperError("There must only be 1 well index. The current"
                                 +" model needs updating if there is more than one or 0!")
            self._input_ind = well_ind[0]
            
        if get_independent_vars:
            self._independent_vars = {}
            ind_vars = self.evaluate_residuals(get_independent_vars=get_independent_vars)
            self._independent_vars[self.time] = ind_vars

        self._get_independent_vars = get_independent_vars

        if len(self.independent_vars_descriptions) != len(self._independent_vars[0.0]):
            raise DeveloperError("The number of independent variables does"
            +" not equal the number of independent variables. Somewhere in "
            +"the code a change has been made where an independent variable"
            +" is not described. You need to check the "
            +"indpendent_vars_descriptions and evaluate_residuals (with "
            +"get_independent_vars=True) property and method and figure out"
            +" which component has a mismatch in the number of variables"
            +" returned in comparison to the number of descriptions!")

        # used for analytics to assure variables in the model are realistic
        # with respect to actual salt cavern gas dynamics.
        if not single_component_test:
            self.components['cavern']._analytics()

    def _form_array(self):
        """
        Create a numpy array and column names for model output.

        """

        keys = np.array(list(self._solutions.keys()))

        values = np.array(list(self._solutions.values()))
        column_names = self.xg_descriptions
        column_names = ["Time (s)"] + column_names
        if self._get_independent_vars:
            keys2 = np.array(list(self._independent_vars.keys()))
            if not (keys == keys2).all():
                raise DeveloperError("The time steps for independnts_vars and"
                +" solutions have become misaligned. This is a developer "
                +"error and needs to be fixed by changing the source code!")
            column_names += self.independent_vars_descriptions
            values2 = np.array(list(self._independent_vars.values()))
            values_all = np.concatenate([values,values2],axis=1)
        else:
            values_all = values

        out_arr = np.concat([keys.reshape([len(keys),1]),values_all],axis=1)
        return column_names, out_arr
    
    def clear_unpickleable(self):
        """
        You have to do this in order for the model object to be
        pickleable.
        
        
        
        """
        for name, val in self.fluids.items():
            if isinstance(val,dict):
                for fluid_name, fluid in val.items():
                    if fluid.is_active:
                        fluid.del_state()
                    if hasattr(fluid,"_logging"):
                        fluid._logging = None
            else:
                if val.is_active:
                    val.del_state()
                if hasattr(val,"_logging"):
                    val._logging = None
                
                    
        if self.components['cavern']._fluid_m1.is_active:
            self.components['cavern']._fluid_m1.del_state()
        if hasattr(self.components['cavern']._fluid_m1,"_logging"):
            self.components['cavern']._fluid_m1._logging = None
            
        self.logging = None
        
        can_pickle = PickleHelp.find_unpickleable_attrs(self)
        
        if not can_pickle:
            raise DeveloperError("The Model class no longer can pickle. "
            +"Please regard the previous messages and add new unpickleable "
            +"attribues to the 'clear_unpickleable' method!")
        
    

    def pickle(self,pickle_filepath):
        self.clear_unpickleable()
        pickle.dump(self,open(pickle_filepath, 'wb'))
                

    def write_results(self,filename="uh2sc_results.csv"):
        """

        Write out results as a CSV

        """
        column_names, out_arr = self._form_array()
        np.savetxt(filename, out_arr, header=",".join(column_names), delimiter=',')

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

        # process for dataframe considerations
        if relative_time:
            index = out_arr[:,column_names.index("Time (s)")]
        else:
            # date considered.
            start_date = pd.to_datetime(start_date_str)
            delta = pd.to_timedelta(out_arr[:,column_names.index("Time (s)")],unit='s')

            index = start_date + delta

        df = pd.DataFrame(out_arr,index=index,columns=column_names)

        return df

    @property
    def independent_vars(self):
        return self._independent_vars


    @property
    def solutions(self):
        return self._solutions

    @property
    def component_type(self):
        return "model"

    def run(self,new_inp=None,log_file=None):
        """
        inputs:
            new_inp : dict - an input object that, if present, resets
                      all of the inputs but does not reinitialize the
                      model. This way, the model can be commanded to change
                      mass flows and such on the fly while maintaining the
                      current state of the cavern.

        """
        sim_start_time = time.perf_counter()
        
        if new_inp is not None:
            raise NotImplementedError("The ability to add new input "
                                 +"has not been completed!")
            self.inputs = self._read_input(new_inp)
            self._validate()

            for component_name, component in self.components.items():
                if component.component_type == 'Well':
                    new_well = Well(component_name,new_inp['wells'][component_name],self, component.global_indices)
                    self.components[component_name] = new_well
        
        if log_file is not None:
            # Remove the console handler from the root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self.solver._options["LOG_LEVEL"])
            
            RUN_ID = str(uuid.uuid4())[:8]
            logger = logging.getLogger(f"uh2sc logger run {RUN_ID}")
            logger.propagate = False
            
            # create a file handler
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(run_id)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            
            # Add the filter that injects the run_id
            class RunIDFilter(logging.Filter):
                def filter(self, record):
                    record.run_id = RUN_ID
                    return True
            
            file_handler.addFilter(RunIDFilter())
            logger.addHandler(file_handler)
            
            logger.setLevel(logging.DEBUG)
            
            self.logging = logger
            self.logging.info(f"  Begin of run {RUN_ID}")
            
        else:
            self.logging = logging

        # record initial state
        self._solutions[self.time] = deepcopy(self.get_x())
        self.time_m1 = self.time

        step_num = 0
        hit_final_time_step = False
        while self.time < self._end_time:
            x_org = self.get_x()
            
            #print before shifting.
            for xval, xdesc in zip(x_org,self.xg_descriptions):
                self.logging.info(f"Cavern state at {self.time}: {xdesc} = {xval}")
            
            

            self.logging.info(f"UH2SC model beginning calculations for time={self.time}.")
            # solve the current time step
            
            stime = time.perf_counter()
            
            try:
                tup = self.solver.solve(self)
            except CavernStateOutOfOperationalBounds as csob:
                self.logging.error("Exiting simulation early because of outofbounds")
                break
            
            etime = time.perf_counter()
            
            elapsed = etime - stime
            
            solver_converged = bool(tup[0])
            
            if solver_converged:
                # shift the time
                
                

                # update the state of the model
                self.shift_solution()
                if self._shift_solution_failed:
                    break

                self.logging.info(f"Completed time {self.time} with time step "
                                  +f"{self.time_step}. The simulation is "
                                  +f"{100 * self.time/self._end_time}% complete.")

                # shift time
                self.time += self.time_step

                # gather the results
                self._solutions[self.time] = deepcopy(self.get_x())
                
            if not self.is_single_component_test:
                input_rate_change = ((self._solutions[self.time][self._input_ind] 
                                     - self._solutions[self.time_m1][self._input_ind])/self.time_step)
                
                self._time_advice.data[step_num,:] = step_num,self.time,self.time_step,tup[-1],elapsed,tup[0],input_rate_change

            if solver_converged:
                
                self.time_m1 = self.time
                
                # increase the time step if it has shrunk
                if self.time_step < self._max_time_step:
                    
                    if step_num > 20:
                        
                        if not self.is_single_component_test:
                            proposed_time_step_mult = self._time_advice.fit_and_predict(step_num,
                                                                                        default_value=1.5)
                        else:
                            proposed_time_step_mult = 1.5
                    else:
                        proposed_time_step_mult = 1.5
                    
                    proposed_time_step = proposed_time_step_mult * self.time_step
                    if proposed_time_step < self._max_time_step:
                        self.time_step = proposed_time_step
                    else:
                        self.time_step = self._max_time_step
                        
                # assure time step does not exceed the end time.
                if self.time + self.time_step > self._end_time:
                    final_time_step = self._end_time - self.time
                    self.time_step = final_time_step
                    hit_final_time_step = True
                    
                    
                # gather independent variables if requested.
                if self._get_independent_vars:
                    self._independent_vars[self.time] = deepcopy(
                        self.evaluate_residuals(get_independent_vars=
                                                self._get_independent_vars))

            else:
                self.logging.info(f"Failed to complete {self.time} with time"
                                  +f" step {self.time_step}. Reducing time step"
                                  +f" to {0.5*self.time_step}")
                
                
                if self.time_step > self._min_time_step:
                    
                    
                    if step_num > 20:
                        
                        if not self.is_single_component_test:
                            proposed_time_step_mult = self._time_advice.fit_and_predict(step_num,
                                                                                        default_value=0.5)
                        else:
                            proposed_time_step_mult = 0.5
                    
                    else:
                        proposed_time_step_mult = 0.5
                    
                    self.time_step = proposed_time_step_mult*self.time_step
                    
                    # assure time step does not exceed the end time.
                    if self.time + self.time_step > self._end_time:
                        final_time_step = self._end_time - self.time
                        self.time_step = final_time_step
                        
                        hit_final_time_step = True
                    
                    # try resetting and trying a smaller time step
                    self.load_var_values_from_x(x_org)
                    if self._load_vars_failed:
                        break


                else:
                    msg = tup[1]
                    error_string = ("The Newton solver returned an"
                            +f" error for time {self.time} and the time step has"
                            +" reached the minimum allowed value. The solver"
                            +f" message is: {msg}")
                    if hit_final_time_step:
                        # we are going to let this go, perhaps a really small
                        # time step is being taken that is not conducive to 
                        # an implicit convergence.
                        
                        warning_string = ("The final time step of "
                        +f"{final_time_step} seconds failed to converge but "
                        +"the output has been provided anyway since the rest "
                        +"of the simulation converged. This sometimes happens"
                        +" when a very small step is needed to reach the end "
                        +f"time. The error is:\n\n {error_string}")
                        
                        self.logging.warning(warning_string)
                        
                    else:
                        self.logging.error(error_string)
                        break
                        
            step_num += 1
        
        sim_end_time = time.perf_counter()
        elapsed = sim_end_time - sim_start_time
        self.logging.info(f"Simulation complete in {elapsed:.4f}"
                     +f" seconds ({elapsed/3600:.4f} hours)")
        
        self.logging.propagate = True
        
        return elapsed

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
        if get_independent_vars:
            ind_vars = []

        if x is None:
            for _cname, component in self.components.items():

                # this is the single x behavior used by
                # local evaluations of residuals
                if get_independent_vars:
                    ind_vars += list(component.evaluate_residuals(
                        get_independent_vars=get_independent_vars))
                else:
                    residuals += list(component.evaluate_residuals())

        else:
            # this is for scipy where a matrix of all the different
            # vectors needed to evaluate the jacobian is fed in
            if x.ndim == 2:
                for xv in x:
                    v_residuals = []
                    for _cname, component in self.components.items():
                        v_residuals += list(component.evaluate_residuals(xv))
                        if get_independent_vars:
                            ind_vars += list(component.evaluate_residuals(
                                x=xv,
                                get_independent_vars=get_independent_vars))
                    residuals.append(v_residuals)
            elif x.ndim == 1:
                for _cname, component in self.components.items():

                    # this is the single x behavior used by
                    # local evaluations of residuals
                    residuals += list(component.evaluate_residuals(x))
                    if get_independent_vars:
                        ind_vars += list(component.evaluate_residuals(
                            x=x,
                            get_independent_vars=get_independent_vars))

        if get_independent_vars:
            return np.array(ind_vars)

        return np.array(residuals)


    def load_var_values_from_x(self, xg_new):
        for cname, component in self.components.items():
            try:
                self._load_vars_failed = False
                component.load_var_values_from_x(xg_new)
            except CavernStateOutOfOperationalBounds as csob:
                self.logging.error("Exiting simulation early because of outofbounds")
                self._load_vars_failed = True


    def shift_solution(self):
        for cname, component in self.components.items():
            try:
                self._shift_solution_failed = False
                component.shift_solution()
            except CavernStateOutOfOperationalBounds as csob:
                self.logging.error("Exiting simulation early because of outofbounds")
                self._shift_solution_failed = True

    @property
    def independent_vars_descriptions(self):
        """
        gives a 1:1 description of each independent variable so that
        a user can easily find what variables mean and column names
        can be constructed for global output in a model.
        """
        desc = []
        for cname, component in self.components.items():
            desc += component.independent_vars_descriptions

        return desc

    def equations_list(self):
        e_list = []
        for cname, component in self.components.items():
            e_list += component.equations_list()
        return e_list

    def plot_solution(self,variables,show=False):
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
                    raise ValueError("Only integers less than the number "
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
            if show:
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
        if not self.is_test_mode:
            # if not, then we are in a test mode.
            find_all_fluids(self)
            self.number_fluids = len(self.fluids["cavern"].fluid_names())

        if num_caverns != 0:
            if not hasattr(self, "fluids"):
                find_all_fluids(self)
            
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
                # backfit missing stuff to get the solution to work.
                self.inputs["calculation"] = {}
                self.inputs["calculation"]["end_time"] = self.test_inputs["end_time"]
                self.inputs["calculation"]["time_step"] = self.test_inputs["dt"]
                self.time_step = t_step
                self._end_time = self.test_inputs["end_time"]
                self._max_time_step = t_step * 1.5
                self._run_parallel = False
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

                t_step = self.time_step

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

            self.time_step = self.inputs["initial"]["time_step"]
            comp, massfracs, compSRK, fluid = process_CP_gas_string(fluid_name,self.inputs["calculation"]["cool_prop_backend"],self)


            fluid.set_state(CP.AbstractState)
            fluid.update(CP.PT_INPUTS,self.inputs['initial']['pressure'],self.inputs['initial']['temperature'])
            
            

            self.fluids["cavern"] = fluid
            for pfluid in fluid.fluid_names():
                tempfluid = CP.AbstractState("HEOS",pfluid)
                tempfluid.set_mass_fractions([1.0])
                
            fluid.del_state()

            self.test_inputs["r_radial"] = (np.log(self.test_inputs['r_out']
                                                   /(self.inputs['cavern']['diameter']/2.0))
                                            / (2 * np.pi * self.inputs["cavern"]["height"]
                                               * self.test_inputs['salt_therm_cond']))
            if '&' in fluid_name:
                fluid_str = "H2[1.0]&Methane[0.0]"  # fill methane cavern with new hydrogen
                rcomp, rmassfracs, rcompSRK, rfluid = process_CP_gas_string(fluid_str,self.inputs["calculation"]["cool_prop_backend"],self)
                rfluid.set_state(CP.AbstractState)
                rfluid.specify_phase(CP.iphase_gas)
                rfluid.update(CP.PT_INPUTS, 20e6, 310)
                rfluid.del_state()

                self.fluids['cavern_well'] = rfluid
                
            self._end_time = self.inputs["calculation"]["end_time"]
            self._run_parallel = True
            self._max_time_step = self.inputs["calculation"]["max_time_step"]

        else:
            self.prev_components = self.inputs['wells']
            ghe_name = self.inputs['cavern']['ghe_name']
            self.next_components = {ghe_name:self.inputs["ghes"][ghe_name]}

        beg_idx = len(xg)
        
        #instantiate for some calculations!
        PT_list = [self.inputs['initial']['pressure'],self.inputs['initial']['temperature']]
        self.fluids["cavern"].set_state(CP.AbstractState,PT_list)
        
        
        #pure water for brine calculations
        water = CP.AbstractState("HEOS","Water")
        water.update(CP.PT_INPUTS,self.inputs['initial']['pressure'],self.inputs['initial']['liquid_temperature'])
        water.set_mole_fractions([1.0])

        #geometry
        area_horizontal = np.pi * self.inputs['cavern']['diameter']**2/4
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
        
        # no longer pass an abstract state so that we can run evaluate_jacobian in parallel
        self.water = ("HEOS","Water",[1.0],p_brine,t_brine)


        # add all cavern variables
        x_desc += ["Cavern gas temperature (K)"]
        xg += [self.inputs["initial"]["temperature"]]
        
        # the initial wall temperature is estimated to be the average of the farfield temperature 
        # and the initial gas temperature
        x_desc += ["Cavern wall temperature (K)"]
        

        if self.is_test_mode:
            farfield_temp = self.inputs["initial"]["temperature"]
        else:
            # only 1 GHE is allowed for now!
            ghe_dict = self.inputs["ghes"]
            for ghe_name, ghe_dict2 in ghe_dict.items():
                farfield_temp = ghe_dict2["farfield_temperature"]
        
        # CHANGE NEEDED - You should be able to calculate the amount of 
        # heat that can possibly be transfered and how much that will 
        # allow flux to become. If the flux cannot match the temperature
        # difference, then you will not converge. A smarter model will not
        # allow this to happen.
        if farfield_temp - self.inputs["initial"]["temperature"] > 5.0:
            self.logging.warning("There is a greater than 5K difference between"+
            " the initial gas temperature {self.inputs['initial']['temperature']}"
            +" and the farfield temperature {farfield_temp}. If the GHE doesn't"
            +" have enough elements, the model will be unlikely to converge "
            +"because flux will not be able to heat/cool the wall off fast "
            +"enough!")
        
        xg += [(self.inputs["initial"]["temperature"]+farfield_temp)/2]

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
        
        # get rid of AbstractState so that we can run parallel later.
        self.fluids['cavern'].del_state()

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
            # use this section to fill in stuff that isn't built 
            # because your just trying to run a well in an isolated sense.
            find_all_fluids(self)
            
            # pulled from Nieland verification.
            self.residual_normalization = {'cavern_gas_energy': 3591431272594.7725, 
                                                'cavern_gas_mass': 726543.9913691991, 
                                                'cavern_pressure': 20027490.643315244, 
                                                'temperature_norm': 110, 
                                                'heat_flux_norm': 20783745.790479008, 
                                                'mass_flow_norm': 4.204536987090273, 
                                                'brine_mass': 726543.9913691991, 
                                                'brine_energy': 3591431272594.7725}
            
            self.PT = [self.inputs['initial']['pressure'],self.inputs['initial']['temperature']]

            self.number_fluids = len(self.fluids['cavern'].fluid_names())
            self.test_inputs["cavern"] = {"cavern":SaltCavern(self.inputs,global_indices=
                                              (-2,-1),
                                              model=self)}
            self._end_time = self.inputs['calculation']['end_time']
            self._run_parallel = True
            self.time_step = self.inputs['initial']['time_step']
            self._max_time_step = self.time_step
            self._min_time_step = 100

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
                        
                    #initialize state for some calculations
                    self.fluids[wname][vname].set_state(CP.AbstractState,self.PT)

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
                    
                    # VerticalPipe opens up the fluid again
                    # making a need to close it now
                    # vertical pipe closes this.
                    self.fluids[wname][vname].del_state()
                    
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
                    
                    # get rid of AbstractState so we can parallelize
                    

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
            self._temperature_bounds["minor_warning"][1] = self.inputs["cavern"]["max_operational_temperature"]
        else:
            self._temperature_bounds["minor_warning"][1] = 340

        if "min_operational_pressure_ratio" in self.inputs["cavern"]:
            min_pres_ratio = self.inputs["cavern"]["min_operational_pressure_ratio"]
            if min_pres_ratio == 0:
                # atmospheric will be the lowest allowed to avoid gas model problems
                self._pressure_bounds["minor_warning"][0] = 100000
                
            else:
                
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
        cfl.set_state(CP.AbstractState)
        

        cfl.update(CP.PT_INPUTS,self._pressure_bounds["minor_warning"][0],
                                self._temperature_bounds["minor_warning"][0])

        # calculate total volume
        total_volume = 0.25 * np.pi * ((self.inputs['cavern']['diameter']) ** 2
                                   ) * self.inputs['cavern']['height']

        mass_min = total_volume * cfl.rhomass()
        energy_min = cfl.hmass() * mass_min

        cfl.update(CP.PT_INPUTS,self._pressure_bounds["minor_warning"][1],
                                self._temperature_bounds["minor_warning"][1])

        mass_max = total_volume * cfl.rhomass()
        energy_max = cfl.hmass() * mass_max
        
        energy_norm = energy_max - energy_min
        mass_norm = mass_max - mass_min
        pressure_norm = self._pressure_bounds["error"][1] - self._pressure_bounds["error"][0]
        temperature_norm = self._temperature_bounds["error"][1] - self._temperature_bounds["error"][0]
        
        self.cavern_working_mass = mass_max - mass_min
        self.cavern_max_mass = mass_max
        self.cavern_min_mass = mass_min
        
        self._mass_flow_upper_limit["minor_warning"] = self.cavern_working_mass * max_vol_fraction / (3600 * 24)
        self._mass_flow_upper_limit["major_warning"] = 1.05 * self._mass_flow_upper_limit["minor_warning"]
        self._mass_flow_upper_limit["error"] = 1.1 * self._mass_flow_upper_limit["minor_warning"]
        
        mass_flow_norm = self._mass_flow_upper_limit["error"]
        
        flux_norm = mass_flow_norm * cfl.hmass()
        
        # brine norms
        _bw = CP.AbstractState("HEOS","Water")
        _bw.update(CP.PT_INPUTS,self._pressure_bounds["error"][1],self._temperature_bounds["error"][1])
        
        extra_factor = 1
        self.residual_normalization = {"cavern_gas_energy":extra_factor * flux_norm * self.inputs['calculation']['max_time_step'],
                                       "cavern_gas_mass":extra_factor * mass_flow_norm * self.inputs['calculation']['max_time_step'],
                                       "cavern_pressure":extra_factor * pressure_norm,
                                       "temperature_norm":extra_factor * temperature_norm,
                                       "heat_flux_norm":extra_factor * flux_norm,
                                       "mass_flow_norm":extra_factor * mass_flow_norm,
                                       "brine_mass": extra_factor * mass_flow_norm * self.inputs['calculation']['max_time_step'],
                                       "brine_energy": extra_factor * flux_norm * self.inputs['calculation']['max_time_step']}

        if "solution_tolerance" in self.inputs["calculation"]:
            tol_factor = self.inputs["calculation"]["solution_tolerance"]
        else:
            tol_factor = 1.0e-5

        self.solution_tolerance = np.min(tol_factor * np.array([energy_min /energy_norm,
                                                         mass_min/mass_norm,
                                                         self._temperature_bounds["error"][0]/temperature_norm,
                                                        self._pressure_bounds["error"][0]/pressure_norm,
                                                        1.0]))
        
        
        cfl.del_state()

        self.bounds_checks = [self._temperature_bounds,
                              self._pressure_bounds,
                              self._mass_flow_upper_limit]
        
        
        
