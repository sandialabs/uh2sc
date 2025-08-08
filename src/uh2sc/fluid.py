#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 08:50:04 2025

@author: dlvilla
"""
import logging
import numpy as np
from typing import List, Optional
from CoolProp import CoolProp as CP
from uh2sc.abstract import AbstractThermoState

class FluidWithFitOption(AbstractThermoState):
    """
    Thermodynamic state wrapper using CoolProp.AbstractState or an override model.

    This class uses CoolProp as the default thermodynamic backend but optionally allows
    overriding its behavior with a user-provided object (`fluid_fit`) that implements
    one or more of the same method names.

    :param backend: CoolProp backend name (e.g., "HEOS", "REFPROP").
    :param fluids: Fluid or mixture name as a single string (e.g., "Methane&Ethane").
    :param fluid_fit: Optional object with method overrides matching this API.
    
    IMPORTANT: This class has been separated from AbstractState so that
    everything can be fed into "evaluate_residuals" as picklable. It can 
    then be instantiated each time using "set_state" and then end with "del_state"
    so that evaluate_jacobian can be run in parallel! You can use the .is_active 
    attribute to find out if the fluid has a current AbstractState object 
    
    """

    def __init__(self, fluid_tup: tuple , fluid_fit: Optional[object] = None, 
                 logger:logging.Logger = None, acceptable_error:float=0.1, PT=None):
        """
        Inputs
        ======
        
        fluid_tup: 3 or 4-tuple : [0] = string: backend model
                                  [1] = string: CoolPropFluids (components separated by '&')
                                  [2] = list[float]: mass fractions of each component
                                  [3] = list[float]: pressure (Pa), temperature (K)
                                  
        fluid_fit: UNDER DEVELOPMENT MUST BE NONE FOR NOW
                  Plans: This will be a ML model that has the exact same 
                  interfaces as AbstractThermoState and can be used in place 
                  of CoolProp's fluid properties. You can use the test_ml
                  to see how accurate the ML model is before and after 
                  solution so that error can be bounded and the model can
                  decide whether using machine learning was worth it. If not,
                  then the CoolProp (slower model) can be used instead.
                  
        logger: Allows messages to be logged in a logging file
        
        acceptable_error: float : threshold at which machine learning output
                 is rejected (%)

        """
        if not isinstance(fluid_tup, tuple):
            raise TypeError("The fluid_tup input must be a tuple!")
        
        self.state = None
        self.backend = fluid_tup[0]
        self._fluid_tup = fluid_tup
        self._fluids = fluid_tup[1]
        self._fit = fluid_fit
        self._acceptable_error = acceptable_error
        self._max_msgs = 100
        self._errors = []
        self._PT = PT
        if logger is None:
            self._logging = logging
        else:
            self._logging = logger
        self._num_msg = 0
        
    
    def set_state(self,abstract_state,PT=None):
        self.state = abstract_state(self._fluid_tup[0],self._fluid_tup[1])
        self.state.set_mass_fractions(self._fluid_tup[2])
        # assure gas state is specified explicitly before trying to update
        self.state.specify_phase(CP.iphase_gas)
        if PT is not None:
            self.state.update(CP.PT_INPUTS,PT[0],PT[1])
        elif self._PT is not None:
            self.state.update(CP.PT_INPUTS,self._PT[0],self._PT[1])
        self.is_active = True

        
        
    def del_state(self):
        self._fluid_tup = (self.backend,"&".join(self.fluid_names()),
                           self.get_mass_fractions(),[self.p(),self.T()])
        self.state = None
        self.is_active = False
        

    def _call(self, method_name: str, test_ml: bool = False, *args, **kwargs):
        
        """Call method from fit object if it exists, else use CoolProp."""
        if self._fit and hasattr(self._fit, method_name):
            ml_val = getattr(self._fit, method_name)(*args, **kwargs)
        if not self._fit or test_ml:
            cp_val = getattr(self.state, method_name)(*args, **kwargs)
            
        if test_ml:
            error = ml_val - cp_val
            perc_error = 100 * error / cp_val
            
            if perc_error > self._acceptable_error:
                self._errors.append({"Percent Error":perc_error,
                                     "Absolute Error":error,
                                     "Method":method_name,
                                     "Temperature":getattr(self.state,"T"),
                                     "Pressure":getattr(self.state,"P"),
                                     "Fluid Names":getattr(self.state,"fluid_names"),
                                     "Fluid Mass Fractions":getattr(self.state,"get_mass_fractions")})
                self._num_msg += 1
                self._logging.warning("The machine learning model has an "
                                      +f"error of {perc_error} which is above"
                                      +f" the acceptable error of {self._acceptable_error}"
                                      +": DETAILS={self._error[-1}")
                return cp_val
            else:
                if self._num_msg < self._max_msgs:
                    self._num_msg += 1
                    self._logging.debug("The machine learning model has an "
                                          +f"error of {perc_error} which is below"
                                          +f" the acceptable error of {self._acceptable_error}")
                return ml_val

        if not test_ml and not self._fit:
            return cp_val

        return ml_val
    
    def test_ml(self, method_name: str, *args, **kwargs):
        """
        test to see if the ml model meets the needed level of accuracy
        
        """
        num_error = len(self._errors)
        self._call(method_name, test_ml=True, *args, **kwargs)
        num_error_2 = len(self._errors)
        if num_error == num_error_2:
            return True
        return False 


    def update(self, input_pair: int, value1: float, value2: float) -> None:
        """
        Update the thermodynamic state with an input pair.

        :param input_pair: CoolProp input pair enum (e.g., PT_INPUTS).
        :param value1: First value (e.g., pressure in Pa).
        :param value2: Second value (e.g., temperature in K).
        """
        if self._fit and hasattr(self._fit, "update"):
            # TODO, handle this correctly
            self._fit.update(input_pair, value1, value2)
        else:
            self.state.update(input_pair, value1, value2)

    def rhomass(self) -> float:
        """
        Get mass density [kg/m³].

        :return: Mass density.
        """
        return self._call("rhomass")
    
    def _check_sum_1(self, fractions):
        frac_sum = np.array(fractions).sum()
        if frac_sum < 0.999 or frac_sum > 1.001:
            self.logging.warning(f"A set of fractions was input that does not "
                                 +f"sum to 1! sum({fractions})={frac_sum} "
                                 +"this has been normalized!")
        return [frac/frac_sum for frac in fractions]
    
    def specify_phase(self, phase_id):
        self.state.specify_phase(phase_id)
        

    def set_mass_fractions(self, fractions: List[float]) -> None:
        """
        Set component mass fractions.

        :param fractions: List of mass fractions.
        """
        fractions = self._check_sum_1(fractions)
        self.state.set_mass_fractions(fractions)
        if self._fit:
            self._fit.set_mass_fractions(fractions)
        return 

    def get_mass_fractions(self) -> List[float]:
        """
        Get current component mass fractions.

        :return: List of mass fractions.
        """
        return self.state.get_mass_fractions()
    
    def get_mole_fractions(self) -> List[float]:
        """
        Get current component mass fractions.

        :return: List of mass fractions.
        """
        return self.state.get_mole_fractions()
    

    def hmass(self) -> float:
        """
        Get specific enthalpy [J/kg].

        :return: Enthalpy.
        """
        return self._call("hmass")

    def compressibility_factor(self) -> float:
        """
        Get compressibility factor Z.

        :return: Compressibility factor.
        """
        return self._call("compressibility_factor")

    def gas_constant(self) -> float:
        """
        Get specific gas constant [J/kg/K].

        :return: Gas constant.
        """
        return self._call("gas_constant")

    def molar_mass(self) -> float:
        """
        Get molar mass [kg/mol].

        :return: Molar mass.
        """
        return self._call("molar_mass")

    def T(self) -> float:
        """
        Get temperature [K].

        :return: Temperature.
        """
        return self._call("T")

    def p(self) -> float:
        """
        Get pressure [Pa].

        :return: Pressure.
        """
        return self._call("p")

    def conductivity(self) -> float:
        """
        Get thermal conductivity [W/m/K].

        :return: Thermal conductivity.
        """
        return self._call("conductivity")

    def viscosity(self) -> float:
        """
        Get dynamic viscosity [Pa·s].

        :return: Dynamic viscosity.
        """
        return self._call("viscosity")

    def cpmass(self) -> float:
        """
        Get specific heat capacity at constant pressure [J/kg/K].

        :return: Specific heat (cp).
        """
        return self._call("cpmass")
    
    def cvmass(self) -> float:
        """
        Get specific heat capacity at constant volume [J/kg/K].

        :return: Specific heat (cv).
        """
        return self._call("cvmass")

    def isobaric_expansion_coefficient(self) -> float:
        """
        Get the isobaric expansion coefficient [1/K].

        :return: Expansion coefficient.
        """
        return self._call("isobaric_expansion_coefficient")

    def fluid_names(self) -> List[str]:
        """
        Get the list of fluid names.

        :return: Fluid names as list.
        """
        return self._fluids.split('&')