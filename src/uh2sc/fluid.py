#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 08:50:04 2025

@author: dlvilla
"""

from CoolProp.CoolProp import AbstractState
from typing import List, Optional
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
    """

    def __init__(self, backend: str, fluids: str, fluid_fit: Optional[object] = None):
        self.state = AbstractState(backend, fluids)
        self._fluids = fluids
        self._fit = fluid_fit

    def _call(self, method_name: str, *args, **kwargs):
        """Call method from fit object if it exists, else use CoolProp."""
        if self._fit and hasattr(self._fit, method_name):
            return getattr(self._fit, method_name)(*args, **kwargs)
        return getattr(self.state, method_name)(*args, **kwargs)

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

    def set_mass_fractions(self, fractions: List[float]) -> None:
        """
        Set component mass fractions.

        :param fractions: List of mass fractions.
        """
        return self._call("set_mass_fractions", fractions)

    def get_mass_fractions(self) -> List[float]:
        """
        Get current component mass fractions.

        :return: List of mass fractions.
        """
        return self._call("get_mass_fractions")

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