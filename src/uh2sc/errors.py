# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:11:01 2023

@author: dlvilla
"""

class NumericAnomaly(Exception):
    pass

class MassTooLow(Exception):
    pass

class InputFileError(ValueError):
    pass

class NewtonSolverError(Exception):
    pass

class FluidStrBracketError(ValueError):
    pass

class FluidStrNumberError(ValueError):
    pass

class FluidStrNumbersDoNotAddToOneError(ValueError):
    pass

class FluidMixtureDoesNotExistInCoolProp(ValueError):
    pass

class FluidMixtureStateInfeasibleInCoolProp(ValueError):
    pass

class FluidMixturePresTempNotValid(ValueError):
    pass

class CavernStateOutOfOperationalBounds(ValueError):
    pass

class DeveloperError(Exception):
    pass
