from abc import ABC, abstractmethod
from typing import List
import numpy as np
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from enum import Enum
import os

def compute_column(i, x, r, dx_val, residual_func):
    dx = np.zeros_like(x)
    dx[i] = dx_val
    xdx = x + dx
    dfdx = (residual_func(xdx) - r) / dx_val
    return i, dfdx


class AbstractThermoState(ABC):
    """
    Abstract base class mimicking CoolProp.CP.AbstractState.
    Defines the thermophysical property interface.
    """

    @abstractmethod
    def update(self, input_pair: int, value1: float, value2: float) -> None:
        """Update the thermodynamic state based on input pair and values."""
        pass

    @abstractmethod
    def rhomass(self) -> float:
        """Return mass density [kg/m³]."""
        pass

    @abstractmethod
    def set_mass_fractions(self, fractions: List[float]) -> None:
        """Set the mass fractions of the fluid components."""
        pass

    @abstractmethod
    def get_mass_fractions(self) -> List[float]:
        """Return the mass fractions of the fluid components."""
        pass
    
    @abstractmethod
    def get_mole_fractions(self) -> List[float]:
        """Return the mass fractions of the fluid components."""
        pass

    @abstractmethod
    def hmass(self) -> float:
        """Return specific enthalpy [J/kg]."""
        pass

    @abstractmethod
    def compressibility_factor(self) -> float:
        """Return the compressibility factor Z."""
        pass

    @abstractmethod
    def gas_constant(self) -> float:
        """Return the specific gas constant [J/kg/K]."""
        pass

    @abstractmethod
    def molar_mass(self) -> float:
        """Return the molar mass [kg/mol]."""
        pass

    @abstractmethod
    def T(self) -> float:
        """Return the temperature [K]."""
        pass

    @abstractmethod
    def p(self) -> float:
        """Return the pressure [Pa]."""
        pass

    @abstractmethod
    def conductivity(self) -> float:
        """Return thermal conductivity [W/m/K]."""
        pass

    @abstractmethod
    def viscosity(self) -> float:
        """Return dynamic viscosity [Pa·s]."""
        pass

    @abstractmethod
    def cpmass(self) -> float:
        """Return specific heat capacity at constant pressure [J/kg/K]."""
        pass
    
    @abstractmethod
    def cvmass(self) -> float:
        """Return specific heat capacity at constant volume [J/kg/K]."""
        pass    

    @abstractmethod
    def isobaric_expansion_coefficient(self) -> float:
        """Return the isobaric expansion coefficient [1/K]."""
        pass

    @abstractmethod
    def fluid_names(self) -> List[str]:
        """Return the list of fluid component names."""
        pass


class ComponentTypes(Enum):
    MODEL = 1
    GHE = 2
    WELL = 3
    CAVERN = 4

class AbstractComponent(ABC):

    """
    AbstractComponent.

    Components are: 1, SaltCavern, Well,
    """
    @property
    @abstractmethod
    def global_indices(self):
        """
        This property must give the indices that give the begin and end location
        in the global variable vector (xg)
        """
        pass

    @property
    @abstractmethod
    def previous_adjacent_components(self):
        """
        Interface variable indices for the previous component
        """
        pass

    @property
    @abstractmethod
    def next_adjacent_components(self):
        """
        interface variable indices for the next component
        """
        pass
    
    @property
    @abstractmethod
    def component_type(self):
        """
        A string that allows the user to identify what kind of component 
        this is so that specific properties and methods can be invoked

        """
        pass

    @abstractmethod
    def evaluate_residuals(self,x=None, get_independent_vars=False):
        """
        Must first evaluate all interface equations for indices produced by interface_var_ind_prev_comp

        Then must evaluate all internal component equations


        Args:
            x numpy.array : global x vector for the entire
                             differiental/algebraic system
        """
        pass

    @abstractmethod
    def get_x(self):
        """
        Must collect all of the local variables into an ordered list that
        will be integrated into the global variable vector.
        
        """
        pass
    
    
    @abstractmethod
    def independent_vars_descriptions(self):
        """
        gives a 1:1 description of each independent variable so that 
        a user can easily find what variables mean and column names 
        can be constructed for global output in a model.
        
        You must keep this ordered list the same as what evaluate_residuals
        returns when  get_independent_vars=True for the component being 
        modeled.
        
        """
        pass
    

    @abstractmethod
    def load_var_values_from_x(self,xg):
        """
        Values must be loaded from the global variable xg
        or else it is likely that you will NOT actually connect
        different components. You need to pull from all variable inside 
        this component and all variables that bridge between components from
        xg!!!
        
        You may have to rederive some quantities if the original xg values do not
        directly transfer. Otherwise you risk never getting an update for the
        next time step.
        """
        pass
    
    @abstractmethod
    def shift_solution(self):
        pass
    
    @abstractmethod
    def equations_list(self):
        """
        The user MUST put in the correct number of equation
        descriptions so that the system's meaning and order
        is well described same as the model's variables are
        in model.x_descriptions
        """
        pass



    def evaluate_jacobian(self, x=None, dx_val=1e-5):
        
        num_processors = 1
        if hasattr(self,"inputs"):
            if "calculation" in self.inputs:
                if "num_processors" in self.inputs["calculation"]:
                    num_processors = self.inputs["calculation"]["num_processors"]
                    cpu_count = os.cpu_count()
                    if num_processors > cpu_count:
                        num_processors = cpu_count
        
        
        if self._run_parallel and hasattr(self,"fluids"):
            # assure there are no active fluids with AbstractState which
            # are not pickleable! 
            for fluid_name, fluid in self.fluids.items():
                if isinstance(fluid,dict):
                    for fluid_name_2, fluid2 in fluid.items():
                        if fluid2.is_active:
                            fluid2.del_state()
                else:
                    if fluid.is_active:
                        fluid.del_state()
                
        
        if x is None:
            x = self.get_x()
        n = len(x)
        r = self.evaluate_residuals(x)
        J = np.zeros((n, n))
    
        if self._run_parallel:
            results = Parallel(n_jobs=num_processors)(
                delayed(compute_column)(i, x, r, dx_val, self.evaluate_residuals) for i in range(n)
            )
            for i, col in results:
                J[:, i] = col
        else:
            for i in range(n):
                dx = np.zeros(n)
                dx[i] = dx_val
                xdx = x + dx
                dfdx = (self.evaluate_residuals(xdx) - r) / dx_val
                J[:, i] = dfdx
    
        return csr_matrix(J)



    # def evaluate_jacobian(self,x=None):
    #     # must do this numerically and we follow the same routine
    #     if False:
    #         # TODO: look into impelmenting an efficient sparse
    #         # Jacobian or even implementing a sparse Jacobian 
    #         # algorithm in Cython.
    #         if x is None:
    #             x = self.get_x()
                
    #         def compute_jacobian(func,x):
    #             return jax.jacfwd(func)(x)
            
            
    #         J = compute_jacobian(self.evaluate_residuals,x)
    #         return csr_matrix(J)
    #     else:
    #         if x is None:
    #             x = self.get_x()
    #         J = np.zeros((len(x),len(x)))
            
    #         r = self.evaluate_residuals(x)
            
    #         for idx in range(len(x)):
    #             dx = np.zeros(len(x))
    #             dx[idx] = 0.00001
    #             xdx = x + dx
    #             dfdx = (self.evaluate_residuals(xdx) - r)/0.00001
            
    #             J[:,idx] = dfdx
                
    #         return csr_matrix(J)
            
                
