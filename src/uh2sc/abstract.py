from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
from enum import Enum

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
    def evaluate_residuals(self,x=None):
        """
        Must first evaluate all interface equations for indices produced by interface_var_ind_prev_comp

        Then must evaluate all internal component equations


        Args:
            xg numpy.array : global x vector for the entire
                             differiental/algebraic system
        """
        pass

    @abstractmethod
    def get_x(self):
        pass

    @abstractmethod
    def load_var_values_from_x(self,xg):
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

    def evaluate_jacobian(self,x=None):
        # must do this numerically and we follow the same routine
        if False:
            # TODO: look into impelmenting an efficient sparse
            # Jacobian or even implementing a sparse Jacobian 
            # algorithm in Cython.
            if x is None:
                x = self.get_x()
                
            def compute_jacobian(func,x):
                return jax.jacfwd(func)(x)
            
            
            J = compute_jacobian(self.evaluate_residuals,x)
            return csr_matrix(J)
        else:
            if x is None:
                x = self.get_x()
            J = np.zeros((len(x),len(x)))
            
            r = self.evaluate_residuals(x)
            
            for idx in range(len(x)):
                dx = np.zeros(len(x))
                dx[idx] = 0.00001
                xdx = x + dx
                dfdx = (self.evaluate_residuals(xdx) - r)/0.00001
            
                J[:,idx] = dfdx
                
            return csr_matrix(J)
            
                
