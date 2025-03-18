from abc import ABC, abstractmethod
import numpy as np
from scipy.differentiate import jacobian

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

    def evaluate_jacobian(self,x=None):
        # must do this numerically and we follow the same routine
        if x is None:
            x = self.get_x()
        return jacobian(self.evaluate_residuals, x)
