from abc import ABC, abstractmethod
import numpy as np
from scipy.differentiate import jacobian

class AbstractComponent(ABC):

    """
    AbstractComponent.

    Components are: 1, SaltCavern, Well,
    """
    @abstractmethod
    def evaluate_residuals(self,xg):
        """_summary_

        Args:
            xg numpy.array : global x vector for the entire
                             differiental/algebraic system
        """
        pass

    @abstractmethod
    def get_x(self):
        pass

    @abstractmethod
    def load_var_values_from(self,xg):
        pass

    def evaluate_jacobian(self,xg):
        # must do this numerically and we follow the same routine
        return jacobian(self.residuals, xg)
