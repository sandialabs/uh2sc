import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from CoolProp import AbstractState
# Example usage:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from uh2sc.utilities import calculate_cavern_pressure

class VectorFunctionEmulator:
    def __init__(self, model, bounds, test_size=0.2, random_state=42, n_jobs=-2):
        """
        Initialize the VectorFunctionEmulator class.

        Parameters:
        - model: A scikit-learn regressor model
        - bounds: A list of tuples representing the bounds for each input parameter
        - test_size: The fraction of the data to use for testing (default: 0.2)
        - random_state: The random seed for reproducibility (default: 42)
        - n_jobs: The number of jobs to run in parallel (default: -2, which means all but one CPU core)
        """
        self.model = model
        self.bounds = bounds
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    def sample_space(self, num_samples):
        """
        Sample the input space using random methods.

        Parameters:
        - num_samples: The number of samples to generate

        Returns:
        - X: A 2D numpy array of input samples
        """
        X = np.random.uniform(low=[b[0] for b in self.bounds], high=[b[1] for b in self.bounds], size=(num_samples, len(self.bounds)))
        return X

    def evaluate_function(self, x, func):
        """
        Evaluate the function at a given point.

        Parameters:
        - x: A 1D numpy array representing the input point
        - func: The vector function to emulate

        Returns:
        - y: A 1D numpy array representing the output value
        """
        return func(x)

    def generate_dataset(self, num_samples, func, parallel=True):
        """
        Generate a dataset by sampling the input space and evaluating the function.

        Parameters:
        - num_samples: The number of samples to generate
        - func: The vector function to emulate
        - parallel: Whether to run the function evaluations in parallel (default: True)

        Returns:
        - X: A 2D numpy array of input samples
        - y: A 2D numpy array of output samples
        """
        X = self.sample_space(num_samples)
        if parallel:
            y = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self.evaluate_function)(x, func) for x in X))
        else:
            y = np.array([self.evaluate_function(x, func) for x in X])
        return X, y

    def split_dataset(self, X, y):
        """
        Split the dataset into training and testing sets.

        Parameters:
        - X: A 2D numpy array of input samples
        - y: A 2D numpy array of output samples

        Returns:
        - X_train: A 2D numpy array of training input samples
        - X_test: A 2D numpy array of testing input samples
        - y_train: A 2D numpy array of training output samples
        - y_test: A 2D numpy array of testing output samples
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train the machine learning model.

        Parameters:
        - X_train: A 2D numpy array of training input samples
        - y_train: A 2D numpy array of training output samples
        """
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the performance of the trained model.

        Parameters:
        - X_test: A 2D numpy array of testing input samples
        - y_test: A 2D numpy array of testing output samples

        Returns:
        - score: The R-squared score of the model
        """
        score = self.model.score(X_test, y_test)
        return score

class CavernPressureEmulator(VectorFunctionEmulator):
    # ...

    def calculate_cavern_pressure_wrapper(self, x, m_cavern, t_cavern, m_brine, t_brine, volume_total, area):
        try:
            fluid = AbstractState("HEOS", "Methane&Ethane")
            water = AbstractState("HEOS", "Water")
            fluid.set_mass_fractions([x[0], 1-x[0]])
            return calculate_cavern_pressure(fluid,
                                             m_cavern*x[1],
                                             t_cavern + x[2],
                                             water,
                                             m_brine*x[3],
                                             t_brine + x[4],
                                             volume_total*x[5],
                                             area*x[6],
                                             1000)
        except Exception as e:
            print(f"Error calculating cavern pressure for x={x}: {e}")
            return None

    def generate_dataset(self, num_samples, m_cavern, t_cavern, m_brine, t_brine, volume_total, area, parallel=True):
        bounds = [(0, 1),  # mass fraction of the first component
                  (10,m_cavern),
                  (275,t_cavern),
                  (10,m_brine),
                  (275,t_brine),
                  (100,volume_total),
                  (10,area)]
        X = self.sample_space(num_samples)
        if parallel:
            y = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_cavern_pressure_wrapper)(X[i],
                                                                                                    m_cavern,
                                                                                                    t_cavern,
                                                                                                    m_brine,
                                                                                                    t_brine,
                                                                                                    volume_total,
                                                                                                    area)
                                                                                   for i in range(num_samples)))
        else:
            y = np.array([self.calculate_cavern_pressure_wrapper(X[i],
                                                                 m_cavern,
                                                                 t_cavern,
                                                                 m_brine,
                                                                 t_brine,
                                                                 volume_total,
                                                                 area)
                          for i in range(num_samples)])
        # Remove any None values from y
        mask = y != None
        X = X[mask]
        y = y[mask]
        return X, y

# Example usage:
m_cavern = 2e5
m_brine = 2e5
volume_total = 10e6
t_cavern = 320
t_brine = 320
area = 10000
bounds = [(0, 1),  # mass fraction of the first component
                  (10,m_cavern),
                  (275,t_cavern),
                  (10,m_brine),
                  (275,t_brine),
                  (100,volume_total),
                  (10,area)]

emulator = CavernPressureEmulator(RandomForestRegressor(), bounds)
X, y = emulator.generate_dataset(1000, m_cavern, t_cavern, m_brine, t_brine, volume_total, area, parallel=False)
X_train, X_test, y_train, y_test = emulator.split_dataset(X, y)
emulator.train_model(X_train, y_train)
score = emulator.evaluate_model(X_test, y_test)
print("R-squared score:", score)
