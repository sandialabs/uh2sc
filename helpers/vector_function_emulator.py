import os
import json
from scipy.io import savemat
import joblib

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from CoolProp import AbstractState
import CoolProp as CP
# Example usage:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from uh2sc.utilities import calculate_cavern_pressure
import matplotlib.pyplot as plt

class VectorFunctionEmulator:
    def __init__(self, model, test_size=0.2, random_state=42, n_jobs=-2):
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

    def generate_dataset(self, dataset_filepath, func, parallel=True):
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
        X = np.loadtxt(dataset_filepath, delimiter=',',skip_header=1)
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

    def calculate_cavern_pressure_wrapper(self, x):
        try:
            fluid = AbstractState("HEOS", "Methane&Ethane")
            water = AbstractState("HEOS", "Water")
            water.update(CP.PT_INPUTS,10e6,x[3])
            fluid.set_mass_fractions([x[7], 1-x[8]])
            return calculate_cavern_pressure(fluid,
                                             x[0], #m_cavern
                                             x[1], #t_cavern
                                             water,
                                             x[2], #m_brine
                                             x[3], #t_brine
                                             x[4], #volume_total
                                             x[5], #area
                                             x[6])
        except Exception as e:
            print(f"Error calculating cavern pressure for x={x}: {e}")
            return [None for idx in range(2)]
    
    def clean_data(self,data):
        cleaned_data = []
        for row in data:
            cleaned_row = []
            for value in row:
                if isinstance(value, np.ndarray):
                    cleaned_row.append(value[0])
                else:
                    cleaned_row.append(value)
            cleaned_data.append(cleaned_row)
        return np.array(cleaned_data)

    def generate_dataset(self, dataset_filepath, parallel=True,output_filepath=None):
        
        X = np.loadtxt(dataset_filepath, delimiter=',',skiprows=1)
        num_samples = X.shape[0]
        
        if output_filepath is None or not os.path.exists(output_filepath):
        
            if parallel:
                y = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_cavern_pressure_wrapper)(X[i])
                                                                                    for i in range(num_samples))
    
            else:
                y = [self.calculate_cavern_pressure_wrapper(X[i])
                              for i in range(num_samples)]
    
            # Remove any None values from y
            y = self.clean_data(y) 
            mask = y != None
            mask_x = np.tile(np.all(mask, axis=1)[:, None], (1, 9))
    
            Xshape = X.shape
            Xvec = X[mask_x]
            X = Xvec.reshape(int(len(Xvec)/Xshape[1]),Xshape[1])
            yshape = y.shape
            yvec = y[mask]
            y = yvec.reshape(int(len(yvec)/yshape[1]),yshape[1])
        
        else:
            
            y = np.genfromtxt(output_filepath, delimiter=',')
        return X, y
    
def plot_error_vs_test_samples(y_test, y_pred,sample_size,filedir, headers):
    num_outputs = y_pred.shape[1]
    fig, axs = plt.subplots(num_outputs, figsize=(10, 4*num_outputs))
    for i, ax in enumerate(axs):
        errors = np.abs(y_test[:, i] - y_pred[:, i])
        ax.hist(errors)
        ax.set_xlabel("Error " + headers(i))
        ax.set_ylabel(f'Count')
    plt.tight_layout()
    plt.savefig(os.path.join(filedir,f"error_for_model_{sample_size}.png"),dpi=300)

for sample_size in [100]: #,1000,10000,100000]:

    filedir = os.path.dirname(__file__)
    dataset_filepath = os.path.join(filedir,f"salt_cavern_training_dataset_{sample_size}.csv")
    output_filepath = os.path.join(filedir,f'salt_cavern_training_dataset_y_vals_{sample_size}.csv')
    
    emulator = CavernPressureEmulator(RandomForestRegressor())
    X, y = emulator.generate_dataset(dataset_filepath=dataset_filepath,parallel=True,output_filepath=output_filepath)
    
    
    np.savetxt(output_filepath, y, delimiter=',')
    
    X_train, X_test, y_train, y_test = emulator.split_dataset(X, y)
    emulator.train_model(X_train, y_train)
    score = emulator.evaluate_model(X_test, y_test)
    y_pred = emulator.model.predict(X_test)
    headers = ["gas pressure (Pa)","gas volume (m3)"]
    plot_error_vs_test_samples(y_test, y_pred,sample_size,filedir,headers)
    
    print("R-squared score:", score)
    
    # Get the model's parameters
    params = emulator.model.get_params()
    
    # Serialize the trained model
    joblib.dump(emulator, os.path.join(filedir,f'emulator_{sample_size}.joblib'))
    
    # Later, deserialize the trained model
    emulator = joblib.load(os.path.join(filedir,f'emulator_{sample_size}.joblib'))
    
    # Make predictions using the restored model
    y_pred2 = emulator.model.predict(X_test)
    
    error = y_pred - y_pred2
    
    print(f"The maximum error in reloading the model is: {error.max()}")


"""
10
100 0.699
1000 0.889
10000
100000 0.9887510621838383


"""

pass


