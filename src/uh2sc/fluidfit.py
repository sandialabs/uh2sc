#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 09:52:19 2025

@author: dlvilla
"""

import os
import numpy as np
import pandas as pd
import itertools
import joblib
import plotly.express as px
import plotly.io as pio
from typing import List, Tuple, Dict, Optional, Callable
from CoolProp.CoolProp import AbstractState
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import scoreatpercentile
from sklearn.base import RegressorMixin


class CoolPropMLFitter:
    """
    Fit scikit-learn ML models to emulate CoolProp AbstractState outputs.
    """

    METHODS = [
        "rhomass", "hmass", "compressibility_factor", "gas_constant", "molar_mass",
        "T", "p", "conductivity", "viscosity", "cpmass", "isobaric_expansion_coefficient"
    ]

    def __init__(
        self,
        output_path: str,
        temp_range: Tuple[float, float],
        temp_samples: int,
        pres_range: Tuple[float, float],
        pres_samples: int,
        fluids: str,
        mass_frac_limits: List[Tuple[float, float]],
        frac_resolution: int,
        model_factory: Callable[[], RegressorMixin],
        backend: str = "HEOS"
    ):
        
        """
        Parameters
        ----------
        output_path : str
            Path to save all output files and models.
        
        temp_range : Tuple[float, float]
            Temperature range as (min_temp, max_temp).
        
        temp_samples : int
            Number of temperature samples.
        
        pres_range : Tuple[float, float]
            Pressure range as (min_pressure, max_pressure).
        
        pres_samples : int
            Number of pressure samples.
        
        fluids : str
            CoolProp fluid string (e.g., 'Methane&Ethane').
        
        mass_frac_limits : List[Tuple[float, float]]
            List of (min, max) tuples for each gas mass fraction.
        
        frac_resolution : int
            Number of mass fraction samples for each gas.
        
        model_factory : Callable[[], RegressorMixin]
            Callable that returns a new instance of an sklearn regression model
            (e.g., lambda: GradientBoostingRegressor(n_estimators=100)).
        """
        
        self.output_path = output_path
        self.temp_range = temp_range
        self.temp_samples = temp_samples
        self.pres_range = pres_range
        self.pres_samples = pres_samples
        self.fluids = fluids
        self.mass_frac_limits = mass_frac_limits
        self.frac_resolution = frac_resolution
        self.backend = backend
        self.fluids_list = fluids.split('&')
        self.model_factory = model_factory

        os.makedirs(self.output_path, exist_ok=True)
        self.state = AbstractState(self.backend, self.fluids)

    def _generate_mass_fraction_grid(self) -> np.ndarray:
        grids = [np.linspace(low, high, self.frac_resolution) for low, high in self.mass_frac_limits]
        full_grid = np.array(list(itertools.product(*grids)))
        valid = np.isclose(full_grid.sum(axis=1), 1.0, atol=1e-4)
        bounded = np.all((full_grid >= 0) & (full_grid <= 1), axis=1)
        return full_grid[valid & bounded]

    def _evaluate_state(self, T: float, P: float, Y: np.ndarray) -> Dict[str, float]:
        try:
            self.state.set_mass_fractions(Y)
            self.state.update(0, T, P)
            return {method: getattr(self.state, method)() for method in self.METHODS}
        except Exception:
            return {method: np.nan for method in self.METHODS}

    def _create_dataset(self):
        temps = np.linspace(*self.temp_range, self.temp_samples)
        press = np.linspace(*self.pres_range, self.pres_samples)
        fractions = self._generate_mass_fraction_grid()

        full_grid = list(itertools.product(temps, press, fractions))
        temp_list, pres_list, frac_list = zip(*full_grid)
        frac_array = np.array(frac_list)

        X = pd.DataFrame(frac_array, columns=[f"Y_{f}" for f in self.fluids_list])
        X["T"] = temp_list
        X["P"] = pres_list

        outputs = [self._evaluate_state(t, p, y) for t, p, y in zip(temp_list, pres_list, frac_array)]
        Y = pd.DataFrame(outputs)

        data = pd.concat([X, Y], axis=1).dropna()
        return data

    def _create_test_dataset(self, n_samples: int) -> pd.DataFrame:
        temps = np.random.uniform(*self.temp_range, n_samples)
        press = np.random.uniform(*self.pres_range, n_samples)
        fractions = np.random.uniform(0, 1, (n_samples, len(self.mass_frac_limits)))
        fractions /= fractions.sum(axis=1)[:, None]

        for i, (lo, hi) in enumerate(self.mass_frac_limits):
            fractions[:, i] = lo + (hi - lo) * fractions[:, i]

        outputs = [self._evaluate_state(t, p, y) for t, p, y in zip(temps, press, fractions)]

        X = pd.DataFrame(fractions, columns=[f"Y_{f}" for f in self.fluids_list])
        X["T"] = temps
        X["P"] = press
        Y = pd.DataFrame(outputs)

        return pd.concat([X, Y], axis=1).dropna()

    def fit(self):
        data = self._create_dataset()
        X = data[["T", "P"] + [f"Y_{f}" for f in self.fluids_list]]
        models = {}
        stats = {}

        test_data = self._create_test_dataset(int(0.3 * len(X)))
        X_test = test_data[X.columns]

        for method in self.METHODS:
            y = data[method]
            y_test = test_data[method]

            model: RegressorMixin = self.model_factory()
            model.fit(X, y)
            y_pred = model.predict(X_test)

            abs_error = np.abs(y_test - y_pred)
            pct_error = np.abs((y_test - y_pred) / y_test) * 100

            models[method] = model
            stats[method] = {
                "mean_abs_error": float(np.mean(abs_error)),
                "max_abs_error": float(np.max(abs_error)),
                "min_abs_error": float(np.min(abs_error)),
                "percent_error_95": float(scoreatpercentile(pct_error, 95)),
                "percent_error_75": float(scoreatpercentile(pct_error, 75)),
                "percent_error_50": float(scoreatpercentile(pct_error, 50)),
                "mean_percent_error": float(np.mean(pct_error)),
            }

            fig = px.scatter(
                X_test.assign(error=abs_error),
                x="T", y="P", color="error",
                hover_data=X_test.columns.tolist(),
                title=f"Error Distribution for {method}"
            )
            pio.write_html(fig, os.path.join(self.output_path, f"error_plot_{method}.html"))

        ranges = {
            "T": self.temp_range,
            "P": self.pres_range,
            "mass_fractions": {f: lim for f, lim in zip(self.fluids_list, self.mass_frac_limits)}
        }

        master = {
            "models": models,
            "statistics": stats,
            "ranges": ranges
        }

        joblib.dump(models, os.path.join(self.output_path, "ml_models.joblib"))
        joblib.dump(stats, os.path.join(self.output_path, "ml_stats.joblib"))
        joblib.dump(ranges, os.path.join(self.output_path, "ml_ranges.joblib"))
        joblib.dump(master, os.path.join(self.output_path, "ml_master.joblib"))

        return master

class RandomForestCoolPropMLFitter(CoolPropMLFitter):
    def __init__(self,
                 output_path: str,
                 temp_range: Tuple[float, float],
                 temp_samples: int,
                 pres_range: Tuple[float, float],
                 pres_samples: int,
                 fluids: str,
                 mass_frac_limits: List[Tuple[float, float]],
                 frac_resolution: int):
        """
        Initializes a CoolPropMLFitter using RandomForestRegressor for all thermodynamic properties.

        Parameters
        ----------
        output_path : str
            Path to save all output files and models.

        temp_range : Tuple[float, float]
            Temperature range as (min_temp, max_temp).

        temp_samples : int
            Number of temperature samples.

        pres_range : Tuple[float, float]
            Pressure range as (min_pressure, max_pressure).

        pres_samples : int
            Number of pressure samples.

        fluids : str
            CoolProp fluid string (e.g., 'Methane&Ethane').

        mass_frac_limits : List[Tuple[float, float]]
            List of (min, max) tuples for each gas mass fraction.

        frac_resolution : int
            Number of mass fraction samples for each gas.
        """
        super().__init__(
            output_path=output_path,
            temp_range=temp_range,
            temp_samples=temp_samples,
            pres_range=pres_range,
            pres_samples=pres_samples,
            fluids=fluids,
            mass_frac_limits=mass_frac_limits,
            frac_resolution=frac_resolution,
            model_factory=lambda: RandomForestRegressor(n_estimators=100, random_state=42)
        )