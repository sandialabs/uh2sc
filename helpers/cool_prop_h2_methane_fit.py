#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 10:09:02 2025

@author: dlvilla
"""

import os
from uh2sc.fluidfit import RandomForestCoolPropMLFitter  # Replace with actual import path

# Define output path
output_dir = os.path.join(os.path.dirname(__file__), "fluid_fit_models")
os.makedirs(output_dir, exist_ok=True)

# Define configuration
temp_range = (290.0, 360.0)
temp_samples = 70

pres_range = (1e6, 31e6)  # From 1 MPa to 31 MPa in Pascals
pres_samples = 30

# Define CoolProp fluid mixture and mass fraction limits
fluids = "REFPROP::Hydrogen&Methane"
mass_frac_limits = [(0.05, 0.95), (0.05, 0.95)]  # H2 and CH4
frac_resolution = int((0.95 - 0.05) / 0.05 + 1)  # => 19 steps

# Run the fitter
fitter = RandomForestCoolPropMLFitter(
    output_path=output_dir,
    temp_range=temp_range,
    temp_samples=temp_samples,
    pres_range=pres_range,
    pres_samples=pres_samples,
    fluids=fluids,
    mass_frac_limits=mass_frac_limits,
    frac_resolution=frac_resolution
)

fitter.fit()