#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 10:54:39 2025

@author: dlvilla
"""

import numpy as np
import CoolProp as CP
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt


def calculate_cavern_properties(gas_temperature, 
                                gas_brine_temp_diff, 
                                gas_pressure, 
                                cavern_height, 
                                cavern_diameter, 
                                total_brine_mass, 
                                gas_species_names, 
                                gas_species_mass_fractions, 
                                gas_density_func, 
                                brine_density_func):
    # Calculate the circular cross-sectional area of the cavern
    cross_sectional_area = np.pi * (cavern_diameter / 2) ** 2
    
    # Calculate the total volume of the cavern
    total_volume = cross_sectional_area * cavern_height
    
    # Calculate the gas density
    gas_density = gas_density_func(gas_species_names, gas_species_mass_fractions, gas_pressure, gas_temperature)
    
    # Calculate the gas volume
    gas_volume = 0
    brine_height = 0
    brine_temperature = gas_temperature - gas_brine_temp_diff
    
    brine_density = brine_density_func(brine_temperature,gas_pressure)
    
    brine_volume = total_brine_mass / brine_density
    
    brine_height = brine_volume / cross_sectional_area
    
    gas_volume = total_volume - brine_volume
    
    if gas_volume < 0:
        raise ValueError("The gas volume should never by less than 0 something is broken!")
    
    # Calculate the total mass of gas in the cavern
    total_gas_mass = gas_density * gas_volume
    
    return total_gas_mass, total_volume, gas_volume, cross_sectional_area

def generate_salt_cavern_samples(input_dict, 
                                 gas_density_func, 
                                 brine_density_func,
                                 gas_species_names,
                                 run_parallel=True):
    number_samples = input_dict['number_samples']
    gas_pressure_range = input_dict['gas_pressure_range']
    gas_temperature_range = input_dict['gas_temperature_range']
    height_range = input_dict['height_range']
    diameter_range = input_dict['diameter_range']
    gas_brine_temp_diff_range = input_dict['gas_brine_temp_diff_range']
    
    gas_pressures = np.random.uniform(gas_pressure_range[0], gas_pressure_range[1], number_samples)
    gas_temperatures = np.random.uniform(gas_temperature_range[0], gas_temperature_range[1], number_samples)
    heights = np.random.uniform(height_range[0], height_range[1], number_samples)
    diameters = np.random.uniform(diameter_range[0], diameter_range[1], number_samples)
    gas_brine_temp_diffs = np.random.uniform(gas_brine_temp_diff_range[0], gas_brine_temp_diff_range[1], number_samples)
    
    gas_species_mass_fractions = np.random.dirichlet(np.ones(len(gas_species_names.split('&'))), number_samples)  # Replace with actual gas species mass fractions
    
    brine_masses = np.zeros(number_samples)
    if run_parallel:
        def parallel_calculate_brine_mass(i):
            cavern_height = heights[i]
            cavern_diameter = diameters[i]
            gas_pressure = gas_pressures[i]
            gas_temperature = gas_temperatures[i]
            gas_brine_temp_diff = gas_brine_temp_diffs[i]
            
            # Calculate the brine mass
            brine_height = np.random.uniform(0, cavern_height)
            brine_density = brine_density_func(gas_temperature - gas_brine_temp_diff, gas_pressure)
            brine_volume = np.pi * (cavern_diameter / 2) ** 2 * brine_height
            return brine_density * brine_volume
    
        num_jobs = -1  # use all available cores
        brine_masses = np.array(Parallel(n_jobs=num_jobs)(delayed(parallel_calculate_brine_mass)(i) for i in range(number_samples)))
    else:
        brine_masses = np.zeros(number_samples)
        for i in range(number_samples):
            cavern_height = heights[i]
            cavern_diameter = diameters[i]
            gas_pressure = gas_pressures[i]
            gas_temperature = gas_temperatures[i]
            gas_brine_temp_diff = gas_brine_temp_diffs[i]
            
            # Calculate the brine mass
            brine_height = np.random.uniform(0, cavern_height)
            brine_density = brine_density_func(gas_temperature - gas_brine_temp_diff, gas_pressure)
            brine_volume = np.pi * (cavern_diameter / 2) ** 2 * brine_height
            brine_masses[i] = brine_density * brine_volume

    
    if run_parallel:
        def parallel_calculate_cavern_properties(i):
            try:
                total_gas_mass, total_volume, gas_volume, cross_sectional_area = calculate_cavern_properties(
                    gas_temperatures[i], 
                    gas_brine_temp_diffs[i], 
                    gas_pressures[i], 
                    heights[i], 
                    diameters[i], 
                    brine_masses[i], 
                    gas_species_names, 
                    gas_species_mass_fractions[i], 
                    gas_density_func, 
                    brine_density_func)
                
                out_list = [total_gas_mass, 
                            gas_temperatures[i], 
                            brine_masses[i], 
                            gas_temperatures[i] - gas_brine_temp_diffs[i], 
                            total_volume, 
                            cross_sectional_area, 
                            gas_volume]
                for frac in gas_species_mass_fractions[i]:
                    out_list.append(frac)
                
                return out_list
            except Exception as exc:
                print(f"Sample {i} failed: gas temp={gas_temperatures[i]}K gas_pressure={gas_pressures[i]} Pa ")
                print("")
                return None
    
        num_jobs = -1  # use all available cores
        results = Parallel(n_jobs=num_jobs)(delayed(parallel_calculate_cavern_properties)(i) for i in range(number_samples))
        dataset = [result for result in results if result is not None]
        dataset = np.array(dataset)
    else:
        dataset = []
        for i in range(number_samples):
            try:
                total_gas_mass, total_volume, gas_volume, cross_sectional_area = calculate_cavern_properties(
                    gas_temperatures[i], 
                    gas_brine_temp_diffs[i], 
                    gas_pressures[i], 
                    heights[i], 
                    diameters[i], 
                    brine_masses[i], 
                    gas_species_names, 
                    gas_species_mass_fractions[i], 
                    gas_density_func, 
                    brine_density_func)
                
                out_list = [total_gas_mass, 
                            gas_temperatures[i], 
                            brine_masses[i], 
                            gas_temperatures[i] - gas_brine_temp_diffs[i], 
                            total_volume, 
                            cross_sectional_area, 
                            gas_volume]
                for frac in gas_species_mass_fractions[i]:
                    out_list.append(frac)
                
                dataset.append(out_list)
            except Exception as exc:
                print(f"Sample {i} failed: gas temp={gas_temperatures[i]}K gas_pressure={gas_pressures[i]} Pa ")
                print("")
                pass
        dataset = np.array(dataset)
    
    return dataset



def plot_histograms(df):
    fig, axs = plt.subplots(len(df.columns), 1, figsize=(8, 12))

    for i, col in enumerate(df.columns):
        axs[i].hist(df[col], bins=50)
        axs[i].set_title(col)
        axs[i].set_xlabel(col)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    input_dict = {
        'number_samples': 100000,
        'gas_pressure_range': [1e6, 20e6],
        'gas_temperature_range': [283, 350],
        'height_range': [10, 1200],
        'diameter_range': [10, 500],
        'gas_brine_temp_diff_range': [-1, 5],
    }
    
    gas_species_names = "Methane&Ethane"
    gas_species_list = gas_species_names.split("&")
    
    def gas_density_func(gas_species_names, gas_species_mass_fractions, gas_pressure, gas_temperature):
        # Replace with actual gas density calculation
        gas = CP.AbstractState("HEOS",gas_species_names)
        gas.set_mass_fractions(gas_species_mass_fractions)
        gas.update(CP.PT_INPUTS,gas_pressure,gas_temperature)
        
        return gas.rhomass() 
        
    
    def brine_density_func(brine_temperature, brine_pressure):
        # Replace with actual brine density calculation
        brine = CP.AbstractState("HEOS","Water")
        brine.update(CP.PT_INPUTS,brine_pressure,brine_temperature)
        
        return brine.rhomass()

    dataset = generate_salt_cavern_samples(input_dict, gas_density_func, 
                                           brine_density_func, gas_species_names, True)
    
    column_names = ['Total Gas Mass (kg)', 'Gas Temperature (K)', 
                    'Brine Mass (kg)', 'Brine Temperature (K)', 
                    'Total Volume (m³)', 'Cross-Sectional Area (m²)', 
                    'Gas Volume (m³)']
    for species in gas_species_list:
        column_names.append(species + " Mass Fraction")
    
    df = pd.DataFrame(dataset, columns=column_names)

    plot_histograms(df)

    df.to_csv(f'salt_cavern_training_dataset_{input_dict["number_samples"]}.csv', index=False)
    
    pass
