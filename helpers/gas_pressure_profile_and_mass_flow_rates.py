#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:44:53 2025

@author: dlvilla
"""
import numpy as np


from CoolProp import CoolProp as CP
from CoolProp.CoolProp import PropsSI

def pressure_temperature_vs_depth(fluid, min_depth, max_depth, delta_depth, 
                                  start_temperature, geothermal_gradient,
                                  cavern_begin_depth, cavern_diameter, start_pressure):
    """
    Calculate pressure and temperature versus depth assuming a geothermal gradient.

    Parameters:
    fluid (str): Name of the fluid (e.g. "Air", "Nitrogen", etc.)
    delta_depth (float): Depth increment (in meters)
    start_temperature (float): temperature (in Kelvin)
    geothermal_gradient (float): Geothermal gradient (in K/m)
    cavern_begin_depth (float): Depth at which the cavern begins (in meters)
    cavern_diameter (float): Diameter of the cavern (in meters)
    start_pressure (float): Start pressure for the gas

    Returns:
    depth (numpy array): Array of depths (in meters)
    pressure (numpy array): Array of pressures (in Pa)
    temperature (numpy array): Array of temperatures (in Kelvin)
    total_mass (float): Total mass of fluid in the cavern (in kg)
    """
    # Check for errors
    if min_depth >= max_depth:
        raise ValueError("min_depth must be less than max_depth")
    if delta_depth <= 0:
        raise ValueError("delta_depth must be positive")
    if cavern_begin_depth < min_depth:
        raise ValueError("cavern_begin_depth must be greater than or equal to min_depth")
    if cavern_diameter <= 0:
        raise ValueError("cavern_diameter must be positive")

    # Convert geothermal gradient to K/m
    geothermal_gradient = geothermal_gradient  # already in K/m

    # Initialize arrays to store results
    depth = np.arange(min_depth, max_depth, delta_depth)  # assume max depth of 4000 m
    pressure = np.zeros_like(depth)
    temperature = np.zeros_like(depth)
    density = np.zeros_like(depth)

    # Set initial temperature and pressure
    temperature[0] = start_temperature
    pressure[0] = start_pressure
    
    model_depth = 0.5 * (cavern_begin_depth + max_depth)
    
    try:
    
        fluid.update(CP.PT_INPUTS,start_pressure, start_temperature)
        
    except:
        breakpoint()

    # Loop over depth increments
    for i in range(1, len(depth)):
        # Calculate new temperature
        temperature[i] = temperature[i-1] + geothermal_gradient * delta_depth
        
        fluid.update(CP.PT_INPUTS,pressure[i-1], temperature[i])

        # Calculate new density
        density[i] = fluid.rhomass()

        # Calculate new pressure
        pressure[i] = pressure[i-1] + density[i] * 9.81 * delta_depth
            
        

    # Calculate total mass
    cavern_mask = depth >= cavern_begin_depth
    total_mass = np.sum(density[cavern_mask] * np.pi * (cavern_diameter / 2)**2 * delta_depth)
    average_pressure = np.mean(pressure[cavern_mask])

    return depth, pressure, temperature, density, total_mass, average_pressure


if __name__ == "__main__":
    
    gas_types = ["H2", "Air", "Methane"]
    delta_depths = [0.01]
    
    min_pressure = 6001213 #Pa # casing seat minimum pressure
    max_pressure = 17001111 #Pa
    
    
    depth_casing_seat = 884.52
    depth_cavern_begin = 914.4 # m
    cavern_diameter = 25.6032
    
    depth_mid_point = 1066.8 #m
    
    depth_bottom = 1219.2 #m
    geothermal_gradient = 0.0219333333 # C/m  i.e. 0.012 F/ft
    ground_surface_temperature = 21.111111 # 70F
    
    start_temperature = depth_casing_seat * geothermal_gradient + ground_surface_temperature
    
    start_temperature = depth_casing_seat * geothermal_gradient + 273.15
    
    results = {}
    
    flow_in_time = 10 * 24 *3600
    flow_out_time = 20 * 24 * 3600
    
    
    for gas in gas_types:

        fluid = CP.AbstractState("HEOS", gas)
        
        results[gas] = {}
        
        for delta_depth in delta_depths:
        
            results[gas][str(delta_depth)] = {}
            for start_pressure,spstring in zip([min_pressure,max_pressure],["min_pressure","max_pressure"]):
                
                
                results[gas][str(delta_depth)][spstring] = {}
                depth, pressure, temperature, density, total_mass, avg_pres =  pressure_temperature_vs_depth(fluid, depth_casing_seat, depth_bottom, delta_depth, 
                                                  start_temperature, geothermal_gradient,
                                                  depth_cavern_begin, cavern_diameter, start_pressure)
                
                
                
                results[gas][str(delta_depth)][spstring]["total_mass"] = total_mass
                
            
            print(f"For {gas} and delta_depth={delta_depth} the flow rate is:\n\n")
            min_mass = results[gas][str(delta_depth)]["min_pressure"]["total_mass"]
            max_mass = results[gas][str(delta_depth)]["max_pressure"]["total_mass"]
            flow_in = (max_mass - min_mass)/flow_in_time
            flow_out = (max_mass - min_mass)/flow_out_time
            
            print(f"In: {flow_in}")
            print(f"Out: {flow_out}")
            print(f"Average Pressure (Pa): {avg_pres}")
            print("\n\n")
            print("-------------------------")
            print("\n\n")
            
            
            
            
                
            
            
                
                
            
            
        
        
        
    












 