#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 18:50:51 2025

@author: dlvilla
"""

seconds_in_day = 24*3600

def build_neilson_time(num_days_in_year,num_days_in_rise,num_days_in_fall,trivial_time,flow_out):
    """
    I goofed up and reversed rise (rise = flow_out happening) and fall (flow_in happening)
    It works though with this reversed convention
    
    """
    my_list = [0]
    flow_list = [flow_out]
    day = 0
    num_times = 0
    flow_in = -flow_out * num_days_in_rise / num_days_in_fall
    
    while day < num_days_in_year:
        if num_days_in_year - day < num_days_in_rise:
            rise_days = num_days_in_year - day
        else:
            rise_days = num_days_in_rise
        if num_times == 0:
            ind = -1
        else:
            ind = -2
        num_times += 1
        my_list.append(my_list[ind]+rise_days*seconds_in_day)
        flow_list.append(flow_out)
        day += rise_days
        if day >= num_days_in_year:
            continue
        
        if day < num_days_in_year:
            my_list.append(my_list[-1]+trivial_time)
            flow_list.append(flow_in)
        
        if num_days_in_year - day < num_days_in_fall:
            fall_days = num_days_in_year - day
        else:
            fall_days = num_days_in_fall
            
        my_list.append(my_list[-2]+fall_days*seconds_in_day)
        flow_list.append(flow_in)
        day += fall_days
        
        if day < num_days_in_year:
            my_list.append(my_list[-1]+trivial_time)
            flow_list.append(flow_out)
    return my_list, flow_list
            

#H2
flow_out_rate = -1.4272448384707068
#Methane
flow_out_rate = -17.116716802624868

num_days_in_year = 360
num_days_in_rise = 10
num_days_in_fall = 20
trivial_time = 0.01
flow_out = flow_out_rate #kg/s


print("12 cycles:\n\n")
my_list, flow_list = build_neilson_time(num_days_in_year,num_days_in_rise,num_days_in_fall,trivial_time,flow_out)
print(my_list)

print("Flow list:")
print(flow_list)

print("\n\n---------------------------")


num_days_in_year = 360
num_days_in_rise = 30
num_days_in_fall = 60
trivial_time = 0.01
flow_out = flow_out_rate/4 #kg/s


print("4 cycles:\n\n")
my_list, flow_list = build_neilson_time(num_days_in_year,num_days_in_rise,num_days_in_fall,trivial_time,flow_out)
print(my_list)

print("Flow list:")
print(flow_list)

print("\n\n---------------------------")

num_days_in_year = 360
num_days_in_rise = 120
num_days_in_fall = 240
trivial_time = 0.01
flow_out = flow_out_rate/12 #kg/s


print("1 cycles:\n\n")
my_list, flow_list = build_neilson_time(num_days_in_year,num_days_in_rise,num_days_in_fall,trivial_time,flow_out)
print(my_list)

print("Flow list:")
print(flow_list)

print("\n\n---------------------------")
        
        
    
    
    