# uh2sc Copyright (c) 2024
# Daniel Villa, dlvilla@sandia.gov
# 
# This file was originally derived from HydDown's validator.py file but the
# fundamental approach has been significantly extended including putting all
# schemas into yaml files and enabling an arbitrary set of wells to be added
# to a salt cavern.

from cerberus import Validator
import numpy as np
from datetime import datetime 
import yaml
import os
from CoolProp import CoolProp as CP
from warnings import warn

class _LocalErrorObj:
    """
    mimic the cerebrus.Validator error attributes for local error checks
    
    """
    def __init__(self):
        self.errors = {}
        
    def validate(self, field,message, boolean):
        self.errors[field] = message
        return boolean
    
def local_check_pipe_arrays(well,name,validate_dict,retval):
    num_pipe = len(well['pipe_lengths'])
    array_name = name+"_array_count_error"
    validate_dict[array_name] = _LocalErrorObj()
    
    if (num_pipe != len(well['pipe_roughness_ratios']) or
        num_pipe != len(well['pipe_thermal_conductivities']) or
        num_pipe != len(well['pipe_total_minor_loss_coefficients'])):
        
        error_msg = ("The ('pipe_lengths, pipe_thermal_conductivities, "+
        "and pipe_rougnness_ratios) arrays must all have the same number"+
        " of elements!")
        succeeded = False
    elif (3 * num_pipe + 1) != len(well['pipe_diameters']):
        error_msg = ("The 'pipe diameters array must have 4 entries for"+
                     " the first pipe and 3 entries for every additional"+
                     " pipe. For the first pipe the entries are: [min "+
                     "outer diameter, min inner diameter, max inner diam"+
                     "eter, max outer diameter]. All additional pipes"+
                     " only have three entries because the first diameter"+
                     " is assumed to be the 'max outer diameter of the"+
                     " previous pipe. If starting with a simple pipe with"+
                     " only one wall (i.e. the first pipe is not annualar)"+
                     " the first two entries should be 0")
        succeeded = False
    elif well['pipe_diameters'][0] != 0.0 or well['pipe_diameters'][1] != 0.0:
        error_msg = ("Having a non-circular pipe as the first pipe is not currently"+
                  " supported by uh2sc!")
        succeeded = False
    else:
        error_msg = "No error"
        succeeded = True
    
    retval[array_name] = validate_dict[array_name].validate(array_name,error_msg, succeeded)
    


def validation(inp):
    """
    Validate inp
    
    Parameters
    ----------
    inp : dict 
        Structure holding input 
    
    Return
    ----------
    retval : bool 
        True for success, False for failure
    
    """
    
    def h_inner_check(field,value,error):
        if isinstance(value,str):
            if value != "calc":
                error(field,"'calc' is the only option besides numeric values")
        else:
            if value <= 0.0 or value >= 1e5:
                error(field,"Value is outside the physical range of possibilities 0.0 <= h_inner <= 1e5 W/m2/K")

    def is_isocalendardate(field,value,error):
        try:
            datetime.fromisoformat(value)
        except:
            error(field,"Must be an ISO format datetime: 'YYYY-MM-DD' where "+
                  " YYYY is the four integer year, MM is the two integer month "+
                  "code with a leading zero when needed, and DD is the two"+
                  " integer day code with a leading zero when needed. Example:"+
                  " 2024-01-01")
            
    def is_increasinglist(field,value,error):
        try:
            if value[0] == 0.0 and value[1] == 0.0:
                value_mod = value[2:]
            else:
                value_mod = value
            if np.array([val<=0.0 for val in np.diff(value_mod)]) > 0.0:        
               error(field,"Must be a list of increasing numbers!")
        except:
            error(field,"Must be a list of increasing numbers!")
            
    def check_pipe_lengths(field,value,error):
        
        depth = inp['cavern']['depth']
        height = inp['cavern']['height']
        for val in value:        
            if depth > val or val > depth + height:
                error(field,"pipe lengths must be longer than the depth of the"+
                      " cavern but not deeper than the lowest point in the"+
                      " cavern (i.e., "+str(depth)+" <= pipe_length <= "+str(depth+height)+".")
                
    def check_pipe_conductivities(field,value,error):
        diamond_thermal_conductivity_WpmpK = 2200
        for val in value:
            if val > diamond_thermal_conductivity_WpmpK or val < 0.0001:
                error(field,"pipe thermal conductivities must be between "+
                      str(0.0001)+ " and " +
                      str(diamond_thermal_conductivity_WpmpK))
                
    def check_pipe_roughness(field,value,error):
        max_roughness = 0.1
        for val in value:
            if val > max_roughness or val < 0.0:
                error(field,"pipe thermal conductivities must be between "+
                      str(0.0)+ " and " +
                      str(max_roughness))
                
    def check_pipe_minor_loss_coef(field,value,error):
        max_coef = 1e6
        for val in value:
            if val > max_coef or val < 0.0:
                error(field,"pipe minor loss coefficients must be between "+
                      str(0.0)+ " and " +
                      str(max_coef))
                
    def fluid_string_check(field,value,error):
        # verify all species are allowed in CoolProps (CP)
        allowed_gas_species = [s.upper() for s in CP.FluidsList()]
        if "&" in value:
            comp_frac_pair = [str.replace("["," ").replace("]","").split(" ") 
                              for str in  value.split("&")] 
            comp = [pair[0] for pair in comp_frac_pair]
            molefracs = np.asarray([float(pair[1]) for pair in comp_frac_pair])
            if sum(molefracs) != 1:
                warn("The mole fractions in the fluid '{value}' did not add"+
                     " to 1.0. They have been normalized to add to 1.0")
        else:
            comp = [value]
            
        for co in comp:
            
            if not co in allowed_gas_species:
                aliases = [s.upper() for s in CP.get_aliases(co)]
                
                found = False
                for alias in aliases:
                    if alias in allowed_gas_species:
                        found = True
                        
                if not found:
                
                    
                    error(field,"The fluid string has an illegal entry={co}."+
                          "Permissible entries for CoolProps are:\n\n"+
                          str(allowed_gas_species))
                    return
        


        
    # you must make a new entry if you put a new name for check_with in 
    # the YAML schemas
    dispatcher = {"h_inner_check":h_inner_check,
                  "is_isocalendardate":is_isocalendardate,
                  "is_increasinglist":is_increasinglist,
                  "check_pipe_lengths":check_pipe_lengths,
                  "check_pipe_conductivities":check_pipe_conductivities,
                  "check_pipe_roughness":check_pipe_roughness,
                  "check_pipe_minor_loss_coef":check_pipe_minor_loss_coef,
                  "fluid_string_check":fluid_string_check}
    
    def _add_check_with_functions(schema,dispatcher):
        # recursive function
        if isinstance(schema,dict):
            for key,val in schema.items():
                if key == "check_with":
                    schema[key] = dispatcher[val]
                elif isinstance(val,dict):
                    _add_check_with_functions(val,dispatcher)

    #LOAD YAML FILE INPUT SCHEMAS
    schema_path = os.path.join(os.path.dirname(__file__),"..","input_schemas")
    schemas = {}
    for filename in os.listdir(schema_path):
        if filename[-4:] == ".yml":
            with open(os.path.join(schema_path,filename),'r') as infile:
                 schema = yaml.load(infile,Loader=yaml.FullLoader) 
                 _add_check_with_functions(schema,dispatcher)
                 schemas[filename[:-4]] = schema

    validate_dict = {}
    retval = {}
    
    validate_dict['main input'] = Validator(schemas["schema_general"])
    retval['main input'] = validate_dict['main input'].validate(inp)
    
    # subset in wells
    for name,well in inp["wells"].items():
        # validate well
        validate_dict[name] = Validator(schemas["well_schema"])
        retval[name] = validate_dict[name].validate(well)
        # validate valve entry in general
        
        # a check not using cerebus
        # Verify the arrays provided for pipes have the right number of elements
        local_check_pipe_arrays(well,name,validate_dict,retval)
        
        # 2. Verify that there are as many valves as there are pipes. Every pipe must have a valve.
        if len(well["valves"]) != len(well["pipe_lengths"]):
            nvnpv = name+"_pipe_mismatch_to_valves"
            validate_dict[nvnpv] = _LocalErrorObj()
            retval[nvnpv] = validate_dict[nvnpv].validate(
                    nvnpv,
                    "Every pipe must have one valve. The first valve goes to"+
                    " the first pipe...etc..",False)
        

        
        # now loop over the valves (1 valve for every pipe)
        for vname,valve in well["valves"].items():
            
            nvn = name+"_"+vname
            
            # 3. verify that any mdot valves have time arrays that are valid input
            if valve["type"] == "mdot":
                if "time" in valve:
                   covers_all_times = valve["time"][-1] >= inp["calculation"]["end_time"]
                   if not covers_all_times:
                       nvnmd = nvn+"_covers_all_times"
                       validate_dict[nvnmd] = _LocalErrorObj()
                       retval[nvnmd] = validate_dict[nvnmd].validate(
                               nvnmd,
                               "The time array in all mdot valves must exceed"+
                               " the end time of the simulation! (i.e. "+
                               "calculation:end_time > wells:"+name+":"+vname+
                               ")",False)

            validate_dict[nvn] = Validator(schemas["valve_schema"])
            if isinstance(valve,dict):
                # validate valve in general 
                retval[nvn] = validate_dict[nvn].validate(valve)
                # validate specific valve type
                vtype = valve["type"]
                type_id = nvn+"_"+vtype
                vtype_schema = schemas["valve_type_schema"]
                validate_dict[type_id] = Validator(vtype_schema[vtype]) 
                retval[type_id] = validate_dict[type_id].validate(valve)
            else:
                nvnd = nvn+"_dictionary_error"
                validate_dict[nvnd] = _LocalErrorObj()
                retval[nvnd] = validate_dict[nvnd].validate(
                                nvnd, "A valve must be"+
                                " of type dictionary with the schema defined"+
                                " in valve_schema.yml", False)
        

    
    return retval, validate_dict
    
    


