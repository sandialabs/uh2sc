# Underground Hydrogen Salt Cavern (UH2SC) Simulation Tool


## Installation
First install Python 3.12 ([miniconda3](https://repo.anaconda.com/miniconda/) works great!).
For just using the software without further development issue:

```bash
pip install uh2sc
```

For developing the code on a branch:

```bash
git clone https://github.com/sandialabs/uh2sc
cd uh2sc
# If you want to create a virtual environment, ucomment below (you have to change the paths, don't take the literally!)
# in windows this would be:
# C:\path\to\python312\python -m venv C:\path\to\where\you\want\your\virtual\environment\vuh2sc
# in linux
# /path/to/python312/python -m venv /path/to/where/you/want/your/virtual/enironment/vuh2sc
# in windows this would be:
# C:\path\to\where\you\want\your\virtual\environment\Scripts\activate
# in linux
# source /path/to/where/you/want/your/virtual/enironment/vuh2sc/bin/activate
pip install -e .[test,docs]
```

You can then test to see if the current unit tests run with:

```bash
pytest
```

## Command Line Interface

Once uh2sc is installed, it functions as a rich_click command line interface.

```bash
uh2sc --help
                                                          
 Usage: uh2sc [OPTIONS] COMMAND [ARGS]...                 
                                                          
 Underground Hydrogen Salt Cavern (UH2SC): This program   
 executes a control volume with gas mixtures (not limited 
 to hydrogen!) and a brine control volume and radial      
 symetric heat away from a salt cavern. It models         
 boundary conditions as flow at constant pressure and     
 temperature through a vertical adiabatic pipe to the     
 cavern. Navigate how to create a valid input file using  
 the 'list' command. Example input files are in           
 tests/test_data/*.yml. Author=dlvilla@sandia.gov,        
 License=Open Source Revised BSD                          
 Sandia National Laboratories is a multimission           
 laboratory managed and operated by National Technology   
 and Engineering Solutions of Sandia, LLC., a wholly      
 owned subsidiary of Honeywell International, Inc., for   
 the U.S. Department of Energy's National Nuclear         
 Security Administration under contract DE-NA-0003525.    
 Creation of UH2SC was funded by an Inter-Agency          
 Agreement between the United States Department of Energy 
 Office of Fossil Energy and Carbon Management and the    
 Pipeline and Hazardous Materials Safety Administration.  
                                                          
╭─ Options ──────────────────────────────────────────────╮
│ --help      Show this message and exit.                │
╰────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────╮
│ list  List schema files or contents of schema or input │
│       files to understand how to make an input file.   │
│ run   Run a UH2SC simulation with an input file        │
│       specified.                                       │
╰────────────────────────────────────────────────────────╯
```

This is explorable using "--help" 

```bash
bash-5.2$ uh2sc run --help
                                                          
 Usage: uh2sc run [OPTIONS]                               
                                                          
 Run a UH2SC simulation with an input file specified.     
                                                          
╭─ Options ──────────────────────────────────────────────╮
│ --input-file     -i  TEXT  Path to a valid UH2SC input │
│                            YAML file.                  │
│                            [default: input.yml]        │
│ --output-file    -o  TEXT  Path to the CSV output      │
│                            file.                       │
│                            [default:                   │
│                            uh2sc_results.csv]          │
│ --pickle-result  -p        If set, create a pickle     │
│                            file for later processing.  │
│ --graph-results  -g        If set, generate PNG graphs │
│                            in a 'graphs' folder.       │
│ --log-file       -l  TEXT  Path to the logging output  │
│                            file.                       │
│                            [default: uh2sc.log]        │
│ --help                     Show this message and exit. │
╰────────────────────────────────────────────────────────╯

```

The list command enables gaining understanding of what inputs are allowed and what their meaning is:

```bash
bash-5.2$ uh2sc list --available
Available schema files:
  - valve_type_schema.yml
  - schema_general.yml
  - ghe_schema.yml
  - well_schema.yml
```

```bash
bash-5.2$ uh2sc list --schema valve_type_schema.yml --inputnames mdot,reservoir,fluid

UH2SC schema valve_type_schema.yml: mdot,reservoir,fluid
--------------------------------------------------------
check_with: fluid_string_check                            
meta:                                                     
  help: "string: Reservoir fluid. Must be a set of valid C
    \ mass fractions delimited by\n  \"&\". Mass fractions
    \ fraction is needed if it is only a\n  single fluid. 
    \ can exist (e.g., \"H2\" or Hydrogen).\n  Refer to Co
    \ gas names. Examples \"H2\", \"H2[0.1]&Methane[0.9]\"
    , \"CO2[0.5]&H2[0.5]\". Even though tertiary systems c
    \ to be sure to understand if CoolProp has accurate ca
    \  gas mixture! http://www.coolprop.org/fluid_properti
required: true                                            
type: string      
```

It can also list and display values from an input file:

```bash
bash-5.2$ uh2sc list --inputfile ./test_data/nieland_verification_methane_12_cycles.yml --inputnames cavern
 
UH2SC input ./test_data/nieland_verification_methane_12_cycles.yml: cavern
--------------------------------------------------------------------------
depth: 914.4                                              
diameter: 25.69                                           
emissivity: 0.99                                          
ghe_name: nieland_ghe                                     
height: 304.8                                             
max_operational_pressure_ratio: 1.0                       
max_operational_temperature: 360                          
max_volume_change_per_day: 0.2                            
min_operational_pressure_ratio: 0.05                      
min_operational_temperature: 290                          
overburden_pressure: 19829198.656747766
```

## Using Python

The following code shows how to run from Python instead. More options to configure the solver are available running from python:

```python
import yaml
from uh2sc.model import Model
# Input a filepath string OR read yaml, change it and run the model
inp_path = "/a/path/to/input/yaml/file.yml"
with open(inp_path, 'r', encoding='utf-8') as infile
   inp = yaml.safe_load(infile)

# alter the input programatically
inp['cavern']['height'] = 1000

model = Model(inp)

# running the model can take awhile depending 
# on the length and whether gas mixtures are needed
model.run()

# dictionary of results by time step
solutions = model.solutions

# write results to a file
model.write_results()

# pickle the model for later use (simple pickle will not work)
# because of C type object stored with CoolProp. These get cleared
# by this routine and saves them in a class that can repopulate
# the fluid models when you want to use the model again.
model.pickle("/path/to/a/pickle/file.pkl")

# dataframe with dates (relative_time=False) or seconds as an index
df = model.dataframe(relative_time=True)

# list the meaning of all global equations in the model
eqn_list = model.equations_list()

# list of global variables solved for
x_desc = model.xg_descriptions

# list of independent variables (like pressure) that
# can also be output
y_desc = model.independent_vars_descriptions()

# plot solutions
model.plot_solution(x_desc[10:20],show=True)

```



## Model description

A single control volume salt cavern of any combination of gases that is supported by [CoolProp](http://www.coolprop.org/) is modeled. Around this control volume, a cylindrical axisymmetric heat transfer model to ground is included This control volume is assumed to be well mixed and of static volume. Interacting with this control volume is an arbitrary number of wells which insert arbitrary mixtures of gas. Each well is composed of one or more concentric pipes. Each well interacts with ground via a cylindrical axisymmetric heat transfer model with an adjustment factor for assymetry in cases where wells are close to eachother.

In its current state, only a single well, single pipe cyclic loading has been evaluated against the work of Nieland (2008) with matches being as close as can be expected for two models with different gas mixture and heat transfer approximations. A more thorough verification of the axisymmetric heat transfer is also in the testing.

Nieland, J. D., 2008. Salt cavern thermodynamics–comparison between hydrogen, natural gas, and air 
storage. In SMRI Fall 2008 Technical Conference Paper Galveston, Texas, USA October 13-14th.


## Input files

UH2SC uses a [YAML](https://yaml.org/) input file to create new scenarios. UH2SC uses [cerberus](https://docs.python-cerberus.org/) to perform input validation. The schema that shows you all of the valid inputs is in `/src/uh2sc/input_schemas`. The `schema_general.yml` is the highest level schema. The schemas tell you what ranges of values and types are allowed and what the names of each input entry are. More complex validation functions are also included in `/src/uh2sc/validator.py`. 


## Python3.13 may not work

uh2sc may not work with Python 3.13 because of an open issue with [CoolProp](http://www.coolprop.org/)

```python
>>> import CoolProp
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    import CoolProp
  File "/Users/dlvilla/venv/vuh2sc/lib/python3.13/site-packages/CoolProp/__init__.py", line 22, in <module>
    __fluids__ = CoolProp.get_global_param_string('fluids_list').split(',')
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "CoolProp.pyx", line 309, in CoolProp.CoolProp.get_global_param_string
  File "<stringsource>", line 15, in string.from_py.__pyx_convert_string_from_py_std__in_string
TypeError: expected bytes, str found
```

## HydDown

UH2SC originates from the open source [HydDown](https://github.com/andr1976/HydDown) code. As allowed by HydDown's MIT license, the code has been updated extensively though and nothing has been left intact but the basic structure for how the problem is solved is still in place. Thanks to the creators of HydDown for a good structure to start from. 

## LICENSE

![LICENSE](LICENSE)

## Sandia Funding Statement

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525

## Acknowledgements

Creation of UH2SC was funded by an Inter-Agency Agreement between the United States Department of Energy Office of Fossil Energy and Carbon Management and the Pipeline and Hazardous Materials Safety Administration.
