# Underground Hydrogen Salt Cavern (UH2SC) Simulation Tool


## Installation
First install Python 3.12 ([miniconda3](https://repo.anaconda.com/miniconda/) works great!).

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

## Model description

A single control volume salt cavern of any combination of gases that is supported by [CoolProp](http://www.coolprop.org/) is modeled. Around this control volume, a cylindrical axisymmetric heat transfer model to ground is included This control volume is assumed to be well mixed and of static volume. Interacting with this control volume is an arbitrary number of wells which insert arbitrary mixtures of gas. Each well is composed of one or more concentric pipes. Each well interacts with ground via a cylindrical axisymmetric heat transfer model with an adjustment factor for assymetry in cases where wells are close to eachother.

In its current state, only a single well, single pipe cyclic loading has been evaluated against the work of Nieland (2008) with matches being as close as can be expected for two models with different gas mixture and heat transfer approximations. A more thorough verification of the axisymmetric heat transfer is also in the testing.

Nieland, J. D., 2008. Salt cavern thermodynamics–comparison between hydrogen, natural gas, and air 
storage. In SMRI Fall 2008 Technical Conference Paper Galveston, Texas, USA October 13-14th.


## How to use uh2sc

**_DISCLOSURE:_** New cases are likely to be buggy and the multiple well, mulitiple pipe code is not complete

```bash
uh2sc input.yml
```


UH2SC uses a [YAML](https://yaml.org/) input file to create new scenarios. UH2SC uses [cerberus](https://docs.python-cerberus.org/) to perform input validation. The schema that shows you all of the valid inputs is in `/src/input_schemas`. The `schema_general.yml` is the highest level schema. The schemas tell you what ranges of values and types are allowed and what the names of each input entry are. More complex validation functions are also included in `/src/uh2sc/validator.py`. 








## Python3.13 does not work

uh2sc will not work with Python 3.13 because of an open issue with [CoolProp](http://www.coolprop.org/)

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
