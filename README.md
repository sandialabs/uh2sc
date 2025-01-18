# uh2sc
Underground H_2 Salt Cavern (UH_2SC) Simulation Tool

This is a place holder for a tool that is in development.

## CoolProp installation issue:
If you get the following error because CoolProp 6.6.0 install with no error but won't import:

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

Then you can work around by changing the "/Users/dlvilla/venv/vuh2sc/lib/python3.13/site-packages/CoolProp/__init__.py" file.

All the strings after fluid (and before the first 'def') must be turned into byte strings 

```python
__fluids__ = CoolProp.get_global_param_string('fluids_list').split(',')

# becomes

__fluids__ = CoolProp.get_global_param_string(b'fluids_list').split(b',') 

```

This was supposed to be fixed a long time ago: https://github.com/CoolProp/CoolProp/issues/1876
