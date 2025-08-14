.. role:: python(code)
    :language: python

.. role:: bash(code)
    :language: bash

.. highlight:: bash

Quick start
-----------

To get started, you must have python 3.12 and pip (other versions may work but are not tested and python 3.13 has previously had a bug in CoolProp) 
to install version 1.0.0 of UH2SC enter the following in a bash command shell (commands will be somewhat similar in Windows). 

.. code-block:: bash
    :name: install

    python -m venv vuh2sc
    source vuh2sc/bin/activate
    pip install uh2sc

A virtual environment (vuh2sc) helps you from overwritting configurations elsewhere with all the libraries uh2sc needs.
If you want to install from the github repository so you can do some further development or changes from a specific branch you can:

.. code-block:: bash
    :name: install_git

    python -m venv vuh2sc
    source vuh2sc/bin/activate
    git clone -b <branch_name> https://github.com/sandialabs/uh2sc.git
    cd uh2sc
    pip install -e .

Once you have used pip to install, there is an available command line interface (CLI) tool that you can use to run UH2SC.

.. code-block:: bash
    :name: cli

    uh2sc --help
    Usage: uh2sc [OPTIONS]                                                                                                                                       
                                                                                                                                                              
    ╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --input_file  -i  TEXT  A file path or comma separated list of paths (no spaces allowed) to one or more valid UH2SC input files (in the repo see:          │
    │                         ./uh2sc/src/input_schemas/schema_general.yml for the schema and ./uh2sc/tests/test_data/... for examples that work (sometimes the  │
    │                         unit testing varies values so they are not guaranteed to work!))                                                                   │
    │                         IMPORTANT:                                                                                                                         │
    │                         You cannot run uh2sc in parallel using joblib. It already does this and you will get                                               │
    │                         AttributeError: 'NoneType' object has no attribute                                                                                 │
    │                         errors if you try to parallelize runs of uh2sc. It is better to use slurm (or similar) to partition uh2sc to several nodes. Uh2sc  │
    │                         will then use all processors on each node to calculate its jacobian in parallel using joblib                                       │
    │ --help                  Show this message and exit.                                                                                                        │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Now all you need to do is provide a valid input YAML file. We go over every entry in the input file in the Details section. The best way to get started is to use one of the working input files like this one:

.. code-block:: yaml
    :name: input_file_example

    cavern:
      depth: 914.4 # meters
      overburden_pressure: 19829198.656747766 # Pascal:  Approximately 1000 m deep overburden pressure
      height: 304.8 # meters
      diameter: 25.69 # meters
      ghe_name: nieland_ghe
      emissivity: 0.99
      max_operational_pressure_ratio: 1.0
      min_operational_pressure_ratio: 0.1
      max_volume_change_per_day: 0.2
      max_operational_temperature: 360
      min_operational_temperature: 290
    wells:  # an arbitrary number of wells can be defined that can pull and push different gasses into the well simultaneously
            # each well can have multiple pipes such that a single well can inject and withdraw at the same time.
      cavern_well:
        ideal_pipes: true  # override everything else such that there are no energy losses along the pipes.
        control_volume_length: 1066.8 # MUST BE THE SAME AS PIPE LENGTHS FOR THE CURRENT VERSION!
        pipe_lengths: [1066.8] # meters must be
        pipe_diameters: [0.0,0.0,0.1,0.12] # meters
        pipe_thermal_conductivities: [45] # W/m/K
        pipe_roughness_ratios: [0.000]  # ratio of pipe diameter to roughness height
        pipe_total_minor_loss_coefficients: [0.0]
        valves: # must have same number of valves as pipe_lengths (every pipe ends in a valve)
          inflow_mdot:
            type: "mdot"
            time: [0, 10368000, 10368000.01, 31104000]
            mdot: [-0.1189370698725589, -0.1189370698725589, 0.05946853493627945, 0.05946853493627945]
            reservoir:
              pressure: 20000000.0
              temperature: 310.93
              fluid: 'H2' # Though you can make any mixture of gasses here, read CoolProp's documentation. You can get completely unrealistic results with untested arbitrary mixtures of fluid!!!
                          # http://www.coolprop.org/fluid_properties/Mixtures.html#binary-pairs WARNING IN CoolProp:
    ghes:
      nieland_ghe:
        distance_to_farfield_temp: 600.0 # meters - calibration term
        density: 2162.4925554846 # kg/m3 salt density
        farfield_temperature: 317.5944444444444 # Kelvin
        heat_capacity: 837.2 #J/kg/K salt heat capacity
        thermal_conductivity: 5.190311418685122 # W/m/K see https://doi.org/10.1016/j.renene.2021.11.080
                                                #for examples this is for 100 C
        modeled_radial_thickness: 100.0 # hollow cylinder thickness of salt that is modeled
        initial_conditions:
          Q0: 0.0
          Qend: 0.0
        number_elements: 50
    initial:  #
      temperature: 326.3
      pressure: 17023743.5813687 #Pascal
      fluid: "H2"
      start_date: "2023-01-01" # Must be in iso format "YYYY-MM-DD"
      liquid_height: 1.0 # in meters
      liquid_temperature: 322.5 # ussually 2-4 C cooler than the gas
      time_step: 86400
    calculation:
      max_time_step: 172800 # seconds - 2 days time steps
      min_time_step: 1200
      end_time: 31104000 #3456000 #
      run_parallel: true
    heat_transfer:
      h_inner: "calc" # can enter a constant value if needed.
      h_cavern_brine: 100 # eventually a model is needed for this
                          # as a function of Rayleigh number!

You can also command uh2sc directly from Python if you want to manipulate the input dictionary (YAML comes in as a Python dictionary). Only parallelize lots of UH2SC runs in python if you set input["calculation"]["run_parallel"]=False in the input though!

.. code-block:: python
    :name: run_uh2sc_from_python

    import yaml
    from uh2sc.model import Model
     
    inp_path = "/your/path/to/an/input/file/input.yml"

    with open(inp_path, 'r', encoding='utf-8') as infile:
        inp = yaml.load(infile, Loader=yaml.FullLoader)

    # do stuff with the input dictionary (or you can create one from scratch!)
    inp["calculation"]["run_parallel"] = False

    model = Model(inp)  # or you can Model(inp_path) directly!

    model.run()

    #post process 
    results = model.dataframe()
    print(model.independent_vars_descriptions)
    model.plot_solution(model.independent_vars_descriptions[1:5])


    #write 
    output_path = /path/to/an/output.csv
    model.write_results(output_path)

    #understand the equations
    print(model.equations_list)

    
