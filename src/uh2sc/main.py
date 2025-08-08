# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license

import yaml
import os
import shutil
import rich_click as click
import pickle
from matplotlib import pyplot as plt

from uh2sc.model import Model 

def find_unpickleable_attrs(obj, path="obj", visited=None):
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return
    visited.add(id(obj))

    if isinstance(obj, (str, int, float, bool, type(None))):
        return  # primitive types are always pickleable

    try:
        pickle.dumps(obj)
    except Exception as e:
        print(f"Unpickleable at {path}: {type(obj)} — {e}")

    if isinstance(obj, dict):
        for k, v in obj.items():
            find_unpickleable_attrs(k, path=f"{path}[{repr(k)}]", visited=visited)
            find_unpickleable_attrs(v, path=f"{path}[{repr(k)}]", visited=visited)
    elif hasattr(obj, "__dict__"):
        for attr_name, attr_val in vars(obj).items():
            find_unpickleable_attrs(attr_val, path=f"{path}.{attr_name}", visited=visited)
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            find_unpickleable_attrs(item, path=f"{path}[{i}]", visited=visited)


@click.command()
@click.option("-i",
              "--input_file",default="input.yml", help="A file path to a valid "
              +"UH2SC input file (in the repo see: ./uh2sc/src/input_schemas/schema_general.yml for the" 
              +" schema and ./uh2sc/tests/test_data/... for examples that work (sometimes the unit "
              +"testing varies values so they are not guaranteed to work!))\n\n"
              +"IMPORTANT:\n\n"
              +"You cannot run uh2sc in parallel using joblib. It already does this and you will get\n\n "
              +"AttributeError: 'NoneType' object has no attribute\n\n"
              +"errors if you try to parallelize runs of uh2sc. It is better to use slurm (or similar) to partition "
              +"uh2sc to several nodes. Uh2sc will then use all processors on each node to calculate its "
              +"jacobian in parallel using joblib")
@click.option("-o",
              "--output_file",default="uh2sc_results.csv", help="A file path to a valid folder and file location")
@click.option("-p",
               "--pickle_result",default=False, help="Boolean, True means a pickle file "
               +"will be created so you can start up python and reload the model "
               +"object for further processing")
@click.option("-g",
               "--graph_results",default=False, help="Boolean, True means"
               +" graphs of all output files are created in a folder called"
               +" graphs in the current working directory." )
@click.option("-l",
               "--log_file", default="uh2sc.log", help="Filepath to the logging output of the run")
def cli(input_file,output_file,pickle_result,graph_results,log_file):
    main(input_file,output_file,pickle_result,graph_results,log_file)


def main(input_file,output_file,pickle_result,graph_results,log_file):
    with open(input_file,'r',encoding='utf-8') as infile:
        inp = yaml.load(infile, Loader=yaml.FullLoader)
    
    print("initializing model!")
    model = Model(inp)
    
    
    print("Running the model (takes the longest....)")
    
    elapsed = model.run(log_file=log_file) 
    print(f"Finished run after {elapsed} seconds.")
    
    print("Writing results.")
    model.write_results()
    
    
    if pickle_result:
        print("Creating pickle file.")
        model.pickle(input_file.split(".")[0] + ".pickle")
    
    if graph_results:
        print("Graphing results.")
        try:
            graph_dir = os.path.join(os.getcwd(),"graphs")
            if os.path.exists(graph_dir):
                print("Deleting the existing graphs folder!")
                shutil.rmtree(graph_dir)
            os.mkdir(graph_dir)
            os.chdir(graph_dir)
            for desc in model.xg_descriptions:
                figd,axd = model.plot_solution([desc])
                figd[desc].savefig(os.path.join(graph_dir,desc+".png"),dpi=300)
                plt.close(figd[desc])
                
                
            os.chdir("..")
        except Exception as e:
            print(f"MAIN: Failed to create graphs folder so no plots were created: {e}")

    print("All done!")

    
    
