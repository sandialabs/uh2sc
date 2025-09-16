import os
import shutil
import importlib
import yaml
import pathlib
import rich_click as click
from matplotlib import pyplot as plt
from rich import print
from rich.syntax import Syntax

from uh2sc.model import Model

def main(input_file, output_file, pickle_result, graph_results, log_file):
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            input_data = yaml.load(infile, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"[red]Error:[/red] Input file '{input_file}' not found.")
        return
    except yaml.YAMLError as err:
        print(f"[red]Error parsing input file[/red] {input_file}: {err}")
        return

    print("[cyan]Initializing model...[/cyan]")
    try:
        model = Model(input_data)
    except Exception as err:  # pylint: disable=broad-except
        print(f"[red]Error initializing model:[/red] {err}")
        raise err
        return

    print("[cyan]Running the model (this may take a while)...[/cyan]")
    try:
        elapsed = model.run(log_file=log_file)
    except Exception as err:  # pylint: disable=broad-except
        print(f"[red]Error running the model:[/red] {err}")
        return

    print(f"[green]Finished run after {elapsed:.2f} seconds.[/green]")

    print("[cyan]Writing results...[/cyan]")
    try:
        model.write_results(output_file)
    except Exception as err:  # pylint: disable=broad-except
        print(f"[red]Error writing results:[/red] {err}")
        return

    if pickle_result:
        print("[cyan]Creating pickle file...[/cyan]")
        try:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            pickle_path = f"{base_name}.pickle"
            model.pickle(pickle_path)
        except Exception as err:  # pylint: disable=broad-except
            print(f"[red]Error creating pickle file:[/red] {err}")

    if graph_results:
        print("[cyan]Generating graphs...[/cyan]")
        graph_dir = os.path.join(os.getcwd(), "graphs")
        try:
            if os.path.exists(graph_dir):
                print("[yellow]Deleting the existing 'graphs' folder...[/yellow]")
                shutil.rmtree(graph_dir)
            os.mkdir(graph_dir)

            for desc in model.xg_descriptions:
                fig_dict, _ = model.plot_solution([desc])
                fig_dict[desc].savefig(os.path.join(graph_dir, f"{desc}.png"), dpi=300)
                plt.close(fig_dict[desc])

        except Exception as err:  # pylint: disable=broad-except
            print(f"[red]Error generating graphs:[/red] {err}")

    print("[bold green]All done![/bold green]")

def main_list(schema, available,inputfile,inputnames):
    # handle both pytest install -e . and pytest install .
    schema_dir = importlib.resources.files("uh2sc") / 'input_schemas'
    if not os.path.exists(schema_dir):
        schema_path = importlib.resources.files("uh2sc") / 'src' / 'uh2sc' /'input_schemas'

    if available:
        try:
            files = [
                f for f in os.listdir(schema_dir)
                if f.endswith(".yml") or f.endswith(".yaml")
            ]
            if not files:
                print("[yellow]No schema files found.[/yellow]")
            else:
                print("[green]Available schema files:[/green]")
                for fname in files:
                    print(f"  - {fname}")
        except FileNotFoundError:
            print(f"[red]Error:[/red] Schema directory '{schema_dir}' not found.")
        return

    if schema is None and inputfile is None:
        print("[yellow]No schema or input file specified. Use --schema FILE, --inputfile FILEPATH or --available.[/yellow]")
        return
    elif schema is not None and inputfile is not None:
        print("[yellow]You cannot include both a schema and an input file. Please choose one or the other!")

    # set paths.
    if schema is not None:
        schema_path = os.path.join(schema_dir, schema)
    else:
        schema_path = None
    if inputfile is not None:
        inputfile_path = inputfile
    else:
        inputfile_path = None

    # read and display one or more files
    for sfpath, ftype in zip([schema_path,inputfile_path],
                             ["schema","input file"]):
        try:
            if sfpath is not None:
                with open(sfpath, "r", encoding="utf-8") as sffile:
                    content = sffile.read()
                    if inputfile is None:
                        header_print_str = f"UH2SC schema {schema}:"
                    else:
                        header_print_str = f"UH2SC input {inputfile}:"

                    if inputnames is not None:
                        header_print_str = f"{header_print_str} {inputnames}"
                        uh2sc_inp = yaml.safe_load(content)

                        keys = inputnames.split(",")
                        num_keys = len(keys)
                        if keys[0] not in uh2sc_inp:
                            print("")
                            print(f"The inputnames input {inputnames} is not valid!"
                                +"You must enter values that belong to the UH2SC schema!"
                                +" Start with entering uh2sc list --schema schema_general.yml"
                                +" or use uh2sc list --available to see the other schemas!")
                            print("")
                            return
                        else:
                            if num_keys > 1:
                                if inputfile is None:
                                    if "schema" in uh2sc_inp[keys[0]]:
                                        rec = uh2sc_inp[keys[0]]["schema"]
                                    else:
                                        rec = uh2sc_inp[keys[0]]
                                else:
                                    rec = uh2sc_inp[keys[0]]
                            else:
                                rec = uh2sc_inp[keys[0]]
                        num_deep = 1
                        for key in keys[1:]:
                            num_deep += 1
                            if key not in rec:
                                print("")
                                print(f"The -n|--inputnames input {inputnames} is not valid! "
                                +"You must enter values that belong to the UH2SC schema!"
                                +" Start with entering uh2sc list --schema schema_general.yml"
                                +" or use uh2sc list --available to see the other schemas!")
                                print("")
                                return
                            else:
                                if num_keys > num_deep:
                                    if inputfile is None:
                                        if "schema" in rec[key]:
                                            rec = rec[key]["schema"]
                                        else:
                                            rec = rec[key]
                                    else:
                                        rec = rec[key]
                                else:
                                    rec = rec[key]
                        content = yaml.safe_dump(rec)

                    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=False)
                    print("")
                    print("[cyan]" + header_print_str)
                    print("-"*len(header_print_str))
                    print(syntax)
        except FileNotFoundError:
            print(f"[red]Error:[/red] {ftype} file '{sfpath}' not found.")
            return
        except yaml.YAMLError as err:
            print(f"[red]Error parsing {ftype} file[/red] {sfpath}: {err}")
            return
        except KeyError:
            print(f"[red]Error the requested key structure is incorrect")
            return



@click.group()
def cli():
    """Underground Hydrogen Salt Cavern (UH2SC): This program executes a control volume
with gas mixtures (not limited to hydrogen!) and a brine control volume and radial symetric heat away
from a salt cavern. It models boundary conditions as flow at constant pressure and temperature through
a vertical adiabatic pipe to the cavern. Navigate how to create a valid input file using the 'list' command.
Example input files are in tests/test_data/*.yml. Author=dlvilla@sandia.gov, License=Open Source Revised BSD

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering
Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's
National Nuclear Security Administration under contract DE-NA-0003525.

Creation of UH2SC was funded by an Inter-Agency Agreement between the United States Department of Energy Office of Fossil
Energy and Carbon Management and the Pipeline and Hazardous Materials Safety Administration."""
    # Nothing needed here; click uses this as a group entrypoint.


@click.command(help="Run a UH2SC simulation with an input file specified.")
@click.option(
    "-i", "--input-file",
    default="input.yml",
    show_default=True,
    help="Path to a valid UH2SC input YAML file."
)
@click.option(
    "-o", "--output-file",
    default="uh2sc_results.csv",
    show_default=True,
    help="Path to the CSV output file."
)
@click.option(
    "-p", "--pickle-result",
    is_flag=True,
    help="If set, create a pickle file for later processing."
)
@click.option(
    "-g", "--graph-results",
    is_flag=True,
    help="If set, generate PNG graphs in a 'graphs' folder."
)
@click.option(
    "-l", "--log-file",
    default="uh2sc.log",
    show_default=True,
    help="Path to the logging output file."
)
def run(input_file, output_file, pickle_result, graph_results, log_file):
    """Run UH2SC hydrogen/other gas program."""
    main(input_file, output_file, pickle_result, graph_results, log_file)



@click.command(help="List schema files or contents of schema or input files to understand how to make an input file.")
@click.option(
    "-s", "--schema",
    default=None,
    help="The schema file to display (no path needed). Use --a|--available to see what names are valid."
)
@click.option(
    "-n", "--inputnames",
    default=None,
    help=("Input a comma seprated list of names that are keys for"
    +" the UH2SC input schema only a sub-section of input docs will be output")
)
@click.option(
    "-a", "--available",
    is_flag=True,
    help="If set, list all available schema files in the input_schemas directory."
)
@click.option(
    "-i", "--inputfile",
    default=None,
    help="A UH2SC input file to display with path to the file included."

)
def list(schema, available,inputfile,inputnames):
    """List the schema file or show available schema files."""
    main_list(schema, available,inputfile,inputnames)


cli.add_command(run)
cli.add_command(list)


if __name__ == "__main__":
    cli()
