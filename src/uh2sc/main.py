# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license

import yaml
import sys

import rich_click as click

from uh2sc import SaltCavern

@click.command()
@click.option("-i",
              "--input_file",default="input.yml", help="A file path to a valid "
              +"UH2SC input file (in the repo see: ./uh2sc/input_schemas/ for the" 
              +" schema and ./uh2sc/tests/test_data/... for examples that work)")
def main(input_file):
    with open(input_file,'r',encoding='utf-8') as infile:
        inp = yaml.load(infile, Loader=yaml.FullLoader)

    sc = SaltCavern(inp)
    sc.run(disable_pbar=False) 


if __name__ == "__main__":
    main()