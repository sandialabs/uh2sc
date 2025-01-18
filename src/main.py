# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license

import yaml
import sys
from uh2sc import SaltCavern


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        input_filename = "input.yml"

    with open(input_filename) as infile:
        inp = yaml.load(infile, Loader=yaml.FullLoader)


    hdown=SaltCavern(inp)
    hdown.run(disable_pbar=False)
        
    hdown.plot()