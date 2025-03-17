


from warnings import warn

from uh2sc import validator
from uh2sc.errors import InputFileError
from uh2sc.solvers import NewtonSolver



class Model(object):

    def __init__(self,inp,single_component_test=False,**kwargs):

        """
        Construct a combined model of 1) a salt cavern, 2) an arbitrary number
        of wells and 3) Radial heat transfer away from the salt cavern and wells.



        Inputs
        ======

        inp : str or dict : str = filepath to a yaml file that conforms to the
                                  uh2sc schema or:
                            dict = dict that conforms to the uh2sc schema

        """

        self.input = self._read_input(inp)

        if not single_component_test:
            if len(kwargs) != 0:
                raise ValueError("kwargs are only allowed in single component"
                                +" testing! Do not call Model with kwargs!")
            self._validate()
            num_caverns = 1
            num_ghes = len(self.input["ghes"])
            num_wells = len(self.input["wells"])
        else:
            num_caverns = 0
            num_ghes = 0
            num_wells = 0
            # each component type has a single component test mode to assure
            # it can solve a simple case that does not include the other parts
            if kwargs["type"] == "ghe":
                num_ghes = 1
            elif kwargs["type"] == "cavern":
                num_caverns = 1
            elif kwargs["type"] == "well":
                num_wells = 1
            else:
                raise ValueError("Only valid kwargs['type']: ghe, cavern, well")

        self._build(num_caverns,num_wells,num_ghes)


    def run(self):
        pass

    def _build(self,inp,num_caverns,num_wells,num_ghes):
        """
        Assemble the order of the global variable vector the ordering is as follows:

        Though it appears that gas/liquid can be modeled interchangeably. The current
        version of uh2sc only allows a pipe to carry either a gas or a liquid permanently
        in a simulation. There is no ability to switch or mix gas and liquid in the pipes
        Such mixtures are much more difficult to model and are beyond the scope of the
        simple model uh2sc uses. The purpose of adding liquid can be to control gas pressure
        uh2sc does not model chemistry inside the liquid.

        Component    Num var        Description
        Cavern            1.             Cavern Gas Temperature
                          1.             Cavern Wall Temperature
                          1.             Cavern Liquid Temperature
                          1.             Cavern Liquid Wall Temperature
                          1.             Cavern Pressure at Liquid Interface
                          1.             Liquid interface distance from cavern top
                          num_gas        Gas fluxes at the liquid interface
                          num_gas        Gas/Liquid fluxes from well 1   pipe 1
           .              num_gas        Gas/Liquid fluxes from well 1   pipe 2
           .              .
           .              .
           .              num_gas        Gas/Liquid fluxes from well 1   pipe np_w1
           .              num_gas        Gas/Liquid fluxes from well 2   pipe 1
           .              .              .
           .              .              .
                          num_gas        Gas/Liquid fluxes from well nw  pipe np_wnw
        Wells
                          1.             Gas/Liquid temperature in well 1 pipe 1 element 1
                          1 or 2         In/Out pipe wall temps in well 1 pipe 1 element 1
                          NOTE: is 1 for a simple pipe, is 2 for a concentric pipe
                          1.             Gas/Liquid pressure    in well 1 pipe 1 element 1
                          num_gas        Gas/Liquid flow rates  in well 1 pipe 1 element 1
                          NOTE: These could be exiting or entering depending on the flow conditions!
                          np_w1          Gas/Liquid temperature of fluid flows element 1
                          np_w1          Gas/Liquid pressure of gas/liquid element 1
        GHEs              1.             Temperature element 1
                          NOTE: The surface temperature of the outermost pipe or salt cavern
                                wall is the same as the first temperature for the GHE.
                          1.             Temperature element 2
                          .
                          .              Temperature element n





        """
        xg_descriptions = []
        num_var = {}
        if num_caverns != 0:
            self._build_cavern(xg_descriptions)
            num_var["caverns"] = len(xg_descriptions)
        else:
            num_var["caverns"] = 0
        if num_wells != 0:
            self._build_wells(xg_descriptions)
            num_var["wells"] = (len(xg_descriptions)
                                             - num_var["caverns"])
        else:
            num_var["wells"] = 0

        if num_ghes != 0:
            self._build_ghes(xg_descriptions)
            num_var["ghes"] = (len(xg_descriptions)
                                             - num_var["caverns"]
                                             - num_var["wells"])
        else:
            num_var["ghes"] = 0






    def _build_ghes(self):

        ghes = self.inputs["ghes"]


    def _build_cavern(self):
        pass

    def _build_wells(self):
        pass

    def _validate(self):
        """
        Validating the provided problem definition dict

        (originally from HydDown class)

        Raises
        ------
        ValueError
            If missing input is detected.
        """
        valid_tup = validator.validation(self.input)
        vobjs = valid_tup[1]
        valid_test = valid_tup[0]
        invalid = False
        error_string = ""
        for key,val in valid_test.items():
            valid = val
            vobj = vobjs[key]
            if valid is False:
                #TODO - process this better so that the user knows the exact
                #       field (or first error) that needs correcting.
                error_string = error_string + "\n\n" + key + "\n\n" + str(vobj.errors)
                invalid = True

        if invalid:
            raise InputFileError("Error in input file:\n\n" + error_string)

    def _read_input(inp):
            # enable reading a file or accepting a valid dictionary
        if isinstance(inp,str):
            with open(inp,'r',encoding='utf-8') as infile:
                input_dict = yaml.load(infile, Loader=yaml.FullLoader)
        elif isinstance(inp,dict):
            input_dict = inp
        else:
            raise TypeError("The input object 'inp' must be a string that "+
                            "gives a path to a uh2sc yaml input file or must"+
                            " be a dictionary with the proper format for uh2sc.")
        return input_dict
