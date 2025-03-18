


from warnings import warn
import numpy as np

from uh2sc import validator
from uh2sc.errors import InputFileError
from uh2sc.solvers import NewtonSolver
from uh2sc.abstract import AbstractComponent
from uh2sc.hdclass import ImplicitEulerAxisymmetricRadialHeatTransfer
import jax.numpy as jnp


ADJ_COMP_TESTING_NAME = "testing"

class Model(AbstractComponent):
    """
    Formulate a model that consistst of components. Each component has an abstract
     class as its base in uh2sc.abstract.AbstractComponent. Interface and boundary
     conditions are handled as additional equations. Each component has a method
     called "evaluate_residuals." Current valid components are:

     1. Cavern
     2. Arbitrary number of wells
     3. Arbitrary number of 1-D axisymmetric heat transfer through the ground

     Future versions may include multiple caverns that are connected via
     surface pipes and may include pumps/compressors.
    """

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

        self.inputs = self._read_input(inp)
        self.test_inputs = kwargs

        if not single_component_test:
            if len(kwargs) != 0:
                raise ValueError("kwargs are only allowed in single component"
                                +" testing! Do not call Model with kwargs!")
            self._validate()
            num_caverns = 1
            num_ghes = len(self.inputs["ghes"])
            num_wells = len(self.inputs["wells"])
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

        self.solver = NewtonSolver()

        time_step = self.inputs["calculation"]["time_step"]
        nstep = int(self.inputs["calculation"]["end_time"]
                    / time_step
                    + 1)
        # constant time steps
        self.times = [idx * time_step for idx in range(nstep)]


    def run(self):
        for _time in self.times:
            self.solver.solve(self)

    @property
    def next_adjacent_components(self):
        """_summary_
        One day we may make this code be able to
        connect two models but we have not yet
        Done this
        """
        return []

    @property
    def previous_adjacent_components(self):
        """_summary_
        One day we may make this code be able to
        connect two models but we have not yet
        Done this
        """
        return []

    @property
    def global_indices(self):
        """
        This property must give the indices that give the begin and end location
        in the global variable vector (xg). Since this is the entire model
        it will return the entire range.
        """
        return (0, len(self.xg))

    def get_x(self):
        xg = self.xg
        for cname, component in self.components.items():
            bind, eind = component.global_indices
            xg[bind:eind] = component.get_x()
        return xg

    def evaluate_residuals(self,x=None):
        residuals = []
        if x is None:
            for _cname, component in self.components.items():

                # this is the single x behavior used by
                # local evaluations of residuals
                residuals += list(component.evaluate_residuals())
                
            return np.array(residuals)
        else:
            # this is for scipy where a matrix of all the different
            # vectors needed to evaluate the jacobian is fed in
            if x.ndim == 2:
                for xv in x:
                    v_residuals = []
                    for _cname, component in self.components.items():
                        v_residuals += list(component.evaluate_residuals(xv))
                    residuals.append(v_residuals)
            elif x.ndim == 1:
                for _cname, component in self.components.items():

                    # this is the single x behavior used by
                    # local evaluations of residuals
                    residuals += list(component.evaluate_residuals(x))



            return jnp.array(residuals)


    def load_var_values_from_x(self, xg_new):
        for cname, component in self.components.items():
            component.load_var_values_from_x(xg_new)


    def _build(self,num_caverns,num_wells,num_ghes):
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
        xg = []
        num_var = {}
        components = {}
        if num_caverns != 0:
            self._build_cavern(xg_descriptions,xg,components)
            num_var["caverns"] = len(xg_descriptions)
        else:
            num_var["caverns"] = 0
        if num_wells != 0:
            self._build_wells(xg_descriptions,xg,components)
            num_var["wells"] = (len(xg_descriptions)
                                             - num_var["caverns"])
        else:
            num_var["wells"] = 0

        if num_ghes != 0:
            self._build_ghes(xg_descriptions,xg,components)
            num_var["ghes"] = (len(xg_descriptions)
                                             - num_var["caverns"]
                                             - num_var["wells"])
        else:
            num_var["ghes"] = 0

        self.xg = np.array(xg)
        self.components = components

    def _assign_max_from_adj_comp(self,adj_comps,name,multfact,arrind=None):
        val = 0.0

        for tup in adj_comps:
            adjacent_comp = tup[1]
            if "pipe_diameters" in adjacent_comp:
                if name == "height":
                    name = "pipe_lengths"
                if name == "diameter":
                    name = "pipe_diameters"
                if arrind is None:
                    val = np.max(val,adjacent_comp[name]*multfact)
                else:
                    val = np.max(val, adjacent_comp[name][arrind]*multfact)

            else:
                val = np.max(val,adjacent_comp[name]*multfact)

        return val


    def _build_ghes(self,x_desc,xg,components):
        """
        Build all GHE's and link them to the
        specified well or the cavern

        Args:
            x_desc list: _description_
            xg list: _description_
            components (_type_): _description_
        """

        ghes = self.inputs["ghes"]

        for name,ghe in ghes.items():
            if len(self.test_inputs) != 0:
                # This is testing mode and you have to add stuff that
                # would normally come from the main ("schema_general.yml") schema.
                t_step = self.test_inputs["dt"]
                height = self.test_inputs["height"]
                inner_radius = self.test_inputs["inner_radius"]
                adjacent_comps = {}
                adjcomp_name = ADJ_COMP_TESTING_NAME
                # backfit missing stuff to get the solution to work.
                self.inputs["calculation"] = {}
                self.inputs["calculation"]["end_time"] = self.test_inputs["end_time"]
                self.inputs["calculation"]["time_step"] = self.test_inputs["dt"]
            else:
                # find the adjacent components to this GHE!
                adjacent_comps = self._find_ghe_adj_comp(name)

                # set the inner_radius and height either based on user input
                # or based on the maximum value from adjacent copmonents for height
                # and diameter.
                if "inner_radius" in ghe:
                    inner_radius = ghe["inner_radius"]
                else:
                    if len(adjacent_comps) == 0:
                        raise ValueError("Axi-symmetric heat transfer must have "
                                         +"at least one adjacent component if the"
                                         +" inner_radius is not defined in the input!")

                    inner_radius = self._assign_max_from_adj_comp(adjacent_comps,
                                                                  "diameter",
                                                                  0.5,
                                                                  -1)
                if "height" in ghe:
                    height = ghe["height"]
                else:
                    if len(adjacent_comps) == 0:
                        raise ValueError("Axi-symmetric heat transfer must have "
                                         +"at least one adjacent component if the"
                                         +" height is not defined in the input!")
                    height = self._assign_max_from_adj_comp(adjacent_comps,
                                                            "height",
                                                            1.0)
                # set the constant time step

                t_step = self.inputs["calculations"]["time_step"]





            # begin global indice
            beg_idx = len(xg)

            xg += [ghe["initial_conditions"]["Q0"]]
            x_desc += [f"GHE name `{name}` heat flux at inner_radius (W)"]

            # add all new terms that belong to the descriptions and to the
            xg += [ghe["farfield_temperature"] for _idx in range(ghe["number_elements"]+1)]
            x_desc += [f"GHE name `{name}` element {idx} outer temperature "
                       +"(inner temperature element {idx-1})"
                       for idx in range(ghe["number_elements"]+1)]

            xg += [ghe["initial_conditions"]["Qend"]]
            x_desc += [f"GHE name `{name}` heat flux at outer_radius (W)"]

            #end of global indices for this component
            end_idx = len(xg)+1

            components[name] = ImplicitEulerAxisymmetricRadialHeatTransfer(inner_radius,
                  ghe["thermal_conductivity"],
                  ghe["density"],
                  ghe["heat_capacity"],
                  height,
                  ghe["number_elements"],
                  ghe["modeled_radial_thickness"],
                  ghe["farfield_temperature"],
                  ghe["distance_to_farfield_temp"],
                  bc=ghe["initial_conditions"],
                  dt0=t_step,
                  adj_comps=adjacent_comps,
                  global_indices=(beg_idx,end_idx))



    def _build_cavern(self,x_desc,xg,components):
        pass

    def _build_wells(self,x_desc,xg,components):
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

    def _read_input(self,inp):
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

    def _find_ghe_adj_comp(self,ghe_name):
        """
        Find out how many adjacent components the axisymmetric
        ground heat exchange (GHE) has. Every component contributes
        a heat flux to the GHE
        """
        adj_comp = []
        for wname, well in self.inputs["wells"].items():
            if well["ghe_name"] == ghe_name:
                adj_comp.append(wname,well)
        if ghe_name == self.inputs["cavern"]["ghe_name"]:
            adj_comp.append("cavern",cavern)
        return adj_comp
