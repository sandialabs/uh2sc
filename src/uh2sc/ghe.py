# HydDown hydrogen/other gas depressurisation
# Copyright (c) 2021 Anders Andreasen
# Published under an MIT license



import numpy as np

from CoolProp import CoolProp as CP
from uh2sc.abstract import AbstractComponent, ComponentTypes
from uh2sc.utilities import calculate_cavern_pressure, conservation_of_volume


class ImplicitEulerAxisymmetricRadialHeatTransfer(AbstractComponent):

    def __init__(self,r_inner,kg,rhog,cpg,length_component,number_elements,dist_next_cavern_wall,Tg,
                 dist_to_Tg_reservoir,dt0,bc,adj_comps,global_indices,Tgvec0=None,model=None):
        """
        Inputs: (all are float or int)
        -------
            r_inner              = inner radius where heat transfer begins
            kg                   = ground thermal conductivity
            rhog                 = ground density
            cpg                  = ground specific heat
            length_component     = length of the adjacent component
            number_elements      = number of elements in the radial heat transfer model
            dist_next_cavern_wall= axis distance between caverns
            Tg                   = average reservoir ground temperature far from
                                   the salt cavern (constant).
            dist_to_Tg_reservoir = distance from the heat transfer upper and
                                   lower surfaces to a reservoir at Tg
            dt0                  = initial time step (always stays the same for
                                   constant time step)
            bc                   = A dictionary with entries "Q0" and "Qend" for the
                                   flux into (Q0) and out of (Qend) the GHE. The value
                                   for Qend is often = 0.0 while Q0 is a variable of
                                   the component that the GHE is connected to.
            adj_comps             = list of The adjacent component (either one or more wells or
                                    the cavern) object. each entry is a tuple with the name
                                    (index 0) and object (index 1)
            global_indices       = tuple, first is the index (for the global xg vector)
                                   that this component starts at and the second is the
                                   last value (+1 for zero indexing) that this component
                                   uses in the global x vector
            Tgvec0               = Initial condition temperatures. If not provided,
                                   then the entire ground is assumed to be at Tg.


        """
        self._model = model
        if model is None:
            self._use_relative_convergence = False
            self.residual_normalization = None
        else:
            self._use_relative_convergence = model._use_relative_convergence
            if self._use_relative_convergence:
                if hasattr(model,"residual_normalization"):
                    self.residual_normalization = model.residual_normalization
                else:
                    self._use_relative_convergence = False
                    model.logging.info("GHE could not use relative converence"
                                       +" because residual normalization was "
                                       +"not provided!")
        self.dt = dt0
        self.number_elements = number_elements

        self.r_inner = r_inner

        self.length_component = length_component

        self.solutions = {}
        self.time_step_num = 0
        self.time = 0.0
        # this is just for not having to reinitialize over and over.

        self.Tg = Tg

        self.Q = np.zeros(number_elements+1)
        self.Q[0] = bc['Q0']
        self.Q[-1] = bc['Qend']

        self.Qg = np.zeros(number_elements)

        self.bc = bc

        self._gindices = global_indices
        self._adjacent_components = adj_comps

        # initial conditions
        if Tgvec0 is None:
            self.Tgvec_m1 = Tg * np.ones(number_elements+1)
            self.Tgvec = self.Tgvec_m1
        else:
            self.Tgvec_m1 = Tgvec0
            self.Tgvec = Tgvec0

        # Space finite volume elements on a log scale to allow a smooth capture of the
        # surface temperature gradient.
        rr = np.logspace(np.log10(r_inner),np.log10(dist_next_cavern_wall/2.0),num=number_elements+1)

        rrCg = [(r2+r1)/2 for r2,r1 in zip(rr[1:],rr[:-1])]
        rrCg.append(rr[-1])
        rrCg.insert(0,rr[0])

        # Radial Heat capacitance of rings of volume of ground
        self.Cg = [rhog * np.pi * (r2**2.0 - r1**2.0) * self.length_component * cpg for r2,r1 in zip(rrCg[1:],rrCg[:-1])]
        # Radial thermal resistivity of rings of volume of ground
        self.Rg = [np.log(rout/rin)/(2.0*np.pi * self.length_component * kg) for rout,rin in zip(rr[1:],rr[:-1])]
        self.Rg.append(self.Rg[-1]*1e4)  # This produces a zero flux boundary condition

        if isinstance(dist_to_Tg_reservoir,(float,int)):
            self.RTg = [dist_to_Tg_reservoir/(np.pi/4.0 * (rout**2 - rin**2) * kg) for rout,rin in zip(rrCg[1:],rrCg[:-1])]
        else:  # it is assumed that dist_to_Tg_reservoir is a list or array
            try:
                if len(dist_to_Tg_reservoir) != number_elements:
                    raise ValueError("The dist_to_Tg_reservoir must be an array or list of equal length as the number of elements")
            except:
                raise TypeError("dist_to_Tg_reservoir must be a float, array, or list!")

            self.RTg = [dist/(np.pi/4.0 * (rout**2 - rin**2) * kg) for rout,rin,dist in zip(
                rr[1:],rr[:-1],dist_to_Tg_reservoir)]

        self.grid = rr
        self._iter = 0
        self._range_index = range(0,self.number_elements)


    def _Qfunc(self,Tgvec,idx):
        return (-Tgvec[idx+1] + Tgvec[idx])/self.Rg[idx]

    @property
    def global_indices(self):
        return self._gindices

    @property
    def previous_adjacent_components(self):
        """
        Interface variable indices for the previous component
        previous means that the residuals for that component
        have already been calculated before this component
        """
        return self._adjacent_components

    @property
    def next_adjacent_components(self):
        """
        There are no next components for axisymmetric heat transfer
        it is the end of the line w/r to the model
        """
        return []
    
    @property
    def component_type(self):
        return ComponentTypes(2).name
    
    def shift_solution(self):
        """
        Make move the solution for the previous time step to the current time 
        step
        """
        self.Tgvec_m1 = self.Tgvec
        

    def evaluate_residuals(self, x=None, get_independent_vars=False):
        """
        This is a dynamic ODE implicit solution via euler integration

        NOTE:
        The solution assumes that each control volume has a flux in
        flux out (radially) and flux out (to ground). This gives
        num_elment + 1 fluxes and num_elemnet + 1 temperatures.
        We assume that the inner_radius flux is an independent variable
        while the last outer_radius flux is determined by the Fourier's
        law of the in and out temperatures of the last control volume

        An alternative would be to assume the inner_radius flux was determined
        by Fourier's law for  the 1st element but we preferred to keep
        this artificial condition at the end of the elements. For an insulated
        boundary condtition, this means the last two temperatures are
        equal to eachother (though they can rise together as heat comes
        into the last control volume).


        Inputs:
            x np.ndarray : list of variables below
                Variables:
                Local Index 0 = Q0 - flux into the system
                Local Inices 1 to num_element + 2 Tgvec:
                    vector of ground temperatures spaced radially from the GSHX
                Local Index Qend - flux out of the system (ussually fixed to 0.0)
            get_independent_vars bool : enables collecting variables not
                    normally output. (NOT IN USE HERE YET)

        Parameters:

            Tgvec_m1 - Tgvec for the previous time step

        Returns
        =======
            residuals - error level from solving the implicit equations
                        np.array of length number_elements + 1
        """
        # get prevous solution one time step ago
        self.dt = self._model.time_step
        Tgvec_m1 = self.Tgvec_m1
        Q = np.zeros(self.number_elements+1)
        #num_elem + 1 temperatures and two flux end conditions!
        residuals = np.zeros(self.number_elements + 3)

        #unpack to local variable names
        if x is None:
            #
            Q[0] = self.Q[0]
            Tgvec = self.Tgvec
            Q[-1] = self.Q[-1]
        else:
            gind = self.global_indices
            Q[0] = x[gind[0]]
            Tgvec = x[gind[0]+1:gind[1]]
            Q[-1] = x[gind[1]]
        
        # if self._model.time > 100000 and self._iter == 0:
        #     breakpoint()
        #     self._iter +=1

        # set the interface condition equations
        # or boundary condition equations
        prev_comp = self.previous_adjacent_components
        if len(prev_comp) == 0:
            residuals[0] = Q[0] - self.bc["Q0"]
        else:
            # convention heat leaving cavern is negative so we put + so that Q[0]
            # will come out positive.
            residuals[0] = Q[0] + self._q_axi_cavern.sum()
        if self._use_relative_convergence:
            residuals[0] = residuals[0]/self.residual_normalization["heat_flux_norm"]
            
        next_comp = self.next_adjacent_components
        if len(next_comp) == 0:
            # flux condition on end
            residuals[-1] = Q[-1] - self.bc["Qend"]
            # flux condition on end extends into the last CV.
            residuals[-2] = Q[-1] - self._Qfunc(Tgvec,self.number_elements-1)
        else:
            raise NotImplementedError("Connecting the GHE to anothe component has not been completed!")
            
        if self._use_relative_convergence:
            residuals[-1] = residuals[-1]/self.residual_normalization["heat_flux_norm"]
            residuals[-2] = residuals[-2]/self.residual_normalization["heat_flux_norm"]

        range_index = self._range_index


        Qg = [(Tgvec[idx] - self.Tg)/self.RTg[idx] for idx in range_index]

        Q[1:-1] = [self._Qfunc(Tgvec,idx) for idx in range(0,self.number_elements-1)]

        #Euler implicit integration of the ODE
            
        residuals[1:-2] = np.array([(Q[idx]-(Q[idx+1]+Qg[idx-1])) * self.dt / self.Cg[idx-1]
                    + Tgvec_m1[idx] - Tgvec[idx] for idx in range_index])
        if self._use_relative_convergence:
            residuals[1:-2] = residuals[1:-2]/self.residual_normalization["temperature_norm"]

        self.Q = Q
        self.Qg = Qg
        
        if get_independent_vars:
            Qind_vars = list(self.Q[1:-1])
            Qg_ind_vars = [Qg_ for Qg_ in self.Qg]
            return Qind_vars + Qg_ind_vars

        return residuals
    
    @property
    def independent_vars_descriptions(self):
        Qlist = [f"Heat flux from GHE element {idx} to {idx+1} (W)" 
                 for idx in range(self.number_elements+1) if idx !=0 and idx != self.number_elements] 
        Qglist = [f"Heat flux lost downward and/or upward from GHE element {idx+1}"
                  for idx in self._range_index]
        
        return Qlist + Qglist
         
         
    def equations_list(self):
        e_list = []
        e_list += ["Axisymmetric heat transfer flux continuity with cavern"]
        for idx in range(0,self.number_elements):
            e_list += [f"Axisymmetric heat transfer energy balance in control volume {idx}"]
        e_list += ["Axisymmetric heat transfer constant flux condition extending into last control volume"]
        e_list += ["Axisymmetric heat transfer constant flux exiting the ground heat exchanger"]
        return e_list
        
        

    def get_x(self):
        return np.array([self.Q[0]] + list(self.Tgvec) + [self.Q[-1]])

    def load_var_values_from_x(self,xg):
        
        
        bind,eind = self.global_indices
        self.Tgvec = xg[bind+1:eind]
        self.Q[0] = xg[bind]
        self.Q[-1] = xg[eind]
        
        if not self._model.is_test_mode:
        
            cavern = self._model.components['cavern']
            
            cgind = cavern.global_indices
            t_cavern = xg[cgind[0]]
            t_cavern_wall = xg[cgind[0]+1]
            m_brine = xg[cgind[0] + 2]
            t_brine = xg[cgind[0]+3]
            fluid = cavern._fluid
            fluid.set_state(CP.AbstractState,PT=[cavern._p_cavern,t_cavern])

            tup = cavern._water
            water = CP.AbstractState(tup[0],tup[1])
            water.update(CP.PT_INPUTS,tup[3][0],tup[3][1])
            water.set_mass_fractions(tup[2])
            
            m_cavern = xg[cgind[1]-cavern._number_fluids+1:cgind[1]+1]
            
            
            cavern_volume_estimate = cavern._VOL_TOTAL - cavern._initial_volume_brine
            (p_cavern,  
             vol_cavern) = calculate_cavern_pressure(fluid,
                                          m_cavern,
                                          t_cavern,
                                          water,
                                          m_brine,
                                          t_brine,
                                          cavern._VOL_TOTAL,
                                          cavern._area_horizontal,
                                          cavern_volume_estimate)
            
            p_brine = water.p()

            (mass_vapor, rho_vapor, h_vapor,
             p_vapor, h_evaporate) = conservation_of_volume(vol_cavern,
                 cavern._VOL_TOTAL, cavern._area_horizontal, water, t_cavern, 
                 t_brine, m_cavern, m_brine, fluid, True)  
                                                      
            
            
            vol_brine = cavern._VOL_TOTAL - vol_cavern                                              
            mass_frac = m_cavern / m_cavern.sum()
            fluid.set_mass_fractions(mass_frac)
            
            water.update(CP.PT_INPUTS,p_brine,t_brine)
            
            
            height_brine = vol_brine / cavern._area_horizontal
            height_cavern = cavern._height_total - height_brine
            height_total = cavern._height_total
            
            
            fluid.update(CP.PT_INPUTS,p_cavern,t_cavern)
            
            q_cavern, q_brine = cavern.cavern_wall_heat_flux(t_cavern,t_cavern_wall,
                                      t_brine,
                                      fluid,water,height_cavern,
                                      height_brine,height_total)
            
            
            
            self._q_axi_cavern = q_cavern + q_brine
        
            fluid.del_state()
        

