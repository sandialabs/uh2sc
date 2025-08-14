if inp["valve"]["type"] == "orifice":
    if inp["valve"]["flow"] == "filling":
        k = self.res_fluid.cp0molar() / (self.res_fluid.cp0molar() - 8.314)
        self.mass_rate[i] = -tp.gas_release_rate(
            self.p_back,
            self.P[i],
            self.res_fluid.rhomass(),
            k,
            self.CD,
            self.D_orifice ** 2 / 4 * math.pi,
        )
    else:
        self.mass_rate[i] = tp.gas_release_rate(
            self.P[i],
            self.p_back,
            self.rho[i],
            cpcv,
            self.CD,
            self.D_orifice ** 2 / 4 * math.pi,
        )
elif inp["valve"]["type"] == "controlvalve":
    Cv = tp.cv_vs_time(self.Cv,self.time_array[i],self.valve_time_constant,self.valve_characteristic)
    if inp["valve"]["flow"] == "filling":
        Z = self.res_fluid.compressibility_factor() 
        MW = self.MW 
        k = self.res_fluid.cp0molar() / (self.res_fluid.cp0molar() - 8.314)
        self.mass_rate[i] = -tp.control_valve(
            self.p_back, self.P[i], self.T0, Z, MW, k, Cv
        )
    else:
        Z = self.fluid.compressibility_factor() 
        MW = self.MW 
        self.mass_rate[i] = tp.control_valve(
            self.P[i], self.p_back, self.T_fluid[i], Z, MW, cpcv, Cv 
        )
elif inp["valve"]["type"] == "psv":
    self.mass_rate[i] = tp.relief_valve(
        self.P[i],
        self.p_back,
        self.Pset,
        self.blowdown,
        cpcv,
        self.CD,
        self.T_fluid[i],
        self.fluid.compressibility_factor(), 
        self.MW, 
        self.D_orifice ** 2 / 4 * math.pi,
    )
if 'end_pressure' in self.input['valve'] and self.P[i] > self.input['valve']['end_pressure']:
    massflow_stop_switch = 1
if massflow_stop_switch:
    self.mass_rate[i]=0