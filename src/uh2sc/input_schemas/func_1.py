if inp["valve"]["type"] == "orifice":
    if inp["valve"]["flow"] == "filling":
        k = self.res_fluid.cp0molar() / (self.res_fluid.cp0molar() - 8.314)
        self.mass_rate[0] = -tp.gas_release_rate(
            self.p_back,
            self.p0,
            self.res_fluid.rhomass(),
            k,
            self.CD,
            self.D_orifice ** 2 / 4 * math.pi,
        )
    else:
        self.mass_rate[0] = tp.gas_release_rate(
            self.p0,
            self.p_back,
            self.rho0,
            cpcv,
            self.CD,
            self.D_orifice ** 2 / 4 * math.pi,
        )
elif inp["valve"]["type"] == "mdot":
    if "mdot" in inp["valve"].keys() and "time" in inp["valve"].keys():
        mdot = np.asarray(inp["valve"]["mdot"])
        time = np.asarray(inp["valve"]["time"])
        max_i = int(time[-1] / self.tstep)+1
        interp_time = self.time_array
        self.mass_rate[:max_i] = np.interp(interp_time, time, mdot)
        if inp["valve"]["flow"] == "filling":
            self.mass_rate = -self.mass_rate
        elif inp["valve"]["flow"] == "discharge":
            self.mass_rate = self.mass_rate
        else:
            raise ValueError("inp['valve']['flow'] must equal one of ['filling','discharge']")
            

    else:
        if inp["valve"]["flow"] == "filling":
            self.mass_rate[:] = -inp["valve"]["mass_flow"]
        else:
            self.mass_rate[:] = inp["valve"]["mass_flow"]

elif inp["valve"]["type"] == "controlvalve":
    Cv = tp.cv_vs_time(self.Cv, 0, self.valve_time_constant, self.valve_characteristic)
    if inp["valve"]["flow"] == "filling":
        Z = self.res_fluid.compressibility_factor() 
        MW = self.MW
        k = self.res_fluid.cp0molar() / (self.res_fluid.cp0molar()-8.314)
        self.mass_rate[0] = -tp.control_valve(
            self.p_back, self.p0, self.T0, Z, MW, k, Cv
        )
    else:
        Z = self.fluid.compressibility_factor() 
        MW = self.MW 
        k = cpcv
        self.mass_rate[0] = tp.control_valve(
            self.p0, self.p_back, self.T0, Z, MW, k, Cv
        )
elif inp["valve"]["type"] == "psv":
    if inp["valve"]["flow"] == "filling":
        raise ValueError(
            "Unsupported valve: ",
            inp["valve"]["type"],
            " for vessel filling.",
        )
    self.mass_rate[0] = tp.relief_valve(
        self.p0,
        self.p_back,
        self.Pset,
        self.blowdown,
        cpcv,
        self.CD,
        self.T0,
        self.fluid.compressibility_factor(),
        self.MW,
        self.D_orifice ** 2 / 4 * math.pi,
    )
else:
    # validation should keep this from ever happening.
    raise ValueError("input valve types allowed are: ['psv','controlvalve','mdot','orifice']")