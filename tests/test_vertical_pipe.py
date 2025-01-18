# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:21:12 2023

Began modifications again on Jan 17, 2025.

@author: dlvilla
"""
import os
import unittest
from uh2sc.well import VerticalPipe,PipeMaterial 
from uh2sc.salt_cavern import SaltCavern
from uh2sc.constants import Constants as const
import numpy as np
from CoolProp import CoolProp as CP

class Test_VerticalPipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # generic inputs.
        cls.plot_results = False
        cls.run_all_tests = False
        
        cls.pipe_material = PipeMaterial(surface_roughness_height_ratio=0.0001,
                                     thermal_conductivity=45)  #m/m and W/m/K
        cls.valve = {"type":"mdot",
                "time":[0.0,2628000.0],
                "mdot":[13.0196,13.0196],
                "reservoir":{"pressure":8e6,
                             "temperature": 310.92777777777777775,
                             "fluid":'H2'}}
        cls.valve_name = "inflow_mdot"
        cls.initial_temperature = 300
        cls.initial_pressure = 5e6
        cls.outside_inner_diameter = 0.0
        cls.inside_inner_diameter = 0.0
        cls.inside_outer_diameter = 0.2
        cls.outside_outer_diameter = 0.25
        cls.number_control_volumes = 100
        cls.total_minor_losses_coefficient = 0.0
        
        cls.initial_pressure = 2.65e4 # Pa
        cls.initial_temperature = -49.90 + const.K_C_offset['value']
        cls.mass_rate0 = 1 # just a dummy value
        comp = "Air"
        molefracs = [1.0]
        length = 10000
        height = 10000
        
        cls.vp1 = VerticalPipe(molefracs,
                          comp,
                          length,
                          height,
                          cls.pipe_material,  
                          cls.valve,
                          cls.valve_name,
                          cls.initial_pressure,
                          cls.initial_temperature,  
                          cls.outside_inner_diameter,  
                          cls.inside_inner_diameter,                 
                          cls.inside_outer_diameter,
                          cls.outside_outer_diameter,
                          cls.number_control_volumes,
                          cls.total_minor_losses_coefficient,
                          )
        
        inp = os.path.join(os.path.dirname(__file__),"test_data",
                           "salt_cavern_mdot_only_test.yml")
        
        cls.cavern = SaltCavern(inp)
        
    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_adiabatic_column(self):
        if self.run_all_tests:
            # US standard atmosphere at 10,000 ft testing to see if dry adiabatic lapse rate
            # and exponential pressure drop come through at the correct values

            # this should produce the dry air adiabatic lapse rate of the atmosphere
            # and standard U.S. atmospheric pressure
            Tfluid, Pfluid, rhofluid = self.vp1._initial_adiabatic_static_column(self.initial_temperature, self.initial_pressure, self.mass_rate0, True)
            
            # verify this is an adiabatic process where Pressure / Density ** (specific heat ratio) = constant
            constant = Pfluid / rhofluid ** 1.4
            
            self.assertTrue(np.sum(np.diff(constant)/constant[1:] * 100 > 0.05)==0)
            
            dry_adiabatic_lapse_rate = 9.8 # C/km
            
            # The dry air adiabatic lapse rate of the atmosphere results.
            self.assertTrue((Tfluid[-1] - Tfluid[0])/10 - dry_adiabatic_lapse_rate < 0.001) 


    @unittest.skip("The development of these features is not complete")
    def test_pipe_pressure_loss(self):
        """

        This test is under development and cannot be expected to pass
        until the pipe equations are completed (dlv - 01)

        """
        
        
        if self.run_all_tests:
            
            # the following values have been set from Adair and Power 1961.
            # "Polytropic Flow of Natural Gas in Vertical Pipes"
            # https://watermark.silverchair.com/spe-30-ms.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2UwggNhBgkqhkiG9w0BBwagggNSMIIDTgIBADCCA0cGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMudAzE3UAGeSveDoPAgEQgIIDGHIjAm0v-uEz0nqR3wW764Mp7XT_wRRX9wIt6GM26KwEe313pAIGQF2LOQ4hR3qt6OzXL_DLJs-TVfXlEtaH_n7Glm4HxnHEKoulJiRJUpKK6ZHPjFshIzHFeybljdsyu4a3sY-Vweq5DfYloLejeJKcBpi_aed3PPW61XpGajx6Q09_yJjhHk8hfiErbdUqzMnNn052uPA__cEyOUDdlzdkKIPPzL6lHsmWfg5LL_D6SzWLemS1JZ_K87svyogs4CBC-Qi65_385RbECOqseu4d-4DFI6ChgV0Y1cR7N2uSZfzMFpCDduPtn0cALpC-HQT3usVEmltTOguXDE0KPkMbSZoLszMPYtbs-62hp89BKhKwTXmOe5Un4PZEOH8WrWsCi8hM0dWyZcgXUALUjWNgDynkpG9FUYehkW2tZ1Y9vhyULhwSCW10f94xMu78kY0l1BtRNiX2YPHgum9F0tIGPbdfeqKoNtgEtfRemPB56GMG6fdvHMnobJ0vIioxAI3cgZW-k3B_LkMnXxSH8eVuXowcSfnaDEckCdiLqkzSY9sgtA6ZvgD7SgsGqjYzdjiwy8XbMbI5OnZbcD6MBdE5pNi7ui0edagYFn1kDlFKCOQpk6oOMpRsmvsxU-PybF7dzKop3mhhgpnbqOBfbWweC1Pkm5iSDiiyBXNVpv6mluQv_J4YDNnU6kRBEJnsGcFXwhcziqx6LTDSOvvyb2f9_d7sX9pueWP7xN4AdcC5K5Pl6f5irYa-DFAXfSOKopLRGTBfvSOvIEKOroivi2pFwWU6Fl-HehuqFS68daKMIYVJT5i0D8hbCdqnUBIBGknwaiK3s9B8rimGKGZlSCeH-jOJxITnAaIl9b0RKDXac23bSN0YK92ANCvzQEPiC4UPmz25nC2FDKr8sBkWxI8o6Ku1mkuVleOMacSaM7vBZ88X--SWFiBUDyiE9Y6izQhv1BiFDA0oM1CzyUwKhsQr3ia04XWupEfO98Peq14M4txVT3op3L-jQE4M3v3MZZ1KGTMbdabJRBJ0O4eE5dy0bwsQsHgPNQ
            pipe_diameter = 1.995 *const.inches_2_meters['value']
            pipe_length = 5790 * const.feet_2_meters['value']
            
            roughness_height_ratio = 0.0006 * const.inches_2_meters['value']/pipe_diameter
            
            volume_flow_rate = 2.760 * const.mmscf_p_day_2_m3_p_s['value'] 
            flow_pressure = 14.65 *const.psi_2_pascal['value'] # Pa
            flow_temperature = const.Rankine_2_Kelvin['value']*(60 + const.F_R_offset['value']) #K
            
            comp = "Methane&Ethane&Propane&Nitrogen&CarbonDioxide"
            
            molefracs = [0.947,0.044,0.002,0.004,0.003]  # from https://www.enbridgegas.com/en/about-enbridge-gas/learn-about-natural-gas
            # added .001 to methane composition to make value sum exactly to
            # one. Many trace components (Oxygen, Hydrogen, pentane, hexanes, etc... not included)
            
            fluid = CP.AbstractState("HEOS", comp)
            fluid.specify_phase(CP.iphase_gas)
            fluid.set_mole_fractions(molefracs)
            
            # find out the density at the conditions set
            fluid.update(CP.PT_INPUTS, flow_pressure,  
                                            flow_temperature)
            
            mass_flow_rate = volume_flow_rate * fluid.rhomass()
            
            
            
            # thermal conductivity does not matter in this test!
            pipe_material = PipeMaterial(surface_roughness_height_ratio=roughness_height_ratio,
                                         thermal_conductivity=45)  #m/m and W/m/K
            
            # reservoir doesn't matter because flow is out of the reservoir (production)
            
            valve = {"type":"mdot",
                    "time":[0.0,6000],
                    "mdot":[-mass_flow_rate,-mass_flow_rate],
                    "reservoir":{"pressure":2598 * const.psi_2_pascal['value'],
                                 "temperature": const.Rankine_2_Kelvin['value']*(77.0 + const.F_R_offset['value']),
                                 "fluid":comp}}

            
            # we set the unknown as the known here since our program will
            # calculate the reverse (i.e. pressure at well-head during production)
            initial_pressure = 2598 * const.psi_2_pascal['value']
            initial_temperature = const.Rankine_2_Kelvin['value']*(160 + const.F_R_offset['value'])
          
            # must set these as well since the cavern is only a dummy and 
            # is not being solved.
            self.cavern.T_cavern[0] = initial_temperature
            self.cavern.P_cavern[0] = initial_pressure
          
            self.vp2 = VerticalPipe(molefracs,
                              comp,
                              pipe_length,
                              pipe_length,
                              pipe_material,  
                              valve,
                              self.valve_name,
                              initial_pressure,
                              initial_temperature,  
                              self.outside_inner_diameter,  
                              self.inside_inner_diameter,                 
                              pipe_diameter,
                              pipe_diameter + .01, # this does not matter
                              self.number_control_volumes,
                              self.total_minor_losses_coefficient,
                              )

            self.vp2.step(self.cavern, "cavern_well", 0)
            
            
            
            P_psi = self.vp2.P_fluid / const.psi_2_pascal['value']
            
            delP_total = P_psi[-1] - P_psi[0]
            
            target_delP = 333.5
            
            error = 100*(delP_total - target_delP)/target_delP
            
            # This is where I left off in December 2023
            # The equations for vertical pipes must be made to work as an independent class that
            # can be coupled to a set of radial equations for another pipe, (on both sides), radial conditions with
            # the ground or to a more simple boundary condition that doesn't require another model and is connected to
            # a constant far-field temperature. This last conditions is.
            # for this validation.
            #breakpoint()
            
            pass

            
    @unittest.skip("Development of the vertical pipe is not complete")
    def test_single_vertical_pipe(self):
        """
        This test needs to be used for a general verification that the
        VerticalPipe class
        can be instantiated and each function can be called. It is not
        A verification test with comparisons to physically meaningful
        results. 
        
        It should run an analysis on a single vertical pipe, 
        2 vertical pipes, 3 vertical pipes, and 4 vertical pipes.
        without the need for an input deck.

        """


        if self.run_all_tests:

            molefracs = [0.1, 0.9]  #fraction
            comp = "H2&CH4"
            length = 1000
            height = length
            
            vp = VerticalPipe(molefracs,
                              comp,
                              length,
                              height,
                              self.pipe_material,  
                              self.valve,
                              self.valve_name,
                              self.initial_pressure,
                              self.initial_temperature,  
                              self.outside_inner_diameter,  
                              self.inside_inner_diameter,                 
                              self.inside_outer_diameter,
                              self.outside_outer_diameter,
                              self.number_control_volumes,
                              self.total_minor_losses_coefficient,
                              )
            
            vp.step(self.cavern,'cavern_well', 0)
            
            if self.plot_results:
                #plot something
                pass
            self.assertTrue(True)

        
        #TODO = VERIFY pipe pressure loss and performance in isothermal conditions
        #    using results from https://www.pressure-drop.com/Online-Calculator/
        

if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_VerticalPipe())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('utilities_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())            