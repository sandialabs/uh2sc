# dlvilla created 01/20/2025 (Trump's inauguration day)
import cProfile
import pstats
import io
import unittest
import os

from uh2sc.main import main


class Test_Main(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_all = True
  
    @classmethod
    def tearDownClass(cls):
        pass

    @unittest.skip("The Vertical Pipe Class is under construction and the current step is to add relative_roughness to the data input schema for wells.")
    def test_main(self):
        if self.run_all:
            main(os.path.join(os.path.dirname(__file__),"test_data","salt_cavern_mdot_only_test.yml"))



if __name__ == "__main__":
    profile = False
    
    if profile:

        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_Main())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('main_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())     