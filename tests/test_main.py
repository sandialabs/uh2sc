# dlvilla created 01/20/2025 (Trump's inauguration day)
import cProfile
import pstats
import io
import unittest
import os

from uh2sc.main import main,main_list


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_all = True

    @classmethod
    def tearDownClass(cls):
        pass

    #@unittest.skip("The Vertical Pipe Class is under construction and the current"
    #    +" step is to add relative_roughness to the data input schema for wells.")
    def test_main(self):
        if self.run_all:
            rundir= os.path.join(os.path.dirname(__file__),"test_data")
            main(os.path.join(os.path.dirname(__file__),"test_data",
                 "nieland_verification_h2_SHORT.yml"),
                 os.path.join(rundir,"test_delete_me.csv"),
                 True,
                 True,
                 os.path.join(rundir,"main_test_delete_me_log.log"))
            main_list("schema_general.yml",None,None,None)
            main_list(None,True,None,None)
            main_list(None,None,os.path.join(rundir,"nieland_verification_h2_SHORT.yml"),None)
            main_list("schema_general.yml",None,None,"cavern")
            main_list("schema_general.yml",None,None,"cavern,height")
            main_list(None,None,os.path.join(rundir,"nieland_verification_h2_SHORT.yml"),"cavern,height")



if __name__ == "__main__":
    PROFILE = False

    if PROFILE:

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(TestMain())

    if PROFILE:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('main_test_profile.txt', 'w+', encoding='utf-8') as f:
            f.write(s.getvalue())
