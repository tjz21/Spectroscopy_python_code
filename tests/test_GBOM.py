from tests.Tester import Tester
import unittest

class test_FC(unittest.TestCase):
    def test_MD_cumulant(self):
        # tester = Tester('input_file_GBOM_mBlue')
        tester = Tester('input_file_GBOM_phenolate')
        tester.run_molspecpy()
        tester.check_files('FC')
        tester.compare_spectra(self, 'FC_spectrum.dat')
        tester.compare_spectra(self, 'lineshape_function.dat')