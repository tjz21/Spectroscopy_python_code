from tests.Tester import Tester
import unittest

class test_MD(unittest.TestCase):
    def test_MD_cumulant(self):
        tester = Tester('input_file_MD_cumulant')
        tester.run_molspecpy()
        tester.check_files('MD_cumulant')
        tester.compare_spectra(self, 'MD_cumulant_spectrum.dat')
        tester.compare_spectra(self, 'MD_resonance_raman.dat')
        tester.compare_spectra(self, 'MD_spectral_density.dat')

    def test_MD_cumulant_ohmic(self):
        tester = Tester('input_file_MD_cumulant_ohmic')
        tester.run_molspecpy()
        tester.check_files('MD_cumulant_ohmic')
        tester.compare_spectra(self, 'MD_cumulant_spectrum.dat')
        tester.compare_spectra(self, 'MD_resonance_raman.dat')
        tester.compare_spectra(self, 'MD_spectral_density.dat')

    def test_MD_ensemble(self):
        tester = Tester('input_file_MD_ensemble')
        tester.run_molspecpy()
        tester.check_files('MD_ensemble')
        tester.compare_spectra(self, 'MD_ensemble_spectrum.dat')

    def test_MD_cumulant_3rd(self):
        tester = Tester('input_file_MD_cumulant_3rd')
        tester.run_molspecpy()
        tester.check_files('MD_cumulant_3rd')
        tester.compare_spectra(self, 'MD_cumulant_spectrum.dat')
        tester.compare_spectra(self, 'MD_resonance_raman.dat')
        tester.compare_spectra(self, 'MD_spectral_density.dat')

