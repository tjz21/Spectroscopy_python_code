from tests.Tester import Tester
from unittest import TestCase

class MD_cumulant(TestCase):
    def runTest(self):
        tester = Tester('input_file_MD_cumulant')
        tester.run_molspecpy()
        tester.check_files('MD_cumulant')
        tester.compare_spectra(self, 'MD_cumulant_spectrum.dat')
        tester.compare_spectra(self, 'MD_resonance_raman.dat')
        tester.compare_spectra(self, 'MD_spectral_density.dat')

class MD_cumulant_ohmic(TestCase):
    def runTest(self):
        tester = Tester('input_file_MD_cumulant_ohmic')
        tester.run_molspecpy()
        tester.check_files('MD_cumulant_ohmic')
        tester.compare_spectra(self, 'MD_cumulant_spectrum.dat')
        tester.compare_spectra(self, 'MD_resonance_raman.dat')
        tester.compare_spectra(self, 'MD_spectral_density.dat')

class MD_ensemble(TestCase):
    def runTest(self):
        tester = Tester('input_file_MD_ensemble')
        tester.run_molspecpy()
        tester.check_files('MD_ensemble')
        tester.compare_spectra(self, 'MD_ensemble_spectrum.dat')

class MD_cumulant_3rd(TestCase):
    def runTest(self):
        tester = Tester('input_file_MD_cumulant_3rd')
        tester.run_molspecpy()
        tester.check_files('MD_cumulant_3rd')
        tester.compare_spectra(self, 'MD_cumulant_spectrum.dat')
        tester.compare_spectra(self, 'MD_resonance_raman.dat')
        tester.compare_spectra(self, 'MD_spectral_density.dat')

