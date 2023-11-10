from tests.Tester import Tester, FakeFile
from unittest import TestCase
from spec_pkg.nonlinear_spectrum import twoDES


class MD_2nd_order(TestCase):
    def runTest(self):
        tester = Tester('input_file_MD_2DES_2nd_order')
        twoDES.file_writer = FakeFile
        tester.run_molspecpy()
        tester.check_files('MD_2DES_2nd_order')
        tester.compare_linear_spectra(self, 'pyp_MD_spectral_density.dat')
        tester.compare_linear_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_linear_spectra(self, 'pyp__2DES_2nd_order_cumulant_averaged_spectrum.txt')
        tester.compare_spectra_2d(self, 'pyp__2DES_0.dat')
        tester.compare_spectra_2d(self, 'pyp__2DES_1.dat')
        tester.compare_spectra_2d(self, 'pyp__2nd_order_cumulant_transient_absorption_spec.txt')
        
