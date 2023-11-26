from tests.Tester import Tester, FakeFile
from unittest import TestCase
from spec_pkg.nonlinear_spectrum import twoDES
import warnings
from numba.core.errors import NumbaPerformanceWarning

class GBOM_2nd_order_CL(TestCase):
    def runTest(self):
        if Tester.HIDE_WARNINGS:
            #   can't find where this file warning is happening, so it's ignored for now
            #   might be only a unittest thing
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
        tester = Tester('input_file_GBOM_2DES_2nd_order_cl')
        twoDES.file_writer = FakeFile
        tester.run_molspecpy()
        tester.check_files('GBOM_2DES_2nd_order_CL')
        tester.compare_linear_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_linear_spectra(self, 'phenolate_2DES_2nd_order_cumulant_averaged_spectrum.txt')
        tester.compare_spectra_2d(self, 'phenolate_2DES_0.dat')
        tester.compare_spectra_2d(self, 'phenolate_2DES_1.dat')
        tester.compare_spectra_2d(self, 'phenolate_2nd_order_cumulant_transient_absorption_spec.txt')

class GBOM_2nd_order_QM(TestCase):
    
    def runTest(self):
        if Tester.HIDE_WARNINGS:
            #   can't find where this file warning is happening, so it's ignored for now
            #   might be only a unittest thing
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
        tester = Tester('input_file_GBOM_2DES_2nd_order_exact')
        twoDES.file_writer = FakeFile
        tester.run_molspecpy()
        tester.check_files('GBOM_2DES_2nd_order_QM')
        tester.compare_linear_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_linear_spectra(self, 'phenolate_2DES_2nd_order_cumulant_averaged_spectrum.txt')
        tester.compare_spectra_2d(self, 'phenolate_2DES_0.dat')
        tester.compare_spectra_2d(self, 'phenolate_2DES_1.dat')
        tester.compare_spectra_2d(self, 'phenolate_2nd_order_cumulant_transient_absorption_spec.txt')