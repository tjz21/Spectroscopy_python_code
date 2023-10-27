from tests.Tester import Tester
from unittest import TestCase
import warnings
from numba.core.errors import NumbaPerformanceWarning

class GBOM_all_CL(TestCase):
    

    def runTest(self):
        # tester = Tester('input_file_GBOM_mBlue')
        if Tester.HIDE_WARNINGS:
            #   can't find where this file worning is happening, so it's ignored for now
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
        tester = Tester('input_file_GBOM_phenolate_cl')
        tester.run_molspecpy()
        tester.check_files('GBOM_cl')
        tester.compare_linear_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_linear_spectra(self, 'lineshape_function.dat')
        tester.compare_linear_spectra(self, 'full_correlation_func_classical.dat')
        tester.compare_linear_spectra(self, 'phenolate_EZTFC_spectrum_boltzmann_dist.dat')
        tester.compare_linear_spectra(self, 'FC_spectrum.dat')
        tester.compare_linear_spectra(self, 'phenolate_cumulant_spectrum_harmonic_qcf.dat')
        tester.compare_linear_spectra(self, 'phenolate_ensemble_spectrum_boltzmann_dist.dat')
        tester.compare_linear_spectra(self, 'phenolate_resonance_raman_harmonic_qcf.dat')
        tester.compare_linear_spectra(self, 'phenolate_spectral_density_harmonic_qcf.dat')

class GBOM_all_QM(TestCase):
    def runTest(self):
        if Tester.HIDE_WARNINGS:
            #   can't find where this file worning is happening, so it's ignored for now
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            warnings.simplefilter("ignore", DeprecationWarning) #   TODO: Fix complex warnings
        tester = Tester('input_file_GBOM_phenolate_exact')
        tester.run_molspecpy()
        tester.check_files('GBOM_exact')
        tester.compare_linear_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_linear_spectra(self, 'lineshape_function.dat')
        tester.compare_linear_spectra(self, 'full_correlation_func_qm.dat')
        tester.compare_linear_spectra(self, 'phenolate_EZTFC_spectrum_boltzmann_dist.dat')
        tester.compare_linear_spectra(self, 'FC_spectrum.dat')
        tester.compare_linear_spectra(self, 'phenolate_cumulant_spectrum_exact_corr.dat')
        tester.compare_linear_spectra(self, 'phenolate_ensemble_spectrum_boltzmann_dist.dat')
        tester.compare_linear_spectra(self, 'phenolate_resonance_raman_exact_corr.dat')
        tester.compare_linear_spectra(self, 'phenolate_spectral_density_exact_corr.dat')

