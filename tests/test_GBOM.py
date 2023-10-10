from tests.Tester import Tester
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning

class test_GBOM(unittest.TestCase):
    warnings.simplefilter("ignore", NumbaPerformanceWarning)

    def test_all_classical(self):
        # tester = Tester('input_file_GBOM_mBlue')
        #   can't find where this file worning is happening, so it's ignored for now
        warnings.simplefilter("ignore", ResourceWarning)
        tester = Tester('input_file_GBOM_phenolate_cl')
        tester.run_molspecpy()
        tester.check_files('GBOM_cl')
        tester.compare_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_spectra(self, 'lineshape_function.dat')
        tester.compare_spectra(self, 'full_correlation_func_classical.dat')
        tester.compare_spectra(self, 'phenolate_EZTFC_spectrum_boltzmann_dist.dat')
        tester.compare_spectra(self, 'FC_spectrum.dat')
        tester.compare_spectra(self, 'phenolate_cumulant_spectrum_harmonic_qcf.dat')
        tester.compare_spectra(self, 'phenolate_ensemble_spectrum_boltzmann_dist.dat')
        tester.compare_spectra(self, 'phenolate_resonance_raman_harmonic_qcf.dat')
        tester.compare_spectra(self, 'phenolate_spectral_density_harmonic_qcf.dat')


    def test_all_exact(self):
        # tester = Tester('input_file_GBOM_mBlue')
        #   can't find where this file worning is happening, so it's ignored for now
        warnings.simplefilter("ignore", ResourceWarning)
        tester = Tester('input_file_GBOM_phenolate_exact')
        tester.run_molspecpy()
        tester.check_files('GBOM_exact')
        tester.compare_spectra(self, '2nd_order_cumulant_from_spectral_dens.dat')
        tester.compare_spectra(self, 'lineshape_function.dat')
        tester.compare_spectra(self, 'full_correlation_func_qm.dat')
        tester.compare_spectra(self, 'phenolate_EZTFC_spectrum_boltzmann_dist.dat')
        tester.compare_spectra(self, 'FC_spectrum.dat')
        tester.compare_spectra(self, 'phenolate_cumulant_spectrum_exact_corr.dat')
        tester.compare_spectra(self, 'phenolate_ensemble_spectrum_boltzmann_dist.dat')
        tester.compare_spectra(self, 'phenolate_resonance_raman_exact_corr.dat')
        tester.compare_spectra(self, 'phenolate_spectral_density_exact_corr.dat')

