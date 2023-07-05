from generate_spectra import main
import io
import numpy as np
# import tests.config as config
from tests.Tester import Tester
from os.path import *

class testMD(Tester):

    def test_MD_cumulant(self):
        self.assertEqual(2, 2)

    def test_MD_ensemble(self):
        input_file = join(self.data_dir, 'input_file_MD_ensemble')
        output_file = io.StringIO()
        main([input_file], outfile=output_file)
        self.check_files('MD_ensemble')

        for name in self.file_data:
            print(f'name={name}')
        
        self.compare_spectra('MD_ensemble_spectrum.dat')

if __name__ == '__main__':
    a = 1
    pass

        

    

