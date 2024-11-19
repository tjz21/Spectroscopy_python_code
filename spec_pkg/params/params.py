import os.path
import numpy as np
from spec_pkg.constants import constants as const

# Function definition
# function searches through input file for given keyword and returns keyword string
def get_param(filename,keyword):
    searchfile = open(filename,"r")
    line_count=0
    keyword_line=9999999
    for line in searchfile:
        # account for comments in input file 
        if keyword in line and keyword_line==9999999 and line[0]!='#':
            keyword_line=line_count
        line_count=line_count+1
    searchfile.close()
    if keyword_line < 9999999:
        linefile=open(filename,"r")
        lines=linefile.readlines()
        keyword=lines[keyword_line].split()
        return keyword[1]
    else:
        return ''


def get_param_list(filename,keyword):
    searchfile = open(filename,"r")

    line_count=0
    keyword_line=9999999
    for line in searchfile:
        # account for comments in input file
        if keyword in line and keyword_line==9999999 and line[0]!='#':
            keyword_line=line_count
        line_count=line_count+1
    searchfile.close()
    if keyword_line < 9999999:
        linefile=open(filename,"r")
        lines=linefile.readlines()
        keyword=lines[keyword_line].split()
        del keyword[0]
        return keyword
    else:
        return ''

#----------------------------------------------------------------------
# class definition

class params:
    def __init__(self,filepath):
        self.freq_cutoff_gbom=-1.0
        self.low_freq_cutoff=-1.0
        self.stdout=open(filepath+'.out','w')  # standard output file
        self.num_trajs=1
        self.num_gboms=1
        self.num_modes=0
        self.num_frozen_atoms=0    # only needed if Gaussian calculation or Terachem calculation and some atoms are frozen
        self.max_t=1000.0/const.fs_to_Ha # by default, start with 1000 fs worth of max_t
        self.num_steps=1000
        self.num_steps_2DES=100
        self.herzberg_teller=False
        self.ht_dipole_dipole_only=False # determine whether we do the full HT term or only the dipole-dipole correlation func in cumulant
        self.integration_points_morse=2000
        self.max_states_morse_gs=50 # maximum number of states considered in each morse oscillator
        self.max_states_morse_ex=50
        self.t_step_2DES=2.0/const.fs_to_Ha
        self.num_time_samples_2DES=50
        self.temperature=300.0   # temperature in kelvin at which the spectrum was simulated
        self.temperature_MD=300.0  # temperature at which the MD simulation was carried out
        self.exact_corr=True
        self.qm_wigner_dist=False  # Whether nuclei in ensemble appraoch are sampled with QM 
					   # wigner distribution. Only relevant for GBOM and ensemble method
        self.gs_reference_dipole=True  # Take the ground state as a reference in MD calculations with HT effects
        self.third_order=False
        self.cumulant_nongaussian_prefactor = False
        self.FC2DES_device = "CPU"
        self.g3_cutoff=-1.0
        self.is_solvent=False
        self.no_dusch=False
        self.solvent_reorg=0.0001
        self.solvent_cutoff_freq=0.0001	
        self.pump_energy=3.0/const.Ha_to_eV  # pump energy for pump probe
        self.omega1=0.0/const.Ha_to_eV   # energies for when 2DES spectrum is supposed
        self.omega3=0.0/const.Ha_to_eV	 # to be computed around a value other than mean
        self.model=''
        self.is_vertical_gradient=False
        self.task=''
        self.method=''
        self.scale_Jmat=False   # do we want to manually scale the Jmatrix?
        self.Jmat_scaling_fac=1.0  # this is the scaling factor
        self.Jpath=''
        self.Kpath=''
        self.freq_gs_path=''
        self.freq_ex_path=''
        self.E_adiabatic_path=''  # these two are needed if we have a batch of GBOMs
        self.dipole_mom_path=''
        self.E_opt_path=''
        self.MD_root=''
        self.MD_input_code=''
        self.GBOM_root=''
        self.add_emission_shift=False # shift omega_eg_av by twice the solvent broadening (GBOM) or the total spectral dens
						# reorganization energy (cumulant MD). This accounts for the fact that we have an 
						# emission spectrum and havent dealt with the changes this does to the average energy
						# gap.

        self.morse_gs_path='' # ground state potential parameters D and alpha, mu
        self.morse_ex_path='' # excited state potential parameters D and alpha, and shift (relative to gs minimum. Reduced mass is assumed to be identical to GS) 
        self.num_atoms=0   # needed for terachem calculation
        self.frozen_atom_path=''  # path to file list detailing the frozen atoms. 
        self.dipole_mom=np.array([1.0,0.0,0.0])  # dipole moment is a vector that by default has x,y,z coordinates
        self.E_adiabatic=0.0
        self.decay_length=500.0/const.fs_to_Ha
        self.md_step=2.0/const.fs_to_Ha
        self.md_num_frames=0   # number of MD frames. Only relevant if we read from a TeraChem file
        self.md_skip_frames=0  # skip the first n frames in an MD trajectory to allow for equilibration
        self.corr_length_3rd=1000
        self.four_phonon_term=True # if set to false, ignore 4 phonon contribution to two-time corr in GBOM (speeds up code)
        self.spectral_window=3.0/const.Ha_to_eV   # width of the window in which the spectrum gets calculated. 3eV as default
        self.target_excited_state=1  # TARGET excited state specifies which is the state we are interested in. Only really necessary for Terachem calculation where we extract excited state parameters directly from its output file. 

        # now start filling keyword list by parsing input file.
        self.task=get_param(filepath,'TASK')  # absorption, emission, 2DES, other spectroscopy techniqes
        self.model=get_param(filepath,'CHROMOPHORE_MODEL') # model for the chromophore degrees of freedom
								   # Current options: MD, GBOM, MORSE
        self.method=get_param(filepath,'METHOD') # ensemble, EZTFC, cumulant, FC etc, EOPT_AV  # EOPT_AV only works for GBOMs
        self.method_2DES=get_param(filepath,'NONLINEAR_EXP')  # 2DES, PUMP_PROBE
        self.Jpath=get_param(filepath,'JMAT')
        self.Kpath=get_param(filepath,'KVEC')
        self.freq_gs_path=get_param(filepath,'GS_FREQ') 
        self.freq_ex_path=get_param(filepath,'EX_FREQ')
        self.morse_gs_path=get_param(filepath,'GS_PARAM_MORSE')
        self.morse_ex_path=get_param(filepath,'EX_PARAM_MORSE')
        self.E_adiabatic_path=get_param(filepath, 'LIST_E_ADIAB')
        self.E_opt_path=get_param(filepath, 'EOPT_PATH')   # path to optimized excitation energies and oscillator strengths. 
        self.dipole_mom_path=get_param(filepath,'LIST_DIP_MOM')
        self.frozen_atom_path=get_param(filepath, 'FROZEN_ATOM_PATH') # path to file list detailing frozen atoms
        self.MD_root=get_param(filepath,'MD_ROOTNAME')	
        self.MD_input_code=get_param(filepath, 'MD_INPUT_CODE')
        self.GBOM_root=get_param(filepath, 'GBOM_ROOTNAME')
        self.GBOM_input_code=get_param(filepath, 'GBOM_INPUT_CODE') # specify whether the GBOM input is a Gaussian or Terachem input
									    # this will be extended to other supported codes

        # dealt with keywords that were names. Now deal with variables

        par = get_param(filepath, 'CUMULANT_NONGAUSSIAN_PREFACTOR')
        if par == 'TRUE':
            self.cumulant_nongaussian_prefactor = True
        if par == 'FALSE':
            self.cumulant_nongaussian_prefactor = False

        par = get_param(filepath, 'FC2DES_DEVICE')
        if par == 'CPU':
            self.FC2DES_device = 'CPU'
        if par == 'GPU':
            self.FC2DES_device = 'GPU'
        par=get_param(filepath,'GS_REFERENCE_DIPOLE')
        if par=='TRUE':
            self.gs_reference_dipole=True
        else:
            self.gs_reference_dipole=False

        par=get_param(filepath,'VERTICAL_GRADIENT')
        if par=='TRUE':
            self.is_vertical_gradient=True
        else:
            self.is_vertical_gradient=False

        par=get_param(filepath,'ADD_EMISSION_SHIFT')
        if par=='TRUE':
            self.add_emission_shift=True
        elif par=='FALSE':
            self.add_emission_shift=False

        # SCaling Jmatrix
        par=get_param(filepath,'SCALE_JMAT')
        if par=='TRUE':
            self.scale_Jmat=True
        elif par=='FALSE':
            self.scale_Jmat=False
        par=get_param(filepath,'JMAT_SCALING_FAC')
        if par != '':
            self.Jmat_scaling_fac=(float(par))


        par=get_param(filepath,'NUM_MODES')
        if par != '':
            self.num_modes=int(par)
        par=get_param(filepath,'INTEGRATION_POINTS_MORSE')
        if par != '':
            self.integration_points_morse=int(par)
        par=get_param(filepath,'MAX_STATES_MORSE_GS')
        if par != '':
            self.max_states_morse_gs=int(par)
        par=get_param(filepath,'MAX_STATES_MORSE_EX')
        if par != '':
            self.max_states_morse_ex=int(par)
        par=get_param(filepath,'NUM_GBOMS')
        if par != '':
            self.num_gboms=int(par)
        # check if we have multiple GBOMs or not. if we do, num frozen atoms is a list
        if self.num_gboms==1:
            par=get_param(filepath,'NUM_FROZEN_ATOMS')
            if par != '':
                self.num_frozen_atoms=int(par)
        else:
            par=get_param_list(filepath,'NUM_FROZEN_ATOMS')
            if par !='' and len(par)==self.num_gboms:
                self.num_frozen_atoms=np.zeros(self.num_gboms)
                counter=0
                for elem in par:
                    self.num_frozen_atoms[counter]=int(elem)
                    counter=counter+1

        par=get_param(filepath,'NUM_ATOMS')
        if par != '':
            self.num_atoms=int(par)
        par=get_param(filepath,'NUM_TRAJS')
        if par != '':
            self.num_trajs=int(par) 
        par=get_param(filepath,'MAX_T')
        if par != '':
            self.max_t=(float(par)/const.fs_to_Ha)
        par=get_param(filepath,'G3_CUTOFF')
        if par != '':
            self.g3_cutoff=(float(par)/const.fs_to_Ha)
        par=get_param(filepath,'LOW_FREQ_CUTOFF')
        if par != '':
            self.low_freq_cutoff=(float(par)/const.Ha_to_cm)
        par=get_param(filepath,'FREQ_CUTOFF_GBOM')   #ignore frequencies lower than that in GBOM
        if par != '':
            self.freq_cutoff_gbom=(float(par)/const.Ha_to_cm)
        par=get_param(filepath,'PUMP_ENERGY')
        if par != '':
            self.pump_energy=(float(par)/const.Ha_to_eV)
        par=get_param(filepath,'OMEGA1')
        if par != '':
            self.omega1=(float(par)/const.Ha_to_eV)
        par=get_param(filepath,'OMEGA3')
        if par != '':
            self.omega3=(float(par)/const.Ha_to_eV)

        par=get_param(filepath,'TIMESTEP_2DES')
        if par != '':
            self.t_step_2DES=(float(par)/const.fs_to_Ha)
        par=get_param(filepath,'DECAY_LENGTH')
        if par != '':
            self.decay_length=(float(par)/const.fs_to_Ha)
        par=get_param(filepath,'TEMPERATURE')
        if par != '':
            self.temperature=(float(par))
        par=get_param(filepath,'TEMPERATURE_MD')
        if par != '':
            self.temperature_MD=(float(par))
        else:
            self.temperature_MD=self.temperature # if Temp_MD is not specified, set it to temp
        par=get_param(filepath,'MD_STEP')
        if par != '':
            self.md_step=(float(par)/const.fs_to_Ha)
        par=get_param(filepath,'MD_NUM_FRAMES')
        if par != '':
            self.md_num_frames=int(par)
        par=get_param(filepath,'MD_SKIP_FRAMES')
        if par != '':
            self.md_skip_frames=int(par)
        par=get_param(filepath,'NUM_STEPS')
        if par != '':
            self.num_steps=int(par)
        par=get_param(filepath,'STEPS_2DES')
        if par != '':
            self.num_steps_2DES=int(par)
        par=get_param(filepath,'TARGET_EXCITED_STATE')
        if par != '':
            self.target_excited_state=int(par)
        par=get_param(filepath,'NUM_TIMESTEPS_2DES')
        if par != '':
            self.num_time_samples_2DES=int(par)
        par=get_param(filepath,'CORRELATION_LENGTH_3RD')
        if par != '':
            self.corr_length_3rd=int(par)
        par=get_param_list(filepath,'DIPOLE_MOM')
        if par != '':
            self.dipole_mom[0]=float(par[0])
            self.dipole_mom[1]=float(par[1])
            self.dipole_mom[2]=float(par[2])
        par=get_param(filepath,'E_ADIABATIC')
        if par != '':
            self.E_adiabatic=float(par)/const.Ha_to_eV
        par=get_param(filepath,'SPECTRAL_WINDOW')
        if par != '':
            self.spectral_window=float(par)/const.Ha_to_eV

        par=get_param(filepath, 'COMPUTE_4PHONON_TERM')
        if par != '':
            if par== 'FALSE':
                self.four_phonon_term=False
            if par== 'TRUE':
                self.four_phonon_term=True

        par=get_param(filepath, 'HERZBERG_TELLER')
        if par != '':
            if par== 'FALSE':
                self.herzberg_teller=False
            if par== 'TRUE':
                self.herzberg_teller=True

        par=get_param(filepath, 'NO_DUSCH')
        if par != '':
            if par== 'FALSE':
                self.no_dusch=False
            if par== 'TRUE':
                self.no_dusch=True

        par=get_param(filepath,'EXACT_CORRELATION_FUNC')
        if par != '':
            if par== 'FALSE':
                self.exact_corr=False
            if par== 'TRUE':
                self.exact_corr=True
        par=get_param(filepath,'QUANTUM_WIGNER_DIST')
        if par != '':
            if par== 'FALSE':
                self.qm_wigner_dist=False
            if par== 'TRUE':
                self.qm_wigner_dist=True
        par=get_param(filepath,'THIRD_ORDER_CUMULANT')
        if par != '':
            if par== 'FALSE':
                self.third_order=False
            if par== 'TRUE':
                self.third_order=True

        par=get_param(filepath,'HT_DIPOLE_DIPOLE_ONLY')
        if par != '':
            if par== 'FALSE':
                self.ht_dipole_dipole_only=False
            if par== 'TRUE':
                self.ht_dipole_dipole_only=True
        par=get_param(filepath,'SOLVENT_REORG')
        if par != '':
            self.solvent_reorg=float(par)
        par=get_param(filepath,'SOLVENT_CUTOFF_FREQ')
        if par != '':
            self.solvent_cutoff_freq=float(par)

        # could expand on this if we have different solvent models (ie. simple gaussian or lorentzian or voigt broadening
        par=get_param(filepath,'SOLVENT_MODEL')
        if par != '':
            if par == 'NONE':
                self.is_solvent=False
            if par == 'OHMIC':
                self.is_solvent=True
