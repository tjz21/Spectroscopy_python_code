# /usr/bin/env python

from scipy import integrate
import numpy as np
import math
import spec_pkg.constants.constants as const
from spec_pkg.GBOM import gbom
from spec_pkg.GBOM import extract_model_params_gaussian as model_params
from spec_pkg.linear_spectrum import linear_spectrum
from spec_pkg.solvent_model import solvent_model
from spec_pkg.cumulant import md_traj

temp=300.0
max_t=30000
num_steps=1000
dipole_mom=1.0
S1_file='methylene_S1.log'
gs_file='methylene_gs.log'
vibronic_file='methylene_vibronic.log'
num_modes=38*3-6

freqs_gs=model_params.extract_normal_mode_freqs(gs_file,num_modes)
freqs_ex=model_params.extract_normal_mode_freqs(S1_file,num_modes)
K=model_params.extract_Kmat(vibronic_file,num_modes)
J=model_params.extract_duschinsky_mat(vibronic_file,num_modes)
E_adiab=model_params.extract_adiabatic_freq(vibronic_file)

methylene_gbom=gbom.gbom(freqs_gs,freqs_ex,J,K,E_adiab,dipole_mom)

methylene_gbom.calc_omega_av_cl(temp)
methylene_gbom.calc_omega_av_qm(temp)

methylene_gbom.calc_fc_response(temp,num_steps,max_t)
methylene_gbom.calc_ensemble_response(temp,num_steps,max_t)

np.savetxt('ensemble_response_imag.dat',methylene_gbom.ensemble_response.imag)

methylene_gbom.calc_g2_cl(temp,num_steps,max_t)
methylene_gbom.calc_g2_qm(temp,num_steps,max_t)

np.savetxt('g2_cl_real.txt',methylene_gbom.g2_cl.real)
np.savetxt('g2_qm_real.txt',methylene_gbom.g2_exact.real)

print methylene_gbom.g2_exact
methylene_gbom.calc_cumulant_response(False,True)

#methylene_gbom.calc_g3_cl(temp,num_steps,max_t)
#methylene_gbom.calc_g3_qm(temp,num_steps,max_t)
#np.savetxt('g3_cl_real.txt',methylene_gbom.g3_cl.real)

decay_length=500.0/const.fs_to_Ha
methylene_gbom.calc_spectral_dens(temp,max_t,num_steps,decay_length,False)

np.savetxt('spectral_dens.dat',methylene_gbom.spectral_dens)

methylene_gbom.calc_spectral_dens(temp,max_t,num_steps,decay_length,True)

np.savetxt('spectral_dens_cl.dat',methylene_gbom.spectral_dens)

energy_range=1.0/const.Ha_to_eV

omega_c=0.0001
lambda_solv=0.0001
solvent_mod=solvent_model.solvent_model(lambda_solv,omega_c)
solvent_mod.calc_spectral_dens(num_steps)
solvent_mod.calc_g2_solvent(temp,num_steps,max_t)
solvent_mod.calc_solvent_response()

print solvent_mod.g2_solvent

start_val=methylene_gbom.E_adiabatic-energy_range/2.0
end_val=methylene_gbom.E_adiabatic+energy_range/2.0
spectrum=linear_spectrum.full_spectrum(methylene_gbom.cumulant_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)

np.savetxt('cumulant_spectrum.dat',spectrum)

#methylene_gbom.calc_cumulant_response(True,False)
#spectrum=linear_spectrum.full_spectrum(methylene_gbom.cumulant_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)

#np.savetxt('cumulant_spectrum_3rd_cl.dat',spectrum)

methylene_gbom.calc_cumulant_response(False,False)
spectrum=linear_spectrum.full_spectrum(methylene_gbom.cumulant_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)
np.savetxt('cumulant_spectrum_cl.dat',spectrum)

spectrum=linear_spectrum.full_spectrum(methylene_gbom.fc_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)

np.savetxt('ftfc_spectrum.dat',spectrum)

spectrum=linear_spectrum.full_spectrum(methylene_gbom.ensemble_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)

np.savetxt('ensemble_spectrum.dat',spectrum)

methylene_gbom.calc_eztfc_response(temp,num_steps,max_t)

spectrum=linear_spectrum.full_spectrum(methylene_gbom.eztfc_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)

np.savetxt('eztfc_spectrum.dat',spectrum)

# MM Traj test:

temp_traj=np.genfromtxt('raw_excitations_methylene_blue_camb3lyp_MMwater_diabatic_traj1.dat')
traj=np.zeros((temp_traj.shape[0],1))
traj[:,0]=temp_traj[:,0]
oscillators=np.zeros((temp_traj.shape[0],1))
oscillators[:,0]=temp_traj[:,1]
time_step=2.0/const.fs_to_Ha
print time_step

methylene_traj=md_traj.MDtrajs(traj,oscillators,decay_length,1,time_step)
methylene_traj.calc_2nd_order_corr()
methylene_traj.calc_spectral_dens(temp)

np.savetxt('methylene_spectral_dens_from_MD.dat',methylene_traj.spectral_dens)

methylene_traj.calc_g2(temp,max_t,num_steps)
np.savetxt('methylene_md_g2_real.dat',methylene_traj.g2.real)

methylene_traj.calc_cumulant_response(False)

spectrum=linear_spectrum.full_spectrum(methylene_traj.cumulant_response,np.zeros((1,1)),num_steps,start_val,end_val,False)

np.savetxt('linear_spectrum_cumulant_methylene_MD.dat', spectrum)

methylene_traj.calc_ensemble_response(max_t,num_steps)

np.savetxt('MD_ensemble_response_func_real.dat', methylene_traj.ensemble_response.real)
np.savetxt('MD_ensemble_response_func_imag.dat', methylene_traj.ensemble_response.imag)

spectrum=linear_spectrum.full_spectrum(methylene_traj.ensemble_response,solvent_mod.solvent_response,num_steps,start_val,end_val,True)

np.savetxt('linear_spectrum_ensemble_methylene_MD.dat', spectrum)

#correlation_length=500
#methylene_traj.calc_3rd_order_corr(correlation_length)
#methylene_traj.calc_g3(temp,max_t,num_steps)

#np.savetxt('methylene_md_g3_real.dat',methylene_traj.g3.real)
#methylene_traj.calc_cumulant_response(True)

#spectrum=linear_spectrum.full_spectrum(methylene_traj.cumulant_response,np.zeros((1,1)),num_steps,start_val,end_val,False)

#np.savetxt('linear_spectrum_3rd_order_cumulant_methylene_MD.dat', spectrum)
