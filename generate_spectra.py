#!/usr/bin/env python

from scipy import integrate
import sys
import os
import numpy as np
import math
import cmath
from numba import config
import spec_pkg.constants.constants as const
from spec_pkg.GBOM import gbom
from spec_pkg.GBOM import extract_model_params_gaussian as gaussian_params
from spec_pkg.GBOM import extract_model_params_from_terachem as terachem_params
from spec_pkg.GBOM import hessian_to_GBOM as hess_to_gbom
from spec_pkg.linear_spectrum import linear_spectrum
from spec_pkg.nonlinear_spectrum import twoDES
from spec_pkg.solvent_model import solvent_model
from spec_pkg.cumulant import md_traj
from spec_pkg.params import params

# TODO ##########################################################################
# 2) Implement emission spectra calculations for the GBOM approach		#
# 3) Write simulation information to stdout (what calculation is done, how it	#
#	 progresses etc.							#
# 4) Make sure we can input shapes of pulses for the ultrafast spectroscopy,	#
#	 rather than just delta-function					#
# 5) Implement GBOM batch absorption calculation				#
# 6) Implement combined GBOM_MD model needed for E-ZTFC and similar appraoches	#
# 7) Implement a PIMD version of all the methods				#
#################################################################################


# specific routines:
# compute absorption spectra and print them if chromophore model is defined purely a single GBOM
def compute_GBOM_absorption(param_list,GBOM_chromophore,solvent,is_emission):
		# first compute solvent response
		solvent.calc_spectral_dens(param_list.num_steps)
		solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t)
		solvent.calc_solvent_response(is_emission)

		# if this is an emission calculation, need to reset some standard gbom parameters:
		if is_emission:
				GBOM_chromophore.set_emission_variables()
				 
		# figure out start and end value for the spectrum.
		if param_list.exact_corr:
				GBOM_chromophore.calc_omega_av_qm(param_list.temperature,is_emission)
				E_start=GBOM_chromophore.omega_av_qm-param_list.spectral_window/2.0
				E_end=GBOM_chromophore.omega_av_qm+param_list.spectral_window/2.0
				print('omega av QM')
				print(GBOM_chromophore.omega_av_qm)
		else:
				GBOM_chromophore.calc_omega_av_cl(param_list.temperature,is_emission)
				E_start=GBOM_chromophore.omega_av_cl-param_list.spectral_window/2.0
				E_end=GBOM_chromophore.omega_av_cl+param_list.spectral_window/2.0
				print('omega av cl')
				print(GBOM_chromophore.omega_av_cl)


		if param_list.method=='ENSEMBLE':
				GBOM_chromophore.calc_ensemble_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)		
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)	
				if param_list.qm_wigner_dist:
						np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_qm_wigner_dist.dat', spectrum)
				else:
						np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_boltzmann_dist.dat', spectrum)
		elif param_list.method=='FC':
				GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.fc_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
				np.savetxt(param_list.GBOM_root+'_FC_spectrum.dat', spectrum)
		elif param_list.method=='EZTFC':
				GBOM_chromophore.calc_eztfc_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.eztfc_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
				if param_list.qm_wigner_dist:
						np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_qm_wigner_dist.dat', spectrum)
				else:
						np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_boltzmann_dist.dat', spectrum)
		elif param_list.method=='CUMULANT':
				if param_list.exact_corr:
						# spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
						GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
						np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
						GBOM_chromophore.calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
						#save lineshape function:
						#np.savetxt(param_list.GBOM_root+'_lineshape_func_2nd_order_exact_corr.dat',(GBOM_chromophore.g2_exact).real)

						# only compute third order cumulant if needed
						if param_list.third_order:
								GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)
								# sace lineshape function
								#temp_lineshape2=(GBOM_chromophore.g2_exact).real
								#temp_lineshape2[:,1]=(GBOM_chromophore.g2_exact[:,1]).real+(GBOM_chromophore.g3_exact[:,1]).real
								#np.savetxt(param_list.GBOM_root+'_lineshape_func_3rd_order_exact_corr.dat',temp_lineshape2)

								# also print only g:
								#temp_lineshape2=(GBOM_chromophore.g3_exact).real
								#np.savetxt(param_list.GBOM_root+'_g3_exact_corr_real.dat',temp_lineshape2)
								#temp_lineshape2=(GBOM_chromophore.g3_exact).real
								#temp_lineshape2[:,1]=(GBOM_chromophore.g3_exact[:,1]).imag
								#np.savetxt(param_list.GBOM_root+'_g3_exact_corr_imag.dat',temp_lineshape2)

				else:
						GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
						np.savetxt(param_list.GBOM_root+'_spectral_density_harmonic_qcf.dat', GBOM_chromophore.spectral_dens)
						GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
						#save lineshape function:
						#np.savetxt(param_list.GBOM_root+'_lineshape_func_2nd_order_harmonic_qcf.dat',(GBOM_chromophore.g2_cl).real)

						if param_list.third_order:
								GBOM_chromophore.calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)
								# sace lineshape function
								#temp_lineshape=(GBOM_chromophore.g2_cl).real
								#temp_lineshape[:,1]=temp_lineshape[:,1]+(GBOM_chromophore.g3_cl[:,1]).real
								#np.savetxt(param_list.GBOM_root+'_lineshape_func_3rd_order_harmonic_qcf.dat',temp_lineshape)


				GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission)		
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.cumulant_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
				if param_list.exact_corr:
						np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_exact_corr.dat', spectrum)
				else:
						np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_harmonic_qfc.dat', spectrum)
				
		# do all approaches, including qm wigner sampling and exact and approximate 
		# quantum correlation functions for the cumulant approach
		elif param_list.method=='ALL':
				GBOM_chromophore.calc_ensemble_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
				if param_list.qm_wigner_dist:
						np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_qm_wigner_dist.dat', spectrum)
				else:
						np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_boltzmann_dist.dat', spectrum)

				GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t, is_emission)
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.fc_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
				np.savetxt(param_list.GBOM_root+'_FC_spectrum.dat', spectrum)
				GBOM_chromophore.calc_eztfc_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.eztfc_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)

				if param_list.qm_wigner_dist:
						np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_qm_wigner_dist.dat', spectrum)
				else:
						np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_boltzmann_dist.dat', spectrum)

				if param_list.exact_corr:
						# spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
						GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
						np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
						GBOM_chromophore.calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
						# only compute third order cumulant if needed
						if param_list.third_order:
								GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)
				else:
						GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
						np.savetxt(param_list.GBOM_root+'_spectral_density_harmonic_qfc.dat', GBOM_chromophore.spectral_dens)
						GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
						if param_list.third_order:
								GBOM_chromophore.calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)
				GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr, is_emission)		
				spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.cumulant_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)

				if param_list.exact_corr:
						np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_exact_corr.dat', spectrum)
				else:
						np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_harmonic_qcf.dat', spectrum)

		else:
				sys.exit('Error: Unknown method '+param_list.method)


# compute absorption spectra when chromophore model is given by a batch of GBOMS
def compute_GBOM_batch_absorption(param_list,GBOM_batch,solvent,is_emission):
		# first compute solvent response
		solvent.calc_spectral_dens(param_list.num_steps)
		solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t)
		solvent.calc_solvent_response(is_emission)
		
		# Now make sure that we have only a single average spectral window for the GBOM batch. 
		# also set the correct omega_av.
		icount=0
		average_Egap=0.0
		while icount<GBOM_batch.num_gboms:
		# figure out start and end value for the spectrum.
				if param_list.exact_corr:
						GBOM_batch.gboms[icount].calc_omega_av_qm(param_list.temperature,is_emission)
						average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
				else:
						GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
						average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl

				icount=icount+1

		average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)

		E_start=average_Egap-param_list.spectral_window/2.0
		E_end=average_Egap+param_list.spectral_window/2.0

		if param_list.method=='FC':
				# Compute FC response for all elements in the GBOM batch
				print('Computing FC response')
				icount=0 
				spectrum=np.zeros((param_list.num_steps,2))
				while icount<GBOM_batch.num_gboms:
						print('Processing GBOM batch '+str(icount))
						GBOM_batch.gboms[icount].calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

						temp_spectrum=linear_spectrum.full_spectrum(GBOM_batch.gboms[icount].fc_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
						if icount==0:
								spectrum=spectrum+temp_spectrum
						else:
								spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]
						icount=icount+1

				spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)
				np.savetxt(param_list.GBOM_root+'_E_FTFC_spectrum.dat', spectrum)

		# cumulant spectrum for all elements in the GBOM batch. The result is the summed spectrum
		elif param_list.method=='CUMULANT':
				print('Computing Cumulant response')
				icount=0
				spectrum=np.zeros((param_list.num_steps,2))
				while icount<GBOM_batch.num_gboms:
						if param_list.exact_corr:
								# spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
								GBOM_batch.gboms[icount].calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

								# only compute third order cumulant if needed
								if param_list.third_order:
										GBOM_batch.gboms[icount].calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)

						else:
								GBOM_batch.gboms[icount].calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

								if param_list.third_order:
										GBOM_batch.gboms[icount].calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)


						GBOM_batch.gboms[icount].calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission)
						temp_spectrum=linear_spectrum.full_spectrum(GBOM_batch.gboms[icount].cumulant_response,solvent.solvent_response,param_list.dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)
						print(E_start,E_end)
						if icount==0:
								spectrum=temp_spectrum
						else:
								spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]
						icount=icount+1

				spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)
				np.savetxt(param_list.GBOM_root+'_E_cumulant_spectrum.dat', spectrum)

		# compute an AVERAGE g2 and g3 and place that on the average energy gap of all GBOMs. This is 
		# equivalent of averaging the spectral density over different instances of the GBOM and then just
		# computing a single, effective response function. 
		elif param_list.method=='CUMULANT_AV':
				print('Computing  response')
				# get list of adiabatic energies and dipole moms. 
				energy_dipole=np.zeros((1,1))
				if os.path.exists(param_set.E_opt_path):
						energy_dipole=np.genfromtxt(param_set.E_opt_path)
				else:
						sys.exit('Error: Requested an Eopt_avCUMULANT type calculation but did not provide optimized vertical energy gaps and dipoles')


				Eopt=energy_dipole[:,0]/const.Ha_to_eV
				# compute average energy
				Eopt_av=np.sum(Eopt)/(1.0*Eopt.shape[0])
				Eopt_fluct=Eopt-Eopt_av # fluctuation of Eopt energies around common mean. 
				print(Eopt_fluct)
				print('EOPT_Av')
				print(Eopt_av)
				average_Eadiab=0.0
				average_E00=0.0    # E00 and Eadiab are not the same. 
				for icount in range(GBOM_batch.num_gboms):
					average_Eadiab=average_Eadiab+GBOM_batch.gboms[icount].E_adiabatic
					average_E00=average_E00+GBOM_batch.gboms[icount].E_adiabatic+0.5*np.sum(GBOM_batch.gboms[icount].freqs_ex)-0.5*np.sum(GBOM_batch.gboms[icount].freqs_gs)
				average_Eadiab=average_Eadiab/(1.0*GBOM_batch.num_gboms)
				average_E00=average_E00/(1.0*GBOM_batch.num_gboms)
				average_Egap=0.0
				icount=0
				while icount<GBOM_batch.num_gboms:
						# Set E_00 to zero and calculate the lineshape function and energy gap. This 
						# guarantees that all cumulant spectra start at the same 0-0 transition
						# Then reset gboms.omega_av and recompute it for 0-0 transitions set to zero. 
						# Then compute lineshape function for that setup. This will generate a cumulant
						# spectrum with the 0-0 transition shifted to 0
						GBOM_batch.gboms[icount].E_adiabatic=0.0
						if param_set.exact_corr:
								average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
								GBOM_batch.gboms[icount].omega_av_qm=0.0
						else:
								average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl
								GBOM_batch.gboms[icount].omega_av_cl=0.0				

						icount=icount+1

				average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)

				
				delta_E_opt_E_adiab=Eopt_av-average_Eadiab   # The average E_adiab value should be unchanged in Eopt_avFTFC
				print('averageEgap, Eopt_av, average_Eadiab, Average E00')
				print(average_Egap,Eopt_av,average_Eadiab, average_E00)

				E_start=average_Egap-param_set.spectral_window/2.0
				E_end=average_Egap+param_set.spectral_window/2.0
				print('Estart, Eend')
				print(E_start,E_end)			

				# NOW overwrite E_adiabatic and dipole moment for all GBOMS. Make sure that all GBOM's have a consistent 0-0 transition equal to average_E00
				icount=0
				while icount<GBOM_batch.num_gboms:
						# convert from the constant E_00 to 
						GBOM_batch.gboms[icount].E_adiabatic=average_E00-0.5*np.sum(GBOM_batch.gboms[icount].freqs_ex)+0.5*np.sum(GBOM_batch.gboms[icount].freqs_gs)
						#GBOM_batch.gboms[icount].E_adiabatic=average_E00
						GBOM_batch.gboms[icount].dipole_mom=energy_dipole[icount,1]
						
						# recompute corrected average energy gap:
						if param_list.exact_corr:
							GBOM_batch.gboms[icount].calc_omega_av_qm(param_list.temperature,is_emission)
						else:
							GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)

						print('Adjusted GBOM energy and dipole mom and omega_av:', GBOM_batch.gboms[icount].E_adiabatic,GBOM_batch.gboms[icount].dipole_mom,GBOM_batch.gboms[icount].omega_av_qm)
						icount=icount+1

				# now compute average response function. Important: Average response function, NOT lineshape function
				average_response=np.zeros((param_list.num_steps,2),dtype=complex)
				icount=0
				while icount<GBOM_batch.num_gboms:
						if param_list.exact_corr:
								# spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
								print('OMEGA_AV_QM:',GBOM_batch.gboms[icount].omega_av_qm)
								GBOM_batch.gboms[icount].calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

								# only compute third order cumulant if needed
								if param_list.third_order:
										GBOM_batch.gboms[icount].calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)

						else:

								GBOM_batch.gboms[icount].calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

								if param_list.third_order:
										GBOM_batch.gboms[icount].calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term)
						# build average response function:
						for j in range(average_response.shape[0]):
							if param_list.exact_corr:
								average_response[j,0]=GBOM_batch.gboms[icount].g2_exact[j,0]
								if param_list.third_order:
									average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_exact[j,1]-GBOM_batch.gboms[icount].g3_exact[j,1])
								else:
									average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_exact[j,1])
							else:
								average_response[j,0]=GBOM_batch.gboms[icount].g2_cl[j,0]
								if param_list.third_order:
									average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_cl[j,1]-GBOM_batch.gboms[icount].g3_cl[j,1])
								else:
									average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_cl[j,1])	

						icount=icount+1
				# now average:
				average_response[:,1]=average_response[:,1]/(1.0*GBOM_batch.num_gboms)


				# now build spectrum.
				spectrum=np.zeros((average_response.shape[0],2))


				# TEST:
				temp_spectrum=linear_spectrum.full_spectrum(average_response,solvent.solvent_response,1.0,param_list.num_steps,0.08,0.13,True,is_emission)
				np.savetxt(param_list.GBOM_root+'_avcumulant_shape_func.dat', temp_spectrum)
				np.savetxt(param_list.GBOM_root+'_avcumulant_response_func_real.dat', average_response.real)
				# END test

				for icount in range(GBOM_batch.num_gboms):
					eff_response_func=average_response
					for jcount in range(eff_response_func.shape[0]):
							eff_response_func[jcount,1]=eff_response_func[jcount,1]*cmath.exp(1j*(Eopt_av-Eopt[icount])*eff_response_func[jcount,0]/math.pi)
					temp_spectrum=linear_spectrum.full_spectrum(eff_response_func,solvent.solvent_response,GBOM_batch.gboms[icount].dipole_mom,param_list.num_steps,E_start,E_end,True,is_emission)

					print(icount,(Eopt[icount]-Eopt_av),GBOM_batch.gboms[icount].dipole_mom)
					np.savetxt('Eopt_spec_snapshot'+str(icount)+'.dat',temp_spectrum)

					if icount==0:
						spectrum=temp_spectrum
					else:
						spectrum[:,0]=temp_spectrum[:,0]
						spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]

				spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)	

				np.savetxt(param_list.GBOM_root+'_Eopt_avcumulant_spectrum.dat', spectrum)
				print(min(Eopt_fluct),max(Eopt_fluct))

		else:
				sys.exit('Unknown method for GBOM_BATCH linear spectrum: '+param_list.method)


# same as GBOM_MD absorption, but this time we have a batch of GBOMs
def compute_hybrid_GBOM_batch_MD_absorption(param_list,MDtraj,GBOM_batch,is_emission):
                # now fix energy range
                E_start=MDtraj.mean-param_list.spectral_window/2.0
                E_end=MDtraj.mean+param_list.spectral_window/2.0

                # initialize GBOM:
                # if this is an emission calculation, need to reset some standard gbom parameters:
                if is_emission:
                                for i in range(param_list.num_gboms):
                                        GBOM_batch.gboms[i].set_emission_variables()

                if param_list.exact_corr:
                                for i in range(param_list.num_gboms):
                                        GBOM_batch.gboms[i].calc_omega_av_qm(param_list.temperature,is_emission)
                                        print('omega av QM')
                                        print(GBOM_batch.gboms[i].omega_av_qm)
                else:
                                for i in range(param_list.num_gboms):
                                        GBOM_batch.gboms[i].calc_omega_av_cl(param_list.temperature,is_emission)
                                        print('omega av cl')
                                        print(GBOM_batch.gboms[i].omega_av_cl)

                # Andres 2nd order cumulant GBOM approach assuming that the energy gap operator is
                # fully separable
                if param_list.method=='CUMUL_FC_SEPARABLE':
                                # first compute 2nd order cumulant response for MD trajectory
                                MDtraj.calc_2nd_order_corr()
                                MDtraj.calc_spectral_dens(param_list.temperature_MD)
                                np.savetxt(param_list.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
                                MDtraj.calc_g2(param_list.temperature,param_list.max_t,param_list.num_steps)
                                MDtraj.calc_cumulant_response(False,is_emission)


                                # Now compute 2nd order cumulant g2 for GBOM
                                if param_list.exact_corr:
                                                for i in range(param_list.num_gboms):
                                                        # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                                                        GBOM_batch.gboms[i].calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
                                                        #np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                        GBOM_batch.gboms[i].calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)


                                else:
                                                for i in range(param_list.num_gboms):
                                                        GBOM_batch.gboms[i].calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                                                        #np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                        GBOM_batch.gboms[i].calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

                                # calculate FC and 2nd order cumulant response functions for GBOM
                                for i in range(param_list.num_gboms):
                                                GBOM_batch.gboms[i].calc_cumulant_response(False,param_list.exact_corr,is_emission)
                                                GBOM_batch.gboms[i].calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
                                                #print('GBOM number '+str(i))
						#print(GBOM_batch.gboms[i].cumulant_response)

                                # now build effective response function. What about dipole moment? Where does it come from? MD or GBOM? If condon approx is valid
                                # it doesnt matter
                                eff_response=np.zeros((MDtraj.cumulant_response.shape[0],2),dtype=complex)
                                eff_response[:,0]=MDtraj.cumulant_response[:,0]
                                for j in range(param_list.num_gboms):
                                        for icount in range(eff_response.shape[0]):
                                                # protect against divide by 0
                                                if abs(GBOM_batch.gboms[j].cumulant_response[icount,1].real)>10e-10:
                                                        eff_response[icount,1]=eff_response[icount,1]+MDtraj.cumulant_response[icount,1]*GBOM_batch.gboms[j].fc_response[icount,1]/GBOM_batch.gboms[j].cumulant_response[icount,1]
                                                else: 
                                                        eff_response[icount,1]=eff_response[icount,1]+MDtraj.cumulant_response[icount,1]
                                eff_response[:,1]=eff_response[:,1]/(1.0*param_list.num_gboms)	

                                # now we can compute the linear spectrum based on eff_response
                                # no need for solvent model. This is taken care of in the MD trajectory
                                spectrum=linear_spectrum.full_spectrum(eff_response,np.zeros((1,1)),MDtraj.dipole_mom_av,param_list.num_steps,E_start,E_end,False,is_emission)
                                np.savetxt(param_list.GBOM_root+'_cumul_FC_separable_spectrum.dat', spectrum)

                else:
                                sys.exit('Error: Method '+param_list.method+' does not work with a mixed GBOM MD model.')


#  compute absorption spectra when chromophore model is given by both GBOM batch and MD batch
# this is mainly relevant for E-ZTFC and related methods defined by ANDRES
def compute_hybrid_GBOM_MD_absorption(param_list,MDtraj,GBOM_chromophore,is_emission):
                # now fix energy range
                E_start=MDtraj.mean-param_list.spectral_window/2.0
                E_end=MDtraj.mean+param_list.spectral_window/2.0

		# initialize GBOM:
		# if this is an emission calculation, need to reset some standard gbom parameters:
                if is_emission:
                                GBOM_chromophore.set_emission_variables()

                if param_list.exact_corr:
                                GBOM_chromophore.calc_omega_av_qm(param_list.temperature,is_emission)
                                print('omega av QM')
                                print(GBOM_chromophore.omega_av_qm)
                else:
                                GBOM_chromophore.calc_omega_av_cl(param_list.temperature,is_emission)
                                print('omega av cl')
                                print(GBOM_chromophore.omega_av_cl)

	
		# Andres 2nd order cumulant GBOM approach assuming that the energy gap operator is
		# fully separable
                if param_list.method=='CUMUL_FC_SEPARABLE':
				# first compute 2nd order cumulant response for MD trajectory
                                MDtraj.calc_2nd_order_corr()
                                MDtraj.calc_spectral_dens(param_list.temperature_MD)
                                np.savetxt(param_list.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
                                MDtraj.calc_g2(param_list.temperature,param_list.max_t,param_list.num_steps)
                                MDtraj.calc_cumulant_response(False,is_emission)


				# Now compute 2nd order cumulant g2 for GBOM
                                if param_list.exact_corr:
                                                # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                                                GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
                                                np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                GBOM_chromophore.calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
					
	
                                else:
                                                GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                                                np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

				# calculate FC and 2nd order cumulant response functions for GBOM
                                GBOM_chromophore.calc_cumulant_response(False,param_list.exact_corr,is_emission)
                                GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)	
				
				# now build effective response function
                                eff_response=GBOM_chromophore.fc_response
                                for icount in range(eff_response.shape[0]):
                                                eff_response[icount,1]=eff_response[icount,1]*MDtraj.cumulant_response[icount,1]/GBOM_chromophore.cumulant_response[icount,1]

				# now we can compute the linear spectrum based on eff_response
				# no need for solvent model. This is taken care of in the MD trajectory
                                spectrum=linear_spectrum.full_spectrum(eff_response,np.zeros((1,1)),MDtraj.dipole_mom_av,param_list.num_steps,E_start,E_end,False,is_emission)
                                np.savetxt(param_list.GBOM_root+'_cumul_FC_separable_spectrum.dat', spectrum)

                else:
                                sys.exit('Error: Method '+param_list.method+' does not work with a mixed GBOM MD model.')

# compute absorption spectrum from pure MD input
# Solvent degrees of freedom are optional here. They can be added to provide additional
# Broadening but in principle all broadening should originate from the MD. 
# Note that a pure MD trajectory can only be used to compute Ensemble or cumulant spectrum
def compute_MD_absorption(param_list,MDtraj,solvent,is_emission):
		# first check if we need a solvent model:
		if param_list.is_solvent:
				solvent.calc_spectral_dens(param_list.num_steps)
				solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t)
				solvent.calc_solvent_response(is_emission)
		# now fix energy range
		E_start=MDtraj.mean-param_list.spectral_window/2.0
		E_end=MDtraj.mean+param_list.spectral_window/2.0

		# now check if this is a cumulant or a classical ensemble calculation
		if param_list.method=='CUMULANT':
				MDtraj.calc_2nd_order_corr()
				MDtraj.calc_spectral_dens(param_list.temperature_MD)
				np.savetxt(param_list.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
				MDtraj.calc_g2(param_list.temperature,param_list.max_t,param_list.num_steps)
				if param_list.third_order:
						MDtraj.calc_3rd_order_corr(param_list.corr_length_3rd)
						# technically, in 3rd order cumulant, can have 2 different temperatures again. one at
						# which the MD was performed and one at wich the spectrum is simulated. Fix this...
						MDtraj.calc_g3(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.low_freq_cutoff)
						MDtraj.calc_cumulant_response(True,is_emission)
				else:
						MDtraj.calc_cumulant_response(False,is_emission)

				# compute linear spectrum
				if param_list.is_solvent:
						spectrum=linear_spectrum.full_spectrum(MDtraj.cumulant_response,solvent.solvent_response,MDtraj.dipole_mom_av,param_list.num_steps,E_start,E_end,True,is_emission)
				else:
						# set solvent response to a zero dummy vector
						spectrum=linear_spectrum.full_spectrum(MDtraj.cumulant_response,np.zeros((1,1)),MDtraj.dipole_mom_av,param_list.num_steps,E_start,E_end,False,is_emission)
				np.savetxt(param_list.MD_root+'MD_cumulant_spectrum.dat', spectrum)

		# now do ensemble approach
		elif param_list.method=='ENSEMBLE':
				MDtraj.calc_ensemble_response(param_list.max_t,param_list.num_steps)
				if param_list.is_solvent:
						spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,solvent.solvent_response,MDtraj.dipole_mom_av,param_list.num_steps,E_start,E_end,True,is_emission)
				else:
						# set solvent response to a zero dummy vector
						spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,np.zeros((1,1)),MDtraj.dipole_mom_av,param_list.num_steps,E_start,E_end,False,is_emission)				 
				np.savetxt(param_list.MD_root+'MD_ensemble_spectrum.dat', spectrum)
		else:
				sys.exit('Error: Method '+param_list.method+' does not work with a pure MD based model. Set Method to ENSEMBLE or CUMULANT.')

# main driver #
input_file=sys.argv[1]
if len(sys.argv)<3:
		num_cores=1
else:
		num_cores=int(sys.argv[2])

config.NUMBA_NUM_THREADS=num_cores

# parse input values
if os.path.exists(input_file):
		param_set=params.params(input_file)
else:
		sys.exit('Error: Could not find input file')

param_set.stdout.write('Successfully parsed the input file!'+'\n')
param_set.stdout.write('Now starting spectrum calculation'+'\n')

# set up solvent model:
if param_set.is_solvent:
		solvent_mod=solvent_model.solvent_model(param_set.solvent_reorg,param_set.solvent_cutoff_freq)
		param_set.stdout.write('Created solvent model!'+'\n')
		param_set.stdout.write('Solvent reorg:	  '+str(param_set.solvent_reorg)+' Ha'+'\n')
		param_set.stdout.write('Cutoff frequency:  '+str(param_set.solvent_cutoff_freq)+' Ha'+'\n')

# set up chromophore model
# pure GBOM model. 
if param_set.model=='GBOM' or param_set.model=='MD_GBOM':
		# sanity check:
		if param_set.num_modes==0:
				sys.exit('Error: Model GBOM requested but number of normal modes in the system is not set!')

		# single GBOM
		if param_set.num_gboms==1:
				# GBOM root is given. This means user requests reading params from Gaussian or Terachem
				if param_set.GBOM_root!='':				
						if param_set.GBOM_input_code=='GAUSSIAN':
								# build list of frozen atoms:
								freqs_gs=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+'_gs.log',param_set.num_modes,param_set.num_frozen_atoms)
								freqs_ex=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+'_ex.log',param_set.num_modes,param_set.num_frozen_atoms)
								K=gaussian_params.extract_Kmat(param_set.GBOM_root+'_vibronic.log',param_set.num_modes)
								J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
								if param_set.no_dusch:
										counter=0
										while counter<freqs_ex.shape[0]:
												J[counter,counter]=1.0
												counter=counter+1
								else:
										J=gaussian_params.extract_duschinsky_mat(param_set.GBOM_root+'_vibronic.log',param_set.num_modes)

								param_set.E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+'_vibronic.log')

								GBOM=gbom.gbom(freqs_gs,freqs_ex,J,K,param_set.E_adiabatic,param_set.dipole_mom)

						elif param_set.GBOM_input_code=='TERACHEM':
								# first obtain coordinates and Hessian. Check if we have frozen atoms.
								# sanity check:
								if param_set.num_atoms<1:
										sys.exit('Error: Trying to read from Terachem input but number of atoms is not set!') 
								frozen_atom_list=np.zeros(param_set.num_atoms)
								if param_set.num_frozen_atoms>0: 
										if os.path.exists(param_set.frozen_atom_path):
												frozen_atom_list=np.genfromtxt(param_set.frozen_atom_path)
										else:
												sys.exit('Error: Trying to perform Terachem calculation with frozen atoms but frozen atom list does not exist!')
								
								# now obtain Hessians and other params.
								masses,gs_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+'_gs.log', param_set.num_atoms)		
								gs_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+'_gs.log',frozen_atom_list,param_set.num_frozen_atoms)
								masses,ex_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+'_ex.log', param_set.num_atoms)
								ex_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+'_ex.log',frozen_atom_list,param_set.num_frozen_atoms)
								dipole_mom,E_adiabatic=terachem_params.get_e_adiabatic_dipole(param_set.GBOM_root+'_gs.log',param_set.GBOM_root+'_ex.log',param_set.target_excited_state)

								# now construct frequencies, J and K from these params. 
								freqs_gs,freqs_ex,J,K=hess_to_gbom.construct_freqs_J_K(gs_geom,ex_geom,gs_hessian,ex_hessian,masses,param_set.num_frozen_atoms,frozen_atom_list)

								# Check if we are artificially switching off the Duschinsky rotation
								if param_set.no_dusch:
										J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
										counter=0
										while counter<freqs_ex.shape[0]:
												J[counter,counter]=1.0
												counter=counter+1

								# GBOM assumes E_0_0 as input rather than E_adiabatic. 
								E_0_0=(E_adiabatic+0.5*(np.sum(freqs_ex)-np.sum(freqs_gs)))

								# construct GBOM
								GBOM=gbom.gbom(freqs_gs,freqs_ex,J,K,E_0_0,dipole_mom)

						# unsupported input code
						else:
								sys.exit('Error: Unrecognized code from which to read the GBOM parameters. Only GAUSSIAN and TERACHEM supported!')

				elif param_set.Jpath!='' and param_set.Kpath!='' and param_set.freq_gs_path!='' and param_set.freq_ex_path!='':
						if param_set.dipole_mom==0.0 and param_set.E_adiabatic==0.0:
								sys.exit('Error: Did not provide dipole moment or adiabatic energy gap for GBOM!')
						# create GBOM from input J, K and freqs
						else:
								freqs_gs=np.genfromtxt(param_set.freq_gs_path)
								J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
								if param_set.no_dusch:
										counter=0
										while counter<freqs_gs.shape[0]:
												J[counter,counter]=1.0
												counter=counter+1
								else:
										J=np.genfromtxt(param_set.Jpath)
								K=np.genfromtxt(param_set.Kpath)
								freqs_ex=np.genfromtxt(param_set.freq_ex_path)
								# created appropriate matrices: now create GBOM.
 
								GBOM=gbom.gbom(freqs_gs,freqs_ex,J,K,param_set.E_adiabatic,param_set.dipole_mom)
				else:
						sys.exit('Error: GBOM calculation requested but no path to model system parameters given!')

		# instead create a GBOM batch
		else:
				batch_count=1
				freqs_gs_batch=np.zeros((param_set.num_gboms,param_set.num_modes))
				freqs_ex_batch=np.zeros((param_set.num_gboms,param_set.num_modes))
				Jbatch=np.zeros((param_set.num_gboms,param_set.num_modes,param_set.num_modes))
				Kbatch=np.zeros((param_set.num_gboms,param_set.num_modes))
				E_batch=np.zeros((param_set.num_gboms))
				dipole_batch=np.zeros((param_set.num_gboms))

				while batch_count<param_set.num_gboms+1:
						if param_set.GBOM_root!='':
								if param_set.GBOM_input_code=='GAUSSIAN':
										print('SANITY CHECK: NUM FROZEN ATOMS:')
										print(param_set.num_frozen_atoms.shape[0])
										print(param_set.num_frozen_atoms)
										freqs_gs=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+str(batch_count)+'_gs.log',param_set.num_modes,param_set.num_frozen_atoms[batch_count-1])
										freqs_ex=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+str(batch_count)+'_ex.log',param_set.num_modes,param_set.num_frozen_atoms[batch_count-1])
										K=gaussian_params.extract_Kmat(param_set.GBOM_root+str(batch_count)+'_vibronic.log',param_set.num_modes)
										J=gaussian_params.extract_duschinsky_mat(param_set.GBOM_root+str(batch_count)+'_vibronic.log',param_set.num_modes)
										E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+str(batch_count)+'_vibronic.log')
										# are we switching off Duschinsky rotation?
										if param_set.no_dusch:
												J=np.zeros((K.shape[0],K.shape[0]))
												counter=0
												while counter<freqs_gs.shape[0]:
														J[counter,counter]=1.0
														counter=counter+1
										# fill batch
										freqs_gs_batch[batch_count-1,:]=freqs_gs
										freqs_ex_batch[batch_count-1,:]=freqs_ex
										Jbatch[batch_count-1,:,:]=J
										Kbatch[batch_count-1,:]=K
										E_batch[batch_count-1]=E_adiabatic
										dipole_batch[batch_count-1]=param_set.dipole_mom
								elif param_set.GBOM_input_code=='TERACHEM':
										atoms_snapshot=0
										frozen_atoms_snapshot=0
										if param_set.num_frozen_atoms>0: 
												if os.path.exists(param_set.frozen_atom_path+str(batch_count)):
														frozen_atom_list=np.genfromtxt(param_set.frozen_atom_path)
												else:
														sys.exit('Error: Trying to perform Terachem calculation with frozen atoms but frozen atom list does not exist for current batch!')
												atoms_snapshot=frozen_atom_list.shape[0]
												frozen_atoms_snapshot=int(np.sum(frozen_atom_list))
										# NOT a frozen atom calculation
										else:
												# sanity check
												if param_set.num_atoms<1:
														sys.exit('Error: Trying to read a batch of Terachem input files with no frozen atoms and NUM_ATOMS is not set!')
												atoms_snapshot=param_set.num_atoms

										# now obtain Hessians and other params.
										masses,gs_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+str(batch_count)+'_gs.log', num_atoms_snapshot)
										gs_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+str(batch_count)+'_gs.log',frozen_atom_list,frozen_atoms_snapshot)
										masses,ex_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+str(batch_count)+'_ex.log', num_atoms_snapshot)
										ex_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+str(batch_count)+'_ex.log',frozen_atom_list,frozen_atoms_snapshot)
										dipole_mom,E_adiabatic=terachem_params.get_e_adiabatic_dipole(param_set.GBOM_root+str(batch_count)+'_gs.log',param_set.GBOM_root+str(batch_count)+'_ex.log',param_set.target_excited_state)

										# now construct frequencies, J and K from these params. 
										freqs_gs,freqs_ex,J,K=hess_to_gbom.construct_freqs_J_K(gs_geom,ex_geom,gs_hessian,ex_hessian,masses,frozen_atoms_snapshot,frozen_atom_list)

										# Check if we are artificially switching off the Duschinsky rotation
										if param_set.no_dusch:
												J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
												counter=0
												while counter<freqs_ex.shape[0]:
														J[counter,counter]=1.0
														counter=counter+1

										# GBOM assumes E_0_0 as input rather than E_adiabatic. 
										E_0_0=(E_adiabatic+0.5*(np.sum(freqs_ex)-np.sum(freqs_gs)))

										# fill batch
										freqs_gs_batch[batch_count-1,:]=freqs_gs
										freqs_ex_batch[batch_count-1,:]=freqs_ex
										Jbatch[batch_count-1,:,:]=J
										Kbatch[batch_count-1,:]=K
										E_batch[batch_count-1]=E_0_0
										dipole_batch[batch_count-1]=dipole_mom
								else:
										sys.exit('Error: Currently only support GBOM_INPUT_CODE=GAUSSIAN or TERACHEM!')

						elif param_set.Jpath!='' and param_set.Kpath!='' and param_set.freq_gs_path!='' and param_set.freq_ex_path!='':
								if param_set.dipole_mom_path=='' and param_set.E_adiabatic_path=='' and not param_set.method=='EOPT_AV':
										sys.exit('Error: Did not provide dipole moment list or adiabatic energy gap list for GBOM batch!')
								# create GBOM from input J, K and freqs
								else:
										J=np.genfromtxt(param_set.Jpath+str(batch_count)+'.dat')
										K=np.genfromtxt(param_set.Kpath+str(batch_count)+'.dat')
										freqs_gs=np.genfromtxt(param_set.freq_gs_path+str(batch_count)+'.dat')
										freqs_ex=np.genfromtxt(param_set.freq_ex_path+str(batch_count)+'.dat')
										if not param_set.method=='EOPT_AV':
												dipole_list=np.genfromtxt(param_set.dipole_mom_path)
												E_adiabatic_list=np.genfromtxt(param_set.E_adiabatic_path)
		
										# fill batch
										freqs_gs_batch[batch_count-1,:]=freqs_gs
										freqs_ex_batch[batch_count-1,:]=freqs_ex
										Jbatch[batch_count-1,:,:]=J
										Kbatch[batch_count-1,:]=K
										if not param_set.method=='EOPT_AV':
												E_batch[batch_count-1]=E_adiabatic_list[batch_count-1]
												dipole_batch[batch_count-1]=dipole_list[batch_count-1]
										else:	# if this is an EOPT_AV calculation we get the adiabatic energy gap from somewhere else
												E_batch[batch_count-1]=0.0
												dipole_batch[batch_count-1]=1.0
								
						else:
								sys.exit('Error: GBOM calculation requested but no path to model system parameters given!')


						batch_count=batch_count+1
				# now construct the model
				GBOM_batch=gbom.gbom_list(freqs_gs_batch,freqs_ex_batch,Jbatch,Kbatch,E_batch,dipole_batch,param_set.num_gboms)
		param_set.stdout.write('Successfully set up a GBOM model!'+'\n')

# pure MD model
elif param_set.model=='MD':
		traj_count=1
		while traj_count<param_set.num_trajs+1:
				if os.path.exists(param_set.MD_root+'traj'+str(traj_count)+'.dat'):
						param_set.stdout.write('Reading in MD trajectory '+str(traj_count)+'!'+'\n')
						traj_dipole=np.genfromtxt(param_set.MD_root+'traj'+str(traj_count)+'.dat')
						if traj_count==1:
								traj_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))
								osc_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))

						traj_batch[:,traj_count-1]=traj_dipole[:,0]
						osc_batch[:,traj_count-1]=traj_dipole[:,1]
				
				else:
						sys.exit('Error: Trajectory file name necessary for MD-based model does not exist!')

				traj_count=traj_count+1
		MDtraj=md_traj.MDtrajs(traj_batch,osc_batch,param_set.decay_length,param_set.num_trajs,param_set.md_step)		
		param_set.stdout.write('Successfully read in MD trajectory data of energy gap fluctuations!'+'\n')		

else:
                sys.exit('Error: Invalid model '+param_set.model)

# both MD and GBOM input ---> E-ZTFC and related approaches 
# need to also construct the MDtraj model. Have already constructed the GBOM type model
if param_set.model=='MD_GBOM':
		# start by setting up MDtraj
                traj_count=1
                while traj_count<param_set.num_trajs+1:
                                if os.path.exists(param_set.MD_root+'traj'+str(traj_count)+'.dat'):
                                                param_set.stdout.write('Reading in MD trajectory '+str(traj_count)+'!'+'\n')
                                                traj_dipole=np.genfromtxt(param_set.MD_root+'traj'+str(traj_count)+'.dat')
                                                if traj_count==1:
                                                                traj_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))
                                                                osc_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))

                                                traj_batch[:,traj_count-1]=traj_dipole[:,0]
                                                osc_batch[:,traj_count-1]=traj_dipole[:,1]

                                else:
                                                sys.exit('Error: Trajectory file name necessary for MD-based model does not exist!')

                                traj_count=traj_count+1
                MDtraj=md_traj.MDtrajs(traj_batch,osc_batch,param_set.decay_length,param_set.num_trajs,param_set.md_step)
                param_set.stdout.write('Successfully read in MD trajectory data of energy gap fluctuations!'+'\n')

# first check whether this is an absorption or a 2DES calculation
if param_set.task=='ABSORPTION':
		print('Setting up linear absorption spectrum calculation:')
		if param_set.model=='GBOM':
				if param_set.num_gboms==1:
						if param_set.is_solvent:		
								compute_GBOM_absorption(param_set,GBOM,solvent_mod, False)
						else:
								sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')
				else:
						if param_set.is_solvent:
								compute_GBOM_batch_absorption(param_set,GBOM_batch,solvent_mod, False)
						else:
								sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')

		elif param_set.model=='MD':
				if param_set.is_solvent:
						compute_MD_absorption(param_set,MDtraj,solvent_mod,False)
				else:
						# set solvent model to dummy variable

						compute_MD_absorption(param_set,MDtraj,0.0,False)
		elif param_set.model=='MD_GBOM':
				if param_set.num_gboms==1:
						compute_hybrid_GBOM_MD_absorption(param_set,MDtraj,GBOM,False)
				else:
						compute_hybrid_GBOM_batch_MD_absorption(param_set,MDtraj,GBOM_batch,False)	

		else:
				sys.exit('Error: Only pure GBOM model or pure MD model or MD_GBOM model implemented so far.')

elif param_set.task=='EMISSION':
		print('Setting up linear emission spectrum calculation:')
		if param_set.model=='GBOM':
				if param_set.is_solvent:
						compute_GBOM_absorption(param_set,GBOM,solvent_mod,True)
				else:
						sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')
		elif param_set.model=='MD':
				if param_set.is_solvent:
						compute_MD_absorption(param_set,MDtraj,solvent_mod,True)
				else:
						# set solvent model to dummy variable
						compute_MD_absorption(param_set,MDtraj,0.0,True)
		else:
				sys.exit('Error: Only pure GBOM model or pure MD model implemented so far.')

elif param_set.task=='2DES':
		print('Setting up 2DES calculation:')
		if param_set.model=='GBOM' and param_set.num_gboms==1:
				if not param_set.is_solvent:
						sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')

				solvent_mod.calc_spectral_dens(param_set.num_steps)
				solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t)
				filename_2DES=param_set.GBOM_root+''
				if param_set.exact_corr:
						GBOM.calc_omega_av_qm(param_set.temperature,False)
						GBOM.calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False)
						# set the start and end values for both the x and the y axis of the
						# 2DES spectrum
						eff_shift1=0.0
						eff_shift2=0.0
						if abs(param_set.omega1)>0.000001:
								eff_shift1=param_set.omega1-GBOM.omega_av_qm
						if abs(param_set.omega3)>0.000001:
								eff_shift2=param_set.omega3-GBOM.omega_av_qm

						E_start1=GBOM.omega_av_qm-param_set.spectral_window/2.0+eff_shift1
						E_end1=GBOM.omega_av_qm+param_set.spectral_window/2.0+eff_shift1
						E_start2=GBOM.omega_av_qm-param_set.spectral_window/2.0+eff_shift2
						E_end2=GBOM.omega_av_qm+param_set.spectral_window/2.0+eff_shift2

						q_func_eff=GBOM.g2_exact
						q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]


						print('Omega_Sq parameter')
						print(GBOM.Omega_sq)

						# if it is a 3rd order cumulant calculation, compute g3 and auxilliary functions h1 and h2
						if param_set.third_order:
								GBOM.calc_g3_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.four_phonon_term)

								if os.path.exists('h1_real.dat') and os.path.exists('h1_imag.dat') and os.path.exists('h2_real.dat') and os.path.exists('h2_imag.dat') and os.path.exists('h4_real.dat') and os.path.exists('h4_imag.dat') and os.path.exists('h5_real.dat') and os.path.exists('h5_imag.dat'):
										# read in all files:
										GBOM.h1_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
										GBOM.h1_exact=GBOM.h1_exact+twoDES.read_2D_spectrum('h1_real.dat',param_set.num_steps)
										temp_imag=1j*twoDES.read_2D_spectrum('h1_imag.dat',param_set.num_steps)
										GBOM.h1_exact[:,:,2]=GBOM.h1_exact[:,:,2]+temp_imag[:,:,2]

										GBOM.h2_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
										GBOM.h2_exact=GBOM.h2_exact+twoDES.read_2D_spectrum('h2_real.dat',param_set.num_steps)
										temp_imag=1j*twoDES.read_2D_spectrum('h2_imag.dat',param_set.num_steps)
										GBOM.h2_exact[:,:,2]=GBOM.h2_exact[:,:,2]+temp_imag[:,:,2]

										GBOM.h4_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
										GBOM.h4_exact=GBOM.h4_exact+twoDES.read_2D_spectrum('h4_real.dat',param_set.num_steps)
										temp_imag=1j*twoDES.read_2D_spectrum('h4_imag.dat',param_set.num_steps)
										GBOM.h4_exact[:,:,2]=GBOM.h4_exact[:,:,2]+temp_imag[:,:,2]

										GBOM.h5_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
										GBOM.h5_exact=GBOM.h5_exact+twoDES.read_2D_spectrum('h5_real.dat',param_set.num_steps)
										temp_imag=1j*twoDES.read_2D_spectrum('h5_imag.dat',param_set.num_steps)
										GBOM.h5_exact[:,:,2]=GBOM.h5_exact[:,:,2]+temp_imag[:,:,2]

								else:
										GBOM.calc_h1_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h1_real.dat',GBOM.h1_exact,False)
										twoDES.print_2D_spectrum('h1_imag.dat',GBOM.h1_exact,True)
										GBOM.calc_h2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h2_real.dat',GBOM.h2_exact,False)
										twoDES.print_2D_spectrum('h2_imag.dat',GBOM.h2_exact,True)
										GBOM.calc_h4_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h4_real.dat',GBOM.h4_exact,False)
										twoDES.print_2D_spectrum('h4_imag.dat',GBOM.h4_exact,True)
										GBOM.calc_h5_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h5_real.dat',GBOM.h5_exact,False)
										twoDES.print_2D_spectrum('h5_imag.dat',GBOM.h5_exact,True)

								# now construct 3rd order correlation function. Needed to speed up evaluation of h3
								#GBOM.compute_corr_func_3rd(param_set.temperature*const.kb_in_Ha,param_set.num_steps,param_set.max_t,True)
								#twoDES.print_2D_spectrum('corr_func_3rd_real.dat',GBOM.corr_func_3rd_qm,False)
								#twoDES.print_2D_spectrum('corr_func_3rd_imag.dat',GBOM.corr_func_3rd_qm,True)
						print('NUMBA environment variable:')
						print(config.NUMBA_NUM_THREADS)

						if param_set.method_2DES=='2DES':
								if param_set.third_order:
										twoDES.calc_2DES_time_series_GBOM_3rd(q_func_eff,GBOM.g3_exact,GBOM.h1_exact,GBOM.h2_exact,GBOM.h4_exact,GBOM.h5_exact,GBOM.corr_func_3rd_qm,GBOM.freqs_gs,GBOM.Omega_sq,GBOM.gamma,param_set.temperature*const.kb_in_Ha,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0,False,param_set.no_dusch,param_set.four_phonon_term)
								else:
										twoDES.calc_2DES_time_series(q_func_eff,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)
						elif param_set.method_2DES=='PUMP_PROBE':
								twoDES.calc_pump_probe_time_series(q_func_eff,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)
				else:
						GBOM.calc_omega_av_cl(param_set.temperature,False)
						GBOM.calc_g2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False)
						# set the start and end values for both the x and the y axis of the
						# 2DES spectrum
						eff_shift1=0.0
						eff_shift2=0.0
						if abs(param_set.omega1)>0.000001:
								eff_shift1=param_set.omega1-GBOM.omega_av_cl
						if abs(param_set.omega3)>0.000001:
								eff_shift2=param_set.omega3-GBOM.omega_av_cl

						E_start1=GBOM.omega_av_cl-param_set.spectral_window/2.0+eff_shift1
						E_end1=GBOM.omega_av_cl+param_set.spectral_window/2.0+eff_shift1
						E_start2=GBOM.omega_av_cl-param_set.spectral_window/2.0+eff_shift2
						E_end2=GBOM.omega_av_cl+param_set.spectral_window/2.0+eff_shift2
						q_func_eff=GBOM.g2_cl
						q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]

						# if it is a 3rd order cumulant calculation, compute g3 and auxilliary functions h1 and h2
						if param_set.third_order:
								GBOM.calc_g3_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.four_phonon_term)
								if os.path.exists('h1_real.dat') and os.path.exists('h1_imag.dat') and os.path.exists('h2_real.dat') and os.path.exists('h2_imag.dat') and os.path.exists('h4_real.dat') and os.path.exists('h4_imag.dat') and os.path.exists('h5_real.dat') and os.path.exists('h5_imag.dat'):
										# read in all files:
										GBOM.h1_cl=twoDES.read_2D_spectrum('h1_real.dat',param_set.num_steps)
										GBOM.h1_cl[:,:,2]=GBOM.h1_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h1_imag.dat',param_set.num_steps))[:,:,2]
										GBOM.h2_cl=twoDES.read_2D_spectrum('h2_real.dat',param_set.num_steps)
										GBOM.h2_cl[:,:,2]=GBOM.h2_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h2_imag.dat',param_set.num_steps))[:,:,2]

										GBOM.h4_cl=twoDES.read_2D_spectrum('h4_real.dat',param_set.num_steps)
										GBOM.h4_cl[:,:,2]=GBOM.h4_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h4_imag.dat',param_set.num_steps))[:,:,2]
										GBOM.h5_cl=twoDES.read_2D_spectrum('h5_real.dat',param_set.num_steps)
										GBOM.h5_cl[:,:,2]=GBOM.h5_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h5_imag.dat',param_set.num_steps))[:,:,2]

								else:
										# Calc h and save to file
										GBOM.calc_h1_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h1_real.dat',(GBOM.h1_cl),False)
										twoDES.print_2D_spectrum('h1_imag.dat',GBOM.h1_cl,True)
										GBOM.calc_h2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h2_real.dat',(GBOM.h2_cl),False)
										twoDES.print_2D_spectrum('h2_imag.dat',GBOM.h2_cl,True)
										GBOM.calc_h4_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h4_real.dat',(GBOM.h4_cl),False)
										twoDES.print_2D_spectrum('h4_imag.dat',GBOM.h4_cl,True)
										GBOM.calc_h5_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
										twoDES.print_2D_spectrum('h5_real.dat',(GBOM.h5_cl),False)
										twoDES.print_2D_spectrum('h5_imag.dat',GBOM.h5_cl,True,)
		
								# now construct 3rd order correlation function. Needed to speed up evaluation of h3
								#GBOM.compute_corr_func_3rd(param_set.temperature*const.kb_in_Ha,param_set.num_steps,param_set.max_t,False)
		
						if param_set.method_2DES=='2DES':
								if param_set.third_order:
										twoDES.calc_2DES_time_series_GBOM_3rd(q_func_eff,GBOM.g3_cl,GBOM.h1_cl,GBOM.h2_cl,GBOM.h4_cl,GBOM.h5_cl,GBOM.corr_func_3rd_cl,GBOM.freqs_gs,GBOM.Omega_sq,GBOM.gamma,param_set.temperature*const.kb_in_Ha,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0,True,param_set.no_dusch,param_set.four_phonon_term)

								else:
										twoDES.calc_2DES_time_series(q_func_eff,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)		
						elif param_set.method_2DES=='PUMP_PROBE':
								twoDES.calc_pump_probe_time_series(q_func_eff,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)

		# GBOM batch. Simplified implementation for the time being. Only 2nd order cumulant, and only standard 2DES
		elif param_set.model=='GBOM' and param_set.num_gboms!=1:

				if param_set.method=='EOPT_AV':		# this is not an E_FTFC calculation but rather an Eopt_avFTFC calculation
						filename_2DES=param_set.GBOM_root+''
						solvent_mod.calc_spectral_dens(param_set.num_steps)
						solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t)
						solvent_mod.calc_solvent_response(False)
						
						# get list of adiabatic energies and dipole moms. 
						energy_dipole=np.zeros((1,1))
						if os.path.exists(param_set.E_opt_path):
								energy_dipole=np.genfromtxt(param_set.E_opt_path)
						else:
								sys.exit('Error: Requested an Eopt_avFTFC type calculation but did not provide optimized vertical energy gaps and dipoles')


						Eopt=energy_dipole[:,0]/const.Ha_to_eV
						# compute average energy
						Eopt_av=np.sum(Eopt)/(1.0*Eopt.shape[0]) 

						icount=0
						average_Egap=0.0
						average_E00=0.0
						
						while icount<GBOM_batch.num_gboms:
						# figure out start and end value for the spectrum.
								average_E00=average_E00+GBOM_batch.gboms[icount].E_adiabatic+0.5*np.sum(GBOM_batch.gboms[icount].freqs_ex)-0.5*np.sum(GBOM_batch.gboms[icount].freqs_gs)
								if param_set.exact_corr:
										# compute energy gap without 0-0 transition
										GBOM_batch.gboms[icount].calc_omega_av_qm(param_set.temperature,False)
										average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
										GBOM_batch.gboms[icount].omega_av_qm=0.0
								else:
										GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
										average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl
										GBOM_batch.gboms[icount].omega_av_cl=0.0
								icount=icount+1

						# figure out average Egap and gap between Eopt_av and average_Egap for the GBOMS
						average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)  # this is the place the spectrum should be centered on
						delta_Eopt_Eav=average_Egap-Eopt_av
						average_E00=average_E00/(1.0*GBOM_batch.num_gboms)
						average_E_adiabatic=0.0
						for i in range(len(GBOM_batch.gboms)):
							average_E_adiabatic=average_E_adiabatic+GBOM_batch.gboms[i].E_adiabatic
						print('AVERAGE E ADIABATIC')	
						average_E_adiabatic=average_E_adiabatic/(1.0*Eopt.shape[0])		
						print(average_E_adiabatic)
						print('AVERAGE E00')
						print(average_E00)
						delta_Eadiab_Eopt_av=average_E_adiabatic-Eopt_av
						Eopt_fluct=Eopt-Eopt_av

						# now set all Eadiabatic to the average Eadiabatic and recompute the g2 function.
						for icount in range(GBOM_batch.num_gboms):
								if param_set.exact_corr:
                                                                                # compute energy gap without 0-0 transition
										GBOM_batch.gboms[icount].E_adiabatic=average_E_adiabatic
										GBOM_batch.gboms[icount].calc_omega_av_qm(param_set.temperature,False)
										GBOM_batch.gboms[icount].calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False)
								else:
										GBOM_batch.gboms[icount].E_adiabatic=average_E_adiabatic
										GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
										GBOM_batch.gboms[icount].calc_g2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False)

						#HACK
						E_start=Eopt_av+average_Egap-param_set.spectral_window/2.0
						E_end=Eopt_av+average_Egap+param_set.spectral_window/2.0
						print(E_start,E_end)
						# now construct list of g functions with the corrected energy shift taken from Eopt
						q_func_eff_batch = []
						icount=0
						while icount<Eopt.shape[0]:
								if param_set.exact_corr:
									g2_temp=GBOM_batch.gboms[icount].g2_exact
								else:
									g2_temp=GBOM_batch.gboms[icount].g2_cl
								tcount=0
								print(Eopt[icount],average_Egap,Eopt_av)
								while tcount<g2_temp.shape[0]:
										g2_temp[tcount,1]=g2_temp[tcount,1]
										tcount=tcount+1
								g2_temp[:,1]=g2_temp[:,1]+solvent_mod.g2_solvent[:,1]
								q_func_eff_batch.append(g2_temp)
								icount=icount+1

						Eopt_const=np.zeros(Eopt.shape[0])
						Eopt_const[:]=Eopt_const[:]+Eopt_av

						# created batch of g functions that are all the same, apart from different Eopt shifts
						# now construct 2DES spectra. 
						print(Eopt)
						twoDES.calc_2DES_time_series_batch_Eopt_av(q_func_eff_batch,Eopt.shape[0],E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,Eopt)
				else:

						filename_2DES=param_set.GBOM_root+''
						solvent_mod.calc_spectral_dens(param_set.num_steps)
						solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t)
						solvent_mod.calc_solvent_response(False)

						# Now make sure that we have only a single average spectral window for the GBOM batch. 
						# also set the correct omega_av. Also, use this opportunity to compute g2 for each GBOM 
						icount=0
						average_Egap=0.0
						while icount<GBOM_batch.num_gboms:
						# figure out start and end value for the spectrum.
								if param_set.exact_corr:
										GBOM_batch.gboms[icount].calc_omega_av_qm(param_set.temperature,False)
										GBOM_batch.gboms[icount].calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False)
										average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
								else:
										GBOM_batch.gboms[icount].calc_g2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False)
										GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
										average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl

								icount=icount+1

						average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)

						E_start=average_Egap-param_set.spectral_window/2.0
						E_end=average_Egap+param_set.spectral_window/2.0

						print(average_Egap,E_start,E_end)
						# create a list of effective q functions
						q_func_eff_batch = []
						icount=0
						while icount<GBOM_batch.num_gboms:
								if param_set.exact_corr:
										q_func_eff=GBOM_batch.gboms[icount].g2_exact
										q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]
								else:
										q_func_eff=GBOM_batch.gboms[icount].g2_cl
										q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]
								q_func_eff_batch.append(q_func_eff)
								icount=icount+1


						# Successfully set up set of GBOMs ready for 2DES calculation.
						twoDES.calc_2DES_time_series_batch(q_func_eff_batch,GBOM_batch.num_gboms,E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)


		elif param_set.model=='MD':
				filename_2DES=param_set.MD_root+''
				# first check if we have a solvent model
				if param_set.is_solvent:
						solvent_mod.calc_spectral_dens(param_set.num_steps)
						solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t)
				# then set up g2 for MDtraj
				MDtraj.calc_2nd_order_corr()
				MDtraj.calc_spectral_dens(param_set.temperature_MD)
				np.savetxt(param_set.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
				MDtraj.calc_g2(param_set.temperature,param_set.max_t,param_set.num_steps)
				# 3rd order cumulant calculation? Then compute g_3, as well as the 3rd order quantum correlation function
				if param_set.third_order:
						MDtraj.calc_3rd_order_corr(param_set.corr_length_3rd)
						# technically, in 3rd order cumulant, can have 2 different temperatures again. one at
						# which the MD was performed and one at wich the spectrum is simulated. Fix this...
						MDtraj.calc_g3(param_set.temperature,param_set.max_t,param_set.num_steps,param_set.low_freq_cutoff)
						MDtraj.calc_corr_func_3rd_qm_freq(param_set.temperature_MD,param_set.low_freq_cutoff)
						# Check if h1 and h2 are already computed and stored. computational savings...
						if os.path.exists('h1_real.dat') and os.path.exists('h1_imag.dat') and os.path.exists('h2_real.dat') and os.path.exists('h2_imag.dat') and os.path.exists('h4_real.dat') and os.path.exists('h4_imag.dat') and os.path.exists('h5_real.dat') and os.path.exists('h5_imag.dat'):

								# read in all files:
								MDtraj.h1=twoDES.read_2D_spectrum('h1_real.dat',param_set.num_steps)
								MDtraj.h1[:,:,2]=MDtraj.h1[:,:,2]+1j*(twoDES.read_2D_spectrum('h1_imag.dat',param_set.num_steps))[:,:,2]		
								MDtraj.h2=twoDES.read_2D_spectrum('h2_real.dat',param_set.num_steps)
								MDtraj.h2[:,:,2]=MDtraj.h2[:,:,2]+1j*(twoDES.read_2D_spectrum('h2_imag.dat',param_set.num_steps))[:,:,2]		

								MDtraj.h4=twoDES.read_2D_spectrum('h4_real.dat',param_set.num_steps)
								MDtraj.h4[:,:,2]=MDtraj.h4[:,:,2]+1j*(twoDES.read_2D_spectrum('h4_imag.dat',param_set.num_steps))[:,:,2]
								MDtraj.h5=twoDES.read_2D_spectrum('h5_real.dat',param_set.num_steps)
								MDtraj.h5[:,:,2]=MDtraj.h5[:,:,2]+1j*(twoDES.read_2D_spectrum('h5_imag.dat',param_set.num_steps))[:,:,2]

						else:
								MDtraj.calc_h1(param_set.max_t,param_set.num_steps)
								twoDES.print_2D_spectrum('h1_real.dat',(MDtraj.h1),False)
								twoDES.print_2D_spectrum('h1_imag.dat',MDtraj.h1,True)
								MDtraj.calc_h2(param_set.max_t,param_set.num_steps)
								twoDES.print_2D_spectrum('h2_real.dat',(MDtraj.h2),False)
								twoDES.print_2D_spectrum('h2_imag.dat',MDtraj.h2,True)

								MDtraj.calc_h4(param_set.max_t,param_set.num_steps)
								twoDES.print_2D_spectrum('h4_real.dat',(MDtraj.h4),False)
								twoDES.print_2D_spectrum('h4_imag.dat',MDtraj.h4,True)
								MDtraj.calc_h5(param_set.max_t,param_set.num_steps)
								twoDES.print_2D_spectrum('h5_real.dat',(MDtraj.h5),False)
								twoDES.print_2D_spectrum('h5_imag.dat',MDtraj.h5,True)


				# set the start and end values for both the x and the y axis of the
				# 2DES spectrum
				eff_shift1=0.0
				eff_shift2=0.0
				if abs(param_set.omega1)>0.000001:
						eff_shift1=param_set.omega1-MDtraj.mean
				if abs(param_set.omega3)>0.000001:
						eff_shift2=param_set.omega3-MDtraj.mean
				print(param_set.omega1,param_set.omega3)

				E_start1=MDtraj.mean-param_set.spectral_window/2.0+eff_shift1
				E_end1=MDtraj.mean+param_set.spectral_window/2.0+eff_shift1		
				E_start2=MDtraj.mean-param_set.spectral_window/2.0+eff_shift2
				E_end2=MDtraj.mean+param_set.spectral_window/2.0+eff_shift2
				
				print(E_start1*27.2114, E_end1*27.2114, E_start2*27.2114, E_end2*27.2114)

				q_func_eff=MDtraj.g2
				if param_set.is_solvent:
						q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]
				# now compute 2DES in 2nd order cumulant approach
				if param_set.method_2DES=='2DES':
						print('Starting 2DES time series calc:')
						# Check if this is a 3rd order cumulant calculation
						if param_set.third_order:
								twoDES.calc_2DES_time_series_3rd(q_func_eff,MDtraj.g3,MDtraj.h1,MDtraj.h2,MDtraj.h4,MDtraj.h5,MDtraj.corr_func_3rd_qm_freq,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,MDtraj.mean)
						else:
								twoDES.calc_2DES_time_series(q_func_eff,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,MDtraj.mean)
				elif param_set.method_2DES=='PUMP_PROBE':
						twoDES.calc_pump_probe_time_series(q_func_eff,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,MDtraj.mean)
				else:
						sys.exit('Error: Invalid nonlinear spectroscopy method '+param_set.method_2DES)

else:
		sys.exit('Error: Invalid task '+param_set.task)
