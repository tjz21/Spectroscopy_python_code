#/usr/bin/env python

from scipy import integrate
import sys
import os
import numpy as np
import math
from numba import config
import spec_pkg.constants.constants as const
from spec_pkg.GBOM import gbom
from spec_pkg.GBOM import extract_model_params_gaussian as gaussian_params
from spec_pkg.linear_spectrum import linear_spectrum
from spec_pkg.nonlinear_spectrum import twoDES
from spec_pkg.solvent_model import solvent_model
from spec_pkg.cumulant import md_traj
from spec_pkg.params import params

# TODO ##########################################################################
# 1) Fix MD ensemble spectrum (currently gives negative absorption)             #
# 2) Implement emission spectra calculations, both in the GBOM approach and for #
#    MD trajectories								#
# 3) Write simulation information to stdout (what calculation is done, how it   #
#    progresses etc.                                                            #
# 4) Make sure we can input shapes of pulses for the ultrafast spectroscopy,   	#
#    rather than just delta-function						#
# 5) Implement GBOM batch absorption calculation                                #
# 6) Implement combined GBOM_MD model needed for E-ZTFC and similar appraoches  #
# 7) Use Ajay's code to establish the functionality to read in Terachem files   #
# 8) Implement a PIMD version of all the methods                                #
# 9) Use FFTs to generate final spectrum 					#
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
		spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)	
		if param_list.qm_wigner_dist:
			np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_qm_wigner_dist.dat', spectrum)
		else:
			np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_boltzmann_dist.dat', spectrum)
	elif param_list.method=='FC':
		GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)

		spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.fc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
		np.savetxt(param_list.GBOM_root+'_FC_spectrum.dat', spectrum)
	elif param_list.method=='EZTFC':
		GBOM_chromophore.calc_eztfc_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)
		spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.eztfc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
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
				GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
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
				GBOM_chromophore.calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
				# sace lineshape function
				#temp_lineshape=(GBOM_chromophore.g2_cl).real
				#temp_lineshape[:,1]=temp_lineshape[:,1]+(GBOM_chromophore.g3_cl[:,1]).real
				#np.savetxt(param_list.GBOM_root+'_lineshape_func_3rd_order_harmonic_qcf.dat',temp_lineshape)


		GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission)	
		spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
		if param_list.exact_corr:
			np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_exact_corr.dat', spectrum)
		else:
			np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_harmonic_qfc.dat', spectrum)
		
	# do all approaches, including qm wigner sampling and exact and approximate 
	# quantum correlation functions for the cumulant approach
	elif param_list.method=='ALL':
                GBOM_chromophore.calc_ensemble_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
                if param_list.qm_wigner_dist:
                        np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_qm_wigner_dist.dat', spectrum)
                else:
                        np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_boltzmann_dist.dat', spectrum)

                GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t, is_emission)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.fc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
                np.savetxt(param_list.GBOM_root+'_FC_spectrum.dat', spectrum)
                GBOM_chromophore.calc_eztfc_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.eztfc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)

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
                                GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
                else:
                        GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                        np.savetxt(param_list.GBOM_root+'_spectral_density_harmonic_qfc.dat', GBOM_chromophore.spectral_dens)
                        GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
                        if param_list.third_order:
                                GBOM_chromophore.calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission)
                GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr, is_emission)      
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)

                if param_list.exact_corr:
                        np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_exact_corr.dat', spectrum)
                else:
                        np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_harmonic_qfc.dat', spectrum)

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

                	temp_spectrum=linear_spectrum.full_spectrum(GBOM_batch.gboms[icount].fc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
			if icount==0:
				spectrum=spectrum+temp_spectrum
			else:
				spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]
			icount=icount+1

		spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)
		np.savetxt(param_list.GBOM_root+'_E_FTFC_spectrum.dat', spectrum)

	else:
		sys.exit('So far, only the FC method is implemented for the GBOM batch model.')


#  compute absorption spectra when chromophore model is given by both GBOM batch and MD batch
#def compute_hybrid_GBOM_MD_absorption:


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
			spectrum=linear_spectrum.full_spectrum(MDtraj.cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
		else:
			# set solvent response to a zero dummy vector
			spectrum=linear_spectrum.full_spectrum(MDtraj.cumulant_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False)
		np.savetxt(param_list.MD_root+'MD_cumulant_spectrum.dat', spectrum)

	# now do ensemble approach
	elif param_list.method=='ENSEMBLE':
		MDtraj.calc_ensemble_response(param_list.max_t,param_list.num_steps)
		if param_list.is_solvent:
                        spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True)
		else:
			# set solvent response to a zero dummy vector
			spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False)               
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
	param_set.stdout.write('Solvent reorg:    '+str(param_set.solvent_reorg)+' Ha'+'\n')
	param_set.stdout.write('Cutoff frequency:  '+str(param_set.solvent_cutoff_freq)+' Ha'+'\n')

# set up chromophore model
# pure GBOM model. 
if param_set.model=='GBOM':
	# sanity check:
	if param_set.num_modes==0:
		sys.exit('Error: Model GBOM requested but number of normal modes in the system is not set!')

	# single GBOM
	if param_set.num_gboms==1:
		# GBOM root is given. This means user requests reading params from Gaussian.
		if param_set.GBOM_root!='':		
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
				freqs_gs=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+str(batch_count)+'_gs.log',param_set.num_modes,param_set.num_frozen_atoms)
				freqs_ex=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+str(batch_count)+'_ex.log',param_set.num_modes,param_set.num_frozen_atoms)
				K=gaussian_params.extract_Kmat(param_set.GBOM_root+str(batch_count)+'vibronic.log',param_set.num_modes)
				J=gaussian_params.extract_duschinsky_mat(param_set.GBOM_root+str(batch_count)+'vibronic.log',param_set.num_modes)
				E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+str(batch_count)+'vibronic.log')
				# fill batch
				freqs_gs_batch[batch_count-1,:]=freqs_gs
				freqs_ex_batch[batch_count-1,:]=freqs_ex
				Jbatch[batch_count-1,:,:]=J
				Kbatch[batch_count-1,:]=K
				E_batch[batch_count-1]=E_adiabatic
				dipole_batch[batch_count-1]=param_set.dipole_mom

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
					else:   # if this is an EOPT_AV calculation we get the adiabatic energy gap from somewhere else
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

# both MD and GBOM input ---> E-ZTFC and related approaches 
elif param_set.model=='MD_GBOM':
	print('Currently not implemented option!')
else:
	sys.exit('Error: Invalid model '+param_set.model)

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
	else:
		sys.exit('Error: Only pure GBOM model or pure MD model implemented so far.')

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
				GBOM.calc_g3_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False)

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
					GBOM.calc_h1_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h1_real.dat',GBOM.h1_exact,False)
                                        twoDES.print_2D_spectrum('h1_imag.dat',GBOM.h1_exact,True)
					GBOM.calc_h2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h2_real.dat',GBOM.h2_exact,False)
                                        twoDES.print_2D_spectrum('h2_imag.dat',GBOM.h2_exact,True)
					GBOM.calc_h4_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h4_real.dat',GBOM.h4_exact,False)
                                        twoDES.print_2D_spectrum('h4_imag.dat',GBOM.h4_exact,True)
					GBOM.calc_h5_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
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
					twoDES.calc_2DES_time_series_GBOM_3rd(q_func_eff,GBOM.g3_exact,GBOM.h1_exact,GBOM.h2_exact,GBOM.h4_exact,GBOM.h5_exact,GBOM.corr_func_3rd_qm,GBOM.freqs_gs,GBOM.Omega_sq,GBOM.gamma,param_set.temperature*const.kb_in_Ha,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0,False,param_set.no_dusch)
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
				GBOM.calc_g3_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False)
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
					GBOM.calc_h1_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h1_real.dat',(GBOM.h1_cl),False)
                                        twoDES.print_2D_spectrum('h1_imag.dat',GBOM.h1_cl,True)
					GBOM.calc_h2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h2_real.dat',(GBOM.h2_cl),False)
                                        twoDES.print_2D_spectrum('h2_imag.dat',GBOM.h2_cl,True)
					GBOM.calc_h4_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h4_real.dat',(GBOM.h4_cl),False)
                                        twoDES.print_2D_spectrum('h4_imag.dat',GBOM.h4_cl,True)
					GBOM.calc_h5_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch)
					twoDES.print_2D_spectrum('h5_real.dat',(GBOM.h5_cl),False)
                                        twoDES.print_2D_spectrum('h5_imag.dat',GBOM.h5_cl,True,)
	
				# now construct 3rd order correlation function. Needed to speed up evaluation of h3
				#GBOM.compute_corr_func_3rd(param_set.temperature*const.kb_in_Ha,param_set.num_steps,param_set.max_t,False)
	
			if param_set.method_2DES=='2DES':
				if param_set.third_order:
					twoDES.calc_2DES_time_series_GBOM_3rd(q_func_eff,GBOM.g3_cl,GBOM.h1_cl,GBOM.h2_cl,GBOM.h4_cl,GBOM.h5_cl,GBOM.corr_func_3rd_cl,GBOM.freqs_gs,GBOM.Omega_sq,GBOM.gamma,param_set.temperature*const.kb_in_Ha,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0,True,param_set.no_dusch)

				else:
					twoDES.calc_2DES_time_series(q_func_eff,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)	
			elif param_set.method_2DES=='PUMP_PROBE':
				twoDES.calc_pump_probe_time_series(q_func_eff,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)

	# GBOM batch. Simplified implementation for the time being. Only 2nd order cumulant, and only standard 2DES
	elif param_set.model=='GBOM' and param_set.num_gboms!=1:

		if param_set.method=='EOPT_AV':     # this is not an E_FTFC calculation but rather an Eopt_avFTFC calculation
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

			average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)  # this is the place the spectrum should be centered on

			#HACK
                        E_start=average_Egap+Eopt_av-param_set.spectral_window/2.0-0.0025
                        E_end=average_Egap+Eopt_av+param_set.spectral_window/2.0-0.0025

			# now compute the average lineshape function g2:
			icount=0
			g2_av=np.zeros((param_set.num_steps,2))
			while icount<GBOM_batch.num_gboms:
				if param_set.exact_corr:
					if icount==0:
						g2_av=GBOM_batch.gboms[icount].g2_exact
					else:
						g2_av[:,1]=g2_av[:,1]+GBOM_batch.gboms[icount].g2_exact[:,1]
				else:
					if icount==0:
                                                g2_av=GBOM_batch.gboms[icount].g2_cl
                                        else:
                                                g2_av[:,1]=g2_av[:,1]+GBOM_batch.gboms[icount].g2_cl[:,1]
				
				icount=icount+1	

			# successfully constructed average effective set of GBOMs
			g2_av[:,1]=g2_av[:,1]/(1.0*GBOM_batch.num_gboms)

			# now construct list of g functions with the corrected energy shift taken from Eopt
			q_func_eff_batch = []
                        icount=0
                        while icount<Eopt.shape[0]:
				g2_temp=g2_av
				tcount=0
				while tcount<g2_temp.shape[0]:
					g2_temp[tcount,1]=g2_temp[tcount,1]+1j*(Eopt[icount]-average_Egap)*g2_temp[tcount,0]
					tcount=tcount+1

				q_func_eff_batch.append(g2_temp)
				icount=icount+1

			# created batch of g functions that are all the same, apart from different Eopt shifts
			# now construct 2DES spectra. 
			twoDES.calc_2DES_time_series_batch(q_func_eff_batch,Eopt.shape[0],E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)


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

			# create a list of effective q functions
			q_func_eff_batch = []
			icount=0
			while icount<GBOM_batch.num_gboms:
				q_func_eff=np.zeros((1,1))
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
