#! /usr/bin/env python

import os.path
import numpy as np
import math
import cmath
import spec_pkg.GBOM.franck_condon_response as franck_condon_response
import spec_pkg.GBOM.gbom_ensemble_response as gbom_ensemble_response
import spec_pkg.GBOM.gbom_cumulant_response as gbom_cumulant_response
from spec_pkg.constants import constants as const

# Global variable definition

#--------------------------------------------------------------------------
# Function definitions
def get_sd_skew(freqs_gs,Omega_sq,gamma,kbT,is_cl):
	# compute 2nd order corr func for t=0 and 
	# third order corr func for t1=t2=0. These 
	# values are related to skew and sd
	skew=gbom_cumulant_response.skew_from_third_order_corr(freqs_gs, Omega_sq, gamma, kbT,is_cl)
	sd=0.0
	if is_cl:
  		sd=gbom_cumulant_response.sd_from_2nd_order_corr_cl(freqs_gs, Omega_sq, gamma, kbT)
	else:
                sd=gbom_cumulant_response.sd_from_2nd_order_corr_qm(freqs_gs, Omega_sq, gamma, kbT)

	return sd,skew/(sd**3.0)

def get_lambda_0(Kmat,Jmat,omega_e_sq):
	Jtrans=np.transpose(Jmat)
	Ktrans=np.transpose(Kmat)
	temp_mat=np.dot(Ktrans,Jmat)
	temp_mat2=np.dot(Jtrans,Kmat)
	temp_mat3=np.dot(omega_e_sq,temp_mat2)
	return np.dot(temp_mat,temp_mat3)*0.5

def get_omega_sq(freqs):
	omega_mat=np.zeros((freqs.shape[0],freqs.shape[0]))
	counter=0
	while counter<freqs.shape[0]:
		omega_mat[counter,counter]=freqs[counter]**2.0
		counter=counter+1
	return omega_mat

def get_gamma(Kmat, Jmat,omega_e_sq):
	Jtrans=np.transpose(Jmat)
	Ktrans=np.transpose(Kmat)
	temp_mat=np.dot(Ktrans,Jmat)
	temp_mat2=np.dot(omega_e_sq,Jtrans)
	return np.dot(temp_mat,temp_mat2)

def get_full_Omega_sq(Jmat,omega_e_sq,freqs_gs):
	Jtrans=np.transpose(Jmat)
	temp_mat=np.dot(omega_e_sq,Jtrans)
	return_mat=0.5*np.dot(Jmat,temp_mat)
	counter=0
	while counter<return_mat.shape[0]:
 		return_mat[counter,counter]=return_mat[counter,counter]-0.5*freqs_gs[counter]**2.0
 		counter=counter+1
	return return_mat

def av_energy_gap_classical(Omega_sq,lambda_0,E_adiabatic,kbT,freqs_gs,is_emission):
	av_gap=E_adiabatic+lambda_0
	counter=0
	while counter<Omega_sq.shape[0]:
		if is_emission:
			av_gap=av_gap-Omega_sq[counter,counter]*kbT/(freqs_gs[counter]**2.0)
		else:
			av_gap=av_gap+Omega_sq[counter,counter]*kbT/(freqs_gs[counter]**2.0)
		counter=counter+1
	return av_gap

def av_energy_gap_exact_qm(Omega_sq,lambda_0,E_adiabatic,kbT,freqs_gs,is_emission):
	av_gap=0.0
	av_gap=lambda_0+E_adiabatic

	counter=0
	while counter<freqs_gs.shape[0]:
		if is_emission:
			av_gap=av_gap-Omega_sq[counter,counter]/(2.0*freqs_gs[counter])*math.cosh(freqs_gs[counter]/(2.0*kbT))/math.sinh(freqs_gs[counter]/(2.0*kbT))
		else:
			av_gap=av_gap+Omega_sq[counter,counter]/(2.0*freqs_gs[counter])*math.cosh(freqs_gs[counter]/(2.0*kbT))/math.sinh(freqs_gs[counter]/(2.0*kbT))
		counter=counter+1

	return av_gap


#--------------------------------------------------------------------------
# Class definition of the Generalized Brownian Oscillator Model. 
# Generates the GBOM model parameters from Gaussian output files
# Can construct linear response functions in the ensemble, E-ZTFC
# and pure Franck-Condon approach
# standard way to initialize the GBOM works 

class gbom:
	def __init__(self,freqs_gs,freqs_ex,J,K,E_adiabatic,dipole_mom,stdout):
		# Ground state, excited state and Duschinsky rotations,
		# adiabatic energy gap. 
		self.num_modes=freqs_gs.shape[0]
		self.freqs_gs=freqs_gs
		self.freqs_ex=freqs_ex
		self.J=J
		self.K=K
		# emission equivalents of J and K. for now set to zero unless they are needed
		self.J_emission=np.zeros((self.J.shape[0],self.J.shape[0]))
		self.K_emission=np.zeros(self.K.shape[0])
		self.freqs_gs_emission=np.zeros(self.freqs_ex.shape[0])
		self.freqs_ex_emission=np.zeros(self.freqs_gs.shape[0])

		# the E_adiabatic passed into this routine is actually E0_0. To go from E00 to Eadiabatic we have to 
		# subtract the zero point energy on the excited state PES and add the zero point energy on the ground- 
		# state PES.
		stdout.write('\n'+'Initializing a GBOM object.'+'\n')
		self.E_adiabatic=E_adiabatic+0.5*(-np.sum(self.freqs_ex)+np.sum(self.freqs_gs)) # the energy gap passed in this routine is the energy gap between 0-0 transitions. 
		# correct for zero-point energy of ground and excited state
		stdout.write('Adiabatic energy gap: '+str(self.E_adiabatic)+'  Ha'+'\n')
		self.dipole_mom=dipole_mom 
		stdout.write('Dipole moment  '+str(self.dipole_mom)+'  Ha'+'\n')

		# by default, this is set to an empty dummy array. It is only needed for HT effects
		self.dipole_deriv=np.zeros((freqs_gs.shape[0],3))

		# by default, initialize temporary variables for an absorption calculation.
		self.set_absorption_variables()

		# print ground and excited state frequencies
		stdout.write('   MODE'+'\t'+'GS FREQUENCY (cm-1)'+'\t'+'EX FREQUENCY (cm-1)'+'\n')
		for i in range(self.num_modes):
			stdout.write("%5d      %10.4f          %10.4f" % (i+1, self.freqs_gs[i]*const.Ha_to_cm,self.freqs_ex[i]*const.Ha_to_cm)+ '\n')
		# Derived terms beyond the model system parameters
		self.omega_av_cl=0.0
		self.omega_av_qm=0.0

		# response functions
		self.ensemble_response=np.zeros((1,1))
		self.fc_response=np.zeros((1,1))
		self.eztfc_response=np.zeros((1,1))
		self.cumulant_response=np.zeros((1,1))

		# Herzberg teller contribution
		self.HT=np.zeros((1,1))

		# cumulant terms:
		self.g2_exact=np.zeros((1,1),dtype=complex)
		self.g3_exact=np.zeros((1,1),dtype=complex)
		self.g2_cl=np.zeros((1,1),dtype=complex)
		self.g3_cl=np.zeros((1,1),dtype=complex)

		# 3rd order cumulant 2DES terms. Needed to evaluate the 3rd order 
		# cumulant correction to the 3rd order response function
		self.h1_exact=np.zeros((1,1,1),dtype=complex)
		self.h2_exact=np.zeros((1,1,1),dtype=complex)
		self.h1_cl=np.zeros((1,1,1),dtype=complex)
		self.h2_cl=np.zeros((1,1,1),dtype=complex)

		self.h4_exact=np.zeros((1,1,1),dtype=complex)
		self.h5_exact=np.zeros((1,1,1),dtype=complex)
		self.h4_cl=np.zeros((1,1,1),dtype=complex)
		self.h5_cl=np.zeros((1,1,1),dtype=complex)

		# spectral density. Only needed for comparisons
		self.spectral_dens=np.zeros((1,1),dtype=complex)

		# 3rd order correlation functions: Only needed for non-linear response beyond 2nd order cumulant
		self.corr_func_3rd_cl=np.zeros((1,1),dtype=complex)
		self.corr_func_3rd_cl_freq=np.zeros((1,1),dtype=complex)
		self.corr_func_3rd_qm=np.zeros((1,1),dtype=complex)
		self.corr_func_3rd_qm_freq=np.zeros((1,1),dtype=complex)

		# second order divergence factor for the cumulant approach
		self.second_order_divergence=0.0

	def calc_2nd_order_divergence(self,temp,is_qm): 
		kbT=const.kb_in_Ha*temp
		divergence_sum=0.0
		print('DIVERGENCE SUM COMPUTED FOR GBOM')
		for i in range(self.Omega_sq.shape[0]):
			if is_qm:
				omega_kbT=self.freqs_gs[i]/kbT
				divergence_sum=divergence_sum+(self.Omega_sq[i,i])**2.0/(2.0*self.freqs_gs[i]**2.0)*math.exp(omega_kbT)/(math.exp(omega_kbT)-1.0)**2.0
			else:
				divergence_sum=divergence_sum+(self.Omega_sq[i,i])**2.0/(2.0*self.freqs_gs[i]**4.0)*kbT**2.0
				print(i,divergence_sum)
		self.second_order_divergence=divergence_sum


	# need an emission equivalent for this, ie where the propagation happens on the excited state PES
	def compute_corr_func_3rd(self,kbT,num_points,max_t,is_qm,four_phonon_term):
		if is_qm:
			self.corr_func_3rd_qm=gbom_cumulant_response.full_third_order_corr_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_t,num_points,False,four_phonon_term)
		else:
			self_corr_func_3rd_cl=gbom_cumulant_response.full_third_order_corr_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_t,num_points,True,four_phonon_term)

	def compute_HT_term(self,temp,num_points,max_t,decay_length,is_qm,is_3rd,dipole_dipole_only,stdout):
		kbT=const.kb_in_Ha*temp
		self.HT=gbom_cumulant_response.full_HT_term(self.freqs_gs,self.K,self.J,self.Omega_sq,self.gamma,self.dipole_mom,self.dipole_deriv,kbT,max_t,num_points,decay_length,is_qm,is_3rd,dipole_dipole_only,stdout)


	def set_absorption_variables(self):
		# temporary variable. Not needed to be accessible to the outside
		omega_e_sq=get_omega_sq(self.freqs_ex)

		# These parameters are needed both in the cumulant and in ensemble approaches
		self.gamma=get_gamma(self.K,self.J,omega_e_sq)
		self.Omega_sq=get_full_Omega_sq(self.J,omega_e_sq,self.freqs_gs)
		self.lambda_0=get_lambda_0(self.K,self.J,omega_e_sq)


	# need to swap around initial and final state and reinitialize omega sq and other derived variables 
	# also set J_emission and K_emission. J_emission=J^T and K_emission=-J^T*K
	def set_emission_variables(self):
		# swap initial and final state
		self.freqs_gs_emission=self.freqs_ex
		self.freqs_ex_emission=self.freqs_gs
		# same as for absorption but with initial and final state swapped
		# in emisson variables, only omega_e_sq, gamma, Omega_sq and lambda_0 get set correctly
		omega_e_sq=get_omega_sq(self.freqs_ex_emission)
		self.K_emission=-np.dot(np.transpose(self.J),self.K)
		self.J_emission=np.transpose(self.J)
		self.gamma=get_gamma(self.K_emission,self.J_emission, omega_e_sq)
		self.Omega_sq=get_full_Omega_sq(self.J_emission,omega_e_sq,self.freqs_gs_emission)
		# in emission calculation, flip sign of lambda_0
		self.lambda_0=-get_lambda_0(self.K_emission,self.J_emission,omega_e_sq) 
		# adjust E_adiabatic:
		#self.E_adiabatic=self.E_adiabatic+0.5*(-np.sum(self.freqs_ex_emission)+np.sum(self.freqs_gs_emission))
		
	def calc_cumulant_response(self,is_3rd_order_cumulant,is_qm,is_emission,is_HT):
		if is_qm:
			self.cumulant_response=gbom_cumulant_response.compute_cumulant_response(self.g2_exact,self.g3_exact,self.dipole_mom,self.HT,is_3rd_order_cumulant,is_HT,is_emission)
		else:
			self.cumulant_response=gbom_cumulant_response.compute_cumulant_response(self.g2_cl,self.g3_cl,self.dipole_mom,self.HT,is_3rd_order_cumulant,is_HT,is_emission)

	def calc_spectral_dens(self,temp,max_t,max_steps,decay_length,is_cl,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.spectral_dens=gbom_cumulant_response.compute_spectral_dens(self.freqs_gs_emission,self.Omega_sq,self.gamma,kbT,max_t,max_steps,decay_length,is_cl)
		else:
			self.spectral_dens=gbom_cumulant_response.compute_spectral_dens(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_t,max_steps,decay_length,is_cl)
	
	def calc_g2_cl(self,temp,num_points,max_time,is_emission,stdout):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			# sanity check. Compute sd and skew to see whether cumulant is reliable
			sd,skew=get_sd_skew(self.freqs_gs_emission,self.Omega_sq,self.gamma,kbT,True)
			stdout.write('\n'+'Standard deviation of energy gap fluctuations for GBOM: '+str(sd)+' Ha. Skewness: '+str(skew)+'\n')
			if abs(skew)>0.3:
				stdout.write('WARNING: Large skewness value of '+str(skew)+' detected in energy gap fluctuations.'+'\n')
				stdout.write('This means that the energy gap fluctuations are likely non-Gaussian in nature and low-order cumulant expansions might be unreliable!'+'\n'+'\n')

			self.g2_cl=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_gs_emission,self.freqs_ex_emission,self.K_emission,self.J_emission,self.gamma,self.Omega_sq,kbT,self.omega_av_cl,max_time,num_points,True,is_emission,self.E_adiabatic,stdout)
		else:
			sd,skew=get_sd_skew(self.freqs_gs,self.Omega_sq,self.gamma,kbT,True)
			stdout.write('\n'+'Standard deviation of energy gap fluctuations for GBOM: '+str(sd)+' Ha. Skewness: '+str(skew)+'\n')
			if abs(skew)>0.3:
				stdout.write('WARNING: Large skewness value of '+str(skew)+' detected in energy gap fluctuations.'+'\n')
				stdout.write('This means that the energy gap fluctuations are likely non-Gaussian in nature and low-order cumulant expansions might be unreliable!'+'\n'+'\n')
			self.g2_cl=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_gs,self.freqs_ex,self.K,self.J,self.gamma,self.Omega_sq,kbT,self.omega_av_cl,max_time,num_points,True,is_emission,self.E_adiabatic,stdout)


	def calc_g3_cl(self,temp,num_points,max_time,is_emission,four_phonon_term,g3_cutoff,stdout):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.g3_cl=gbom_cumulant_response.full_third_order_lineshape(self.freqs_gs_emission,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,four_phonon_term,g3_cutoff,stdout)
		else:
			self.g3_cl=gbom_cumulant_response.full_third_order_lineshape(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,four_phonon_term,g3_cutoff,stdout)

	def calc_g3_qm(self,temp,num_points,max_time,is_emission,four_phonon_term,g3_cutoff,stdout):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.g3_exact=gbom_cumulant_response.full_third_order_lineshape(self.freqs_gs_emission,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,four_phonon_term,g3_cutoff,stdout)
		else:
			self.g3_exact=gbom_cumulant_response.full_third_order_lineshape(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,four_phonon_term,g3_cutoff,stdout)

	def calc_h4_qm(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h4_exact=gbom_cumulant_response.full_h4_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch,four_phonon_term)

	def calc_h5_qm(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h5_exact=gbom_cumulant_response.full_h5_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch,four_phonon_term)

	def calc_h4_cl(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h4_cl=gbom_cumulant_response.full_h4_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch,four_phonon_term)

	def calc_h5_cl(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h5_cl=gbom_cumulant_response.full_h5_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch,four_phonon_term)

	def calc_h1_qm(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h1_exact=gbom_cumulant_response.full_h1_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch,four_phonon_term)

	def calc_h2_qm(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h2_exact=gbom_cumulant_response.full_h2_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch,four_phonon_term)

	def calc_h1_cl(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h1_cl=gbom_cumulant_response.full_h1_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch,four_phonon_term)

	def calc_h2_cl(self,temp,num_points,max_time,no_dusch,four_phonon_term):
		kbT=const.kb_in_Ha*temp
		self.h2_cl=gbom_cumulant_response.full_h2_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch,four_phonon_term)

	def calc_g2_qm(self,temp,num_points,max_time,is_emission,stdout):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			sd,skew=get_sd_skew(self.freqs_gs_emission,self.Omega_sq,self.gamma,kbT,False)
			stdout.write('\n'+'Standard deviation of energy gap fluctuations for GBOM: '+str(sd)+' Ha. Skewness: '+str(skew)+'\n')
			if abs(skew)>0.3:
				stdout.write('WARNING: Large skewness value of '+str(skew)+' detected in energy gap fluctuations.'+'\n')
				stdout.write('This means that the energy gap fluctuations are likely non-Gaussian in nature and low-order cumulant expansions might be unreliable!'+'\n'+'\n')
			self.g2_exact=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_gs_emission,self.freqs_ex_emission,self.K_emission,self.J_emission,self.gamma,self.Omega_sq,kbT,self.omega_av_qm,max_time,num_points,False,is_emission,self.E_adiabatic,stdout)
		else:
			sd,skew=get_sd_skew(self.freqs_gs,self.Omega_sq,self.gamma,kbT,False)
			stdout.write('\n'+'Standard deviation of energy gap fluctuations for GBOM: '+str(sd)+' Ha. Skewness: '+str(skew)+'\n')
			if abs(skew)>0.3:
				stdout.write('WARNING: Large skewness value of '+str(skew)+' detected in energy gap fluctuations.'+'\n')
				stdout.write('This means that the energy gap fluctuations are likely non-Gaussian in nature and low-order cumulant expansions might be unreliable!'+'\n'+'\n')
			self.g2_exact=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_gs,self.freqs_ex,self.K,self.J,self.gamma,self.Omega_sq,kbT,self.omega_av_qm,max_time,num_points,False,is_emission,self.E_adiabatic,stdout)

	def calc_omega_av_cl(self,temp,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.omega_av_cl=av_energy_gap_classical(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_gs_emission,is_emission)
		else:
			self.omega_av_cl=av_energy_gap_classical(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_gs,is_emission)

	def calc_omega_av_qm(self,temp,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.omega_av_qm=av_energy_gap_exact_qm(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_gs_emission,is_emission)
		else:
			self.omega_av_qm=av_energy_gap_exact_qm(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_gs,is_emission)

	def calc_ensemble_response(self,temp,num_points,max_time,is_qm,is_emission,stdout):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			omega_e_sq=get_omega_sq(self.freqs_ex_emission)
			omega_g_sq=get_omega_sq(self.freqs_gs_emission)
			self.ensemble_response=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs_emission,self.freqs_ex_emission,self.J_emission,self.K_emission,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm,is_emission,stdout)
		else:
			omega_e_sq=get_omega_sq(self.freqs_ex)
			omega_g_sq=get_omega_sq(self.freqs_gs)
			self.ensemble_response=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm,is_emission,stdout)

	def calc_fc_response(self,temp,num_points,max_time,is_emission,is_HT,stdout):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			 self.fc_response=franck_condon_response.compute_full_response_func(self.freqs_gs_emission,self.freqs_ex_emission,self.J_emission,self.K_emission,self.E_adiabatic,self.dipole_mom,self.dipole_deriv,kbT,num_points,max_time,is_emission,is_HT,stdout)
		else:
			self.fc_response=franck_condon_response.compute_full_response_func(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.dipole_mom,self.dipole_deriv,kbT,num_points,max_time,is_emission,is_HT,stdout)

	# differentiate between absorption and emission, and whether this is a Quantum Wigner distribution or classical 
	# distribution for the ensemble sampling
	def calc_eztfc_response(self,temp,num_points,max_time,is_qm,is_emission,is_HT,stdout):
		low_temp_kbT=10.0*const.kb_in_Ha   # needed for computing low temperature FC limit. Set T to 10 K
		if is_emission:
			ztfc_resp=franck_condon_response.compute_full_response_func(self.freqs_gs_emission,self.freqs_ex_emission,self.J_emission,self.K_emission,self.E_adiabatic,low_temp_kbT,self.dipole_mom,self.dipole_deriv,num_points,max_time,is_emission,is_HT,stdout)
		else: 
			ztfc_resp=franck_condon_response.compute_full_response_func(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.dipole_mom,self.dipole_deriv,low_temp_kbT,num_points,max_time,is_emission,is_HT,stdout)
		kbT=temp*const.kb_in_Ha
		# implement this so it works either with quantum wigner or classical distribution of nuclei
		if is_emission:
			omega_e_sq=get_omega_sq(self.freqs_ex_emission)
			omega_g_sq=get_omega_sq(self.freqs_gs_emission)
			if is_qm:
				# compute 'zero temperature' ensemble spectrum
				ensemble_resp_zero_t=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs_emission,self.freqs_ex_emission,self.J_emission,self.K_emission,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,low_temp_kbT,num_points,max_time,True,is_emission,stdout)
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs_emission,self.freqs_ex_emission,self.J_emission,self.K_emission,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,True,is_emission,stdout)
			else:
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs_emission,self.freqs_ex_emission,self.J_emission,self.K_emission,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,False,is_emission,stdout)
		else:
			omega_e_sq=get_omega_sq(self.freqs_ex)
			omega_g_sq=get_omega_sq(self.freqs_gs)
			if is_qm:
				# compute 'zero temperature' ensemble spectrum
				ensemble_resp_zero_t=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,low_temp_kbT,num_points,max_time,True,is_emission,stdout)
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,True,is_emission,stdout)
			else:
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,False,is_emission,stdout)

		total_response=np.zeros((ztfc_resp.shape[0],2),dtype=complex)
		counter=0
		# now combine both the ZTFC and the Ensemble lineshapes. Add the magnitude and the phase 
		# individually. We also need to add a term to the phase factor to ensure that the Ensemble
		# spectrum is centered around 0. Also, if this is a calculation using a QM wigner distribution
		# for computing the ensemlbe spectrum, make sure that we add the appropriate correction factor (the zero temperature
		# limit of the QM ensemble spectrum).  		
		while counter<ztfc_resp.shape[0]:
			total_response[counter,0]=ztfc_resp[counter,0]
			if is_qm:
				total_response[counter,1]=ztfc_resp[counter,1]*ensemble_resp[counter,1]/ensemble_resp_zero_t[counter,1]
			else:
				total_response[counter,1]=ztfc_resp[counter,1]*ensemble_resp[counter,1]*cmath.exp(1j*(self.E_adiabatic)*ztfc_resp[counter,0]) 
			counter=counter+1

		self.eztfc_response=total_response


# class definition of a list of GBOM models. This is relevant for the E-ZTFC approach as applied to condensed phase systems, but also more accurate 
# approaches like E-avFTFC and E-FTFC.  
# FIX emission approach for GBOM list, similar to the correction done in GBOM
class gbom_list:
	def __init__(self,freqs_gs_batch,freqs_ex_batch,J_batch,K_batch,E_adiabatic_batch,dipole_mom_batch,num_gboms,stdout):
		self.gboms = []	
		# gboms is a list of GBOM objects
		for i in range(num_gboms):
			stdout.write('\n'+'\n'+'ADDING GBOM '+str(i+1)+' OUT OF '+str(num_gboms)+' TO BATCH.'+'\n')
			self.gboms.append(gbom(freqs_gs_batch[i,:],freqs_ex_batch[i,:],J_batch[i,:,:],K_batch[i,:],E_adiabatic_batch[i],dipole_mom_batch[i],stdout))
			
		self.num_gboms=num_gboms

		# response functions
		self.avZTFC_response=np.zeros((1,1))  # average ZTFC response function, calculated by setting E_adiabatic to the same value for all of them
		self.avFTFC_response=np.zeros((1,1))  # average FTFC response function, calculated by setting E_adiabatic to the same value for all of them
		self.full_FTFC_response=np.zeros((1,1)) # Just a sum of all FTFC response functions for each GBOM, divided by num_gboms 
		self.avEnsemble_response=np.zeros((1,1)) # sum of all ensemble response functions
		self.avCumulant_response=np.zeros((1,1)) # sum of all cumulant response functions

	
	# Routines for calculating the response functions
	def calc_avEnsemble_response(self,temp,num_points,max_time,is_qm,is_emission,stdout):
		kbT=const.kb_in_Ha*temp
		counter=0
		while counter<self.num_gboms:
			omega_e_sq=get_omega_sq(self[counter].freqs_ex)
			omega_g_sq=get_omega_sq(self[counter].freqs_gs)

			if is_emission:
				eff_K=-np.dot(np.transpose(self[counter].J),self[counter].K)
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self[counter].freqs_ex,self[counter].freqs_gs,np.transpose(self[counter].J),eff_K,self[counter].E_adiabatic,self[counter].lambda_0,self[counter].gamma,self[counter].Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm,is_emission,stdout)
			else:
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[counter].K,self[counter].E_adiabatic,self[counter].lambda_0,self[counter].gamma,self[counter].Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm,is_emission,stdout)
			if counter==0:
				self.avEnsemble_response=ensemble_resp
			else:
				self.avEnsemble_response[:,1]=self.avEnsemble_response[:,1]+ensemble_resp[:,1]
			counter=counter+1
		self.avEnsemble_response[:,1]=self.avEnsemble_response[:,1]/(1.0*self.num_gboms)

	def calc_avZTFC_response(self,num_points,max_time,is_emission,is_HT,stdout):
		low_temp_kbT=10.0*const.kb_in_Ha
		counter=0
		while counter<self.num_gboms:
			ztfc_resp=franck_condon_response.compute_full_response_func(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[counter].K,0.0,self.dipole_mom,self.dipole_deriv,low_temp_kbT,num_points,max_time,is_emission,stdout)
			if counter==0:
				self.avZTFC_response=ztfc_resp
			else:
				self.avZTFC_response[:,1]=self.avZTFC_response[:,1]+ztfc_rsponse[:,1]		

			counter=counter+1
		self.avZTFC_response[:,1]=self.avZTFC_response[:,1]/(1.0*self.num_gboms)

	def calc_avFTFC_response(self,temp,num_points,max_time,is_emission,is_HT,stdout):     
		kbT=temp*const.kb_in_Ha
		counter=0
		print('Compute FTFC response for GBOM batch.')
		while counter<self.num_gboms:
			stdout.write('PROCESSING BATCH	'+str(counter))
			ftfc_resp=franck_condon_response.compute_full_response_func(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[couinter].K,0.0,self.dipole_mom,self.dipole_deriv,kbT,num_points,max_time,is_emission,is_HT,stdout)
			if counter==0:
				self.avFTFC_response=ftfc_resp
			else:
				self.avFTFC_response[:,1]=self.avFTFC_response[:,1]+ftfc_rsponse[:,1]

			counter=counter+1
		self.avFTFC_response[:,1]=self.avFTFC_response[:,1]/(1.0*self.num_gboms)

	def calc_full_FTFC_response(self,temp,num_points,max_time,is_emission,is_HT,stdout):
		kbT=temp*const.kb_in_Ha
		counter=0
		while counter<self.num_gboms:
			stdout.write('PROCESSING BATCH  '+str(counter))
			ftfc_resp=franck_condon_response.compute_full_response_func(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[counter].K,self[counter].E_adiabatic,self.dipole_mom,self.dipole_deriv,kbT,num_points,max_time,is_emission,is_HT,stdout)
			if counter==0:
				self.full_FTFC_response=ftfc_resp
			else:
				self.full_FTFC_response[:,1]=self.full_FTFC_response[:,1]+ftfc_rsponse[:,1]

			counter=counter+1
		self.full_FTFC_response[:,1]=self.full_FTFC_response[:,1]/(1.0*self.num_gboms)

