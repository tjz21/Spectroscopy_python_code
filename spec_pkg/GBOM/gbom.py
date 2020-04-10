#! /usr/bin/env python

import os.path
import numpy as np
import math
import cmath
import franck_condon_response
import gbom_ensemble_response
import gbom_cumulant_response
from ..constants import constants as const

# Global variable definition

#--------------------------------------------------------------------------
# Function definitions
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
    if is_emission:
	av_gap=-lambda_0+E_adiabatic
    else:
    	av_gap=lambda_0+E_adiabatic
    counter=0
    while counter<Omega_sq.shape[0]:
	if is_emission:
		av_gap=av_gap-Omega_sq[counter,counter]*kbT/(freqs_gs[counter]**2.0)
	else:
        	av_gap=av_gap+Omega_sq[counter,counter]*kbT/(freqs_gs[counter]**2.0)
        counter=counter+1

    return av_gap

def av_energy_gap_exact_qm(Omega_sq,lambda_0,E_adiabatic,kbT,freqs_gs,is_emission):
    # first construct matrices needed:
    if is_emission:
	av_gap=-lambda_0+E_adiabatic
    else:
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
	def __init__(self,freqs_gs,freqs_ex,J,K,E_adiabatic,dipole_mom):
		# Ground state, excited state and Duschinsky rotations,
		# adiabatic energy gap. 
		self.num_modes=freqs_gs.shape[0]
		self.freqs_gs=freqs_gs
		self.freqs_ex=freqs_ex
		self.J=J
		self.K=K
		self.E_adiabatic=E_adiabatic+0.5*(-np.sum(self.freqs_ex)+np.sum(self.freqs_gs)) # the energy gap passed in this routine is the energy gap between 0-0 transitions. 
		# correct for zero-point energy of ground and excited state
		self.dipole_mom=dipole_mom 
		
		# by default, initialize temporary variables for an absorption calculation.
		self.set_absorption_variables()
	
		# Derived terms beyond the model system parameters
		self.omega_av_cl=0.0
		self.omega_av_qm=0.0
	
		# response functions
		self.ensemble_response=np.zeros((1,1))
		self.fc_response=np.zeros((1,1))
		self.eztfc_response=np.zeros((1,1))
		self.cumulant_response=np.zeros((1,1))

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

	def compute_corr_func_3rd(self,kbT,num_points,max_t,is_qm):
		if is_qm:
			self.corr_func_3rd_qm=gbom_cumulant_response.full_third_order_corr_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_t,num_points,False)
		else:
			self_corr_func_3rd_cl=gbom_cumulant_response.full_third_order_corr_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_t,num_points,True)


	def set_absorption_variables(self):
                # temporary variable. Not needed to be accessible to the outside
                omega_e_sq=get_omega_sq(self.freqs_ex)

                # These parameters are needed both in the cumulant and in ensemble approaches
                self.gamma=get_gamma(self.K,self.J,omega_e_sq)
                self.Omega_sq=get_full_Omega_sq(self.J,omega_e_sq,self.freqs_gs)
                self.lambda_0=get_lambda_0(self.K,self.J,omega_e_sq)

	# need to swap around initial and final state and reinitialize omega sq and other derived variables 
	def set_emission_variables(self):
		# same as for absorption but with initial and final state swapped
		omega_e_sq=get_omega_sq(self.freqs_gs)
		eff_K=-np.dot(np.transpose(self.J),self.K)
		self.gamma=get_gamma(eff_K,np.transpose(self.J), omega_e_sq)
		self.Omega_sq=get_full_Omega_sq(np.transpose(self.J),omega_e_sq,self.freqs_ex)
		self.lambda_0=get_lambda_0(eff_K,np.transpose(self.J),omega_e_sq)
		print self.lambda_0

	def calc_cumulant_response(self,is_3rd_order_cumulant,is_qm,is_emission):
		if is_qm:
			self.cumulant_response=gbom_cumulant_response.compute_cumulant_response(self.g2_exact,self.g3_exact,is_3rd_order_cumulant,is_emission)
		else:
			self.cumulant_response=gbom_cumulant_response.compute_cumulant_response(self.g2_cl,self.g3_cl,is_3rd_order_cumulant,is_emission)

	def calc_spectral_dens(self,temp,max_t,max_steps,decay_length,is_cl,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.spectral_dens=gbom_cumulant_response.compute_spectral_dens(self.freqs_ex,-self.Omega_sq,-self.gamma,kbT,max_t,max_steps,decay_length,is_cl)
		else:
			self.spectral_dens=gbom_cumulant_response.compute_spectral_dens(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_t,max_steps,decay_length,is_cl)
	
	def calc_g2_cl(self,temp,num_points,max_time,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			eff_K=-np.dot(np.transpose(self.J),self.K)
			self.g2_cl=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_ex,self.freqs_gs,eff_K,np.transpose(self.J),-self.gamma,-self.Omega_sq,kbT,self.omega_av_cl,max_time,num_points,True)
		else:
			self.g2_cl=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_gs,self.freqs_ex,self.K,self.J,self.gamma,self.Omega_sq,kbT,self.omega_av_cl,max_time,num_points,True)

        def calc_g3_cl(self,temp,num_points,max_time,is_emission):
                kbT=const.kb_in_Ha*temp
		if is_emission:
			self.g3_cl=gbom_cumulant_response.full_third_order_lineshape(self.freqs_ex,-self.Omega_sq,-self.gamma,kbT,max_time,num_points,True)
		else:
                	self.g3_cl=gbom_cumulant_response.full_third_order_lineshape(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True)

        def calc_g3_qm(self,temp,num_points,max_time,is_emission):
                kbT=const.kb_in_Ha*temp
		if is_emission:
			self.g3_exact=gbom_cumulant_response.full_third_order_lineshape(self.freqs_ex,-self.Omega_sq,-self.gamma,kbT,max_time,num_points,False)
		else:
                	self.g3_exact=gbom_cumulant_response.full_third_order_lineshape(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False)

        def calc_h4_qm(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h4_exact=gbom_cumulant_response.full_h4_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch)

        def calc_h5_qm(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h5_exact=gbom_cumulant_response.full_h5_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch)

        def calc_h4_cl(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h4_cl=gbom_cumulant_response.full_h4_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch)

        def calc_h5_cl(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h5_cl=gbom_cumulant_response.full_h5_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch)




        def calc_h1_qm(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h1_exact=gbom_cumulant_response.full_h1_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch)

	def calc_h2_qm(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h2_exact=gbom_cumulant_response.full_h2_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,False,no_dusch)

        def calc_h1_cl(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h1_cl=gbom_cumulant_response.full_h1_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch)

        def calc_h2_cl(self,temp,num_points,max_time,no_dusch):
                kbT=const.kb_in_Ha*temp
                self.h2_cl=gbom_cumulant_response.full_h2_func(self.freqs_gs,self.Omega_sq,self.gamma,kbT,max_time,num_points,True,no_dusch)

	def calc_g2_qm(self,temp,num_points,max_time,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			eff_K=-np.dot(np.transpose(self.J),self.K)
			self.g2_exact=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_ex,self.freqs_gs,eff_K,np.transpose(self.J),self.gamma,self.Omega_sq,kbT,self.omega_av_qm,max_time,num_points,False)
		else:
                	self.g2_exact=gbom_cumulant_response.full_2nd_order_lineshape(self.freqs_gs,self.freqs_ex,self.K,self.J,self.gamma,self.Omega_sq,kbT,self.omega_av_qm,max_time,num_points,False)

	def calc_omega_av_cl(self,temp,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.omega_av_cl=av_energy_gap_classical(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_ex,is_emission)
		else:
			self.omega_av_cl=av_energy_gap_classical(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_gs,is_emission)

	def calc_omega_av_qm(self,temp,is_emission):
		kbT=const.kb_in_Ha*temp
		if is_emission:
			self.omega_av_qm=av_energy_gap_exact_qm(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_ex,is_emission)
		else:
                	self.omega_av_qm=av_energy_gap_exact_qm(self.Omega_sq,self.lambda_0,self.E_adiabatic,kbT,self.freqs_gs,is_emission)

	def calc_ensemble_response(self,temp,num_points,max_time,is_qm,is_emission):
		kbT=const.kb_in_Ha*temp
		omega_e_sq=get_omega_sq(self.freqs_ex)
		omega_g_sq=get_omega_sq(self.freqs_gs)
		if is_emission:
			eff_K=-np.dot(np.transpose(self.J),self.K)
			self.ensemble_response=gbom_ensemble_response.compute_ensemble_response(self.freqs_ex,self.freqs_gs,np.transpose(self.J),eff_K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm)
		else:
			self.ensemble_response=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm)

	def calc_fc_response(self,temp,num_points,max_time,is_emission):
		kbT=const.kb_in_Ha*temp
                self.fc_response=franck_condon_response.compute_full_response_func(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,kbT,num_points,max_time,is_emission)

	# differentiate between absorption and emission, and whether this is a Quantum Wigner distribution or classical 
	# distribution for the ensemble sampling
	def calc_eztfc_response(self,temp,num_points,max_time,is_qm,is_emission):
		low_temp_kbT=10.0*const.kb_in_Ha   # needed for computing low temperature FC limit. 
		ztfc_resp=franck_condon_response.compute_full_response_func(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,low_temp_kbT,num_points,max_time,is_emission)
		kbT=temp*const.kb_in_Ha
		omega_e_sq=get_omega_sq(self.freqs_ex)
                omega_g_sq=get_omega_sq(self.freqs_gs)
		# implement this so it works either with quantum wigner or classical distribution of nuclei
		if is_emission:
			eff_K=-np.dot(np.transpose(self.J),self.K)
			if is_qm:
				# compute 'zero temperature' ensemble spectrum
				ensemble_resp_zero_t=gbom_ensemble_response.compute_ensemble_response(self.freqs_ex,self.freqs_gs,np.transpose(self.J),eff_K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,low_temp_kbT,num_points,max_time,True)
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_ex,self.freqs_gs,np.transpose(self.J),eff_K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,True)
			else:
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_ex,self.freqs_gs,np.transpose(self.J),eff_K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,False)
		else:
			if is_qm:
				# compute 'zero temperature' ensemble spectrum
                                ensemble_resp_zero_t=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,low_temp_kbT,num_points,max_time,True)
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,True)
			else:
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self.freqs_gs,self.freqs_ex,self.J,self.K,self.E_adiabatic,self.lambda_0,self.gamma,self.Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,False)

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
class gbom_list:
        def __init__(self,freqs_gs_batch,freqs_ex_batch,J_batch,K_batch,E_adiabatic_batch,dipole_mom_batch,num_gboms):
		self.gboms = []	
		# gboms is a list of GBOM objects
		for i in range(num_gboms):
			self.gboms.append(gbom(freqs_gs_batch[i,:],freqs_ex_batch[i,:],J_batch[i,:,:],K_batch[i,:],E_adiabatic_batch[i],dipole_mom_batch[i]))
			
		self.num_gboms=num_gboms

		# response functions
		self.avZTFC_response=np.zeros((1,1))  # average ZTFC response function, calculated by setting E_adiabatic to the same value for all of them
		self.avFTFC_response=np.zeros((1,1))  # average FTFC response function, calculated by setting E_adiabatic to the same value for all of them
		self.full_FTFC_response=np.zeros((1,1)) # Just a sum of all FTFC response functions for each GBOM, divided by num_gboms 
		self.avEnsemble_response=np.zeros((1,1)) # sum of all ensemble response functions
		self.avCumulant_response=np.zeros((1,1)) # sum of all cumulant response functions

	
	# Routines for calculating the response functions
	def calc_avEnsemble_response(self,temp,num_points,max_time,is_qm,is_emission):
		kbT=const.kb_in_Ha*temp
		counter=0
		while counter<self.num_gboms:
			omega_e_sq=get_omega_sq(self[counter].freqs_ex)
                	omega_g_sq=get_omega_sq(self[counter].freqs_gs)

			if is_emission:
				eff_K=-np.dot(np.transpose(self[counter].J),self[counter].K)
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self[counter].freqs_ex,self[counter].freqs_gs,np.transpose(self[counter].J),eff_K,self[counter].E_adiabatic,self[counter].lambda_0,self[counter].gamma,self[counter].Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm)
			else:
				ensemble_resp=gbom_ensemble_response.compute_ensemble_response(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[counter].K,self[counter].E_adiabatic,self[counter].lambda_0,self[counter].gamma,self[counter].Omega_sq,omega_e_sq,omega_g_sq,kbT,num_points,max_time,is_qm)
			if counter==0:
				self.avEnsemble_response=ensemble_resp
			else:
				self.avEnsemble_response[:,1]=self.avEnsemble_response[:,1]+ensemble_resp[:,1]
			counter=counter+1
		self.avEnsemble_response[:,1]=self.avEnsemble_response[:,1]/(1.0*self.num_gboms)

	def calc_avZTFC_response(self,num_points,max_time,is_emission):
		low_temp_kbT=10.0*const.kb_in_Ha
		counter=0
		while counter<self.num_gboms:
			ztfc_resp=franck_condon_response.compute_full_response_func(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[counter].K,0.0,low_temp_kbT,num_points,max_time,is_emission)
			if counter==0:
				self.avZTFC_response=ztfc_resp
			else:
				self.avZTFC_response[:,1]=self.avZTFC_response[:,1]+ztfc_rsponse[:,1]		

			counter=counter+1
		self.avZTFC_response[:,1]=self.avZTFC_response[:,1]/(1.0*self.num_gboms)

	def calc_avFTFC_response(self,temp,num_points,max_time,is_emission):     
                kbT=temp*const.kb_in_Ha
                counter=0
		print('Compute FTFC response for GBOM batch.')
                while counter<self.num_gboms:
			print('Processing batch	'+str(counter))
                        ftfc_resp=franck_condon_response.compute_full_response_func(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[couinter].K,0.0,kbT,num_points,max_time,is_emission)
                        if counter==0:
                                self.avFTFC_response=ftfc_resp
                        else:
                                self.avFTFC_response[:,1]=self.avFTFC_response[:,1]+ftfc_rsponse[:,1]

                        counter=counter+1
                self.avFTFC_response[:,1]=self.avFTFC_response[:,1]/(1.0*self.num_gboms)

        def calc_full_FTFC_response(self,temp,num_points,max_time,is_emission):
                kbT=temp*const.kb_in_Ha
                counter=0
                while counter<self.num_gboms:
                        ftfc_resp=franck_condon_response.compute_full_response_func(self[counter].freqs_gs,self[counter].freqs_ex,self[counter].J,self[counter].K,self[counter].E_adiabatic,kbT,num_points,max_time,is_emission)
                        if counter==0:
                                self.full_FTFC_response=ftfc_resp
                        else:
                                self.full_FTFC_response[:,1]=self.full_FTFC_response[:,1]+ftfc_rsponse[:,1]

                        counter=counter+1
                self.full_FTFC_response[:,1]=self.full_FTFC_response[:,1]/(1.0*self.num_gboms)

