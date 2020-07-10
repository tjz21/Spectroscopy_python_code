#! /usr/bin/env python

import numpy as np
import math
import cmath
from numba import jit
from ..constants import constants as const

# contains all routines to construct the classical ensemble spectrum for the GBOM
@jit
def get_Bmat(full_Omega_sq,freqs_gs,kbT,t,is_QM):
	Bmat=full_Omega_sq*1j*t
	counter=0
	while counter<Bmat.shape[0]:
		if is_QM:
 			Bmat[counter,counter]=Bmat[counter,counter]+freqs_gs[counter]*math.tanh(freqs_gs[counter]/(2.0*kbT))
		else:
			Bmat[counter,counter]=Bmat[counter,counter]+freqs_gs[counter]**2.0/(2.0*kbT)
		counter=counter+1
	return Bmat

@jit
def get_inv_freqs_gs_sq(freqs_gs):
	inv_freqs_gs=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
	counter=0
	while counter<freqs_gs.shape[0]:
		inv_freqs_gs[counter,counter]=1.0/(freqs_gs[counter]**2.0)
		counter=counter+1
	return inv_freqs_gs

@jit
# Check if we have quantum nuclei or not
def get_Bdash_mat(Jmat,freqs_gs,omega_e_sq,kbT,inv_omega_g_sq,t,is_QM):
	Jtrans=np.transpose(Jmat)
	temp_mat=np.dot(omega_e_sq,Jtrans)
	temp_mat2=np.dot(Jmat,temp_mat)
	if is_QM:
		# create effective inv_omega_sq matrix which in this case is given by 1/2omega*coth(beta omega/2)
		eff_inv_mat=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]),dtype=np.complex_)
		counter=0
		while counter<eff_inv_mat.shape[0]:
			eff_inv_mat[counter,counter]=1j*t/(2.0*freqs_gs[counter]*math.tanh(freqs_gs[counter]/(2.0*kbT)))
			counter=counter+1
		Bdash=np.dot(eff_inv_mat,temp_mat2)
	else:
		Bdash=1j*kbT*t*np.dot(inv_omega_g_sq,temp_mat2)
	counter=0
	while counter<Bdash.shape[0]:
		if is_QM:
			Bdash[counter,counter]=Bdash[counter,counter]-1j*freqs_gs[counter]/(2.0*math.tanh(freqs_gs[counter]/(2.0*kbT)))*t+1.0
		else:
			Bdash[counter,counter]=Bdash[counter,counter]-1j*kbT*t+1.0
		counter=counter+1
	return Bdash

@jit
def get_prefac(Bdash_mat):
	return 1.0/cmath.sqrt(np.linalg.det(Bdash_mat))

@jit
def calc_chi_for_given_time(freq_gs,freq_ex,Jmat,Kmat,lambda_0,gamma,Omega_sq,omega_e_sq,omega_g_sq,kBT,time,is_QM):
	# calculate full value of chi(t) for the given value of t.
	inv_omega_g_sq=get_inv_freqs_gs_sq(freq_gs)
	Bmat=get_Bmat(Omega_sq.astype(np.complex_),freq_gs,kBT,time,is_QM)
	Bdash_mat=get_Bdash_mat(Jmat.astype(np.complex_),freq_gs,omega_e_sq.astype(np.complex_),kBT,inv_omega_g_sq.astype(np.complex_),time,is_QM)
	# successfully gotten auxillary matrices. First construct prefactor 
	# protect against small values of time, for which c and a diverge
	if time<0.00000001:
		prefac=1.0
	else:
		prefac=get_prefac(Bdash_mat)

	Binv=np.linalg.inv(Bmat)
	gammatrans=np.transpose(gamma.astype(np.complex_))
	temp=np.dot(Binv,gamma.astype(np.complex_))
	total_val=-0.25*time**2.0*np.dot(gammatrans,temp)

	chi_t=prefac*cmath.exp(total_val)
    
	return cmath.polar(chi_t)

def compute_ensemble_response(freq_gs,freq_ex,Jmat,Kmat,E_adiabatic,lambda_0,gamma,Omega_sq,omega_e_sq,omega_g_sq,kBT,steps,max_time,is_QM,is_emission,stdout):
	stdout.write('Constructing the ensemble response function for a GBOM: '+'\n')
	stdout.write('Calculating the response function for '+str(steps)+' time steps and a maximum time of '+str(max_time*const.fs_to_Ha)+'  fs'+'\n')
	chi=np.zeros((steps,3))
	response_func=np.zeros((steps,2),dtype=complex)
	step_length=max_time/steps
	eV_to_Ha=1.0/const.Ha_to_eV
	start_val=0.0000001
	counter=0
	stdout.write('\n'+'  Step       Time (fs)          Re[Chi]         Im[Chi]'+'\n')
	while counter<steps:
		current_t=start_val+step_length*counter
		chi[counter,0]=current_t
		if is_emission:
			# all variables coming into this routine (Jmat,Kmat etc) have already been adjusted to be correct. All we have to do is reverse time
			chi_val=calc_chi_for_given_time(freq_gs,freq_ex,Jmat,Kmat,lambda_0,gamma,Omega_sq,omega_e_sq,omega_g_sq,kBT,-current_t, is_QM)
		else:
			chi_val=calc_chi_for_given_time(freq_gs,freq_ex,Jmat,Kmat,lambda_0,gamma,Omega_sq,omega_e_sq,omega_g_sq,kBT,current_t, is_QM)
		chi[counter,1]=chi_val[0]
		chi[counter,2]=chi_val[1]
		counter=counter+1

	# now make sure the phase is a continuous function
	counter=0
	phase_fac=0.0
	while counter<steps-1:
		chi[counter,2]=chi[counter,2]+phase_fac
		if abs(chi[counter,2]-phase_fac-chi[counter+1,2])>0.7*math.pi: #check for discontinuous jump.
			diff=chi[counter+1,2]-(chi[counter,2]-phase_fac)
			frac=diff/math.pi
			n=int(round(frac))
			phase_fac=phase_fac-math.pi*n
		chi[steps-1,2]=chi[steps-1,2]+phase_fac
		counter=counter+1

	# now build total response function
	counter=0
	while counter<steps:
		response_func[counter,0]=chi[counter,0]
		# again, double check this for emission
		if is_emission:
			response_func[counter,1]=chi[counter,1]*cmath.exp(1j*chi[counter,2]-1j*chi[counter,0]*(E_adiabatic+lambda_0-0.5*(np.sum(freq_ex)-np.sum(freq_gs))))
		else:
			response_func[counter,1]=chi[counter,1]*cmath.exp(1j*chi[counter,2]-1j*chi[counter,0]*(E_adiabatic+lambda_0))

		stdout.write("%5d      %10.4f          %10.4e       %10.4e" % (counter+1,current_t*const.fs_to_Ha, np.real(response_func[counter,1]), np.imag(response_func[counter,1]))+'\n')
		counter=counter+1

	return response_func
