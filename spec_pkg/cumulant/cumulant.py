#! /usr/bin/env python

import os.path
import numpy as np
import math
import cmath
from scipy import integrate
from spec_pkg.constants import constants as const
from spec_pkg.GBOM import gbom_cumulant_response as gbom_cumul
from numba import jit, njit, prange

# 2D version of the simpson integral. Parallelized accross cores
@jit(fastmath=True)
def simpson_integral_2D(integrant):
	limit1=integrant.shape[0]
	limit2=integrant.shape[1]
	result=0.0

	# make sure we have an even number of points in each dimension
	if limit1%2>0.5:
		limit1=limit1-1
	if limit2%2>0.5:
		limit2=limit2-1

	step_x=integrant[1,0,0]-integrant[0,0,0]
	step_y=integrant[0,1,1]-integrant[0,0,1]

	for icount in range(limit1):
		if icount==0 or icount==limit1-1:
			prefac_x=1.0
		elif icount%2<0.5:
			prefac_x=2.0
		else:
			prefac_x=4.0

		for jcount in range(limit2):

			if jcount==0 or jcount==limit1-1:
				prefac_y=1.0
			elif jcount%2<0.5:
				prefac_y=2.0
			else:
				prefac_y=4.0

			result+=prefac_x*prefac_y*integrant[icount,jcount,2]
	return step_x*step_y*result/(9.0)

# construct the effective quantum correlation function in frequency space using the Jung prefactor.
# This is necessary for 2DES calculations.
def construct_corr_func_3rd_qm_freq(corr_func,kbT,sampling_rate_in_fs,low_freq_filter):
	step_length_corr=corr_func[1,1,0]-corr_func[0,0,0]
	sample_rate=sampling_rate_in_fs*math.pi*2.0*const.hbar_in_eVfs

	# pad corr func with zeros to have a finer resolution in the frequency range. Double the range, see if results change
	new_dim=corr_func.shape[0]*2+1  # pad with a factor of 2
	padding=np.array([new_dim,new_dim])

	# try padding:
	extended_func=np.zeros((new_dim,new_dim),dtype=complex)

	# pad with zeros:
	start_index=(new_dim-corr_func.shape[0])/2
	end_index=start_index+corr_func.shape[0]	
	icount=0
	while icount<new_dim:
		jcount=0
		while jcount<new_dim:
			if icount<start_index or icount>end_index-1 or jcount<start_index or jcount>end_index-1:
				extended_func[icount,jcount]=0.0+0.0j
			else:
				extended_func[icount,jcount]=corr_func[icount-start_index,jcount-start_index,2]
			jcount=jcount+1
		icount=icount+1

	# remember: Need to normalize correlation func in frequency domain by multiplying by dt**2.0
	corr_func_freq=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func[:,:]))))*step_length_corr*step_length_corr

	freq_list=np.fft.fftshift(np.fft.fftfreq(corr_func_freq.shape[0],d=1./sample_rate))/const.Ha_to_eV

	eff_corr=np.zeros((corr_func_freq.shape[0],corr_func_freq.shape[0],3))
	# loop over corr func and filter out imaginary part. This should be zero by the symmetries of the correlation function, but is not quite
	icount=0
	while icount<corr_func_freq.shape[0]:
		jcount=0
		while jcount<corr_func_freq.shape[0]:
			eff_corr[icount,jcount,0]=freq_list[icount]
			eff_corr[icount,jcount,1]=freq_list[jcount]
			prefac=prefactor_jung(freq_list[icount],freq_list[jcount],kbT)
			eff_corr[icount,jcount,2]=(prefac*corr_func_freq[icount,jcount]).real
			# now apply low freq filter:
			if abs(freq_list[icount])<low_freq_filter and abs(freq_list[jcount])<low_freq_filter:
				eff_corr[icount,jcount,2]=0.0
			jcount=jcount+1
		icount=icount+1

	return eff_corr

# construct a quantum 3rd order correlation function  in time domain using the Jung prefactor
def construct_corr_func_3rd_qm(corr_func,kbT,sampling_rate_in_fs,low_freq_filter):
        step_length_corr=corr_func[1,1,0]-corr_func[0,0,0]
        sample_rate=sampling_rate_in_fs*math.pi*2.0*const.hbar_in_eVfs

        # no padding:
        extended_func=corr_func[:,:,2]

        # remember: Need to normalize correlation func in frequency domain by multiplying by dt**2.0. Not necessary here because we do an inverse FFt
        corr_func_freq=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func[:,:]))))

        freq_list=np.fft.fftshift(np.fft.fftfreq(corr_func_freq.shape[0],d=1./sample_rate))/const.Ha_to_eV

        eff_corr=np.zeros((corr_func_freq.shape[0],corr_func_freq.shape[0]))
        # loop over corr func and filter out imaginary part. This should be zero by the symmetries of the correlation function, but is not quite
        icount=0
        while icount<corr_func_freq.shape[0]:
                jcount=0
                while jcount<corr_func_freq.shape[0]:
                        prefac=prefactor_jung(freq_list[icount],freq_list[jcount],kbT)
                        eff_corr[icount,jcount]=(prefac*corr_func_freq[icount,jcount]).real
                        # now apply low freq filter:
                        if abs(freq_list[icount])<low_freq_filter and abs(freq_list[jcount])<low_freq_filter:
                                eff_corr[icount,jcount,2]=0.0
                        jcount=jcount+1
                icount=icount+1

        corr_func_qm=(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(eff_corr))))


        new_corr_func=np.zeros((corr_func.shape[0],corr_func.shape[0],3),dtype=complex)
	
        new_corr_func[:,:,0]=corr_func[:,:,0]
        new_corr_func[:,:,1]=corr_func[:,:,1]
        new_corr_func[:,:,2]=corr_func_qm[:,:]

        return new_corr_func


# function constructing the complete 3rd order lineshape function from a correlation function
def compute_lineshape_func_3rd(corr_func,kbT,sampling_rate_in_fs,max_t,steps,low_freq_filter,g3_cutoff,stdout):
	stdout.write('\n'+"Computing third order cumulant lineshape function."+'\n')
	if low_freq_filter>0.0:
		 stdout.write("Applying a low pass frequency filter of "+str(low_freq_filter*const.Ha_to_cm)+' cm^(-1)'+'\n')
	g_func=np.zeros((steps,2),dtype=np.complex)
	# need to constuct 3rd order correlation function in the frequency domain. 
	step_length_corr=corr_func[1,1,0]-corr_func[0,0,0]
 
	sample_rate=sampling_rate_in_fs*math.pi*2.0*const.hbar_in_eVfs

	# pad corr func with zeros to have a finer resolution in the frequency range. Double the range, see if results change
	new_dim=corr_func.shape[0]*2+1  # pad with a factor of 2 . Resulting value is guaranteed to be odd
	padding=np.array([new_dim,new_dim])

	# try padding:
	extended_func=np.zeros((new_dim,new_dim),dtype=np.complex)

	# pad with zeros:
	start_index=int((new_dim-corr_func.shape[0])/2)
	end_index=start_index+corr_func.shape[0]

	icount=0
	while icount<new_dim:
		jcount=0
		while jcount<new_dim:
			if icount<start_index or icount>end_index-1 or jcount<start_index or jcount>end_index-1:
				extended_func[icount,jcount]=0.0+0.0j
			else:
				extended_func[icount,jcount]=corr_func[icount-start_index,jcount-start_index,2]
			jcount=jcount+1
		icount=icount+1

	# remember: Need to normalize correlation func in frequency domain by multiplying by dt**2.0
	corr_func_freq=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func[:,:]))))*step_length_corr*step_length_corr

	freq_list=np.fft.fftshift(np.fft.fftfreq(corr_func_freq.shape[0],d=1./sample_rate))/const.Ha_to_eV

	# loop over corr func and filter out imaginary part. This should be zero by the symmetries of the correlation function, but is not quite
	icount=0
	while icount<corr_func_freq.shape[0]:
		jcount=0
		while jcount<corr_func_freq.shape[0]:
			corr_func_freq[icount,jcount]=corr_func_freq[icount,jcount].real
			# now apply low freq filter:
			if abs(freq_list[icount])<low_freq_filter and abs(freq_list[jcount])<low_freq_filter:
				corr_func_freq[icount,jcount]=0.0
		
			jcount=jcount+1
		icount=icount+1


	stdout.write('\n'+'  Step       Time (fs)          Re[g_3]         Im[g_3]'+'\n')
	step_length=max_t/steps
	counter=0
	while counter<steps:
		t_current=counter*step_length
		g_func[counter,0]=t_current
		integrant=integrant_3rd_order_cumulant_lineshape(corr_func_freq,freq_list,t_current,kbT)
		# if Necessary, apply a cutoff to the 3rd order correlation function to tame 
		# long timescale divergences
		if g3_cutoff>0.0:
			g_func[counter,1]=np.exp(-t_current/g3_cutoff)*simpson_integral_2D(integrant)
		else:
			g_func[counter,1]=simpson_integral_2D(integrant)   #  give x and y axis
		stdout.write("%5d      %10.4f          %10.4e           %10.4e" % (counter,t_current*const.fs_to_Ha, np.real(g_func[counter,1]), np.imag(g_func[counter,1]))+'\n')
		counter=counter+1
	return g_func


def compute_h1_func(qm_corr_func_freq,max_t,steps):
	step_length=max_t/steps
	h1_func=np.zeros((steps,steps,3),dtype=complex)
	icount=0
	print('Computing H1')
	while icount<steps:
		jcount=0
		print(icount)
		while jcount<steps:
			t1=icount*step_length
			t2=jcount*step_length
			h1_func[icount,jcount,0]=t1
			h1_func[icount,jcount,1]=t2
			integrant=integrant_h1(qm_corr_func_freq,t1,t2)
			h1_func[icount,jcount,2]=simpson_integral_2D(integrant)
			jcount=jcount+1
		icount=icount+1

	return h1_func

def compute_h2_func(qm_corr_func_freq,max_t,steps):
	step_length=max_t/steps
	h2_func=np.zeros((steps,steps,3),dtype=complex)
	icount=0
	print('Computing H2')
	while icount<steps:
		jcount=0
		print(icount)
		while jcount<steps:
			t1=icount*step_length
			t2=jcount*step_length
			h2_func[icount,jcount,0]=t1
			h2_func[icount,jcount,1]=t2
			integrant=integrant_h2(qm_corr_func_freq,t1,t2)
			h2_func[icount,jcount,2]=simpson_integral_2D(integrant)
			jcount=jcount+1
		icount=icount+1

	return h2_func

def compute_h4_func(qm_corr_func_freq,max_t,steps):
	step_length=max_t/steps	
	h4_func=np.zeros((steps,steps,3),dtype=complex)
	icount=0
	print('Computing H4')
	while icount<steps:
		jcount=0
		print(icount)
		while jcount<steps:
			t1=icount*step_length
			t2=jcount*step_length
			h4_func[icount,jcount,0]=t1
			h4_func[icount,jcount,1]=t2
			integrant=integrant_h4(qm_corr_func_freq,t1,t2)
			h4_func[icount,jcount,2]=simpson_integral_2D(integrant)
			jcount=jcount+1
		icount=icount+1

	return h4_func


def compute_h5_func(qm_corr_func_freq,max_t,steps):
	step_length=max_t/steps
	h5_func=np.zeros((steps,steps,3),dtype=complex)
	icount=0
	print('Computing H5')
	while icount<steps:
		jcount=0
		print(icount)
		while jcount<steps:
			t1=icount*step_length
			t2=jcount*step_length
			h5_func[icount,jcount,0]=t1
			h5_func[icount,jcount,1]=t2
			integrant=integrant_h5(qm_corr_func_freq,t1,t2)
			h5_func[icount,jcount,2]=simpson_integral_2D(integrant)
			jcount=jcount+1
		icount=icount+1

	return h5_func

# compute h3 for just a single set of t1,t2,t3 values. This is done because h3 is a function 
# with 3 indices and is too large to be stored, but rather has to be computed on the fly.
@jit
def compute_h3_val(qm_corr_func_freq,t1,t2,t3):
	integrant=integrant_h3(qm_corr_func_freq,t1,t2,t3)
	return simpson_integral_2D(integrant)

# the analogous routine to the second order lineshape integrant. We have to carefully treat the limits of the integrant
# when omega1 or omega2=0, or when omega1=-omega2. Double checked all limits. They seem to be correct.   
@jit(fastmath=True)
def integrant_3rd_order_cumulant_lineshape(corr_func_freq,freq_list,t_val,kbT):
	tol=10e-15
	integrant=np.zeros((corr_func_freq.shape[0],corr_func_freq.shape[0],3),dtype=np.complex_)
	for icount in range(corr_func_freq.shape[0]):
		for jcount in range(corr_func_freq.shape[0]):
			omega1=freq_list[icount]
			omega2=freq_list[jcount]
			omega_bar=omega1+omega2
			integrant[icount,jcount,0]=omega1
			integrant[icount,jcount,1]=omega2

			if abs(omega1)<tol and abs(omega2)<tol:
				integrant[icount,jcount,2]=1j*1.0/3.0*corr_func_freq[icount,jcount]*t_val**3.0/(8.0*math.pi**2.0)
			elif abs(omega_bar)<tol:
				num=1j*(-omega1**2.0*t_val**2.0/2.0+1.0-cmath.exp(1j*omega1*t_val)+1j*omega1*t_val)
				denom=omega1*(-omega2/kbT-math.exp(-omega2/kbT)+1.0)  # i think this is right...
				integrant[icount,jcount,2]=1j*corr_func_freq[icount,jcount]*num/(8.0*math.pi**2.0*kbT**2.0*denom)
			elif abs(omega1)<tol:
				num=1j*(2.0*(cmath.exp(-1j*omega2*t_val)-1.0)+1j*omega2*t_val*(cmath.exp(-1j*omega2*t_val)+1.0))
				denom=omega2*(1.0-omega2/kbT*math.exp(-omega2/kbT)-math.exp(-omega2/kbT))	
				integrant[icount,jcount,2]=1j*corr_func_freq[icount,jcount]*num/(8.0*math.pi**2.0*kbT**2.0*denom)
			elif abs(omega2)<tol:
				num=1j*(-omega1**2.0*t_val**2.0/2.0+1.0-1j*omega1*t_val-cmath.exp(-1j*omega1*t_val))
				denom=omega1*(omega1/kbT+math.exp(-omega1/kbT)-1.0)
				integrant[icount,jcount,2]=1j*corr_func_freq[icount,jcount]*num/(8.0*math.pi**2.0*kbT**2.0*denom)
			else:
				num=1j*(cmath.exp(-1j*omega2*t_val)-1.0+omega2/omega_bar*(1.0-cmath.exp(-1j*omega_bar*t_val))+omega1/omega2*(cmath.exp(-1j*omega2*t_val)-1.0+1j*omega2*t_val))
				denom=omega2*math.exp(-omega_bar/kbT)-omega_bar*math.exp(-omega2/kbT)+omega1
				integrant[icount,jcount,2]=1j*corr_func_freq[icount,jcount]/(8.0*math.pi**2.0*kbT**2.0)*num/denom

			jcount=jcount+1
		icount=icount+1

	return integrant

# function evaluates the integrant for the auxilliary term h1 needed when constructing the 3rd order cumulant correction
# to the 3rd order response function. Note that this function takes as an input the approximately reconstructed QUANTUM
# correlation function ins frequency space. 
@jit(fastmath=True, parallel=True)
def integrant_h1(qm_corr_func_freq,t1,t2):
	tol=10e-15
	integrant=np.zeros((qm_corr_func_freq.shape[0],qm_corr_func_freq.shape[0],3),dtype=complex)
	icount=0
	while icount<qm_corr_func_freq.shape[0]:
		jcount=0
		while jcount<qm_corr_func_freq.shape[0]:
			omega1=qm_corr_func_freq[icount,jcount,0]
			omega2=qm_corr_func_freq[icount,jcount,1]
			integrant[icount,jcount,0]=omega1
			integrant[icount,jcount,1]=omega2

			# kbT shouldnt be necessary: dummy variable
			integrant[icount,jcount,2]=gbom_cumul.prefactor_2DES_h1_QM(qm_corr_func_freq[icount,jcount,2],omega1,omega2,0.0,t1,t2)

			jcount=jcount+1
		icount=icount+1

	return integrant

# function evaluates the integrant for the auxilliary term h2 needed when constructing the 3rd order cumulant correction
# to the 3rd order response function. Note that this function takes as an input the approximately reconstructed QUANTUM
# correlation function ins frequency space. 
@jit(fastmath=True, parallel=True)
def integrant_h2(qm_corr_func_freq,t1,t2):
	tol=10e-15
	integrant=np.zeros((qm_corr_func_freq.shape[0],qm_corr_func_freq.shape[0],3),dtype=complex)
	icount=0
	while icount<qm_corr_func_freq.shape[0]:
		jcount=0
		while jcount<qm_corr_func_freq.shape[0]:
			omega1=qm_corr_func_freq[icount,jcount,0]
			omega2=qm_corr_func_freq[icount,jcount,1]
			integrant[icount,jcount,0]=omega1
			integrant[icount,jcount,1]=omega2

 
			integrant[icount,jcount,2]=gbom_cumul.prefactor_2DES_h2_QM(qm_corr_func_freq[icount,jcount,2],omega1,omega2,0.0,t1,t2)

			jcount=jcount+1
		icount=icount+1

	return integrant


@jit(fastmath=True, parallel=True)
def integrant_h4(qm_corr_func_freq,t1,t2):
	tol=10e-15
	integrant=np.zeros((qm_corr_func_freq.shape[0],qm_corr_func_freq.shape[0],3),dtype=complex)
	icount=0
	while icount<qm_corr_func_freq.shape[0]:
		jcount=0
		while jcount<qm_corr_func_freq.shape[0]:
			omega1=qm_corr_func_freq[icount,jcount,0]
			omega2=qm_corr_func_freq[icount,jcount,1]
			integrant[icount,jcount,0]=omega1
			integrant[icount,jcount,1]=omega2


			integrant[icount,jcount,2]=gbom_cumul.prefactor_2DES_h4_QM(qm_corr_func_freq[icount,jcount,2],omega1,omega2,0.0,t1,t2)

			jcount=jcount+1
		icount=icount+1

	return integrant


@jit(fastmath=True, parallel=True)
def integrant_h5(qm_corr_func_freq,t1,t2):
	tol=10e-15
	integrant=np.zeros((qm_corr_func_freq.shape[0],qm_corr_func_freq.shape[0],3),dtype=complex)
	icount=0
	while icount<qm_corr_func_freq.shape[0]:
		jcount=0
		while jcount<qm_corr_func_freq.shape[0]:
			omega1=qm_corr_func_freq[icount,jcount,0]
			omega2=qm_corr_func_freq[icount,jcount,1]
			integrant[icount,jcount,0]=omega1	
			integrant[icount,jcount,1]=omega2
			integrant[icount,jcount,2]=gbom_cumul.prefactor_2DES_h5_QM(qm_corr_func_freq[icount,jcount,2],omega1,omega2,0.0,t1,t2)

			jcount=jcount+1
		icount=icount+1
	return integrant


# function evaluates the integrant for the auxilliary term h3 needed when constructing the 3rd order cumulant correction
# to the 3rd order response function. Note that this function takes as an input the approximately reconstructed QUANTUM
# correlation function ins frequency space. 
@jit(fastmath=True, parallel=True)
def integrant_h3(qm_corr_func_freq,t1,t2,t3):
	tol=10e-15
	integrant=np.zeros((qm_corr_func_freq.shape[0],qm_corr_func_freq.shape[0],3),dtype=complex)
	icount=0
	while icount<qm_corr_func_freq.shape[0]:
		jcount=0	
		while jcount<qm_corr_func_freq.shape[0]:
			omega1=qm_corr_func_freq[icount,jcount,0]
			omega2=qm_corr_func_freq[icount,jcount,1]

			integrant[icount,jcount,0]=omega1
			integrant[icount,jcount,1]=omega2
			integrant[icount,jcount,2]=gbom_cumul.prefactor_2DES_h3_QM(qm_corr_func_freq[icount,jcount,2],omega1,omega2,0.0,t1,t2,t3)

			jcount=jcount+1
		icount=icount+1

	return integrant

# prefactor defined by Jung that transforms the classical correlation function in Frequency space to its QM counterpart
@njit(fastmath=True, parallel=True)
def prefactor_jung(omega1,omega2,kbT):
	omega_bar=omega1+omega2
	tol=10e-15
	prefac=0.0+1j*0.0    

	if abs(omega1)<tol and abs(omega2)<tol:
		prefac=1.0
	elif abs(omega_bar)<tol:
		prefac=omega1**2.0/(2.0*kbT**2.0*(1.0+omega1/kbT-np.exp(omega1/kbT)))
	elif abs(omega1)<tol:
		prefac=omega2**2.0/(2.0*kbT**2.0*(1.0-np.exp(-omega2/kbT)-omega2/kbT*np.exp(-omega2/kbT)))
	elif abs(omega2)<tol:
		prefac=omega1**2.0/(2.0*kbT**2.0*(np.exp(-omega1/kbT)+omega1/kbT-1.0))
	else:
		prefac=omega_bar*omega1*omega2/(2.0*kbT**2.0*(omega2*np.exp(-omega_bar/kbT)-omega_bar*np.exp(-omega2/kbT)+omega1))

	return prefac

# This function constructs the 2D classical correlation function in time domain from a single trajectory.
#@jit(fastmath=True)
def construct_classical_3rd_order_corr_from_single_traj(fluctuations,correlation_length,time_step):
	func_size=correlation_length*2+1
	corr_func=np.zeros((func_size,func_size,3))      
	for icount in range(corr_func.shape[0]):
		print(icount)
		for jcount in range(icount,corr_func.shape[0]):
			integrant=get_correlation_integrant_3rd(fluctuations,icount,jcount,correlation_length,time_step)
			corr_func[icount,jcount,0]=(icount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
			corr_func[icount,jcount,1]=(jcount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
			corr_func[icount,jcount,2]=integrate.simps(integrant[:,1],dx=time_step)/(1.0*integrant.shape[0])
			# exploit symmetry of classical correlation function under exchange of w and w'
			corr_func[jcount,icount,0]=corr_func[icount,jcount,1]
			corr_func[jcount,icount,1]=corr_func[icount,jcount,0]
			corr_func[jcount,icount,2]=corr_func[icount,jcount,2]

	# make sure that the step length does not get multiplied in the trapezium integral function
	for icount in range(corr_func.shape[0]):
		for jcount in range(corr_func.shape[0]):
			corr_func[icount,jcount,2]=corr_func[icount,jcount,2]/(time_step)

	return corr_func

# construct the full classical 3rd order correlation function by making use of fast Fourier transforms. 
def construct_corr_func_3rd(fluctuations,num_trajs,correlation_length,tau,time_step,stdout):
	stdout.write('\n'+"Constructing classical 3rd order correlation function from MD trajectory."+'\n')
	classical_corr=np.zeros((correlation_length*2+1,correlation_length*2+1,3))

	traj_counter=0
	while traj_counter<num_trajs:
		# get classical correlation function:
		stdout.write('Processing Trajectory '+str(traj_counter+1)+' of '+str(num_trajs))
		temp_corr=construct_classical_3rd_order_corr_from_single_traj(fluctuations[:,traj_counter],correlation_length,time_step)
		icount=0
		while icount<temp_corr.shape[0]:
			jcount=0
			while jcount<temp_corr.shape[0]:
				classical_corr[icount,jcount,0]=temp_corr[icount,jcount,0]
				classical_corr[icount,jcount,1]=temp_corr[icount,jcount,1]
				classical_corr[icount,jcount,2]=classical_corr[icount,jcount,2]+temp_corr[icount,jcount,2]
				jcount=jcount+1
			icount=icount+1
		traj_counter=traj_counter+1

	# have summed over all trajectories. Now average and multiply by decaying exponential.
	icount=0
	while icount<temp_corr.shape[0]:
		jcount=0
		while jcount<temp_corr.shape[0]:
			classical_corr[icount,jcount,2]=classical_corr[icount,jcount,2]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
			jcount=jcount+1	
		icount=icount+1

	return classical_corr

@jit(fastmath=True)
def get_correlation_integrant_3rd(fluctuations,current_corr_i,current_corr_j,correlation_length,MD_steplength):
	relative_start_i=current_corr_i-correlation_length   # these can be negative or positive and define a range
	relative_start_j=current_corr_j-correlation_length   # over which the correlation function gets calculated
#	print(current_corr_i,current_corr_j,correlation_length,relative_start_i,relative_start_j)
	min_index=min(relative_start_i,relative_start_j)
	max_index=max(relative_start_i,relative_start_j)
	if max_index<0:
		max_index=0
	if min_index>0:
 		min_index=0
	# counter must be choosen in such a way that 
#	print(max_index,min_index)
#
	# set maximum size that the integrant can have. This is chosen such that counter+min_index and counter+max_index
	# are guaranteed to be within the range of the integrant
	integrant_size=fluctuations.shape[0]-(abs(min_index)+max_index+2)
	integrant=np.zeros((integrant_size,2))
	counter=abs(min_index)
	while counter<integrant.shape[0]:
		integrant[counter,0]=(counter*MD_steplength)
		integrant[counter,1]=(fluctuations[counter])*(fluctuations[counter+relative_start_i])*(fluctuations[counter+relative_start_j])
		counter=counter+1
	return integrant

#ALTERNATIVE
#@jit(fastmath=True,parallel=True)
#def get_correlation_integrant_3rd(fluctuations,current_corr_i,current_corr_j,correlation_length,MD_steplength):
#       max_index_i=fluctuations.shape[0]-current_corr_i
#       max_index_j=fluctuations.shape[0]-current_corr_j
#       max_index=min(max_index_i,max_index_j)
#
#       integrant=np.zeros((max_index,2))
#       counter=0
#       while counter<integrant.shape[0]:
#               integrant[counter,0]=(counter*MD_steplength)
#               integrant[counter,1]=(fluctuations[counter])*(fluctuations[counter+current_corr_i])*(fluctuations[counter+current_corr_j])
#               counter=counter+1
#       return integrant


def construct_corr_func(fluctuations,num_trajs,tau,time_step):
	corr_func=np.zeros(fluctuations.shape[0]*2-1)
	dim=fluctuations.shape[0]

	# need to fourier transform all trajs, then average, then multiply by a decaying exponential
	traj_count=0
	while traj_count<num_trajs:
		# extract the column belonging to a given trajectory 
		temp_array=np.zeros(dim)
		counter=0	
		while counter<dim:
			temp_array[counter]=fluctuations[counter,traj_count]
			counter=counter+1

		transform_time=np.correlate(temp_array,temp_array,mode='full')

		# add it to full corr func:
		counter=0
		while counter<transform_time.shape[-1]:
			corr_func[counter]=corr_func[counter]+transform_time[counter]
			counter=counter+1

		traj_count=traj_count+1

 	# normalize and multiply by a decaying exponential
	eff_decay_length=tau/time_step
	current_index=-(corr_func.shape[-1]-1)/2*1.0
	counter=0
	while counter<corr_func.shape[-1]:
		corr_func[counter]=corr_func[counter]/num_trajs*math.exp(-abs(current_index)/eff_decay_length)/dim
		current_index=current_index+1.0
		counter=counter+1
	return corr_func

def calc_2nd_order_cumulant_divergence(corr_func,omega_step,time_step):
	integral=integrate.simps(corr_func,dx=time_step)
	return 	(1.0/(2.0*math.pi))*integral*omega_step

def compute_spectral_dens(corr_func,kbT, sample_rate,time_step):
	# fourier transform correlation func and apply the harmonic prefactor. Watch for normalization factors in
	# the FFT and the prefactor. The FFT is only a sum over components. Does not account for the time step 
	corr_freq=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func)))

	spectral_dens=np.zeros((int((corr_freq.shape[-1]+1)/2),2))
	freqs=np.fft.fftshift(np.fft.fftfreq(corr_func.size,d=1.0/sample_rate))

	counter=0
	shift_index=corr_freq.shape[-1]-spectral_dens.shape[0]
	while counter<spectral_dens.shape[0]:
		spectral_dens[counter,0]=freqs[counter+shift_index]
		spectral_dens[counter,1]=freqs[counter+shift_index]/(2.0*kbT)*corr_freq[counter+shift_index].real
		counter=counter+1

	return spectral_dens

# define the maximum number of t points this should be calculated for and the maximum number of steps
def compute_2nd_order_cumulant_from_spectral_dens(spectral_dens,kbT,max_t,steps,stdout):
	q_func=np.zeros((steps,2),dtype=complex)
	stdout.write('\n'+"Computing second order cumulant lineshape function."+'\n')
	stdout.write('\n'+'  Step       Time (fs)          Re[g_2]         Im[g_2]'+'\n')
	step_length=max_t/steps
	step_length_omega=spectral_dens[1,0]-spectral_dens[0,0]
	counter=0
	while counter<steps:
		t_current=counter*step_length
		q_func[counter,0]=t_current
		integrant=integrant_2nd_order_cumulant_lineshape(spectral_dens,t_current,kbT)
		q_func[counter,1]=integrate.simps(integrant[:,1],dx=(integrant[1,0]-integrant[0,0]))   #  give x and y axis
		stdout.write("%5d      %10.4f          %10.4e           %10.4e" % (counter,t_current*const.fs_to_Ha, np.real(q_func[counter,1]), np.imag(q_func[counter,1]))+'\n')
		counter=counter+1
	return q_func

# fix limit of x-->0, Sign in imaginary term?
@jit(fastmath=True)
def integrant_2nd_order_cumulant_lineshape(spectral_dens,t_val,kbT):
	integrant=np.zeros((spectral_dens.shape[0],spectral_dens.shape[1]),dtype=np.complex_)
	for counter in range(spectral_dens.shape[0]):
		omega=spectral_dens[counter,0]
		integrant[counter,0]=omega
		if counter==0:
			integrant[counter,1]=0.0
		else:
			integrant[counter,1]=1.0/math.pi*spectral_dens[counter,1]/(omega**2.0)*(2.0*cmath.cosh(omega/(2.0*kbT))/cmath.sinh(omega/(2.0*kbT))*(math.sin(omega*t_val/2.0))**2.0+1j*(math.sin(omega*t_val)-omega*t_val))

	return integrant

