#! /usr/bin/env python
  
import os.path
import numpy as np
import math
import cmath
from scipy import integrate
from spec_pkg.constants import constants as const
from numba import jit, njit, prange


# Construct Dipole dipole fluctuation correlation function
def construct_corr_func_dipole(dipole_flucts,num_trajs,tau,time_step):
        corr_func=np.zeros(dipole_flucts.shape[0]*2-1)
        dim=dipole_flucts.shape[0]

        # need to fourier transform all trajs, then average, then multiply by a decaying exponential
        traj_count=0
        while traj_count<num_trajs:
                # extract the column belonging to a given trajectory 
                temp_array=np.zeros(dim)
                counter=0
                while counter<dim:
                        temp_array[counter]=dipole_flucts[counter,traj_count]
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

# Construct Dipole-energy gap fluctuation correlation function
def construct_corr_func_cross(dipole_flucts,energy_flucts,num_trajs,tau,time_step):
        corr_func=np.zeros(dipole_flucts.shape[0]*2-1)
        dim=dipole_flucts.shape[0]

        # need to fourier transform all trajs, then average, then multiply by a decaying exponential
        traj_count=0
        while traj_count<num_trajs:
                # extract the column belonging to a given trajectory 
                temp_array=np.zeros(dim)
                temp_array2=np.zeros(dim)
                counter=0
                while counter<dim:
                        temp_array[counter]=dipole_flucts[counter,traj_count]
                        temp_array2[counter]=energy_flucts[counter,traj_count]
                        counter=counter+1

                transform_time=np.correlate(temp_array,temp_array2,mode='full')

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

def compute_dipole_reorg(corr_func_cross, kbT,sample_rate, time_step):
	# fourier transform correlation func and apply the harmonic prefactor. Watch for normalization factors in
        # the FFT and the prefactor. The FFT is only a sum over components. Does not account for the time step 
        corr_freq=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func_cross)))

        spectral_dens=np.zeros((int((corr_freq.shape[-1]+1)/2),2))
        freqs=np.fft.fftshift(np.fft.fftfreq(corr_func_cross.size,d=1.0/sample_rate))

        counter=0
        shift_index=corr_freq.shape[-1]-spectral_dens.shape[0]
        while counter<spectral_dens.shape[0]:
                spectral_dens[counter,0]=freqs[counter+shift_index]
                spectral_dens[counter,1]=freqs[counter+shift_index]/(2.0*kbT)*corr_freq[counter+shift_index].real
                counter=counter+1

        dipole_reorg=1.0/math.pi*integrate.simps(spectral_dens[:,1],dx=(spectral_dens[1,0]-spectral_dens[0,0]))

        return dipole_reorg

def compute_corr_func_freq(corr_func,sample_rate,time_step):
        corr_freq=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func)))

        full_corr_func_freq=np.zeros((corr_freq.shape[-1],2),dtype=np.complex_)
        freqs=np.fft.fftshift(np.fft.fftfreq(corr_func.size,d=1.0/sample_rate))
        for i in range(freqs.shape[-1]):
                full_corr_func_freq[i,0]=freqs[i]
                full_corr_func_freq[i,1]=corr_freq[i]

        return full_corr_func_freq

@jit(fastmath=True)
def HT_2nd_order_integrant(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,t):
	integrant=np.zeros((corr_func_freq.shape[0],2),dtype=np.complex_)
	tol=1.0e-15
	for i in range(corr_func_freq.shape[0]):
		integrant[i,0]=corr_func_freq[i,0]
		# check for omega=0 condition
		omega=integrant[i,0]
		if abs(omega)<tol:
			denom=mu_renorm**2.0*2.0*math.pi
			num=2.0*(mu_reorg-mu_av)*t*corr_func_cross_freq[i,1]+corr_func_freq[i,1]
			integrant[i,1]=num/denom

		else:
			denom=kBT*mu_renorm**2.0*2.0*math.pi*(1.0-np.exp(-omega/kBT))
			num=2.0*(mu_reorg-mu_av)*corr_func_cross_freq[i,1]*(1.0-cmath.exp(-1j*omega*t))+corr_func_freq[i,1]*omega*np.exp(-1j*omega*t)
			integrant[i,1]=num/denom

	return integrant


def compute_HT_term_2nd_order(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,max_t,steps):
	Afunc=np.zeros((steps,2),dtype=np.complex_)
	step_length=max_t/steps
	for i in range(steps):
		t_current=i*step_length
		Afunc[i,0]=t_current
		integrant=HT_2nd_order_integrant(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,t_current)
		Afunc[i,1]=1.0+integrate.simps(integrant[:,1],dx=(integrant[1,0]-integrant[0,0]))

	return Afunc
