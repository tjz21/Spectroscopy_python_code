#! /usr/bin/env python
  
import os.path
import numpy as np
import math
import cmath
from scipy import integrate
from spec_pkg.constants import constants as const
from numba import jit, njit, prange


# Construct Dipole dipole fluctuation correlation function
# this is NOT a vector quantity, but the dipole flucts are vector quantities
def construct_corr_func_dipole(dipole_flucts,num_trajs,tau,time_step):
        corr_func=np.zeros(dipole_flucts.shape[0]*2-1)
        dim=dipole_flucts.shape[0]

        # need to fourier transform all trajs, then average, then multiply by a decaying exponential
        traj_count=0
        while traj_count<num_trajs:
                # extract the column belonging to a given trajectory 
                temp_array=np.zeros((dim,3))
                temp_array[:,0]=dipole_flucts[:,traj_count,0]  
                temp_array[:,1]=dipole_flucts[:,traj_count,1]
                temp_array[:,2]=dipole_flucts[:,traj_count,2]

                transform_time_x=np.correlate(temp_array[:,0],temp_array[:,0],mode='full')
                transform_time_y=np.correlate(temp_array[:,1],temp_array[:,1],mode='full')
                transform_time_z=np.correlate(temp_array[:,2],temp_array[:,2],mode='full')
	
                # add it to full corr func:
                counter=0
                while counter<transform_time_x.shape[-1]:
			# corr func is a scalar quantitiy. 
                        corr_func[counter]=transform_time_x[counter]+transform_time_y[counter]+transform_time_z[counter]
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
# The cross corrlation function IS a vector quantity
def construct_corr_func_cross(dipole_flucts,energy_flucts,num_trajs,tau,time_step):
        corr_func=np.zeros((dipole_flucts.shape[0]*2-1,3))
        dim=dipole_flucts.shape[0]

        # need to fourier transform all trajs, then average, then multiply by a decaying exponential
        traj_count=0
        while traj_count<num_trajs:
                # extract the column belonging to a given trajectory 
                temp_array=np.zeros((dim,3))
                temp_array2=np.zeros(dim)

                temp_array[:,0]=dipole_flucts[:,traj_count,0]
                temp_array[:,1]=dipole_flucts[:,traj_count,1]
                temp_array[:,2]=dipole_flucts[:,traj_count,2]
                temp_array2[:]=energy_flucts[:,traj_count]

                transform_time_x=np.correlate(temp_array[:,0],temp_array2,mode='full')
                transform_time_y=np.correlate(temp_array[:,1],temp_array2,mode='full')
                transform_time_z=np.correlate(temp_array[:,2],temp_array2,mode='full')

                # add it to full corr func:
                counter=0
                while counter<transform_time_x.shape[-1]:
                        corr_func[counter,0]=transform_time_x[counter]
                        corr_func[counter,1]=transform_time_y[counter]
                        corr_func[counter,2]=transform_time_z[counter]
                        counter=counter+1

                traj_count=traj_count+1

        # normalize and multiply by a decaying exponential
        eff_decay_length=tau/time_step
        current_index=-(corr_func.shape[0]-1)/2*1.0
        counter=0
        while counter<corr_func.shape[0]:
                corr_func[counter,:]=corr_func[counter,:]/num_trajs*math.exp(-abs(current_index)/eff_decay_length)/dim
                current_index=current_index+1.0
                counter=counter+1
        return corr_func

# cross correlation function is a vector quantity
# consequently dipole reorg is a vector quantity
def compute_dipole_reorg(corr_func_cross, kbT,sample_rate, time_step):
	# fourier transform correlation func and apply the harmonic prefactor. Watch for normalization factors in
        # the FFT and the prefactor. The FFT is only a sum over components. Does not account for the time step 
        corr_freq=np.zeros((corr_func_cross.shape[0],3),dtype=np.complex_)
	
        corr_freq[:,0]=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func_cross[:,0])))
        corr_freq[:,1]=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func_cross[:,1])))
        corr_freq[:,2]=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func_cross[:,2])))

        spectral_dens=np.zeros((int((corr_freq.shape[0]+1)/2),4))
        freqs=np.fft.fftshift(np.fft.fftfreq(corr_func_cross.shape[0],d=1.0/sample_rate))

        counter=0
        shift_index=corr_freq.shape[0]-spectral_dens.shape[0]
        while counter<spectral_dens.shape[0]:
                spectral_dens[counter,0]=freqs[counter+shift_index]
                spectral_dens[counter,1]=freqs[counter+shift_index]/(2.0*kbT)*corr_freq[counter+shift_index,0].real
                spectral_dens[counter,2]=freqs[counter+shift_index]/(2.0*kbT)*corr_freq[counter+shift_index,1].real
                spectral_dens[counter,3]=freqs[counter+shift_index]/(2.0*kbT)*corr_freq[counter+shift_index,2].real
                counter=counter+1

	# add the x, y and z components
        dipole_reorg=np.zeros(3)
        dipole_reorg[0]=1.0/math.pi*integrate.simps(spectral_dens[:,1],dx=(spectral_dens[1,0]-spectral_dens[0,0]))
        dipole_reorg[1]=1.0/math.pi*integrate.simps(spectral_dens[:,2],dx=(spectral_dens[1,0]-spectral_dens[0,0]))
        dipole_reorg[2]=1.0/math.pi*integrate.simps(spectral_dens[:,3],dx=(spectral_dens[1,0]-spectral_dens[0,0]))

        return dipole_reorg

def compute_corr_func_freq(corr_func,sample_rate,time_step):
        corr_freq=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corr_func)))

        full_corr_func_freq=np.zeros((corr_freq.shape[-1],2),dtype=np.complex_)
        freqs=np.fft.fftshift(np.fft.fftfreq(corr_func.size,d=1.0/sample_rate))
        for i in range(freqs.shape[-1]):
                full_corr_func_freq[i,0]=freqs[i]
                full_corr_func_freq[i,1]=corr_freq[i]

        return full_corr_func_freq

def compute_cross_corr_func_freq(cross_corr_func,sample_rate,time_step):
        cross_corr_func_freq=np.zeros((cross_corr_func.shape[0],3),dtype=np.complex_)  # vector quantity
        cross_corr_func_freq[:,0]=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cross_corr_func[:,0])))
        cross_corr_func_freq[:,1]=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cross_corr_func[:,1])))
        cross_corr_func_freq[:,2]=time_step*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cross_corr_func[:,2])))

        full_corr_func_freq=np.zeros((cross_corr_func_freq.shape[0],4),dtype=np.complex_)
        freqs=np.fft.fftshift(np.fft.fftfreq(cross_corr_func.shape[0],d=1.0/sample_rate))
        for i in range(freqs.shape[-1]):
                full_corr_func_freq[i,0]=freqs[i]
                full_corr_func_freq[i,1]=cross_corr_func_freq[i,0]
                full_corr_func_freq[i,2]=cross_corr_func_freq[i,1]
                full_corr_func_freq[i,3]=cross_corr_func_freq[i,2]

        return full_corr_func_freq

# remember, mu_av and mu_reonorm and mu reorg, as well as corr_func_cross_freq, are vector quantities
@jit(fastmath=True)
def HT_2nd_order_integrant(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,t):
	integrant=np.zeros((corr_func_freq.shape[0],corr_func_freq.shape[1]),dtype=np.complex_)
	tol=1.0e-15
	for i in range(corr_func_freq.shape[0]):
		integrant[i,0]=corr_func_freq[i,0]
		# check for omega=0 condition
		omega=integrant[i,0]
		if abs(omega)<tol:
			denom=mu_renorm**2.0*2.0*math.pi
			num=2.0*(mu_reorg[0]-mu_av[0])*t*corr_func_cross_freq[i,1]+2.0*(mu_reorg[1]-mu_av[1])*t*corr_func_cross_freq[i,2]+2.0*(mu_reorg[2]-mu_av[2])*t*corr_func_cross_freq[i,3]+corr_func_freq[i,1]
			integrant[i,1]=num/denom

		else:
			denom=kBT*mu_renorm**2.0*2.0*math.pi*(1.0-np.exp(-omega/kBT))
			num=2.0*((mu_reorg[0]-mu_av[0])*corr_func_cross_freq[i,1]+(mu_reorg[1]-mu_av[1])*corr_func_cross_freq[i,2]+(mu_reorg[2]-mu_av[2])*corr_func_cross_freq[i,3])*(1.0-cmath.exp(-1j*omega*t))+corr_func_freq[i,1]*omega*np.exp(-1j*omega*t)
			integrant[i,1]=num/denom

	return integrant

# The full A func. NOTE THAT MU RENORM IS A SCALAR QUANTITY, NOT A VECTOR QUANTITY
def compute_HT_term_2nd_order(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,max_t,steps):
	Afunc=np.zeros((steps,2),dtype=np.complex_)
	step_length=max_t/steps
	for i in range(steps):
		t_current=i*step_length
		Afunc[i,0]=t_current
		integrant=HT_2nd_order_integrant(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,t_current)
		Afunc[i,1]=1.0+integrate.simps(integrant[:,1],dx=(integrant[1,0]-integrant[0,0]))

	Afunc[:,1]=Afunc[:,1]*mu_renorm**2.0

	return Afunc
