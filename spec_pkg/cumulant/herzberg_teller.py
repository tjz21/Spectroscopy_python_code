#! /usr/bin/env python
  
import os.path
import numpy as np
import math
import cmath
from scipy import integrate
from spec_pkg.constants import constants as const
from numba import jit, njit, prange
from spec_pkg.cumulant import cumulant as cumulant

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
                        corr_func[counter]=corr_func[counter]+transform_time_x[counter]+transform_time_y[counter]+transform_time_z[counter]
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

# This correlation function is a scalar as it contains two instances of mu
def construct_corr_func_3rd_mu_U_mu(dipole_flucts,energy_flucts, num_trajs,correlation_length,tau,time_step,stdout):
        stdout.write('\n'+"Constructing classical 3rd order correlation function C_muUmu from MD trajectory."+'\n')
        classical_corr=np.zeros((correlation_length*2+1,correlation_length*2+1,3))
        traj_counter=0
        while traj_counter<num_trajs:
                # get classical correlation function:
                stdout.write('Processing Trajectory '+str(traj_counter+1)+' of '+str(num_trajs))
                temp_corr=construct_classical_mu_U_mu_from_single_traj(dipole_flucts[:,traj_counter,:],energy_flucts[:,traj_counter],correlation_length,time_step)
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

# constructing muUmu from single trajectory: CURRENTLY THIS ASSUMES ONLY A SINGLE TRAJ
def construct_classical_mu_U_mu_from_single_traj(dipole_flucts,energy_flucts,correlation_length,time_step):
        func_size=correlation_length*2+1
        corr_func=np.zeros((func_size,func_size,3))
        for icount in range(corr_func.shape[0]):
                print(icount)
                for jcount in range(corr_func.shape[0]):  # DO NOT exploit symmetry. Calculate full corr func
                        integrant=get_correlation_integrant_muUmu_3rd(dipole_flucts,energy_flucts,icount,jcount,correlation_length,time_step)
                        corr_func[icount,jcount,0]=(icount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
                        corr_func[icount,jcount,1]=(jcount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
                        corr_func[icount,jcount,2]=integrate.simps(integrant[:,1],dx=time_step)/(1.0*integrant.shape[0])

        # make sure that the step length does not get multiplied in the trapezium integral function
        for icount in range(corr_func.shape[0]):
                for jcount in range(corr_func.shape[0]):
                        corr_func[icount,jcount,2]=corr_func[icount,jcount,2]/(time_step)

        return corr_func

@jit(fastmath=True) 
def get_correlation_integrant_muUmu_3rd(dipole_flucts,energy_flucts,current_corr_i,current_corr_j,correlation_length,MD_steplength):
        relative_start_i=current_corr_i-correlation_length   # these can be negative or positive and define a range
        relative_start_j=current_corr_j-correlation_length   # over which the correlation function gets calculated
        min_index=min(relative_start_i,relative_start_j)
        max_index=max(relative_start_i,relative_start_j)
        if max_index<0:
                max_index=0
        if min_index>0:
                min_index=0
#
        # set maximum size that the integrant can have. This is chosen such that counter+min_index and counter+max_index
        # are guaranteed to be within the range of the integrant
        integrant_size=energy_flucts.shape[0]-(abs(min_index)+max_index+2)
        integrant=np.zeros((integrant_size,2))
        counter=abs(min_index)
        while counter<integrant.shape[0]:
                integrant[counter,0]=(counter*MD_steplength)
                integrant[counter,1]=np.dot(dipole_flucts[counter,:],dipole_flucts[counter+relative_start_j,:])*(energy_flucts[counter+relative_start_i])
                counter=counter+1
        return integrant

# construct UUmu from single trajectory
def construct_classical_U_U_mu_from_single_traj(dipole_flucts,energy_flucts,correlation_length,time_step):
        func_size=correlation_length*2+1
        corr_func=np.zeros((func_size,func_size,5)) # vector quantity 
        for icount in range(corr_func.shape[0]):
                print(icount)
                for jcount in range(corr_func.shape[0]):  # DO NOT exploit symmetry. Calculate full corr func
                        integrant=get_correlation_integrant_UUmu_3rd(dipole_flucts,energy_flucts,icount,jcount,correlation_length,time_step)
                        corr_func[icount,jcount,0]=(icount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
                        corr_func[icount,jcount,1]=(jcount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
                        corr_func[icount,jcount,2]=integrate.simps(integrant[:,0],dx=time_step)/(1.0*integrant.shape[0])
                        corr_func[icount,jcount,3]=integrate.simps(integrant[:,1],dx=time_step)/(1.0*integrant.shape[0])
                        corr_func[icount,jcount,4]=integrate.simps(integrant[:,2],dx=time_step)/(1.0*integrant.shape[0])

        # make sure that the step length does not get multiplied in the trapezium integral function
        for icount in range(corr_func.shape[0]):
                for jcount in range(corr_func.shape[0]):
                        corr_func[icount,jcount,2]=corr_func[icount,jcount,2]/(time_step)
                        corr_func[icount,jcount,3]=corr_func[icount,jcount,3]/(time_step)
                        corr_func[icount,jcount,4]=corr_func[icount,jcount,4]/(time_step)

        return corr_func

@jit(fastmath=True)
def get_correlation_integrant_UUmu_3rd(dipole_flucts,energy_flucts,current_corr_i,current_corr_j,correlation_length,MD_steplength):
        relative_start_i=current_corr_i-correlation_length   # these can be negative or positive and define a range
        relative_start_j=current_corr_j-correlation_length   # over which the correlation function gets calculated
        min_index=min(relative_start_i,relative_start_j)
        max_index=max(relative_start_i,relative_start_j)
        if max_index<0:
                max_index=0
        if min_index>0:
                min_index=0
#
        # set maximum size that the integrant can have. This is chosen such that counter+min_index and counter+max_index
        # are guaranteed to be within the range of the integrant
        integrant_size=energy_flucts.shape[0]-(abs(min_index)+max_index+2)
        integrant=np.zeros((integrant_size,3)) # integrant is a vector
        counter=abs(min_index)
        while counter<integrant.shape[0]:
                integrant[counter,:]=(dipole_flucts[counter,:])*(energy_flucts[counter+relative_start_j])*(energy_flucts[counter+relative_start_i])
                counter=counter+1
        return integrant

# This correlation function is a VECTOR, as it contains only one instance of mu
def construct_corr_func_3rd_U_U_mu(dipole_flucts,energy_flucts, num_trajs,correlation_length,tau,time_step,stdout):
        stdout.write('\n'+"Constructing classical 3rd order correlation function C_muUmu from MD trajectory."+'\n')
        classical_corr=np.zeros((correlation_length*2+1,correlation_length*2+1,5))
        traj_counter=0
        while traj_counter<num_trajs:
                # get classical correlation function:
                stdout.write('Processing Trajectory '+str(traj_counter+1)+' of '+str(num_trajs))
                temp_corr=construct_classical_U_U_mu_from_single_traj(dipole_flucts[:,traj_counter,:],energy_flucts[:,traj_counter],correlation_length,time_step)
                icount=0
                while icount<temp_corr.shape[0]:
                        jcount=0
                        while jcount<temp_corr.shape[0]:
                                classical_corr[icount,jcount,0]=temp_corr[icount,jcount,0]
                                classical_corr[icount,jcount,1]=temp_corr[icount,jcount,1]
                                classical_corr[icount,jcount,2]=classical_corr[icount,jcount,2]+temp_corr[icount,jcount,2]
                                classical_corr[icount,jcount,3]=classical_corr[icount,jcount,3]+temp_corr[icount,jcount,3]
                                classical_corr[icount,jcount,4]=classical_corr[icount,jcount,4]+temp_corr[icount,jcount,4]
                                jcount=jcount+1
                        icount=icount+1
                traj_counter=traj_counter+1
        # have summed over all trajectories. Now average and multiply by decaying exponential.
        icount=0
        while icount<temp_corr.shape[0]:
                jcount=0
                while jcount<temp_corr.shape[0]:
                        classical_corr[icount,jcount,2]=classical_corr[icount,jcount,2]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
                        classical_corr[icount,jcount,3]=classical_corr[icount,jcount,3]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
                        classical_corr[icount,jcount,4]=classical_corr[icount,jcount,4]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
                        jcount=jcount+1
                icount=icount+1

        return classical_corr


# constructing muUU from single trajectory
def construct_classical_mu_U_U_from_single_traj(dipole_flucts,energy_flucts,correlation_length,time_step):
        func_size=correlation_length*2+1
        corr_func=np.zeros((func_size,func_size,5)) # vector quantity 
        for icount in range(corr_func.shape[0]):
                print(icount)
                for jcount in range(corr_func.shape[0]):  # DO NOT exploit symmetry. Calculate full corr func
                        integrant=get_correlation_integrant_muUU_3rd(dipole_flucts,energy_flucts,icount,jcount,correlation_length,time_step)
                        corr_func[icount,jcount,0]=(icount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
                        corr_func[icount,jcount,1]=(jcount*time_step)-(corr_func.shape[0]-1)/2.0*time_step
                        corr_func[icount,jcount,2]=integrate.simps(integrant[:,0],dx=time_step)/(1.0*integrant.shape[0])
                        corr_func[icount,jcount,3]=integrate.simps(integrant[:,1],dx=time_step)/(1.0*integrant.shape[0])
                        corr_func[icount,jcount,4]=integrate.simps(integrant[:,2],dx=time_step)/(1.0*integrant.shape[0])

        # make sure that the step length does not get multiplied in the trapezium integral function
        for icount in range(corr_func.shape[0]):
                for jcount in range(corr_func.shape[0]):
                        corr_func[icount,jcount,2]=corr_func[icount,jcount,2]/(time_step)
                        corr_func[icount,jcount,3]=corr_func[icount,jcount,3]/(time_step)
                        corr_func[icount,jcount,4]=corr_func[icount,jcount,4]/(time_step)
	
        return corr_func

@jit(fastmath=True)
def get_correlation_integrant_muUU_3rd(dipole_flucts,energy_flucts,current_corr_i,current_corr_j,correlation_length,MD_steplength):
        relative_start_i=current_corr_i-correlation_length   # these can be negative or positive and define a range
        relative_start_j=current_corr_j-correlation_length   # over which the correlation function gets calculated
        min_index=min(relative_start_i,relative_start_j)
        max_index=max(relative_start_i,relative_start_j)
        if max_index<0:
                max_index=0
        if min_index>0:
                min_index=0
#
        # set maximum size that the integrant can have. This is chosen such that counter+min_index and counter+max_index
        # are guaranteed to be within the range of the integrant
        integrant_size=energy_flucts.shape[0]-(abs(min_index)+max_index+2)
        integrant=np.zeros((integrant_size,3)) # integrant is a vector
        counter=abs(min_index)
        while counter<integrant.shape[0]:
                integrant[counter,:]=(dipole_flucts[counter+relative_start_j,:])*(energy_flucts[counter+relative_start_i])*(energy_flucts[counter])
                counter=counter+1
        return integrant

# This correlation function is a VECTOR, as it contains only one instance of mu
def construct_corr_func_3rd_mu_U_U(dipole_flucts,energy_flucts, num_trajs,correlation_length,tau,time_step,stdout):
        stdout.write('\n'+"Constructing classical 3rd order correlation function C_muUmu from MD trajectory."+'\n')
        classical_corr=np.zeros((correlation_length*2+1,correlation_length*2+1,5))
        traj_counter=0
        while traj_counter<num_trajs:
                # get classical correlation function:
                stdout.write('Processing Trajectory '+str(traj_counter+1)+' of '+str(num_trajs))
                temp_corr=construct_classical_mu_U_U_from_single_traj(dipole_flucts[:,traj_counter,:],energy_flucts[:,traj_counter],correlation_length,time_step)
                icount=0
                while icount<temp_corr.shape[0]:
                        jcount=0
                        while jcount<temp_corr.shape[0]:
                                classical_corr[icount,jcount,0]=temp_corr[icount,jcount,0]
                                classical_corr[icount,jcount,1]=temp_corr[icount,jcount,1]
                                classical_corr[icount,jcount,2]=classical_corr[icount,jcount,2]+temp_corr[icount,jcount,2]
                                classical_corr[icount,jcount,3]=classical_corr[icount,jcount,3]+temp_corr[icount,jcount,3]
                                classical_corr[icount,jcount,4]=classical_corr[icount,jcount,4]+temp_corr[icount,jcount,4]
                                jcount=jcount+1
                        icount=icount+1
                traj_counter=traj_counter+1
        # have summed over all trajectories. Now average and multiply by decaying exponential.
        icount=0
        while icount<temp_corr.shape[0]:
                jcount=0
                while jcount<temp_corr.shape[0]:
                        classical_corr[icount,jcount,2]=classical_corr[icount,jcount,2]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
                        classical_corr[icount,jcount,3]=classical_corr[icount,jcount,3]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
                        classical_corr[icount,jcount,4]=classical_corr[icount,jcount,4]/num_trajs*math.exp(-abs(classical_corr[icount,jcount,0])/tau)*math.exp(-abs(classical_corr[icount,jcount,1])/tau)
                        jcount=jcount+1
                icount=icount+1

        return classical_corr

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
                        corr_func[counter,0]=corr_func[counter,0]+transform_time_x[counter]
                        corr_func[counter,1]=corr_func[counter,1]+transform_time_y[counter]
                        corr_func[counter,2]=corr_func[counter,2]+transform_time_z[counter]
                        counter=counter+1

                traj_count=traj_count+1

	# SYMMETRIZE. THIS SHOULD IN PRINCIPLE NOT BE NECESSARY FOR WELL ENOUGH BEHAVED CROSS CORRELATION FUNC
        for i in range(corr_func.shape[0]):
                corr_func[i,0]=0.5*(corr_func[i,0]+corr_func[corr_func.shape[0]-1-i,0])
                corr_func[corr_func.shape[0]-1-i,0]=corr_func[i,0]
                corr_func[i,1]=0.5*(corr_func[i,1]+corr_func[corr_func.shape[0]-1-i,1])
                corr_func[corr_func.shape[0]-1-i,1]=corr_func[i,1]
                corr_func[i,2]=0.5*(corr_func[i,2]+corr_func[corr_func.shape[0]-1-i,2])
                corr_func[corr_func.shape[0]-1-i,2]=corr_func[i,2]
       

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


# build the vector quantity in the frequency domain:
def compute_mu_U_U_corr_func_freq(corr_func_mu_U_U,sampling_rate_in_fs,low_freq_filter):
        step_length_corr=corr_func_mu_U_U[1,1,0]-corr_func_mu_U_U[0,0,0]
        sample_rate=sampling_rate_in_fs*math.pi*2.0*const.hbar_in_eVfs
        new_dim=corr_func_mu_U_U.shape[0]*2+1  # pad with a factor of 2 . Resulting value is guaranteed to be odd
	# try padding:
        extended_func_x=np.zeros((new_dim,new_dim),dtype=np.complex)
        extended_func_y=np.zeros((new_dim,new_dim),dtype=np.complex)
        extended_func_z=np.zeros((new_dim,new_dim),dtype=np.complex)

        # pad with zeros:
        start_index=int((new_dim-corr_func_mu_U_U.shape[0])/2)
        end_index=start_index+corr_func_mu_U_U.shape[0]

        for icount in range(new_dim):
                for jcount in range(new_dim):
                        if icount<start_index or icount>end_index-1 or jcount<start_index or jcount>end_index-1:
                                extended_func_x[icount,jcount]=0.0+0.0j
                                extended_func_y[icount,jcount]=0.0+0.0j
                                extended_func_z[icount,jcount]=0.0+0.0j
                        else:
                                extended_func_x[icount,jcount]=corr_func_mu_U_U[icount-start_index,jcount-start_index,2]
                                extended_func_y[icount,jcount]=corr_func_mu_U_U[icount-start_index,jcount-start_index,3]
                                extended_func_z[icount,jcount]=corr_func_mu_U_U[icount-start_index,jcount-start_index,4]

        # remember: Need to normalize correlation func in frequency domain by multiplying by dt**2.0
        corr_func_freq_x=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func_x[:,:]))))*step_length_corr*step_length_corr
        corr_func_freq_y=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func_y[:,:]))))*step_length_corr*step_length_corr
        corr_func_freq_z=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func_z[:,:]))))*step_length_corr*step_length_corr

        freq_list=np.fft.fftshift(np.fft.fftfreq(corr_func_freq_x.shape[0],d=1./sample_rate))/const.Ha_to_eV

        full_corr_func_freq=np.zeros((corr_func_freq_x.shape[0],corr_func_freq_x.shape[0],5),dtype=np.complex_)

        for icount in range (full_corr_func_freq.shape[0]):
                for jcount in range(full_corr_func_freq.shape[1]):
                        full_corr_func_freq[icount,jcount,0]=freq_list[icount]
                        full_corr_func_freq[icount,jcount,1]=freq_list[jcount]
                        full_corr_func_freq[icount,jcount,2]=corr_func_freq_x[icount,jcount].real
                        full_corr_func_freq[icount,jcount,3]=corr_func_freq_y[icount,jcount].real
                        full_corr_func_freq[icount,jcount,4]=corr_func_freq_z[icount,jcount].real
                        if abs(freq_list[icount])<low_freq_filter and abs(freq_list[jcount])<low_freq_filter:
                                full_corr_func_freq[icount,jcount,2]=0.0
                                full_corr_func_freq[icount,jcount,3]=0.0
                                full_corr_func_freq[icount,jcount,4]=0.0

        return full_corr_func_freq

# compute the classical correlation function of U_mu_U in the frequency domain
def compute_mu_U_mu_corr_func_freq(corr_func_mu_U_mu,sampling_rate_in_fs,low_freq_filter): # this is a scalar func
        step_length_corr=corr_func_mu_U_mu[1,1,0]-corr_func_mu_U_mu[0,0,0]
        sample_rate=sampling_rate_in_fs*math.pi*2.0*const.hbar_in_eVfs
        new_dim=corr_func_mu_U_mu.shape[0]*2+1  # pad with a factor of 2 . Resulting value is guaranteed to be odd
	# try padding:
        extended_func=np.zeros((new_dim,new_dim),dtype=np.complex)

        # pad with zeros:
        start_index=int((new_dim-corr_func_mu_U_mu.shape[0])/2)
        end_index=start_index+corr_func_mu_U_mu.shape[0]

        for icount in range(new_dim):
                for jcount in range(new_dim):
                        if icount<start_index or icount>end_index-1 or jcount<start_index or jcount>end_index-1:
                                extended_func[icount,jcount]=0.0+0.0j
                        else:
                                extended_func[icount,jcount]=corr_func_mu_U_mu[icount-start_index,jcount-start_index,2]

	# remember: Need to normalize correlation func in frequency domain by multiplying by dt**2.0
        corr_func_freq=(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(extended_func[:,:]))))*step_length_corr*step_length_corr

        freq_list=np.fft.fftshift(np.fft.fftfreq(corr_func_freq.shape[0],d=1./sample_rate))/const.Ha_to_eV

        full_corr_func_freq=np.zeros((corr_func_freq.shape[0],corr_func_freq.shape[0],3),dtype=np.complex_)

        for icount in range (full_corr_func_freq.shape[0]):
                for jcount in range(full_corr_func_freq.shape[1]):
                        full_corr_func_freq[icount,jcount,0]=freq_list[icount]
                        full_corr_func_freq[icount,jcount,1]=freq_list[jcount]
                        full_corr_func_freq[icount,jcount,2]=corr_func_freq[icount,jcount].real
                        if abs(freq_list[icount])<low_freq_filter and abs(freq_list[jcount])<low_freq_filter:
                                full_corr_func_freq[icount,jcount,2]=0.0

        return full_corr_func_freq


# remember, mu_av and mu_reonorm and mu reorg, as well as corr_func_cross_freq, are vector quantities
@jit(fastmath=True)
def HT_2nd_order_integrant(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,t,is_dipole_only):
        integrant=np.zeros((corr_func_freq.shape[0],corr_func_freq.shape[1]),dtype=np.complex_)
        tol=1.0e-15
        for i in range(corr_func_freq.shape[0]):
                integrant[i,0]=corr_func_freq[i,0]
                omega=integrant[i,0]
                # check for omega=0 condition
                if abs(omega)<tol:
                        denom=mu_renorm**2.0*2.0*math.pi
                        if is_dipole_only: # only dipole dipole corr func contributes
                            num=corr_func_freq[i,1]
                        else:
                            num=2.0*(mu_reorg[0]-mu_av[0])*t*corr_func_cross_freq[i,1]+2.0*(mu_reorg[1]-mu_av[1])*t*corr_func_cross_freq[i,2]+2.0*(mu_reorg[2]-mu_av[2])*t*corr_func_cross_freq[i,3]+corr_func_freq[i,1]
                        integrant[i,1]=num/denom

                else:
                        denom=kBT*mu_renorm**2.0*2.0*math.pi*(1.0-np.exp(-omega/kBT))
                        if is_dipole_only:
                            num=corr_func_freq[i,1]*omega*np.exp(-1j*omega*t)
                        else:
                            num=2.0*((mu_reorg[0]-mu_av[0])*corr_func_cross_freq[i,1]+(mu_reorg[1]-mu_av[1])*corr_func_cross_freq[i,2]+(mu_reorg[2]-mu_av[2])*corr_func_cross_freq[i,3])*(1.0-cmath.exp(-1j*omega*t))+corr_func_freq[i,1]*omega*np.exp(-1j*omega*t)
                        integrant[i,1]=num/denom

        return integrant

# first of the 3 integrant terms required for 3rd order cumulant correction to HT term
@jit(fastmath=True)
def HT_integrant_mu_U_mu(corr_func_mu_U_mu_freq,mu_renorm,kBT,t_current):
	beta=1.0/kBT
	tol=1.0e-15
	integrant=np.zeros((corr_func_mu_U_mu_freq.shape[0],corr_func_mu_U_mu_freq.shape[1],3),dtype=np.complex_) 
	for i in range(integrant.shape[0]):
		for j in range(integrant.shape[1]):
			w1=corr_func_mu_U_mu_freq[i,j,0]
			w2=corr_func_mu_U_mu_freq[i,j,1]
			integrant[i,j,0]=w1
			integrant[i,j,1]=w2
			const_fac=corr_func_mu_U_mu_freq[i,j,2]/mu_renorm**2.0
			w_bar=w1+w2
			if abs(w1)<tol and abs(w2)<tol:
				num=t_current
				denom=4.0*math.pi**2.0
			elif abs(w1)<tol:
				num=beta**2.0*w2**2.0*t_current*np.exp(beta*w2)*np.exp(-1j*w2*t_current)
				denom=8.0*math.pi**2.0*(np.exp(beta*w2)-beta*w2-1.0)
			elif abs(w2)<tol:
				num=-1j*beta**2.0*w1*np.exp(beta*w1)*np.exp(-1j*w1*t_current)*(np.exp(1j*w1*t_current)-1.0)
				denom=8.0*math.pi**2.0*(1.0+np.exp(beta*w1)*(beta*w1-1.0))
			elif abs(w_bar)<tol:
				num=1j*beta**2.0*w1*(np.exp(1j*w1*t_current)-1.0)
				denom=8.0*math.pi**2.0*(1.0-np.exp(beta*w1)-beta*w1)
			else:
				num=-1j*w_bar*w2*beta**2.0*(np.exp(-1j*w2*t_current)*(np.exp(1j*w2*t_current)-np.exp(-1j*w_bar*t_current)))
				denom=8.0*math.pi**2.0*(w2*np.exp(-beta*w_bar)-w_bar*np.exp(-beta*w2)+w1)
		
			integrant[i,j,2]=num/denom*const_fac
			
	return integrant

# Correct
@jit(fastmath=True)
def HT_integrant_mu_U_U(corr_func_mu_U_U_freq,mu_reorg,mu_renorm,kBT,t_current):
        beta=1.0/kBT
        tol=1.0e-15
        integrant=np.zeros((corr_func_mu_U_U_freq.shape[0],corr_func_mu_U_U_freq.shape[1],3),dtype=np.complex_)
        for i in range(integrant.shape[0]):
                for j in range(integrant.shape[1]):
                        w1=corr_func_mu_U_U_freq[i,j,0]
                        w2=corr_func_mu_U_U_freq[i,j,1]
                        integrant[i,j,0]=w1
                        integrant[i,j,1]=w2
                        w_bar=w1+w2
                        const_fac=(corr_func_mu_U_U_freq[i,j,2]*mu_reorg[0]+corr_func_mu_U_U_freq[i,j,3]*mu_reorg[1]+corr_func_mu_U_U_freq[i,j,4]*mu_reorg[2])/mu_renorm**2.0
                        if abs(w1)<tol and abs(w2)<tol:
                                num=-t_current**2.0
                                denom=8.0*math.pi**2.0
                        elif abs(w1)<tol:
                                num=beta**2.0*np.exp(beta*w2)*np.exp(-1j*w2*t_current)*(np.exp(1j*w2*t_current)-1j*w2*t_current-1.0)
                                denom=8.0*math.pi**2.0*(np.exp(beta*w2)-beta*w2-1.0)
                        elif abs(w2)<tol:
                                num=beta**2.0*np.exp(beta*w1)*(np.exp(-1j*w1*t_current)+1j*w1*t_current-1.0)
                                denom=8.0*math.pi**2.0*(1.0+np.exp(beta*w1)*(beta*w1-1.0))
                        elif abs(w_bar)<tol:
                                num=beta**2.0*(np.exp(1j*w1*t_current)-1j*w1*t_current-1.0)
                                denom=8.0*math.pi**2.0*(np.exp(beta*w1)-beta*w1-1.0)
                        else:
                                num=beta**2.0*w1*w2*np.exp(-1j*w2*t_current)*((np.exp(1j*w2*t_current)-1.0)/w2+(np.exp(-1j*w1*t_current)-1.0)/w1)
                                denom=8.0*math.pi**2.0*(w2*np.exp(-beta*w_bar)-w_bar*np.exp(-beta*w2)+w1)

                        integrant[i,j,2]=num/denom*const_fac

        return integrant

# Correct
@jit(fastmath=True)
def HT_integrant_U_U_mu(corr_func_U_U_mu_freq,mu_reorg,mu_renorm,mu_av,kBT,t_current):
        mu_eff=mu_reorg-2.0*mu_av
        beta=1.0/kBT
        tol=1.0e-15
        integrant=np.zeros((corr_func_U_U_mu_freq.shape[0],corr_func_U_U_mu_freq.shape[1],3),dtype=np.complex_)
        for i in range(integrant.shape[0]):
                for j in range(integrant.shape[1]):
                        w1=corr_func_U_U_mu_freq[i,j,0]
                        w2=corr_func_U_U_mu_freq[i,j,1]
                        integrant[i,j,0]=w1
                        integrant[i,j,1]=w2
                        w_bar=w1+w2
                        const_fac=(corr_func_U_U_mu_freq[i,j,2]*mu_eff[0]+corr_func_U_U_mu_freq[i,j,3]*mu_eff[1]+corr_func_U_U_mu_freq[i,j,4]*mu_eff[2])/mu_renorm**2.0
                        if abs(w1)<tol and abs(w2)<tol:
                                num=t_current**2.0
                                denom=8.0*math.pi**2.0
                        elif abs(w1)<tol:
                                num=beta**2.0*np.exp(beta*w2)*np.exp(-1j*w2*t_current)*(np.exp(1j*w2*t_current)-1j*w2*t_current-1.0)
                                denom=8.0*math.pi**2.0*(1.0-np.exp(beta*w2)+beta*w2)
                        elif abs(w2)<tol:
                                num=beta**2.0*np.exp(beta*w1)*np.exp(-1j*w1*t_current)*(np.exp(1j*w1*t_current)*(1.0+1j*w1*t_current)-1.0)
                                denom=8.0*math.pi**2.0*(1.0+np.exp(beta*w1)*(beta*w1-1.0))
                        elif abs(w_bar)<tol:
                                num=beta**2.0*(np.exp(1j*w1*t_current)-1j*w1*t_current-1.0)
                                denom=8.0*math.pi**2.0*(1.0-np.exp(beta*w1)+beta*w1)
                        else:
                                num=beta**2.0*w_bar*w2*((1.0-np.exp(-1j*w_bar*t_current))/w_bar-(1.0-np.exp(-1j*w2*t_current))/w2)
                                denom=8.0*math.pi**2.0*(w2*np.exp(-beta*w_bar)-w_bar*np.exp(-beta*w2)+w1)

                        integrant[i,j,2]=num/denom*const_fac

        return integrant

def compute_HT_term_3rd_order(corr_func_mu_U_U_freq,corr_func_U_U_mu_freq,corr_func_mu_U_mu_freq,mu_av,mu_renorm,mu_reorg,kBT,max_t,steps):
	Afunc=np.zeros((steps,2),dtype=np.complex_)
	step_length=max_t/steps
	for i in range(steps):
		t_current=i*step_length
		Afunc[i,0]=t_current
		integrant1=HT_integrant_U_U_mu(corr_func_U_U_mu_freq,mu_reorg,mu_renorm,mu_av,kBT,t_current)	
		integrant2=HT_integrant_mu_U_mu(corr_func_mu_U_mu_freq,mu_renorm,kBT,t_current)
		integrant3=HT_integrant_mu_U_U(corr_func_mu_U_U_freq,mu_reorg,mu_renorm,kBT,t_current)
		tot_integrant=integrant1
		tot_integrant[:,:,2]=tot_integrant[:,:,2]-1j*integrant2[:,:,2]+integrant3[:,:,2]
		Afunc[i,1]=cumulant.simpson_integral_2D(tot_integrant)


	Afunc[:,1]=Afunc[:,1]*mu_renorm**2.0
	return Afunc


# The full A func. NOTE THAT MU RENORM IS A SCALAR QUANTITY, NOT A VECTOR QUANTITY
def compute_HT_term_2nd_order(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,max_t,steps,is_dipole_only):
        Afunc=np.zeros((steps,2),dtype=np.complex_)
        step_length=max_t/steps
        for i in range(steps):
                t_current=i*step_length
                Afunc[i,0]=t_current
                integrant=HT_2nd_order_integrant(corr_func_freq,corr_func_cross_freq,mu_av,mu_renorm,mu_reorg,kBT,t_current,is_dipole_only)
                Afunc[i,1]=1.0+integrate.simps(integrant[:,1],dx=(integrant[1,0]-integrant[0,0]))

        Afunc[:,1]=Afunc[:,1]*mu_renorm**2.0

        # STUPID TEST: Integrate over corr funcs:
        effective_cross_corr=np.zeros(3)
        effective_cross_corr[0]=integrate.simps(corr_func_cross_freq[:,1],dx=(integrant[1,0]-integrant[0,0]))
        effective_cross_corr[1]=integrate.simps(corr_func_cross_freq[:,2],dx=(integrant[1,0]-integrant[0,0]))
        effective_cross_corr[2]=integrate.simps(corr_func_cross_freq[:,3],dx=(integrant[1,0]-integrant[0,0]))
        print('Effective magnitude of cross correlation term:')
        print(effective_cross_corr,np.dot(effective_cross_corr,mu_av))

        return Afunc
