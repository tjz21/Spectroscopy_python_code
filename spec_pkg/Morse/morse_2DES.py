#! /usr/bin/env python
  
import numpy as np
import math
import cmath
from numba import jit
from spec_pkg.constants import constants as const
from spec_pkg.nonlinear_spectrum import twoDES
from spec_pkg.cumulant import cumulant as cumul

###########################################################
# This routine computes the exact third order response of #
# the Morse oscillator                                    #
# Since we have the eigenstates and the overlaps of the   #
# Morse oscillator, this expression can be evaluated in a #
# Straightforward manner.                                 #
###########################################################

# this version of the spectrum integrant assumes that rfuncs is already the exponentiated form
@jit(fastmath=True)
def twoD_spectrum_integrant_morse(rfuncs,omega1,omega2,t_delay):
        step_length=(rfuncs[1,0,0]-rfuncs[0,0,0]).real
        steps_in_t_delay=int(t_delay/step_length)  # NOTE: delay time is rounded to match with steps in the response function
        # this means that by default, the resolution of delay time in the 2DES spectrum is 1 fs. 
        max_index=rfuncs.shape[0]
        integrant=np.zeros((max_index,max_index,3),dtype=np.complex_)

        for count1 in range(max_index):
                for count2 in range(max_index):
                                integrant[count1,count2,0]=rfuncs[count1,0,0]
                                integrant[count1,count2,1]=rfuncs[count2,0,0]
                                # rephasing diagrams=2,3. Non-rephasing=1,4. Rephasing has a negative sign, non-rephasing a positive sign. 
                                integrant[count1,count2,2]=np.exp(1j*(rfuncs[count1,0,0]*(omega1)+rfuncs[count2,0,0]*(omega2)))*(rfuncs[count1,count2,2]+rfuncs[count1,count2,5])+np.exp(1j*(-rfuncs[count1,0,0]*(omega1)+rfuncs[count2,0,0]*(omega2)))*(rfuncs[count1,count2,3]+rfuncs[count1,count2,4])

        return integrant



def calc_2DES_time_series_morse(morse,g2_solvent,E_min1,E_max1,E_min2,E_max2,num_points_2D,rootname,num_times,time_step):
                averaged_val=np.zeros((num_times,2))

                # make sure chosen delay times break down into an integer number of timesteps in the response function
                step_length_t=(g2_solvent[1,0]-g2_solvent[0,0]).real
                eff_time_index_2DES=int(round(time_step/step_length_t))
                eff_time_step_2DES=eff_time_index_2DES*step_length_t

                current_delay=0.0
                current_delay_index=0
                counter=0
                while counter<num_times:
                                print(counter,current_delay)
                                spectrum_2D=calc_2D_spectrum_exact_morse(morse.gs_energies,morse.ex_energies,morse.wf_overlaps,morse.boltzmann_fac,g2_solvent,current_delay,current_delay_index,E_min1,E_max1,E_min2,E_max2,num_points_2D)
                                twoDES.print_2D_spectrum(rootname+'_2DES_'+str(counter)+'.dat',spectrum_2D,False)

                                averaged_val[counter,0]=current_delay
                                averaged_val[counter,1]=cumul.simpson_integral_2D(spectrum_2D)
                                counter=counter+1
                                current_delay=current_delay+eff_time_step_2DES
                                current_delay_index=current_delay_index+eff_time_index_2DES

                np.savetxt(rootname+'_Morse_averaged_spectrum.txt',averaged_val)


def calc_2DES_time_series_morse_list(morse_list,g2_solvent,E_min1,E_max1,E_min2,E_max2,num_points_2D,rootname,num_times,time_step):
                averaged_val=np.zeros((num_times,2))
                # make sure chosen delay times break down into an integer number of timesteps in the response function
                step_length_t=(g2_solvent[1,0]-g2_solvent[0,0]).real
                eff_time_index_2DES=int(round(time_step/step_length_t))
                eff_time_step_2DES=eff_time_index_2DES*step_length_t

                current_delay=0.0
                current_delay_index=0
                counter=0
                while counter<num_times:
                                print(counter,current_delay)
                                spectrum_2D=calc_2D_spectrum_exact_morse_list(morse_list,g2_solvent,current_delay,current_delay_index,E_min1,E_max1,E_min2,E_max2,num_points_2D)
                                twoDES.print_2D_spectrum(rootname+'_2DES_'+str(counter)+'.dat',spectrum_2D,False)
                                
                                averaged_val[counter,0]=current_delay
                                averaged_val[counter,1]=cumul.simpson_integral_2D(spectrum_2D)
                                counter=counter+1
                                current_delay=current_delay+eff_time_step_2DES
                                current_delay_index=current_delay_index+eff_time_index_2DES

                np.savetxt(rootname+'_Morse_averaged_spectrum.txt',averaged_val)


# this routine is for a morse list, which means that gs_energies, ex_energies, overlap and boltmann_fac are all lists
# This routine works for several uncoupled Morse oscillators
def calc_2D_spectrum_exact_morse_list(morse_list,g2_solvent,delay_time,delay_index,E_min1,E_max1,E_min2,E_max2,num_points_2D):
        full_2D_spectrum=np.zeros((num_points_2D,num_points_2D,3))
        step_length=((E_max1-E_min1)/num_points_2D)
        num_steps=g2_solvent.shape[0]
        max_t=g2_solvent[num_steps-1,0]

        rfunc=total_Rfuncs_exact_list(morse_list,g2_solvent,delay_time,delay_index)
        for counter1 in range(num_points_2D):
                omega1=E_min1+counter1*step_length
                for counter2 in range(num_points_2D):
                                omega2=E_min2+counter2*step_length
                                full_2D_integrant=twoD_spectrum_integrant_morse(rfunc,omega1,omega2,delay_time)
                                full_2D_spectrum[counter1,counter2,0]=omega1
                                full_2D_spectrum[counter1,counter2,1]=omega2
                                full_2D_spectrum[counter1,counter2,2]=cumul.simpson_integral_2D(full_2D_integrant).real

        return full_2D_spectrum


# Calculation routine for the exact full 2DES spectrum for a morse potential described by ground and excited state energies and 
# wavefunction overlaps. This is for a Single morse oscillator.  
@jit
def calc_2D_spectrum_exact_morse(gs_energies,ex_energies,overlap,boltzmann_fac,g2_solvent,delay_time,delay_index,E_min1,E_max1,E_min2,E_max2,num_points_2D):
        full_2D_spectrum=np.zeros((num_points_2D,num_points_2D,3))
        step_length=((E_max1-E_min1)/num_points_2D)

        rfunc=total_Rfuncs_exact(gs_energies,ex_energies,overlap,g2_solvent,delay_time,delay_index)

        for counter1 in range(num_points_2D):
                omega1=E_min1+counter1*step_length
                for counter2 in range(num_points_2D):
                                omega2=E_min2+counter2*step_length
                                full_2D_integrant=twoD_spectrum_integrant_morse(rfunc,omega1,omega2,delay_time)
                                full_2D_spectrum[counter1,counter2,0]=omega1
                                full_2D_spectrum[counter1,counter2,1]=omega2
                                full_2D_spectrum[counter1,counter2,2]=cumul.simpson_integral_2D(full_2D_integrant).real

        return full_2D_spectrum


# same as Rfuncs_exact, but for several uncoupled Morse oscillators. 
def total_Rfuncs_exact_list(morse_list,g2_solvent,tdelay,delay_index):
	rfuncs_solvent=twoDES.calc_Rfuncs_tdelay(g2_solvent,tdelay,delay_index)
	num_steps=rfuncs_solvent.shape[0]
	step_length=g2_solvent[1,0]-g2_solvent[0,0]

	rfuncs_solute=np.zeros((rfuncs_solvent.shape[0],rfuncs_solvent.shape[0],6),dtype=np.complex_)
	for i in range(len(morse_list)):
		temp=compute_rfuncs_tdelay(morse_list[i].gs_energies,morse_list[i].ex_energies,morse_list[i].wf_overlaps,morse_list[i].boltzmann_fac,step_length,num_steps,tdelay)
		print('COMPUtED RFUNC NUMBER:' +str(i))
		print(temp)
		if i==0:
                        rfuncs_solute=temp
		else:
			rfuncs_solute[:,:,2]=rfuncs_solute[:,:,2]*temp[:,:,2]
			rfuncs_solute[:,:,3]=rfuncs_solute[:,:,3]*temp[:,:,3]
			rfuncs_solute[:,:,4]=rfuncs_solute[:,:,4]*temp[:,:,4]
			rfuncs_solute[:,:,5]=rfuncs_solute[:,:,5]*temp[:,:,5]
		
		print('COMBINDED RFUNC')
		print(rfuncs_solute)
	rfuncs_solute[:,:,2]=rfuncs_solute[:,:,2]*np.exp(rfuncs_solvent[:,:,2])
	rfuncs_solute[:,:,3]=rfuncs_solute[:,:,3]*np.exp(rfuncs_solvent[:,:,3])
	rfuncs_solute[:,:,4]=rfuncs_solute[:,:,4]*np.exp(rfuncs_solvent[:,:,4])
	rfuncs_solute[:,:,5]=rfuncs_solute[:,:,5]*np.exp(rfuncs_solvent[:,:,5])

	return rfuncs_solute


def total_Rfuncs_exact(gs_energies,ex_energies,overlap,boltzmann_fac,g2_solvent,tdelay,delay_index):

	rfuncs_solvent=twoDES.calc_Rfuncs_tdelay(g2_solvent,tdelay,delay_index)
	num_steps=rfuncs_solvent.shape[0]
	step_length=g2_solvent[1,0]-g2_solvent[0,0]


	rfuncs_solute=compute_rfuncs_tdelay(gs_energies,ex_energies,overlap,boltzmann_fac,step_length,num_steps,tdelay)

	rfuncs_solute[:,:,2]=rfuncs_solute[:,:,2]*np.exp(rfuncs_solvent[:,:,2])
	rfuncs_solute[:,:,3]=rfuncs_solute[:,:,3]*np.exp(rfuncs_solvent[:,:,3])
	rfuncs_solute[:,:,4]=rfuncs_solute[:,:,4]*np.exp(rfuncs_solvent[:,:,4])
	rfuncs_solute[:,:,5]=rfuncs_solute[:,:,5]*np.exp(rfuncs_solvent[:,:,5])

	return rfuncs_solute

# build the 3rd order response func for a given tdelay
# overlaps <gs|ex> structure overlap(t)=e^iE_gst <gs|ex>e^-iE_ext
def compute_rfuncs_tdelay(gs_energies,ex_energies,overlap,boltzmann_fac,step_length,num_steps,tdelay): 
	rfuncs=np.zeros((num_steps,num_steps,6),dtype=np.complex_)

	Z=sum(boltzmann_fac)

	for i in range(num_steps):
		print(i)
		for j in range(num_steps):
			rfuncs[i,j,0]=i*step_length
			rfuncs[i,j,1]=j*step_length
			
			# define tau1,tau2,tau3		
			t1=i*step_length
			t2=t1+tdelay
			t3=t2+j*step_length

			# Now build vectors of time evolution
			vec_gs_t1=np.exp(1j*gs_energies*t1)
			vec_gs_t2=np.exp(1j*gs_energies*t2)
			vec_gs_t3=np.exp(1j*gs_energies*t3)
	
			vec_ex_t1=np.exp(1j*ex_energies*t1)
			vec_ex_t2=np.exp(1j*ex_energies*t2)	
			vec_ex_t3=np.exp(1j*ex_energies*t3)

			# now we can construct the overlap matrix 
			# in the time dependent version
			overlap_t1=np.zeros((overlap.shape[0],overlap.shape[1]),dtype=np.complex_) 
			overlap_t2=np.zeros((overlap.shape[0],overlap.shape[1]),dtype=np.complex_) 
			overlap_t3=np.zeros((overlap.shape[0],overlap.shape[1]),dtype=np.complex_) 
			overlap_t1=overlap_t1+overlap
			overlap_t2=overlap_t2+overlap
			overlap_t3=overlap_t3+overlap

			for a in range(overlap.shape[0]):
				for b in range(overlap.shape[1]):
					overlap_t1[a,b]=overlap_t1[a,b]*vec_gs_t1[a]*np.conjugate(vec_ex_t1[b])
					overlap_t2[a,b]=overlap_t2[a,b]*vec_gs_t2[a]*np.conjugate(vec_ex_t2[b])
					overlap_t3[a,b]=overlap_t3[a,b]*vec_gs_t3[a]*np.conjugate(vec_ex_t3[b])

			# now the response functions can be easily constructed
			rfuncs[i,j,2]=np.sum(np.dot(np.dot(np.dot(overlap_t1,overlap_t2.conj().T),np.dot(overlap_t3,overlap.conj().T)),boltzmann_fac))/Z
			rfuncs[i,j,3]=np.sum(np.dot(np.dot(np.dot(overlap,overlap_t2.conj().T),np.dot(overlap_t3,overlap_t1.conj().T)),boltzmann_fac))/Z
			rfuncs[i,j,4]=np.sum(np.dot(np.dot(np.dot(overlap,overlap_t1.conj().T),np.dot(overlap_t3,overlap_t2.conj().T)),boltzmann_fac))/Z
			rfuncs[i,j,5]=np.sum(np.dot(np.dot(np.dot(overlap_t3,overlap_t2.conj().T),np.dot(overlap_t1,overlap.conj().T)),boltzmann_fac))/Z

	return rfuncs
