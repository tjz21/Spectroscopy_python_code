#! /usr/bin/env python
  
import os.path
import numpy as np
import math
import cmath
import sys
from scipy import interpolate
from scipy import integrate
from scipy import special
from numba import jit
from spec_pkg.constants import constants as const
from spec_pkg.GBOM import gbom
from spec_pkg.cumulant import cumulant
from spec_pkg.GBOM import gbom_cumulant_response
from spec_pkg.nonlinear_spectrum import twoDES
from spec_pkg.Morse import morse_2DES

#---------------------------------------------------------------------------------------------
# Class definition of a single discplaced Morse oscillator, where ground and excited state PES 
# can have different parameters. For this Morse oscillator, we should be able to compute the 
# Exact response function using analytic expressions for the nuclear wavefunctions,
# we should also be able to construct an approximate Franck-Condon spectrum. Furthermore,
# since the ground state Wavefunctions can be expressed analytically, we should also, in 
# principle, be able to construct the exact quantum autocorrelation function of the Morse
# oscillator.
#--------------------------------------------------------------------------------------------

# return the result of a ground state wavefunction acting on an excited state Hamiltonian
#@jit(fastmath=True)
def H_e_on_gs_wavefunc(wavefunc,D_ex,alpha_ex,shift,mu,E_adiabatic):
	# first interpolate the wavefunction as a spline. Then compute derivative
	spl=interpolate.UnivariateSpline(wavefunc[:,0],wavefunc[:,1],s=0,k=4)
	second_deriv=spl.derivative(n=2)
	# calculate the 2nd derivative data only at the finite number of x values. 
	second_deriv_dat=second_deriv(wavefunc[:,0])
	
	op_wavefunc=np.zeros((wavefunc.shape[0],2))
	op_wavefunc[:,0]=wavefunc[:,0]
	op_wavefunc[:,1]=-1.0/(2.0*mu)*second_deriv_dat[:]
	for i in range(op_wavefunc.shape[0]):
		op_wavefunc[i,1]=op_wavefunc[i,1]+(D_ex*(1.0-np.exp(-alpha_ex*(op_wavefunc[i,0]-shift)))**2.0)*wavefunc[i,1]

	return op_wavefunc


# same as H_e_on_gs_wavefunc, with the difference that this is for 2 Morse oscillators
# coupled by a Duschinsky rotation. This means that normal modes are no longer separable
# and we have to consider a full 2 dimensional wavefunction
def H_e_on_gs_wavefunc_2D(wavefunc_x,wavefunc_y,D_ex,alpha_ex,J,shift,mu,E_adiabatic):
	# first interpolate the one dimensional wavefunction for each ground state oscillator and compute its
	# derivativ
	spl_x=interpolate.UnivariateSpline(wavefunc_x[:,0],wavefunc_x[:,1],s=0,k=4)
	spl_y=interpolate.UnivariateSpline(wavefunc_y[:,0],wavefunc_y[:,1],s=0,k=4)
	second_deriv_x=spl_x.derivative(n=2)
	second_deriv_y=spl_y.derivative(n=2)
	second_deriv_dat_x=second_deriv_x(wavefunc_x[:,0])
	second_deriv_dat_y=second_deriv_y(wavefunc_y[:,0])

	op_wavefunc=np.zeros((wavefunc_x.shape[0],wavefunc_y.shape[0],3))
	for i in range(wavefunc_x.shape[0]):
		for j in range(wavefunc_y.shape[0]):
			op_wavefunc[i,j,0]=wavefunc_x[i,0]
			op_wavefunc[i,j,1]=wavefunc_y[j,0]
			x=np.array([wavefunc_x[i,0],wavefunc_y[j,0]])
			x=x-shift
			eff_x=np.dot(np.transpose(J),x)
			op_wavefunc[i,j,2]=(-1.0/(2.0*mu[0])*second_deriv_dat_x[i]*wavefunc_y[j,1])+(-1.0/(2.0*mu[1])*second_deriv_dat_y[j]*wavefunc_x[i,1])
			op_wavefunc[i,j,2]=op_wavefunc[i,j,2]+(D_ex[0]*(1.0-np.exp(-alpha_ex[0]*(eff_x[0])))**2.0+D_ex[1]*(1.0-np.exp(-alpha_ex[1]*(eff_x[1])))**2.0)*wavefunc_x[i,1]*wavefunc_y[j,1]

	return op_wavefunc

@jit
def g2_from_corr_func_t(corr_func_freq,t):
	integrant=np.zeros(corr_func_freq.shape[0],dtype=np.complex_)
	tiny=10e-15
	for i in range(corr_func_freq.shape[0]):
		if abs(np.real(corr_func_freq[i,0]))<tiny:
			integrant[i]=1.0/(2.0*math.pi)*t**2.0/2.0*corr_func_freq[i,1]
		else:
			integrant[i]=1.0/(2.0*math.pi*corr_func_freq[i,0]**2.0)*corr_func_freq[i,1]*(1.0-cmath.exp(-1j*corr_func_freq[i,0]*t)-1j*corr_func_freq[i,0]*t)

	return integrate.simps(integrant,dx=corr_func_freq[1,0]-corr_func_freq[0,0]) 

def g2_from_corr_func(corr_func_freq,num_steps,max_t):
	step_length=max_t/num_steps
	g_func=np.zeros((num_steps,2),dtype=np.complex_)

	for i in range(num_steps):
		current_t=step_length*(i*1.0)
		g_func[i,0]=current_t
		g_func[i,1]=g2_from_corr_func_t(corr_func_freq,current_t)

	return g_func

# evaluate the matrix elements <phi_gs|H_e|phi_gs> for the ground state nuclear wavefunctions acting on the
# excited state hamiltonian. 
def gs_wavefunc_He_matrix(num_points,start_point,end_point,D_gs,D_ex,alpha_gs,alpha_ex,shift,mu,E_adiabatic,max_n_gs):
	overlap_matrix=np.zeros((max_n_gs,max_n_gs))
	for i in range(max_n_gs):
		# compute gs wavefunction and the action of excited state Hamiltonian He on it.
		func1=compute_wavefunction_n(num_points, start_point,end_point,D_gs,alpha_gs,mu,i,0.0)
		He_func1=H_e_on_gs_wavefunc(func1,D_ex,alpha_ex,shift,mu,E_adiabatic)
		# exploint symmetry
		for j in range(i,max_n_gs):
			func2=compute_wavefunction_n(num_points, start_point,end_point,D_gs,alpha_gs,mu,j,0.0)
			eff_func=func2[:,1]*He_func1[:,1]
			overlap_matrix[i,j]=integrate.simps(eff_func,dx=func2[1,0]-func2[0,0])
			overlap_matrix[j,i]=overlap_matrix[i,j]			

	return overlap_matrix

# 2 dimensional version of the matrix <phi_gs|H_e|phi_ex> where the modes are coupled through a Duschinsky rotation
def gs_wavefunc_He_matrix_2D(num_points,start_point,end_point,D_gs,D_ex,alpha_gs,alpha_ex,J,shift,mu,E_adiabatic,max_n_gs):
	overlap_matrix=np.zeros((int(max_n_gs[0]*max_n_gs[1]),int(max_n_gs[0]*max_n_gs[1])))
	for i in range(int(max_n_gs[0])):
		# compute first gs wavefunction
		func1_x=compute_wavefunction_n(num_points[0], start_point[0],end_point[0],D_gs[0],alpha_gs[0],mu[0],i,0.0)
		for j in range(int(max_n_gs[1])):
			func1_y=compute_wavefunction_n(num_points[1], start_point[1],end_point[1],D_gs[1],alpha_gs[1],mu[1],j,0.0)
			func1=np.zeros((func1_x.shape[0],func1_y.shape[0],3))
			# successfully constructed ground state wavefunction. Now apply excited state Hamiltonian
			He_func1=H_e_on_gs_wavefunc_2D(func1_x,func1_y,D_ex,alpha_ex,J,shift,mu,E_adiabatic)

			# exploit symmetry
			for k in range(i,int(max_n_gs[0])):
				func2_x=compute_wavefunction_n(num_points[0], start_point[0],end_point[0],D_gs[0],alpha_gs[0],mu[0],k,0.0)
				for l in range(j,int(max_n_gs[1])):
					func2_y=compute_wavefunction_n(num_points[1], start_point[1],end_point[1],D_gs[1],alpha_gs[1],mu[1],l,0.0)
					func2=np.zeros((func2_x.shape[0],func2_y.shape[0],3))
					for a in range(func2.shape[0]):
                                		for b in range(func2.shape[1]):
                                        		func2[a,b,0]=func2_x[a,0]
                                        		func2[a,b,1]=func2_y[b,0]
                                        		func2[a,b,2]=func2_x[a,1]*func2_y[b,1]
					#successfuly computed additional wavefunction
					eff_func=func2
					eff_func[:,:,2]=eff_func[:,:,2]*He_func1[:,:,2]
					index1=int(i*max_n_gs[1]+j)
					index2=int(k*max_n_gs[1]+l)
					overlap_matrix[index1,index2]=cumulant.simpson_integral_2D(eff_func)
					overlap_matrix[index2,index1]=overlap_matrix[index1,index2]
					print(i,j,k,l,overlap_matrix[index1,index2])
	return overlap_matrix

# compute the thermal average of the energy gap operator
def compute_omega_av_qm(He_matrix,freq_gs,D_gs,kbT):
	energy_list=np.zeros(He_matrix.shape[0])
	for i in range(energy_list.shape[0]):
		energy_list[i]=compute_morse_eval_n(freq_gs,D_gs,i)

	boltzmann_fac=np.exp(-energy_list/kbT)
	Z=np.sum(boltzmann_fac)
	omega_av=0.0
	
	for i in range(energy_list.shape[0]):
		omega_av=omega_av+(He_matrix[i,i]-energy_list[i])*boltzmann_fac[i]

	omega_av=omega_av/Z

	return omega_av

def compute_omega_av_qm_2D(He_matrix,freq_gs,D_gs,kbT,n_max_gs):
	energy_list=np.zeros(He_matrix.shape[0])
	for i in range(int(n_max_gs[0])):
		for j in range(int(n_max_gs[1])):
			index=int(i*n_max_gs[1]+j)
			energy_list[index]=compute_morse_eval_n(freq_gs[0],D_gs[0],i)+compute_morse_eval_n(freq_gs[1],D_gs[1],j)

	boltzmann_fac=np.exp(-energy_list/kbT)
	Z=np.sum(boltzmann_fac)
	omega_av=0.0

	for i in range(energy_list.shape[0]):
		omega_av=omega_av+(He_matrix[i,i]-energy_list[i])*boltzmann_fac[i]

	omega_av=omega_av/Z

	return omega_av

# compute the energy gap autocorrelation function for a given value of t. 
#@jit
def exact_corr_func_t(gs_energy_list,boltzmann_list, delta_U_matrix,t):
	pos_phase_mat=np.zeros((gs_energy_list.shape[0],gs_energy_list.shape[0]),dtype=np.complex_)
	neg_phase_mat=np.zeros((gs_energy_list.shape[0],gs_energy_list.shape[0]),dtype=np.complex_)
	for i in range(gs_energy_list.shape[0]):
		pos_phase_mat[i,i]=cmath.exp(1j*gs_energy_list[i]*t)
		neg_phase_mat[i,i]=cmath.exp(-1j*gs_energy_list[i]*t)


	Z=np.sum(boltzmann_list)
	corr_val=0.0+1j*0.0
	boltzmann_mat=np.zeros((boltzmann_list.shape[0],boltzmann_list.shape[0]))
	for i in range(boltzmann_mat.shape[0]):
		boltzmann_mat[i,i]=boltzmann_list[i]

	posU=np.dot(pos_phase_mat,delta_U_matrix)
	negU=np.dot(neg_phase_mat,delta_U_matrix)
	temp_mat=np.dot(posU,negU)
	tot_mat=np.dot(temp_mat,boltzmann_mat)	

	corr_val=np.trace(tot_mat)/Z
	return corr_val

# compute the exact two-time correlation function of the energy gap for a given value of t,t'.
def exact_corr_func_3rd_t(gs_energy_list,boltzmann_list,delta_U_matrix,t1,t2):
	pos_phase_mat_t1=np.zeros((gs_energy_list.shape[0],gs_energy_list.shape[0]),dtype=np.complex_)
	pos_phase_mat_t2=np.zeros((gs_energy_list.shape[0],gs_energy_list.shape[0]),dtype=np.complex_)
	for i in range(gs_energy_list.shape[0]):
		pos_phase_mat_t1[i,i]=cmath.exp(1j*gs_energy_list[i]*t1)
		pos_phase_mat_t2[i,i]=cmath.exp(1j*gs_energy_list[i]*t2)

	Z=np.sum(boltzmann_list)
	corr_val=0.0+1j*0.0
	boltzmann_mat=np.zeros((boltzmann_list.shape[0],boltzmann_list.shape[0]))
	for i in range(boltzmann_mat.shape[0]):
		boltzmann_mat[i,i]=boltzmann_list[i]

	t1t2_mat=np.dot(np.dot(np.conjugate(pos_phase_mat_t2),pos_phase_mat_t),delta_U_matrix)
	posU=np.dot(pos_phase_mat_t2,delta_U_mat)
	negU=np.dot(np.conjugate(pos_phase_mat_t1),deltaU_mat)
	
	temp_mat=np.dot(np.dot(negU,t1t2_mat),negU)
	tot_mat=np.dot(temp_mat,boltzmann_mat)

	corr_val=np.trace(tot_mat)/Z
	return corr_val

# compute exact correlation function of energy gap fluctuations.
def exact_corr_func(He_mat,D_gs,freq_gs,omega_av,kbT,num_points,max_t):
	gs_energy_list=np.zeros(He_mat.shape[0])
	delta_U_mat=He_mat
	for i in range(gs_energy_list.shape[0]):
		gs_energy_list[i]=compute_morse_eval_n(freq_gs,D_gs,i)
		delta_U_mat[i,i]=delta_U_mat[i,i]-gs_energy_list[i]		

	boltzmann_list=np.exp(-gs_energy_list/kbT)

	time_step=max_t/num_points
	corr_func=np.zeros((2*num_points+1,2),np.complex_)
	current_t=-max_t
	for i in range(corr_func.shape[0]):
		corr_func[i,0]=current_t
		corr_func[i,1]=exact_corr_func_t(gs_energy_list,boltzmann_list, delta_U_mat,current_t)
		current_t=current_t+time_step

	return corr_func

def exact_corr_func_3rd(He_mat,D_gs,freqs_gs,omega_av,kbT,num_points,max_t):
	gs_energy_list=np.zeros(He_mat.shape[0])
	delta_U_mat=He_mat
	for i in range(gs_energy_list.shape[0]):
		gs_energy_list[i]=compute_morse_eval_n(freq_gs,D_gs,i)
		delta_U_mat[i,i]=delta_U_mat[i,i]-gs_energy_list[i]
	boltzmann_list=np.exp(-gs_energy_list/kbT)

	time_step=max_t/num_points
	corr_func=np.zeros((2*num_points+1,2*num_points+1,3),np.complex_)
	t_start=-max_t

	for i in range(corr_func.shape[0]):
		for j in range(corr_func.shape[0]):
			corr_func[i,j,0]=t_start+i*time_step
			corr_func[i,j,1]=t_start+j*time_step
			corr_func[i,j,2]=exact_corr_func_3rd_t(gs_energy_list,boltzmann_list, delta_U_mat,corr_func[i,j,0],corr_func[i,j,1])

	return corr_func

# construct exact correlation func in the case we have a Duschinsky rotation coupling 2 modes
def exact_corr_func_2D(He_mat,D_gs,freq_gs,kbT,n_max_gs,num_points,max_t):
        gs_energy_list=np.zeros(He_mat.shape[0])
        delta_U_mat=He_mat
        for i in range(int(n_max_gs[0])):
                for j in range(int(n_max_gs[1])):
                        index=int(i*n_max_gs[1]+j)
                        gs_energy_list[index]=compute_morse_eval_n(freq_gs[0],D_gs[0],i)+compute_morse_eval_n(freq_gs[1],D_gs[1],j)
        for i in range(delta_U_mat.shape[0]):	
                delta_U_mat[i,i]=delta_U_mat[i,i]-gs_energy_list[i]

        boltzmann_list=np.exp(-gs_energy_list/kbT)

        time_step=max_t/num_points
        corr_func=np.zeros((2*num_points+1,2),np.complex_)
        current_t=-max_t
        for i in range(corr_func.shape[0]):
                corr_func[i,0]=current_t
                corr_func[i,1]=exact_corr_func_t(gs_energy_list,boltzmann_list, delta_U_mat,current_t)
                current_t=current_t+time_step

        return corr_func

# compute response function at a given point in time
@jit(fastmath=True)
def compute_morse_chi_func_t(freq_gs,D_gs,kbT,factors,energies,t):
        chi=0.0+0.0j
        Z=0.0 # partition function
        for n_gs in range(factors.shape[0]):
                Egs=compute_morse_eval_n(freq_gs,D_gs,n_gs)
                boltzmann=math.exp(-Egs/kbT)
                Z=Z+boltzmann
                for n_ex in range(factors.shape[1]):
                        chi=chi+boltzmann*factors[n_gs,n_ex]*cmath.exp(-1j*energies[n_gs,n_ex]*t)
	
        chi=chi/Z	

        return cmath.polar(chi)

@jit(fastmath=True)
def compute_morse_chi_func_2D_t(freq_gs,D_gs,n_max_gs,kbT,factors,energies,t):
	chi=0.0+0.0j
	Z=0.0
	for i in range(int(n_max_gs[0])):
		for j in range(int(n_max_gs[1])):
			E_gs=compute_morse_eval_n(freq_gs[0],D_gs[0],i)+compute_morse_eval_n(freq_gs[1],D_gs[1],j)
			boltzmann=np.exp(-E_gs/kbT)
			Z=Z+boltzmann
			gs_index=int(i*n_max_gs[1]+j)
			for n_ex in range(factors.shape[1]):
				chi=chi+boltzmann*factors[gs_index,n_ex]*cmath.exp(-1j*energies[gs_index,n_ex]*t)
	chi=chi/Z
	return cmath.polar(chi)

def compute_exact_response_func_2D(factors,energies,freq_gs,n_max_gs,D_gs,kbT,max_t,num_steps):
        step_length=max_t/num_steps
        # end fc integral definition
        chi_full=np.zeros((num_steps,3))
        response_func = np.zeros((num_steps, 2), dtype=np.complex_)
        current_t=0.0
        for counter in range(num_steps):
                chi_t=compute_morse_chi_func_2D_t(freq_gs,D_gs,n_max_gs,kbT,factors,energies,current_t)
                chi_full[counter,0]=current_t
                chi_full[counter,1]=chi_t[0]
                chi_full[counter,2]=chi_t[1]
                current_t=current_t+step_length

        # now make sure that phase is a continuous function:
        phase_fac=0.0
        for counter in range(num_steps-1):
                chi_full[counter,2]=chi_full[counter,2]+phase_fac
                if abs(chi_full[counter,2]-phase_fac-chi_full[counter+1,2])>0.7*math.pi: #check for discontinuous jump.
                        diff=chi_full[counter+1,2]-(chi_full[counter,2]-phase_fac)
                        frac=diff/math.pi
                        n=int(round(frac))
                        phase_fac=phase_fac-math.pi*n
                chi_full[num_steps-1,2]=chi_full[num_steps-1,2]+phase_fac

        # now construct response function
        for counter in range(num_steps):
                response_func[counter,0]=chi_full[counter,0]
                response_func[counter,1]=chi_full[counter,1]*cmath.exp(1j*chi_full[counter,2])

        return response_func

def compute_exact_response_func(factors,energies,freq_gs,D_gs,kbT,max_t,num_steps):
        step_length=max_t/num_steps
        # end fc integral definition
        chi_full=np.zeros((num_steps,3))
        response_func = np.zeros((num_steps, 2), dtype=np.complex_)
        current_t=0.0
        for counter in range(num_steps):
                chi_t=compute_morse_chi_func_t(freq_gs,D_gs,kbT,factors,energies,current_t)
                chi_full[counter,0]=current_t
                chi_full[counter,1]=chi_t[0]
                chi_full[counter,2]=chi_t[1]
                current_t=current_t+step_length

        # now make sure that phase is a continuous function:
        phase_fac=0.0
        for counter in range(num_steps-1):
                chi_full[counter,2]=chi_full[counter,2]+phase_fac
                if abs(chi_full[counter,2]-phase_fac-chi_full[counter+1,2])>0.7*math.pi: #check for discontinuous jump.
                        diff=chi_full[counter+1,2]-(chi_full[counter,2]-phase_fac)
                        frac=diff/math.pi
                        n=int(round(frac))
                        phase_fac=phase_fac-math.pi*n
                chi_full[num_steps-1,2]=chi_full[num_steps-1,2]+phase_fac

	# now construct response function
        for counter in range(num_steps):
                response_func[counter,0]=chi_full[counter,0]
                response_func[counter,1]=chi_full[counter,1]*cmath.exp(1j*chi_full[counter,2])

        return response_func

# some common functions:
@jit(fastmath=True)
def compute_morse_eval_n(omega,D,n):
        return omega*(n+0.5)-(omega*(n+0.5))**2.0/(4.0*D)

def find_classical_turning_points_morse(n_max_gs,n_max_ex,freq_gs,freq_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift):
	E_max_gs=compute_morse_eval_n(freq_gs,D_gs,n_max_gs) # compute the energies for the highest energy morse 
	E_max_ex=compute_morse_eval_n(freq_ex,D_ex,n_max_ex) # state considered

	# find the two classical turning points for the ground state PES
	point1_gs=math.log(math.sqrt(E_max_gs/D_gs)+1.0)/(-alpha_gs)
	point2_gs=math.log(-math.sqrt(E_max_gs/D_gs)+1.0)/(-alpha_gs)

	# same for excited state. Include shift vector
	point1_ex=math.log(math.sqrt(E_max_ex/D_ex)+1.0)/(-alpha_ex)+shift
	point2_ex=math.log(-math.sqrt(E_max_ex/D_ex)+1.0)/(-alpha_ex)+shift

	# now find the smallest value and the largest value
	start_point=min(point1_gs,point2_gs)
	end_point=max(point1_ex,point2_ex)

	return start_point,end_point

# returns an array containing the nth order wavefunction of either the ground state or 
# excited state potential energy surface. Allows for the inclusion of a shift vector
def compute_wavefunction_n(num_points, start_point,end_point,D,alpha,mu,n,shift):
        # first start by filling array with position points:
        wavefunc=np.zeros((num_points,2))
        lamda=math.sqrt(2.0*D*mu)/(alpha)
        step_x=(end_point-start_point)/num_points
        denom=special.gamma(2.0*lamda-n)
        if np.isinf(denom):
                denom=10e280
        num=(math.factorial(n)*(2.0*lamda-2.0*n-1.0))
        normalization=math.sqrt(num/denom)
        counter=0
        for x in wavefunc:
                x[0]=start_point+counter*step_x
                r_val=(start_point+counter*step_x)*alpha
                r_shift_val=(shift)*alpha
                z_val=2.0*lamda*math.exp(-(r_val-r_shift_val))
                func_val=normalization*z_val**(lamda-n-0.5)*math.exp(-0.5*z_val)*special.assoc_laguerre(z_val,n,2.0*lamda-2.0*n-1.0)
                x[1]=func_val
                counter=counter+1

	# fix normalization regardless of value of denominator to avoid rounding errors
        wavefunc_sq=np.zeros(wavefunc.shape[0])
        wavefunc_sq[:]=wavefunc[:,1]*wavefunc[:,1]

        normalization=integrate.simps(wavefunc_sq,dx=step_x)
        for counter in range(wavefunc.shape[0]):
                wavefunc[counter,1]=wavefunc[counter,1]/math.sqrt(normalization)

        return wavefunc

# compute effective excited state wavefunction for a Duschinsky-coupled excited state PES
# this assumes we have exactly 2 normal modes
def compute_excited_state_wf_2D(num_points,start_point,end_point,D,alpha,mu,n,shift,J):
	wavefunc=np.zeros((num_points[0],num_points[1],3))
	lamda=np.zeros(2)
	lamda[0]=math.sqrt(2.0*D[0]*mu[0])/(alpha[0])
	lamda[1]=math.sqrt(2.0*D[1]*mu[1])/(alpha[1])
	step_x=(end_point[0]-start_point[0])/num_points[0]
	step_y=(end_point[1]-start_point[1])/num_points[1]
	denom=special.gamma(2.0*lamda[0]-n[0])*special.gamma(2.0*lamda[1]-n[1])
	if np.isinf(denom):
		denom=10e280
	num=(math.factorial(n[0])*(2.0*lamda[0]-2.0*n[0]-1.0))*(math.factorial(n[1])*(2.0*lamda[1]-2.0*n[1]-1.0))
	if np.isinf(num):
		num=10e280
	normalization=math.sqrt(num/denom)
	for i in range(wavefunc.shape[0]):
		for j in range(wavefunc.shape[1]):
			x=np.zeros(2)
			x[0]=start_point[0]+i*step_x
			x[1]=start_point[1]+j*step_y
			wavefunc[i,j,0]=x[0]
			wavefunc[i,j,1]=x[1]
			temp=x-shift
			eff_x=np.dot(np.transpose(J),temp) # apply Duschinsky rotation
			rval=alpha*eff_x
			z_val=2.0*lamda*np.exp(-rval)
			func_val_x=z_val[0]**(lamda[0]-n[0]-0.5)*math.exp(-0.5*z_val[0])*special.assoc_laguerre(z_val[0],n[0],2.0*lamda[0]-2.0*n[0]-1.0)
			func_val_y=z_val[1]**(lamda[1]-n[1]-0.5)*math.exp(-0.5*z_val[1])*special.assoc_laguerre(z_val[1],n[1],2.0*lamda[1]-2.0*n[1]-1.0)
			wavefunc[i,j,2]=normalization*func_val_x*func_val_y

	# fix normalization regardless of value of denominator to avoid rounding errors
	wavefunc_sq=np.zeros((wavefunc.shape[0],wavefunc.shape[1],3))
	for i in range(wavefunc.shape[0]):
		for j in range(wavefunc.shape[1]):
			wavefunc_sq[i,j,0]=wavefunc[i,j,0]
			wavefunc_sq[i,j,1]=wavefunc[i,j,1]
			wavefunc_sq[i,j,2]=wavefunc[i,j,2]*wavefunc[i,j,2]

	normalization=cumulant.simpson_integral_2D(wavefunc_sq)
	wavefunc[:,:,2]=wavefunc[:,:,2]/np.sqrt(normalization)

	return wavefunc


# compute the effective Franck-Condon wavefunction overlap between two wavefunctions
# This shouldnt be squared!
def get_fc_factor(num_points,start_point,end_point,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,n_gs,n_ex):
        #print start_point,end_point,D,alpha,n_gs,n_ex,d
        func1=compute_wavefunction_n(num_points, start_point,end_point,D_gs,alpha_gs,mu,n_gs,0.0)
        func2=compute_wavefunction_n(num_points, start_point,end_point,D_ex,alpha_ex,mu,n_ex,K)
        counter=0
        for x in func1:
                x[1]=x[1]*func2[counter,1]
                counter=counter+1

        return integrate.simps(func1[:,1],dx=func1[1,0]-func1[0,0])


# 2D version of get_fc_factor. NOTe that WF overlaps are Not squared
def get_fc_factor_2D(num_points,start_point,end_point,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,J,n_gs,n_ex):
	# n_gs and n_ex are vectors
	# first build 2D gs wavefunc
	#print(num_points[0], start_point[0],end_point[0],D_gs[0],alpha_gs[0],mu[0],n_gs[0])
	func_x=compute_wavefunction_n(num_points[0], start_point[0],end_point[0],D_gs[0],alpha_gs[0],mu[0],n_gs[0],0.0)
	func_y=compute_wavefunction_n(num_points[1], start_point[1],end_point[1],D_gs[1],alpha_gs[1],mu[1],n_gs[1],0.0)
	gs_wavefunc=np.zeros((func_x.shape[0],func_y.shape[0],3))
	for i in range(gs_wavefunc.shape[0]):
		for j in range(gs_wavefunc.shape[1]):
			gs_wavefunc[i,j,0]=func_x[i,0]
			gs_wavefunc[i,j,1]=func_y[j,0]
			gs_wavefunc[i,j,2]=func_x[i,1]*func_y[j,1]

	#twoDES.print_2D_spectrum('2D_gs_wavefunction.dat',gs_wavefunc,False)

	ex_wavefunc=compute_excited_state_wf_2D(num_points,start_point,end_point,D_ex,alpha_ex,mu,n_ex,K,J)
	#twoDES.print_2D_spectrum('2D_ex_wavefunction.dat',ex_wavefunc,False)
	overlap=gs_wavefunc
	overlap[:,:,2]=overlap[:,:,2]*ex_wavefunc[:,:,2]
	return cumulant.simpson_integral_2D(overlap)

class morse: 
	def __init__(self,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,E_adiabatic,dipole_mom,max_states_gs,max_states_ex,num_points):
		# Adiabatic energy gap, dipole moment, ground and excited state well depths D
		# and alpha parameters, as well as the shift vector K between the two surfaces
		self.alpha_gs=alpha_gs
		self.alpha_ex=alpha_ex
		self.D_gs=D_gs
		self.D_ex=D_ex
		self.mu=mu # reduced mass
		self.K=K
		self.E_adiabatic=E_adiabatic
		self.dipole_mom=dipole_mom

		# derived quantities:
		# effective harmonic ground and excited state frequencies
		self.freq_gs=0.0
		self.freq_ex=0.0

		# variables defining the numerical grid over which we compute and integrate
		# ground and excited state wavefunctions
		self.grid_start=0.0
		self.grid_end=0.0
		self.grid_n_points=num_points
		self.grid_step=0.0   # step length of the grid 
		
		# number of states in the ground and excited state that are considered
		# this number is either max_states, or the number of bound states in the 
		# specific morse oscillator, whatever parameter is smaller
		self.n_max_gs=max_states_gs
		self.n_max_ex=max_states_ex


		# values needed for the calculation of quantum correlation functions
		self.omega_av_cl=0.0
		self.omega_av_qm=0.0 # this value does NOT include E_adiabatic

		# set derived absorption vactiables, mainly n_max and freq
		self.set_absorption_variables()


		# modulus squared of the overlaps of nuclear wavefunctions between ground
		# and excited state PES
		self.wf_overlaps=np.zeros((self.n_max_gs,self.n_max_ex))
		self.wf_overlaps_sq=np.zeros((self.n_max_gs,self.n_max_ex))
		self.transition_energies=np.zeros((self.n_max_gs,self.n_max_ex))
		self.gs_energies=np.zeros(self.n_max_gs)
		self.boltzmann_fac=np.zeros(self.n_max_gs)
		self.ex_energies=np.zeros(self.n_max_ex)	
	
		# effective overlap between ground state wavefunctions and excited state hamiltonian
		self.He_mat=np.zeros((self.n_max_gs,self.n_max_gs))

		# response functions 
		self.exact_response_func=np.zeros((1,1),dtype=np.complex_)

		# cumulant parameters
		self.exact_2nd_order_corr=np.zeros((1,1),dtype=np.complex_)
		self.exact_3rd_order_corr=np.zeros((1,1,1),dtype=np.complex_)
		self.exact_g2=np.zeros((1,1),dtype=np.complex_)

	def compute_boltzmann_fac(self,temp):
		kbT=const.kb_in_Ha*temp
		self.boltzmann_fac=np.exp(-self.gs_energies/kbT)	
	
	def compute_exact_corr_3rd(self,temp,num_points,max_t):
		kbT=const.kb_in_Ha*temp
		self.He_mat=gs_wavefunc_He_matrix(self.grid_n_points,self.grid_start,self.grid_end,self.D_gs,self.D_ex,self.alpha_gs,self.alpha_ex,self.K,self.mu,self.E_adiabatic,self.n_max_gs)
		self.omega_av_qm= compute_omega_av_qm(self.He_mat,self.freq_gs,self.D_gs,kbT)
		self.exact_3rd_order_corr=exact_corr_func_3rd(self.He_mat,self.D_gs,self.freq_gs,self.omega_av_qm,kbT,num_points,max_t)

	# Now declare functions of the Morse oscillator 
	def compute_exact_corr(self,temp,num_points,max_t):
		kbT=const.kb_in_Ha*temp
		self.He_mat=gs_wavefunc_He_matrix(self.grid_n_points,self.grid_start,self.grid_end,self.D_gs,self.D_ex,self.alpha_gs,self.alpha_ex,self.K,self.mu,self.E_adiabatic,self.n_max_gs)
		self.omega_av_qm= compute_omega_av_qm(self.He_mat,self.freq_gs,self.D_gs,kbT)
		self.exact_2nd_order_corr=exact_corr_func(self.He_mat,self.D_gs,self.freq_gs,self.omega_av_qm,kbT,num_points,max_t)

	def set_absorption_variables(self):
		self.freq_gs=math.sqrt(2.0*self.D_gs*self.alpha_gs**2.0/self.mu)
		self.freq_ex=math.sqrt(2.0*self.D_ex*self.alpha_ex**2.0/self.mu)
			
		# calculate number of bound states
		nbound_gs=int((2.0*self.D_gs-self.freq_gs)/self.freq_gs)
		nbound_ex=int((2.0*self.D_ex-self.freq_ex)/self.freq_ex)
			
		if nbound_gs<self.n_max_gs:
			self.n_max_gs=nbound_gs
		if nbound_ex<self.n_max_ex:
			self.n_max_ex=nbound_ex

		# now define numerical grid. Find classical turning points
		# on ground and excited state PES
		start_point,end_point=find_classical_turning_points_morse(self.n_max_gs,self.n_max_ex,self.freq_gs,self.freq_ex,self.alpha_gs,self.alpha_ex,self.D_gs,self.D_ex,self.K)

		cl_range=end_point-start_point
		# make sure that the effective qm range is 10% larger than the effective classical range
		# to account for tunneling effects 
		self.grid_start=start_point-0.05*cl_range
		self.grid_end=end_point+0.05*cl_range
		self.grid_step=(self.grid_end-self.grid_start)/self.grid_n_points

	def compute_exact_response(self,temp,max_t,num_steps):
		kbT=const.kb_in_Ha*temp
		self.compute_overlaps_and_transition_energies()
		self.exact_response_func=compute_exact_response_func(self.wf_overlaps_sq,self.transition_energies,self.freq_gs,self.D_gs,kbT,max_t,num_steps)

	# now compute all wavefunction overlaps and Transition energies 
	def compute_overlaps_and_transition_energies(self):
		for i in range(self.n_max_gs):
			self.gs_energies[i]=compute_morse_eval_n(self.freq_gs,self.D_gs,i)
			for j in range(self.n_max_ex):
				self.ex_energies[j]=compute_morse_eval_n(self.freq_ex,self.D_ex,j)+self.E_adiabatic
				self.transition_energies[i,j]=self.transition_energy(i,j)
				self.wf_overlaps[i,j]=get_fc_factor(self.grid_n_points,self.grid_start,self.grid_end,self.D_gs,self.D_ex,self.alpha_gs,self.alpha_ex,self.mu,self.K,i,j)
				self.wf_overlaps_sq[i,j]=self.wf_overlaps[i,j]**2.0

		print('Transition energies relative to GS')
		print(self.transition_energies[0,:])


	# calculate transition energy between two specific morse oscillators.
	def transition_energy(self,n_gs,n_ex):
		E_gs=compute_morse_eval_n(self.freq_gs,self.D_gs,n_gs)
		E_ex=compute_morse_eval_n(self.freq_ex,self.D_ex,n_ex)

		return E_ex-E_gs

# Now define a class for a batch of independent Morse oscillators. This only works if the Morse oscillators are
# NOT coupled through a Duschinsky type rotation. 
class morse_list:
	def __init__(self,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,E_adiabatic,dipole_mom,max_states_gs,max_states_ex,num_points,num_morse_oscillators,stdout):
		self.morse_oscs = []
		for i in range(num_morse_oscillators):
			self.morse_oscs.append(morse(D_gs[i],D_ex[i],alpha_gs[i],alpha_ex[i],mu[i],K[i],E_adiabatic,dipole_mom,max_states_gs,max_states_ex,num_points))
		self.num_morse_oscillators=num_morse_oscillators

		# GBOM parameters obtained from doing a harmonic expansion of the ground and excited state
		# PES. These are temporary variables that get filled and an effective GBOM is created 
		gs_freqs=np.zeros(self.num_morse_oscillators)
		ex_freqs=np.zeros(self.num_morse_oscillators)
		# Jmat is always taken to be the identity matrix for the time being
		Jmat=np.zeros((self.num_morse_oscillators,self.num_morse_oscillators))
		eff_shift_vec_GBOM=np.zeros(self.num_morse_oscillators)

		# fill effective GBOM parameters:
		for i in range(self.num_morse_oscillators):
			Jmat[i,i]=1.0
			gs_freqs[i]=self.morse_oscs[i].freq_gs
			ex_freqs[i]=self.morse_oscs[i].freq_ex
			eff_shift_vec_GBOM[i]=self.morse_oscs[i].K/self.morse_oscs[i].freq_gs # need to convert this to a dimensionless shift vec

		eff_0_0=E_adiabatic-0.5*np.sum(gs_freqs)+0.5*np.sum(ex_freqs)
		self.eff_gbom=gbom.gbom(gs_freqs,ex_freqs,Jmat,eff_shift_vec_GBOM,eff_0_0,dipole_mom,stdout)

		self.total_exact_response_func=np.zeros((1,1),dtype=np.complex_)
		self.harmonic_fc_response_func=np.zeros((1,1),dtype=np.complex_)
		self.harmonic_cumulant_response_func=np.zeros((1,1),dtype=np.complex_)
		self.cumulant_response_func=np.zeros((1,1),dtype=np.complex_)
		self.hybrid_cumul_fc_response_func=np.zeros((1,1),dtype=np.complex_)
		self.E_adiabatic=E_adiabatic

		self.omega_av_qm=0.0
		self.omega_av_cl=0.0

		# cumulant
		self.exact_2nd_order_corr=np.zeros((1,1),dtype=np.complex_)
		self.exact_2nd_order_corr_freq=np.zeros((1,1),dtype=np.complex_)
		self.exact_3rd_order_corr=np.zeros((1,1,1),dtype=np.complex_)
		self.exact_3rd_order_corr_freq=np.zeros((1,1,1),dtype=np.complex_)
		self.spectral_dens=np.zeros((1,1))
		self.g2_exact=np.zeros((1,1),dtype=np.complex_)

		
	def compute_total_exact_response(self,temp,max_t,num_steps):
		for i in range(self.num_morse_oscillators):
			self.morse_oscs[i].compute_exact_response(temp,max_t,num_steps)
			print('Computed response func!')
			print(self.morse_oscs[i].exact_response_func)
			if i==0:
				self.total_exact_response_func=self.morse_oscs[i].exact_response_func

			else:
				for j in range(self.total_exact_response_func.shape[0]):
					self.total_exact_response_func[j,1]=self.total_exact_response_func[j,1]*self.morse_oscs[i].exact_response_func[j,1]

		# shift final response function by the adiabatic energy gap
		for j in range(self.total_exact_response_func.shape[0]):
			self.total_exact_response_func[j,1]=self.total_exact_response_func[j,1]*cmath.exp(-1j*self.E_adiabatic*self.total_exact_response_func[j,0])

	# routine computes the exact 3rd order correlation function and its Fourier transform
	def compute_total_corr_func_3rd_exact(self,temp,decay_length,max_t,num_steps):
		for i in range(self.num_morse_oscillators):
			self.morse_oscs[i].compute_exact_corr_3rd(temp,num_steps,max_t)
			self.omega_av_qm=self.omega_av_qm+(self.morse_oscs[i].omega_av_qm)
			if i==0:
				self.exact_3rd_order_corr=self.morse_oscs[i].exact_3rd_order_corr
			else:
				self.exact_3rd_order_corr[:,:,2]=self.exact_3rd_order_corr[:,:,2]+self.morse_oscs[i].exact_3rd_order_corr[:,:,2]
		skew=(np.sqrt(np.sum(np.real(self.exact_2nd_order_corr[:,1]))/(1.0*self.exact_2nd_order_corr.shape[0])))**3.0
		
		self.exact_3rd_order_corr[:,:,2]=self.exact_3rd_order_corr[:,:,2]-skew

		# successfully computed the total 3rd order correlation function. Now need to construct its Fourier
		# transform. 


	# routine computes the exact correlation function and its Fourier transform
	def compute_total_corr_func_exact(self,temp,decay_length,max_t,num_steps):
		for i in range(self.num_morse_oscillators):
			self.morse_oscs[i].compute_exact_corr(temp,num_steps,max_t)
			self.omega_av_qm=self.omega_av_qm+(self.morse_oscs[i].omega_av_qm)
			if i==0:
				self.exact_2nd_order_corr=self.morse_oscs[i].exact_2nd_order_corr
			else:
				self.exact_2nd_order_corr[:,1]=self.exact_2nd_order_corr[:,1]+self.morse_oscs[i].exact_2nd_order_corr[:,1]

		# mean_sq has to be real
		mean_sq=np.sum(np.real(self.exact_2nd_order_corr[:,1]))/(1.0*self.exact_2nd_order_corr.shape[0])

		# subtract the square of the total omega_av_qm from the values
		self.exact_2nd_order_corr[:,1]=self.exact_2nd_order_corr[:,1]-mean_sq

		# add a decaying exponential to make Fourier transforms well-behaved
		for i in range(self.exact_2nd_order_corr.shape[0]):
			self.exact_2nd_order_corr[i,1]=self.exact_2nd_order_corr[i,1]*np.exp(-abs(np.real(self.exact_2nd_order_corr[i,0]))/decay_length)
		np.savetxt('Morse_exact_corr_func_real.dat',np.real(self.exact_2nd_order_corr))
		print('Average energy gap Morse: '+str(self.E_adiabatic+np.sqrt(mean_sq)))

		self.omega_av_qm=self.omega_av_qm+self.E_adiabatic # add E_adiabatic to get the total average energy gap 

		# compute the Fourier transform of the correlation function.
		corr_freq=max_t/num_steps*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.exact_2nd_order_corr[:,1])))
		sample_rate=1.0/(max_t/num_steps)*math.pi*2.0
		freqs=np.fft.fftshift(np.fft.fftfreq(corr_freq.size,d=1.0/sample_rate))
		self.exact_2nd_order_corr_freq=np.zeros((corr_freq.size,2),dtype=np.complex_)
		for i in range(self.exact_2nd_order_corr_freq.shape[0]):
			self.exact_2nd_order_corr_freq[i,0]=freqs[i]
			self.exact_2nd_order_corr_freq[i,1]=corr_freq[i]

		np.savetxt('Morse_exact_corr_func_freq_real.dat',np.real(self.exact_2nd_order_corr_freq))

	def compute_spectral_dens(self):
		step_length=np.real(self.exact_2nd_order_corr[1,0]-self.exact_2nd_order_corr[0,0])
		corr_func_freq=np.fft.fft(np.fft.ifftshift(self.exact_2nd_order_corr[:, 1].imag))
		freqs = np.fft.fftfreq(self.exact_2nd_order_corr.shape[0], step_length)*2.0*math.pi
		self.spectral_dens = np.zeros((int((corr_func_freq.shape[0])/2) - 1, 2))
		counter = 0
		while counter < self.spectral_dens.shape[0]:
			self.spectral_dens[counter, 0] = freqs[counter]
			self.spectral_dens[counter, 1] = -step_length*np.real(1j*corr_func_freq[counter]) 
			counter = counter + 1

	def compute_g2_exact(self,temp,max_t,num_points,stdout):
		kbT=const.kb_in_Ha*temp
		self.g2_exact=cumulant.compute_2nd_order_cumulant_from_spectral_dens(self.spectral_dens,kbT,max_t,num_points,stdout)

	def compute_2nd_order_cumulant_response(self,temp,max_t,num_points,stdout):
		kbT=const.kb_in_Ha*temp	
		#self.g2_exact=g2_from_corr_func(self.exact_2nd_order_corr_freq,num_points,max_t)
		self.g2_exact=cumulant.compute_2nd_order_cumulant_from_spectral_dens(self.spectral_dens,kbT,max_t,num_points,stdout)
		self.cumulant_response_func=gbom_cumulant_response.compute_cumulant_response(self.g2_exact, np.zeros((1,1)),self.eff_gbom.dipole_mom,np.zeros((1,1)),False,False, False)
		for i in range(self.cumulant_response_func.shape[0]):
			self.cumulant_response_func[i,1]=self.cumulant_response_func[i,1]*cmath.exp(-1j*self.cumulant_response_func[i,0]*self.omega_av_qm)

	def compute_harmonic_FC_response_func(self,temp,max_t,num_steps,is_emission,is_HT,stdout):
		self.eff_gbom.calc_fc_response(temp,num_steps,max_t,is_emission,is_HT,stdout)
		self.harmonic_fc_response_func=self.eff_gbom.fc_response

	def compute_harmonic_exact_cumulant_response_func(self,temp,max_t,num_steps,is_emission,stdout):
		self.eff_gbom.calc_g2_qm(temp,num_steps,max_t,is_emission,stdout)
		self.eff_gbom.calc_cumulant_response(False,True,is_emission,False) # no HT option for the time being
		self.harmonic_cumulant_response_func=self.eff_gbom.cumulant_response

	def compute_cumul_fc_hybrid_response_func(self,temp,decay_length,max_t,num_steps,is_emission,stdout):
		# first compute effective response functions for the GBOM under the harmonic approximation of the PES
		self.compute_harmonic_FC_response_func(temp,max_t,num_steps,is_emission,False,stdout)  # disable HT option for now
		self.compute_harmonic_exact_cumulant_response_func(temp,max_t,num_steps,is_emission,stdout)
	
		# now compute the 2nd order cumulant response function for the morse oscillator
		# start by computing the classical correlation function
		self.compute_total_corr_func_exact(temp,decay_length,max_t,num_steps)
		# now compute the spectral density
		self.compute_spectral_dens()
		# finally compute 2nd order cumulant response
		self.compute_2nd_order_cumulant_response(temp,max_t,num_steps,stdout)

		# now can construct the full hybrid response function
		self.hybrid_cumul_fc_response_func=self.cumulant_response_func
		self.hybrid_cumul_fc_response_func[:,1]=self.hybrid_cumul_fc_response_func[:,1]*self.eff_gbom.fc_response[:,1]/self.eff_gbom.cumulant_response[:,1]

# Class description for several morse oscillators that are coupled by a Duschinsky rotation. Evaluating the FC integrals for this 
# is very expensive, so usage of this class should be limited to a couple of modes. It can however be combined with Morse_list, to 
# describe a system with a large number of anharmonic modes, only a few of which are coupled by a Duschinsky rotation. For now, limit
# the discussion to only two modes 
class morse_coupled:
        def __init__(self,D_gs,D_ex,alpha_gs,alpha_ex,mu,J,K,E_adiabatic,dipole_mom,max_states_gs,max_states_ex,num_points,num_morse_oscillators,stdout):
                if num_morse_oscillators!=2:	
                        sys.exit('Error: Coupled Morse oscillator code currently only implemented for exactly 2 coupled oscillators.')
                self.morse_oscs = []
                for i in range(num_morse_oscillators):
                        self.morse_oscs.append(morse(D_gs[i],D_ex[i],alpha_gs[i],alpha_ex[i],mu[i],K[i],E_adiabatic,dipole_mom,max_states_gs,max_states_ex,num_points))
                self.num_morse_oscillators=num_morse_oscillators

                # GBOM parameters obtained from doing a harmonic expansion of the ground and excited state
                # PES. 
                self.gs_freqs=np.zeros(self.num_morse_oscillators)
                self.ex_freqs=np.zeros(self.num_morse_oscillators)
                self.D_gs=D_gs
                self.D_ex=D_ex
                self.alpha_gs=alpha_gs
                self.alpha_ex=alpha_ex
                self.J=J
                self.K=K
                self.mu=mu
                self.E_adiabatic=E_adiabatic

		# variables defining the grid:
                self.grid_start=np.array([self.morse_oscs[0].grid_start,self.morse_oscs[1].grid_start])
                self.grid_end=np.array([self.morse_oscs[0].grid_end,self.morse_oscs[1].grid_end])
                self.grid_n_points=np.array([num_points,num_points])

		# variables defining number of bound states per morse oscillator
                self.n_max_gs=np.zeros(self.num_morse_oscillators)
                self.n_max_ex=np.zeros(self.num_morse_oscillators)

                # Jmat is always taken to be the identity matrix for the time being
                eff_shift_vec_GBOM=np.zeros(self.num_morse_oscillators)

                for i in range(self.num_morse_oscillators):
                        self.gs_freqs[i]=self.morse_oscs[i].freq_gs
                        self.ex_freqs[i]=self.morse_oscs[i].freq_ex
                        self.n_max_gs[i]=self.morse_oscs[i].n_max_gs
                        self.n_max_ex[i]=self.morse_oscs[i].n_max_ex
                        eff_shift_vec_GBOM[i]=self.morse_oscs[i].K/self.morse_oscs[i].freq_gs # need to convert this to a dimensionless shift vec


                eff_0_0=E_adiabatic-0.5*np.sum(self.gs_freqs)+0.5*np.sum(self.ex_freqs)
                self.eff_gbom=gbom.gbom(self.gs_freqs,self.ex_freqs,self.J,eff_shift_vec_GBOM,eff_0_0,dipole_mom,stdout)

		# all possible wf overlaps for a combination of Morse wavefunctions on the ground and excited state PES saved in a N*N matrix
                self.wf_overlaps=np.zeros((int(self.n_max_gs[0]*self.n_max_gs[1]),int(self.n_max_ex[0]*self.n_max_ex[1])))
                self.wf_overlaps_sq=np.zeros((int(self.n_max_gs[0]*self.n_max_gs[1]),int(self.n_max_ex[0]*self.n_max_ex[1])))
                self.transition_energies=np.zeros((int(self.n_max_gs[0]*self.n_max_gs[1]),int(self.n_max_ex[0]*self.n_max_ex[1])))
                self.boltmann_fac=np.zeros(int(self.n_max_gs[0]*self.n_max_gs[1]))
                self.gs_energies=np.zeros(int(self.n_max_gs[0]*self.n_max_gs[1]))
                self.ex_energies=np.zeros(int(self.n_max_ex[0]*self.n_max_ex[1]))

		# He mat needed for exact response func
                self.He_mat=np.zeros((int(self.n_max_gs[0]*self.n_max_gs[1]),int(self.n_max_gs[0]*self.n_max_gs[1])))		

		# response functions:
                self.exact_response_func=np.zeros((1,1),dtype=np.complex_)
                self.harmonic_FC_response_func=np.zeros((1,1),dtype=np.complex_)
                self.harmonic_cumulant_response_func=np.zeros((1,1),dtype=np.complex_)
                self.cumulant_response_func=np.zeros((1,1),dtype=np.complex_)
                self.hybrid_cumul_response_func=np.zeros((1,1),dtype=np.complex_)

		# Exact cumulant fucntion
                self.exact_2nd_order_corr=np.zeros((1,1),dtype=np.complex_)
                self.spectal_dens=np.zeros((1,1))	
                self.g2_exact=np.zeros((1,1),dtype=np.complex_)
	
        def compute_boltzmann_fac(self,temp):
                kbT=const.kb_in_Ha*temp
                self.boltmann_fac=np.exp(-self.gs_energies/kbT)

        def compute_wf_overlaps_energies(self):
                print('COMPUTING WF overlaps:')
                for i in range(int(self.n_max_gs[0])):
                        print(str(i)+'  of  '+str(self.n_max_gs[0])+'  completed.')
                        for j in range(int(self.n_max_gs[1])):
                                for k in range(int(self.n_max_ex[0])):
                                        for l in range(int(self.n_max_ex[1])):
                                                index_gs=int(i*self.n_max_gs[1]+j)
                                                index_ex=int(k*self.n_max_ex[1]+l)
                                                n_gs=np.array([i,j])
                                                n_ex=np.array([k,l])
                                                self.transition_energies[index_gs,index_ex]=self.transition_energy(n_gs,n_ex)
                                                self.wf_overlaps[index_gs,index_ex]=get_fc_factor_2D(self.grid_n_points,self.grid_start,self.grid_end,self.D_gs,self.D_ex,self.alpha_gs,self.alpha_ex,self.mu,self.K,self.J,n_gs,n_ex)
                                                self.wf_overlaps_sq[index_gs,index_ex]=self.wf_overlaps[index_gs,index_ex]**2.0

                                                self.gs_energies[index_gs]=compute_morse_eval_n(self.gs_freqs[0],self.D_gs[0],n_gs[0])+compute_morse_eval_n(self.gs_freqs[1],self.D_gs[1],n_gs[1])
                                                self.ex_energies[index_ex]=compute_morse_eval_n(self.ex_freqs[0],self.D_ex[0],n_ex[0])+compute_morse_eval_n(self.ex_freqs[1],self.D_ex[1],n_ex[1])+self.E_adiabatic
                                                print(i,j,k,l,self.wf_overlaps[index_gs,index_ex])


	# calculate transition energy between two specific morse oscillators.
        def transition_energy(self,n_gs,n_ex):
                E_gs=compute_morse_eval_n(self.gs_freqs[0],self.D_gs[0],n_gs[0])+compute_morse_eval_n(self.gs_freqs[1],self.D_gs[1],n_gs[1])
                E_ex=compute_morse_eval_n(self.ex_freqs[0],self.D_ex[0],n_ex[0])+compute_morse_eval_n(self.ex_freqs[1],self.D_ex[1],n_ex[1])

                return E_ex-E_gs+self.E_adiabatic

        def compute_harmonic_FC_response_func(self,temp,max_t,num_steps,is_emission,is_HT,stdout):
                self.eff_gbom.calc_fc_response(temp,num_steps,max_t,is_emission,is_HT,stdout)
                self.harmonic_fc_response_func=self.eff_gbom.fc_response

        def compute_harmonic_exact_cumulant_response_func(self,temp,max_t,num_steps,is_emission,stdout):
                self.eff_gbom.calc_g2_qm(temp,num_steps,max_t,is_emission,stdout)
                self.eff_gbom.calc_cumulant_response(False,True,is_emission,False)
                self.harmonic_cumulant_response_func=self.eff_gbom.cumulant_response

	        # Now declare functions of the Morse oscillator 
        def compute_exact_corr(self,temp,decay_length,num_points,max_t):
                kbT=const.kb_in_Ha*temp
                self.He_mat=gs_wavefunc_He_matrix_2D(self.grid_n_points,self.grid_start,self.grid_end,self.D_gs,self.D_ex,self.alpha_gs,self.alpha_ex,self.J,self.K,self.mu,self.E_adiabatic,self.n_max_gs)
                self.omega_av_qm=compute_omega_av_qm_2D(self.He_mat,self.gs_freqs,self.D_gs,kbT,self.n_max_gs)
                self.exact_2nd_order_corr=exact_corr_func_2D(self.He_mat,self.D_gs,self.gs_freqs,kbT,self.n_max_gs,num_points,max_t)

                mean_sq=np.sum(np.real(self.exact_2nd_order_corr[:,1]))/(1.0*self.exact_2nd_order_corr.shape[0])

                # subtract the square of the total omega_av_qm from the values
                self.exact_2nd_order_corr[:,1]=self.exact_2nd_order_corr[:,1]-mean_sq

                # add a decaying exponential to make Fourier transforms well-behaved
                for i in range(self.exact_2nd_order_corr.shape[0]):
                        self.exact_2nd_order_corr[i,1]=self.exact_2nd_order_corr[i,1]*np.exp(-abs(np.real(self.exact_2nd_order_corr[i,0]))/decay_length)

                self.omega_av_qm=self.omega_av_qm+self.E_adiabatic # add E_adiabatic to get the total average energy gap 

                # compute the Fourier transform of the correlation function.
                corr_freq=max_t/num_points*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.exact_2nd_order_corr[:,1])))
                sample_rate=1.0/(max_t/num_points)*math.pi*2.0
                freqs=np.fft.fftshift(np.fft.fftfreq(corr_freq.size,d=1.0/sample_rate))
                self.exact_2nd_order_corr_freq=np.zeros((corr_freq.size,2),dtype=np.complex_)
                for i in range(self.exact_2nd_order_corr_freq.shape[0]):
                        self.exact_2nd_order_corr_freq[i,0]=freqs[i]
                        self.exact_2nd_order_corr_freq[i,1]=corr_freq[i]

                np.savetxt('Morse_with_Dusch_exact_corr_func_freq_real.dat',np.real(self.exact_2nd_order_corr_freq))

        def compute_spectral_dens(self):
                step_length=np.real(self.exact_2nd_order_corr[1,0]-self.exact_2nd_order_corr[0,0])
                corr_func_freq=np.fft.fft(np.fft.ifftshift(self.exact_2nd_order_corr[:, 1].imag))
                freqs = np.fft.fftfreq(self.exact_2nd_order_corr.shape[0], step_length)*2.0*math.pi
                self.spectral_dens = np.zeros((int((corr_func_freq.shape[0])/2) - 1, 2))
                counter = 0
                while counter < self.spectral_dens.shape[0]:
                        self.spectral_dens[counter, 0] = freqs[counter]
                        self.spectral_dens[counter, 1] = -step_length*np.real(1j*corr_func_freq[counter]) # HACK
                        counter = counter + 1

        def compute_g2_exact(self,temp,max_t,num_points):
                kbT=const.kb_in_Ha*temp
                self.g2_exact=cumulant.compute_2nd_order_cumulant_from_spectral_dens(self.spectral_dens,kbT,max_t,num_points)

        def compute_2nd_order_cumulant_response(self,temp,max_t,num_points,stdout):
                kbT=const.kb_in_Ha*temp 
                self.g2_exact=cumulant.compute_2nd_order_cumulant_from_spectral_dens(self.spectral_dens,kbT,max_t,num_points,stdout)
                self.cumulant_response_func=gbom_cumulant_response.compute_cumulant_response(self.g2_exact, np.zeros((1,1)),self.eff_gbom.dipole_mom,np.zeros((1,1)),False,False, False)
                for i in range(self.cumulant_response_func.shape[0]):
                        self.cumulant_response_func[i,1]=self.cumulant_response_func[i,1]*cmath.exp(-1j*self.cumulant_response_func[i,0]*self.omega_av_qm)


        def compute_cumul_fc_hybrid_response_func(self,temp,decay_length,max_t,num_steps,is_emission,stdout):
                # first compute effective response functions for the GBOM under the harmonic approximation of the PES
                self.compute_harmonic_FC_response_func(temp,max_t,num_steps,is_emission,False,stdout)  # disable HT option for now
                self.compute_harmonic_exact_cumulant_response_func(temp,max_t,num_steps,is_emission,stdout)

                # now compute the 2nd order cumulant response function for the morse oscillator
                # start by computing the classical correlation function
                self.compute_exact_corr(temp,decay_length,num_steps,max_t)
                # now compute the spectral density
                self.compute_spectral_dens()
                # finally compute 2nd order cumulant response
                self.compute_2nd_order_cumulant_response(temp,max_t,num_steps,stdout)

                # now can construct the full hybrid response function
                self.hybrid_cumul_fc_response_func=self.cumulant_response_func
                self.hybrid_cumul_fc_response_func[:,1]=self.hybrid_cumul_fc_response_func[:,1]*self.eff_gbom.fc_response[:,1]/self.eff_gbom.cumulant_response[:,1]


        def compute_exact_response(self,temp,max_t,num_steps):
                kbT=const.kb_in_Ha*temp
                self.compute_wf_overlaps_energies()
                self.exact_response_func=compute_exact_response_func_2D(self.wf_overlaps_sq,self.transition_energies,self.gs_freqs,self.n_max_gs,self.D_gs,kbT,max_t,num_steps)
