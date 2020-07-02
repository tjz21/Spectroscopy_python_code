#! /usr/bin/env python
  
import os.path
import numpy as np
import math
import cmath
from scipy import integrate
from scipy import special
from numba import jit
from spec_pkg.constants import constants as const

#---------------------------------------------------------------------------------------------
# Class definition of a single discplaced Morse oscillator, where ground and excited state PES 
# can have different parameters. For this Morse oscillator, we should be able to compute the 
# Exact response function using analytic expressions for the nuclear wavefunctions,
# we should also be able to construct an approximate Franck-Condon spectrum. Furthermore,
# since the ground state Wavefunctions can be expressed analytically, we should also, in 
# principle, be able to construct the exact quantum autocorrelation function of the Morse
# oscillator.
#--------------------------------------------------------------------------------------------

# compute response function at a given point in time
@jit(fastmath=True)
def compute_morse_chi_func_t(freq_gs,D_gs,kbT,factors,energies,t):
        chi=0.0+0.0j
        for n_gs in range(factors.shape[0]):
                Egs=compute_morse_eval_n(freq_gs,D_gs,n_gs)
                boltzmann=math.exp(-Egs/kbT)
                for n_ex in range(factors.shape[1]):
                        chi=chi+boltzmann*factors[n_gs,n_ex]*cmath.exp(-1j*energies[n_gs,n_ex]*t)

        return cmath.polar(chi)

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

        if np.isinf(special.gamma(2.0*lamda-n)):
                wavefunc_sq=np.zeros(wavefunc.shape[0])
                for counter in range(wavefunc.shape[0]):
                        wavefunc_sq[counter]=wavefunc[counter,1]*wavefunc[counter,1]

                normalization=integrate.simps(wavefunc_sq,dx=step_x)
                for counter in range(wavefunc.shape[0]):
                        wavefunc[counter,1]=wavefunc[counter,1]/math.sqrt(normalization)

        return wavefunc

# compute the effective Franck-Condon wavefunction overlap between two wavefunctions
def get_fc_factor(num_points,start_point,end_point,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,n_gs,n_ex):
        #print start_point,end_point,D,alpha,n_gs,n_ex,d
        func1=compute_wavefunction_n(num_points, start_point,end_point,D_gs,alpha_gs,mu,n_gs,0.0)
        func2=compute_wavefunction_n(num_points, start_point,end_point,D_ex,alpha_ex,mu,n_ex,K)
        counter=0
        for x in func1:
                x[1]=x[1]*func2[counter,1]
                counter=counter+1

        return (integrate.simps(func1[:,1],dx=func1[1,0]-func1[0,0]))**2.0

class morse: 
	def __init__(self,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,E_adiabatic,dipole_mom,max_states,num_points):
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
		self.n_max_gs=max_states
		self.n_max_ex=max_states

		# set derived absorption vactiables, mainly n_max and freq
		self.set_absorption_variables()

		# modulus squared of the overlaps of nuclear wavefunctions between ground
		# and excited state PES
		self.wf_overlaps=np.zeros((self.n_max_gs,self.n_max_ex))
		self.transition_energies=np.zeros((self.n_max_gs,self.n_max_ex))

		self.omega_av_cl=0.0
		self.omega_av_qm=0.0

		# response functions
		self.exact_response_func=np.zeros((1,1),dtype=np.complex_)

	# Now declare functions of the Morse oscillator
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
		self.exact_response_func=compute_exact_response_func(self.wf_overlaps,self.transition_energies,self.freq_gs,self.D_gs,kbT,max_t,num_steps)

	# now compute all wavefunction overlaps and Transition energies 
	def compute_overlaps_and_transition_energies(self):
		for i in range(self.n_max_gs):
			for j in range(self.n_max_ex):
				self.transition_energies[i,j]=self.transition_energy(i,j)
				self.wf_overlaps[i,j]=get_fc_factor(self.grid_n_points,self.grid_start,self.grid_end,self.D_gs,self.D_ex,self.alpha_gs,self.alpha_ex,self.mu,self.K,i,j)
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
	def __init__(self,D_gs,D_ex,alpha_gs,alpha_ex,mu,K,E_adiabatic,dipole_mom,max_states,num_points,num_morse_oscillators):
		self.morse_oscs = []
		for i in range(num_morse_oscillators):
			self.morse_oscs.append(morse(D_gs[i],D_ex[i],alpha_gs[i],alpha_ex[i],mu[i],K[i],E_adiabatic,dipole_mom,max_states,num_points))
		self.num_morse_oscillators=num_morse_oscillators

		self.total_exact_response_func=np.zeros((1,1),dtype=np.complex_)
		self.E_adiabatic=E_adiabatic
		
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
	
