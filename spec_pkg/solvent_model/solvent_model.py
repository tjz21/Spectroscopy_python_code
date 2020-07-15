#! /usr/bin/env python

import os.path
import numpy as np
import math
import cmath
from ..cumulant import cumulant
from ..constants import constants as const

def solvent_spectral_dens(omega_cut,reorg,max_omega,num_steps):
	spectral_dens=np.zeros((num_steps,2))
	step_length=max_omega/num_steps
	counter=0
	omega=0.0
	while counter<num_steps:
		spectral_dens[counter,0]=omega
		# this definition of the spectral density guarantees that integrating the spectral dens yields the 
		# reorganization energy. We thus have a physical motivation for the chosen parameters
		spectral_dens[counter,1]=2.0*reorg*omega/((1.0+(omega/omega_cut)**2.0)*omega_cut)
		omega=omega+step_length
		counter=counter+1
	return spectral_dens


# class definition
class solvent_model:
	def __init__(self,E_reorg, omega_c):
		self.reorg=E_reorg
		self.cutoff_freq=omega_c
		self.spectral_dens=np.zeros((1,1))
		self.g2_solvent=np.zeros((1,1))
		self.solvent_response=np.zeros((1,1))

	def calc_spectral_dens(self,num_points):
		self.spectral_dens=solvent_spectral_dens(self.cutoff_freq,self.reorg,self.cutoff_freq*20.0,num_points)

	def calc_g2_solvent(self,temp,num_points,max_t,stdout):
		stdout.write('Compute the cumulant lineshape function for a solvent bath made up of an infinite set of harmonic oscillators.')
		kbT=const.kb_in_Ha*temp
		self.g2_solvent=cumulant.compute_2nd_order_cumulant_from_spectral_dens(self.spectral_dens,kbT,max_t,num_points,stdout)

	def calc_solvent_response(self,is_emission):
		counter=0
		response_func=np.zeros((self.g2_solvent.shape[0],2),dtype=complex)
		while counter<self.g2_solvent.shape[0]:
			response_func[counter,0]=self.g2_solvent[counter,0].real
			if is_emission:
				response_func[counter,1]=cmath.exp(-np.conj(self.g2_solvent[counter,1]))
			else:
				response_func[counter,1]=cmath.exp(-self.g2_solvent[counter,1])
			counter=counter+1
		self.solvent_response=response_func

