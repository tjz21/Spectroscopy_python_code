# /usr/bin/env python

from scipy import integrate
import numpy as np
import cmath
import math
from ..constants import constants as const

# analysis routines for linear spectrum
def compute_mean_sd_skew(spectrum): 
	# first make sure spectrum has no negative data points:
	counter=0
	while counter<spectrum.shape[0]:
		if spectrum[counter,1]<0.0:
			spectrum[counter,1]=0.0
		counter=counter+1
	mean=0.0
	for x in spectrum:
		mean=mean+x[1]
	mean=mean/(1.0*spectrum.shape[0])

	sd=0.0
	for x in spectrum:
		sd=sd+(x[1]-mean)**2.0
	sd=math.sqrt(sd/(1.0*spectrum.shape[0]))

	skew=0.0
	for x in spectrum:
		skew=skew+(x[1]-mean)**3.0
	skew=skew/(sd**3.0)

	return mean,sd,skew


def spectrum_prefactor(Eval,dipole_mom,is_emission):
	# prefactor alpha in Ha atomic units
	# Absorption: prefac=10*pi*omega*mu**2*alpha/(3*epsilon_0*ln(10))
	# Emission:   prefac=2*mu**2*omega**4*alpha**3/(3*epsilon_0)
	# note that 4pi*epslion0=1=> epslilon0=1/(4pi)

	prefac=0.0
	if not is_emission:
		# absorption constants
		prefac=40.0*math.pi**2.0*dipole_mom**2.0*const.fine_struct*Eval/(3.0*math.log(10.0))
	else:
		# emission constants
		prefac=2.0*dipole_mom**2.0*const.fine_struct**3.0*Eval**4.0*4.0*math.pi/3.0

	return prefac

def full_spectrum(response_func,solvent_response_func,dipole_mom,steps_spectrum,start_val,end_val,is_solvent,is_emission):
	spectrum=np.zeros((steps_spectrum,2))
	counter=0
	step_length=((end_val-start_val)/steps_spectrum)
	while counter<spectrum.shape[0]:
		E_val=start_val+counter*step_length
		prefac=spectrum_prefactor(E_val,dipole_mom,is_emission)
		integrant=full_spectrum_integrant(response_func,solvent_response_func,E_val,is_solvent)
		spectrum[counter,0]=E_val
		spectrum[counter,1]=prefac*(integrate.simps(integrant,dx=response_func[1,0].real-response_func[0,0].real))
		counter=counter+1

	return spectrum

def full_spectrum_integrant(response_func,solvent_response_func,E_val,is_solvent):
	integrant=np.zeros(response_func.shape[0])
	counter=0
	while counter<integrant.shape[0]:
		if is_solvent:
			integrant[counter]=(response_func[counter,1]*solvent_response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
		else:
			integrant[counter]=(response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
		counter=counter+1
	return integrant


