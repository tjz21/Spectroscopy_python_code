# /usr/bin/env python

from scipy import integrate
import numpy as np
import cmath
import math

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

def full_spectrum(response_func,solvent_response_func,steps_spectrum,start_val,end_val,is_solvent):
    spectrum=np.zeros((steps_spectrum,2))
    counter=0
    step_length=((end_val-start_val)/steps_spectrum)
    while counter<spectrum.shape[0]:
        E_val=start_val+counter*step_length
        integrant=full_spectrum_integrant(response_func,solvent_response_func,E_val,is_solvent)
        spectrum[counter,0]=E_val
        spectrum[counter,1]=integrate.simps(integrant,dx=response_func[1,0].real-response_func[0,0].real)
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


