# /usr/bin/env python

from scipy import integrate
import numpy as np
import cmath
import math
from ..constants import constants as const

# analysis routines for linear spectrum
def compute_mean_sd_skew(spec):
	# first make sure spectrum has no negative data points:
	counter=0
	while counter<spec.shape[0]:
		if spec[counter,1]<0.0:
			spec[counter,1]=0.0
		counter=counter+1
	step=spec[1,0]-spec[0,0]

	# now compute normlization factor
	norm=0.0
	for x in spec:
                norm=norm+x[1]*step

	mean=0.0
	for x in spec:
		mean=mean+x[0]*x[1]*step
	mean=mean/norm

	sd=0.0
	for x in spec:
		sd=sd+(x[0]-mean)**2.0*x[1]*step
	sd=math.sqrt(sd)/norm

	skew=0.0
	for x in spec:
		skew=skew+(x[0]-mean)**3.0*x[1]*step
	skew=skew/(sd**3.0)
	skew=skew/norm

	return mean,sd,skew


def spectrum_prefactor(Eval,is_emission):
	# prefactor alpha in Ha atomic units
	# Absorption: prefac=10*pi*omega*mu**2*alpha/(3*epsilon_0*ln(10))
	# Emission:   prefac=2*mu**2*omega**4*alpha**3/(3*epsilon_0)
	# note that 4pi*epslion0=1=> epslilon0=1/(4pi)

	prefac=0.0
	if not is_emission:
		# absorption constants
		prefac=40.0*math.pi**2.0*const.fine_struct*Eval/(3.0*math.log(10.0))
	else:
		# emission constants
		prefac=2.0*const.fine_struct**3.0*Eval**4.0*4.0*math.pi/3.0

	return prefac

def full_spectrum(response_func,solvent_response_func,steps_spectrum,start_val,end_val,is_solvent,is_emission,stdout):
	spectrum=np.zeros((steps_spectrum,2))
	counter=0
	# print total response function
	stdout.write('\n'+'Total Chromophore linear response function of the system:'+'\n')
	stdout.write('\n'+'  Step       Time (fs)          Re[Chi]         Im[Chi]'+'\n')
	for i in range(response_func.shape[0]):
		stdout.write("%5d      %10.4f          %10.4e       %10.4e" % (i+1,np.real(response_func[i,0])*const.fs_to_Ha, np.real(response_func[i,1]), np.imag(response_func[i,1]))+'\n')

	stdout.write('\n'+'Computing linear spectrum of the system between '+str(start_val*const.Ha_to_eV)+' and '+str(end_val*const.Ha_to_eV)+' eV.')
	stdout.write('\n'+'Total linear spectrum of the system:'+'\n')
	stdout.write('Energy (Ha)         Absorbance (Ha)'+'\n')	
	step_length=((end_val-start_val)/steps_spectrum)
	while counter<spectrum.shape[0]:
		E_val=start_val+counter*step_length
		prefac=spectrum_prefactor(E_val,is_emission)
		integrant=full_spectrum_integrant(response_func,solvent_response_func,E_val,is_solvent)
		spectrum[counter,0]=E_val
		spectrum[counter,1]=prefac*(integrate.simps(integrant,dx=response_func[1,0].real-response_func[0,0].real))
		stdout.write("%2.5f          %10.4e" % (spectrum[counter,0], spectrum[counter,1])+'\n') 
		counter=counter+1

	# Dont do it
	# compute mean, skew and SD of spectrum
#	temp_spec=spectrum
#	print(spectrum)
#	mean,sd,skew=compute_mean_sd_skew(temp_spec)
#	print('Spectrum positive?')
#	stdout.write('\n'+'Mean of spectrum: '+str(mean)+' Ha, SD: '+str(sd)+' Ha, Skew: '+str(skew)+'\n')
#	print(spectrum)
#	# unit conversion: Print X-Axis in eV
	spectrum[:,0]=spectrum[:,0]*const.Ha_to_eV	

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


