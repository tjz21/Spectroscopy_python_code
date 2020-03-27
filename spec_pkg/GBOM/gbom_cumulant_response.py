#! /usr/bin/env python

import numpy as np
import math
from numba import jit
import cmath

# check whether this is an absorption or an emission calculation
def compute_cumulant_response(g2,g3,is_3rd_order_cumulant,is_emission):
    # compute the effective response function of energy gap fluctuations in the 2nd or 3rd order cumulant
    response_func=np.zeros((g2.shape[0],2),dtype=complex)
    counter=0
    while counter<g2.shape[0]:
	response_func[counter,0]=g2[counter,0].real
	# if required add 3rd order contribution
	if is_emission:
		if is_3rd_order_cumulant:
			eff_g3=1j*g3[counter,1]
			response_func[counter,1]=cmath.exp(-np.conj(g2[counter,1])-1j*np.conj(eff_g3))
		else:
			response_func[counter,1]=cmath.exp(-np.conj(g2[counter,1]))
	else:
		if is_3rd_order_cumulant:
			response_func[counter,1]=cmath.exp(-g2[counter,1]-g3[counter,1])
		else:
			response_func[counter,1]=cmath.exp(-g2[counter,1])
	counter=counter+1
    return response_func

def compute_spectral_dens(freqs_gs,Omega_sq,gamma,kbT,max_t,max_steps,decay_length,is_cl):
    # compute a longer version of the spectral density
    max_t=max_t*10
    max_steps=max_steps*10

    # is the correlation function derived from the classical correlation function or is it the exact 
    # quantum correlation function?
    if is_cl:
	full_corr_func=full_2nd_order_corr_cl(freqs_gs,Omega_sq,gamma,kbT,max_t,max_steps,decay_length)
	np.savetxt('full_correlation_func_classical.dat', full_corr_func)
    else:
    	full_corr_func=full_2nd_order_corr_qm(freqs_gs,Omega_sq,gamma,kbT,max_t,max_steps,decay_length)
	np.savetxt('full_correlation_func_qm.dat', full_corr_func)
    # get frequencies and perform inverse FFT. The imaginary part is the spectral density
    # multiply FFT by step length to get correct units for full spectral density
    corr_func_freq=np.fft.fft(np.fft.ifftshift(full_corr_func[:,1].imag))*max_t/max_steps*2.0*math.pi   # 2pi is the missing normalization factor. double check
    freqs=np.fft.fftfreq(corr_func_freq.shape[0],max_t/max_steps)*2.0*math.pi

    spectral_dens=np.zeros((corr_func_freq.shape[0]/2-1,2))
    counter=0
    while counter<spectral_dens.shape[0]:
	spectral_dens[counter,0]=freqs[counter]
	spectral_dens[counter,1]=corr_func_freq[counter].real
	counter=counter+1

    return spectral_dens

def full_2nd_order_corr_qm(freqs_gs,Omega_sq,gamma,kbT,max_t,max_steps,decay_length):
    step_length=max_t/max_steps
    corr_func=np.zeros((max_steps*2-1,2),dtype=complex)
    current_t=0.0

    # compute freqs, freqs_gs and 
    Omega_sq_sq=np.multiply(Omega_sq,Omega_sq)

    n_i=np.zeros(freqs_gs.shape[0])
    n_i_p=np.zeros(freqs_gs.shape[0])

    # fill n_i and n_i_p vectors:
    icount=0
    while icount<freqs_gs.shape[0]:
        n_i[icount]=bose_einstein(freqs_gs[icount],kbT)
        n_i_p[icount]=n_i[icount]+1.0
        icount=icount+1

    counter=0
    while counter<corr_func.shape[0]:
        corr_func[counter,0]=-max_t+step_length*counter
        corr_func[counter,1]=second_order_corr_t_qm(freqs_gs,Omega_sq_sq,gamma,n_i,n_i_p,corr_func[counter,0])
        corr_func[counter,1]=corr_func[counter,1]*math.exp(-abs(corr_func[counter,0].real)/decay_length)
        counter=counter+1

    return corr_func

def full_2nd_order_corr_cl(freqs_gs,Omega_sq,gamma,kbT,max_t,max_steps,decay_length):
    step_length=max_t/max_steps
    corr_func=np.zeros((max_steps*2-1,2),dtype=complex)
    current_t=0.0

    # compute freqs, freqs_gs and 
    Omega_sq_sq=np.multiply(Omega_sq,Omega_sq)

    n_i=np.zeros(freqs_gs.shape[0])
    n_i_p=np.zeros(freqs_gs.shape[0])
    n_ij_p=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
    n_ij_m=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))

    # fill n_i vector:
    icount=0
    while icount<freqs_gs.shape[0]:
        n_i[icount]=bose_einstein(freqs_gs[icount],kbT)
	n_i_p[icount]=n_i[icount]+1.0
        icount=icount+1

    # fill n_ij vectors. Make sure to deal corretly with cases where omegai==omegaj:
    icount=0
    while icount<freqs_gs.shape[0]:
	jcount=0
	while jcount<freqs_gs.shape[0]:
		n_ij_p[icount,jcount]=bose_einstein(freqs_gs[icount]+freqs_gs[jcount],kbT)
		if icount==jcount:
			n_ij_m[icount,jcount]=1.0
		else:
			n_ij_m[icount,jcount]=bose_einstein(freqs_gs[icount]-freqs_gs[jcount],kbT)

		jcount=jcount+1
	icount=icount+1

    counter=0
    while counter<corr_func.shape[0]:
        corr_func[counter,0]=-max_t+step_length*counter
        corr_func[counter,1]=second_order_corr_t_cl(freqs_gs,Omega_sq_sq,gamma,n_i,n_i_p,n_ij_p,n_ij_m,kbT,corr_func[counter,0])
        corr_func[counter,1]=corr_func[counter,1]*math.exp(-abs(corr_func[counter,0].real)/decay_length)
        counter=counter+1

    return corr_func

# compact vectorized representation of classically derived quantum autocorrelation function
@jit
def second_order_corr_t_cl(freqs_gs,Omega_sq,gamma,n_i,n_i_p,n_ij_p,n_ij_m,kbT,t):
    # vectorized version: 
    exp_wt=np.zeros(freqs_gs.shape[0],dtype=complex)
    inv_w=np.zeros(freqs_gs.shape[0])

    # fill exp_wt vectors
    icount=0
    while icount<freqs_gs.shape[0]:
        inv_w[icount]=1.0/freqs_gs[icount]
        exp_wt[icount]=cmath.exp(1j*freqs_gs[icount]*t)
        icount=icount+1

    # compact representation of single phonon term
    term1=0.5*np.dot(np.multiply(np.conj(exp_wt),gamma),np.multiply(inv_w,np.multiply(n_i_p,gamma)))+0.5*np.dot(np.multiply(exp_wt,gamma),np.multiply(inv_w,np.multiply(n_i,gamma)))

    term2=0.0
    term3=0.0
    
    icount=0
    while icount<freqs_gs.shape[0]:
	jcount=0
	while jcount<freqs_gs.shape[0]:
		w_ij_p=freqs_gs[icount]+freqs_gs[jcount]
		w_ij_m=freqs_gs[icount]-freqs_gs[jcount]

		const_fac=0.5*kbT*Omega_sq[icount,jcount]/(freqs_gs[icount]*freqs_gs[jcount])**2.0
		term2=term2+const_fac*w_ij_p*((n_ij_p[icount,jcount]+1.0)*cmath.exp(-1j*w_ij_p*t)+n_ij_p[icount,jcount]*cmath.exp(1j*w_ij_p*t))
		if icount!=jcount:
			term3=term3+const_fac*w_ij_p*((n_ij_m[icount,jcount]+1.0)*cmath.exp(-1j*w_ij_m*t)+n_ij_m[icount,jcount]*cmath.exp(1j*w_ij_m*t))
		else:
			term3=term3+const_fac*2.0*kbT

		jcount=jcount+1
	icount=icount+1
  
    return term1+term2+term3

# compact vectorized representation of exact quantum autocorrelation function
@jit
def second_order_corr_t_qm(freqs_gs,Omega_sq,gamma,n_i,n_i_p,t):
    # vectorized version: 
    exp_wt=np.zeros(freqs_gs.shape[0],dtype=complex)
    inv_w=np.zeros(freqs_gs.shape[0])

    # fill exp_wt vectors
    icount=0
    while icount<freqs_gs.shape[0]:
        inv_w[icount]=1.0/freqs_gs[icount]
        exp_wt[icount]=cmath.exp(1j*freqs_gs[icount]*t)
        icount=icount+1

    # compact representation of single phonon term
    term1=0.5*np.dot(np.multiply(np.conj(exp_wt),gamma),np.multiply(inv_w,np.multiply(n_i_p,gamma)))+0.5*np.dot(np.multiply(exp_wt,gamma),np.multiply(inv_w,np.multiply(n_i,gamma)))

    # compact representation of multiple phonon term
    term2=0.5*np.dot(np.multiply(np.multiply(n_i_p,inv_w),np.conj(exp_wt)),np.dot(Omega_sq,np.multiply(np.multiply(n_i_p,inv_w),np.conj(exp_wt))))
    term2=term2+0.5*np.dot(np.multiply(np.multiply(n_i,inv_w),exp_wt),np.dot(Omega_sq,np.multiply(np.multiply(n_i,inv_w),exp_wt)))
    term2=term2+0.5*np.dot(np.multiply(np.multiply(n_i_p,inv_w),np.conj(exp_wt)),np.dot(Omega_sq,np.multiply(np.multiply(n_i,inv_w),exp_wt)))
    term2=term2+0.5*np.dot(np.multiply(np.multiply(n_i,inv_w),exp_wt),np.dot(Omega_sq,np.multiply(np.multiply(n_i_p,inv_w),np.conj(exp_wt))))

    return term1+term2

@jit
def bose_einstein(freq,kbT):
    n=math.exp(freq/kbT)-1.0
    return 1.0/n

# g2 as derived from either the exact classical correlation function or the exact QM correlation function
def full_2nd_order_lineshape(freqs_gs,freqs_ex,Kmat,Jmat,gamma,Omega_sq,kbT,av_energy_gap,max_t,num_points,is_cl):
    lineshape_func=np.zeros((num_points,2),dtype=complex)
    print 'Computing lineshape function'
    print 'av energy gap:'
    print av_energy_gap
    print 'gamma:'
    print gamma
    print Omega_sq
    print 'kbT:'
    print kbT
    step_length=max_t/num_points
    t=0.0
    count1=0
    while count1<num_points:
        lineshape_func[count1,0]=t
	if is_cl:
        	lineshape_func[count1,1]=second_order_lineshape_cl_t(freqs_gs,Omega_sq,gamma,kbT,t)+1j*t*av_energy_gap
	else:
		lineshape_func[count1,1]=second_order_lineshape_qm_t(freqs_gs,Omega_sq,gamma,kbT,t)+1j*t*av_energy_gap
        count1=count1+1
        t=t+step_length
        print lineshape_func[count1-1,0],lineshape_func[count1-1,1]

    return lineshape_func

# third order cumulant lineshape
def full_third_order_lineshape(freqs_gs,Omega_sq,gamma,kbT,max_t,num_points,is_cl):
    lineshape_func=np.zeros((num_points,2),dtype=complex)
    step_length=max_t/num_points
    # only compute n_i_vec if this is a QM lineshape calculation:
    n_i_vec=np.zeros(freqs_gs.shape[0])
    if not is_cl:
	icount=0
     	while icount<freqs_gs.shape[0]:
         	n_i_vec[icount]=bose_einstein(freqs_gs[icount],kbT)
            	icount=icount+1
    t=0.0
    count1=0
    while count1<num_points:
        lineshape_func[count1,0]=t
	if is_cl:
		lineshape_func[count1,1]=third_order_lineshape_cl_t(freqs_gs,Omega_sq,gamma,kbT,t)
	else:
        	lineshape_func[count1,1]=third_order_lineshape_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t)
        count1=count1+1
        t=t+step_length
        print lineshape_func[count1-1,0],lineshape_func[count1-1,1]
    return lineshape_func

# h1 factor necessary for computing 2DES in 3rd order cumulant:
def full_h1_func(freqs_gs,Omega_sq,gamma,kbT,max_t,num_points,is_cl):
    h1_func=np.zeros((num_points,num_points,3),dtype=complex)
    step_length=max_t/num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec=np.zeros(freqs_gs.shape[0])
    if not is_cl:
    	icount=0
      	while icount<freqs_gs.shape[0]:
              	n_i_vec[icount]=bose_einstein(freqs_gs[icount],kbT)
             	icount=icount+1

    t1=0.0
    count1=0
    print 'COMPUTING H1'
    while count1<num_points:
	count2=0
	t2=0.0
	print count1
	while count2<num_points:
        	h1_func[count1,count2,0]=t1
		h1_func[count1,count2,1]=t2
        	if is_cl:
                	h1_func[count1,count2,2]=h1_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2)
        	else:
                	h1_func[count1,count2,2]=h1_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2)
		count2=count2+1
		t2=t2+step_length
        count1=count1+1
        t1=t1+step_length
    return h1_func

# h2 factor necessary for computing 2DES in 3rd order cumulant:
def full_h2_func(freqs_gs,Omega_sq,gamma,kbT,max_t,num_points,is_cl):
    h2_func=np.zeros((num_points,num_points,3),dtype=complex)
    step_length=max_t/num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec=np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount=0
        while icount<freqs_gs.shape[0]:
                n_i_vec[icount]=bose_einstein(freqs_gs[icount],kbT)
                icount=icount+1

    t1=0.0
    count1=0
    print 'COMPUTING H2'
    while count1<num_points:
        count2=0
        t2=0.0
	print count1
        while count2<num_points:
                h2_func[count1,count2,0]=t1
                h2_func[count1,count2,1]=t2
                if is_cl:
                        h2_func[count1,count2,2]=h2_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2)
                else:
                        h2_func[count1,count2,2]=h2_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2)
                count2=count2+1
                t2=t2+step_length
        count1=count1+1
        t1=t1+step_length
    return h2_func

# h4 factor necessary for computing 2DES in 3rd order cumulant:
def full_h4_func(freqs_gs,Omega_sq,gamma,kbT,max_t,num_points,is_cl):
    h4_func=np.zeros((num_points,num_points,3),dtype=complex)
    step_length=max_t/num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec=np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount=0
        while icount<freqs_gs.shape[0]:
                n_i_vec[icount]=bose_einstein(freqs_gs[icount],kbT)
                icount=icount+1

    t1=0.0
    count1=0
    print 'COMPUTING H4'
    while count1<num_points:
        count2=0
        t2=0.0
        print count1
        while count2<num_points:
                h4_func[count1,count2,0]=t1
                h4_func[count1,count2,1]=t2
                if is_cl:
                        h4_func[count1,count2,2]=h4_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2)
                else:
                        h4_func[count1,count2,2]=h4_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2)
                count2=count2+1
                t2=t2+step_length
        count1=count1+1
        t1=t1+step_length
    return h4_func


# h5 factor necessary for computing 2DES in 3rd order cumulant:
def full_h5_func(freqs_gs,Omega_sq,gamma,kbT,max_t,num_points,is_cl):
    h5_func=np.zeros((num_points,num_points,3),dtype=complex)
    step_length=max_t/num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec=np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount=0
        while icount<freqs_gs.shape[0]:
                n_i_vec[icount]=bose_einstein(freqs_gs[icount],kbT)
                icount=icount+1

    t1=0.0
    count1=0
    print 'COMPUTING H5'
    while count1<num_points:
        count2=0
        t2=0.0
        print count1
        while count2<num_points:
                h5_func[count1,count2,0]=t1
                h5_func[count1,count2,1]=t2
                if is_cl:
                        h5_func[count1,count2,2]=h5_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2)
                else:
                        h5_func[count1,count2,2]=h5_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2)
                count2=count2+1
                t2=t2+step_length
        count1=count1+1
        t1=t1+step_length
    return h5_func

@jit
def h2_func_cl_no_dusch(freqs_gs,Omega_sq,kbT,t1,t2):
    	gamma_term=0.0+0.0j
    	omega_term=0.0+0.0j
    	icount=0
    	while icount<freqs_gs.shape[0]:
                const_fac=2.0*math.pi**2.0*Omega_sq[icount,icount]*gamma[icount]*gamma[icount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[icount])**2.0)
                omega_p=freqs_gs[icount]+freqs_gs[icount]
                omega_m=freqs_gs[icount]-freqs_gs[icount]
                omegai=freqs_gs[icount]
                omegaj=freqs_gs[icount]

                # term 1
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
                # term 2  
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
                # term 3
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

                const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*Omega_sq[icount,icount]*Omega_sq[icount,icount]
                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[icount]*freqs_gs[icount])**2.0
                ik_p=freqs_gs[icount]+freqs_gs[icount]
                ik_m=freqs_gs[icount]-freqs_gs[icount]
                ij_p=freqs_gs[icount]+freqs_gs[icount]
                ij_m=freqs_gs[icount]-freqs_gs[icount]
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

        	icount=icount+1

        return omega_term+gamma_term

@jit
def h1_func_cl_no_dusch(freqs_gs,Omega_sq,kbT,t1,t2):
    	gamma_term=0.0+0.0j
    	omega_term=0.0+0.0j
    	icount=0
    	while icount<freqs_gs.shape[0]:
                const_fac=2.0*math.pi**2.0*Omega_sq[icount,icount]*gamma[icount]*gamma[icount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[icount])**2.0)
                omega_p=freqs_gs[icount]+freqs_gs[icount]
                omega_m=freqs_gs[icount]-freqs_gs[icount]
                omegai=freqs_gs[icount]
                omegaj=freqs_gs[icount]

                # term 1
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
                # term 2  
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
                # term 3
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

                const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*Omega_sq[icount,icount]*Omega_sq[icount,icount]
                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[icount]*freqs_gs[icount])**2.0
                ik_p=freqs_gs[icount]+freqs_gs[icount]
                ik_m=freqs_gs[icount]-freqs_gs[icount]
                ij_p=freqs_gs[icount]+freqs_gs[icount]
                ij_m=freqs_gs[icount]-freqs_gs[icount]
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

        	icount=icount+1

        return omega_term+gamma_term


@jit 
def h5_func_cl_no_dusch(freqs_gs,Omega_sq,kbT,t1,t2):
    	gamma_term=0.0+0.0j
    	omega_term=0.0+0.0j
    	icount=0
    	while icount<freqs_gs.shape[0]:
                const_fac=2.0*math.pi**2.0*Omega_sq[icount,icount]*gamma[icount]*gamma[icount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[icount])**2.0)
                omega_p=freqs_gs[icount]+freqs_gs[icount]
                omega_m=freqs_gs[icount]-freqs_gs[icount]
                omegai=freqs_gs[icount]
                omegaj=freqs_gs[icount]

                # term 1
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
                # term 2  
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
                # term 3
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

                const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*Omega_sq[icount,icount]*Omega_sq[icount,icount]
                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[icount]*freqs_gs[icount])**2.0
                ik_p=freqs_gs[icount]+freqs_gs[icount]
                ik_m=freqs_gs[icount]-freqs_gs[icount]
                ij_p=freqs_gs[icount]+freqs_gs[icount]
                ij_m=freqs_gs[icount]-freqs_gs[icount]
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

        	icount=icount+1

        return omega_term+gamma_term


@jit
def h4_func_cl_no_dusch(freqs_gs,Omega_sq,kbT,t1,t2):
    	gamma_term=0.0+0.0j
    	omega_term=0.0+0.0j
    	icount=0
    	while icount<freqs_gs.shape[0]:
                const_fac=2.0*math.pi**2.0*Omega_sq[icount,icount]*gamma[icount]*gamma[icount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[icount])**2.0)
                omega_p=freqs_gs[icount]+freqs_gs[icount]
                omega_m=freqs_gs[icount]-freqs_gs[icount]
                omegai=freqs_gs[icount]
                omegaj=freqs_gs[icount]

                # term 1
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
                # term 2  
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
                # term 3
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
                gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

                const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*Omega_sq[icount,icount]*Omega_sq[icount,icount]
                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[icount]*freqs_gs[icount])**2.0
                ik_p=freqs_gs[icount]+freqs_gs[icount]
                ik_m=freqs_gs[icount]-freqs_gs[icount]
                ij_p=freqs_gs[icount]+freqs_gs[icount]
                ij_m=freqs_gs[icount]-freqs_gs[icount]
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

        	icount=icount+1

        return omega_term+gamma_term




@jit 
def h3_func_cl_no_dusch(freqs_gs,Omega_sq,kbT,t1,t2,t3):
    	gamma_term=0.0+0.0j
    	omega_term=0.0+0.0j
    	icount=0
    	while icount<freqs_gs.shape[0]:
         	const_fac=2.0*math.pi**2.0*Omega_sq[icount,icount]*gamma[icount]*gamma[icount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[icount])**2.0)
             	omega_p=freqs_gs[icount]+freqs_gs[icount]
            	omega_m=freqs_gs[icount]-freqs_gs[icount]
          	omegai=freqs_gs[icount]
             	omegaj=freqs_gs[icount]

            	# term 1
            	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
            	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
              	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
              	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
             	# term 2  
             	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
               	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
            	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
              	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
               	# term 3
            	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegaj,omegai,kbT,t1,t2,t3)
             	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
             	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
            	gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

		const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*Omega_sq[icount,icount]*Omega_sq[icount,icount]
               	const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[icount]*freqs_gs[icount])**2.0
             	ik_p=freqs_gs[icount]+freqs_gs[icount]
            	ik_m=freqs_gs[icount]-freqs_gs[icount]
            	ij_p=freqs_gs[icount]+freqs_gs[icount]
              	ij_m=freqs_gs[icount]-freqs_gs[icount]
          	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
             	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
              	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
               	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

              	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
             	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
             	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
              	omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

        	icount=icount+1

	return omega_term+gamma_term

@jit
def h3_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2,t3):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:

        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=2.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]

                        # term 1
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
                        # term 2  
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
                        # term 3
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegaj,omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+prefactor_2DES_h3_cl(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])**2.0
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                                omega_term=omega_term+prefactor_2DES_h3_cl(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1

    corr_val=omega_term+gamma_term
    return corr_val



@jit
def h4_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=2.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]

                        # term 1
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h4_cl(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])**2.0
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h4_cl(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term
    return corr_val


@jit
def h5_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=2.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]

                        # term 1
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h5_cl(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])**2.0
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h5_cl(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1

    corr_val=omega_term+gamma_term
    return corr_val


@jit
def h2_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=2.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]

                        # term 1
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h2_cl(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])**2.0
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h2_cl(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1

    corr_val=omega_term+gamma_term
    return corr_val


@jit
def h1_func_cl_t(freqs_gs,Omega_sq,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=2.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]

                        # term 1
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+prefactor_2DES_h1_cl(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])**2.0
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+prefactor_2DES_h1_cl(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term
    return corr_val


@jit
def third_order_lineshape_cl_t(freqs_gs,Omega_sq,gamma,kbT,t):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=2.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]*(kbT**2.0/(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]

                        # term 1
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,omega_p,-omegai,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,-omega_m,omegai,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,omega_m,-omegai,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,-omega_p,omegai,kbT,t)
                        # term 2  
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,-omegai,omega_p,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,omegai,-omega_m,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,-omegai,omega_m,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,omegai,-omega_p,kbT,t)
                        # term 3
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,omegaj,omegai,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,omegaj,-omegai,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,-omegaj,omegai,kbT,t)
                        gamma_term=gamma_term+prefactor_3rd_order_lineshape(const_fac,-omegaj,-omegai,kbT,t)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac*(kbT)**3.0/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])**2.0
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,-ik_m,ij_p,kbT,t)
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,-ik_p,ij_p,kbT,t)
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,ik_p,-ij_p,kbT,t)
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,ik_m,-ij_p,kbT,t)

                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,-ik_m,ij_m,kbT,t)
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,-ik_p,ij_m,kbT,t)
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,ik_p,-ij_m,kbT,t)
                                omega_term=omega_term+prefactor_3rd_order_lineshape(const_fac,ik_m,-ij_m,kbT,t)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1

    corr_val=omega_term+gamma_term
    return corr_val


@jit 
def h3_func_qm_t_no_dusch(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2,t3):
    	gamma_term=0.0+0.0j
    	omega_term=0.0+0.0j
    	# start with gamma term first:
    	icount=0
    	while icount<freqs_gs.shape[0]:
              	const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*gamma[icount]*gamma[icount]/(2.0*freqs_gs[icount]*freqs_gs[icount])
              	omega_p=freqs_gs[icount]+freqs_gs[icount]
               	omega_m=freqs_gs[icount]-freqs_gs[icount]
               	omegai=freqs_gs[icount]
              	omegaj=freqs_gs[icount]
              	ni=n_i_vec[icount]
               	nj=n_i_vec[icount]
              	ni_p=ni+1.0
             	nj_p=nj+1.0

             	# term 1
            	gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h3_QM(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
            	gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
              	gamma_term=gamma_term+ni*nj*prefactor_2DES_h3_QM(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
             	gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
             	# term 2  
             	gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
             	gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h3_QM(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
              	gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
              	gamma_term=gamma_term+ni*nj*prefactor_2DES_h3_QM(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
               	# term 3
             	gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,omegaj,omegai,kbT,t1,t2,t3)
               	gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h3_QM(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
             	gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
        	gamma_term=gamma_term+ni*nj*prefactor_2DES_h3_QM(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

		const_fac=4.0*math.pi**2.0*Omega_sq[icount,icount]*Omega_sq[icount,icount]*Omega_sq[icount,icount]
              	const_fac=const_fac/(freqs_gs[icount]*freqs_gs[icount]*freqs_gs[icount])
             	ik_p=freqs_gs[icount]+freqs_gs[icount]
             	ik_m=freqs_gs[icount]-freqs_gs[icount]
             	ij_p=freqs_gs[icount]+freqs_gs[icount]
              	ij_m=freqs_gs[icount]-freqs_gs[icount]
              	ni=n_i_vec[icount]
              	nj=n_i_vec[icount]
               	nk=n_i_vec[icount]
              	ni_p=ni+1.0
              	nj_p=nj+1.0
              	nk_p=nk+1.0

            	omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_2DES_h3_QM(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
             	omega_term=omega_term+ni_p*nj_p*nk*prefactor_2DES_h3_QM(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
              	omega_term=omega_term+ni*nj*nk_p*prefactor_2DES_h3_QM(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
             	omega_term=omega_term+ni*nj*nk*prefactor_2DES_h3_QM(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

         	omega_term=omega_term+ni_p*nj*nk_p*prefactor_2DES_h3_QM(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
              	omega_term=omega_term+ni_p*nj*nk*prefactor_2DES_h3_QM(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
            	omega_term=omega_term+ni*nj_p*nk_p*prefactor_2DES_h3_QM(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
            	omega_term=omega_term+ni*nj_p*nk*prefactor_2DES_h3_QM(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

        	icount=icount+1

	return omega_term+gamma_term



@jit
def h3_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2,t3):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]/(2.0*freqs_gs[icount]*freqs_gs[jcount])
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]
                        ni=n_i_vec[icount]
                        nj=n_i_vec[jcount]
                        ni_p=ni+1.0
                        nj_p=nj+1.0

                        # term 1
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h3_QM(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h3_QM(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omega_p,omegai,kbT,t1,t2,t3)
                        # term 2  
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h3_QM(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h3_QM(const_fac,omegai,-omega_p,kbT,t1,t2,t3)
                        # term 3
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,omegaj,omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h3_QM(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h3_QM(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                ni=n_i_vec[icount]
                                nj=n_i_vec[jcount]
                                nk=n_i_vec[kcount]
                                ni_p=ni+1.0
                                nj_p=nj+1.0
                                nk_p=nk+1.0

                                omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_2DES_h3_QM(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                                omega_term=omega_term+ni_p*nj_p*nk*prefactor_2DES_h3_QM(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                                omega_term=omega_term+ni*nj*nk_p*prefactor_2DES_h3_QM(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                                omega_term=omega_term+ni*nj*nk*prefactor_2DES_h3_QM(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                                omega_term=omega_term+ni_p*nj*nk_p*prefactor_2DES_h3_QM(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                                omega_term=omega_term+ni_p*nj*nk*prefactor_2DES_h3_QM(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                                omega_term=omega_term+ni*nj_p*nk_p*prefactor_2DES_h3_QM(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                                omega_term=omega_term+ni*nj_p*nk*prefactor_2DES_h3_QM(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term
    return corr_val

@jit
def h5_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]/(2.0*freqs_gs[icount]*freqs_gs[jcount])
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]
                        ni=n_i_vec[icount]
                        nj=n_i_vec[jcount]
                        ni_p=ni+1.0
                        nj_p=nj+1.0

                        # term 1
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h5_QM(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h5_QM(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h5_QM(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h5_QM(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h5_QM(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h5_QM(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h5_QM(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h5_QM(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h5_QM(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h5_QM(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h5_QM(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h5_QM(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                ni=n_i_vec[icount]
                                nj=n_i_vec[jcount]
                                nk=n_i_vec[kcount]
                                ni_p=ni+1.0
                                nj_p=nj+1.0
                                nk_p=nk+1.0

                                omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_2DES_h5_QM(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj_p*nk*prefactor_2DES_h5_QM(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk_p*prefactor_2DES_h5_QM(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk*prefactor_2DES_h5_QM(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+ni_p*nj*nk_p*prefactor_2DES_h5_QM(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj*nk*prefactor_2DES_h5_QM(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk_p*prefactor_2DES_h5_QM(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk*prefactor_2DES_h5_QM(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term
    return corr_val


@jit
def h4_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]/(2.0*freqs_gs[icount]*freqs_gs[jcount])
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]
                        ni=n_i_vec[icount]
                        nj=n_i_vec[jcount]
                        ni_p=ni+1.0
                        nj_p=nj+1.0

                        # term 1
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h4_QM(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h4_QM(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h4_QM(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h4_QM(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h4_QM(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h4_QM(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h4_QM(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h4_QM(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h4_QM(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h4_QM(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h4_QM(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h4_QM(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                ni=n_i_vec[icount]
                                nj=n_i_vec[jcount]
                                nk=n_i_vec[kcount]
                                ni_p=ni+1.0
                                nj_p=nj+1.0
                                nk_p=nk+1.0

                                omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_2DES_h4_QM(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj_p*nk*prefactor_2DES_h4_QM(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk_p*prefactor_2DES_h4_QM(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk*prefactor_2DES_h4_QM(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+ni_p*nj*nk_p*prefactor_2DES_h4_QM(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj*nk*prefactor_2DES_h4_QM(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk_p*prefactor_2DES_h4_QM(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk*prefactor_2DES_h4_QM(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1

    corr_val=omega_term+gamma_term
    return corr_val


@jit
def h2_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]/(2.0*freqs_gs[icount]*freqs_gs[jcount])
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]
                        ni=n_i_vec[icount]
                        nj=n_i_vec[jcount]
                        ni_p=ni+1.0
                        nj_p=nj+1.0

                        # term 1
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h2_QM(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h2_QM(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h2_QM(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h2_QM(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h2_QM(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h2_QM(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h2_QM(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h2_QM(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h2_QM(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h2_QM(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h2_QM(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h2_QM(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                ni=n_i_vec[icount]
                                nj=n_i_vec[jcount]
                                nk=n_i_vec[kcount]
                                ni_p=ni+1.0
                                nj_p=nj+1.0
                                nk_p=nk+1.0

                                omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_2DES_h2_QM(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj_p*nk*prefactor_2DES_h2_QM(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk_p*prefactor_2DES_h2_QM(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk*prefactor_2DES_h2_QM(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+ni_p*nj*nk_p*prefactor_2DES_h2_QM(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj*nk*prefactor_2DES_h2_QM(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk_p*prefactor_2DES_h2_QM(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk*prefactor_2DES_h2_QM(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1

    corr_val=omega_term+gamma_term
    return corr_val



@jit
def h1_func_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t1,t2):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]/(2.0*freqs_gs[icount]*freqs_gs[jcount])
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]
                        ni=n_i_vec[icount]
                        nj=n_i_vec[jcount]
                        ni_p=ni+1.0
                        nj_p=nj+1.0

                        # term 1
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h1_QM(const_fac,omega_p,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h1_QM(const_fac,-omega_m,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h1_QM(const_fac,omega_m,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h1_QM(const_fac,-omega_p,omegai,kbT,t1,t2)
                        # term 2  
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h1_QM(const_fac,-omegai,omega_p,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h1_QM(const_fac,omegai,-omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h1_QM(const_fac,-omegai,omega_m,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h1_QM(const_fac,omegai,-omega_p,kbT,t1,t2)
                        # term 3
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_2DES_h1_QM(const_fac,omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj_p*prefactor_2DES_h1_QM(const_fac,omegaj,-omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni_p*nj*prefactor_2DES_h1_QM(const_fac,-omegaj,omegai,kbT,t1,t2)
                        gamma_term=gamma_term+ni*nj*prefactor_2DES_h1_QM(const_fac,-omegaj,-omegai,kbT,t1,t2)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                ni=n_i_vec[icount]
                                nj=n_i_vec[jcount]
                                nk=n_i_vec[kcount]
                                ni_p=ni+1.0
                                nj_p=nj+1.0
                                nk_p=nk+1.0

                                omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_2DES_h1_QM(const_fac,-ik_m,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj_p*nk*prefactor_2DES_h1_QM(const_fac,-ik_p,ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk_p*prefactor_2DES_h1_QM(const_fac,ik_p,-ij_p,kbT,t1,t2)
                                omega_term=omega_term+ni*nj*nk*prefactor_2DES_h1_QM(const_fac,ik_m,-ij_p,kbT,t1,t2)

                                omega_term=omega_term+ni_p*nj*nk_p*prefactor_2DES_h1_QM(const_fac,-ik_m,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni_p*nj*nk*prefactor_2DES_h1_QM(const_fac,-ik_p,ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk_p*prefactor_2DES_h1_QM(const_fac,ik_p,-ij_m,kbT,t1,t2)
                                omega_term=omega_term+ni*nj_p*nk*prefactor_2DES_h1_QM(const_fac,ik_m,-ij_m,kbT,t1,t2)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term

    return corr_val


@jit
def third_order_lineshape_qm_t(freqs_gs,Omega_sq,n_i_vec,gamma,kbT,t):
    corr_val=0.0+0.0j
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                        const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*gamma[icount]*gamma[jcount]/(2.0*freqs_gs[icount]*freqs_gs[jcount])
                        omega_p=freqs_gs[icount]+freqs_gs[jcount]
                        omega_m=freqs_gs[icount]-freqs_gs[jcount]
                        omegai=freqs_gs[icount]
                        omegaj=freqs_gs[jcount]
                        ni=n_i_vec[icount]
                        nj=n_i_vec[jcount]
                        ni_p=ni+1.0
                        nj_p=nj+1.0

                        # term 1
                        gamma_term=gamma_term+ni*nj_p*prefactor_3rd_order_lineshape_QM(const_fac,omega_p,-omegai,kbT,t)
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_3rd_order_lineshape_QM(const_fac,-omega_m,omegai,kbT,t)
                        gamma_term=gamma_term+ni*nj*prefactor_3rd_order_lineshape_QM(const_fac,omega_m,-omegai,kbT,t)
                        gamma_term=gamma_term+ni_p*nj*prefactor_3rd_order_lineshape_QM(const_fac,-omega_p,omegai,kbT,t)
                        # term 2  
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_3rd_order_lineshape_QM(const_fac,-omegai,omega_p,kbT,t)
                        gamma_term=gamma_term+ni*nj_p*prefactor_3rd_order_lineshape_QM(const_fac,omegai,-omega_m,kbT,t)
                        gamma_term=gamma_term+ni_p*nj*prefactor_3rd_order_lineshape_QM(const_fac,-omegai,omega_m,kbT,t)
                        gamma_term=gamma_term+ni*nj*prefactor_3rd_order_lineshape_QM(const_fac,omegai,-omega_p,kbT,t)
                        # term 3
                        gamma_term=gamma_term+ni_p*nj_p*prefactor_3rd_order_lineshape_QM(const_fac,omegaj,omegai,kbT,t)
                        gamma_term=gamma_term+ni*nj_p*prefactor_3rd_order_lineshape_QM(const_fac,omegaj,-omegai,kbT,t)
                        gamma_term=gamma_term+ni_p*nj*prefactor_3rd_order_lineshape_QM(const_fac,-omegaj,omegai,kbT,t)
                        gamma_term=gamma_term+ni*nj*prefactor_3rd_order_lineshape_QM(const_fac,-omegaj,-omegai,kbT,t)
                        jcount=jcount+1
        icount=icount+1

    # now do the more complicated term that is a sum over 3 indices:
    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                kcount=0
                while kcount<freqs_gs.shape[0]:
                                const_fac=4.0*math.pi**2.0*Omega_sq[icount,jcount]*Omega_sq[jcount,kcount]*Omega_sq[icount,kcount]
                                const_fac=const_fac/(freqs_gs[icount]*freqs_gs[jcount]*freqs_gs[kcount])
                                ik_p=freqs_gs[icount]+freqs_gs[kcount]
                                ik_m=freqs_gs[icount]-freqs_gs[kcount]
                                ij_p=freqs_gs[icount]+freqs_gs[jcount]
                                ij_m=freqs_gs[icount]-freqs_gs[jcount]
                                ni=n_i_vec[icount]
                                nj=n_i_vec[jcount]
                                nk=n_i_vec[kcount]
                                ni_p=ni+1.0
                                nj_p=nj+1.0
                                nk_p=nk+1.0

                                omega_term=omega_term+ni_p*nj_p*nk_p*prefactor_3rd_order_lineshape_QM(const_fac,-ik_m,ij_p,kbT,t)
                                omega_term=omega_term+ni_p*nj_p*nk*prefactor_3rd_order_lineshape_QM(const_fac,-ik_p,ij_p,kbT,t)
                                omega_term=omega_term+ni*nj*nk_p*prefactor_3rd_order_lineshape_QM(const_fac,ik_p,-ij_p,kbT,t)
                                omega_term=omega_term+ni*nj*nk*prefactor_3rd_order_lineshape_QM(const_fac,ik_m,-ij_p,kbT,t)

                                omega_term=omega_term+ni_p*nj*nk_p*prefactor_3rd_order_lineshape_QM(const_fac,-ik_m,ij_m,kbT,t)
                                omega_term=omega_term+ni_p*nj*nk*prefactor_3rd_order_lineshape_QM(const_fac,-ik_p,ij_m,kbT,t)
                                omega_term=omega_term+ni*nj_p*nk_p*prefactor_3rd_order_lineshape_QM(const_fac,ik_p,-ij_m,kbT,t)
                                omega_term=omega_term+ni*nj_p*nk*prefactor_3rd_order_lineshape_QM(const_fac,ik_m,-ij_m,kbT,t)

                                kcount=kcount+1
                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term
    return corr_val

@jit
def prefactor_3rd_order_lineshape_QM(const_fac,omega1,omega2,kbT,t):
    val=const_fac/(4.0*math.pi**2.0)
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0+0.0j

    # deal separately with omega12=0 and omega2=0
    if abs(omega1)<tol and abs(omega2)<tol:
        full_contribution=-val/6.0*1j*t**3.0   # check this limit!!!!
    elif abs(omega12)<tol:
        num=1j*omega1*t-t**2.0*omega1**2.0/2.0+1.0-cmath.exp(1j*omega1*t)   # double check. Is the limit correct?
        denom=omega1**2.0*omega2
        full_contribution=val*num/denom
    elif abs(omega1)<tol:
        num=2.0*(cmath.exp(-1j*omega2*t)-1.0)+1j*omega2*t*(cmath.exp(-1j*omega2*t)+1.0)
        denom=omega2*omega2*omega12
        full_contribution=val*num/denom

    elif abs(omega2)<tol:     # double check the t^2 sign here as well
        num=-1j*omega1*t+(1.0-cmath.exp(-1j*omega1*t))-1.0/2.0*omega1**2.0*t**2.0
        denom=omega1*omega12*omega1
        full_contribution=val*num/denom
    else:
        num=cmath.exp(-1j*omega2*t)-1.0+omega2/omega12*(1.0-cmath.exp(-1j*omega12*t))+omega1/omega2*(cmath.exp(-1j*omega2*t)-1.0+1j*omega2*t)
        denom=omega1*omega2*omega12
        full_contribution=num/denom*val

    return full_contribution


# prefactor by Jung that turns the classical two time correlation function into its quantum counterpart:
@jit 
def prefactor_jung(omega1,omega2,kbT):
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0
    beta=1.0/kbT
    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1)<tol and abs(omega2)<tol:
        full_contribution=1.0
    elif abs(omega12)<tol:
        num=beta*beta*omega2*omega1
        denom=2.0*(1.0+beta*omega1-np.exp(beta*omega1))
        full_contribution=num/denom
    elif abs(omega1)<tol:
        num=beta*beta*omega2*omega2
        denom=2.0*(1.0-np.exp(-beta*omega2)-beta*omega2*np.exp(-beta*omega2))
        full_contribution=num/denom
    elif abs(omega2)<tol:
        num=beta*beta*omega1*omega1
        denom=2.0*(np.exp(-beta*omega1)+beta*omega1-1.0)
        full_contribution=num/denom
    else:
        num=beta*beta*omega1*omega2*omega12
        denom=2.0*(omega2*np.exp(-beta*omega12)-omega12*np.exp(-beta*omega2)+omega1)
        full_contribution=num/denom

    return full_contribution


# Exact contribution to the 3rd order correlation function correction h1(t1,t2) to 
# the 3rd order response function, for a specific value of omega1 and omega2. 
# WORK THIS OUT PROPERLY, THEN SPEED UP THE CODE
@jit
def prefactor_2DES_h1_QM(const_fac,omega1,omega2,kbT,t1,t2):
    val=const_fac/(4.0*math.pi**2.0)
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0+0.0j
    
    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1)<tol and abs(omega2)<tol:
        full_contribution=-1j*val/2.0*t1**2.0*t2   
    elif abs(omega12)<tol:
        num=-1j*t2*(1j*t1-(1.0-cmath.exp(-1j*omega2*t1))/omega2)   # double check. Is the limit correct?
        denom=omega1
        full_contribution=val*num/denom
    elif abs(omega1)<tol:
        num=(1.0-cmath.exp(1j*omega2*t2))*cmath.exp(-1j*omega2*t1)*(1j*omega2*t1-cmath.exp(1j*omega2*t1)+1.0)
        denom=omega2**3.0
        full_contribution=val*num/denom
    elif abs(omega2)<tol:     
        num=(1.0-cmath.exp(1j*omega1*t2))*((1.0-cmath.exp(-1j*omega1*t1))/omega1-1j*t1)
        denom=omega1**2.0
        full_contribution=val*num/denom
    else:
        num=(1.0-cmath.exp(1j*omega12*t2))*((1.0-cmath.exp(-1j*omega12*t1))/omega12-(1.0-cmath.exp(-1j*omega2*t1))/omega2)
        denom=omega1*omega12
        full_contribution=num/denom*val

    return full_contribution

# h1 prefactor if the quantum correlation function is approximately reconstructed from its 
# classical counterpart
@jit
def prefactor_2DES_h1_cl(const_fac,omega1,omega2,kbT,t1,t2):
    full_contribution=prefactor_jung(omega1,omega2,kbT)*prefactor_2DES_h1_QM(const_fac,omega1,omega2,kbT,t1,t2)

    return full_contribution


# Exact contribution h2 is very closely related to h1. Reuse the code for h1. 
@jit
def prefactor_2DES_h2_QM(const_fac,omega1,omega2,kbT,t1,t2):
    return prefactor_2DES_h1_QM(const_fac,omega2,omega1,kbT,t1,t2)


# h2 prefactor if the quantum correlation function is approximately reconstructed from its 
# classical counterpart, which breaks the symmetry between omega1 and omega2 that allows one
# to write the h2_QM prefactor in terms of the h1_QM prefactor
@jit
def prefactor_2DES_h2_cl(const_fac,omega1,omega2,kbT,t1,t2):
    full_contribution=prefactor_jung(omega1,omega2,kbT)*prefactor_2DES_h2_QM(const_fac,omega1,omega2,kbT,t1,t2)

    return full_contribution




# H3 is a function of 3 variables. We will probably run into storage issues here (16 GB of storage to store full array)
# therefore, h3 contribution has to be computed on the fly 
@jit
def prefactor_2DES_h3_QM(const_fac,omega1,omega2,kbT,t1,t2,t3):
    val=const_fac/(4.0*math.pi**2.0)
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0+0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1)<tol and abs(omega2)<tol:
        full_contribution=-1j*val*t1*t2*t3
    elif abs(omega12)<tol:
        num=1j*t3*(1.0-cmath.exp(-1j*omega2*t1))*(1.0-cmath.exp(-1j*omega1*t2))
        denom=omega1*omega2
        full_contribution=val*num/denom
    elif abs(omega1)<tol:
        num=-1j*(1.0-cmath.exp(1j*omega2*t3))*(1.0-cmath.exp(-1j*omega2*t1))*t2
        denom=omega2**2.0
        full_contribution=val*num/denom
    elif abs(omega2)<tol:
        num=-1j*(1.0-cmath.exp(1j*omega1*t3))*(1.0-cmath.exp(-1j*omega1*t2))*t1
        denom=omega1**2.0
        full_contribution=val*num/denom
    else:
        num=-(1.0-cmath.exp(1j*omega12*t3))*(1.0-cmath.exp(-1j*omega2*t1))*(1.0-cmath.exp(-1j*omega1*t2))
        denom=omega1*omega12*omega2
        full_contribution=num/denom*val

    return full_contribution

# H3 is a function of 3 variables. We will probably run into storage issues here (16 GB of storage to store full array)
# therefore, h3 contribution has to be computed on the fly 
@jit
def prefactor_2DES_h3_cl(const_fac,omega1,omega2,kbT,t1,t2,t3):
    full_contribution=prefactor_jung(omega1,omega2,kbT)*prefactor_2DES_h3_QM(const_fac,omega1,omega2,kbT,t1,t2,t3)

    return full_contribution



# Exact contribution to the 3rd order correlation function correction h4(t1,t2) to 
# the 3rd order response function, for a specific value of omega1 and omega2. 
# WORK THIS OUT PROPERLY, THEN SPEED UP THE CODE
@jit
def prefactor_2DES_h4_QM(const_fac,omega1,omega2,kbT,t1,t2):
    val=const_fac/(4.0*math.pi**2.0)
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0+0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1)<tol and abs(omega2)<tol:
        full_contribution=-1j*val/2.0*t1*t2**2.0
    elif abs(omega12)<tol:
        num=(1.0-cmath.exp(-1j*omega2*t1))*(1.0+cmath.exp(1j*omega2*t2)*(1j*omega2*t2-1.0))  # double check. Is the limit correct?
        denom=omega2**3.0
        full_contribution=val*num/denom
    elif abs(omega1)<tol:
        num=-(1.0-cmath.exp(-1j*omega2*t1))*(1j*t2+(1.0-cmath.exp(1j*omega2*t2))/omega2)
        denom=omega2**2.0
        full_contribution=val*num/denom
    elif abs(omega2)<tol:
        num=-1j*t1*((1.0-cmath.exp(-1j*omega1*t2))/omega1-1j*t2)
        denom=omega1
        full_contribution=val*num/denom
    else:
        num=-(1.0-cmath.exp(-1j*omega2*t1))*((1.0-cmath.exp(-1j*omega1*t2))/omega1+(1.0-cmath.exp(1j*omega2*t2))/omega2)
        denom=omega2*omega12
        full_contribution=num/denom*val

    return full_contribution


@jit
def prefactor_2DES_h4_cl(const_fac,omega1,omega2,kbT,t1,t2):
    full_contribution=prefactor_jung(omega1,omega2,kbT)*prefactor_2DES_h4_QM(const_fac,omega1,omega2,kbT,t1,t2)

    return full_contribution


# Exact contribution to the 3rd order correlation function correction h5(t1,t2) to 
# the 3rd order response function, for a specific value of omega1 and omega2. 
# WORK THIS OUT PROPERLY, THEN SPEED UP THE CODE
@jit
def prefactor_2DES_h5_QM(const_fac,omega1,omega2,kbT,t1,t2):
    val=const_fac/(4.0*math.pi**2.0)
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0+0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1)<tol and abs(omega2)<tol:
        full_contribution=-1j*val/2.0*t1*t2**2.0
    elif abs(omega12)<tol:
        num=-(1.0-cmath.exp(-1j*omega2*t1))*(1j*t2+(1.0-cmath.exp(1j*omega2*t2))/omega2)  # double check. Is the limit correct?
        denom=omega2**2.0
        full_contribution=val*num/denom
    elif abs(omega1)<tol:
        num=-(1.0-cmath.exp(-1j*omega2*t1))*(cmath.exp(1j*omega2*t2)*(1.0-1j*omega2*t2)-1.0)
        denom=omega2**3.0
        full_contribution=val*num/denom
    elif abs(omega2)<tol:
        num=-1j*t1*((1.0-cmath.exp(1j*omega1*t2))/omega1+1j*t2)
        denom=omega1
        full_contribution=val*num/denom
    else:
        num=-(1.0-cmath.exp(-1j*omega2*t1))*((1.0-cmath.exp(1j*omega12*t2))/omega12-(1.0-cmath.exp(1j*omega2*t2))/omega2)
        denom=omega1*omega2
        full_contribution=num/denom*val

    return full_contribution


@jit
def prefactor_2DES_h5_cl(const_fac,omega1,omega2,kbT,t1,t2):
    full_contribution=prefactor_jung(omega1,omega2,kbT)*prefactor_2DES_h5_QM(const_fac,omega1,omega2,kbT,t1,t2)

    return full_contribution


# This is the Jung correction factor applied to the 3rd order lineshape function
@jit
def prefactor_3rd_order_lineshape(const_fac,omega1,omega2,kbT,t):
    val=const_fac/(8.0*math.pi**2.0*kbT**2.0)
    omega12=omega1+omega2
    tol=10.0**(-15)
    full_contribution=0.0+0.0j

    # deal separately with omega12=0 and omega2=0
    if abs(omega1)/kbT<tol and abs(omega2)/kbT<tol:
        full_contribution=-val/3.0*1j*kbT**2.0*t**3.0   
    elif abs(omega12)/kbT<tol:
        num=1j*omega1*t-t**2.0*omega1**2.0/2.0+1.0-cmath.exp(1j*omega1*t)   
        denom=omega1*(1.0+omega1/kbT-cmath.exp(omega1/kbT))
        full_contribution=val*num/denom
    elif abs(omega1)/kbT<tol:
        num=2.0*(cmath.exp(-1j*omega2*t)-1.0)+1j*omega2*t*(cmath.exp(-1j*omega2*t)+1.0)
        denom=omega2*(1.0-cmath.exp(-omega2/kbT)-omega2/kbT*cmath.exp(-omega2/kbT))
        full_contribution=val*num/denom

    elif abs(omega2)/kbT<tol:     
        num=-1j*omega1*t+(1.0-cmath.exp(-1j*omega1*t))-1.0/2.0*omega1**2.0*t**2.0
        denom=omega1*(cmath.exp(-omega1/kbT)+omega1/kbT-1.0)
        full_contribution=val*num/denom
    else:
        num=cmath.exp(-1j*omega2*t)-1.0+omega2/omega12*(1.0-cmath.exp(-1j*omega12*t))+omega1/omega2*(cmath.exp(-1j*omega2*t)-1.0+1j*omega2*t)
        denom=omega2*cmath.exp(-omega12/kbT)-omega12*cmath.exp(-omega2/kbT)+omega1
        full_contribution=num/denom*val

    return full_contribution

@jit
def second_order_lineshape_cl_t(freqs_gs,Omega_sq,gamma,kbT,t):
    corr_val=0.0+0.0j
    tol=10.0e-12
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        const_fac=gamma[icount]**2.0/(2.0*freqs_gs[icount]**3.0)
        # GAMMA term of the lineshape function
        n=bose_einstein(freqs_gs[icount],kbT)
        gamma_term=gamma_term+const_fac*(2.0*n+1.0-1j*freqs_gs[icount]*t-(n+1.0)*cmath.exp(-1j*freqs_gs[icount]*t)-n*cmath.exp(1j*freqs_gs[icount]*t))

        icount=icount+1

    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
        while jcount<freqs_gs.shape[0]:
                const_val=kbT*Omega_sq[icount,jcount]**2.0/(2.0*(freqs_gs[icount]*freqs_gs[jcount])**2.0)
                omega_p=freqs_gs[icount]+freqs_gs[jcount]
                omega_m=freqs_gs[icount]-freqs_gs[jcount]

                n_p=bose_einstein(omega_p,kbT)
                omega_p_term=const_val*(2.0*n_p+1.0-1j*omega_p*t-(n_p+1.0)*cmath.exp(-1j*omega_p*t)-n_p*cmath.exp(1j*omega_p*t))/omega_p

		# deal with the limit of omega_i=omega_j
                if abs(omega_m)>tol:
                        n_m=bose_einstein(omega_m,kbT)
                        omega_m_term=const_val*(2.0*n_m+1.0-1j*omega_m*t-(n_m+1.0)*cmath.exp(-1j*omega_m*t)-n_m*cmath.exp(1j*omega_m*t))/omega_m
                else:
                        omega_m_term=const_val*(kbT*t**2.0)   # check this limit. 

                omega_term=omega_term+omega_p_term+omega_m_term

                jcount=jcount+1
        icount=icount+1
    corr_val=omega_term+gamma_term
    return corr_val

@jit
def second_order_lineshape_qm_t(freqs_gs,Omega_sq,gamma,kbT,t):
    corr_val=0.0+0.0j
    tol=10.0e-12
    gamma_term=0.0+0.0j
    omega_term=0.0+0.0j
    # start with gamma term first:
    icount=0
    while icount<freqs_gs.shape[0]:
        const_fac=gamma[icount]**2.0/(2.0*freqs_gs[icount]**3.0)
        # GAMMA term of the lineshape function
        n=bose_einstein(freqs_gs[icount],kbT)
        gamma_term=gamma_term+const_fac*(2.0*n+1.0-1j*freqs_gs[icount]*t-(n+1.0)*cmath.exp(-1j*freqs_gs[icount]*t)-n*cmath.exp(1j*freqs_gs[icount]*t))

        icount=icount+1

    icount=0
    while icount<freqs_gs.shape[0]:
        jcount=0
	n_i=bose_einstein(freqs_gs[icount],kbT)
        while jcount<freqs_gs.shape[0]:
		n_j=bose_einstein(freqs_gs[jcount],kbT)
                const_val=Omega_sq[icount,jcount]**2.0/(2.0*(freqs_gs[icount]*freqs_gs[jcount]))
                omega_p=freqs_gs[icount]+freqs_gs[jcount]
                omega_m=freqs_gs[icount]-freqs_gs[jcount]

                omega_p_term=const_val*(2.0*n_i*n_j+n_i+n_j+1.0-(n_i+n_j+1.0)*1j*omega_p*t-(n_i+1.0)*(n_j+1.0)*cmath.exp(-1j*omega_p*t)-n_i*n_j*cmath.exp(1j*omega_p*t))/omega_p**2.0

		#deal with the limit of omega_i=omega_j
                if abs(omega_m)>tol:
			# CORRECTION?
			omega_m_term=const_val*(2.0*n_i*n_j+n_i+n_j+(n_i-n_j)*1j*omega_m*t-(n_i+1.0)*n_j*cmath.exp(-1j*omega_m*t)-n_i*(n_j+1.0)*cmath.exp(1j*omega_m*t))/omega_m**2.0
                        #omega_m_term=const_val*(2.0*n_i*n_j+n_i+n_j-(n_i-n_j)*1j*omega_m*t-(n_i+1.0)*n_j*cmath.exp(-1j*omega_m*t)-n_i*(n_j+1.0)*cmath.exp(1j*omega_m*t))/omega_m**2.0
                else:
                        omega_m_term=const_val*((n_i*n_j+n_i)*t**2.0)   # check this limit. 

                omega_term=omega_term+omega_p_term+omega_m_term
		
                jcount=jcount+1
        icount=icount+1
 
    corr_val=omega_term+gamma_term
    return corr_val



