#!/usr/bin/env python

from scipy import integrate
import time
import sys
import os
import numpy as np
import math
import cmath
from numba import config
import scipy.integrate
import scipy.interpolate
import spec_pkg.constants.constants as const
from spec_pkg.GBOM import FC2DES
from spec_pkg.GBOM import gbom
from spec_pkg.GBOM import extract_model_params_gaussian as gaussian_params
from spec_pkg.GBOM import extract_model_params_from_terachem as terachem_params
from spec_pkg.GBOM import hessian_to_GBOM as hess_to_gbom
from spec_pkg.linear_spectrum import linear_spectrum
from spec_pkg.nonlinear_spectrum import twoDES
from spec_pkg.solvent_model import solvent_model
from spec_pkg.cumulant import md_traj
from spec_pkg.cumulant import extract_MD_params_terachem as terachem_params_MD 
from spec_pkg.params import params
from spec_pkg.Morse import morse
from spec_pkg.Morse import morse_2DES


# TODO ##########################################################################
# 1) Write simulation information to stdout (what calculation is done, how it   #
#    progresses etc.                            #
# 2) Expand the GBOM_MD type approaches. E-ZTFC fits in that framework, as do   #
#        a variety of other approaches.                                         #
# 3) Make sure the input file format works smoothly with the GBOM batch format  #
#        ie the case where we have multiple GBOMs to average over (this could   #
#        be the E-FTFC type approach or other approaches). Make sure that we    #
#        can specify different amounts of frozen solvent environment per GBOM   #
# 4) Interface with more electronic structure codes                             #
# 5) PIMD version of GBOM approaches?                                           #
# 6) PIMD-bead version of MDtraj data type                                      #
# 7) Expand nonlinear spectroscopy. Speed up 3rd order cumulant 2DES. Laser     #
#        pulse shapes as user defined input                                     #
# 8) Curvilinear coordinates instead of normal mode coordinates for the GBOM    #
#        which will help with large amplitude motion for low frequency modes    #
#        that tend to break the Condon approximation                            #
# 9) Cumulant approach is only valid if energy gap fluctuations are Gaussian    #
#    We need to implement a measure of Gaussian-ness for the energy gap flucts  #
#    of the GBOM and calulate it in the beginning.                              #
#################################################################################

def print_banner(stdout):
        stdout.write('\n')
        stdout.write('    ###########################################################'+'\n')
        stdout.write('    ###########################################################'+'\n')
        stdout.write('    #  __  __       _ ____                  _    ____         #'+'\n')
        stdout.write(r'    # |  \/  | ___ | / ___| ____   ___  ___| | _|  _ \ _  __  #'+'\n')
        stdout.write(r'    # | |\/| |/ _ \| \___ \| ._ \ / _ \/ __| |/ / |_) | | | | #'+'\n')
        stdout.write('    # | |  | | (_) | |___) | |_) |  __/ (__|   <|  __/| |_| | #'+'\n')
        stdout.write(r'    # |_|  |_|\___/|_|____/| .__/ \___|\___|_|\_\_|    \__, | #'+'\n')
        stdout.write('    #                      |_|                         |___/  #'+'\n')
        stdout.write('    ###########################################################'+'\n')
        stdout.write('    ###########################################################'+'\n')
        stdout.write('    #                     Version 0.1                         #'+'\n')
        stdout.write('    # Copyright (C) 2019-2020 Tim J. Zuehlsdorff. The source  #'+'\n')
        stdout.write('    # code is subject to the terms of the Mozilla Public      #'+'\n')
        stdout.write('    # License, v. 2.0. This program is distributed in the     #'+'\n')
        stdout.write('    # hope that it will be useful, but WITHOUT ANY WARRANTY.  #'+'\n')
        stdout.write('    # See the Mozilla Public License, v. 2.0 for details.     #'+'\n')
        stdout.write('    ###########################################################'+'\n')
        stdout.write('    #                   Acknowledgements:                     #'+'\n')
        stdout.write('    # The purpose of this code is to generate linear and non- #'+'\n')
        stdout.write('    # linear optical spectra for a variety of realistic and   #'+'\n')
        stdout.write('    # simplified model systems in a variety of different      #'+'\n')
        stdout.write('    # approximations.                                         #'+'\n')
        stdout.write('    # The underlying algorithms implemented in this code are  #'+'\n')
        stdout.write('    # described in the following publications:                #'+'\n')
        stdout.write('    # 1) T. J. Zuehlsdorff, A. Montoya-Castillo, J. A. Napoli,#'+'\n')
        stdout.write('    #    T. E. Markland, and C. M. Isborn, J. Chem. Phys. 151 #'+'\n')
        stdout.write('    #    074111 (2019).                                       #'+'\n')
        stdout.write('    # 2) T. J. Zuehlsdorff, H. Hong, L. Shi, and C. M. Isborn,#'+'\n')
        stdout.write('    #    J. Chem. Phys. 153, 044127 (2020).                   #'+'\n')
        stdout.write('    # 3) T. J. Zuehlsdorff, and C. M. Isborn, J. Chem. Phys.  #'+'\n')
        stdout.write('    #    148, 024110 (2018).                                  #'+'\n')
        stdout.write('    # 4) T. J. Zuehlsdorff, J. A. Napoli, J. M. Milanese, T.  #'+'\n')
        stdout.write('    #    E. Markland, and C. M. Isborn, J. Chem. Phys. 149,   #'+'\n')
        stdout.write('    #    024107 (2018).                                       #'+'\n')
        stdout.write('    # The code for computing Franck-Condon linear absorption  #'+'\n')
        stdout.write('    # and emission spectra is based on the algorithm des-     #'+'\n')
        stdout.write('    # cribed in: B. de Souza, F. Neese, and R. Izsak, J.      #'+'\n')
        stdout.write('    # Chem. Phys. 148, 034104 (2018).                         #'+'\n')
        stdout.write('    # Part of the interface between MolSpeckPy and Terachem   #'+'\n')
        stdout.write('    # is based on code originally written by Ajay Khanna.     #'+'\n')
        stdout.write('    ###########################################################'+'\n'+'\n')                                 

# Compute the absorption spectrum for a Morse oscillator with 2 frequencies coupled by 
# a Duschinsky rotation 
def compute_coupled_morse_absorption(param_list,coupled_morse,solvent,is_emission):
        # first compute solvent response. This is NOT optional for the Morse oscillator, same
                # as in the GBOM
                solvent.calc_spectral_dens(param_list.num_steps)
                solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
                solvent.calc_solvent_response(is_emission)

                # figure out start and end values over which we compute the spectrum
                # at the moment this is a Hack because we have no expression to analytically 
                # evaluate the average energy gap of the Morse oscillator. 
                E_start=param_list.E_adiabatic-param_list.spectral_window/2.0
                E_end=param_list.E_adiabatic+param_list.spectral_window/2.0

                # exact solution to the morse oscillator
                if param_list.method=='EXACT':
                        coupled_morse.compute_exact_response(param_list.temperature,param_list.max_t,param_list.num_steps)
                        spectrum=linear_spectrum.full_spectrum(coupled_morse.exact_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                        np.savetxt('Morse_Duschinsky_coupled_exact_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                elif param_list.method=='FC_HARMONIC':
                        coupled_morse.compute_harmonic_FC_response_func(param_list.temperature,param_list.max_t,param_list.num_steps,False,False,param_list.stdout)
                        spectrum=linear_spectrum.full_spectrum(coupled_morse.harmonic_fc_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                        np.savetxt('Morse_Duschinsky_harmonic_fc_spectrum.dat', spectrum,header='Energy (eV)      Intensity (arb. units)')

                # cumulant based approach:
                elif param_list.method=='CUMULANT':
                        coupled_morse.compute_exact_corr(param_list.temperature,param_list.decay_length,param_list.num_steps*10,param_list.max_t*10.0)
                        np.savetxt('Morse_duschinsky_2nd_order_corr_real.dat',np.real(coupled_morse.exact_2nd_order_corr))
                        temp_func=np.real(coupled_morse.exact_2nd_order_corr)
                        temp_func[:,1]=np.imag(coupled_morse.exact_2nd_order_corr[:,1])
                        np.savetxt('Morse_duschinsky_2nd_order_corr_imag.dat',temp_func)
                        coupled_morse.compute_spectral_dens()
                        np.savetxt('Morse_duschinsky_spectral_dens.dat',coupled_morse.spectral_dens)
                        coupled_morse.compute_2nd_order_cumulant_response(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.stdout)
                        spectrum=linear_spectrum.full_spectrum(coupled_morse.cumulant_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                        np.savetxt('Morse_Duschinsky_second_order_cumulant_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

                # Andres Hybrid approach
                elif param_list.method=='CUMUL_FC_SEPARABLE':
                        # Set average energy gap for GBOM
                        coupled_morse.eff_gbom.calc_omega_av_qm(param_list.temperature,is_emission)
                        coupled_morse.compute_cumul_fc_hybrid_response_func(param_list.temperature,param_list.decay_length,param_list.max_t,param_list.num_steps,is_emission,param_list.stdout)
                        spectrum=linear_spectrum.full_spectrum(coupled_morse.hybrid_cumul_fc_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                        np.savetxt('Morse_duschinsky_hybrid_cumul_harmonic_FC_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')


                else:
                        sys.exit('Error: Unknown method '+param_list.method)


# currently the is_emission option does not work 
def compute_morse_absorption(param_list,morse_oscs,solvent,is_emission):
        # first compute solvent response. This is NOT optional for the Morse oscillator, same
        # as in the GBOM
        solvent.calc_spectral_dens(param_list.num_steps)
        solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
        solvent.calc_solvent_response(is_emission)

        # figure out start and end values over which we compute the spectrum
        # at the moment this is a Hack because we have no expression to analytically 
        # evaluate the average energy gap of the Morse oscillator. 
        E_start=param_list.E_adiabatic-param_list.spectral_window/2.0
        E_end=param_list.E_adiabatic+param_list.spectral_window/2.0

        # exact solution to the morse oscillator
        if param_list.method=='EXACT':
            morse_oscs.compute_total_exact_response(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.herzberg_teller)
            spectrum=linear_spectrum.full_spectrum(morse_oscs.total_exact_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
            np.savetxt('Morse_exact_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
        # The effective FC spectrum for this oscillator
        elif param_list.method=='FC_HARMONIC':
            morse_oscs.compute_harmonic_FC_response_func(param_list.temperature,param_list.max_t,param_list.num_steps,False,False,param_list.stdout)  # NO emission and No Herzberg-Teller implemented at the moment
            spectrum=linear_spectrum.full_spectrum(morse_oscs.harmonic_fc_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
            np.savetxt('Morse_harmonic_fc_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

        # cumulant based approach:
        elif param_list.method=='CUMULANT':
            morse_oscs.compute_total_corr_func_exact(param_list.temperature,param_list.decay_length,param_list.max_t*10.0,param_list.num_steps*10)
            np.savetxt('Morse_oscs_2nd_order_corr_real.dat',np.real(morse_oscs.exact_2nd_order_corr))
            temp_func=np.real(morse_oscs.exact_2nd_order_corr)
            temp_func[:,1]=np.imag(morse_oscs.exact_2nd_order_corr[:,1])
            np.savetxt('Morse_oscs_2nd_order_corr_imag.dat',temp_func)
            morse_oscs.compute_spectral_dens()
            np.savetxt('Morse_oscs_spectral_dens.dat',morse_oscs.spectral_dens)
            morse_oscs.compute_2nd_order_cumulant_response(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.stdout,param_list.herzberg_teller)
            spectrum=linear_spectrum.full_spectrum(morse_oscs.cumulant_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
            np.savetxt('Morse_second_order_cumulant_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

        # Andres Hybrid approach
        elif param_list.method=='CUMUL_FC_SEPARABLE':
            # Set average energy gap for GBOM
            morse_oscs.eff_gbom.calc_omega_av_qm(param_list.temperature,is_emission)
            morse_oscs.compute_cumul_fc_hybrid_response_func(param_list.temperature,param_list.decay_length,param_list.max_t,param_list.num_steps,is_emission,param_list.stdout)
            spectrum=linear_spectrum.full_spectrum(morse_oscs.hybrid_cumul_fc_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout) 
            np.savetxt('Morse_hybrid_cumul_harmonic_FC_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

        # nothing else implemented yet. However, in the future, we could have
        # a 2nd order cumulant approach by analytically evaluating classical or
        # quantum correlation functions. 
        else:
            sys.exit('Error: Unknown method '+param_list.method)
        

# specific routines:
# compute absorption spectra and print them if chromophore model is defined purely a single GBOM
def compute_GBOM_absorption(param_list,GBOM_chromophore,solvent,is_emission):
        # first compute solvent response
        solvent.calc_spectral_dens(param_list.num_steps)
        solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
        solvent.calc_solvent_response(is_emission)

        # if this is an emission calculation, need to reset some standard gbom parameters:
        if is_emission:
                GBOM_chromophore.set_emission_variables()        
        # figure out start and end value for the spectrum.
        if param_list.exact_corr:
                GBOM_chromophore.calc_omega_av_qm(param_list.temperature,is_emission)
                E_start=GBOM_chromophore.omega_av_qm-param_list.spectral_window/2.0
                E_end=GBOM_chromophore.omega_av_qm+param_list.spectral_window/2.0
        else:
                GBOM_chromophore.calc_omega_av_cl(param_list.temperature,is_emission)
                E_start=GBOM_chromophore.omega_av_cl-param_list.spectral_window/2.0
                E_end=GBOM_chromophore.omega_av_cl+param_list.spectral_window/2.0

        # Set an additional solvent emission shift if requried
        if is_emission and param_list.add_emission_shift:
                if param_list.exact_corr:
                    GBOM_chromophore.omega_av_qm=GBOM_chromophore.omega_av_qm-2.0*solvent.reorg
                else:
                    GBOM_chromophore.omega_av_cl=GBOM_chromophore.omega_av_cl-2.0*solvent.reorg
                E_start=E_start-2.0*solvent.reorg
                E_end=E_end-2.0*solvent.reorg

        if param_list.method=='ENSEMBLE':
                GBOM_chromophore.calc_ensemble_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission,param_list.stdout)       
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)   
                if param_list.qm_wigner_dist:
                        np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_qm_wigner_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                else:
                        np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_boltzmann_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
        elif param_list.method=='FC':
                GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.herzberg_teller,param_list.stdout)

                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.fc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                np.savetxt(param_list.GBOM_root+'_FC_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
        elif param_list.method=='EZTFC':
                GBOM_chromophore.calc_eztfc_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission,param_list.herzberg_teller,param_list.stdout)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.eztfc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                if param_list.qm_wigner_dist:
                        np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_qm_wigner_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                else:
                        np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_boltzmann_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
        elif param_list.method=='CUMULANT':
                if param_list.exact_corr:
                        # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                        GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
                        np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                        GBOM_chromophore.calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                        # only compute third order cumulant if needed
                        if param_list.third_order:
                                GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)
                                if param_list.cumulant_nongaussian_prefactor:
                                        #generate ensemble spectra to extract statistics as was done in the GBOM scan
                                        GBOM_chromophore.calc_ensemble_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission,param_list.stdout)       
                                        ensemble_spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)   
                                        y = ensemble_spectrum[:,1]
                                        x = ensemble_spectrum[:,0]
                                        y[np.where(y < 0.001)] = 0 
                                        vol = scipy.integrate.simpson(y, ensemble_spectrum[:,0], ensemble_spectrum[1,0] - ensemble_spectrum[0,0])
                                        y_normalized = y/vol
                                        mean = scipy.integrate.simpson(x*y_normalized,x, dx= x[1] - x[0])
                                        var = scipy.integrate.simpson((x-mean)**2*y_normalized, x, dx=x[1]-x[0])
                                        skew = scipy.integrate.simpson(((x-mean)**3 * y_normalized) ,x, dx= x[1] - x[0])/ (var**(3/2))
                                        kurtosis = scipy.integrate.simpson(y_normalized * (x-mean)**4  / var**2, x, x[1] - x[0]) - 3
                                        #build spline and predict prefactor, rescale g_3 exact
                                        cofs = [0.5704207280544358,0.5121728697163654,0.19152366513038535,14.037923420679276,1.9225644473520938
                                                ,1.731540247289854,-0.2850403241337555,0.27133579735792684,-1.1394462432833181,1.2203032510849585, -0.48192256127602257, 0.30922580126562926,-3.5581880060082836,0.5262346303535038, 0.20572744884039815, 0.12231112351678848]
                                        tx = [-0.45615384615384613,-0.45615384615384613,-0.45615384615384613,-0.45615384615384613,1.086923076923077,1.086923076923077
                                              , 1.086923076923077,1.086923076923077]
                                        ty = [-0.08476923076923078, -0.08476923076923078, -0.08476923076923078,-0.08476923076923078,1.6819999999999997,1.6819999999999997,1.6819999999999997,1.6819999999999997]
                                        tck = (tx,ty,cofs,3,3)
                                        spline = scipy.interpolate.SmoothBivariateSpline._from_tck(tck)
                                        prefactor = spline(skew, kurtosis)
                                        if prefactor < 0:
                                                prefactor = 0
                                        if prefactor > 1:
                                                prefactor = 1
                                        GBOM_chromophore.g3_exact[:,1] = prefactor * GBOM_chromophore.g3_exact[:,1]
                                        #check if we're interpolating or extrapolating
                                        p1,p2,p3,p4,p5 = [-0.5, 0.5],[0.1, -0.1],[0.1, 0.5],[0.66, 1.7],[1.1,1.7]
                                        interpolate = False
                                        C1,C2 = ((p5[1] - p2[1])/(p5[0] - p2[0])**2) * (skew - p2[0])**2 + p2[1], ((p4[1] - p3[1])/(p4[0] - p3[0])**2) * (skew - p3[0])**2 + p3[1]    
                                        if p1[0] <= skew and skew <= p3[0]:
                                                if C1 <= kurtosis and kurtosis <= p1[1]:
                                                        interpolate = True
                                        if p3[0] <= skew and skew <= p4[0]:
                                                if C1<= kurtosis and kurtosis <= C2:
                                                        interpolate = True
                                        if p4[0] <= skew and skew <= p5[0]:
                                                if C1 <= kurtosis and kurtosis <= p4[1]:
                                                        interpolate = True
                                        
                                        if interpolate:
                                                print("PREFACTOR: ", prefactor, " SKEW: ", skew, " KURTOSIS: ", kurtosis, " VALUE LIES IN SAMPLED REGION")
                                        else:
                                                print("PREFACTOR: ", prefactor, " SKEW: ", skew, " KURTOSIS: ", kurtosis, " WARNING! VALUE LIES OUTSIDE OF SAMPLED REGION. THIS IS AN ESTIMATE")

  

                                        
                else:
                        GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                        np.savetxt(param_list.GBOM_root+'_spectral_density_harmonic_qcf.dat', GBOM_chromophore.spectral_dens)
                        GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)

                        if param_list.third_order:
                                GBOM_chromophore.calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)


                # Check if I need HT term:
                if param_list.herzberg_teller:
                    GBOM_chromophore.compute_HT_term(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.decay_length,param_list.exact_corr,param_list.third_order,param_list.ht_dipole_dipole_only,is_emission,param_list.stdout)

                GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission,param_list.herzberg_teller)        
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                if param_list.exact_corr:
                        np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_exact_corr.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                else:
                        np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_harmonic_qcf.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
    
                # now also print out resonance raman. 
                rr_spectrum=np.zeros((GBOM_chromophore.spectral_dens.shape[0],GBOM_chromophore.spectral_dens.shape[1]))
                for i in range(GBOM_chromophore.spectral_dens.shape[0]):
                    rr_spectrum[i,0]=GBOM_chromophore.spectral_dens[i,0]
                    rr_spectrum[i,1]=GBOM_chromophore.spectral_dens[i,0]**2.0*GBOM_chromophore.spectral_dens[i,1]

                if param_list.exact_corr:
                    np.savetxt(param_list.GBOM_root+'_resonance_raman_exact_corr.dat', rr_spectrum)
                else:
                    np.savetxt(param_list.GBOM_root+'_resonance_raman_harmonic_qcf.dat', rr_spectrum)

    
        # do all approaches, including qm wigner sampling and exact and approximate 
        # quantum correlation functions for the cumulant approach
        elif param_list.method=='ALL':
                GBOM_chromophore.calc_ensemble_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission,param_list.stdout)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                if param_list.qm_wigner_dist:
                        np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_qm_wigner_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                else:
                        np.savetxt(param_list.GBOM_root+'_ensemble_spectrum_boltzmann_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

                GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t, is_emission,param_list.herzberg_teller,param_list.stdout)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.fc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                np.savetxt(param_list.GBOM_root+'_FC_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                GBOM_chromophore.calc_eztfc_response(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.qm_wigner_dist,is_emission,param_list.herzberg_teller,param_list.stdout)
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.eztfc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)

                if param_list.qm_wigner_dist:
                        np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_qm_wigner_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                else:
                        np.savetxt(param_list.GBOM_root+'_EZTFC_spectrum_boltzmann_dist.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

                if param_list.exact_corr:
                        # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                        GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
                        np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                        GBOM_chromophore.calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                        # only compute third order cumulant if needed
                        if param_list.third_order:
                                GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)
                else:
                        GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                        np.savetxt(param_list.GBOM_root+'_spectral_density_harmonic_qcf.dat', GBOM_chromophore.spectral_dens)
                        GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                        if param_list.third_order:
                                GBOM_chromophore.calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list,g3_cutoff,param_list.stdout)
                GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr, is_emission,param_list.herzberg_teller)       
                spectrum=linear_spectrum.full_spectrum(GBOM_chromophore.cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)

                if param_list.exact_corr:
                        np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_exact_corr.dat', spectrum,header='Energy (eV)      Intensity (arb. units)')
                else:
                        np.savetxt(param_list.GBOM_root+'_cumulant_spectrum_harmonic_qcf.dat', spectrum,header='Energy (eV)      Intensity (arb. units)')

                        # now also print out resonance raman. 
                rr_spectrum=np.zeros((GBOM_chromophore.spectral_dens.shape[0],GBOM_chromophore.spectral_dens.shape[1]))
                for i in range(GBOM_chromophore.spectral_dens.shape[0]):
                    rr_spectrum[i,0]=GBOM_chromophore.spectral_dens[i,0]
                    rr_spectrum[i,1]=GBOM_chromophore.spectral_dens[i,0]**2.0*GBOM_chromophore.spectral_dens[i,1]

                if param_list.exact_corr:
                    np.savetxt(param_list.GBOM_root+'_resonance_raman_exact_corr.dat', rr_spectrum)
                else:   
                    np.savetxt(param_list.GBOM_root+'_resonance_raman_harmonic_qcf.dat', rr_spectrum)


        else:
                sys.exit('Error: Unknown method '+param_list.method)


# compute absorption spectra when chromophore model is given by a batch of GBOMS
def compute_GBOM_batch_absorption(param_list,GBOM_batch,solvent,is_emission):
        # first compute solvent response
        solvent.calc_spectral_dens(param_list.num_steps)
        solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
        solvent.calc_solvent_response(is_emission)
        
        # Now make sure that we have only a single average spectral window for the GBOM batch. 
        # also set the correct omega_av.
        icount=0
        average_Egap=0.0
        while icount<GBOM_batch.num_gboms:
        # figure out start and end value for the spectrum.
                if param_list.exact_corr:
                        GBOM_batch.gboms[icount].calc_omega_av_qm(param_list.temperature,is_emission)
                        average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
                else:
                        GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
                        average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl

                icount=icount+1

        average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)

        E_start=average_Egap-param_list.spectral_window/2.0
        E_end=average_Egap+param_list.spectral_window/2.0

        if param_list.method=='FC':
                # Compute FC response for all elements in the GBOM batch
                icount=0 
                spectrum=np.zeros((param_list.num_steps,2))
                while icount<GBOM_batch.num_gboms:
                        GBOM_batch.gboms[icount].calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.herzberg_teller,param_list.stdout)

                        temp_spectrum=linear_spectrum.full_spectrum(GBOM_batch.gboms[icount].fc_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                        if icount==0:
                                spectrum=spectrum+temp_spectrum
                        else:
                                spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]
                        icount=icount+1

                spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)
                np.savetxt(param_list.GBOM_root+'_E_FTFC_spectrum.dat', spectrum,header='Energy (eV)      Intensity (arb. units)')

        # cumulant spectrum for all elements in the GBOM batch. The result is the summed spectrum
        elif param_list.method=='CUMULANT':
                icount=0
                spectrum=np.zeros((param_list.num_steps,2))
                while icount<GBOM_batch.num_gboms:
                        if param_list.exact_corr:
                                # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                                GBOM_batch.gboms[icount].calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)

                                # only compute third order cumulant if needed
                                if param_list.third_order:
                                        GBOM_batch.gboms[icount].calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)

                        else:
                                GBOM_batch.gboms[icount].calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)

                                if param_list.third_order:
                                        GBOM_batch.gboms[icount].calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)

                        GBOM_batch.gboms[icount].calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,param_list.exact_corr,is_emission)

                        GBOM_batch.gboms[icount].calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission)
                        temp_spectrum=linear_spectrum.full_spectrum(GBOM_batch.gboms[icount].cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                        if icount==0:
                                sd=GBOM_batch.gboms[icount].spectral_dens
                                spectrum=temp_spectrum
                        else:
                                sd[:,1]=sd[:,1]+GBOM_batch.gboms[icount].spectral_dens[:,1]
                                spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]
                        icount=icount+1

                sd[:,1]=sd[:,1]/(1.0*GBOM_batch.num_gboms)
                np.savetxt(param_list.GBOM_root+'_av_spectral_dens.dat',sd)
                spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)
                np.savetxt(param_list.GBOM_root+'_E_cumulant_spectrum.dat', spectrum,header='Energy (eV)      Intensity (arb. units)')

        # compute an AVERAGE g2 and g3 and place that on the average energy gap of all GBOMs. This is 
        # equivalent of averaging the spectral density over different instances of the GBOM and then just
        # computing a single, effective response function. 
        elif param_list.method=='CUMULANT_AV':
                # get list of adiabatic energies and dipole moms. 
                energy_dipole=np.zeros((1,1))
                if os.path.exists(param_set.E_opt_path):
                        energy_dipole=np.genfromtxt(param_set.E_opt_path)
                else:
                        sys.exit('Error: Requested an Eopt_avCUMULANT type calculation but did not provide optimized vertical energy gaps and dipoles')


                Eopt=energy_dipole[:,0]/const.Ha_to_eV
                # compute average energy
                Eopt_av=np.sum(Eopt)/(1.0*Eopt.shape[0])
                Eopt_fluct=Eopt-Eopt_av # fluctuation of Eopt energies around common mean. 
                average_Eadiab=0.0
                average_E00=0.0    # E00 and Eadiab are not the same. 
                for icount in range(GBOM_batch.num_gboms):
                    average_Eadiab=average_Eadiab+GBOM_batch.gboms[icount].E_adiabatic
                    average_E00=average_E00+GBOM_batch.gboms[icount].E_adiabatic+0.5*np.sum(GBOM_batch.gboms[icount].freqs_ex)-0.5*np.sum(GBOM_batch.gboms[icount].freqs_gs)
                average_Eadiab=average_Eadiab/(1.0*GBOM_batch.num_gboms)
                average_E00=average_E00/(1.0*GBOM_batch.num_gboms)
                average_Egap=0.0
                icount=0
                while icount<GBOM_batch.num_gboms:
                        # Set E_00 to zero and calculate the lineshape function and energy gap. This 
                        # guarantees that all cumulant spectra start at the same 0-0 transition
                        # Then reset gboms.omega_av and recompute it for 0-0 transitions set to zero. 
                        # Then compute lineshape function for that setup. This will generate a cumulant
                        # spectrum with the 0-0 transition shifted to 0
                        GBOM_batch.gboms[icount].E_adiabatic=0.0
                        if param_set.exact_corr:
                                average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
                                GBOM_batch.gboms[icount].omega_av_qm=0.0
                        else:
                                average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl
                                GBOM_batch.gboms[icount].omega_av_cl=0.0                

                        icount=icount+1

                average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)

                
                delta_E_opt_E_adiab=Eopt_av-average_Eadiab   # The average E_adiab value should be unchanged in Eopt_avFTFC

                E_start=average_Egap-param_set.spectral_window/2.0
                E_end=average_Egap+param_set.spectral_window/2.0

                # NOW overwrite E_adiabatic and dipole moment for all GBOMS. Make sure that all GBOM's have a consistent 0-0 transition equal to average_E00
                icount=0
                while icount<GBOM_batch.num_gboms:
                        # convert from the constant E_00 to 
                        GBOM_batch.gboms[icount].E_adiabatic=average_E00-0.5*np.sum(GBOM_batch.gboms[icount].freqs_ex)+0.5*np.sum(GBOM_batch.gboms[icount].freqs_gs)
                        #GBOM_batch.gboms[icount].E_adiabatic=average_E00
                        GBOM_batch.gboms[icount].dipole_mom=energy_dipole[icount,1]
                        
                        # recompute corrected average energy gap:
                        if param_list.exact_corr:
                            GBOM_batch.gboms[icount].calc_omega_av_qm(param_list.temperature,is_emission)
                        else:
                            GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)

                        icount=icount+1

                # now compute average response function. Important: Average response function, NOT lineshape function
                average_response=np.zeros((param_list.num_steps,2),dtype=complex)
                icount=0
                while icount<GBOM_batch.num_gboms:
                        if param_list.exact_corr:
                                # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                                GBOM_batch.gboms[icount].calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)

                                # only compute third order cumulant if needed
                                if param_list.third_order:
                                        GBOM_batch.gboms[icount].calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)

                        else:

                                GBOM_batch.gboms[icount].calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)

                                if param_list.third_order:
                                        GBOM_batch.gboms[icount].calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)
                        # build average response function:
                        for j in range(average_response.shape[0]):
                            if param_list.exact_corr:
                                average_response[j,0]=GBOM_batch.gboms[icount].g2_exact[j,0]
                                if param_list.third_order:
                                    average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_exact[j,1]-GBOM_batch.gboms[icount].g3_exact[j,1])
                                else:
                                    average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_exact[j,1])
                            else:
                                average_response[j,0]=GBOM_batch.gboms[icount].g2_cl[j,0]
                                if param_list.third_order:
                                    average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_cl[j,1]-GBOM_batch.gboms[icount].g3_cl[j,1])
                                else:
                                    average_response[j,1]=average_response[j,1]+GBOM_batch.gboms[icount].dipole_mom*cmath.exp(-GBOM_batch.gboms[icount].g2_cl[j,1]) 

                        icount=icount+1
                # now average:
                average_response[:,1]=average_response[:,1]/(1.0*GBOM_batch.num_gboms)


                # now build spectrum.
                spectrum=np.zeros((average_response.shape[0],2))

                for icount in range(GBOM_batch.num_gboms):
                    eff_response_func=average_response
                    for jcount in range(eff_response_func.shape[0]):
                            eff_response_func[jcount,1]=eff_response_func[jcount,1]*cmath.exp(1j*(Eopt_av-Eopt[icount])*eff_response_func[jcount,0]/math.pi)
                    temp_spectrum=linear_spectrum.full_spectrum(eff_response_func,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)

                    np.savetxt('Eopt_spec_snapshot'+str(icount)+'.dat',temp_spectrum, header='Energy (eV)      Intensity (arb. units)')

                    if icount==0:
                        spectrum=temp_spectrum
                    else:
                        spectrum[:,0]=temp_spectrum[:,0]
                        spectrum[:,1]=spectrum[:,1]+temp_spectrum[:,1]

                spectrum[:,1]=spectrum[:,1]/(1.0*GBOM_batch.num_gboms)  

                np.savetxt(param_list.GBOM_root+'_Eopt_avcumulant_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

        else:
                sys.exit('Unknown method for GBOM_BATCH linear spectrum: '+param_list.method)


# same as GBOM_MD absorption, but this time we have a batch of GBOMs
def compute_hybrid_GBOM_batch_MD_absorption(param_list,MDtraj,GBOM_batch,solvent,is_emission):
                # first check if we need a solvent model:
                if param_list.is_solvent:
                                solvent.calc_spectral_dens(param_list.num_steps)
                                solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
                                solvent.calc_solvent_response(is_emission)

                # now fix energy range
                E_start=MDtraj.mean-param_list.spectral_window/2.0
                E_end=MDtraj.mean+param_list.spectral_window/2.0

                # initialize GBOM:
                # if this is an emission calculation, need to reset some standard gbom parameters:
                if is_emission:
                                for i in range(param_list.num_gboms):
                                        GBOM_batch.gboms[i].set_emission_variables()

                if param_list.exact_corr:
                                for i in range(param_list.num_gboms):
                                        GBOM_batch.gboms[i].calc_omega_av_qm(param_list.temperature,is_emission)
                else:
                                for i in range(param_list.num_gboms):
                                        GBOM_batch.gboms[i].calc_omega_av_cl(param_list.temperature,is_emission)
                                        print('omega av cl')
                                        print(GBOM_batch.gboms[i].omega_av_cl)

                # Andres 2nd order cumulant GBOM approach assuming that the energy gap operator is
                # fully separable
                if param_list.method=='CUMUL_FC_SEPARABLE':
                                # first compute 2nd order cumulant response for MD trajectory
                                MDtraj.calc_2nd_order_corr()
                                MDtraj.calc_spectral_dens(param_list.temperature_MD)
                                np.savetxt(param_list.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
                                MDtraj.calc_g2(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.stdout)
                                if param_list.third_order:
                                                MDtraj.calc_3rd_order_corr(param_list.corr_length_3rd,param_list.stdout)
                                                # technically, in 3rd order cumulant, can have 2 different temperatures again. one at
                                                # which the MD was performed and one at wich the spectrum is simulated. Fix this...
                                                MDtraj.calc_g3(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.low_freq_cutoff,param_list.g3_cutoff,param_list.stdout)
                                                MDtraj.calc_cumulant_response(True,is_emission,False)
                                else:
                                                MDtraj.calc_cumulant_response(False,is_emission,False)

                                # calculate 2nd order divergence for MD trajectory:
                                MDtraj.calc_2nd_order_divergence()

                                # Now compute 2nd order cumulant g2 for GBOM
                                if param_list.exact_corr:
                                                for i in range(param_list.num_gboms):
                                                        # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                                                        GBOM_batch.gboms[i].calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
                                                        #np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                        GBOM_batch.gboms[i].calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                                                        # check if this is a 3rd order cumulant calculation
                                                        if param_list.third_order:
                                                                GBOM_batch.gboms[i].calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)

                                else:
                                                for i in range(param_list.num_gboms):
                                                        GBOM_batch.gboms[i].calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                                                        #np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                        GBOM_batch.gboms[i].calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                                                        if param_list.third_order:
                                                                GBOM_batch.gboms[i].calc_g3_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)

                                # calculate FC and 2nd order cumulant response functions for GBOM
                                for i in range(param_list.num_gboms):
                                                GBOM_batch.gboms[i].calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission,param_list.herzberg_teller)
                                                GBOM_batch.gboms[i].calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.herzberg_teller,param_list.stdout)
                                                # compute 2nd order cumulant divergence:
                                                GBOM_batch.gboms[i].calc_2nd_order_divergence(param_list.temperature,param_list.exact_corr)

                                # now build effective response function. What about dipole moment? Where does it come from? MD or GBOM? If condon approx is valid
                                # it doesnt matter
                                eff_response=np.zeros((MDtraj.cumulant_response.shape[0],2),dtype=complex)
                                eff_response[:,0]=MDtraj.cumulant_response[:,0]
                                num_gboms_averaged=0
                                for j in range(param_list.num_gboms):
                                        # can i average over this GBOM? Check
                           # print('Divergence GBOM, Divergence MDtraj')
                           # print(GBOM_batch.gboms[j].second_order_divergence, MDtraj.second_order_divergence)
                                        #if GBOM_batch.gboms[j].second_order_divergence<MDtraj.second_order_divergence:
                                                num_gboms_averaged=num_gboms_averaged+1
                                                for icount in range(eff_response.shape[0]):
                                                        # protect against divide by 0
                                                        if abs(GBOM_batch.gboms[j].cumulant_response[icount,1].real)>10e-30:
                                                                eff_response[icount,1]=eff_response[icount,1]+MDtraj.cumulant_response[icount,1]*GBOM_batch.gboms[j].fc_response[icount,1]/GBOM_batch.gboms[j].cumulant_response[icount,1]
                                                        else: 
                                                                eff_response[icount,1]=eff_response[icount,1]+MDtraj.cumulant_response[icount,1]
                                eff_response[:,1]=eff_response[:,1]/(1.0*num_gboms_averaged)    

                                # now we can compute the linear spectrum based on eff_response
                                # no need for solvent model. This is taken care of in the MD trajectory
                                if param_list.is_solvent:
                                                spectrum=linear_spectrum.full_spectrum(eff_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                                else:
                                                spectrum=linear_spectrum.full_spectrum(eff_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False,is_emission,param_list.stdout)

                                np.savetxt(param_list.GBOM_root+'_cumul_FC_separable_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

                else:
                                sys.exit('Error: Method '+param_list.method+' does not work with a mixed GBOM MD model.')


#  compute absorption spectra when chromophore model is given by both GBOM batch and MD batch
# this is mainly relevant for E-ZTFC and related methods defined by ANDRES
def compute_hybrid_GBOM_MD_absorption(param_list,MDtraj,GBOM_chromophore,solvent,is_emission):
                # first check if we need a solvent model:
                if param_list.is_solvent:
                                solvent.calc_spectral_dens(param_list.num_steps)
                                solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
                                solvent.calc_solvent_response(is_emission)
                # now fix energy range
                E_start=MDtraj.mean-param_list.spectral_window/2.0
                E_end=MDtraj.mean+param_list.spectral_window/2.0

        # initialize GBOM:
        # if this is an emission calculation, need to reset some standard gbom parameters:
                if is_emission:
                                GBOM_chromophore.set_emission_variables()

                if param_list.exact_corr:
                                GBOM_chromophore.calc_omega_av_qm(param_list.temperature,is_emission)
                else:
                                GBOM_chromophore.calc_omega_av_cl(param_list.temperature,is_emission)

    
        # Andres 2nd order cumulant GBOM approach assuming that the energy gap operator is
        # fully separable
                if param_list.method=='CUMUL_FC_SEPARABLE':
                # first compute 2nd order cumulant response for MD trajectory
                                MDtraj.calc_2nd_order_corr()
                                MDtraj.calc_spectral_dens(param_list.temperature_MD)
                                np.savetxt(param_list.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
                                MDtraj.calc_g2(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.stdout)
                                if param_list.third_order:
                                                MDtraj.calc_3rd_order_corr(param_list.corr_length_3rd,param_list.stdout)
                                                # technically, in 3rd order cumulant, can have 2 different temperatures again. one at
                                                # which the MD was performed and one at wich the spectrum is simulated. Fix this...
                                                MDtraj.calc_g3(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.low_freq_cutoff,param_list.g3_cutoff,param_list.stdout)
                                                MDtraj.calc_cumulant_response(True,is_emission,False)
                                else:
                                                MDtraj.calc_cumulant_response(False,is_emission,False)


                # Now compute 2nd order cumulant g2 for GBOM
                                if param_list.exact_corr:
                                                # spectral density not needed for calculation purposes in the GBOM. just print it out anyway for analysis
                                                GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,False,is_emission)
                                                np.savetxt(param_list.GBOM_root+'_spectral_density_exact_corr.dat', GBOM_chromophore.spectral_dens)
                                                GBOM_chromophore.calc_g2_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                                                if param_list.third_order:
                                                                GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)
    
                                else:
                                                GBOM_chromophore.calc_spectral_dens(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.decay_length,True,is_emission)
                                                np.savetxt(param_list.GBOM_root+'_spectral_density_harmonic_qcf.dat', GBOM_chromophore.spectral_dens)
                                                GBOM_chromophore.calc_g2_cl(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.stdout)
                                                if param_list.third_order:
                                                                GBOM_chromophore.calc_g3_qm(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.four_phonon_term,param_list.g3_cutoff,param_list.stdout)

                # calculate FC and 2nd order cumulant response functions for GBOM
                                GBOM_chromophore.calc_cumulant_response(param_list.third_order,param_list.exact_corr,is_emission,param_list.herzberg_teller)
                                GBOM_chromophore.calc_fc_response(param_list.temperature,param_list.num_steps,param_list.max_t,is_emission,param_list.herzberg_teller,param_list.stdout)    
                
                # now build effective response function
                                # HACK... DIVIDE OUT A CERTAIN PART OF THE LINESHAPE FUNCTION --> LONG TIMESCALE DIVERGENCE
                                GBOM_chromophore.calc_2nd_order_divergence(param_list.temperature,param_list.exact_corr)
                                print(GBOM_chromophore.second_order_divergence)
                                print(GBOM_chromophore.cumulant_response[1,:])
                                for icount in range(GBOM_chromophore.cumulant_response.shape[0]):
                                                GBOM_chromophore.cumulant_response[icount,1]=GBOM_chromophore.cumulant_response[icount,1]*cmath.exp(GBOM_chromophore.second_order_divergence*GBOM_chromophore.cumulant_response[icount,0]**2.0)

                                eff_response=GBOM_chromophore.fc_response
                                for icount in range(eff_response.shape[0]):
                                                eff_response[icount,1]=eff_response[icount,1]*MDtraj.cumulant_response[icount,1]/GBOM_chromophore.cumulant_response[icount,1]

                # now we can compute the linear spectrum based on eff_response
                                if param_list.is_solvent:
                                                spectrum=linear_spectrum.full_spectrum(eff_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                                else:
                                                spectrum=linear_spectrum.full_spectrum(eff_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False,is_emission,param_list.stdout)
                                np.savetxt(param_list.GBOM_root+'_cumul_FC_separable_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

                else:
                                sys.exit('Error: Method '+param_list.method+' does not work with a mixed GBOM MD model.')

# compute absorption spectrum from pure MD input
# Solvent degrees of freedom are optional here. They can be added to provide additional
# Broadening but in principle all broadening should originate from the MD. 
# Note that a pure MD trajectory can only be used to compute Ensemble or cumulant spectrum
def compute_MD_absorption(param_list,MDtraj,solvent,is_emission):
        # first check if we need a solvent model:
        if param_list.is_solvent:
                solvent.calc_spectral_dens(param_list.num_steps)
                solvent.calc_g2_solvent(param_list.temperature,param_list.num_steps,param_list.max_t,param_list.stdout)
                solvent.calc_solvent_response(is_emission)
        # now fix energy range
        E_start=MDtraj.mean-param_list.spectral_window/2.0
        E_end=MDtraj.mean+param_list.spectral_window/2.0

        # now check if this is a cumulant or a classical ensemble calculation
        if param_list.method=='CUMULANT':
                MDtraj.calc_2nd_order_corr()
                MDtraj.calc_spectral_dens(param_list.temperature_MD)
                np.savetxt(param_list.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)

                MDtraj.calc_g2(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.stdout)
                # check if we need to compute HT corrections:
                if param_list.herzberg_teller:
                        print('COMPUTING HERZBERG TELLER TERM')
                        MDtraj.calc_ht_correction(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.corr_length_3rd,param_list.low_freq_cutoff,param_list.third_order,param_list.gs_reference_dipole,param_list.ht_dipole_dipole_only,is_emission,param_list.stdout)
                        print('Renormalized dipole mom vs standard dipole mom:')
                        print(np.dot(MDtraj.dipole_mom_av,MDtraj.dipole_mom_av),MDtraj.dipole_renorm**2.0, np.dot(MDtraj.dipole_reorg,MDtraj.dipole_reorg))
                if param_list.third_order:
                        MDtraj.calc_3rd_order_corr(param_list.corr_length_3rd,param_list.stdout)
                        # technically, in 3rd order cumulant, can have 2 different temperatures again. one at
                        # which the MD was performed and one at wich the spectrum is simulated. Fix this...
                        MDtraj.calc_g3(param_list.temperature,param_list.max_t,param_list.num_steps,param_list.low_freq_cutoff,param_list.g3_cutoff,param_list.stdout)
                        if param_list.cumulant_nongaussian_prefactor:

                                        #generate ensemble spectra to extract statistics as was done in the GBOM scan
                                        MDtraj.calc_ensemble_response(param_list.max_t,param_list.num_steps)
                                        if param_list.is_solvent:
                                                ensemble_spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                                        else:
                                                ensemble_spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False,is_emission,param_list.stdout)              
                                        y = ensemble_spectrum[:,1]
                                        x = ensemble_spectrum[:,0]
                                        y[np.where(y < 0.001)] = 0 
                                        vol = scipy.integrate.simpson(y, ensemble_spectrum[:,0], ensemble_spectrum[1,0] - ensemble_spectrum[0,0])
                                        y_normalized = y/vol
                                        mean = scipy.integrate.simpson(x*y_normalized,x, dx= x[1] - x[0])
                                        var = scipy.integrate.simpson((x-mean)**2*y_normalized, x, dx=x[1]-x[0])
                                        skew = scipy.integrate.simpson(((x-mean)**3 * y_normalized) ,x, dx= x[1] - x[0])/ (var**(3/2))
                                        kurtosis = scipy.integrate.simpson(y_normalized * (x-mean)**4  / var**2, x, x[1] - x[0]) - 3
                                        #build spline and predict prefactor, rescale g_3 exact
                                        cofs = [0.5704207280544358,0.5121728697163654,0.19152366513038535,14.037923420679276,1.9225644473520938
                                                ,1.731540247289854,-0.2850403241337555,0.27133579735792684,-1.1394462432833181,1.2203032510849585, -0.48192256127602257, 0.30922580126562926,-3.5581880060082836,0.5262346303535038, 0.20572744884039815, 0.12231112351678848]
                                        tx = [-0.45615384615384613,-0.45615384615384613,-0.45615384615384613,-0.45615384615384613,1.086923076923077,1.086923076923077
                                              , 1.086923076923077,1.086923076923077]
                                        ty = [-0.08476923076923078, -0.08476923076923078, -0.08476923076923078,-0.08476923076923078,1.6819999999999997,1.6819999999999997,1.6819999999999997,1.6819999999999997]
                                        tck = (tx,ty,cofs,3,3)
                                        spline = scipy.interpolate.SmoothBivariateSpline._from_tck(tck)
                                        prefactor = spline(skew, kurtosis)
                                        if prefactor < 0:
                                                prefactor = 0
                                        if prefactor > 1:
                                                prefactor = 1
                                        MDtraj.g3[:,1] = prefactor * MDtraj.g3[:,1]
                                        #check if we're interpolating or extrapolating
                                        p1,p2,p3,p4,p5 = [-0.5, 0.5],[0.1, -0.1],[0.1, 0.5],[0.66, 1.7],[1.1,1.7]
                                        interpolate = False
                                        C1,C2 = ((p5[1] - p2[1])/(p5[0] - p2[0])**2) * (skew - p2[0])**2 + p2[1], ((p4[1] - p3[1])/(p4[0] - p3[0])**2) * (skew - p3[0])**2 + p3[1]    
                                        if p1[0] <= skew and skew <= p3[0]:
                                                if C1 <= kurtosis and kurtosis <= p1[1]:
                                                        interpolate = True
                                        if p3[0] <= skew and skew <= p4[0]:
                                                if C1<= kurtosis and kurtosis <= C2:
                                                        interpolate = True
                                        if p4[0] <= skew and skew <= p5[0]:
                                                if C1 <= kurtosis and kurtosis <= p4[1]:
                                                        interpolate = True
                                        
                                        if interpolate:
                                                print("PREFACTOR: ", prefactor, " SKEW: ", skew, " KURTOSIS: ", kurtosis, " VALUE LIES IN SAMPLED REGION")
                                        else:
                                                print("PREFACTOR: ", prefactor, " SKEW: ", skew, " KURTOSIS: ", kurtosis, " WARNING! VALUE LIES OUTSIDE OF SAMPLED REGION. THIS IS AN ESTIMATE")

                        MDtraj.calc_cumulant_response(True,is_emission,param_list.herzberg_teller)
               
                else:
                        MDtraj.calc_cumulant_response(False,is_emission,param_list.herzberg_teller)



                # compute linear spectrum
                if param_list.is_solvent:
                        spectrum=linear_spectrum.full_spectrum(MDtraj.cumulant_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                else:
                        # set solvent response to a zero dummy vector
                        spectrum=linear_spectrum.full_spectrum(MDtraj.cumulant_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False,is_emission,param_list.stdout)
                np.savetxt(param_list.MD_root+'MD_cumulant_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
                # DO THIS LAST! CURRENTLY THIS OVERWRITES SD.

                # Also print raman intensity:
                resonance_raman=np.zeros((MDtraj.spectral_dens.shape[0],MDtraj.spectral_dens.shape[1]))                                        

                for i in range(resonance_raman.shape[0]):
                    resonance_raman[i,0]=MDtraj.spectral_dens[i,0]
                    resonance_raman[i,1]=MDtraj.spectral_dens[i,1]*MDtraj.spectral_dens[i,0]**2.0

                np.savetxt(param_list.MD_root+'MD_resonance_raman.dat', resonance_raman)

        # now do ensemble approach
        elif param_list.method=='ENSEMBLE':
                MDtraj.calc_ensemble_response(param_list.max_t,param_list.num_steps)
                if param_list.is_solvent:
                        spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,solvent.solvent_response,param_list.num_steps,E_start,E_end,True,is_emission,param_list.stdout)
                else:
                        # set solvent response to a zero dummy vector
                        spectrum=linear_spectrum.full_spectrum(MDtraj.ensemble_response,np.zeros((1,1)),param_list.num_steps,E_start,E_end,False,is_emission,param_list.stdout)              
                np.savetxt(param_list.MD_root+'MD_ensemble_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')
        else:
                sys.exit('Error: Method '+param_list.method+' does not work with a pure MD based model. Set Method to ENSEMBLE or CUMULANT.')

# main driver #
# start timer.
start_time=time.time()

input_file=sys.argv[1]
if len(sys.argv)<3:
        num_cores=1
else:
        num_cores=int(sys.argv[2])

config.NUMBA_NUM_THREADS=num_cores

# parse input values
if os.path.exists(input_file):
        param_set=params.params(input_file)
else:
        sys.exit('Error: Could not find input file')

print('PARAMSET: NUM_FROZEN_ATOMS')
print(param_set.num_frozen_atoms)

print_banner(param_set.stdout)

param_set.stdout.write('Successfully parsed the input file!'+'\n')
param_set.stdout.write('Now starting spectrum calculation'+'\n')

# set up solvent model:
if param_set.is_solvent:
        solvent_mod=solvent_model.solvent_model(param_set.solvent_reorg,param_set.solvent_cutoff_freq)
        param_set.stdout.write('Created solvent model!'+'\n')
        param_set.stdout.write('Solvent reorg:    '+str(param_set.solvent_reorg)+' Ha'+'\n')
        param_set.stdout.write('Cutoff frequency:  '+str(param_set.solvent_cutoff_freq)+' Ha'+'\n')

# set up chromophore model
# pure GBOM model. 
if param_set.model=='GBOM' or param_set.model=='MD_GBOM':
        param_set.stdout.write('\n'+'Requested a GBOM model calculation with parameter: MODEL   '+param_set.model+'\n'+'Constructing a GBOM model.'+'\n')
        # sanity check:
        if param_set.num_modes==0:
                param_set.stdout.write('Error: Model GBOM requested but number of normal modes in the system is not set!')
                sys.exit('Error: Model GBOM requested but number of normal modes in the system is not set!')
        # single GBOM
        if param_set.num_gboms==1:
                param_set.stdout.write('This is a calculation involving only a single GBOM model with '+str(param_set.num_modes)+' normal modes.'+'\n')
                # GBOM root is given. This means user requests reading params from Gaussian or Terachem
                if param_set.GBOM_root!='':             
                        if param_set.GBOM_input_code=='GAUSSIAN':
                                # build list of frozen atoms:
                                # CURRENTLY ONLY IMPLEMENTED FOR GAUSSIAN WITH HPMODES, NO HT EFFECTS AND SINGLE GBOM
                                if param_set.is_vertical_gradient:
                                    param_set.stdout.write('Attempting to compute vertical gradinet GBOM model from the following Gaussian output files: '+'\n')
                                    param_set.stdout.write(param_set.GBOM_root+'_gs.log'+'\t'+'\t'+param_set.GBOM_root+'_grad.log'+'\t'+'\n')
                                    freqs_gs,freqs_ex,K,J=gaussian_params.construct_vertical_gradient_model(param_set.GBOM_root+'_gs.log',param_set.GBOM_root+'_grad.log',param_set.num_modes)
                                    # need to implement adiabatic energy from vertical gradient and fix dipole mom to harmonic part. 
                                    param_set.E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+'_vibronic.log')  # for now just supply vibronic file
                                else:                               
                                    param_set.stdout.write('Attempting to compute GBOM model from the following Gaussian output files: '+'\n')
                                    param_set.stdout.write(param_set.GBOM_root+'_gs.log'+'\t'+param_set.GBOM_root+'_ex.log'+'\t'+param_set.GBOM_root+'_vibronic.log'+'\t'+'\n')
                                    freqs_gs=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+'_gs.log',param_set.num_modes)
                                    freqs_ex=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+'_ex.log',param_set.num_modes)
                                    K=gaussian_params.extract_Kmat(param_set.GBOM_root+'_vibronic.log',param_set.num_modes)
                                    J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
                                    if param_set.no_dusch:
                                        counter=0
                                        while counter<freqs_ex.shape[0]:
                                                J[counter,counter]=1.0
                                                counter=counter+1
                                        param_set.dipole_mom=gaussian_params.extract_transition_dipole(param_set.GBOM_root+'_ex.log',param_set.target_excited_state)
                                        param_set.E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+'_vibronic.log')

                                    else:
                                        J=gaussian_params.extract_duschinsky_mat(param_set.GBOM_root+'_vibronic.log',param_set.num_modes)
                                        param_set.dipole_mom=gaussian_params.extract_transition_dipole(param_set.GBOM_root+'_ex.log',param_set.target_excited_state)
                                        param_set.E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+'_vibronic.log')

                                                                # if requested, remove low frequency vibrational modes:
                                if param_set.freq_cutoff_gbom>0.0:
                                    for i in range(freqs_gs.shape[0]):
                                        if freqs_gs[i]<param_set.freq_cutoff_gbom:
                                            freqs_ex[i]=freqs_gs[i]
                                            J[i,:]=0.0
                                            J[:,i]=0.0
                                            J[i,i]=1.0
                                            K[i]=0.0 


                                GBOM=gbom.gbom(freqs_gs,freqs_ex,J,K,param_set.E_adiabatic,param_set.dipole_mom,param_set.stdout)

                                if param_set.herzberg_teller:
                                    GBOM.dipole_deriv=np.genfromtxt(param_set.GBOM_root+'_dipole_deriv.dat')
                                    # First work out units
                                    GBOM.dipole_deriv=GBOM.dipole_deriv*const.gaussian_to_debye*const.debye_in_au/const.ang_to_bohr
                                    # now units should be in a.u/(amu)*0.5
                                    # convert from AMU to hartree...
                                    GBOM.dipole_deriv=GBOM.dipole_deriv*np.sqrt(const.emass_in_au)
                                    print(GBOM.dipole_deriv)
                                    # HACK! TRY AND SCALE MASSES. DONT THINK THAT's correct? 
                                    reduced_masses=np.genfromtxt(param_set.GBOM_root+'_reduced_masses.dat')
                                    for i in range(GBOM.dipole_deriv.shape[0]):
                                        # HACK: Try and scale by excited state freq squared?
                                        GBOM.dipole_deriv[i,:]=GBOM.dipole_deriv[i,:]*freqs_ex[i]**2.0  # scale by square of ex freq
                                        #GBOM.dipole_deriv[i,:]=GBOM.dipole_deriv[i,:]*np.sqrt(reduced_masses[i]*const.emass_in_au)

                        elif param_set.GBOM_input_code=='TERACHEM':
                                if param_set.is_vertical_gradient:  # CURRENTLY SINGLE GBOM AND NO HT EFFECTS
                                    param_set.stdout.write('Attempting to compute GBOM model from the following TeraChem output files: '+'\n')
                                    param_set.stdout.write(param_set.GBOM_root+'_gs.log'+'\t'+param_set.GBOM_root+'_grad.log'+'\n')
                                else:
                                    # first obtain coordinates and Hessian. Check if we have frozen atoms.
                                    # sanity check:
                                    param_set.stdout.write('Attempting to compute GBOM model from the following TeraChem output files: '+'\n')
                                    param_set.stdout.write(param_set.GBOM_root+'_gs.log'+'\t'+param_set.GBOM_root+'_ex.log'+'\n')
                                    if param_set.num_atoms<1:
                                        param_set.stdout.write('Error: Trying to read from Terachem input but number of atoms is not set!'+'\n')
                                        sys.exit('Error: Trying to read from Terachem input but number of atoms is not set!') 
                                    frozen_atom_list=np.zeros(param_set.num_atoms)
                                    if param_set.num_frozen_atoms>0: 
                                        if os.path.exists(param_set.frozen_atom_path):
                                            frozen_atom_list=np.genfromtxt(param_set.frozen_atom_path)
                                        else:
                                            # assume the frozen atoms are all at the end of file
                                            for i in range(frozen_atom_list.shape[0]):
                                                if i<param_set.num_atoms-param_set.num_frozen_atoms:
                                                    frozen_atom_list[i]=0  # unfrozen
                                                else:
                                                    frozen_atom_list[i]=1 # frozen                                              

                                    #param_set.stdout.write('Error: Trying to perform Terachem calculation with frozen atoms but frozen atom list does not exist!')
                                    #sys.exit('Error: Trying to perform Terachem calculation with frozen atoms but frozen atom list does not exist!')
                                
                                    # now obtain Hessians and other params.
                                    masses,gs_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+'_gs.log', param_set.num_atoms)        
                                    gs_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+'_gs.log',frozen_atom_list,param_set.num_frozen_atoms)
                                    masses,ex_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+'_ex.log', param_set.num_atoms)
                                    ex_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+'_ex.log',frozen_atom_list,param_set.num_frozen_atoms)
                                    dipole_mom,E_adiabatic=terachem_params.get_e_adiabatic_dipole(param_set.GBOM_root+'_gs.log',param_set.GBOM_root+'_ex.log',param_set.target_excited_state)

                                    dipole_deriv_cart=terachem_params.get_dipole_deriv_from_terachem(param_set.GBOM_root+'_ex.log',frozen_atom_list,param_set.num_frozen_atoms,param_set.target_excited_state)

                                    # now construct frequencies, J and K from these params. 
                                    # also construct new transition dipole moment in Eckart frame and 
                                    # the dipole derivative with respect to mass weighted normal modes
                                    freqs_gs,freqs_ex,J,K,dipole_transformed,dipole_deriv_nm=hess_to_gbom.construct_freqs_J_K(gs_geom,ex_geom,gs_hessian,ex_hessian,dipole_mom,dipole_deriv_cart,masses,param_set.num_frozen_atoms,frozen_atom_list)

                                    # Check if we are artificially switching off the Duschinsky rotation
                                    if param_set.no_dusch:
                                        J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
                                        counter=0
                                        while counter<freqs_ex.shape[0]:
                                                J[counter,counter]=1.0
                                                counter=counter+1
                                    #SCALE JMAT
                                    # if requested, scale duschinsky rotation off diagonal elements to increase coupling
                                    if param_set.scale_Jmat:
                                        J=hess_to_gbom.scale_J_mixing(J,freqs_gs,param_set.freq_cutoff_gbom,param_set.Jmat_scaling_fac)

                                    # if requested, remove low frequency vibrational modes (only happens if we do not scale the J matrix):
                                    if param_set.freq_cutoff_gbom>0.0 and not param_set.scale_Jmat:
                                        for i in range(freqs_gs.shape[0]):
                                            if freqs_gs[i]<param_set.freq_cutoff_gbom:
                                                freqs_ex[i]=freqs_gs[i]
                                                J[i,:]=0.0      
                                                J[:,i]=0.0
                                                J[i,i]=1.0
                                                K[i]=0.0 


                                    # GBOM assumes E_0_0 as input rather than E_adiabatic. 
                                    E_0_0=(E_adiabatic+0.5*(np.sum(freqs_ex)-np.sum(freqs_gs)))

                                    # construct GBOM 
                                    # HACK!!!!! SET GS AND EX FREQS EQUAL. REMOVE!!! SECOND FREQ SHOULD BE FREQ EX
                                    GBOM=gbom.gbom(freqs_gs,freqs_ex,J,K,E_0_0,dipole_transformed,param_set.stdout)
                                    if param_set.herzberg_teller:
                                        GBOM.dipole_deriv=dipole_deriv_nm

                        # unsupported input code
                        else:
                                sys.exit('Error: Unrecognized code from which to read the GBOM parameters. Only GAUSSIAN and TERACHEM supported!')

                elif param_set.Jpath!='' and param_set.Kpath!='' and param_set.freq_gs_path!='' and param_set.freq_ex_path!='':
                        if np.dot(param_set.dipole_mom,param_set.dipole_mom)<0.000000000001 and param_set.E_adiabatic==0.0000000000001:
                                sys.exit('Error: Did not provide dipole moment or adiabatic energy gap for GBOM!')
                        # create GBOM from input J, K and freqs
                        else:
                                freqs_gs=np.genfromtxt(param_set.freq_gs_path)
                                J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
                                if param_set.no_dusch:
                                        counter=0
                                        while counter<freqs_gs.shape[0]:
                                                J[counter,counter]=1.0
                                                counter=counter+1
                                else:
                                        J=np.genfromtxt(param_set.Jpath)
                                K=np.genfromtxt(param_set.Kpath)
                                freqs_ex=np.genfromtxt(param_set.freq_ex_path)
                                # created appropriate matrices: now create GBOM.


                                                                # if requested, remove low frequency vibrational modes:
                                if param_set.freq_cutoff_gbom>0.0:
                                    for i in range(freqs_gs.shape[0]):
                                        if freqs_gs[i]<param_set.freq_cutoff_gbom:
                                            freqs_ex[i]=freqs_gs[i]
                                            J[i,:]=0.0      
                                            J[:,i]=0.0
                                            J[i,i]=1.0
                                            K[i]=0.0 

                                # GBOM assumes E_0_0 as input rather than E_adiabatic. 
                                E_0_0=param_set.E_adiabatic+0.5*(np.sum(freqs_ex)-np.sum(freqs_gs))
                                GBOM=gbom.gbom(freqs_gs,freqs_ex,J,K,E_0_0,param_set.dipole_mom,param_set.stdout)
                else:
                        sys.exit('Error: GBOM calculation requested but no path to model system parameters given!')

        # instead create a GBOM batch
        else:
                batch_count=1
                freqs_gs_batch=np.zeros((param_set.num_gboms,param_set.num_modes))
                freqs_ex_batch=np.zeros((param_set.num_gboms,param_set.num_modes))
                Jbatch=np.zeros((param_set.num_gboms,param_set.num_modes,param_set.num_modes))
                Kbatch=np.zeros((param_set.num_gboms,param_set.num_modes))
                E_batch=np.zeros((param_set.num_gboms))
                dipole_batch=np.zeros((param_set.num_gboms,3))
                dipole_deriv_batch=np.zeros((param_set.num_gboms,param_set.num_modes,3))

                while batch_count<param_set.num_gboms+1:
                        if param_set.GBOM_root!='':
                                if param_set.GBOM_input_code=='GAUSSIAN':
                                        print(param_set.GBOM_root+str(batch_count)+'_gs.log',param_set.num_modes)
                                        freqs_gs=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+str(batch_count)+'_gs.log',param_set.num_modes)
                                        freqs_ex=gaussian_params.extract_normal_mode_freqs(param_set.GBOM_root+str(batch_count)+'_ex.log',param_set.num_modes)
                                        K=gaussian_params.extract_Kmat(param_set.GBOM_root+str(batch_count)+'_vibronic.log',param_set.num_modes)
                                        J=gaussian_params.extract_duschinsky_mat(param_set.GBOM_root+str(batch_count)+'_vibronic.log',param_set.num_modes)
                                        E_adiabatic=gaussian_params.extract_adiabatic_freq(param_set.GBOM_root+str(batch_count)+'_vibronic.log')
                                        param_set.dipole_mom=gaussian_params.extract_transition_dipole(param_set.GBOM_root+str(batch_count)+'_ex.log',param_set.target_excited_state)
                                        # are we switching off Duschinsky rotation?
                                        if param_set.no_dusch:
                                                J=np.zeros((K.shape[0],K.shape[0]))
                                                counter=0
                                                while counter<freqs_gs.shape[0]:
                                                        J[counter,counter]=1.0
                                                        counter=counter+1


                                                                        # if requested, remove low frequency vibrational modes:
                                        if param_set.freq_cutoff_gbom>0.0:
                                            for i in range(freqs_gs.shape[0]):
                                                if freqs_gs[i]<param_set.freq_cutoff_gbom:
                                                    freqs_ex[i]=freqs_gs[i]
                                                    J[i,:]=0.0      
                                                    J[:,i]=0.0
                                                    J[i,i]=1.0
                                                    K[i]=0.0 

                                        # fill batch
                                        freqs_gs_batch[batch_count-1,:]=freqs_gs
                                        freqs_ex_batch[batch_count-1,:]=freqs_ex
                                        Jbatch[batch_count-1,:,:]=J
                                        Kbatch[batch_count-1,:]=K



                                        E_batch[batch_count-1]=E_adiabatic
                                        dipole_batch[batch_count-1,:]=param_set.dipole_mom
                                elif param_set.GBOM_input_code=='TERACHEM':
                                        atoms_snapshot=0
                                        frozen_atoms_snapshot=0
                                        print('FROZEN ATOMS')
                                        print(param_set.num_frozen_atoms)
                                        if param_set.num_frozen_atoms[batch_count-1]>0: 
                                                if os.path.exists(param_set.frozen_atom_path+str(batch_count)):
                                                        frozen_atom_list=np.genfromtxt(param_set.frozen_atom_path+str(batch_count))
                                                else:
                                                        sys.exit('Error: Trying to perform Terachem calculation with frozen atoms but frozen atom list does not exist for current batch!')
                                                atoms_snapshot=frozen_atom_list.shape[0]
                                                frozen_atoms_snapshot=int(np.sum(frozen_atom_list))
                                        # NOT a frozen atom calculation
                                        else:
                                                # sanity check
                                                if param_set.num_atoms<1:
                                                        sys.exit('Error: Trying to read a batch of Terachem input files with no frozen atoms and NUM_ATOMS is not set!')
                                                atoms_snapshot=param_set.num_atoms

                                        # now obtain Hessians and other params.
                                        masses,gs_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+str(batch_count)+'_gs.log', atoms_snapshot)
                                        gs_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+str(batch_count)+'_gs.log',frozen_atom_list,frozen_atoms_snapshot)
                                        masses,ex_geom=terachem_params.get_masses_geom_from_terachem(param_set.GBOM_root+str(batch_count)+'_ex.log', atoms_snapshot)
                                        ex_hessian=terachem_params.get_hessian_from_terachem(param_set.GBOM_root+str(batch_count)+'_ex.log',frozen_atom_list,frozen_atoms_snapshot)
                                        dipole_mom,E_adiabatic=terachem_params.get_e_adiabatic_dipole(param_set.GBOM_root+str(batch_count)+'_gs.log',param_set.GBOM_root+str(batch_count)+'_ex.log',param_set.target_excited_state)
                                        dipole_deriv_cart=terachem_params.get_dipole_deriv_from_terachem(param_set.GBOM_root+str(batch_count)+'_ex.log',frozen_atom_list,frozen_atoms_snapshot,param_set.target_excited_state)

                                        # now construct frequencies, J and K from these params. 
                                        freqs_gs,freqs_ex,J,K,dipole_transformed,dipole_deriv_nm=hess_to_gbom.construct_freqs_J_K(gs_geom,ex_geom,gs_hessian,ex_hessian,dipole_mom,dipole_deriv_cart,masses,frozen_atoms_snapshot,frozen_atom_list)

                                        # Check if we are artificially switching off the Duschinsky rotation
                                        if param_set.no_dusch:
                                                J=np.zeros((freqs_gs.shape[0],freqs_gs.shape[0]))
                                                counter=0
                                                while counter<freqs_ex.shape[0]:
                                                        J[counter,counter]=1.0
                                                        counter=counter+1


                                                                        # if requested, remove low frequency vibrational modes:
                                        if param_set.freq_cutoff_gbom>0.0:
                                            for i in range(freqs_gs.shape[0]):
                                                if freqs_gs[i]<param_set.freq_cutoff_gbom:
                                                    print('Remove_freq:  ', str(i))
                                                    freqs_ex[i]=freqs_gs[i]
                                                    J[i,:]=0.0      
                                                    J[:,i]=0.0
                                                    J[i,i]=1.0
                                                    K[i]=0.0 

                                        # GBOM assumes E_0_0 as input rather than E_adiabatic. 
                                        E_0_0=(E_adiabatic+0.5*(np.sum(freqs_ex)-np.sum(freqs_gs)))

                                        # fill batch
                                        freqs_gs_batch[batch_count-1,:]=freqs_gs
                                        freqs_ex_batch[batch_count-1,:]=freqs_ex
                                        Jbatch[batch_count-1,:,:]=J
                                        Kbatch[batch_count-1,:]=K
                                        E_batch[batch_count-1]=E_0_0
                                        dipole_batch[batch_count-1,:]=dipole_transformed
                                        dipole_deriv_batch[batch_count-1,:,:]=dipole_deriv_nm
                                else:
                                        sys.exit('Error: Currently only support GBOM_INPUT_CODE=GAUSSIAN or TERACHEM!')

                        elif param_set.Jpath!='' and param_set.Kpath!='' and param_set.freq_gs_path!='' and param_set.freq_ex_path!='':
                                if param_set.dipole_mom_path=='' and param_set.E_adiabatic_path=='' and not param_set.method=='EOPT_AV':
                                        sys.exit('Error: Did not provide dipole moment list or adiabatic energy gap list for GBOM batch!')
                                # create GBOM from input J, K and freqs
                                else:
                                        J=np.genfromtxt(param_set.Jpath+str(batch_count)+'.dat')
                                        K=np.genfromtxt(param_set.Kpath+str(batch_count)+'.dat')
                                        freqs_gs=np.genfromtxt(param_set.freq_gs_path+str(batch_count)+'.dat')
                                        freqs_ex=np.genfromtxt(param_set.freq_ex_path+str(batch_count)+'.dat')
                                        if not param_set.method=='EOPT_AV':
                                                dipole_list=np.genfromtxt(param_set.dipole_mom_path)
                                                E_adiabatic_list=np.genfromtxt(param_set.E_adiabatic_path)
        
                                        # fill batch
                                        freqs_gs_batch[batch_count-1,:]=freqs_gs
                                        freqs_ex_batch[batch_count-1,:]=freqs_ex
                                        Jbatch[batch_count-1,:,:]=J
                                        Kbatch[batch_count-1,:]=K
                                        if not param_set.method=='EOPT_AV':
                                                E_batch[batch_count-1]=E_adiabatic_list[batch_count-1]
                                                dipole_batch[batch_count-1]=dipole_list[batch_count-1]
                                        else:   # if this is an EOPT_AV calculation we get the adiabatic energy gap from somewhere else
                                                E_batch[batch_count-1]=0.0
                                                dipole_batch[batch_count-1]=1.0
                                
                        else:
                                sys.exit('Error: GBOM calculation requested but no path to model system parameters given!')


                        batch_count=batch_count+1
                # now construct the model
                GBOM_batch=gbom.gbom_list(freqs_gs_batch,freqs_ex_batch,Jbatch,Kbatch,E_batch,dipole_batch,param_set.num_gboms,param_set.stdout)
                

        param_set.stdout.write('Successfully set up a GBOM model!'+'\n')

# Morse oscillator model
elif param_set.model=='MORSE':
                if param_set.morse_gs_path!='' and param_set.morse_ex_path!='':
                                                if np.dot(param_set.dipole_mom,param_set.dipole_mom)==0.0 and param_set.E_adiabatic==0.0:
                                                                sys.exit('Error: Did not provide dipole moment or adiabatic energy gap for Morse oscillator!')
                                                # create effective morse oscillator from input values
                                                else:
                                                        gs_params=np.genfromtxt(param_set.morse_gs_path)
                                                        ex_params=np.genfromtxt(param_set.morse_ex_path)

                                                        D_gs=gs_params[:,0]
                                                        alpha_gs=gs_params[:,1]
                                                        D_ex=ex_params[:,0]
                                                        alpha_ex=ex_params[:,1]
                                                        mu=gs_params[:,2]
                                                        shift=ex_params[:,2]

                                                        # check if HT params are given
                                                        if gs_params.shape[1]>3:
                                                            lambda_param=gs_params[:,3]
                                                            HT_overlaps=True
                                                        else:
                                                            lambda_param=np.zeros(1)
                                                            HT_overlaps=False
                
                                                        # sanity check
                                                        num_morse=gs_params.shape[0]
                                                        if gs_params.shape[0] != ex_params.shape[0]:
                                                                 sys.exit('Error: Inconsistent number of parameters in ground and excited state morse oscillator files!')
                                                        morse_oscs=morse.morse_list(D_gs,D_ex,alpha_gs,alpha_ex,mu,shift,param_set.E_adiabatic,param_set.dipole_mom,param_set.max_states_morse_gs,param_set.max_states_morse_ex,param_set.integration_points_morse,lambda_param,HT_overlaps,param_set.gs_reference_dipole,num_morse,param_set.stdout)
                else:
                                                sys.exit('Error: Did not provide ground and excited state Morse oscillator parameters!')

# Morse oscillator model
elif param_set.model=='COUPLED_MORSE':
                if param_set.morse_gs_path!='' and param_set.morse_ex_path!='' and param_set.Jpath!='':
                                                if np.dot(param_set.dipole_mom,param_set.dipole_mom)==0.0 and param_set.E_adiabatic==0.0:
                                                                sys.exit('Error: Did not provide dipole moment or adiabatic energy gap for Morse oscillator!')
                                                # create effective morse oscillator from input values
                                                else:
                                                        gs_params=np.genfromtxt(param_set.morse_gs_path)
                                                        ex_params=np.genfromtxt(param_set.morse_ex_path)

                                                        D_gs=gs_params[:,0]
                                                        alpha_gs=gs_params[:,1]
                                                        D_ex=ex_params[:,0]
                                                        alpha_ex=ex_params[:,1]
                                                        mu=gs_params[:,2]
                                                        shift=ex_params[:,2]
                                                        J=np.genfromtxt(param_set.Jpath)
                                                        # sanity check
                                                        num_morse=gs_params.shape[0]
                                                        if gs_params.shape[0] != ex_params.shape[0]:
                                                                 sys.exit('Error: Inconsistent number of parameters in ground and excited state morse oscillator files!')
                                                        coupled_morse=morse.morse_coupled(D_gs,D_ex,alpha_gs,alpha_ex,mu,J,shift,param_set.E_adiabatic,param_set.dipole_mom,param_set.max_states_morse_gs,param_set.max_states_morse_ex,param_set.integration_points_morse,num_morse,param_set.stdout)
                else:
                                                sys.exit('Error: Did not provide ground and excited state Morse oscillator parameters!')




# pure MD model
elif param_set.model=='MD':
    if param_set.MD_input_code=='TERACHEM':   # Read directly from a TeraChem input file 
        # currently only works for a single traj. 
        traj_count=1
        if os.path.exists(param_set.MD_root+'traj'+str(traj_count)+'.out') and os.path.exists(param_set.MD_root+'traj'+str(traj_count)+'.xyz') and os.path.exists(param_set.MD_root+'ref.out') and os.path.exists(param_set.MD_root+'ref.xyz'):  # make sure the out
            param_set.stdout.write('Reading in MD trajectory '+str(traj_count)+'  from TeraChem output file '+param_set.MD_root+'traj'+str(traj_count)+'.dat'+'\n')

            traj_dipole=terachem_params_MD.get_full_energy_dipole_moms_from_MD(param_set.MD_root+'traj'+str(traj_count)+'.out',param_set.MD_root+'traj'+str(traj_count)+'.xyz',param_set.MD_root+'ref.out',param_set.MD_root+'ref.xyz',param_set.num_atoms,param_set.target_excited_state,param_set.md_num_frames,param_set.md_skip_frames)

            np.savetxt('full_traj_dipole_from_terachem.dat',traj_dipole)

            if traj_count==1:
                traj_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))
                dipole_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs,3))

            traj_batch[:,traj_count-1]=traj_dipole[:,0]
            dipole_batch[:,traj_count-1,0]=traj_dipole[:,2]
            dipole_batch[:,traj_count-1,1]=traj_dipole[:,3]
            dipole_batch[:,traj_count-1,2]=traj_dipole[:,4]

        else:
            param_set.stdout.write('Error: Trajectory and coordinate file names necessary for MD-based model does not exist!')
            sys.exit('Error: Trajectory and coordinate file names necessary for MD-based model does not exist!')

        MDtraj=md_traj.MDtrajs(traj_batch,dipole_batch,param_set.decay_length,param_set.num_trajs,param_set.md_step,param_set.stdout)   
        param_set.stdout.write('Successfully read in MD trajectory data of energy gap fluctuations!'+'\n')


    else:
        traj_count=1
        while traj_count<param_set.num_trajs+1:
                # if this is not a HT calculation, we expect the data to be in two colums: Energy and oscillator strength
                # we then use the oscillator strength to compute an effective transition dipole moment (all the strength being in
                # the arbitrary x direction). 
                # if this is a HT calculation, we expect 5 columsn. Energy, oscillator strenght and the actual transition dipole moment vector
                if os.path.exists(param_set.MD_root+'traj'+str(traj_count)+'.dat'):
                        param_set.stdout.write('Reading in MD trajectory '+str(traj_count)+'  from file '+param_set.MD_root+'traj'+str(traj_count)+'.dat'+'\n')
                        traj_dipole=np.genfromtxt(param_set.MD_root+'traj'+str(traj_count)+'.dat')

                        # Sanity check. 
                        if param_set.herzberg_teller and traj_dipole.shape[1] != 5:
                            sys.exit('Error: Requested HT calculation but input trajectory data only has '+str(traj_dipole.shape[1])+' columns!')

                        if traj_count==1:
                                traj_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))
                                dipole_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs,3))
                        if traj_dipole.shape[1]==2:   # check if we have 2 or 5 parameters in input file
                        #if not param_set.herzberg_teller:  # THIS IS WRONG! NEED TO CHECK DIMENSIONS OF INCOMING DATA FILE. 
                            traj_batch[:,traj_count-1]=traj_dipole[:,0]
                            # compute dipole moment from oscillator strength for each snapshot. Since direction of the 
                            # dipole moment does not matter for non-HT calculations, simply let the dipole moment point pure
                            # ly in the X direction.
                            dipole_batch[:,traj_count-1,0]=np.sqrt(3.0*traj_dipole[:,1]/(2.0*traj_dipole[:,0]/const.Ha_to_eV))
                
                        else:
                            # This is a HT calculation. read in full data from file. 
                            traj_batch[:,traj_count-1]=traj_dipole[:,0]
                            dipole_batch[:,traj_count-1,0]=traj_dipole[:,2]
                            dipole_batch[:,traj_count-1,1]=traj_dipole[:,3]
                            dipole_batch[:,traj_count-1,2]=traj_dipole[:,4]

                else:
                        param_set.stdout.write('Error: Trajectory file name necessary for MD-based model does not exist!')
                        sys.exit('Error: Trajectory file name necessary for MD-based model does not exist!')

                traj_count=traj_count+1
        MDtraj=md_traj.MDtrajs(traj_batch,dipole_batch,param_set.decay_length,param_set.num_trajs,param_set.md_step,param_set.stdout)       
        param_set.stdout.write('Successfully read in MD trajectory data of energy gap fluctuations!'+'\n')      

else:
                sys.exit('Error: Invalid model '+param_set.model)

# both MD and GBOM input ---> E-ZTFC and related approaches 
# need to also construct the MDtraj model. Have already constructed the GBOM type model
if param_set.model=='MD_GBOM':
        # start by setting up MDtraj
                traj_count=1
                while traj_count<param_set.num_trajs+1:
                                if os.path.exists(param_set.MD_root+'traj'+str(traj_count)+'.dat'):
                                                param_set.stdout.write('Reading in MD trajectory '+str(traj_count)+'  from file '+param_set.MD_root+'traj'+str(traj_count)+'.dat'+'\n')
                                                traj_dipole=np.genfromtxt(param_set.MD_root+'traj'+str(traj_count)+'.dat')

                                                # Sanity check. 
                                                if param_set.herzberg_teller and traj_dipole.shape[1] != 5:
                                                        sys.exit('Error: Requested HT calculation but input trajectory data only has '+str(traj_dipole.shape[1])+' columns!')

                                                if traj_count==1:
                                                                traj_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs))
                                                                dipole_batch=np.zeros((traj_dipole.shape[0],param_set.num_trajs,3))

                                                if not param_set.herzberg_teller:
                                                        traj_batch[:,traj_count-1]=traj_dipole[:,0]
                                                        # compute dipole moment from oscillator strength for each snapshot. Since direction of the 
                                                        # dipole moment does not matter for non-HT calculations, simply let the dipole moment point pure
                                                        # ly in the X direction.
                                                        dipole_batch[:,traj_count-1,0]=np.sqrt(3.0*traj_dipole[:,1]/(2.0*traj_dipole[:,0]/const.Ha_to_eV))

                                                else:
                                                        # This is a HT calculation. read in full data from file. 
                                                        traj_batch[:,traj_count-1]=traj_dipole[:,0]
                                                        dipole_batch[:,traj_count-1,0]=traj_dipole[:,2]
                                                        dipole_batch[:,traj_count-1,1]=traj_dipole[:,3]
                                                        dipole_batch[:,traj_count-1,2]=traj_dipole[:,4]


                                else:
                                                param_set.stdout.write('Error: Trajectory file name necessary for MD-based model does not exist!')
                                                sys.exit('Error: Trajectory file name necessary for MD-based model does not exist!')

                                traj_count=traj_count+1
                MDtraj=md_traj.MDtrajs(traj_batch,dipole_batch,param_set.decay_length,param_set.num_trajs,param_set.md_step,param_set.stdout)
                param_set.stdout.write('Successfully read in MD trajectory data of energy gap fluctuations!'+'\n')

# first check whether this is an absorption or a 2DES calculation
if param_set.task=='ABSORPTION':
        param_set.stdout.write('\n'+'Setting up linear absorption spectrum calculation:'+'\n')
        if param_set.model=='GBOM':
                if param_set.num_gboms==1:
                        if param_set.is_solvent:        
                                compute_GBOM_absorption(param_set,GBOM,solvent_mod, False)
                        else:
                                sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')
                else:
                        if param_set.is_solvent:
                                compute_GBOM_batch_absorption(param_set,GBOM_batch,solvent_mod, False)
                        else:
                                sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')

        elif param_set.model=='MORSE':
                compute_morse_absorption(param_set,morse_oscs,solvent_mod,False)

        elif param_set.model=='COUPLED_MORSE':
                compute_coupled_morse_absorption(param_set,coupled_morse,solvent_mod,False)

        elif param_set.model=='MD':
                if param_set.is_solvent:
                        compute_MD_absorption(param_set,MDtraj,solvent_mod,False)
                else:
                        # set solvent model to dummy variable

                        compute_MD_absorption(param_set,MDtraj,0.0,False)
        elif param_set.model=='MD_GBOM':
                if param_set.num_gboms==1:
                        if param_set.is_solvent:
                            compute_hybrid_GBOM_MD_absorption(param_set,MDtraj,GBOM,solvent_mod,False)
                        else:
                            compute_hybrid_GBOM_MD_absorption(param_set,MDtraj,GBOM,0.0,False)
                else:
                        if param_set.is_solvent:
                            compute_hybrid_GBOM_batch_MD_absorption(param_set,MDtraj,GBOM_batch,solvent_mod,False)  
                        else:
                            compute_hybrid_GBOM_batch_MD_absorption(param_set,MDtraj,GBOM_batch,0.0,False)
        else:
                sys.exit('Error: Only pure GBOM model or pure MD model or MD_GBOM model implemented so far.')

elif param_set.task=='EMISSION':
        param_set.stdout.write('\n'+'Setting up linear emission spectrum calculation:'+'\n')
        if param_set.model=='GBOM':
                if param_set.is_solvent:
                        compute_GBOM_absorption(param_set,GBOM,solvent_mod,True)
                else:
                        sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')
        elif param_set.model=='MD':
                if param_set.is_solvent:
                        compute_MD_absorption(param_set,MDtraj,solvent_mod,True)
                else:
                        # set solvent model to dummy variable
                        compute_MD_absorption(param_set,MDtraj,0.0,True)
        else:
                sys.exit('Error: Only pure GBOM model or pure MD model implemented so far.')

elif param_set.task=='2DES':
        param_set.stdout.write('\n'+'Setting up 2DES calculation:'+'\n')
        if param_set.model=='GBOM' and param_set.num_gboms==1:
                if param_set.method == 'FC':
                        #RUNS OFF FC2DES FORMULA DERIVED BY LUKE ALLAN (THAT'S MEEEE :])
                        #RIGHT NOW IT HAS CPU CALCULATION CAPABLITY FOR (J!=I) and (J==I) FORMULAS
                        #WORKFLOW: BUILD g_2 SOLVENT -> BUILD SOLVENT RESPONSE FUNC -> FIND DECAY POINT FC2DES CALCULATION CAN BE TRUNCATED AT
                        #->RUN FC2DES CALCULATION -> FT->PRINT SPECTRA (2DES and TRANSIENT)
                        print("FC2DES CALCULATION")
                        if not param_set.is_solvent:
                                sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')
                        solvent_mod.calc_spectral_dens(param_set.num_steps)
                        solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                        FC2DES.Calc_2DES_time_series(solvent_mod.g2_solvent, param_set.temperature, GBOM,param_set.E_adiabatic, param_set.num_steps_2DES, param_set.spectral_window, param_set.num_time_samples_2DES, param_set.t_step_2DES,param_set.FC2DES_device)
                if param_set.method == 'CUMULANT':
                        if not param_set.is_solvent:
                                sys.exit('Error: Pure GBOM calculations require some form of additional solvent broadening provided by a solvent model')

                        solvent_mod.calc_spectral_dens(param_set.num_steps)
                        solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                        filename_2DES=param_set.GBOM_root+''
                        if param_set.exact_corr:
                                GBOM.calc_omega_av_qm(param_set.temperature,False)
                                GBOM.calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)
                                # set the start and end values for both the x and the y axis of the
                                # 2DES spectrum
                                eff_shift1=0.0
                                eff_shift2=0.0
                                if abs(param_set.omega1)>0.000001:
                                        eff_shift1=param_set.omega1-GBOM.omega_av_qm
                                if abs(param_set.omega3)>0.000001:
                                        eff_shift2=param_set.omega3-GBOM.omega_av_qm

                                E_start1=GBOM.omega_av_qm-param_set.spectral_window/2.0+eff_shift1
                                E_end1=GBOM.omega_av_qm+param_set.spectral_window/2.0+eff_shift1
                                E_start2=GBOM.omega_av_qm-param_set.spectral_window/2.0+eff_shift2
                                E_end2=GBOM.omega_av_qm+param_set.spectral_window/2.0+eff_shift2

                                q_func_eff=GBOM.g2_exact
                                q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]


                                # if it is a 3rd order cumulant calculation, compute g3 and auxilliary functions h1 and h2
                                if param_set.third_order:
                                        GBOM.calc_g3_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.four_phonon_term,param_set.g3_cutoff,param_set.stdout)

                                        if os.path.exists('h1_real.dat') and os.path.exists('h1_imag.dat') and os.path.exists('h2_real.dat') and os.path.exists('h2_imag.dat') and os.path.exists('h4_real.dat') and os.path.exists('h4_imag.dat') and os.path.exists('h5_real.dat') and os.path.exists('h5_imag.dat'):
                                                # read in all files:
                                                GBOM.h1_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
                                                GBOM.h1_exact=GBOM.h1_exact+twoDES.read_2D_spectrum('h1_real.dat',param_set.num_steps)
                                                temp_imag=1j*twoDES.read_2D_spectrum('h1_imag.dat',param_set.num_steps)
                                                GBOM.h1_exact[:,:,2]=GBOM.h1_exact[:,:,2]+temp_imag[:,:,2]

                                                GBOM.h2_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
                                                GBOM.h2_exact=GBOM.h2_exact+twoDES.read_2D_spectrum('h2_real.dat',param_set.num_steps)
                                                temp_imag=1j*twoDES.read_2D_spectrum('h2_imag.dat',param_set.num_steps)
                                                GBOM.h2_exact[:,:,2]=GBOM.h2_exact[:,:,2]+temp_imag[:,:,2]

                                                GBOM.h4_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
                                                GBOM.h4_exact=GBOM.h4_exact+twoDES.read_2D_spectrum('h4_real.dat',param_set.num_steps)
                                                temp_imag=1j*twoDES.read_2D_spectrum('h4_imag.dat',param_set.num_steps)
                                                GBOM.h4_exact[:,:,2]=GBOM.h4_exact[:,:,2]+temp_imag[:,:,2]

                                                GBOM.h5_exact=np.zeros((param_set.num_steps,param_set.num_steps,3),dtype=complex)
                                                GBOM.h5_exact=GBOM.h5_exact+twoDES.read_2D_spectrum('h5_real.dat',param_set.num_steps)
                                                temp_imag=1j*twoDES.read_2D_spectrum('h5_imag.dat',param_set.num_steps)
                                                GBOM.h5_exact[:,:,2]=GBOM.h5_exact[:,:,2]+temp_imag[:,:,2]

                                        else:
                                                GBOM.calc_h1_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h1_real.dat',GBOM.h1_exact,False)
                                                twoDES.print_2D_spectrum('h1_imag.dat',GBOM.h1_exact,True)
                                                GBOM.calc_h2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h2_real.dat',GBOM.h2_exact,False)
                                                twoDES.print_2D_spectrum('h2_imag.dat',GBOM.h2_exact,True)
                                                GBOM.calc_h4_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h4_real.dat',GBOM.h4_exact,False)
                                                twoDES.print_2D_spectrum('h4_imag.dat',GBOM.h4_exact,True)
                                                GBOM.calc_h5_qm(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h5_real.dat',GBOM.h5_exact,False)
                                                twoDES.print_2D_spectrum('h5_imag.dat',GBOM.h5_exact,True)


                                if param_set.method_2DES=='2DES':
                                        if param_set.third_order:
                                                twoDES.calc_2DES_time_series_GBOM_3rd(q_func_eff,GBOM.dipole_mom,GBOM.g3_exact,GBOM.h1_exact,GBOM.h2_exact,GBOM.h4_exact,GBOM.h5_exact,GBOM.corr_func_3rd_qm,GBOM.freqs_gs,GBOM.Omega_sq,GBOM.gamma,param_set.temperature*const.kb_in_Ha,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0,False,param_set.no_dusch,param_set.four_phonon_term)
                                        else:
                                                twoDES.calc_2DES_time_series(q_func_eff,GBOM.dipole_mom,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)
                                elif param_set.method_2DES=='PUMP_PROBE':
                                        twoDES.calc_pump_probe_time_series(q_func_eff,GBOM.dipole_mom,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)
                        else:
                                GBOM.calc_omega_av_cl(param_set.temperature,False)
                                GBOM.calc_g2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)
                                # set the start and end values for both the x and the y axis of the
                                # 2DES spectrum
                                eff_shift1=0.0
                                eff_shift2=0.0
                                if abs(param_set.omega1)>0.000001:
                                        eff_shift1=param_set.omega1-GBOM.omega_av_cl
                                if abs(param_set.omega3)>0.000001:
                                        eff_shift2=param_set.omega3-GBOM.omega_av_cl

                                E_start1=GBOM.omega_av_cl-param_set.spectral_window/2.0+eff_shift1
                                E_end1=GBOM.omega_av_cl+param_set.spectral_window/2.0+eff_shift1
                                E_start2=GBOM.omega_av_cl-param_set.spectral_window/2.0+eff_shift2
                                E_end2=GBOM.omega_av_cl+param_set.spectral_window/2.0+eff_shift2
                                q_func_eff=GBOM.g2_cl
                                q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]

                                # if it is a 3rd order cumulant calculation, compute g3 and auxilliary functions h1 and h2
                                if param_set.third_order:
                                        GBOM.calc_g3_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.four_phonon_term,param_set.g3_cutoff,param_set.stdout)
                                        if os.path.exists('h1_real.dat') and os.path.exists('h1_imag.dat') and os.path.exists('h2_real.dat') and os.path.exists('h2_imag.dat') and os.path.exists('h4_real.dat') and os.path.exists('h4_imag.dat') and os.path.exists('h5_real.dat') and os.path.exists('h5_imag.dat'):
                                                # read in all files:
                                                GBOM.h1_cl=twoDES.read_2D_spectrum('h1_real.dat',param_set.num_steps)
                                                GBOM.h1_cl[:,:,2]=GBOM.h1_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h1_imag.dat',param_set.num_steps))[:,:,2]
                                                GBOM.h2_cl=twoDES.read_2D_spectrum('h2_real.dat',param_set.num_steps)
                                                GBOM.h2_cl[:,:,2]=GBOM.h2_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h2_imag.dat',param_set.num_steps))[:,:,2]

                                                GBOM.h4_cl=twoDES.read_2D_spectrum('h4_real.dat',param_set.num_steps)
                                                GBOM.h4_cl[:,:,2]=GBOM.h4_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h4_imag.dat',param_set.num_steps))[:,:,2]
                                                GBOM.h5_cl=twoDES.read_2D_spectrum('h5_real.dat',param_set.num_steps)
                                                GBOM.h5_cl[:,:,2]=GBOM.h5_cl[:,:,2]+1j*(twoDES.read_2D_spectrum('h5_imag.dat',param_set.num_steps))[:,:,2]

                                        else:
                                                # Calc h and save to file
                                                GBOM.calc_h1_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h1_real.dat',(GBOM.h1_cl),False)
                                                twoDES.print_2D_spectrum('h1_imag.dat',GBOM.h1_cl,True)
                                                GBOM.calc_h2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h2_real.dat',(GBOM.h2_cl),False)
                                                twoDES.print_2D_spectrum('h2_imag.dat',GBOM.h2_cl,True)
                                                GBOM.calc_h4_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h4_real.dat',(GBOM.h4_cl),False)
                                                twoDES.print_2D_spectrum('h4_imag.dat',GBOM.h4_cl,True)
                                                GBOM.calc_h5_cl(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.no_dusch,param_set.four_phonon_term)
                                                twoDES.print_2D_spectrum('h5_real.dat',(GBOM.h5_cl),False)
                                                twoDES.print_2D_spectrum('h5_imag.dat',GBOM.h5_cl,True,)
                
                                        # now construct 3rd order correlation function. Needed to speed up evaluation of h3
                                        #GBOM.compute_corr_func_3rd(param_set.temperature*const.kb_in_Ha,param_set.num_steps,param_set.max_t,False)
                
                                if param_set.method_2DES=='2DES':
                                        if param_set.third_order:
                                                twoDES.calc_2DES_time_series_GBOM_3rd(q_func_eff,GBOM.dipole_mom,GBOM.g3_cl,GBOM.h1_cl,GBOM.h2_cl,GBOM.h4_cl,GBOM.h5_cl,GBOM.corr_func_3rd_cl,GBOM.freqs_gs,GBOM.Omega_sq,GBOM.gamma,param_set.temperature*const.kb_in_Ha,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0,True,param_set.no_dusch,param_set.four_phonon_term)

                                        else:
                                                twoDES.calc_2DES_time_series(q_func_eff,GBOM.dipole_mom,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)       
                                elif param_set.method_2DES=='PUMP_PROBE':
                                        twoDES.calc_pump_probe_time_series(q_func_eff,GBOM.dipole_mom,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)

        # GBOM batch. Simplified implementation for the time being. Only 2nd order cumulant, and only standard 2DES
        elif param_set.model=='GBOM' and param_set.num_gboms!=1:

                if param_set.method=='EOPT_AV':     # this is not an E_FTFC calculation but rather an Eopt_avFTFC calculation
                        filename_2DES=param_set.GBOM_root+''
                        solvent_mod.calc_spectral_dens(param_set.num_steps)
                        solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                        solvent_mod.calc_solvent_response(False)
                        
                        # get list of adiabatic energies and dipole moms. 
                        energy_dipole=np.zeros((1,1))
                        if os.path.exists(param_set.E_opt_path):
                                energy_dipole=np.genfromtxt(param_set.E_opt_path)
                        else:
                                sys.exit('Error: Requested an Eopt_avFTFC type calculation but did not provide optimized vertical energy gaps and dipoles')


                        Eopt=energy_dipole[:,0]/const.Ha_to_eV
                        # compute average energy
                        Eopt_av=np.sum(Eopt)/(1.0*Eopt.shape[0]) 

                        icount=0
                        average_Egap=0.0
                        average_E00=0.0
                        
                        while icount<GBOM_batch.num_gboms:
                        # figure out start and end value for the spectrum.
                                average_E00=average_E00+GBOM_batch.gboms[icount].E_adiabatic+0.5*np.sum(GBOM_batch.gboms[icount].freqs_ex)-0.5*np.sum(GBOM_batch.gboms[icount].freqs_gs)
                                if param_set.exact_corr:
                                        # compute energy gap without 0-0 transition
                                        GBOM_batch.gboms[icount].calc_omega_av_qm(param_set.temperature,False)
                                        average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
                                        GBOM_batch.gboms[icount].omega_av_qm=0.0
                                else:
                                        GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
                                        average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl
                                        GBOM_batch.gboms[icount].omega_av_cl=0.0
                                icount=icount+1

                        # figure out average Egap and gap between Eopt_av and average_Egap for the GBOMS
                        average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)  # this is the place the spectrum should be centered on
                        delta_Eopt_Eav=average_Egap-Eopt_av
                        average_E00=average_E00/(1.0*GBOM_batch.num_gboms)
                        average_E_adiabatic=0.0
                        for i in range(len(GBOM_batch.gboms)):
                            average_E_adiabatic=average_E_adiabatic+GBOM_batch.gboms[i].E_adiabatic
                        average_E_adiabatic=average_E_adiabatic/(1.0*Eopt.shape[0])     
                        delta_Eadiab_Eopt_av=average_E_adiabatic-Eopt_av
                        Eopt_fluct=Eopt-Eopt_av

                        # now set all Eadiabatic to the average Eadiabatic and recompute the g2 function.
                        for icount in range(GBOM_batch.num_gboms):
                                if param_set.exact_corr:
                                                                                # compute energy gap without 0-0 transition
                                        GBOM_batch.gboms[icount].E_adiabatic=average_E_adiabatic
                                        GBOM_batch.gboms[icount].calc_omega_av_qm(param_set.temperature,False)
                                        GBOM_batch.gboms[icount].calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)
                                else:
                                        GBOM_batch.gboms[icount].E_adiabatic=average_E_adiabatic
                                        GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission)
                                        GBOM_batch.gboms[icount].calc_g2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)

                        #HACK
                        E_start=Eopt_av+average_Egap-param_set.spectral_window/2.0
                        E_end=Eopt_av+average_Egap+param_set.spectral_window/2.0
                        # now construct list of g functions with the corrected energy shift taken from Eopt
                        q_func_eff_batch = []
                        dipole_mom_eff_batch = []
                        icount=0
                        while icount<Eopt.shape[0]:
                                if param_set.exact_corr:
                                    g2_temp=GBOM_batch.gboms[icount].g2_exact
                                else:
                                    g2_temp=GBOM_batch.gboms[icount].g2_cl
                                tcount=0
                                while tcount<g2_temp.shape[0]:
                                        g2_temp[tcount,1]=g2_temp[tcount,1]
                                        tcount=tcount+1
                                g2_temp[:,1]=g2_temp[:,1]+solvent_mod.g2_solvent[:,1]
                                q_func_eff_batch.append(g2_temp)
                                current_dipole=np.zeros(3)
                                current_dipole[0]=3.0*energy_dipole[icount,1]/(2.0*Eopt[icount])
                                dipole_mom_eff_batch.append(current_dipole)
                                icount=icount+1

                        Eopt_const=np.zeros(Eopt.shape[0])
                        Eopt_const[:]=Eopt_const[:]+Eopt_av

                        # created batch of g functions that are all the same, apart from different Eopt shifts
                        # now construct 2DES spectra. 
                        twoDES.calc_2DES_time_series_batch_Eopt_av(q_func_eff_batch,dipole_mom_eff_batch,Eopt.shape[0],E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,Eopt)
                else:

                        filename_2DES=param_set.GBOM_root+''
                        solvent_mod.calc_spectral_dens(param_set.num_steps)
                        solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                        solvent_mod.calc_solvent_response(False)

                        # Now make sure that we have only a single average spectral window for the GBOM batch. 
                        # also set the correct omega_av. Also, use this opportunity to compute g2 for each GBOM 
                        icount=0
                        average_Egap=0.0
                        while icount<GBOM_batch.num_gboms:
                        # figure out start and end value for the spectrum.
                                if param_set.exact_corr:
                                        GBOM_batch.gboms[icount].calc_omega_av_qm(param_set.temperature,False)
                                        GBOM_batch.gboms[icount].calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)
                                        average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_qm
                                else:
                                        GBOM_batch.gboms[icount].calc_g2_cl(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)
                                        GBOM_batch.gboms[icount].calc_omega_av_cl(param_list.temperature,is_emission,param_set.stdout)
                                        average_Egap=average_Egap+GBOM_batch.gboms[icount].omega_av_cl

                                icount=icount+1

                        average_Egap=average_Egap/(1.0*GBOM_batch.num_gboms)

                        E_start=average_Egap-param_set.spectral_window/2.0
                        E_end=average_Egap+param_set.spectral_window/2.0

                        # create a list of effective q functions
                        q_func_eff_batch = []
                        dipole_mom_eff_batch = []
                        icount=0
                        while icount<GBOM_batch.num_gboms:
                                if param_set.exact_corr:
                                        q_func_eff=GBOM_batch.gboms[icount].g2_exact
                                        q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]
                                else:
                                        q_func_eff=GBOM_batch.gboms[icount].g2_cl
                                        q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]
                                q_func_eff_batch.append(q_func_eff)
                                dipole_mom_eff_batch.append(GBOM_batch.gboms[icount].dipole_mom)
                                icount=icount+1


                        # Successfully set up set of GBOMs ready for 2DES calculation.
                        twoDES.calc_2DES_time_series_batch(q_func_eff_batch,dipole_mom_eff_batch,GBOM_batch.num_gboms,E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)


        # For time being, only exact morse spectrum is implemented. In principle, here we can also compute a CUMULANT morse spectrum and 
        # a Franck-Condon type Morse spectrum 
        elif param_set.model=='MORSE':
            if param_set.method=='EXACT':
                # only need to compute corr func so we have a value for the average energy gap
                morse_oscs.compute_total_corr_func_exact(param_set.temperature,param_set.decay_length,param_set.max_t,param_set.num_steps)
                E_start=morse_oscs.omega_av_qm-param_set.spectral_window/2.0
                E_end=morse_oscs.omega_av_qm+param_set.spectral_window/2.0
                filename_2DES='Decoupled_Morse_'
                solvent_mod.calc_spectral_dens(param_set.num_steps)
                solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                                                # set variables in each of the morse oscillators
                for i in range(len(morse_oscs.morse_oscs)):
                    morse_oscs.morse_oscs[i].compute_overlaps_and_transition_energies()
                    morse_oscs.morse_oscs[i].compute_boltzmann_fac(param_set.temperature)

                # now construct 2DES
                morse_2DES.calc_2DES_time_series_morse_list(morse_oscs.morse_oscs,solvent_mod.g2_solvent,E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES)
        
            elif param_set.method=='CUMULANT':
                filename_2DES='Decoupled_Morse_cumulant'
                solvent_mod.calc_spectral_dens(param_set.num_steps)
                solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                morse_oscs.compute_total_corr_func_exact(param_set.temperature,param_set.decay_length,param_set.max_t,param_set.num_steps)
                morse_oscs.compute_spectral_dens()
                morse_oscs.compute_g2_exact(param_set.temperature,param_set.max_t,param_set.num_steps,param_set.stdout)
            
                E_start=morse_oscs.omega_av_qm-param_set.spectral_window/2.0
                E_end=morse_oscs.omega_av_qm+param_set.spectral_window/2.0
    
                q_func_eff=morse_oscs.g2_exact
                q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]

                twoDES.calc_2DES_time_series(q_func_eff,param_set.dipole_mom,E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)

            elif param_set.method=='FC_HARMONIC':
                filename_2DES='Decoupled_Morse_harmonic_FC'
                solvent_mod.calc_spectral_dens(param_set.num_steps)
                solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)

                morse_oscs.eff_gbom.calc_omega_av_qm(param_set.temperature,False)
                morse_oscs.eff_gbom.calc_g2_qm(param_set.temperature,param_set.num_steps,param_set.max_t,False,param_set.stdout)
                # set the start and end values for both the x and the y axis of the
                # 2DES spectrum
                eff_shift1=0.0
                eff_shift2=0.0
                if abs(param_set.omega1)>0.000001:
                    eff_shift1=param_set.omega1-morse_oscs.eff_gbom.omega_av_qm
                if abs(param_set.omega3)>0.000001:
                    eff_shift2=param_set.omega3-morse_oscs.eff_gbom.omega_av_qm

                E_start1=morse_oscs.eff_gbom.omega_av_qm-param_set.spectral_window/2.0+eff_shift1
                E_end1=morse_oscs.eff_gbom.omega_av_qm+param_set.spectral_window/2.0+eff_shift1
                E_start2=morse_oscs.eff_gbom.omega_av_qm-param_set.spectral_window/2.0+eff_shift2
                E_end2=morse_oscs.eff_gbom.omega_av_qm+param_set.spectral_window/2.0+eff_shift2

                q_func_eff=morse_oscs.eff_gbom.g2_exact
                q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]


                twoDES.calc_2DES_time_series(q_func_eff,param_set.dipole_mom,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,0.0)

        elif param_set.model=='COUPLED_MORSE':
            if param_set.method=='EXACT':
                E_start=coupled_morse.E_adiabatic-param_set.spectral_window/2.0
                E_end=coupled_morse.E_adiabatic+param_set.spectral_window/2.0

                filename_2DES='Coupled_Morse_'
                solvent_mod.calc_spectral_dens(param_set.num_steps)
                solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                coupled_morse.compute_overlaps_and_transition_energies()
                coupled_morse.compute_boltzmann_fac(param_set.temperature)
                morse_2DES.calc_2DES_time_series_morse_list(coupled_morse,solvent_mod.g2_solvent,E_start,E_end,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES)
    

        elif param_set.model=='MD':
                filename_2DES=param_set.MD_root+''
                # first check if we have a solvent model
                if param_set.is_solvent:
                        solvent_mod.calc_spectral_dens(param_set.num_steps)
                        solvent_mod.calc_g2_solvent(param_set.temperature,param_set.num_steps,param_set.max_t,param_set.stdout)
                # then set up g2 for MDtraj
                MDtraj.calc_2nd_order_corr()
                MDtraj.calc_spectral_dens(param_set.temperature_MD)
                np.savetxt(param_set.MD_root+'MD_spectral_density.dat', MDtraj.spectral_dens)
                MDtraj.calc_g2(param_set.temperature,param_set.max_t,param_set.num_steps,param_set.stdout)
                # 3rd order cumulant calculation? Then compute g_3, as well as the 3rd order quantum correlation function
                if param_set.third_order:
                        MDtraj.calc_3rd_order_corr(param_set.corr_length_3rd,param_set.stdout)
                        # technically, in 3rd order cumulant, can have 2 different temperatures again. one at
                        # which the MD was performed and one at wich the spectrum is simulated. Fix this...
                        MDtraj.calc_g3(param_set.temperature,param_set.max_t,param_set.num_steps,param_set.low_freq_cutoff,param_set.g3_cutoff,param_set.stdout)
                        MDtraj.calc_corr_func_3rd_qm_freq(param_set.temperature_MD,param_set.low_freq_cutoff)
                        # Check if h1 and h2 are already computed and stored. computational savings...
                        if os.path.exists('h1_real.dat') and os.path.exists('h1_imag.dat') and os.path.exists('h2_real.dat') and os.path.exists('h2_imag.dat') and os.path.exists('h4_real.dat') and os.path.exists('h4_imag.dat') and os.path.exists('h5_real.dat') and os.path.exists('h5_imag.dat'):

                                # read in all files:
                                MDtraj.h1=twoDES.read_2D_spectrum('h1_real.dat',param_set.num_steps)
                                MDtraj.h1[:,:,2]=MDtraj.h1[:,:,2]+1j*(twoDES.read_2D_spectrum('h1_imag.dat',param_set.num_steps))[:,:,2]        
                                MDtraj.h2=twoDES.read_2D_spectrum('h2_real.dat',param_set.num_steps)
                                MDtraj.h2[:,:,2]=MDtraj.h2[:,:,2]+1j*(twoDES.read_2D_spectrum('h2_imag.dat',param_set.num_steps))[:,:,2]        

                                MDtraj.h4=twoDES.read_2D_spectrum('h4_real.dat',param_set.num_steps)
                                MDtraj.h4[:,:,2]=MDtraj.h4[:,:,2]+1j*(twoDES.read_2D_spectrum('h4_imag.dat',param_set.num_steps))[:,:,2]
                                MDtraj.h5=twoDES.read_2D_spectrum('h5_real.dat',param_set.num_steps)
                                MDtraj.h5[:,:,2]=MDtraj.h5[:,:,2]+1j*(twoDES.read_2D_spectrum('h5_imag.dat',param_set.num_steps))[:,:,2]

                        else:
                                MDtraj.calc_h1(param_set.max_t,param_set.num_steps)
                                twoDES.print_2D_spectrum('h1_real.dat',(MDtraj.h1),False)
                                twoDES.print_2D_spectrum('h1_imag.dat',MDtraj.h1,True)
                                MDtraj.calc_h2(param_set.max_t,param_set.num_steps)
                                twoDES.print_2D_spectrum('h2_real.dat',(MDtraj.h2),False)
                                twoDES.print_2D_spectrum('h2_imag.dat',MDtraj.h2,True)

                                MDtraj.calc_h4(param_set.max_t,param_set.num_steps)
                                twoDES.print_2D_spectrum('h4_real.dat',(MDtraj.h4),False)
                                twoDES.print_2D_spectrum('h4_imag.dat',MDtraj.h4,True)
                                MDtraj.calc_h5(param_set.max_t,param_set.num_steps)
                                twoDES.print_2D_spectrum('h5_real.dat',(MDtraj.h5),False)
                                twoDES.print_2D_spectrum('h5_imag.dat',MDtraj.h5,True)


                # set the start and end values for both the x and the y axis of the
                # 2DES spectrum
                eff_shift1=0.0
                eff_shift2=0.0
                if abs(param_set.omega1)>0.000001:
                        eff_shift1=param_set.omega1-MDtraj.mean
                if abs(param_set.omega3)>0.000001:
                        eff_shift2=param_set.omega3-MDtraj.mean

                E_start1=MDtraj.mean-param_set.spectral_window/2.0+eff_shift1
                E_end1=MDtraj.mean+param_set.spectral_window/2.0+eff_shift1     
                E_start2=MDtraj.mean-param_set.spectral_window/2.0+eff_shift2
                E_end2=MDtraj.mean+param_set.spectral_window/2.0+eff_shift2
                

                q_func_eff=MDtraj.g2
                if param_set.is_solvent:
                        q_func_eff[:,1]=q_func_eff[:,1]+solvent_mod.g2_solvent[:,1]
                # now compute 2DES in 2nd order cumulant approach
                if param_set.method_2DES=='2DES':
                        # Check if this is a 3rd order cumulant calculation
                        if param_set.third_order:
                                twoDES.calc_2DES_time_series_3rd(q_func_eff,MDtraj.g3,MDtraj.h1,MDtraj.h2,MDtraj.h4,MDtraj.h5,MDtraj.corr_func_3rd_qm_freq,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,MDtraj.mean)
                        else:
                                twoDES.calc_2DES_time_series(q_func_eff,param_set.dipole_mom,E_start1,E_end1,E_start2,E_end2,param_set.num_steps_2DES,filename_2DES,param_set.num_time_samples_2DES,param_set.t_step_2DES,MDtraj.mean)
                elif param_set.method_2DES=='PUMP_PROBE':
                        twoDES.calc_pump_probe_time_series(q_func_eff,E_start,E_end,param_set.num_steps_2DES,filename_2DES,param_set.pump_energy,param_set.num_time_samples_2DES,param_set.t_step_2DES,MDtraj.mean)
                else:
                        sys.exit('Error: Invalid nonlinear spectroscopy method '+param_set.method_2DES)

else:
            sys.exit('Error: Invalid task '+param_set.task)


# end timer and print the result:
end_time=time.time()
param_set.stdout.write('\n'+'###############################################################################'+'\n')
param_set.stdout.write('Successfully finished spectrum calculation. Elapsed time: '+str(end_time-start_time)+' s.')
param_set.stdout.write('\n'+'###############################################################################'+'\n')
