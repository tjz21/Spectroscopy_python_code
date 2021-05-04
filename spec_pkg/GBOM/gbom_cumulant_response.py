#! /usr/bin/env python

import numpy as np
import math
from numba import jit, njit, prange
import cmath
from spec_pkg.constants import constants as const
from scipy import integrate 

# check whether this is an absorption or an emission calculation
# also include HT capabilities
def compute_cumulant_response(g2, g3, dipole_mom, HT_term,is_3rd_order_cumulant, is_HT,is_emission):
    # compute the effective response function of energy gap fluctuations in the 2nd or 3rd order cumulant
    response_func = np.zeros((g2.shape[0], 2), dtype=complex)
    counter = 0
    while counter < g2.shape[0]:
        response_func[counter, 0] = g2[counter, 0].real
        # if required add 3rd order contribution
        if is_emission:
            # Fix exmission expression for 3rd order cumulant
            if is_3rd_order_cumulant:
                eff_g3 = (
                    1j * g3[counter, 1]
                )  # eff_g now coincides with the Fidler definition of g_3
                response_func[counter, 1] = cmath.exp(
                    -np.conj(g2[counter, 1]) - 1j * np.conj(eff_g3)
                )  # this should be the correct expression
            else:
                response_func[counter, 1] = cmath.exp(-np.conj(g2[counter, 1]))
        else:
            if is_3rd_order_cumulant:
                response_func[counter, 1] = cmath.exp(-g2[counter, 1] - g3[counter, 1])
            else:
                response_func[counter, 1] = cmath.exp(-g2[counter, 1])

        counter = counter + 1

    if is_HT:
            response_func[:,1]=response_func[:,1]*HT_term[:,1]

    else:
            response_func[:,1]=np.dot(dipole_mom,dipole_mom)*response_func[:,1]  

    return response_func




@jit(fastmath=True)
def two_time_corr_func_term_jung(prefac, omega1, omega2, kbT, t1, t2):
    prefac_jung = prefactor_jung(omega1, omega2, kbT)
    full_value = (
        prefac
        * prefac_jung
        / (4.0 * math.pi ** 2.0)
        * cmath.exp(-1j * omega1 * t1)
        * cmath.exp(-1j * omega2 * t2)
    )
    return full_value


@jit(fastmath=True)
def skew_from_third_order_corr(freqs_gs, Omega_sq, gamma, kbT,is_cl):
    n_i_vec = np.zeros(freqs_gs.shape[0])
    icount = 0
    skew=0.0
    while icount < freqs_gs.shape[0]:
        n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
        icount = icount + 1
    if is_cl:
        skew=third_order_corr_t_cl(
                    freqs_gs, Omega_sq, gamma, kbT,0.0,0.0, True)
    else:
        skew=third_order_corr_t_QM(
                    freqs_gs, Omega_sq, gamma, n_i_vec,0.0, 0.0, True)

    return np.real(skew)

@jit(fastmath=True)
def full_third_order_corr_func(
    freqs_gs, Omega_sq, gamma, kbT, max_t, num_points, is_cl, four_phonon_term
):
    corr_func = np.zeros((num_points * 2 + 1, num_points * 2 + 1, 3), dtype=complex)
    step_length = max_t / num_points
    # create n_i_vec
    n_i_vec = np.zeros(freqs_gs.shape[0])
    icount = 0
    while icount < freqs_gs.shape[0]:
        n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
        icount = icount + 1

    count1 = 0
    t1 = -max_t
    while count1 < num_points * 2 + 1:
        t2 = -max_t
        count2 = 0
        while count2 < num_points * 2 + 1:
            corr_func[count1, count2, 0] = t1
            corr_func[count1, count2, 1] = t2
            if is_cl:
                corr_func[count1, count2, 2] = third_order_corr_t_cl(
                    freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term
                )
            else:
                corr_func[count1, count2, 2] = third_order_corr_t_QM(
                    freqs_gs, Omega_sq, gamma, n_i_vec, t1, t2, four_phonon_term
                )
            count2 = count2 + 1
            t2 = t2 + step_length
        count1 = count1 + 1
        t1 = t1 + step_length
    return corr_func


# compute third order quantum correlation function constructed from the classical correlation function using the
# jung prefactor
@njit(fastmath=True)
def third_order_corr_t_cl(freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            # term 1
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, omegai, omegaj, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, omegai, -omegaj, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, -omegai, omegaj, kbT, t1, t2
            )
            gamma_term = gamma_term + two_time_corr_func_term_jung(
                const_fac, -omegai, -omegaj, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1
    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )

                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    # first two terms
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    # Swapped signs of the second term here. Correct?
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + two_time_corr_func_term_jung(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


# compute the third order quantum correlation function for two points in time
@njit(fastmath=True)
def third_order_corr_t_QM(freqs_gs, Omega_sq, gamma, n_i_vec, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j

    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        n_i = n_i_vec[icount]
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            n_j = n_i_vec[jcount]
            const_fac = (
                Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            # term 1
            term1 = (n_i + 1.0) * (n_j + 1.0) * cmath.exp(
                -1j * (omegai * t2 + omegaj * t1)
            ) + n_i * n_j * cmath.exp(1j * (omegai * t2 + omegaj * t1))
            term2 = (n_i + 1.0) * n_j * cmath.exp(
                -1j * (omegai * t2 - omegaj * t1)
            ) + n_i * (n_j + 1.0) * cmath.exp(1j * (omegai * t2 - omegaj * t1))
            term3 = (n_i + 1.0) * (n_j + 1.0) * cmath.exp(
                -1j * (omegai * t2 - omega_m * t1)
            ) + n_i * n_j * cmath.exp(1j * (omegai * t2 - omega_m * t1))
            term4 = (n_i + 1.0) * n_j * cmath.exp(
                -1j * (omegai * t2 - omega_p * t1)
            ) + n_i * (n_j + 1.0) * cmath.exp(1j * (omegai * t2 - omega_p * t1))
            term5 = (n_i + 1.0) * (n_j + 1.0) * cmath.exp(
                -1j * (omega_p * t2 - omegai * t1)
            ) + n_i * n_j * cmath.exp(1j * (omega_p * t2 - omegai * t1))
            term6 = (n_i + 1.0) * n_j * cmath.exp(
                -1j * (omega_m * t2 - omegai * t1)
            ) + n_i * (n_j + 1.0) * cmath.exp(1j * (omega_m * t2 - omegai * t1))

            gamma_term = (
                gamma_term + (term1 + term2 + term3 + term4 + term5 + term6) * const_fac
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            n_i = n_i_vec[icount]
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                n_j = n_i_vec[jcount]
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    n_k = n_i_vec[kcount]
                    const_fac = (
                        Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )

                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]

                    term1 = (n_i + 1.0) * (n_j + 1.0) * (n_k + 1.0) * cmath.exp(
                        -1j * (ij_p * t2 - ik_m * t1)
                    ) + n_i * n_j * n_k * cmath.exp(1j * (ij_p * t2 - ik_m * t1))
                    term2 = (n_i + 1.0) * n_j * (n_k + 1.0) * cmath.exp(
                        -1j * (ij_m * t2 - ik_m * t1)
                    ) + n_i * (n_j + 1.0) * n_k * cmath.exp(
                        1j * (ij_m * t2 - ik_m * t1)
                    )
                    term3 = (n_i + 1.0) * n_j * n_k * cmath.exp(
                        -1j * (ij_m * t2 - ik_p * t1)
                    ) + n_i * (n_j + 1.0) * (n_k + 1.0) * cmath.exp(
                        1j * (ij_m * t2 - ik_p * t1)
                    )
                    term4 = (n_i + 1.0) * (n_j + 1.0) * n_k * cmath.exp(
                        -1j * (ij_p * t2 - ik_p * t1)
                    ) + n_i * n_j * (n_k + 1.0) * cmath.exp(
                        1j * (ij_p * t2 - ik_p * t1)
                    )

                    omega_term = omega_term + const_fac * (
                        term1 + term2 + term3 + term4
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


def compute_spectral_dens(
    freqs_gs, Omega_sq, gamma, kbT, max_t, max_steps, decay_length, is_cl
):
    # compute a longer version of the spectral density
    max_t = max_t * 10
    max_steps = max_steps * 10

    # is the correlation function derived from the classical correlation function or is it the exact
    # quantum correlation function?
    if is_cl:
        full_corr_func = full_2nd_order_corr_cl(
            freqs_gs, Omega_sq, gamma, kbT, max_t, max_steps, decay_length
        )
        np.savetxt("full_correlation_func_classical.dat", full_corr_func)
    else:
        full_corr_func = full_2nd_order_corr_qm(
            freqs_gs, Omega_sq, gamma, kbT, max_t, max_steps, decay_length
        )
        np.savetxt("full_correlation_func_qm.dat", full_corr_func)
    # get frequencies and perform inverse FFT. The imaginary part is the spectral density
    # multiply FFT by step length to get correct units for full spectral density
    corr_func_freq = (
        np.fft.fft(np.fft.ifftshift(full_corr_func[:, 1].imag))
        * max_t
        / max_steps
        * 2.0
        * math.pi
    )  # 2pi is the missing normalization factor. double check
    freqs = np.fft.fftfreq(corr_func_freq.shape[0], max_t / max_steps) * 2.0 * math.pi

    spectral_dens = np.zeros((int(corr_func_freq.shape[0]/2) - 1, 2))
    counter = 0
    while counter < spectral_dens.shape[0]:
        spectral_dens[counter, 0] = freqs[counter]
        spectral_dens[counter, 1] = corr_func_freq[counter].real
        counter = counter + 1

    return spectral_dens


def full_2nd_order_corr_qm(
    freqs_gs, Omega_sq, gamma, kbT, max_t, max_steps, decay_length
):
    step_length = max_t / max_steps
    corr_func = np.zeros((max_steps * 2 - 1, 2), dtype=complex)
    current_t = 0.0

    # compute freqs, freqs_gs and
    Omega_sq_sq = np.multiply(Omega_sq, Omega_sq)

    n_i = np.zeros(freqs_gs.shape[0])
    n_i_p = np.zeros(freqs_gs.shape[0])

    # fill n_i and n_i_p vectors:
    icount = 0
    while icount < freqs_gs.shape[0]:
        n_i[icount] = bose_einstein(freqs_gs[icount], kbT)
        n_i_p[icount] = n_i[icount] + 1.0
        icount = icount + 1

    counter = 0
    while counter < corr_func.shape[0]:
        corr_func[counter, 0] = -max_t + step_length * counter
        corr_func[counter, 1] = second_order_corr_t_qm(
            freqs_gs, Omega_sq_sq.astype(np.complex_), gamma.astype(np.complex_), n_i.astype(np.complex), n_i_p.astype(np.complex), corr_func[counter, 0]
        )
        corr_func[counter, 1] = corr_func[counter, 1] * math.exp(
            -abs(corr_func[counter, 0].real) / decay_length
        )
        counter = counter + 1

    return corr_func

def sd_from_2nd_order_corr_qm(freqs_gs, Omega_sq, gamma, kbT):
    # compute freqs, freqs_gs and
    Omega_sq_sq = np.multiply(Omega_sq, Omega_sq)

    n_i = np.zeros(freqs_gs.shape[0])
    n_i_p = np.zeros(freqs_gs.shape[0])

    # fill n_i and n_i_p vectors:
    icount = 0
    while icount < freqs_gs.shape[0]:
        n_i[icount] = bose_einstein(freqs_gs[icount], kbT)
        n_i_p[icount] = n_i[icount] + 1.0
        icount = icount + 1

    sd=np.sqrt(second_order_corr_t_qm(
            freqs_gs, Omega_sq_sq.astype(np.complex_), gamma.astype(np.complex_), n_i.astype(np.complex), n_i_p.astype(np.complex),0.0))

    return np.real(sd)

def sd_from_2nd_order_corr_cl(freqs_gs, Omega_sq, gamma, kbT):
    Omega_sq_sq = np.multiply(Omega_sq, Omega_sq)

    n_i = np.zeros(freqs_gs.shape[0])
    n_i_p = np.zeros(freqs_gs.shape[0])
    n_ij_p = np.zeros((freqs_gs.shape[0], freqs_gs.shape[0]))
    n_ij_m = np.zeros((freqs_gs.shape[0], freqs_gs.shape[0]))

    # fill n_i vector:
    icount = 0
    while icount < freqs_gs.shape[0]:
        n_i[icount] = bose_einstein(freqs_gs[icount], kbT)
        n_i_p[icount] = n_i[icount] + 1.0
        icount = icount + 1

    # fill n_ij vectors. Make sure to deal corretly with cases where omegai==omegaj:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            n_ij_p[icount, jcount] = bose_einstein(
                freqs_gs[icount] + freqs_gs[jcount], kbT
            )
            if icount == jcount:
                n_ij_m[icount, jcount] = 1.0
            else:
                n_ij_m[icount, jcount] = bose_einstein(
                    freqs_gs[icount] - freqs_gs[jcount], kbT
                )

            jcount = jcount + 1
        icount = icount + 1
    sd=np.sqrt(second_order_corr_t_cl(
            freqs_gs,
            Omega_sq_sq.astype(np.complex_),
            gamma.astype(np.complex_),
            n_i.astype(np.complex_),
            n_i_p.astype(np.complex_),
            n_ij_p.astype(np.complex_),
            n_ij_m.astype(np.complex_),
            kbT,0.0))
    return np.real(sd)

def full_2nd_order_corr_cl(
    freqs_gs, Omega_sq, gamma, kbT, max_t, max_steps, decay_length
):
    step_length = max_t / max_steps
    corr_func = np.zeros((max_steps * 2 - 1, 2), dtype=complex)
    current_t = 0.0

    # compute freqs, freqs_gs and
    Omega_sq_sq = np.multiply(Omega_sq, Omega_sq)

    n_i = np.zeros(freqs_gs.shape[0])
    n_i_p = np.zeros(freqs_gs.shape[0])
    n_ij_p = np.zeros((freqs_gs.shape[0], freqs_gs.shape[0]))
    n_ij_m = np.zeros((freqs_gs.shape[0], freqs_gs.shape[0]))

    # fill n_i vector:
    icount = 0
    while icount < freqs_gs.shape[0]:
        n_i[icount] = bose_einstein(freqs_gs[icount], kbT)
        n_i_p[icount] = n_i[icount] + 1.0
        icount = icount + 1

    # fill n_ij vectors. Make sure to deal corretly with cases where omegai==omegaj:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            n_ij_p[icount, jcount] = bose_einstein(
                freqs_gs[icount] + freqs_gs[jcount], kbT
            )
            if icount == jcount:
                n_ij_m[icount, jcount] = 1.0
            else:
                n_ij_m[icount, jcount] = bose_einstein(
                    freqs_gs[icount] - freqs_gs[jcount], kbT
                )

            jcount = jcount + 1
        icount = icount + 1

    counter = 0
    while counter < corr_func.shape[0]:
        corr_func[counter, 0] = -max_t + step_length * counter
        corr_func[counter, 1] = second_order_corr_t_cl(
            freqs_gs,
            Omega_sq_sq.astype(np.complex_),
            gamma.astype(np.complex_),
            n_i.astype(np.complex_),
            n_i_p.astype(np.complex_),
            n_ij_p.astype(np.complex_),
            n_ij_m.astype(np.complex_),
            kbT,
            corr_func[counter, 0])
        corr_func[counter, 1] = corr_func[counter, 1] * math.exp(
            -abs(corr_func[counter, 0].real) / decay_length
        )
        counter = counter + 1

    return corr_func


# compact vectorized representation of classically derived quantum autocorrelation function
@jit
def second_order_corr_t_cl(
    freqs_gs, Omega_sq, gamma, n_i, n_i_p, n_ij_p, n_ij_m, kbT, t
):
    # vectorized version:
    exp_wt = np.zeros(freqs_gs.shape[0], dtype=np.complex_)
    inv_w = np.zeros(freqs_gs.shape[0], dtype=np.complex_)

    # fill exp_wt vectors
    icount = 0
    while icount < freqs_gs.shape[0]:
        inv_w[icount] = 1.0 / freqs_gs[icount]
        exp_wt[icount] = cmath.exp(1j * freqs_gs[icount] * t)
        icount = icount + 1

    # compact representation of single phonon term
    term1 = 0.5 * np.dot(
        np.multiply(np.conj(exp_wt), gamma),
        np.multiply(inv_w, np.multiply(n_i_p, gamma)),
    ) + 0.5 * np.dot(
        np.multiply(exp_wt, gamma), np.multiply(inv_w, np.multiply(n_i, gamma))
    )

    term2 = 0.0
    term3 = 0.0

    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            w_ij_p = freqs_gs[icount] + freqs_gs[jcount]
            w_ij_m = freqs_gs[icount] - freqs_gs[jcount]

            const_fac = (
                0.5
                * kbT
                * Omega_sq[icount, jcount]
                / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0
            )
            term2 = term2 + const_fac * w_ij_p * (
                (n_ij_p[icount, jcount] + 1.0) * cmath.exp(-1j * w_ij_p * t)
                + n_ij_p[icount, jcount] * cmath.exp(1j * w_ij_p * t)
            )
            if icount != jcount:
                term3 = term3 + const_fac * w_ij_p * (
                    (n_ij_m[icount, jcount] + 1.0) * cmath.exp(-1j * w_ij_m * t)
                    + n_ij_m[icount, jcount] * cmath.exp(1j * w_ij_m * t)
                )
            else:
                term3 = term3 + const_fac * 2.0 * kbT

            jcount = jcount + 1
        icount = icount + 1

    return term1 + term2 + term3


# compact vectorized representation of exact quantum autocorrelation function
@jit
def second_order_corr_t_qm(freqs_gs, Omega_sq, gamma, n_i, n_i_p, t):
    # vectorized version:
    exp_wt = np.zeros(freqs_gs.shape[0], dtype=np.complex_)
    inv_w = np.zeros(freqs_gs.shape[0], dtype=np.complex_)

    # fill exp_wt vectors
    for icount in range(freqs_gs.shape[0]):
        inv_w[icount] = 1.0 / freqs_gs[icount]
        exp_wt[icount] = cmath.exp(1j * freqs_gs[icount] * t)

    # compact representation of single phonon term
    term1 = 0.5 * np.dot(
        np.multiply(np.conj(exp_wt), gamma),
        np.multiply(inv_w, np.multiply(n_i_p, gamma)),
    ) + 0.5 * np.dot(
        np.multiply(exp_wt, gamma), np.multiply(inv_w, np.multiply(n_i, gamma))
    )

    # compact representation of multiple phonon term
    term2 = 0.5 * np.dot(
        np.multiply(np.multiply(n_i_p, inv_w), np.conj(exp_wt)),
        np.dot(Omega_sq, np.multiply(np.multiply(n_i_p, inv_w), np.conj(exp_wt))),
    )
    term2 = term2 + 0.5 * np.dot(
        np.multiply(np.multiply(n_i, inv_w), exp_wt),
        np.dot(Omega_sq, np.multiply(np.multiply(n_i, inv_w), exp_wt)),
    )
    term2 = term2 + 0.5 * np.dot(
        np.multiply(np.multiply(n_i_p, inv_w), np.conj(exp_wt)),
        np.dot(Omega_sq, np.multiply(np.multiply(n_i, inv_w), exp_wt)),
    )
    term2 = term2 + 0.5 * np.dot(
        np.multiply(np.multiply(n_i, inv_w), exp_wt),
        np.dot(Omega_sq, np.multiply(np.multiply(n_i_p, inv_w), np.conj(exp_wt))),
    )

    return term1 + term2


@njit(fastmath=True)
def bose_einstein(freq, kbT):
    n = math.exp(freq / kbT) - 1.0
    return 1.0 / n


#  TEST: COMPUTE DIPOLE ENERGY AND DIPOLE DIPOLE CORR FUNC:
#  computing classical correlation funcs, then FFT to build SD
def compute_spectral_dens_dipole(freqs_gs,Jmat,gamma,dipole_deriv,dipole_mom,kbT,decay_length,max_t,num_points):
    num_points=num_points*10
    max_t=max_t*10.0 
    dipole_dipole=np.zeros((num_points*2-1,2),dtype=complex) 
    dipole_energy=np.zeros((num_points*2-1,2),dtype=complex) # dipole energy corr func is computed by forming the dot product with the dipole mom 

    n_i=np.zeros(freqs_gs.shape[0])
    for i in range(n_i.shape[0]):
        n_i[i]=bose_einstein(freqs_gs[i],kbT)

    step_length=max_t/num_points
    for i in range(dipole_dipole.shape[0]):
        t= -max_t + step_length * i
        dipole_dipole[i,0]=t
        dipole_dipole[i,1]=compute_corr_func_dipole_dipole_qm_t(freqs_gs,dipole_deriv,Jmat,n_i,t)
        dipole_energy[i,0]=t
        dipole_energy[i,1]=compute_corr_func_dipole_energy_qm_t(freqs_gs,dipole_mom,dipole_deriv,gamma,Jmat,n_i,t)

    dipole_dipole[:,1]=dipole_dipole[:,1]*np.exp(-abs(dipole_dipole[:,0].real)/decay_length)
    dipole_energy[:,1]=dipole_energy[:,1]*np.exp(-abs(dipole_energy[:,0].real)/decay_length)

    np.savetxt('dipole_dipole_corr_func_qm.txt',dipole_dipole)
 
    corr_func_freq = (
        np.fft.fft(np.fft.ifftshift(dipole_dipole[:, 1].imag))
        * max_t
        / num_points
        * 2.0
        * math.pi
    )  # 2pi is the missing normalization factor. double check
    corr_func_freq_dipole_energy = (
        np.fft.fft(np.fft.ifftshift(dipole_energy[:, 1].imag))
        * max_t
        / num_points
        * 2.0
        * math.pi
    )  # 2pi is the missing normalization factor. double check
    freqs = np.fft.fftfreq(corr_func_freq.shape[0], max_t / num_points) * 2.0 * math.pi

    spectral_dens = np.zeros((int(corr_func_freq.shape[0]/2) - 1, 2))
    spectral_dens_dipole_energy=np.zeros((int(corr_func_freq.shape[0]/2) - 1, 2))
    counter = 0
    while counter < spectral_dens.shape[0]:
        spectral_dens[counter, 0] = freqs[counter]
        spectral_dens[counter, 1] = corr_func_freq[counter].real
        spectral_dens_dipole_energy[counter,0]=freqs[counter]
        spectral_dens_dipole_energy[counter,1]=corr_func_freq_dipole_energy[counter].real
        counter = counter + 1

    np.savetxt('dipole_dipole_SD.txt',spectral_dens)
    np.savetxt('dipole_energy_SD.txt',spectral_dens_dipole_energy)


#  computing classical correlation funcs, then FFT to build SD
def compute_spectral_dens_dipole_cl(freqs_gs,Jmat,gamma,dipole_deriv,dipole_mom,reorg_dipole,kbT,decay_length,max_t,num_points):
    num_points=num_points*10
    max_t=max_t*10.0 
    dipole_dipole=np.zeros((num_points*2-1,2),dtype=complex) 
    dipole_energy=np.zeros((num_points*2-1,2),dtype=complex) # dipole energy corr func is computed by forming the dot product with the dipole mom 

    step_length=max_t/num_points
    for i in range(dipole_dipole.shape[0]):
        t= -max_t + step_length * i
        dipole_dipole[i,0]=t
        dipole_dipole[i,1]=compute_corr_func_dipole_dipole_cl_t(freqs_gs,dipole_deriv,Jmat,kbT,t)
        dipole_energy[i,0]=t
        dipole_energy[i,1]=compute_corr_func_dipole_energy_cl_t(freqs_gs,dipole_mom,reorg_dipole,dipole_deriv,gamma,Jmat,kbT,t)

    dipole_dipole[:,1]=dipole_dipole[:,1]*np.exp(-abs(dipole_dipole[:,0].real)/decay_length)
    dipole_energy[:,1]=dipole_energy[:,1]*np.exp(-abs(dipole_energy[:,0].real)/decay_length)

    np.savetxt('dipole_dipole_corr_func_cl.txt',np.real(dipole_dipole))
    np.savetxt('dipole_energy_corr_func_cl.txt',np.real(dipole_energy))

    corr_func_freq = (
        np.fft.fft(np.fft.ifftshift(dipole_dipole[:, 1]))
        * max_t
        / num_points
        * 2.0
        * math.pi
    )  # 2pi is the missing normalization factor. double check
    corr_func_freq_dipole_energy = (
        np.fft.fft(np.fft.ifftshift(dipole_energy[:, 1]))
        * max_t
        / num_points
        * 2.0
        * math.pi
    )  # 2pi is the missing normalization factor. double check
    freqs = np.fft.fftfreq(corr_func_freq.shape[0], max_t / num_points) * 2.0 * math.pi

    spectral_dens = np.zeros((int(corr_func_freq.shape[0]/2) - 1, 2))
    spectral_dens_dipole_energy=np.zeros((int(corr_func_freq.shape[0]/2) - 1, 2))
    cross_corr_freq=np.zeros((int(corr_func_freq.shape[0]/2) - 1, 2))
 
    counter = 0
    while counter < spectral_dens.shape[0]:
        spectral_dens[counter, 0] = freqs[counter]
        spectral_dens[counter, 1] = spectral_dens[counter,0]/(2.0*kbT)*corr_func_freq[counter].real
        spectral_dens_dipole_energy[counter,0]=freqs[counter]
        spectral_dens_dipole_energy[counter,1]=spectral_dens[counter,0]/(2.0*kbT)*corr_func_freq_dipole_energy[counter].real
        cross_corr_freq[counter,0]=freqs[counter]
        cross_corr_freq[counter,1]=corr_func_freq_dipole_energy[counter].real
        counter = counter + 1

    np.savetxt('dipole_dipole_SD_cl.txt',spectral_dens)
    np.savetxt('dipole_energy_SD_cl.txt',spectral_dens_dipole_energy)

    return cross_corr_freq

def compute_corr_func_dipole_dipole_qm_t(freqs_gs,dipole_deriv,Jmat,n,t):
    val=0.0+0.0*1j
    dipole_deriv_J=np.zeros((Jmat.shape[0],3))
    dipole_deriv_J[:,0]=np.dot(dipole_deriv[:,0],np.transpose(Jmat))
    dipole_deriv_J[:,1]=np.dot(dipole_deriv[:,1],np.transpose(Jmat))
    dipole_deriv_J[:,2]=np.dot(dipole_deriv[:,2],np.transpose(Jmat))


    for i in range(freqs_gs.shape[0]):
        val=val+1.0/(2.0*freqs_gs[i])*np.dot(dipole_deriv_J[i,:],dipole_deriv_J[i,:])*((n[i]+1.0)*np.exp(-1j*freqs_gs[i]*t)+n[i]*np.exp(1j*freqs_gs[i]*t))

    return val

def compute_corr_func_dipole_energy_qm_t(freqs_gs,dipole_mom,dipole_deriv,gamma,Jmat,n,t):
    val=0.0+0.0*1j
    dipole_deriv_J=np.zeros((Jmat.shape[0],3))
    dipole_deriv_J[:,0]=np.dot(dipole_deriv[:,0],np.transpose(Jmat))
    dipole_deriv_J[:,1]=np.dot(dipole_deriv[:,1],np.transpose(Jmat))
    dipole_deriv_J[:,2]=np.dot(dipole_deriv[:,2],np.transpose(Jmat))

    for i in range(freqs_gs.shape[0]):
        val=val-gamma[i]*np.dot(dipole_deriv_J[i,:],dipole_mom)/(2.0*freqs_gs[i])*((n[i]+1.0)*np.exp(-1j*freqs_gs[i]*t)+n[i]*np.exp(1j*freqs_gs[i]*t))

    return val 


def compute_corr_func_dipole_dipole_cl_t(freqs_gs,dipole_deriv,Jmat,kbT,t):
    val=0.0+0.0*1j
    dipole_deriv_J=np.zeros((Jmat.shape[0],3))
    dipole_deriv_J[:,0]=np.dot(dipole_deriv[:,0],np.transpose(Jmat))
    dipole_deriv_J[:,1]=np.dot(dipole_deriv[:,1],np.transpose(Jmat))
    dipole_deriv_J[:,2]=np.dot(dipole_deriv[:,2],np.transpose(Jmat))


    for i in range(freqs_gs.shape[0]):
        val=val+kbT/(2.0*freqs_gs[i]**2.0)*np.dot(dipole_deriv_J[i,:],dipole_deriv_J[i,:])*(np.cos(freqs_gs[i]*t))

    return val

def compute_corr_func_dipole_energy_cl_t(freqs_gs,dipole_mom,reorg_dipole,dipole_deriv,gamma,Jmat,kbT,t):
    val=0.0+0.0*1j
    dipole_deriv_J=np.zeros((Jmat.shape[0],3))
    dipole_deriv_J[:,0]=np.dot(dipole_deriv[:,0],np.transpose(Jmat))
    dipole_deriv_J[:,1]=np.dot(dipole_deriv[:,1],np.transpose(Jmat))
    dipole_deriv_J[:,2]=np.dot(dipole_deriv[:,2],np.transpose(Jmat))

    for i in range(freqs_gs.shape[0]):
        val=val-gamma[i]*np.dot(dipole_deriv_J[i,:],(reorg_dipole-dipole_mom))*kbT/(2.0*freqs_gs[i]**2.0)*np.cos(freqs_gs[i]*t)

    return val

# full HT function
def full_HT_term(freqs_gs,Kmat,Jmat,Omega_sq,gamma,dipole_mom,dipole_deriv,kbT,max_t,num_points,decay_length,is_qm,is_3rd_order,dipole_dipole_only,stdout):
    # Before doing anything, compute the dipole dipole correlation function
    Jtrans=np.transpose(Jmat)
    HT_func=np.zeros((num_points, 2), dtype=complex)
    n_i_vec = np.zeros(freqs_gs.shape[0])
    # correct order
    Jdip_deriv=np.zeros((freqs_gs.shape[0],3))
    Jdip_deriv[:,0]=np.dot(dipole_deriv[:,0],Jtrans)
    Jdip_deriv[:,1]=np.dot(dipole_deriv[:,1],Jtrans)
    Jdip_deriv[:,2]=np.dot(dipole_deriv[:,2],Jtrans)
    reorg_dipole=np.zeros(3)
    reorg_dipole[0]=np.dot(Jdip_deriv[:,0],Kmat)
    reorg_dipole[1]=np.dot(Jdip_deriv[:,1],Kmat)
    reorg_dipole[2]=np.dot(Jdip_deriv[:,2],Kmat)

    # SET reorg_dipole to zero. THIS IS Equivalent to expanding around the GS PES.
    reorg_dipole=np.zeros(3)

    compute_spectral_dens_dipole(freqs_gs,Jmat,gamma,dipole_deriv,dipole_mom,kbT,decay_length,max_t,num_points)
    cross_corr_freq=compute_spectral_dens_dipole_cl(freqs_gs,Jmat,gamma,dipole_deriv,dipole_mom,reorg_dipole,kbT,decay_length,max_t,num_points)


    # TEST HOW BIG the FCHT TERM IS:
    inv_w=1.0/freqs_gs[:]
    inv_w_sq=np.multiply(inv_w,inv_w)
    gamma_inv_sq=np.multiply(inv_w_sq,gamma)
    eff_size=np.zeros(3)
    eff_size[0]=1.0/kbT*(np.dot(gamma_inv_sq,Jdip_deriv[:,0]))
    eff_size[1]=1.0/kbT*(np.dot(gamma_inv_sq,Jdip_deriv[:,1]))
    eff_size[2]=1.0/kbT*(np.dot(gamma_inv_sq,Jdip_deriv[:,2]))
 
    print('Size of corr func')
    print(eff_size,np.dot(dipole_mom,eff_size))


    # compute the total renormalized dipole moment
    total_renorm_dipole=np.dot(dipole_mom,dipole_mom)-2.0*np.dot(reorg_dipole,dipole_mom)+np.dot(reorg_dipole,reorg_dipole)

    # create bose einstein factors
    for icount in range(freqs_gs.shape[0]):
        n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
 
    t=0.0
    step_length=max_t/num_points
    for i in range(num_points):
        HT_func[i,0]=t
        if is_qm:
            HT_func[i,1]=HT_qm_t(freqs_gs,Kmat,Jtrans,Omega_sq,gamma,n_i_vec,dipole_mom,Jdip_deriv,reorg_dipole,total_renorm_dipole,is_3rd_order,dipole_dipole_only,t) 
        else:
            HT_func[i,1]=HT_cl_t(freqs_gs,cross_corr_freq,Kmat,Jtrans,Omega_sq,gamma,n_i_vec,kbT,dipole_mom,Jdip_deriv,reorg_dipole,total_renorm_dipole,is_3rd_order,dipole_dipole_only,t)
        t=t+step_length

    return HT_func

def one_deriv_term_intergrant(cross_corr_func_freq,kbT,t):
    integrant=np.zeros((cross_corr_func_freq.shape[0],cross_corr_func_freq.shape[1]),dtype=np.complex_)
    tol=1.0e-15
    for i in range(cross_corr_func_freq.shape[0]):
        integrant[i,0]=cross_corr_func_freq[i,0]
        # check for omega=0 condition
        omega=integrant[i,0]
        if abs(omega)<tol:
           denom=2.0*math.pi
           num=2.0*t*cross_corr_func_freq[i,1]
           integrant[i,1]=num/denom

        else:
           denom=kbT*2.0*math.pi*(1.0-np.exp(-omega/kbT))
           num=2.0*cross_corr_func_freq[i,1]*(1.0-cmath.exp(-1j*omega*t))
           integrant[i,1]=num/denom

    return integrant



def HT_cl_t(freqs_gs,cross_corr_func_freq,Kmat,Jtrans,Omega_sq,gamma,n,kBT,dipole_mom,Jdip_deriv,reorg_dipole,total_renorm_dipole,is_3rd_order,dipole_dipole_only,t):
    one_deriv_term=0.0+0.0*1j
    # 2nd order HT term is unchanged from qm version of the same term.
    for i in range(freqs_gs.shape[0]):
        term_xyz=-2.0*(np.dot(reorg_dipole-dipole_mom,Jdip_deriv[i,:]))*gamma[i]/(4.0*math.pi*(freqs_gs[i])**2.0)*((n[i]+1.0)*(1.0-cmath.exp(-1j*freqs_gs[i]*t))-n[i]*(1.0-cmath.exp(1j*freqs_gs[i])*t))  # (n+1) and n term are opposite sign!)
        one_deriv_term=one_deriv_term+term_xyz

#    print(t)
#    integrant=one_deriv_term_intergrant(cross_corr_func_freq,kBT,t)
#    one_deriv_term=-integrate.simps(integrant[:,1],dx=integrant[1,0]-integrant[0,0])   


    two_deriv_term=0.0+1j*0.0
    for i in range(freqs_gs.shape[0]):
        term_xyz=(Jdip_deriv[i,0]**2.0+Jdip_deriv[i,1]**2.0+Jdip_deriv[i,2]**2.0)/(2.0*freqs_gs[i])*((n[i]+1.0)*cmath.exp(-1j*freqs_gs[i]*t)+n[i]*cmath.exp(1j*freqs_gs[i]*t))
        two_deriv_term=two_deriv_term+term_xyz

    if is_3rd_order:
        # Compute 3rd order cumulant correction
        dUdUdmu=0.0+0.0j
        dmudUdU=0.0+0.0j
        dmudUdmu=0.0+0.0j

        # loop over two phonon terms
        for i in range(freqs_gs.shape[0]):
            for j in range(freqs_gs.shape[0]):
                # start with dmudUdmu
                omega_m=freqs_gs[i]-freqs_gs[j]
                omega_p=freqs_gs[i]+freqs_gs[j]

                # absorb the extra minus sign in the prefactor
                prefac1=-1j*math.pi**2.0*kBT**2.0*(Omega_sq[i,j]+Omega_sq[j,i])*np.dot(Jdip_deriv[i,:],Jdip_deriv[j,:])/(freqs_gs[i]*freqs_gs[j])**2.0

                dmudUdmu=dmudUdmu+HT_fac_dmu_dU_dmu_cl(prefac1,kBT,-omega_m,freqs_gs[i],t)
                dmudUdmu=dmudUdmu+HT_fac_dmu_dU_dmu_cl(prefac1,kBT,omega_m,-freqs_gs[i],t)
                dmudUdmu=dmudUdmu+HT_fac_dmu_dU_dmu_cl(prefac1,kBT,-omega_p,freqs_gs[i],t)
                dmudUdmu=dmudUdmu+HT_fac_dmu_dU_dmu_cl(prefac1,kBT,omega_p,-freqs_gs[i],t)

                # now dUdUdmu
                prefac2=-(np.dot(Jdip_deriv[i,:],reorg_dipole)-2.0*np.dot(Jdip_deriv[i,:],dipole_mom))*math.pi**2.0*kBT**2.0*(Omega_sq[i,j]+Omega_sq[j,i])*gamma[j]/(freqs_gs[i]*freqs_gs[j])**2.0
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,-freqs_gs[i],omega_p,t)
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,freqs_gs[i],-omega_p,t)
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,-freqs_gs[i],omega_m,t)
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,freqs_gs[i],-omega_m,t)

                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,-omega_m,freqs_gs[i],t)
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,omega_m,-freqs_gs[i],t)
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,-omega_p,freqs_gs[i],t)
                dUdUdmu=dUdUdmu+HT_fac_dU_dU_dmu_cl(prefac2,kBT,omega_p,-freqs_gs[i],t)


                # now dmudUdU. Slightly different prefactor. 
                prefac3=-np.dot(Jdip_deriv[i,:],reorg_dipole)*math.pi**2.0*kBT**2.0*(Omega_sq[i,j]+Omega_sq[j,i])*gamma[j]/(freqs_gs[i]*freqs_gs[j])**2.0
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,freqs_gs[i],freqs_gs[j],t)
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,-freqs_gs[i],-freqs_gs[j],t)
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,-freqs_gs[i],freqs_gs[j],t)
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,freqs_gs[i],-freqs_gs[j],t)

                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,-freqs_gs[i],omega_p,t)
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,freqs_gs[i],-omega_p,t)
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,-freqs_gs[i],omega_m,t)
                dmudUdU=dmudUdU+HT_fac_dmu_dU_dU_cl(prefac3,kBT,freqs_gs[i],-omega_m,t)


        total=total_renorm_dipole+one_deriv_term+two_deriv_term+dUdUdmu+dmudUdU+dmudUdmu
    else:
        if dipole_dipole_only:  # only two derivative term
            total=total_renorm_dipole+two_deriv_term
        else:
            total=total_renorm_dipole+one_deriv_term+two_deriv_term

    return total

# HT prefactor for a given value of t if the exact quantum correlation function is used. 
def HT_qm_t(freqs_gs,Kmat,Jtrans,Omega_sq,gamma,n,dipole_mom,Jdip_deriv,reorg_dipole,total_renorm_dipole,is_3rd_order,dipole_dipole_only,t):
    one_deriv_term=0.0+0.0*1j
    for i in range(freqs_gs.shape[0]):
        term_xyz=-2.0*(np.dot(reorg_dipole-dipole_mom,Jdip_deriv[i,:]))*gamma[i]/(4.0*math.pi*(freqs_gs[i])**2.0)*((n[i]+1.0)*(1.0-cmath.exp(-1j*freqs_gs[i]*t))-n[i]*(1.0-cmath.exp(1j*freqs_gs[i])*t))
	# n+1 and n term are opposite sign!
        one_deriv_term=one_deriv_term+term_xyz

    two_deriv_term=0.0+1j*0.0
    for i in range(freqs_gs.shape[0]):
        term_xyz=(np.dot(Jdip_deriv[i,:],Jdip_deriv[i,:]))/(2.0*freqs_gs[i])*((n[i]+1.0)*cmath.exp(-1j*freqs_gs[i]*t)+n[i]*cmath.exp(1j*freqs_gs[i]*t))
        two_deriv_term=two_deriv_term+term_xyz


    if is_3rd_order:
        # Compute 3rd order cumulant correction
        dUdUdmu=0.0+0.0j
        dmudUdU=0.0+0.0j
        dmudUdmu=0.0+0.0j
  
        # loop over two phonon terms
        for i in range(freqs_gs.shape[0]):
            for j in range(freqs_gs.shape[0]):
                # start with dmudUdmu
                omega_m=freqs_gs[i]-freqs_gs[j]
                omega_p=freqs_gs[i]+freqs_gs[j]

                # absorb the extra minus sign in the prefactor 
                prefac1=-(Omega_sq[i,j]+Omega_sq[j,i])*np.dot(Jdip_deriv[i,:],Jdip_deriv[j,:])/(4.0*freqs_gs[i]*freqs_gs[j])
                 
                dmudUdmu=dmudUdmu+prefac1*HT_fac_dmu_dU_dmu((n[i]+1.0)*(n[j]+1.0),-omega_m,freqs_gs[i],t)
                dmudUdmu=dmudUdmu+prefac1*HT_fac_dmu_dU_dmu(n[i]*n[j],omega_m,-freqs_gs[i],t)
                dmudUdmu=dmudUdmu+prefac1*HT_fac_dmu_dU_dmu((n[i]+1.0)*n[j],-omega_p,freqs_gs[i],t)
                dmudUdmu=dmudUdmu+prefac1*HT_fac_dmu_dU_dmu((n[j]+1.0)*n[i],omega_p,-freqs_gs[i],t)

                # now dUdUdmu
                prefac2=-(np.dot(Jdip_deriv[i,:],reorg_dipole)-2.0*np.dot(Jdip_deriv[i,:],dipole_mom))*(Omega_sq[i,j]+Omega_sq[j,i])*gamma[j]/(4.0*freqs_gs[i]*freqs_gs[j])
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu((n[i]+1.0)*(n[j]+1.0),-freqs_gs[i],omega_p,t)
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu(n[i]*n[j],freqs_gs[i],-omega_p,t)
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu((n[i]+1.0)*n[j],-freqs_gs[i],omega_m,t)
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu((n[j]+1.0)*n[i],freqs_gs[i],-omega_m,t)

                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu((n[i]+1.0)*(n[j]+1.0),-omega_m,freqs_gs[i],t)
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu(n[i]*n[j],omega_m,-freqs_gs[i],t)
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu((n[i]+1.0)*n[j],-omega_p,freqs_gs[i],t)
                dUdUdmu=dUdUdmu+prefac2*HT_fac_dU_dU_dmu((n[j]+1.0)*n[i],omega_p,-freqs_gs[i],t)


                # now dmudUdU. Slightly different prefactor. 
                prefac3=-np.dot(Jdip_deriv[i,:],reorg_dipole)*(Omega_sq[i,j]+Omega_sq[j,i])*gamma[j]/(4.0*freqs_gs[i]*freqs_gs[j])
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU((n[i]+1.0)*(n[j]+1.0),freqs_gs[i],freqs_gs[j],t)
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU(n[i]*n[j]+1.0,-freqs_gs[i],-freqs_gs[j],t)
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU((n[j]+1.0)*n[i],-freqs_gs[i],freqs_gs[j],t)
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU((n[i]+1.0)*n[j],freqs_gs[i],-freqs_gs[j],t)
             
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU((n[i]+1.0)*(n[j]+1.0),-freqs_gs[i],omega_p,t)
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU(n[i]*n[j],freqs_gs[i],-omega_p,t)
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU((n[i]+1.0)*n[j],-freqs_gs[i],omega_m,t)
                dmudUdU=dmudUdU+prefac3*HT_fac_dmu_dU_dU((n[j]+1.0)*n[i],freqs_gs[i],-omega_m,t)
                

        total=total_renorm_dipole+one_deriv_term+dUdUdmu+dmudUdU+two_deriv_term+dmudUdmu
    else:
        if dipole_dipole_only:
            total=total_renorm_dipole+two_deriv_term
        else:
            total=total_renorm_dipole+one_deriv_term+two_deriv_term

    return total

# the factor that arises from performing the time-ordered integral over the dUdUdmu corr func
@jit
def HT_fac_dU_dU_dmu(prefac,omega1,omega2,t):
    omega_bar=omega1+omega2
    tiny=10e-16
    term=0.0+1j*0.0

    if abs(omega2)<tiny:
       term=(1.0-cmath.exp(-1j*omega_bar*t))/(omega1*omega_bar)-1j*t/omega1
    elif abs(omega_bar)<tiny:
    
       term=1j*t/omega1-(1.0-cmath.exp(-1j*omega2*t))/(omega1*omega2)
    elif abs(omega1)<tiny:
       term=1j*t*cmath.exp(-1j*omega2*t)/omega2  # double check this limit

    else:
       term=(1.0-cmath.exp(-1j*omega_bar*t))/(omega1*omega_bar)-(1.0-cmath.exp(-1j*omega2*t))/(omega1*omega2)

    return term*prefac


# unlike for the normal response function, we do not have to account for the w1,w2=0 term as there are no 
# 0 frequency modes
@jit
def HT_fac_dU_dU_dmu_cl(prefac,kBT,omega1,omega2,t):
    omega_bar=omega1+omega2
    tiny=10e-16
    term=0.0+1j*0.0
    
    if abs(omega_bar)<tiny:
        num=(np.exp(1j*omega1*t)-1j*omega1*t-1.0)
        denom=8.0*kBT**2.0*math.pi**2.0*(1.0-np.exp(omega1/kBT)+omega1/kBT)
        term=num/denom
    elif abs(omega1)<tiny:
        num=np.exp(omega2/kBT)*np.exp(-1j*omega2*t)*(np.exp(1j*omega2*t)-1j*omega2*t-1.0)
        denom=8.0*kBT**2.0*math.pi**2.0*(1.0-np.exp(omega2/kBT)+omega2/kBT)
        term=num/denom
    elif abs(omega2)<tiny:
        num=np.exp(omega1/kBT)*np.exp(-1j*omega1*t)*(np.exp(1j*omega1*t)*(1.0-1j*omega1*t)-1.0)
        denom=8.0*kBT**2.0*math.pi**2.0*(1.0+np.exp(omega1/kBT)*(omega1/kBT-1.0))
        term=num/denom
    else:
        num=omega_bar*omega2*((1.0-np.exp(-1j*omega_bar*t))/omega_bar-(1.0-np.exp(-1j*omega2*t))/omega2)
        denom=8.0*kBT**2.0*math.pi**2.0*(omega2*np.exp(-omega_bar/kBT)-omega_bar*np.exp(-omega2/kBT)+omega1)
        term=num/denom

    return term*prefac

@jit
def HT_fac_dmu_dU_dmu(prefac,omega1,omega2,t):
    omega_bar=omega1+omega2
    tiny=10e-16
    term=0.0+1j*0.0

    if abs(omega1)<tiny:
       term=1j*t*cmath.exp(-1j*omega2*t)
    else:
       term=(cmath.exp(-1j*omega2*t)-cmath.exp(-1j*omega_bar*t))/omega1

    return term*prefac

@jit
def HT_fac_dmu_dU_dmu_cl(prefac,kBT,omega1,omega2,t):
    omega_bar=omega1+omega2
    tiny=10e-16
    term=0.0+1j*0.0
   
    if abs(omega_bar)<tiny:
       num=1j*omega1*(np.exp(1j*omega1*t)-1.0)	
       denom=8.0*kBT**2.0*math.pi**2.0*(1.0-np.exp(omega1/kBT)-omega1/kBT)
       term=num/denom
    elif abs(omega1)<tiny:
       num=omega2**2.0*t*np.exp(omega2/kBT)*np.exp(-1j*omega2*t)
       denom=8.0*kBT**2.0*math.pi**2.0*(np.exp(omega2/kBT)-omega2/kBT-1.0)
       term=num/denom
    elif abs(omega2)<tiny:
       num=-1j*omega1*np.exp(omega1/kBT)*np.exp(-1j*omega1*t)*(np.exp(1j*omega1*t)-1.0)
       denom=8.0*kBT**2.0*math.pi**2.0*(1.0+np.exp(omega1/kBT)*(omega1/kBT-1.0))
       term=num/denom
    else:
       num=-1j*omega_bar*omega2*(np.exp(-1j*omega2*t)-np.exp(-1j*omega_bar*t))
       denom=8.0*kBT**2.0*math.pi**2.0*(omega2*np.exp(-omega_bar/kBT)-omega_bar*np.exp(-omega2/kBT)+omega1)
       term=num/denom
    return term*prefac

@jit
def HT_fac_dmu_dU_dU(prefac,omega1,omega2,t):
    omega_bar=omega1+omega2
    tiny=10e-16
    term=0.0+1j*0.0

    if abs(omega1)<tiny:
       term=cmath.exp(-1j*omega2*t)/omega_bar*((cmath.exp(1j*omega2*t)-1.0)/omega2-1j*t)      
    elif abs(omega2)<tiny:
       term=cmath.exp(-1j*omega2*t)/omega_bar*(1j*t+(cmath.exp(-1j*omega1*t)-1.0)/omega1) 
    elif abs(omega_bar)<tiny:
       term=0.0+0.0j # NOT THE CORRECT LIMIT. WORK OUT ANALYTICALLY
    else:
       term=cmath.exp(-1j*omega2*t)/omega_bar*((cmath.exp(1j*omega2*t)-1.0)/omega2+(cmath.exp(-1j*omega1*t)-1.0)/omega1) 
    
    return prefac*term

@jit
def HT_fac_dmu_dU_dU_cl(prefac,kBT,omega1,omega2,t):
    omega_bar=omega1+omega2
    tiny=10e-16
    term=0.0+1j*0.0

    if abs(omega1)<tiny:
         num=np.exp(omega2/kBT)*np.exp(-1j*omega2*t)*(np.exp(1j*omega2*t)-1j*omega2*t-1.0)
         denom=8.0*math.pi**2.0*kBT**2.0*(np.exp(omega2/kBT)-omega2/kBT-1.0)
         term=num/denom
    elif abs(omega2)<tiny:
         num=np.exp(omega1/kBT)*(np.exp(-1j*omega1*t)+1j*omega1*t-1.0)
         denom=8.0*math.pi**2.0*kBT**2.0*(1.0+np.exp(omega1/kBT)*(omega1/kBT)-1.0)
         term=num/denom
    elif abs(omega_bar)<tiny:
         num=(np.exp(1j*omega1*t)-1j*omega1*t-1.0)
         denom=8.0*math.pi**2.0*kBT**2.0*(np.exp(omega1/kBT)-omega1/kBT-1.0)
         term=num/denom
    else:
         num=omega1*omega2*np.exp(-1j*omega2*t)*((np.exp(1j*omega2*t)-1.0)/omega2+(np.exp(-1j*omega1*t)-1.0)/omega1)
         denom=8.0*kBT**2.0*math.pi**2.0*(omega2*np.exp(-omega_bar/kBT)-omega_bar*np.exp(-omega2/kBT)+omega1)
         term=num/denom
    return prefac*term

# g2 as derived from either the exact classical correlation function or the exact QM correlation function
def full_2nd_order_lineshape(
    freqs_gs,
    freqs_ex,
    Kmat,
    Jmat,
    gamma,
    Omega_sq,
    kbT,
    av_energy_gap,
    max_t,
    num_points,
    is_cl,
    is_emission,
    E_adiabatic,
stdout):
    lineshape_func = np.zeros((num_points, 2), dtype=complex)
    stdout.write('\n'+"Computing second order cumulant lineshape function."+'\n')
    stdout.write("Av energy gap:  "+str(av_energy_gap)+'  Ha          kBT:  '+str(kbT)+ '  Ha'+'\n')
    step_length = max_t / num_points
    stdout.write('\n'+'  Step       Time (fs)          Re[g_2]         Im[g_2]'+'\n')
    t = 0.0
    count1 = 0
    while count1 < num_points:
        lineshape_func[count1, 0] = t
        if is_emission:
            if is_cl:
                lineshape_func[count1, 1] = (
                    second_order_lineshape_cl_t(freqs_gs, Omega_sq, gamma, kbT, t)
                    - 1j * t * av_energy_gap
                )
            else:
                lineshape_func[count1, 1] = (
                    second_order_lineshape_qm_t(freqs_gs, Omega_sq, gamma, kbT, t)
                    - 1j * t * av_energy_gap
                )
        else:
            if is_cl:
                lineshape_func[count1, 1] = (
                    second_order_lineshape_cl_t(freqs_gs, Omega_sq, gamma, kbT, t)
                    + 1j * t * av_energy_gap
                )
            else:
                lineshape_func[count1, 1] = (
                    second_order_lineshape_qm_t(freqs_gs, Omega_sq, gamma, kbT, t)
                    + 1j * t * av_energy_gap
                )

        count1 = count1 + 1
        stdout.write("%5d      %10.4f          %10.4e		%10.4e" % (count1,t*const.fs_to_Ha, np.real(lineshape_func[count1-1,1]), np.imag(lineshape_func[count1-1,1]))+'\n')
        t = t + step_length

    return lineshape_func


# third order cumulant lineshape
def full_third_order_lineshape(
    freqs_gs, Omega_sq, gamma, kbT, max_t, num_points, is_cl, four_phonon_term
,g3_cutoff,stdout):
    stdout.write('\n'+"Computing third order cumulant lineshape function."+'\n')
    stdout.write('\n'+'  Step       Time (fs)          Re[g_3]         Im[g_3]'+'\n')
    lineshape_func = np.zeros((num_points, 2), dtype=complex)
    step_length = max_t / num_points
    # only compute n_i_vec if this is a QM lineshape calculation:
    n_i_vec = np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount = 0
        while icount < freqs_gs.shape[0]:
            n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
            icount = icount + 1
    t = 0.0
    count1 = 0
    while count1 < num_points:
        lineshape_func[count1, 0] = t
        if is_cl:
            lineshape_func[count1, 1] = third_order_lineshape_cl_t(
                freqs_gs, Omega_sq, gamma, kbT, t, four_phonon_term
            )
        else:
            lineshape_func[count1, 1] = third_order_lineshape_qm_t(
                freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t, four_phonon_term
            )
        # apply cutoff to tame divergences for long t in 3rd order cumulant lineshape func
        if g3_cutoff>0.0:
                lineshape_func[count1,1]=lineshape_func[count1,1]*np.exp(-t/g3_cutoff)

        count1 = count1 + 1
        stdout.write("%5d      %10.4f          %10.4e       %10.4e" % (count1,t*const.fs_to_Ha, np.real(lineshape_func[count1-1,1]), np.imag(lineshape_func[count1-1,1]))+'\n')
        t = t + step_length

    return lineshape_func


# h1 factor necessary for computing 2DES in 3rd order cumulant:
def full_h1_func(
    freqs_gs, Omega_sq, gamma, kbT, max_t, num_points, is_cl, no_dusch, four_phonon_term
):
    h1_func = np.zeros((num_points, num_points, 3), dtype=complex)
    step_length = max_t / num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec = np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount = 0
        while icount < freqs_gs.shape[0]:
            n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
            icount = icount + 1

    t1 = 0.0
    count1 = 0
    print("COMPUTING H1")
    while count1 < num_points:
        count2 = 0
        t2 = 0.0
        print(count1)
        while count2 < num_points:
            h1_func[count1, count2, 0] = t1
            h1_func[count1, count2, 1] = t2

            if no_dusch:
                if is_cl:
                    h1_func[count1, count2, 2] = h1_func_cl_t_no_dusch(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2
                    )
                else:
                    h1_func[count1, count2, 2] = h1_func_qm_t_no_dusch(
                        freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2
                    )

            else:
                if is_cl:
                    h1_func[count1, count2, 2] = h1_func_cl_t(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term
                    )
                else:
                    h1_func[count1, count2, 2] = h1_func_qm_t(
                        freqs_gs,
                        Omega_sq,
                        n_i_vec,
                        gamma,
                        kbT,
                        t1,
                        t2,
                        four_phonon_term,
                    )

            count2 = count2 + 1
            t2 = t2 + step_length
        count1 = count1 + 1
        t1 = t1 + step_length
    return h1_func


# h2 factor necessary for computing 2DES in 3rd order cumulant:
def full_h2_func(
    freqs_gs, Omega_sq, gamma, kbT, max_t, num_points, is_cl, no_dusch, four_phonon_term
):
    h2_func = np.zeros((num_points, num_points, 3), dtype=complex)
    step_length = max_t / num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec = np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount = 0
        while icount < freqs_gs.shape[0]:
            n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
            icount = icount + 1

    t1 = 0.0
    count1 = 0
    print("COMPUTING H2")
    while count1 < num_points:
        count2 = 0
        t2 = 0.0
        print(count1)
        while count2 < num_points:
            h2_func[count1, count2, 0] = t1
            h2_func[count1, count2, 1] = t2

            if no_dusch:
                if is_cl:
                    h2_func[count1, count2, 2] = h2_func_cl_t_no_dusch(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2
                    )
                else:
                    h2_func[count1, count2, 2] = h2_func_qm_t_no_dusch(
                        freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2
                    )

            else:
                if is_cl:
                    h2_func[count1, count2, 2] = h2_func_cl_t(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term
                    )
                else:
                    h2_func[count1, count2, 2] = h2_func_qm_t(
                        freqs_gs,
                        Omega_sq,
                        n_i_vec,
                        gamma,
                        kbT,
                        t1,
                        t2,
                        four_phonon_term,
                    )

            count2 = count2 + 1
            t2 = t2 + step_length
        count1 = count1 + 1
        t1 = t1 + step_length
    return h2_func


# h4 factor necessary for computing 2DES in 3rd order cumulant:
def full_h4_func(
    freqs_gs, Omega_sq, gamma, kbT, max_t, num_points, is_cl, no_dusch, four_phonon_term
):
    h4_func = np.zeros((num_points, num_points, 3), dtype=complex)
    step_length = max_t / num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec = np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount = 0
        while icount < freqs_gs.shape[0]:
            n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
            icount = icount + 1

    t1 = 0.0
    count1 = 0
    print("COMPUTING H4")
    while count1 < num_points:
        count2 = 0
        t2 = 0.0
        print(count1)
        while count2 < num_points:
            h4_func[count1, count2, 0] = t1
            h4_func[count1, count2, 1] = t2
            if no_dusch:
                if is_cl:
                    h4_func[count1, count2, 2] = h4_func_cl_t_no_dusch(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2
                    )
                else:
                    h4_func[count1, count2, 2] = h4_func_qm_t_no_dusch(
                        freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2
                    )

            else:
                if is_cl:
                    h4_func[count1, count2, 2] = h4_func_cl_t(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term
                    )
                else:
                    h4_func[count1, count2, 2] = h4_func_qm_t(
                        freqs_gs,
                        Omega_sq,
                        n_i_vec,
                        gamma,
                        kbT,
                        t1,
                        t2,
                        four_phonon_term,
                    )
            count2 = count2 + 1
            t2 = t2 + step_length
        count1 = count1 + 1
        t1 = t1 + step_length
    return h4_func


# h5 factor necessary for computing 2DES in 3rd order cumulant:
def full_h5_func(
    freqs_gs, Omega_sq, gamma, kbT, max_t, num_points, is_cl, no_dusch, four_phonon_term
):
    h5_func = np.zeros((num_points, num_points, 3), dtype=complex)
    step_length = max_t / num_points
    # precompute n_i_vec. Only necessary for h1_func_qm
    n_i_vec = np.zeros(freqs_gs.shape[0])
    if not is_cl:
        icount = 0
        while icount < freqs_gs.shape[0]:
            n_i_vec[icount] = bose_einstein(freqs_gs[icount], kbT)
            icount = icount + 1

    t1 = 0.0
    count1 = 0
    print("COMPUTING H5")
    while count1 < num_points:
        count2 = 0
        t2 = 0.0
        print(count1)
        while count2 < num_points:
            h5_func[count1, count2, 0] = t1
            h5_func[count1, count2, 1] = t2
            if no_dusch:
                if is_cl:
                    h5_func[count1, count2, 2] = h5_func_cl_t_no_dusch(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2
                    )
                else:
                    h5_func[count1, count2, 2] = h5_func_qm_t_no_dusch(
                        freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2
                    )

            else:
                if is_cl:
                    h5_func[count1, count2, 2] = h5_func_cl_t(
                        freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term
                    )
                else:
                    h5_func[count1, count2, 2] = h5_func_qm_t(
                        freqs_gs,
                        Omega_sq,
                        n_i_vec,
                        gamma,
                        kbT,
                        t1,
                        t2,
                        four_phonon_term,
                    )

            count2 = count2 + 1
            t2 = t2 + step_length
        count1 = count1 + 1
        t1 = t1 + step_length
    return h5_func


@jit
def h2_func_cl_t_no_dusch(freqs_gs, Omega_sq, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            2.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[icount]) ** 2.0)
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]

        # term 1
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h2_cl(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = (
            const_fac
            * (kbT) ** 3.0
            / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount]) ** 2.0
        )
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h2_cl(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit
def h1_func_cl_t_no_dusch(freqs_gs, Omega_sq, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            2.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[icount]) ** 2.0)
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]

        # term 1
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h1_cl(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = (
            const_fac
            * (kbT) ** 3.0
            / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount]) ** 2.0
        )
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h1_cl(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit
def h5_func_cl_t_no_dusch(freqs_gs, Omega_sq, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            2.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[icount]) ** 2.0)
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]

        # term 1
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h5_cl(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = (
            const_fac
            * (kbT) ** 3.0
            / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount]) ** 2.0
        )
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h5_cl(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit
def h4_func_cl_t_no_dusch(freqs_gs, Omega_sq, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            2.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[icount]) ** 2.0)
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]

        # term 1
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + prefactor_2DES_h4_cl(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = (
            const_fac
            * (kbT) ** 3.0
            / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount]) ** 2.0
        )
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + prefactor_2DES_h4_cl(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit(fastmath=True, parallel=True)
def h3_func_cl_t_no_dusch(freqs_gs, Omega_sq, gamma, kbT, t1, t2, t3):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    icount = 0
    for icount in range(freqs_gs.shape[0]):
        const_fac = (
            2.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[icount]) ** 2.0)
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]

        # term 1
        gamma_term += prefactor_2DES_h3_cl(const_fac, omega_p, -omegai, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, -omega_m, omegai, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, omega_m, -omegai, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, -omega_p, omegai, kbT, t1, t2, t3)
        # term 2
        gamma_term += prefactor_2DES_h3_cl(const_fac, -omegai, omega_p, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, omegai, -omega_m, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, -omegai, omega_m, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, omegai, -omega_p, kbT, t1, t2, t3)
        # term 3
        gamma_term += prefactor_2DES_h3_cl(const_fac, omegaj, omegai, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, omegaj, -omegai, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, -omegaj, omegai, kbT, t1, t2, t3)
        gamma_term += prefactor_2DES_h3_cl(const_fac, -omegaj, -omegai, kbT, t1, t2, t3)

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = (
            const_fac
            * (kbT) ** 3.0
            / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount]) ** 2.0
        )
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        omega_term += prefactor_2DES_h3_cl(const_fac, -ik_m, ij_p, kbT, t1, t2, t3)
        omega_term += prefactor_2DES_h3_cl(const_fac, -ik_p, ij_p, kbT, t1, t2, t3)
        omega_term += prefactor_2DES_h3_cl(const_fac, ik_p, -ij_p, kbT, t1, t2, t3)
        omega_term += prefactor_2DES_h3_cl(const_fac, ik_m, -ij_p, kbT, t1, t2, t3)

        omega_term += prefactor_2DES_h3_cl(const_fac, -ik_m, ij_m, kbT, t1, t2, t3)
        omega_term += prefactor_2DES_h3_cl(const_fac, -ik_p, ij_m, kbT, t1, t2, t3)
        omega_term += prefactor_2DES_h3_cl(const_fac, ik_p, -ij_m, kbT, t1, t2, t3)
        omega_term += prefactor_2DES_h3_cl(const_fac, ik_m, -ij_m, kbT, t1, t2, t3)

    return omega_term + gamma_term


@jit(fastmath=True)
def calc_h3_time_domain(full_corr_func_3rd, t1_index, t2_index, t3_index):
    h_val = 0.0
    start_index = (full_corr_func_3rd.shape[0] - 1) / 2
    step_length = full_corr_func_3rd[1, 1, 0] - full_corr_func_3rd[0, 0, 0]

    if t1_index == 0 or t2_index == 0 or t3_index == 0:
        h_val = 0.0 + 0.0j
    else:
        tau3 = 0
        while tau3 < t3_index + 1:
            tau2 = 0
            while tau2 < t2_index + 1:
                tau1 = 0

                while tau1 < t1_index + 1:
                    eff_tau1 = tau1 + start_index
                    eff_tau2 = tau2 + start_index
                    eff_tau3 = tau3 + start_index
                    h_val = (
                        h_val
                        + (-1j)
                        * full_corr_func_3rd[
                            eff_tau2 - eff_tau1, eff_tau3 - eff_tau1, 2
                        ]
                    )

                    tau1 = tau1 + 1
                tau2 = tau2 + 1

            tau3 = tau3 + 1

        h_val = h_val * step_length ** 3.0

    return h_val


@njit(fastmath=True, parallel=True)
def h3_func_cl_t(freqs_gs, Omega_sq, gamma, kbT, t1, t2, t3, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    for icount in range(freqs_gs.shape[0]):
        for jcount in range(freqs_gs.shape[0]):
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            # term 1
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, omega_p, -omegai, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, -omega_m, omegai, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, omega_m, -omegai, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, -omega_p, omegai, kbT, t1, t2, t3
            )
            # term 2
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, -omegai, omega_p, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, omegai, -omega_m, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, -omegai, omega_m, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, omegai, -omega_p, kbT, t1, t2, t3
            )
            # term 3
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, omegaj, omegai, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, omegaj, -omegai, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, -omegaj, omegai, kbT, t1, t2, t3
            )
            gamma_term += prefactor_2DES_h3_cl(
                const_fac, -omegaj, -omegai, kbT, t1, t2, t3
            )

    # now do the more complicated term that is a sum over 3 indices:
    # only parallelize this part of the loop
    if four_phonon_term:
        for icount in range(freqs_gs.shape[0]):
            for jcount in range(freqs_gs.shape[0]):
                for kcount in range(freqs_gs.shape[0]):
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, -ik_m, ij_p, kbT, t1, t2, t3
                    )
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, -ik_p, ij_p, kbT, t1, t2, t3
                    )
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, ik_p, -ij_p, kbT, t1, t2, t3
                    )
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, ik_m, -ij_p, kbT, t1, t2, t3
                    )

                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, -ik_m, ij_m, kbT, t1, t2, t3
                    )
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, -ik_p, ij_m, kbT, t1, t2, t3
                    )
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, ik_p, -ij_m, kbT, t1, t2, t3
                    )
                    omega_term += prefactor_2DES_h3_cl(
                        const_fac, ik_m, -ij_m, kbT, t1, t2, t3
                    )

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h4_func_cl_t(freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            # term 1
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h4_cl(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h4_cl(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1
    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h5_func_cl_t(freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            # term 1
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h5_cl(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h5_cl(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h2_func_cl_t(freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            # term 1
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h2_cl(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h2_cl(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h1_func_cl_t(freqs_gs, Omega_sq, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            # term 1
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + prefactor_2DES_h1_cl(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + prefactor_2DES_h1_cl(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1
    corr_val = omega_term + gamma_term
    return corr_val


@jit(fastmath=True)
def third_order_lineshape_cl_t(freqs_gs, Omega_sq, gamma, kbT, t, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                2.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                * (kbT ** 2.0 / (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            # term 1
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, omega_p, -omegai, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, -omega_m, omegai, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, omega_m, -omegai, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, -omega_p, omegai, kbT, t
            )
            # term 2
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, -omegai, omega_p, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, omegai, -omega_m, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, -omegai, omega_m, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, omegai, -omega_p, kbT, t
            )
            # term 3
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, omegaj, omegai, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, omegaj, -omegai, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, -omegaj, omegai, kbT, t
            )
            gamma_term = gamma_term + prefactor_3rd_order_lineshape(
                const_fac, -omegaj, -omegai, kbT, t
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = (
                        const_fac
                        * (kbT) ** 3.0
                        / (freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount])
                        ** 2.0
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, -ik_m, ij_p, kbT, t
                    )
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, -ik_p, ij_p, kbT, t
                    )
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, ik_p, -ij_p, kbT, t
                    )
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, ik_m, -ij_p, kbT, t
                    )

                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, -ik_m, ij_m, kbT, t
                    )
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, -ik_p, ij_m, kbT, t
                    )
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, ik_p, -ij_m, kbT, t
                    )
                    omega_term = omega_term + prefactor_3rd_order_lineshape(
                        const_fac, ik_m, -ij_m, kbT, t
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h2_func_qm_t_no_dusch(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            / (2.0 * freqs_gs[icount] * freqs_gs[icount])
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0

        # term 1
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h2_QM(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h2_QM(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h2_QM(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h2_QM(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h2_QM(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h2_QM(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h2_QM(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h2_QM(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h2_QM(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h2_QM(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h2_QM(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h2_QM(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = const_fac / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount])
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        nk = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0
        nk_p = nk + 1.0

        omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h2_QM(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h2_QM(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h2_QM(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk * prefactor_2DES_h2_QM(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h2_QM(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h2_QM(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h2_QM(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h2_QM(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit
def h1_func_qm_t_no_dusch(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            / (2.0 * freqs_gs[icount] * freqs_gs[icount])
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0

        # term 1
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h1_QM(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h1_QM(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h1_QM(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h1_QM(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h1_QM(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h1_QM(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h1_QM(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h1_QM(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h1_QM(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h1_QM(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h1_QM(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h1_QM(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = const_fac / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount])
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        nk = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0
        nk_p = nk + 1.0

        omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h1_QM(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h1_QM(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h1_QM(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk * prefactor_2DES_h1_QM(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h1_QM(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h1_QM(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h1_QM(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h1_QM(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit
def h5_func_qm_t_no_dusch(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            / (2.0 * freqs_gs[icount] * freqs_gs[icount])
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0

        # term 1
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h5_QM(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h5_QM(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h5_QM(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h5_QM(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h5_QM(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h5_QM(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h5_QM(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h5_QM(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h5_QM(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h5_QM(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h5_QM(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h5_QM(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = const_fac / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount])
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        nk = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0
        nk_p = nk + 1.0

        omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h5_QM(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h5_QM(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h5_QM(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk * prefactor_2DES_h5_QM(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h5_QM(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h5_QM(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h5_QM(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h5_QM(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@jit
def h4_func_qm_t_no_dusch(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            / (2.0 * freqs_gs[icount] * freqs_gs[icount])
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0

        # term 1
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h4_QM(
            const_fac, omega_p, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h4_QM(
            const_fac, -omega_m, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h4_QM(
            const_fac, omega_m, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h4_QM(
            const_fac, -omega_p, omegai, kbT, t1, t2
        )
        # term 2
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h4_QM(
            const_fac, -omegai, omega_p, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h4_QM(
            const_fac, omegai, -omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h4_QM(
            const_fac, -omegai, omega_m, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h4_QM(
            const_fac, omegai, -omega_p, kbT, t1, t2
        )
        # term 3
        gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h4_QM(
            const_fac, omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h4_QM(
            const_fac, omegaj, -omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h4_QM(
            const_fac, -omegaj, omegai, kbT, t1, t2
        )
        gamma_term = gamma_term + ni * nj * prefactor_2DES_h4_QM(
            const_fac, -omegaj, -omegai, kbT, t1, t2
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = const_fac / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount])
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        nk = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0
        nk_p = nk + 1.0

        omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h4_QM(
            const_fac, -ik_m, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h4_QM(
            const_fac, -ik_p, ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h4_QM(
            const_fac, ik_p, -ij_p, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj * nk * prefactor_2DES_h4_QM(
            const_fac, ik_m, -ij_p, kbT, t1, t2
        )

        omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h4_QM(
            const_fac, -ik_m, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h4_QM(
            const_fac, -ik_p, ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h4_QM(
            const_fac, ik_p, -ij_m, kbT, t1, t2
        )
        omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h4_QM(
            const_fac, ik_m, -ij_m, kbT, t1, t2
        )

        icount = icount + 1

    return omega_term + gamma_term


@njit(fastmath=True, parallel=True)
def h3_func_qm_t_no_dusch(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, t3):
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    for icount in range(freqs_gs.shape[0]):
        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * gamma[icount]
            * gamma[icount]
            / (2.0 * freqs_gs[icount] * freqs_gs[icount])
        )
        omega_p = freqs_gs[icount] + freqs_gs[icount]
        omega_m = freqs_gs[icount] - freqs_gs[icount]
        omegai = freqs_gs[icount]
        omegaj = freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0

        # term 1
        gamma_term += (
            ni
            * nj_p
            * prefactor_2DES_h3_QM(const_fac, omega_p, -omegai, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni_p
            * nj_p
            * prefactor_2DES_h3_QM(const_fac, -omega_m, omegai, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni * nj * prefactor_2DES_h3_QM(const_fac, omega_m, -omegai, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni_p
            * nj
            * prefactor_2DES_h3_QM(const_fac, -omega_p, omegai, kbT, t1, t2, t3)
        )
        # term 2
        gamma_term += (
            ni_p
            * nj_p
            * prefactor_2DES_h3_QM(const_fac, -omegai, omega_p, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni
            * nj_p
            * prefactor_2DES_h3_QM(const_fac, omegai, -omega_m, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni_p
            * nj
            * prefactor_2DES_h3_QM(const_fac, -omegai, omega_m, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni * nj * prefactor_2DES_h3_QM(const_fac, omegai, -omega_p, kbT, t1, t2, t3)
        )
        # term 3
        gamma_term += (
            ni_p
            * nj_p
            * prefactor_2DES_h3_QM(const_fac, omegaj, omegai, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni
            * nj_p
            * prefactor_2DES_h3_QM(const_fac, omegaj, -omegai, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni_p
            * nj
            * prefactor_2DES_h3_QM(const_fac, -omegaj, omegai, kbT, t1, t2, t3)
        )
        gamma_term += (
            ni * nj * prefactor_2DES_h3_QM(const_fac, -omegaj, -omegai, kbT, t1, t2, t3)
        )

        const_fac = (
            4.0
            * math.pi ** 2.0
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
            * Omega_sq[icount, icount]
        )
        const_fac = const_fac / (freqs_gs[icount] * freqs_gs[icount] * freqs_gs[icount])
        ik_p = freqs_gs[icount] + freqs_gs[icount]
        ik_m = freqs_gs[icount] - freqs_gs[icount]
        ij_p = freqs_gs[icount] + freqs_gs[icount]
        ij_m = freqs_gs[icount] - freqs_gs[icount]
        ni = n_i_vec[icount]
        nj = n_i_vec[icount]
        nk = n_i_vec[icount]
        ni_p = ni + 1.0
        nj_p = nj + 1.0
        nk_p = nk + 1.0

        omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h3_QM(
            const_fac, -ik_m, ij_p, kbT, t1, t2, t3
        )
        omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h3_QM(
            const_fac, -ik_p, ij_p, kbT, t1, t2, t3
        )
        omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h3_QM(
            const_fac, ik_p, -ij_p, kbT, t1, t2, t3
        )
        omega_term = omega_term + ni * nj * nk * prefactor_2DES_h3_QM(
            const_fac, ik_m, -ij_p, kbT, t1, t2, t3
        )

        omega_term += (
            ni_p
            * nj
            * nk_p
            * prefactor_2DES_h3_QM(const_fac, -ik_m, ij_m, kbT, t1, t2, t3)
        )
        omega_term += (
            ni_p
            * nj
            * nk
            * prefactor_2DES_h3_QM(const_fac, -ik_p, ij_m, kbT, t1, t2, t3)
        )
        omega_term += (
            ni
            * nj_p
            * nk_p
            * prefactor_2DES_h3_QM(const_fac, ik_p, -ij_m, kbT, t1, t2, t3)
        )
        omega_term += (
            ni
            * nj_p
            * nk
            * prefactor_2DES_h3_QM(const_fac, ik_m, -ij_m, kbT, t1, t2, t3)
        )

    return omega_term + gamma_term


@njit(fastmath=True, parallel=True)
def h3_func_qm_t_fast(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, t3):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j

    # precompute the exponentials. They are expensive to evaluate. Only has to be done once for the entire routine
    expt1 = np.exp(1j * t1 * freqs_gs)
    expt2 = np.exp(1j * t2 * freqs_gs)
    expt3 = np.exp(1j * t3 * freqs_gs)

    # start with gamma term first:
    for icount in range(freqs_gs.shape[0]):
        for jcount in range(freqs_gs.shape[0]):
            const_fac = (
                4.0
                * math.pi
                * math.pi
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # prefactor_2DES_h3_QM_fast(const_fac,omega1,omega2,kbT,expw1t2,exptw2t1,expw1w2t3,expw1t3,expw2t3,t1,t2,t3)
            # we need the following terms: e^(-iomega1*t2),e^(-iomega2*t1),e^(i(omega1+omega2)*t3)

            # term 1 old:
            # gamma_term+=ni*nj_p*prefactor_2DES_h3_QM(const_fac,omega_p,-omegai,kbT,t1,t2,t3)
            # gamma_term+=ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,-omega_m,omegai,kbT,t1,t2,t3)
            # gamma_term+=ni*nj*prefactor_2DES_h3_QM(const_fac,omega_m,-omegai,kbT,t1,t2,t3)
            # gamma_term+=ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omega_p,omegai,kbT,t1,t2,t3)

            # term2 old:
            # gamma_term+=ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,-omegai,omega_p,kbT,t1,t2,t3)
            # gamma_term+=ni*nj_p*prefactor_2DES_h3_QM(const_fac,omegai,-omega_m,kbT,t1,t2,t3)
            # gamma_term+=ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omegai,omega_m,kbT,t1,t2,t3)
            # gamma_term+=ni*nj*prefactor_2DES_h3_QM(const_fac,omegai,-omega_p,kbT,t1,t2,t3)

            # term3 old:
            # gamma_term+=ni_p*nj_p*prefactor_2DES_h3_QM(const_fac,omegaj,omegai,kbT,t1,t2,t3)
            # gamma_term+=ni*nj_p*prefactor_2DES_h3_QM(const_fac,omegaj,-omegai,kbT,t1,t2,t3)
            # gamma_term+=ni_p*nj*prefactor_2DES_h3_QM(const_fac,-omegaj,omegai,kbT,t1,t2,t3)
            # gamma_term+=ni*nj*prefactor_2DES_h3_QM(const_fac,-omegaj,-omegai,kbT,t1,t2,t3)

            # term1 new
            gamma_term += (
                ni
                * nj_p
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    omega_p,
                    -omegai,
                    kbT,
                    expt2[icount] * expt2[jcount],
                    np.conj(expt1[icount]),
                    expt3[icount] * expt3[jcount] * np.conj(expt3[icount]),
                    expt3[icount] * expt3[jcount],
                    np.conj(expt3[icount]),
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni_p
                * nj_p
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    -omega_m,
                    omegai,
                    kbT,
                    np.conj(expt2[icount]) * expt2[jcount],
                    expt1[icount],
                    np.conj(expt3[icount]) * expt3[jcount] * expt3[icount],
                    np.conj(expt3[icount]) * expt3[jcount],
                    expt3[icount],
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni
                * nj
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    omega_m,
                    -omegai,
                    kbT,
                    expt2[icount] * np.conj(expt2[jcount]),
                    np.conj(expt1[icount]),
                    expt3[icount] * np.conj(expt3[jcount]) * np.conj(expt3[icount]),
                    expt3[icount] * np.conj(expt3[jcount]),
                    np.conj(expt3[icount]),
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni_p
                * nj
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    -omega_p,
                    omegai,
                    kbT,
                    np.conj(expt2[icount]) * np.conj(expt2[jcount]),
                    expt1[icount],
                    np.conj(expt3[icount]) * np.conj(expt3[jcount]) * expt3[icount],
                    np.conj(expt3[icount]) * np.conj(expt3[jcount]),
                    expt3[icount],
                    t1,
                    t2,
                    t3,
                )
            )

            # term 2 new
            gamma_term += (
                ni_p
                * nj_p
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    -omegai,
                    omega_p,
                    kbT,
                    np.conj(expt2[icount]),
                    expt1[icount] * expt2[jcount],
                    np.conj(expt3[icount]) * expt3[icount] * expt3[jcount],
                    np.conj(expt3[icount]),
                    expt3[icount] * expt3[jcount],
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni
                * nj_p
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    omegai,
                    -omega_m,
                    kbT,
                    expt2[icount],
                    np.conj(expt1[icount]) * expt2[jcount],
                    expt3[icount] * np.conj(expt3[icount]) * expt3[jcount],
                    expt3[icount],
                    np.conj(expt3[icount]) * expt3[jcount],
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni_p
                * nj
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    -omegai,
                    omega_m,
                    kbT,
                    np.conj(expt2[icount]),
                    expt1[icount] * np.conj(expt2[jcount]),
                    np.conj(expt3[icount]) * expt3[icount] * np.conj(expt3[jcount]),
                    np.conj(expt3[icount]),
                    expt3[icount] * np.conj(expt3[jcount]),
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni
                * nj
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    omegai,
                    -omega_p,
                    kbT,
                    expt2[icount],
                    np.conj(expt1[icount]) * np.conj(expt2[jcount]),
                    expt3[icount] * np.conj(expt3[icount]) * np.conj(expt3[jcount]),
                    expt3[icount],
                    np.conj(expt3[icount]) * np.conj(expt3[jcount]),
                    t1,
                    t2,
                    t3,
                )
            )

            # term 3 new
            gamma_term += (
                ni_p
                * nj_p
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    omegaj,
                    omegai,
                    kbT,
                    expt2[jcount],
                    expt1[icount],
                    expt3[jcount] * expt3[icount],
                    expt3[jcount],
                    expt3[icount],
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni
                * nj_p
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    omegaj,
                    -omegai,
                    kbT,
                    expt2[jcount],
                    np.conj(expt1[icount]),
                    expt3[jcount] * np.conj(expt3[icount]),
                    expt3[jcount],
                    np.conj(expt3[icount]),
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni_p
                * nj
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    -omegaj,
                    omegai,
                    kbT,
                    np.conj(expt2[jcount]),
                    expt1[icount],
                    np.conj(expt3[jcount]) * expt3[icount],
                    np.conj(expt3[jcount]),
                    expt3[icount],
                    t1,
                    t2,
                    t3,
                )
            )
            gamma_term += (
                ni
                * nj
                * prefactor_2DES_h3_QM_fast(
                    const_fac,
                    -omegaj,
                    -omegai,
                    kbT,
                    np.conj(expt2[jcount]),
                    np.conj(expt1[icount]),
                    np.conj(expt3[jcount]) * np.conj(expt3[icount]),
                    np.conj(expt3[jcount]),
                    np.conj(expt3[icount]),
                    t1,
                    t2,
                    t3,
                )
            )

    # now do the more complicated term that is a sum over 3 indices:
    # only parallelize the outer loop here. Inner loop is a nested double loop so worth the effort
    for icount in range(freqs_gs.shape[0]):
        for jcount in range(freqs_gs.shape[0]):
            for kcount in range(freqs_gs.shape[0]):
                const_fac = (
                    4.0
                    * math.pi
                    * math.pi
                    * Omega_sq[icount, jcount]
                    * Omega_sq[jcount, kcount]
                    * Omega_sq[icount, kcount]
                )
                const_fac = const_fac / (
                    freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                )
                ik_p = freqs_gs[icount] + freqs_gs[kcount]
                ik_m = freqs_gs[icount] - freqs_gs[kcount]
                ij_p = freqs_gs[icount] + freqs_gs[jcount]
                ij_m = freqs_gs[icount] - freqs_gs[jcount]
                ni = n_i_vec[icount]
                nj = n_i_vec[jcount]
                nk = n_i_vec[kcount]
                ni_p = ni + 1.0
                nj_p = nj + 1.0
                nk_p = nk + 1.0

                # we need the following terms: e^(-iomega1*t2),e^(-iomega2*t1),e^(i(omega1+omega2)*t3)

                # Term 1 old
                # omega_term+=ni_p*nj_p*nk_p*prefactor_2DES_h3_QM(const_fac,-ik_m,ij_p,kbT,t1,t2,t3)
                # omega_term+=ni_p*nj_p*nk*prefactor_2DES_h3_QM(const_fac,-ik_p,ij_p,kbT,t1,t2,t3)
                # omega_term+=ni*nj*nk_p*prefactor_2DES_h3_QM(const_fac,ik_p,-ij_p,kbT,t1,t2,t3)
                # omega_term+=ni*nj*nk*prefactor_2DES_h3_QM(const_fac,ik_m,-ij_p,kbT,t1,t2,t3)

                # Term 2 old
                # omega_term+=ni_p*nj*nk_p*prefactor_2DES_h3_QM(const_fac,-ik_m,ij_m,kbT,t1,t2,t3)
                # omega_term+=ni_p*nj*nk*prefactor_2DES_h3_QM(const_fac,-ik_p,ij_m,kbT,t1,t2,t3)
                # omega_term+=ni*nj_p*nk_p*prefactor_2DES_h3_QM(const_fac,ik_p,-ij_m,kbT,t1,t2,t3)
                # omega_term+=ni*nj_p*nk*prefactor_2DES_h3_QM(const_fac,ik_m,-ij_m,kbT,t1,t2,t3)

                # Term 1 new
                omega_term += (
                    ni_p
                    * nj_p
                    * nk_p
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        -ik_m,
                        ij_p,
                        kbT,
                        np.conj(expt2[icount]) * expt2[kcount],
                        expt1[icount] * expt1[jcount],
                        np.conj(expt3[icount])
                        * expt3[kcount]
                        * expt3[icount]
                        * expt3[jcount],
                        np.conj(expt3[icount]) * expt3[kcount],
                        expt3[icount] * expt3[jcount],
                        t1,
                        t2,
                        t3,
                    )
                )
                omega_term += (
                    ni_p
                    * nj_p
                    * nk
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        -ik_p,
                        ij_p,
                        kbT,
                        np.conj(expt2[icount]) * np.conj(expt2[kcount]),
                        expt1[icount] * expt1[jcount],
                        np.conj(expt3[icount])
                        * np.conj(expt3[kcount])
                        * expt3[icount]
                        * expt3[jcount],
                        np.conj(expt3[icount]) * np.conj(expt3[kcount]),
                        expt3[icount] * expt3[jcount],
                        t1,
                        t2,
                        t3,
                    )
                )
                omega_term += (
                    ni
                    * nj
                    * nk_p
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        ik_p,
                        -ij_p,
                        kbT,
                        expt2[icount] * expt2[kcount],
                        np.conj(expt1[icount]) * np.conj(expt1[jcount]),
                        expt3[icount]
                        * expt3[kcount]
                        * np.conj(expt3[icount])
                        * np.conj(expt3[jcount]),
                        expt3[icount] * expt3[kcount],
                        np.conj(expt3[icount]) * np.conj(expt3[jcount]),
                        t1,
                        t2,
                        t3,
                    )
                )
                omega_term += (
                    ni
                    * nj
                    * nk
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        ik_m,
                        -ij_p,
                        kbT,
                        expt2[icount] * np.conj(expt2[kcount]),
                        np.conj(expt1[icount]) * np.conj(expt1[jcount]),
                        np.conj(expt3[icount] * expt3[kcount])
                        * np.conj(expt3[icount])
                        * np.conj(expt3[jcount]),
                        expt3[icount] * np.conj(expt3[kcount]),
                        np.conj(expt3[icount]) * np.conj(expt3[jcount]),
                        t1,
                        t2,
                        t3,
                    )
                )

                # Term 2
                omega_term += (
                    ni_p
                    * nj
                    * nk_p
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        -ik_m,
                        ij_m,
                        kbT,
                        np.conj(expt2[icount]) * expt2[kcount],
                        expt1[icount] * np.conj(expt1[jcount]),
                        np.conj(expt3[icount])
                        * expt3[kcount]
                        * expt3[icount]
                        * np.conj(expt3[jcount]),
                        np.conj(expt3[icount]) * expt3[kcount],
                        expt3[icount] * np.conj(expt3[jcount]),
                        t1,
                        t2,
                        t3,
                    )
                )
                omega_term += (
                    ni_p
                    * nj
                    * nk
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        -ik_p,
                        ij_m,
                        kbT,
                        np.conj(expt2[icount]) * np.conj(expt2[kcount]),
                        expt1[icount] * np.conj(expt1[jcount]),
                        np.conj(expt3[icount])
                        * np.conj(expt3[kcount])
                        * expt3[icount]
                        * np.conj(expt3[jcount]),
                        np.conj(expt3[icount]) * np.conj(expt3[kcount]),
                        expt3[icount] * np.conj(expt3[jcount]),
                        t1,
                        t2,
                        t3,
                    )
                )
                omega_term += (
                    ni
                    * nj_p
                    * nk_p
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        ik_p,
                        -ij_m,
                        kbT,
                        expt2[icount] * expt2[kcount],
                        np.conj(expt1[icount]) * expt1[jcount],
                        expt3[icount]
                        * expt3[kcount]
                        * np.conj(expt3[icount])
                        * expt3[jcount],
                        expt3[icount] * expt3[kcount],
                        np.conj(expt3[icount]) * expt3[jcount],
                        t1,
                        t2,
                        t3,
                    )
                )
                omega_term += (
                    ni
                    * nj_p
                    * nk
                    * prefactor_2DES_h3_QM_fast(
                        const_fac,
                        ik_m,
                        -ij_m,
                        kbT,
                        expt2[icount] * np.conj(expt2[kcount]),
                        np.conj(expt1[icount]) * expt1[jcount],
                        expt3[icount]
                        * np.conj(expt3[kcount])
                        * np.conj(expt3[icount])
                        * expt3[jcount],
                        expt3[icount] * np.conj(expt3[kcount]),
                        np.conj(expt3[icount]) * expt3[jcount],
                        t1,
                        t2,
                        t3,
                    )
                )

    corr_val = omega_term + gamma_term
    return corr_val


@njit(fastmath=True, parallel=True)
def h3_func_qm_t(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, t3, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j

    # start with gamma term first:
    for icount in range(freqs_gs.shape[0]):
        for jcount in range(freqs_gs.shape[0]):
            const_fac = (
                4.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]

            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # we need the following terms: e^(-iomega1*t2),e^(-iomega2*t1),e^(i(omega1+omega2)*t3)

            # term 1
            gamma_term += (
                ni
                * nj_p
                * prefactor_2DES_h3_QM(const_fac, omega_p, -omegai, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni_p
                * nj_p
                * prefactor_2DES_h3_QM(const_fac, -omega_m, omegai, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni
                * nj
                * prefactor_2DES_h3_QM(const_fac, omega_m, -omegai, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni_p
                * nj
                * prefactor_2DES_h3_QM(const_fac, -omega_p, omegai, kbT, t1, t2, t3)
            )
            # term 2
            gamma_term += (
                ni_p
                * nj_p
                * prefactor_2DES_h3_QM(const_fac, -omegai, omega_p, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni
                * nj_p
                * prefactor_2DES_h3_QM(const_fac, omegai, -omega_m, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni_p
                * nj
                * prefactor_2DES_h3_QM(const_fac, -omegai, omega_m, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni
                * nj
                * prefactor_2DES_h3_QM(const_fac, omegai, -omega_p, kbT, t1, t2, t3)
            )
            # term 3
            gamma_term += (
                ni_p
                * nj_p
                * prefactor_2DES_h3_QM(const_fac, omegaj, omegai, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni
                * nj_p
                * prefactor_2DES_h3_QM(const_fac, omegaj, -omegai, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni_p
                * nj
                * prefactor_2DES_h3_QM(const_fac, -omegaj, omegai, kbT, t1, t2, t3)
            )
            gamma_term += (
                ni
                * nj
                * prefactor_2DES_h3_QM(const_fac, -omegaj, -omegai, kbT, t1, t2, t3)
            )

    # now do the more complicated term that is a sum over 3 indices:
    # only parallelize the outer loop here. Inner loop is a nested double loop so worth the effort
    if four_phonon_term:
        for icount in range(freqs_gs.shape[0]):
            for jcount in range(freqs_gs.shape[0]):
                for kcount in range(freqs_gs.shape[0]):
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ni = n_i_vec[icount]
                    nj = n_i_vec[jcount]
                    nk = n_i_vec[kcount]
                    ni_p = ni + 1.0
                    nj_p = nj + 1.0
                    nk_p = nk + 1.0

                    # we need the following terms: e^(-iomega1*t2),e^(-iomega2*t1),e^(i(omega1+omega2)*t3)

                    omega_term += (
                        ni_p
                        * nj_p
                        * nk_p
                        * prefactor_2DES_h3_QM(const_fac, -ik_m, ij_p, kbT, t1, t2, t3)
                    )
                    omega_term += (
                        ni_p
                        * nj_p
                        * nk
                        * prefactor_2DES_h3_QM(const_fac, -ik_p, ij_p, kbT, t1, t2, t3)
                    )
                    omega_term += (
                        ni
                        * nj
                        * nk_p
                        * prefactor_2DES_h3_QM(const_fac, ik_p, -ij_p, kbT, t1, t2, t3)
                    )
                    omega_term += (
                        ni
                        * nj
                        * nk
                        * prefactor_2DES_h3_QM(const_fac, ik_m, -ij_p, kbT, t1, t2, t3)
                    )

                    omega_term += (
                        ni_p
                        * nj
                        * nk_p
                        * prefactor_2DES_h3_QM(const_fac, -ik_m, ij_m, kbT, t1, t2, t3)
                    )
                    omega_term += (
                        ni_p
                        * nj
                        * nk
                        * prefactor_2DES_h3_QM(const_fac, -ik_p, ij_m, kbT, t1, t2, t3)
                    )
                    omega_term += (
                        ni
                        * nj_p
                        * nk_p
                        * prefactor_2DES_h3_QM(const_fac, ik_p, -ij_m, kbT, t1, t2, t3)
                    )
                    omega_term += (
                        ni
                        * nj_p
                        * nk
                        * prefactor_2DES_h3_QM(const_fac, ik_m, -ij_m, kbT, t1, t2, t3)
                    )

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h5_func_qm_t(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                4.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # term 1
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h5_QM(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h5_QM(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h5_QM(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h5_QM(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h5_QM(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h5_QM(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h5_QM(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h5_QM(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h5_QM(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h5_QM(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h5_QM(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h5_QM(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ni = n_i_vec[icount]
                    nj = n_i_vec[jcount]
                    nk = n_i_vec[kcount]
                    ni_p = ni + 1.0
                    nj_p = nj + 1.0
                    nk_p = nk + 1.0

                    omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h5_QM(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h5_QM(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h5_QM(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk * prefactor_2DES_h5_QM(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h5_QM(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h5_QM(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h5_QM(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h5_QM(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1
    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h4_func_qm_t(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                4.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # term 1
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h4_QM(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h4_QM(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h4_QM(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h4_QM(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h4_QM(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h4_QM(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h4_QM(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h4_QM(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h4_QM(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h4_QM(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h4_QM(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h4_QM(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ni = n_i_vec[icount]
                    nj = n_i_vec[jcount]
                    nk = n_i_vec[kcount]
                    ni_p = ni + 1.0
                    nj_p = nj + 1.0
                    nk_p = nk + 1.0

                    omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h4_QM(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h4_QM(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h4_QM(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk * prefactor_2DES_h4_QM(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h4_QM(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h4_QM(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h4_QM(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h4_QM(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h2_func_qm_t(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                4.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # term 1
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h2_QM(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h2_QM(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h2_QM(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h2_QM(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h2_QM(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h2_QM(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h2_QM(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h2_QM(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h2_QM(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h2_QM(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h2_QM(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h2_QM(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ni = n_i_vec[icount]
                    nj = n_i_vec[jcount]
                    nk = n_i_vec[kcount]
                    ni_p = ni + 1.0
                    nj_p = nj + 1.0
                    nk_p = nk + 1.0

                    omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h2_QM(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h2_QM(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h2_QM(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk * prefactor_2DES_h2_QM(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h2_QM(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h2_QM(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h2_QM(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h2_QM(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val


@jit
def h1_func_qm_t(freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t1, t2, four_phonon_term):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_fac = (
                4.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # term 1
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h1_QM(
                const_fac, omega_p, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h1_QM(
                const_fac, -omega_m, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h1_QM(
                const_fac, omega_m, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h1_QM(
                const_fac, -omega_p, omegai, kbT, t1, t2
            )
            # term 2
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h1_QM(
                const_fac, -omegai, omega_p, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h1_QM(
                const_fac, omegai, -omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h1_QM(
                const_fac, -omegai, omega_m, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h1_QM(
                const_fac, omegai, -omega_p, kbT, t1, t2
            )
            # term 3
            gamma_term = gamma_term + ni_p * nj_p * prefactor_2DES_h1_QM(
                const_fac, omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_2DES_h1_QM(
                const_fac, omegaj, -omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_2DES_h1_QM(
                const_fac, -omegaj, omegai, kbT, t1, t2
            )
            gamma_term = gamma_term + ni * nj * prefactor_2DES_h1_QM(
                const_fac, -omegaj, -omegai, kbT, t1, t2
            )
            jcount = jcount + 1
        icount = icount + 1

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        icount = 0
        while icount < freqs_gs.shape[0]:
            jcount = 0
            while jcount < freqs_gs.shape[0]:
                kcount = 0
                while kcount < freqs_gs.shape[0]:
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ni = n_i_vec[icount]
                    nj = n_i_vec[jcount]
                    nk = n_i_vec[kcount]
                    ni_p = ni + 1.0
                    nj_p = nj + 1.0
                    nk_p = nk + 1.0

                    omega_term = omega_term + ni_p * nj_p * nk_p * prefactor_2DES_h1_QM(
                        const_fac, -ik_m, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj_p * nk * prefactor_2DES_h1_QM(
                        const_fac, -ik_p, ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk_p * prefactor_2DES_h1_QM(
                        const_fac, ik_p, -ij_p, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj * nk * prefactor_2DES_h1_QM(
                        const_fac, ik_m, -ij_p, kbT, t1, t2
                    )

                    omega_term = omega_term + ni_p * nj * nk_p * prefactor_2DES_h1_QM(
                        const_fac, -ik_m, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni_p * nj * nk * prefactor_2DES_h1_QM(
                        const_fac, -ik_p, ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk_p * prefactor_2DES_h1_QM(
                        const_fac, ik_p, -ij_m, kbT, t1, t2
                    )
                    omega_term = omega_term + ni * nj_p * nk * prefactor_2DES_h1_QM(
                        const_fac, ik_m, -ij_m, kbT, t1, t2
                    )

                    kcount = kcount + 1
                jcount = jcount + 1
            icount = icount + 1
    corr_val = omega_term + gamma_term

    return corr_val


@jit(fastmath=True)
def third_order_lineshape_qm_t(
    freqs_gs, Omega_sq, n_i_vec, gamma, kbT, t, four_phonon_term
):
    corr_val = 0.0 + 0.0j
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    for icount in range(freqs_gs.shape[0]):
        for jcount in range(freqs_gs.shape[0]):
            const_fac = (
                4.0
                * math.pi ** 2.0
                * Omega_sq[icount, jcount]
                * gamma[icount]
                * gamma[jcount]
                / (2.0 * freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]
            omegai = freqs_gs[icount]
            omegaj = freqs_gs[jcount]
            ni = n_i_vec[icount]
            nj = n_i_vec[jcount]
            ni_p = ni + 1.0
            nj_p = nj + 1.0

            # term 1
            gamma_term = gamma_term + ni * nj_p * prefactor_3rd_order_lineshape_QM(
                const_fac, omega_p, -omegai, kbT, t
            )
            gamma_term = gamma_term + ni_p * nj_p * prefactor_3rd_order_lineshape_QM(
                const_fac, -omega_m, omegai, kbT, t
            )
            gamma_term = gamma_term + ni * nj * prefactor_3rd_order_lineshape_QM(
                const_fac, omega_m, -omegai, kbT, t
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_3rd_order_lineshape_QM(
                const_fac, -omega_p, omegai, kbT, t
            )
            # term 2
            gamma_term = gamma_term + ni_p * nj_p * prefactor_3rd_order_lineshape_QM(
                const_fac, -omegai, omega_p, kbT, t
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_3rd_order_lineshape_QM(
                const_fac, omegai, -omega_m, kbT, t
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_3rd_order_lineshape_QM(
                const_fac, -omegai, omega_m, kbT, t
            )
            gamma_term = gamma_term + ni * nj * prefactor_3rd_order_lineshape_QM(
                const_fac, omegai, -omega_p, kbT, t
            )
            # term 3
            gamma_term = gamma_term + ni_p * nj_p * prefactor_3rd_order_lineshape_QM(
                const_fac, omegaj, omegai, kbT, t
            )
            gamma_term = gamma_term + ni * nj_p * prefactor_3rd_order_lineshape_QM(
                const_fac, omegaj, -omegai, kbT, t
            )
            gamma_term = gamma_term + ni_p * nj * prefactor_3rd_order_lineshape_QM(
                const_fac, -omegaj, omegai, kbT, t
            )
            gamma_term = gamma_term + ni * nj * prefactor_3rd_order_lineshape_QM(
                const_fac, -omegaj, -omegai, kbT, t
            )

    # now do the more complicated term that is a sum over 3 indices:
    if four_phonon_term:
        for icount in range(freqs_gs.shape[0]):
            for jcount in range(freqs_gs.shape[0]):
                for kcount in range(freqs_gs.shape[0]):
                    const_fac = (
                        4.0
                        * math.pi ** 2.0
                        * Omega_sq[icount, jcount]
                        * Omega_sq[jcount, kcount]
                        * Omega_sq[icount, kcount]
                    )
                    const_fac = const_fac / (
                        freqs_gs[icount] * freqs_gs[jcount] * freqs_gs[kcount]
                    )
                    ik_p = freqs_gs[icount] + freqs_gs[kcount]
                    ik_m = freqs_gs[icount] - freqs_gs[kcount]
                    ij_p = freqs_gs[icount] + freqs_gs[jcount]
                    ij_m = freqs_gs[icount] - freqs_gs[jcount]
                    ni = n_i_vec[icount]
                    nj = n_i_vec[jcount]
                    nk = n_i_vec[kcount]
                    ni_p = ni + 1.0
                    nj_p = nj + 1.0
                    nk_p = nk + 1.0

                    omega_term = (
                        omega_term
                        + ni_p
                        * nj_p
                        * nk_p
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, -ik_m, ij_p, kbT, t
                        )
                    )
                    omega_term = (
                        omega_term
                        + ni_p
                        * nj_p
                        * nk
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, -ik_p, ij_p, kbT, t
                        )
                    )
                    omega_term = (
                        omega_term
                        + ni
                        * nj
                        * nk_p
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, ik_p, -ij_p, kbT, t
                        )
                    )
                    omega_term = (
                        omega_term
                        + ni
                        * nj
                        * nk
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, ik_m, -ij_p, kbT, t
                        )
                    )

                    omega_term = (
                        omega_term
                        + ni_p
                        * nj
                        * nk_p
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, -ik_m, ij_m, kbT, t
                        )
                    )
                    omega_term = (
                        omega_term
                        + ni_p
                        * nj
                        * nk
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, -ik_p, ij_m, kbT, t
                        )
                    )
                    omega_term = (
                        omega_term
                        + ni
                        * nj_p
                        * nk_p
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, ik_p, -ij_m, kbT, t
                        )
                    )
                    omega_term = (
                        omega_term
                        + ni
                        * nj_p
                        * nk
                        * prefactor_3rd_order_lineshape_QM(
                            const_fac, ik_m, -ij_m, kbT, t
                        )
                    )

    corr_val = omega_term + gamma_term
    return corr_val


@njit(fastmath=True)
def prefactor_3rd_order_lineshape_QM(const_fac, omega1, omega2, kbT, t):
    val = const_fac / (4.0 * math.pi ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega12=0 and omega2=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = -val / 6.0 * 1j * t ** 3.0  # check this limit!!!!
    elif abs(omega12) < tol:
        num = (
            1j * omega1 * t
            - t ** 2.0 * omega1 ** 2.0 / 2.0
            + 1.0
            - cmath.exp(1j * omega1 * t)
        )  # double check. Is the limit correct?
        denom = omega1 ** 2.0 * omega2
        full_contribution = val * num / denom
    elif abs(omega1) < tol:
        num = 2.0 * (cmath.exp(-1j * omega2 * t) - 1.0) + 1j * omega2 * t * (
            cmath.exp(-1j * omega2 * t) + 1.0
        )
        denom = omega2 * omega2 * omega12
        full_contribution = val * num / denom

    elif abs(omega2) < tol:  # double check the t^2 sign here as well
        num = (
            -1j * omega1 * t
            + (1.0 - cmath.exp(-1j * omega1 * t))
            - 1.0 / 2.0 * omega1 ** 2.0 * t ** 2.0
        )
        denom = omega1 * omega12 * omega1
        full_contribution = val * num / denom
    else:
        num = (
            cmath.exp(-1j * omega2 * t)
            - 1.0
            + omega2 / omega12 * (1.0 - cmath.exp(-1j * omega12 * t))
            + omega1 / omega2 * (cmath.exp(-1j * omega2 * t) - 1.0 + 1j * omega2 * t)
        )
        denom = omega1 * omega2 * omega12
        full_contribution = num / denom * val

    return full_contribution


# prefactor by Jung that turns the classical two time correlation function into its quantum counterpart:
@njit(fastmath=True)
def prefactor_jung(omega1, omega2, kbT):
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0
    beta = 1.0 / kbT
    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = 1.0
    elif abs(omega12) < tol:
        num = beta * beta * omega2 * omega1
        denom = 2.0 * (1.0 + beta * omega1 - np.exp(beta * omega1))
        full_contribution = num / denom
    elif abs(omega1) < tol:
        num = beta * beta * omega2 * omega2
        denom = 2.0 * (
            1.0 - np.exp(-beta * omega2) - beta * omega2 * np.exp(-beta * omega2)
        )
        full_contribution = num / denom
    elif abs(omega2) < tol:
        num = beta * beta * omega1 * omega1
        denom = 2.0 * (np.exp(-beta * omega1) + beta * omega1 - 1.0)
        full_contribution = num / denom
    else:
        num = beta * beta * omega1 * omega2 * omega12
        denom = 2.0 * (
            omega2 * np.exp(-beta * omega12) - omega12 * np.exp(-beta * omega2) + omega1
        )
        full_contribution = num / denom

    return full_contribution


# Exact contribution to the 3rd order correlation function correction h1(t1,t2) to
# the 3rd order response function, for a specific value of omega1 and omega2.
# WORK THIS OUT PROPERLY, THEN SPEED UP THE CODE
@njit(fastmath=True, parallel=True)
def prefactor_2DES_h1_QM(const_fac, omega1, omega2, kbT, t1, t2):
    val = const_fac / (4.0 * math.pi ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = -1j * val / 2.0 * t1 ** 2.0 * t2
    elif abs(omega12) < tol:
        num = (
            -1j * t2 * (1j * t1 - (1.0 - cmath.exp(-1j * omega2 * t1)) / omega2)
        )  # double check. Is the limit correct?
        denom = omega1
        full_contribution = val * num / denom
    elif abs(omega1) < tol:
        num = (
            (1.0 - cmath.exp(1j * omega2 * t2))
            * cmath.exp(-1j * omega2 * t1)
            * (1j * omega2 * t1 - cmath.exp(1j * omega2 * t1) + 1.0)
        )
        denom = omega2 ** 3.0
        full_contribution = val * num / denom
    elif abs(omega2) < tol:
        num = (1.0 - cmath.exp(1j * omega1 * t2)) * (
            (1.0 - cmath.exp(-1j * omega1 * t1)) / omega1 - 1j * t1
        )
        denom = omega1 ** 2.0
        full_contribution = val * num / denom
    else:
        num = (1.0 - cmath.exp(1j * omega12 * t2)) * (
            (1.0 - cmath.exp(-1j * omega12 * t1)) / omega12
            - (1.0 - cmath.exp(-1j * omega2 * t1)) / omega2
        )
        denom = omega1 * omega12
        full_contribution = num / denom * val

    return full_contribution


# h1 prefactor if the quantum correlation function is approximately reconstructed from its
# classical counterpart
@jit
def prefactor_2DES_h1_cl(const_fac, omega1, omega2, kbT, t1, t2):
    full_contribution = prefactor_jung(omega1, omega2, kbT) * prefactor_2DES_h1_QM(
        const_fac, omega1, omega2, kbT, t1, t2
    )

    return full_contribution


# Exact contribution h2 is very closely related to h1. Reuse the code for h1.
@jit
def prefactor_2DES_h2_QM(const_fac, omega1, omega2, kbT, t1, t2):
    return prefactor_2DES_h1_QM(const_fac, omega2, omega1, kbT, t1, t2)


# h2 prefactor if the quantum correlation function is approximately reconstructed from its
# classical counterpart, which breaks the symmetry between omega1 and omega2 that allows one
# to write the h2_QM prefactor in terms of the h1_QM prefactor
@jit
def prefactor_2DES_h2_cl(const_fac, omega1, omega2, kbT, t1, t2):
    full_contribution = prefactor_jung(omega1, omega2, kbT) * prefactor_2DES_h2_QM(
        const_fac, omega1, omega2, kbT, t1, t2
    )

    return full_contribution


# H3 is a function of 3 variables. We will probably run into storage issues here (16 GB of storage to store full array)
# therefore, h3 contribution has to be computed on the fly
@njit(fastmath=True, parallel=True)
def prefactor_2DES_h3_QM(const_fac, omega1, omega2, kbT, t1, t2, t3):
    val = const_fac / (4.0 * math.pi ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = -1j * val * t1 * t2 * t3
    elif abs(omega12) < tol:
        num = (
            1j
            * t3
            * (1.0 - cmath.exp(-1j * omega2 * t1))
            * (1.0 - cmath.exp(-1j * omega1 * t2))
        )
        denom = omega1 * omega2
        full_contribution = val * num / denom
    elif abs(omega1) < tol:
        num = (
            -1j
            * (1.0 - cmath.exp(1j * omega2 * t3))
            * (1.0 - cmath.exp(-1j * omega2 * t1))
            * t2
        )
        denom = omega2 ** 2.0
        full_contribution = val * num / denom
    elif abs(omega2) < tol:
        num = (
            -1j
            * (1.0 - cmath.exp(1j * omega1 * t3))
            * (1.0 - cmath.exp(-1j * omega1 * t2))
            * t1
        )
        denom = omega1 ** 2.0
        full_contribution = val * num / denom
    else:
        num = (
            -(1.0 - cmath.exp(1j * omega12 * t3))
            * (1.0 - cmath.exp(-1j * omega2 * t1))
            * (1.0 - cmath.exp(-1j * omega1 * t2))
        )
        denom = omega1 * omega12 * omega2
        full_contribution = num / denom * val

    return full_contribution


@njit(fastmath=True, parallel=True)
def prefactor_2DES_h3_QM_fast(
    const_fac,
    omega1,
    omega2,
    kbT,
    expw1t2,
    expw2t1,
    expw1w2t3,
    expw1t3,
    expw2t3,
    t1,
    t2,
    t3,
):
    val = const_fac / (4.0 * math.pi ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = -1j * val * t1 * t2 * t3
    elif abs(omega12) < tol:
        num = 1j * t3 * (1.0 - np.conj(expw2t1)) * (1.0 - np.conj(expw1t2))
        # num=1j*t3*(1.0-cmath.exp(-1j*omega2*t1))*(1.0-cmath.exp(-1j*omega1*t2))
        denom = omega1 * omega2
        full_contribution = val * num / denom
    elif abs(omega1) < tol:
        # num=-1j*(1.0-cmath.exp(1j*omega2*t3))*(1.0-cmath.exp(-1j*omega2*t1))*t2
        num = -1j * (1.0 - expw2t3) * (1.0 - np.conj(expw2t1)) * t2
        denom = omega2 ** 2.0
        full_contribution = val * num / denom
    elif abs(omega2) < tol:
        # num=-1j*(1.0-cmath.exp(1j*omega1*t3))*(1.0-cmath.exp(-1j*omega1*t2))*t1
        num = -1j * (1.0 - expw1t3) * (1.0 - np.conj(expw1t2)) * t1
        denom = omega1 ** 2.0
        full_contribution = val * num / denom
    else:
        # num=-(1.0-cmath.exp(1j*omega12*t3))*(1.0-cmath.exp(-1j*omega2*t1))*(1.0-cmath.exp(-1j*omega1*t2))
        num = -(1.0 - expw1w2t3) * (1.0 - np.conj(expw2t1)) * (1.0 - np.conj(expw1t2))
        denom = omega1 * omega12 * omega2
        full_contribution = num / denom * val

    return full_contribution


# H3 is a function of 3 variables. We will probably run into storage issues here (16 GB of storage to store full array)
# therefore, h3 contribution has to be computed on the fly
@jit
def prefactor_2DES_h3_cl(const_fac, omega1, omega2, kbT, t1, t2, t3):
    full_contribution = prefactor_jung(omega1, omega2, kbT) * prefactor_2DES_h3_QM(
        const_fac, omega1, omega2, kbT, t1, t2, t3
    )

    return full_contribution


# Exact contribution to the 3rd order correlation function correction h4(t1,t2) to
# the 3rd order response function, for a specific value of omega1 and omega2.
# WORK THIS OUT PROPERLY, THEN SPEED UP THE CODE
@njit(fastmath=True, parallel=True)
def prefactor_2DES_h4_QM(const_fac, omega1, omega2, kbT, t1, t2):
    val = const_fac / (4.0 * math.pi ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = -1j * val / 2.0 * t1 * t2 ** 2.0
    elif abs(omega12) < tol:
        num = (1.0 - cmath.exp(-1j * omega2 * t1)) * (
            1.0 + cmath.exp(1j * omega2 * t2) * (1j * omega2 * t2 - 1.0)
        )  # double check. Is the limit correct?
        denom = omega2 ** 3.0
        full_contribution = val * num / denom
    elif abs(omega1) < tol:
        num = -(1.0 - cmath.exp(-1j * omega2 * t1)) * (
            1j * t2 + (1.0 - cmath.exp(1j * omega2 * t2)) / omega2
        )
        denom = omega2 ** 2.0
        full_contribution = val * num / denom
    elif abs(omega2) < tol:
        num = -1j * t1 * ((1.0 - cmath.exp(-1j * omega1 * t2)) / omega1 - 1j * t2)
        denom = omega1
        full_contribution = val * num / denom
    else:
        num = -(1.0 - cmath.exp(-1j * omega2 * t1)) * (
            (1.0 - cmath.exp(-1j * omega1 * t2)) / omega1
            + (1.0 - cmath.exp(1j * omega2 * t2)) / omega2
        )
        denom = omega2 * omega12
        full_contribution = num / denom * val

    return full_contribution


@jit
def prefactor_2DES_h4_cl(const_fac, omega1, omega2, kbT, t1, t2):
    full_contribution = prefactor_jung(omega1, omega2, kbT) * prefactor_2DES_h4_QM(
        const_fac, omega1, omega2, kbT, t1, t2
    )

    return full_contribution


# Exact contribution to the 3rd order correlation function correction h5(t1,t2) to
# the 3rd order response function, for a specific value of omega1 and omega2.
# WORK THIS OUT PROPERLY, THEN SPEED UP THE CODE
@njit(fastmath=True, parallel=True)
def prefactor_2DES_h5_QM(const_fac, omega1, omega2, kbT, t1, t2):
    val = const_fac / (4.0 * math.pi ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega1=0,omega2=0, omega12=0
    if abs(omega1) < tol and abs(omega2) < tol:
        full_contribution = -1j * val / 2.0 * t1 * t2 ** 2.0
    elif abs(omega12) < tol:
        num = -(1.0 - cmath.exp(-1j * omega2 * t1)) * (
            1j * t2 + (1.0 - cmath.exp(1j * omega2 * t2)) / omega2
        )  # double check. Is the limit correct?
        denom = omega2 ** 2.0
        full_contribution = val * num / denom
    elif abs(omega1) < tol:
        num = -(1.0 - cmath.exp(-1j * omega2 * t1)) * (
            cmath.exp(1j * omega2 * t2) * (1.0 - 1j * omega2 * t2) - 1.0
        )
        denom = omega2 ** 3.0
        full_contribution = val * num / denom
    elif abs(omega2) < tol:
        num = -1j * t1 * ((1.0 - cmath.exp(1j * omega1 * t2)) / omega1 + 1j * t2)
        denom = omega1
        full_contribution = val * num / denom
    else:
        num = -(1.0 - cmath.exp(-1j * omega2 * t1)) * (
            (1.0 - cmath.exp(1j * omega12 * t2)) / omega12
            - (1.0 - cmath.exp(1j * omega2 * t2)) / omega2
        )
        denom = omega1 * omega2
        full_contribution = num / denom * val

    return full_contribution


@jit
def prefactor_2DES_h5_cl(const_fac, omega1, omega2, kbT, t1, t2):
    full_contribution = prefactor_jung(omega1, omega2, kbT) * prefactor_2DES_h5_QM(
        const_fac, omega1, omega2, kbT, t1, t2
    )

    return full_contribution


# This is the Jung correction factor applied to the 3rd order lineshape function
@njit(fastmath=True)
def prefactor_3rd_order_lineshape(const_fac, omega1, omega2, kbT, t):
    val = const_fac / (8.0 * math.pi ** 2.0 * kbT ** 2.0)
    omega12 = omega1 + omega2
    tol = 10.0 ** (-15)
    full_contribution = 0.0 + 0.0j

    # deal separately with omega12=0 and omega2=0
    if abs(omega1) / kbT < tol and abs(omega2) / kbT < tol:
        full_contribution = -val / 3.0 * 1j * kbT ** 2.0 * t ** 3.0
    elif abs(omega12) / kbT < tol:
        num = (
            1j * omega1 * t
            - t ** 2.0 * omega1 ** 2.0 / 2.0
            + 1.0
            - cmath.exp(1j * omega1 * t)
        )
        denom = omega1 * (1.0 + omega1 / kbT - cmath.exp(omega1 / kbT))
        full_contribution = val * num / denom
    elif abs(omega1) / kbT < tol:
        num = 2.0 * (cmath.exp(-1j * omega2 * t) - 1.0) + 1j * omega2 * t * (
            cmath.exp(-1j * omega2 * t) + 1.0
        )
        denom = omega2 * (
            1.0 - cmath.exp(-omega2 / kbT) - omega2 / kbT * cmath.exp(-omega2 / kbT)
        )
        full_contribution = val * num / denom

    elif abs(omega2) / kbT < tol:
        num = (
            -1j * omega1 * t
            + (1.0 - cmath.exp(-1j * omega1 * t))
            - 1.0 / 2.0 * omega1 ** 2.0 * t ** 2.0
        )
        denom = omega1 * (cmath.exp(-omega1 / kbT) + omega1 / kbT - 1.0)
        full_contribution = val * num / denom
    else:
        num = (
            cmath.exp(-1j * omega2 * t)
            - 1.0
            + omega2 / omega12 * (1.0 - cmath.exp(-1j * omega12 * t))
            + omega1 / omega2 * (cmath.exp(-1j * omega2 * t) - 1.0 + 1j * omega2 * t)
        )
        denom = (
            omega2 * cmath.exp(-omega12 / kbT)
            - omega12 * cmath.exp(-omega2 / kbT)
            + omega1
        )
        full_contribution = num / denom * val

    return full_contribution


@jit
def second_order_lineshape_cl_t(freqs_gs, Omega_sq, gamma, kbT, t):
    corr_val = 0.0 + 0.0j
    tol = 10.0e-12
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = gamma[icount] ** 2.0 / (2.0 * freqs_gs[icount] ** 3.0)
        # GAMMA term of the lineshape function
        n = bose_einstein(freqs_gs[icount], kbT)
        gamma_term = gamma_term + const_fac * (
            2.0 * n
            + 1.0
            - 1j * freqs_gs[icount] * t
            - (n + 1.0) * cmath.exp(-1j * freqs_gs[icount] * t)
            - n * cmath.exp(1j * freqs_gs[icount] * t)
        )

        icount = icount + 1

    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        while jcount < freqs_gs.shape[0]:
            const_val = (
                kbT
                * Omega_sq[icount, jcount] ** 2.0
                / (2.0 * (freqs_gs[icount] * freqs_gs[jcount]) ** 2.0)
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]

            n_p = bose_einstein(omega_p, kbT)
            omega_p_term = (
                const_val
                * (
                    2.0 * n_p
                    + 1.0
                    - 1j * omega_p * t
                    - (n_p + 1.0) * cmath.exp(-1j * omega_p * t)
                    - n_p * cmath.exp(1j * omega_p * t)
                )
                / omega_p
            )

            # deal with the limit of omega_i=omega_j
            if abs(omega_m) > tol:
                n_m = bose_einstein(omega_m, kbT)
                omega_m_term = (
                    const_val
                    * (
                        2.0 * n_m
                        + 1.0
                        - 1j * omega_m * t
                        - (n_m + 1.0) * cmath.exp(-1j * omega_m * t)
                        - n_m * cmath.exp(1j * omega_m * t)
                    )
                    / omega_m
                )
            else:
                omega_m_term = const_val * (kbT * t ** 2.0)  # check this limit.

            omega_term = omega_term + omega_p_term + omega_m_term

            jcount = jcount + 1
        icount = icount + 1
    corr_val = omega_term + gamma_term
    return corr_val


@jit
def second_order_lineshape_qm_t(freqs_gs, Omega_sq, gamma, kbT, t):
    corr_val = 0.0 + 0.0j
    tol = 10.0e-12
    gamma_term = 0.0 + 0.0j
    omega_term = 0.0 + 0.0j
    # start with gamma term first:
    icount = 0
    while icount < freqs_gs.shape[0]:
        const_fac = gamma[icount] ** 2.0 / (2.0 * freqs_gs[icount] ** 3.0)
        # GAMMA term of the lineshape function
        n = bose_einstein(freqs_gs[icount], kbT)
        gamma_term = gamma_term + const_fac * (
            2.0 * n
            + 1.0
            - 1j * freqs_gs[icount] * t
            - (n + 1.0) * cmath.exp(-1j * freqs_gs[icount] * t)
            - n * cmath.exp(1j * freqs_gs[icount] * t)
        )

        icount = icount + 1

    icount = 0
    while icount < freqs_gs.shape[0]:
        jcount = 0
        n_i = bose_einstein(freqs_gs[icount], kbT)
        while jcount < freqs_gs.shape[0]:
            n_j = bose_einstein(freqs_gs[jcount], kbT)
            const_val = Omega_sq[icount, jcount] ** 2.0 / (
                2.0 * (freqs_gs[icount] * freqs_gs[jcount])
            )
            omega_p = freqs_gs[icount] + freqs_gs[jcount]
            omega_m = freqs_gs[icount] - freqs_gs[jcount]

            omega_p_term = (
                const_val
                * (
                    2.0 * n_i * n_j
                    + n_i
                    + n_j
                    + 1.0
                    - (n_i + n_j + 1.0) * 1j * omega_p * t
                    - (n_i + 1.0) * (n_j + 1.0) * cmath.exp(-1j * omega_p * t)
                    - n_i * n_j * cmath.exp(1j * omega_p * t)
                )
                / omega_p ** 2.0
            )

            # deal with the limit of omega_i=omega_j
            if abs(omega_m) > tol:
                # CORRECTION?
                omega_m_term = (
                    const_val
                    * (
                        2.0 * n_i * n_j
                        + n_i
                        + n_j
                        + (n_i - n_j) * 1j * omega_m * t
                        - (n_i + 1.0) * n_j * cmath.exp(-1j * omega_m * t)
                        - n_i * (n_j + 1.0) * cmath.exp(1j * omega_m * t)
                    )
                    / omega_m ** 2.0
                )
                # omega_m_term=const_val*(2.0*n_i*n_j+n_i+n_j-(n_i-n_j)*1j*omega_m*t-(n_i+1.0)*n_j*cmath.exp(-1j*omega_m*t)-n_i*(n_j+1.0)*cmath.exp(1j*omega_m*t))/omega_m**2.0
            else:
                omega_m_term = const_val * (
                    (n_i * n_j + n_i) * t ** 2.0
                )  # check this limit.

            omega_term = omega_term + omega_p_term + omega_m_term

            jcount = jcount + 1
        icount = icount + 1

    corr_val = omega_term + gamma_term
    return corr_val
