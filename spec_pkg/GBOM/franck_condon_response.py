#! /usr/bin/env python

import sys

sys.path.append(".")
import numpy as np
import math
import cmath
from numba import jit
from ..constants import constants as const


# This file contains all routines to compute a temperature-dependent Franck-Condon response function from
# the GBOM. The expressions implemented here are for the numerically stable version detailed by deSouza and
# coworkers
# TODO: Write the zero-temperature version


@jit
def get_a_gs(freq_gs, kBT, time):
    tau_gs = -time - 1j / kBT
    a_mat = np.zeros((freq_gs.shape[0], freq_gs.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_gs.shape[0]:
        a_mat[counter, counter] = freq_gs[counter] / cmath.sin(
            freq_gs[counter] * tau_gs
        )
        counter = counter + 1
    return a_mat


@jit
def get_a_ex(freq_ex, time):
    tau_ex = time
    a_mat = np.zeros((freq_ex.shape[0], freq_ex.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_ex.shape[0]:
        a_mat[counter, counter] = freq_ex[counter] / cmath.sin(
            freq_ex[counter] * tau_ex
        )
        counter = counter + 1
    return a_mat


@jit
def get_b_gs(freq_gs, kBT, time):
    tau_gs = -time - 1j / kBT
    b_mat = np.zeros((freq_gs.shape[0], freq_gs.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_gs.shape[0]:
        b_mat[counter, counter] = freq_gs[counter] / cmath.tan(
            freq_gs[counter] * tau_gs
        )
        counter = counter + 1

    return b_mat


@jit
def get_b_ex(freq_ex, time):
    tau_ex = time
    b_mat = np.zeros((freq_ex.shape[0], freq_ex.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_ex.shape[0]:
        b_mat[counter, counter] = freq_ex[counter] / cmath.tan(
            freq_ex[counter] * tau_ex
        )
        counter = counter + 1

    return b_mat


@jit
def get_d_gs(freq_gs, kBT, time):
    tau_gs = -1j * time + 1.0 / kBT
    d_mat = np.zeros((freq_gs.shape[0], freq_gs.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_gs.shape[0]:
        d_mat[counter, counter] = freq_gs[counter] * cmath.tanh(
            freq_gs[counter] * tau_gs / 2.0
        )
        counter = counter + 1

    return d_mat


@jit
def get_d_ex(freq_ex, time):
    tau_ex = 1j * time
    d_mat = np.zeros((freq_ex.shape[0], freq_ex.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_ex.shape[0]:
        d_mat[counter, counter] = freq_ex[counter] * cmath.tanh(
            freq_ex[counter] * tau_ex / 2.0
        )
        counter = counter + 1

    return d_mat


# this matrix is related to the partition function of the harmonic oscillator
@jit
def get_Pmat(freq_gs, kBT):
    Pmat = np.zeros((freq_gs.shape[0], freq_gs.shape[0]), dtype=np.complex_)
    counter = 0
    while counter < freq_gs.shape[0]:
        Pmat[counter, counter] = 2.0 * cmath.sinh(freq_gs[counter] / (2.0 * kBT))
        counter = counter + 1

    return Pmat


# now construct the actual useful matrices:
@jit
def get_Amat(a_gs, a_ex, Jmat):
    Amat = a_ex
    Jtrans = np.transpose(Jmat)
    temp_mat = np.dot(a_gs, Jmat)
    Amat = Amat + np.dot(Jtrans, temp_mat)

    return Amat


@jit
def get_Bmat(b_gs, b_ex, Jmat):
    Bmat = b_ex
    Jtrans = np.transpose(Jmat)
    temp_mat = np.dot(b_gs, Jmat)
    Bmat = Bmat + np.dot(Jtrans, temp_mat)

    return Bmat


@jit
def get_Dmat(d_gs, d_ex, Jmat):
    Dmat = d_ex
    Jtrans = np.transpose(Jmat)
    temp_mat = np.dot(d_gs, Jmat)
    Dmat = Dmat + np.dot(Jtrans, temp_mat)

    return Dmat


@jit
def get_prefac(a_gs, a_ex, Amat, Bmat, Pmat):
    Psq = np.dot(Pmat, Pmat)
    asq = np.dot(a_gs, a_ex)
    Binv = np.linalg.inv(Bmat)
    temp_mat = np.dot(Binv, Amat)
    temp_mat2 = np.dot(Amat, temp_mat)
    temp_mat = Bmat - temp_mat2
    middle_mat = np.dot(Bmat, temp_mat)
    middle_mat_inv = np.linalg.inv(middle_mat)
    temp_mat = np.dot(middle_mat_inv, Psq)
    rmat = np.dot(asq, temp_mat)

    return cmath.sqrt(np.linalg.det(rmat))


@jit
def calc_chi_for_given_time(freq_gs, freq_ex, Jmat, Kmat, kBT, time):
    # calculate full value of chi(t) for the given value of t.
    a_gs = get_a_gs(freq_gs, kBT, time)
    a_ex = get_a_ex(freq_ex, time)
    b_gs = get_b_gs(freq_gs, kBT, time)
    b_ex = get_b_ex(freq_ex, time)
    d_gs = get_d_gs(freq_gs, kBT, time)
    d_ex = get_d_ex(freq_ex, time)
    Amat = get_Amat(a_gs, a_ex, Jmat.astype(np.complex_))
    Bmat = get_Bmat(b_gs, b_ex, Jmat.astype(np.complex_))
    Pmat = get_Pmat(freq_gs, kBT)
    Dmat = get_Dmat(d_gs, d_ex, Jmat.astype(np.complex_))

    # successfully gotten auxillary matrices. First construct prefactor
    # protect against small values of time, for which c and a diverge
    if time < 0.000001:
        prefac = cmath.sqrt((-1.0) ** (freq_gs.shape[0]))
    else:
        prefac = get_prefac(a_gs, a_ex, Amat, Bmat, Pmat)

    Dinv = np.linalg.inv(Dmat)
    Ktrans = np.transpose(Kmat.astype(np.complex_))
    temp = np.dot(d_gs, Jmat.astype(np.complex_))
    Vtrans = np.dot(Ktrans, temp)
    V = np.transpose(Vtrans)
    temp = np.dot(d_gs, Kmat.astype(np.complex_))
    temp1 = np.dot(Ktrans, temp)
    temp = np.dot(Dinv, V)
    temp2 = np.dot(Vtrans, temp)
    total_val = -temp1 + temp2

    chi_t = prefac * cmath.exp(total_val)

    return cmath.polar(chi_t)


@jit
def calc_lineshape_for_given_time(freq_gs, freq_ex, Jmat, Kmat, kBT, time):
    # calculate full value of g_inf(t) for the given value of t.
    a_gs = get_a_gs(freq_gs, kBT, time)
    a_ex = get_a_ex(freq_ex, time)
    b_gs = get_b_gs(freq_gs, kBT, time)
    b_ex = get_b_ex(freq_ex, time)
    d_gs = get_d_gs(freq_gs, kBT, time)
    d_ex = get_d_ex(freq_ex, time)
    Amat = get_Amat(a_gs, a_ex, Jmat.astype(np.complex_))
    Bmat = get_Bmat(b_gs, b_ex, Jmat.astype(np.complex_))
    Pmat = get_Pmat(freq_gs, kBT)
    Dmat = get_Dmat(d_gs, d_ex, Jmat.astype(np.complex_))

    # successfully gotten auxillary matrices. First construct prefactor
    # protect against small values of time, for which c and a diverge
    if time < 0.000001:
        prefac = cmath.sqrt((-1.0) ** (freq_gs.shape[0]))
    else:
        prefac = get_prefac(a_gs, a_ex, Amat, Bmat, Pmat)

    Dinv = np.linalg.inv(Dmat)
    Ktrans = np.transpose(Kmat.astype(np.complex_))
    temp = np.dot(d_gs, Jmat.astype(np.complex_))
    Vtrans = np.dot(Ktrans, temp)
    V = np.transpose(Vtrans)
    temp = np.dot(d_gs, Kmat.astype(np.complex_))
    temp1 = np.dot(Ktrans, temp)
    temp = np.dot(Dinv, V)
    temp2 = np.dot(Vtrans, temp)
    total_val = -temp1 + temp2

    return (-np.log(prefac) - total_val).real


def compute_full_response_func(
    freq_gs, freq_ex, Jmat, Kmat, E_adiabatic, kBT, steps, max_time, is_emission
):
    chi = np.zeros((steps, 3))
    lineshape = np.zeros((steps, 2))
    response_func = np.zeros((steps, 2), dtype=np.complex_)
    step_length = max_time / steps
    start_val = 0.0000001
    counter = 0
    while counter < steps:
        current_t = start_val + step_length * counter
        chi[counter, 0] = current_t
        lineshape[counter, 0] = current_t
        # if it is an emission spectrum, switch definition of initial and final state around.
	# This is already done outside of the routine. All we have to do is revert time. 
        if is_emission:
            chi_t = calc_chi_for_given_time(
                freq_gs, freq_ex, Jmat, Kmat, kBT, -current_t
            )
            g_inf = calc_lineshape_for_given_time(
                freq_gs, freq_ex, Jmat, Kmat, kBT, -current_t
            )
        else:
            # calculate the effective lineshape function as well as chi_t
            chi_t = calc_chi_for_given_time(
                freq_gs, freq_ex, Jmat, Kmat, kBT, current_t
            )
            g_inf = calc_lineshape_for_given_time(
                freq_gs, freq_ex, Jmat, Kmat, kBT, current_t
            )
        lineshape[counter, 1] = g_inf
        chi[counter, 1] = chi_t[0]
        chi[counter, 2] = chi_t[1]
        counter = counter + 1

    np.savetxt("FC_lineshape_function.dat", lineshape)
    # now make sure the phase is a continuous function
    counter = 0
    phase_fac = 0.0
    while counter < steps - 1:
        chi[counter, 2] = chi[counter, 2] + phase_fac
        if (
            abs(chi[counter, 2] - phase_fac - chi[counter + 1, 2]) > 0.7 * math.pi
        ):  # check for discontinuous jump.
            diff = chi[counter + 1, 2] - (chi[counter, 2] - phase_fac)
            frac = diff / math.pi
            n = int(round(frac))
            phase_fac = phase_fac - math.pi * n
        chi[steps - 1, 2] = chi[steps - 1, 2] + phase_fac

        counter = counter + 1

    # now construct response function from chi:
    counter = 0
    while counter < steps:
        response_func[counter, 0] = chi[counter, 0]
        if is_emission:
            response_func[counter, 1] = chi[counter, 1] * cmath.exp(
            1j * chi[counter, 2] - 1j * (E_adiabatic-0.5*(np.sum(freq_ex)-np.sum(freq_gs))) * chi[counter, 0]
            )
        else:
            response_func[counter, 1] = chi[counter, 1] * cmath.exp(
            1j * chi[counter, 2] - 1j * E_adiabatic * chi[counter, 0]
            )
        counter = counter + 1

    return response_func
