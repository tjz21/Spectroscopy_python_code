import numpy.linalg
import torch
import cmath
import itertools
import numpy as np
import scipy.integrate as sciint
import spec_pkg.GBOM.gbom
import spec_pkg.params.params
import spec_pkg.constants.constants

def FULL_ft(R1_R4,R2_R3, t_dat, freqs,central_freq, fname):
    out_file = open(fname+"2DES_SPECTRUM.dat", 'w+')
    out_file.write("E_1 E_2 Val" + '\n')
    spec = np.zeros((len(freqs), len(freqs)))
    print("\33[100C")
    for i in range(0, len(freqs)):
        intergrant_r1r4 = np.multiply(R1_R4, np.outer(np.exp(1j*(freqs[i]-central_freq)*t_dat), np.ones_like(t_dat))) 
        intergrant_over_t_1_r1r4 = np.zeros_like(t_dat, dtype= complex)
        intergrant_r2r3 =  np.multiply(R2_R3, np.outer(np.exp(-1j*(freqs[i]-central_freq)*t_dat), np.ones_like(t_dat)))
        intergrant_over_t_1_r2r3 = np.zeros_like(t_dat, dtype= complex)
        for t_i in range(0, len(t_dat)):
            intergrant_over_t_1_r1r4[t_i] = sciint.simpson(intergrant_r1r4[:,t_i], t_dat, t_dat[1] - t_dat[0])
            intergrant_over_t_1_r2r3[t_i] = sciint.simpson(intergrant_r2r3[:,t_i], t_dat, t_dat[1] - t_dat[0])
        for j in range(0, len(freqs)):
            full_intergrant_r1r4 = intergrant_over_t_1_r1r4*np.exp(1j * (freqs[j]-central_freq) * t_dat)
            full_intergrant_r2r3 = intergrant_over_t_1_r2r3*np.exp(1j * (freqs[j]-central_freq) * t_dat)
            spec_val = sciint.simpson(full_intergrant_r1r4+full_intergrant_r2r3, t_dat, t_dat[1] - t_dat[0]).real
            spec[j,i] += spec_val
            out_file.write(f"{freqs[j]} {freqs[i]} {spec_val}" + '\n')
        print(f"TAKING FOURIER TRANSFORM: %{int(round(i/ len(freqs)*100, 0))}", end = '\r')
    out_file.close()
    out_file = open(fname+"TA_SPECTRUM.dat", 'w+')
    out_file.write("E_3 Val" +'\n')
    spec/=np.max(spec)
    for i in range(0, spec.shape[0]):
        ta_val = np.mean(spec[i,:])
        out_file.write(f"{freqs[i]} {ta_val}"+'\n')
    out_file.close()
    print("\33[100C")



with torch.no_grad():
    def compute_response_at_time_GPU(gs_freqs, es_freqs, J_mat, k_vec, num_modes:int, tau_sets, p_mat):
            D_tensor = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            D_inv = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            E_tensor = torch.zeros((4, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            D3_tensor = torch.zeros((4, 2* num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            D_inv_pref = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device = 'cuda')
            A_12_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            A_34_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            Q_ind_vals = torch.zeros(4, dtype= torch.complex128, device= 'cuda')
            ###Build all inital tensors across all response pathways...toss saving all a's and b's for all modes in next go?
            count = 0
            while count < 4:
                tau_set = tau_sets[count, :]
                a_1_tensor= torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[0])).to(torch.complex128).cuda()
                b_1_tensor = torch.diag(gs_freqs/ torch.tan(gs_freqs*tau_set[0])).to(torch.complex128).cuda() 
                a_3_tensor = torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[2])).to(torch.complex128).cuda() 
                b_3_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[2])).to(torch.complex128).cuda()
                a_2_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[1])).to(torch.complex128).cuda()
                b_2_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[1])).to(torch.complex128).cuda()
                a_4_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[3])).to(torch.complex128).cuda()
                b_4_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[3])).to(torch.complex128).cuda()
                b_1_tensor = torch.nan_to_num(b_1_tensor)
                b_2_tensor = torch.nan_to_num(b_2_tensor)
                b_3_tensor = torch.nan_to_num(b_3_tensor)
                b_4_tensor = torch.nan_to_num(b_4_tensor)
                ###THIS COULD BE HACKY, MAP LARGE VALUES OF B (AND A?) TO SOMETHING SYNTHETICALLY SMALLER TO AVOID ASYMPTOTES?
                c_3_tensor = b_3_tensor - a_3_tensor
                c_1_tensor = b_1_tensor - a_1_tensor
                E_tensor[count, :num_modes] = E_tensor[count, 3*num_modes:] = J_mat.t() @(c_3_tensor) @ k_vec
                E_tensor[count, num_modes:2*num_modes] = E_tensor[count, 2*num_modes:3*num_modes] = J_mat.t() @ (c_1_tensor)@k_vec
                D_tensor[count, :num_modes, :num_modes] = b_4_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, num_modes:2*num_modes, num_modes:2*num_modes] = b_4_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 2*num_modes:3*num_modes] = b_2_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 3*num_modes:, 3*num_modes:] = b_2_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, :num_modes, num_modes:2*num_modes] = D_tensor[count, num_modes:2*num_modes, :num_modes] = -a_4_tensor
                D_tensor[count, :num_modes, 3*num_modes:4*num_modes] = D_tensor[count, 3*num_modes: 4*num_modes, :num_modes]=- J_mat.t() @ a_3_tensor @J_mat
                D_tensor[count, num_modes:2*num_modes, 2*num_modes:3*num_modes] =D_tensor[count, 2*num_modes:3*num_modes, num_modes:2*num_modes]= - J_mat.t()@a_1_tensor @ J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 3*num_modes:] = D_tensor[count, 3*num_modes:, 2*num_modes:3*num_modes] = -a_2_tensor
                A_12_pref_tensor[count, :num_modes, :num_modes] = a_1_tensor
                A_12_pref_tensor[count, num_modes:, num_modes:] = a_2_tensor
                A_34_pref_tensor[count, :num_modes, :num_modes] = a_3_tensor
                A_34_pref_tensor[count, num_modes:, num_modes: ] = a_4_tensor
                Q_ind_vals[count] = k_vec.t()@(c_1_tensor + c_3_tensor)@k_vec
                D3_tensor[count, :,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]
                count +=1 
            D_inv = D_tensor.inverse()
            D3_inv = D3_tensor.inverse()
            count = 0
            while count < 4:
                D_inv_pref[count,:,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]@(
                     D_tensor[count, :2*num_modes, :2*num_modes] - D_tensor[count, :2*num_modes, 2*num_modes: ] @ D3_inv[count,:,:] @ D_tensor[count, 2*num_modes:, :2*num_modes ])
                count+=1
            D_inv_pref = D_inv_pref.inverse()
            pref_1 = A_12_pref_tensor[0,:,:] @ A_34_pref_tensor[0,:,:] @ D_inv_pref[0,:,:] @ p_mat
            s,l = torch.slogdet(pref_1)
            pref_1 = torch.sqrt(s*torch.exp(l))
            pref_2 = A_12_pref_tensor[1,:,:] @ A_34_pref_tensor[1,:,:] @ D_inv_pref[1,:,:] @ p_mat 
            s,l = torch.slogdet(pref_2)
            pref_2 = torch.sqrt(s*torch.exp(l))
            pref_3 = A_12_pref_tensor[2,:,:] @ A_34_pref_tensor[2,:,:] @ D_inv_pref[2,:,:] @ p_mat 
            s,l = torch.slogdet(pref_3)
            pref_3 = torch.sqrt(s * torch.exp(l))
            pref_4 = A_12_pref_tensor[3,:,:] @ A_34_pref_tensor[3,:,:] @ D_inv_pref[3,:,:] @ p_mat
            s,l = torch.slogdet(pref_4)
            pref_4 = torch.sqrt(s*torch.exp(l))
            exp_1 = 1j* Q_ind_vals[0] -0.5j*E_tensor[0,:].t()@ D_inv[0,:,:]@ E_tensor[0,:]
            exp_2 = 1j* Q_ind_vals[1] -0.5j*E_tensor[1,:].t()@ D_inv[1,:,:]@ E_tensor[1,:] 
            exp_3 = 1j* Q_ind_vals[2] -0.5j*E_tensor[2,:].t()@ D_inv[2,:,:]@ E_tensor[2,:]
            exp_4 = 1j*  Q_ind_vals[3] -0.5j*E_tensor[3,:].t()@ D_inv[3,:,:]@ E_tensor[3,:]
            return pref_1* torch.exp(exp_1), pref_2* torch.exp(exp_2), pref_3* torch.exp(exp_3), pref_4* torch.exp(exp_4)
    torch_response_at_time_GPU = torch.jit.script(compute_response_at_time_GPU)




with torch.no_grad():
    def compute_response_at_time_GPU_NOBATCH(gs_freqs, es_freqs, J_mat, k_vec, num_modes:int, tau_sets, p_mat):
            D_tensor = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            D_inv = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            E_tensor = torch.zeros((4, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            D3_tensor = torch.zeros((4, 2* num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            D_inv_pref = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device = 'cuda')
            A_12_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            A_34_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            Q_ind_vals = torch.zeros(4, dtype= torch.complex64, device= 'cuda')
            ###Build all inital tensors across all response pathways...toss saving all a's and b's for all modes in next go?
            count = 0
            while count < 4:
                tau_set = tau_sets[count, :]
                a_1_tensor= torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[0])).to(torch.complex128).cuda()
                b_1_tensor = torch.diag(gs_freqs/ torch.tan(gs_freqs*tau_set[0])).to(torch.complex128).cuda() 
                a_3_tensor = torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[2])).to(torch.complex128).cuda() 
                b_3_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[2])).to(torch.complex128).cuda()
                a_2_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[1])).to(torch.complex128).cuda()
                b_2_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[1])).to(torch.complex128).cuda()
                a_4_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[3])).to(torch.complex128).cuda()
                b_4_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[3])).to(torch.complex128).cuda()
                b_1_tensor = torch.nan_to_num(b_1_tensor)
                b_2_tensor = torch.nan_to_num(b_2_tensor)
                b_3_tensor = torch.nan_to_num(b_3_tensor)
                b_4_tensor = torch.nan_to_num(b_4_tensor)
                ###THIS COULD BE HACKY, MAP LARGE VALUES OF B (AND A?) TO SOMETHING SYNTHETICALLY SMALLER TO AVOID ASYMPTOTES?
                c_3_tensor = b_3_tensor - a_3_tensor
                c_1_tensor = b_1_tensor - a_1_tensor
                E_tensor[count, :num_modes] = E_tensor[count, 3*num_modes:] = J_mat.t() @(c_3_tensor) @ k_vec
                E_tensor[count, num_modes:2*num_modes] = E_tensor[count, 2*num_modes:3*num_modes] = J_mat.t() @ (c_1_tensor)@k_vec
                D_tensor[count, :num_modes, :num_modes] = b_4_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, num_modes:2*num_modes, num_modes:2*num_modes] = b_4_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 2*num_modes:3*num_modes] = b_2_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 3*num_modes:, 3*num_modes:] = b_2_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, :num_modes, num_modes:2*num_modes] = D_tensor[count, num_modes:2*num_modes, :num_modes] = -a_4_tensor
                D_tensor[count, :num_modes, 3*num_modes:4*num_modes] = D_tensor[count, 3*num_modes: 4*num_modes, :num_modes]=- J_mat.t() @ a_3_tensor @J_mat
                D_tensor[count, num_modes:2*num_modes, 2*num_modes:3*num_modes] =D_tensor[count, 2*num_modes:3*num_modes, num_modes:2*num_modes]= - J_mat.t()@a_1_tensor @ J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 3*num_modes:] = D_tensor[count, 3*num_modes:, 2*num_modes:3*num_modes] = -a_2_tensor
                A_12_pref_tensor[count, :num_modes, :num_modes] = a_1_tensor
                A_12_pref_tensor[count, num_modes:, num_modes:] = a_2_tensor
                A_34_pref_tensor[count, :num_modes, :num_modes] = a_3_tensor
                A_34_pref_tensor[count, num_modes:, num_modes: ] = a_4_tensor
                Q_ind_vals[count] = k_vec.t()@(c_1_tensor + c_3_tensor)@k_vec
                D3_tensor[count, :,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]
                count +=1 
            count = 0
            while count < 4:
                D_inv[count,:,:] = D_tensor[count,:,:].inverse()
                count+=1
            D3_inv = D3_tensor.inverse()
            count = 0
            while count < 4:
                D_inv_pref[count,:,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]@(
                     D_tensor[count, :2*num_modes, :2*num_modes] - D_tensor[count, :2*num_modes, 2*num_modes: ] @ D3_inv[count,:,:] @ D_tensor[count, 2*num_modes:, :2*num_modes ])
                count+=1
            D_inv_pref = D_inv_pref.inverse()
            pref_1 = A_12_pref_tensor[0,:,:] @ A_34_pref_tensor[0,:,:] @ D_inv_pref[0,:,:] @ p_mat
            s,l = torch.slogdet(pref_1)
            pref_1 = torch.sqrt(s*torch.exp(l))
            pref_2 = A_12_pref_tensor[1,:,:] @ A_34_pref_tensor[1,:,:] @ D_inv_pref[1,:,:] @ p_mat
            s,l = torch.slogdet(pref_2)
            pref_2 = torch.sqrt(s*torch.exp(l))
            pref_3 = A_12_pref_tensor[2,:,:] @ A_34_pref_tensor[2,:,:] @ D_inv_pref[2,:,:] @ p_mat
            s,l = torch.slogdet(pref_3)
            pref_3 = torch.sqrt(s * torch.exp(l))
            pref_4 = A_12_pref_tensor[3,:,:] @ A_34_pref_tensor[3,:,:] @ D_inv_pref[3,:,:] @ p_mat
            s,l = torch.slogdet(pref_4)
            pref_4 = torch.sqrt(s * torch.exp(l))
            exp_1 = 1j* Q_ind_vals[0] -0.5j*E_tensor[0,:].t()@ D_inv[0,:,:]@ E_tensor[0,:] 
            exp_2 = 1j* Q_ind_vals[1] -0.5j*E_tensor[1,:].t()@ D_inv[1,:,:]@ E_tensor[1,:]
            exp_3 = 1j* Q_ind_vals[2] -0.5j*E_tensor[2,:].t()@ D_inv[2,:,:]@ E_tensor[2,:] 
            exp_4 = 1j*  Q_ind_vals[3] -0.5j*E_tensor[3,:].t()@ D_inv[3,:,:]@ E_tensor[3,:] 
            return pref_1* torch.exp(exp_1), pref_2* torch.exp(exp_2), pref_3* torch.exp(exp_3), pref_4* torch.exp(exp_4)
    torch_response_at_time_GPU_NOBATCH = torch.jit.script(compute_response_at_time_GPU_NOBATCH)
with torch.no_grad():
    def compute_response_at_time_CPU(gs_freqs, es_freqs, J_mat, k_vec, num_modes:int, tau_sets,   p_mat):
            ###CALCULATES THE 4 RESPONSE PATHWAYS USING THE FRANCK-CONDON 2DES SOLUTION DERIVED BY LUKE ALLAN (MEEEEE)
            ###USING A CPU
            ###NOTE: THE TIME VARIABLES ARE LABELED BASED ON A FIRST DERIVATION THAT MADE THEM MORE AWKWARD, BUT THE RESULT IS THE SAME
            D_tensor = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128)
            E_tensor = torch.zeros((4, 4*num_modes), dtype= torch.complex128)
            D3_tensor = torch.zeros((4, 2* num_modes, 2*num_modes), dtype= torch.complex128)
            D_inv_pref = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128)
            A_12_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128)
            A_34_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128)
            Q_ind_vals = torch.zeros(4, dtype= torch.complex128)
            ###HERE WE BUILD ALL THE ELEMENTS NEEDED ON TENSORS FOR THE ABLITY TO RUN A BATCH INVERSE
            count = 0
            while count < 4:
                tau_set = tau_sets[count, :]
                a_1_tensor= torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[0])).to(torch.complex128)
                b_1_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[0])).to(torch.complex128) 
                a_3_tensor = torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[2])).to(torch.complex128) 
                b_3_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[2])).to(torch.complex128)
                a_2_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[1])).to(torch.complex128)
                b_2_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[1])).to(torch.complex128)
                a_4_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[3])).to(torch.complex128)
                b_4_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[3])).to(torch.complex128)
                c_3_tensor = b_3_tensor - a_3_tensor
                c_1_tensor = b_1_tensor - a_1_tensor
                E_tensor[count, :num_modes] = E_tensor[count, 3*num_modes:] = J_mat.t() @(c_3_tensor) @ k_vec
                E_tensor[count, num_modes:2*num_modes] = E_tensor[count, 2*num_modes:3*num_modes] = J_mat.t() @ (c_1_tensor)@k_vec
                D_tensor[count, :num_modes, :num_modes] = b_4_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, num_modes:2*num_modes, num_modes:2*num_modes] = b_4_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 2*num_modes:3*num_modes] = b_2_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 3*num_modes:, 3*num_modes:] = b_2_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, :num_modes, num_modes:2*num_modes] = D_tensor[count, num_modes:2*num_modes, :num_modes] = -a_4_tensor
                D_tensor[count, :num_modes, 3*num_modes:4*num_modes] = D_tensor[count, 3*num_modes: 4*num_modes, :num_modes]=- J_mat.t() @ a_3_tensor @J_mat
                D_tensor[count, num_modes:2*num_modes, 2*num_modes:3*num_modes] =D_tensor[count, 2*num_modes:3*num_modes, num_modes:2*num_modes]= - J_mat.t()@a_1_tensor @ J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 3*num_modes:] = D_tensor[count, 3*num_modes:, 2*num_modes:3*num_modes] = -a_2_tensor
                A_12_pref_tensor[count, :num_modes, :num_modes] = a_1_tensor
                A_12_pref_tensor[count, num_modes:, num_modes:] = a_2_tensor
                A_34_pref_tensor[count, :num_modes, :num_modes] = a_3_tensor
                A_34_pref_tensor[count, num_modes:, num_modes: ] = a_4_tensor
                Q_ind_vals[count] = k_vec.t()@(c_1_tensor + c_3_tensor)@k_vec
                D3_tensor[count, :,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]
                count +=1 
            D_inv = D_tensor.inverse()
            D3_inv = D3_tensor.inverse()
            count = 0
            while count < 4:
                D_inv_pref[count,:,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]@(
                     D_tensor[count, :2*num_modes, :2*num_modes] - D_tensor[count, :2*num_modes, 2*num_modes: ] @ D3_inv[count,:,:] @ D_tensor[count, 2*num_modes:, :2*num_modes ])
                count+=1
            ###NOTE: PYTORCH DET FUNCTION IS MORE FUSSY THAN NUMPY'S USING SLOG DET HELPS STABLIZE 
            D_inv_pref = D_inv_pref.inverse()
            pref_1 = A_12_pref_tensor[0,:,:] @ A_34_pref_tensor[0,:,:] @ D_inv_pref[0,:,:] @ p_mat
            s,l = torch.slogdet(pref_1)
            pref_1 = torch.sqrt(s*torch.exp(l))
            pref_2 = A_12_pref_tensor[1,:,:] @ A_34_pref_tensor[1,:,:] @ D_inv_pref[1,:,:] @ p_mat 
            s,l = torch.slogdet(pref_2)
            pref_2 = torch.sqrt(s*torch.exp(l))
            pref_3 = A_12_pref_tensor[2,:,:] @ A_34_pref_tensor[2,:,:] @ D_inv_pref[2,:,:] @ p_mat
            s,l = torch.slogdet(pref_3)
            pref_3 = torch.sqrt(s * torch.exp(l))
            pref_4 = A_12_pref_tensor[3,:,:] @ A_34_pref_tensor[3,:,:] @ D_inv_pref[3,:,:] @ p_mat
            s,l = torch.slogdet(pref_4)
            pref_4 = torch.sqrt(s*torch.exp(l))
            exp_1 = 1j* Q_ind_vals[0] -0.5j*E_tensor[0,:].t()@ D_inv[0,:,:]@ E_tensor[0,:] 
            exp_2 = 1j* Q_ind_vals[1] -0.5j*E_tensor[1,:].t()@ D_inv[1,:,:]@ E_tensor[1,:]  
            exp_3 = 1j* Q_ind_vals[2] -0.5j*E_tensor[2,:].t()@ D_inv[2,:,:]@ E_tensor[2,:]
            exp_4 = 1j*  Q_ind_vals[3] -0.5j*E_tensor[3,:].t()@ D_inv[3,:,:]@ E_tensor[3,:]  
            return pref_1* torch.exp(exp_1), pref_2* torch.exp(exp_2), pref_3* torch.exp(exp_3), pref_4* torch.exp(exp_4)

torch_response_at_time_CPU = torch.jit.script(compute_response_at_time_CPU)


def compute_response_at_time_NO_ROT(gs_freqs, es_freqs,  k_vec, num_modes:int, tau_sets,  p_mat):

            ###Build all inital tensors across all response pathways...toss saving all a's and b's for all modes in next go?
            val_vec = torch.zeros(4, dtype=torch.complex64)
            count = 0
            while count < 4:
                tau_set = tau_sets[count, :]
                a_1_tensor= (gs_freqs/torch.sin(gs_freqs*tau_set[0])).to(torch.complex64)
                b_1_tensor = (gs_freqs/torch.tan(gs_freqs*tau_set[0])).to(torch.complex64) 
                a_3_tensor = (gs_freqs/torch.sin(gs_freqs*tau_set[2])).to(torch.complex64) 
                b_3_tensor = (gs_freqs/torch.tan(gs_freqs*tau_set[2])).to(torch.complex64)
                a_2_tensor  = (es_freqs/ torch.sin(es_freqs*tau_set[1])).to(torch.complex64)
                b_2_tensor  = (es_freqs/ torch.tan(es_freqs*tau_set[1])).to(torch.complex64)
                a_4_tensor  = (es_freqs/ torch.sin(es_freqs*tau_set[3])).to(torch.complex64)
                b_4_tensor  = (es_freqs/ torch.tan(es_freqs*tau_set[3])).to(torch.complex64)
                c_3_tensor = b_3_tensor - a_3_tensor
                c_1_tensor = b_1_tensor - a_1_tensor
                E_tensor = torch.zeros(4*num_modes, dtype= torch.complex64)
                E_tensor[:num_modes] = E_tensor[3*num_modes:] = c_3_tensor*k_vec
                E_tensor[num_modes:2*num_modes] = E_tensor[2*num_modes: 3*num_modes] = c_1_tensor*k_vec
                B41 = b_4_tensor + b_1_tensor
                B43 = b_4_tensor + b_3_tensor
                B21 = b_2_tensor + b_1_tensor
                B23 = b_2_tensor + b_3_tensor
                ###1) solve inverse of d1 block
                B43_inv = B43**-1
                lamda_inv = (B41 - a_4_tensor*B43_inv*a_4_tensor)**-1
                d11_inv = B43_inv + B43_inv*a_4_tensor*lamda_inv*a_4_tensor*B43_inv
                d12_inv = B43_inv*a_4_tensor*lamda_inv
                d21_inv = lamda_inv*a_4_tensor*B43_inv
                d22_inv = lamda_inv
                ###2) evaluate Schur complement of D matrix
                Omega11 = (B21-a_1_tensor*d22_inv*a_1_tensor)
                Omega12 = -a_2_tensor - a_1_tensor*d21_inv*a_3_tensor
                Omega21 = -a_2_tensor - a_3_tensor*d12_inv*a_1_tensor
                Omega22 = B23 - a_3_tensor*d11_inv*a_3_tensor
                gamma_inv = (Omega22 - Omega21*Omega11**-1*Omega12)**-1
                Omega11_inv = Omega11**-1 + Omega11**-1*Omega12*gamma_inv*Omega21*Omega11**-1
                Omega12_inv = -Omega11**-1*Omega12*gamma_inv
                Omega21_inv = -gamma_inv*Omega21*Omega11**-1
                Omega22_inv = gamma_inv
                ###3) Build 4x4 block matrix D^-1
                D_inv33 = Omega11_inv
                D_inv34 = Omega12_inv
                D_inv43 = Omega21_inv
                D_inv44 = Omega22_inv
                D_inv31 = Omega11_inv*a_1_tensor*d21_inv + Omega12_inv*a_3_tensor*d11_inv
                D_inv32 = Omega11_inv*a_1_tensor*d22_inv + Omega12_inv*a_3_tensor*d12_inv
                D_inv41 = Omega21_inv*a_1_tensor*d21_inv + Omega22_inv*a_3_tensor*d11_inv
                D_inv42 = Omega21_inv*a_1_tensor*d22_inv + Omega22_inv*a_3_tensor*d12_inv
                D_inv11 = d11_inv + d11_inv*a_3_tensor*D_inv41 + d12_inv*a_1_tensor*D_inv31
                D_inv12 = d12_inv + d11_inv*a_3_tensor*D_inv42 + d12_inv*a_1_tensor*D_inv32
                D_inv21 = d21_inv + d21_inv*a_3_tensor*D_inv41 + d22_inv*a_1_tensor*D_inv31
                D_inv22 = d22_inv + d21_inv*a_3_tensor*D_inv42 + d22_inv*a_1_tensor*D_inv32
                D_inv13 = d11_inv*a_3_tensor*Omega21_inv + d12_inv*a_1_tensor*Omega11_inv
                D_inv14 = d11_inv*a_3_tensor*Omega22_inv + d12_inv*a_1_tensor*Omega12_inv
                D_inv23 = d21_inv*a_3_tensor*Omega21_inv + d22_inv*a_1_tensor*Omega11_inv
                D_inv24 = d21_inv*a_3_tensor*Omega22_inv + d22_inv*a_1_tensor*Omega12_inv
                ###4) Evaluate E^T D^-1 E 
                #print((D_tensor[count,:,:].inverse()@E_tensor[count,:]).numpy().real)
                D_inv_E = torch.zeros(4*num_modes, dtype= torch.complex64)
                kc3 = k_vec*c_3_tensor
                kc1 = k_vec*c_1_tensor
                D_inv_E[:num_modes] = D_inv11*kc3 + D_inv12*kc1 + D_inv13*kc1 + D_inv14*kc3
                D_inv_E[num_modes:2*num_modes] = D_inv21*kc3 + D_inv22*kc1 + D_inv23*kc1 + D_inv24*kc3
                D_inv_E[2*num_modes:3*num_modes] = D_inv31*kc3 + D_inv32*kc1 + D_inv33*kc1 + D_inv34*kc3
                D_inv_E[3*num_modes:4*num_modes] = D_inv41*kc3 + D_inv42*kc1 + D_inv43*kc1 + D_inv44*kc3
                FINAL_exp = -0.5j*E_tensor.t() @ D_inv_E + 1j*k_vec.t()@((c_1_tensor + c_3_tensor)*k_vec)
                ####MOVING ONTO PREFACTOR d3(d1-d2d3^-1d2^T)
                ###5) solve d3^-1
                Sigma_inv = (B23 - a_2_tensor*B21**-1*a_2_tensor)**-1
                d3_11_inv = B21**-1 + B21**-1 *a_2_tensor*Sigma_inv*a_2_tensor*B21**-1
                d3_12_inv = B21**-1*a_2_tensor*Sigma_inv
                d3_21_inv = Sigma_inv*a_2_tensor*B21**-1
                d3_22_inv = Sigma_inv
                ###6) F = d1 -d2d3^-1d2^2 -> solve entries
                F_11 = B43 - a_3_tensor*d3_22_inv*a_3_tensor
                F_12 = -a_4_tensor -a_3_tensor*d3_21_inv*a_1_tensor
                F_21 = -a_4_tensor -a_1_tensor*d3_12_inv*a_3_tensor
                F_22 = B41 -a_1_tensor*d3_11_inv*a_1_tensor
                ###7) P = d3(F) prefactor matrix that must be inverted
                P_11 = B21*F_11 - a_2_tensor*F_21
                P_12 = B21*F_12 - a_2_tensor*F_22
                P_21 = -a_2_tensor*F_11 + B23*F_21
                P_22 = -a_2_tensor*F_12 + B23*F_22
                ###8) Invert P
                Sigma_inv =  (P_22 - P_21* P_11**-1 * P_12)**-1
                P_11_inv = P_11**-1 + P_11**-1*P_12*Sigma_inv*P_21*P_11**-1
                P_12_inv = -P_11**-1*P_12*Sigma_inv
                P_21_inv = -Sigma_inv*P_21*P_11**-1
                P_22_inv = Sigma_inv
                ###8) matrix to take det of
                Z_inv = torch.diag(p_mat[:num_modes, :num_modes]) #### MAKE THIS NEATER IN J=I method
                pref_11 = a_1_tensor *a_3_tensor*P_11_inv*Z_inv**2
                pref_12 = a_1_tensor*a_3_tensor*P_12_inv
                pref_21 = a_2_tensor*a_4_tensor*P_21_inv*Z_inv**2
                pref_22 = a_2_tensor * a_4_tensor * P_22_inv
                p_det = pref_11*pref_22 - pref_12*pref_21
                #FINAL_pref = torch.sqrt(torch.prod(p_det))
                p_det = torch.diag(p_det)
                s,l = torch.slogdet(p_det)
                FINAL_pref = torch.sqrt(s*torch.exp(l))#MOVE THIS INTO Slogdet for stablity
                val_vec[count] = FINAL_pref*torch.exp(FINAL_exp)
                count +=1 
            return val_vec[0], val_vec[1], val_vec[2], val_vec[3]
def calc_G_2_SOLVENT(q_func,steps_in_t_delay,cut_off_tol):
    ###RUN ACROSS DIAOGNAL OF R2 TO FIND RESONABLE CUTOFF POINT
    not_cutoff = True
    count1 = 0
    while not_cutoff and count1<int(0.5*(q_func.shape[0] - steps_in_t_delay)):
        count2 = count1
        test_val=-q_func[count1,1].conjugate()+q_func[steps_in_t_delay,1]-q_func[count2,1].conjugate()-q_func[count1+steps_in_t_delay,1].conjugate()-q_func[count2+steps_in_t_delay,1]+q_func[count1+count2+steps_in_t_delay,1].conjugate()
        test_val = np.exp(test_val)
        if np.abs(test_val) < cut_off_tol:
            not_cutoff = False
        else:
            count1 +=1
    if not_cutoff:
        print("WARNING: MAX TIME IS TOO SHORT TO DECAY RESPONSE FUNCTION!")
        quit()
    cut_off = count1
    R1 = np.zeros((cut_off, cut_off), dtype= np.complex64)
    R2 = np.zeros((cut_off, cut_off), dtype=np.complex64)
    R3 = np.zeros((cut_off, cut_off), dtype=np.complex64)
    R4 = np.zeros((cut_off, cut_off), dtype=np.complex64)
    for count1 in range(0,cut_off):
        for count2 in range(0,cut_off):
            R1[count1,count2]=-q_func[count1,1]-q_func[steps_in_t_delay,1].conjugate()-q_func[count2,1].conjugate()+q_func[count1+steps_in_t_delay,1]+q_func[count2+steps_in_t_delay,1].conjugate()-q_func[count1+count2+steps_in_t_delay,1]
            R2[count1,count2]=-q_func[count1,1].conjugate()+q_func[steps_in_t_delay,1]-q_func[count2,1].conjugate()-q_func[count1+steps_in_t_delay,1].conjugate()-q_func[count2+steps_in_t_delay,1]+q_func[count1+count2+steps_in_t_delay,1].conjugate()
            R3[count1,count2]=-q_func[count1,1].conjugate()+q_func[steps_in_t_delay,1].conjugate()-q_func[count2,1]-q_func[count1+steps_in_t_delay,1].conjugate()-q_func[count2+steps_in_t_delay,1].conjugate()+q_func[count1+count2+steps_in_t_delay,1].conjugate()
            R4[count1,count2]=-q_func[count1,1]-q_func[steps_in_t_delay,1]-q_func[count2,1]+q_func[count1+steps_in_t_delay,1]+q_func[count2+steps_in_t_delay,1]-q_func[count1+count2+steps_in_t_delay,1]
    return np.exp(R1),np.exp(R2),np.exp(R3),np.exp(R4)



def Calc_2DES_time_series(g2_solvent,param_set, GBOM_model:spec_pkg.GBOM.gbom):
        print(f"RUNNING {param_set.num_time_samples_2DES} FC2DES CALCULATIONS WITH A {param_set.t_step_2DES*spec_pkg.constants.constants.fs_to_Ha}FS TIME STEP")
        KbT = param_set.temperature * (8.6173303*10.0**(-5.0)/27.211396132)
        J_mat= torch.from_numpy(GBOM_model.J).to(dtype= torch.complex128)
        gs_freqs = torch.from_numpy(GBOM_model.freqs_gs).to(dtype= torch.complex128)
        ex_freqs = torch.from_numpy(GBOM_model.freqs_ex).to(dtype= torch.complex128)
        k_vec = torch.from_numpy(GBOM_model.K).to(dtype=torch.complex128)
        num_modes = len(k_vec)
        P_mat = torch.zeros((2*num_modes, 2* num_modes), dtype= torch.complex128)
        P_mat[0:num_modes, 0:num_modes] = P_mat[num_modes:2*num_modes, num_modes:2*num_modes] =torch.diag(2 * torch.sinh(gs_freqs/(2*KbT)))
        t_dat = g2_solvent[:,0].real
        t_dat[0] = 10
        freqs = np.arange(param_set.E_adiabatic -0.5*param_set.spectral_window , param_set.E_adiabatic + 0.5*param_set.spectral_window, param_set.spectral_window/param_set.num_steps_2DES)
        for t_step_index in range(0, param_set.num_time_samples_2DES):
            print(f"CALCULATION #{t_step_index+1}")
            t_delay = t_step_index*param_set.t_step_2DES
            delay_in_steps = int(divmod(float(t_delay), float(t_dat[1] - t_dat[0]))[0])
            t_delay_rounded = delay_in_steps* (t_dat[1] - t_dat[0])
            R1_solv,R2_solv, R3_solv, R4_solv = calc_G_2_SOLVENT(g2_solvent, delay_in_steps, 10**-8)
            effictive_cutoff = R1_solv.shape[0]
            if t_delay_rounded == 0.0:
                t_delay_rounded = 10 ###PROTECTS AGAINST NO DELAY ISSUE: CAN DERIVE FIX LATER
            if not param_set.herzberg_teller:
                R1,R2,R3,R4 = Compute_fc2des_response_function(gs_freqs, ex_freqs, J_mat, k_vec,P_mat, t_dat[:effictive_cutoff], t_delay_rounded, param_set.temperature, param_set.FC2DES_device)
                R1 = Stabilize_phase(R1)*R1_solv
                R2 = Stabilize_phase(R2)*R2_solv
                R3 = Stabilize_phase(R3)*R3_solv
                R4 = Stabilize_phase(R4)*R4_solv
                R1_R4 = np.zeros((len(t_dat), len(t_dat)), dtype= complex)
                R2_R3 = np.zeros((len(t_dat), len(t_dat)), dtype= complex)
                R1_R4[:effictive_cutoff,:effictive_cutoff] = R1 + R4
                R2_R3[:effictive_cutoff,:effictive_cutoff] = R2 + R3
                f_name = f"FC2DES_{t_step_index}_"
                FULL_ft(R1_R4, R2_R3, t_dat, freqs, param_set.E_adiabatic, f_name)
            if param_set.herzberg_teller:
                print("HT_CALCULATION")
                R1P,R1,R2P,R2,R3P,R3,R4P,R4 = Compute_fc2des_response_function_W_HT(gs_freqs, ex_freqs, J_mat, k_vec,GBOM_model.dipole_mom, GBOM_model.dipole_deriv, t_dat[:effictive_cutoff], t_delay_rounded,param_set.temperature, param_set.FC2DES_device)
                FULL_R1 = R1P*Stabilize_phase(R1)*R1_solv
                FULL_R2 = R2P*Stabilize_phase(R2)*R2_solv
                FULL_R3 = R3P*Stabilize_phase(R3)*R3_solv
                FULL_R4 = R4P*Stabilize_phase(R4)*R4_solv
                R1_R4 = np.zeros((len(t_dat), len(t_dat)), dtype= complex)
                R2_R3 = np.zeros((len(t_dat), len(t_dat)), dtype= complex)
                R1_R4[:effictive_cutoff,:effictive_cutoff] = FULL_R1 + FULL_R4
                R2_R3[:effictive_cutoff,:effictive_cutoff] = FULL_R2 + FULL_R3
                f_name = f"HTFC2DES_{t_step_index}_"
                FULL_ft(R1_R4, R2_R3, t_dat, freqs, param_set.E_adiabatic, f_name)
                ###Might as well do standard FC as well.
                R1 = Stabilize_phase(R1)*R1_solv
                R2 = Stabilize_phase(R2)*R2_solv
                R3 = Stabilize_phase(R3)*R3_solv
                R4 = Stabilize_phase(R4)*R4_solv
                R1_R4 = np.zeros((len(t_dat), len(t_dat)), dtype= complex)
                R2_R3 = np.zeros((len(t_dat), len(t_dat)), dtype= complex)
                R1_R4[:effictive_cutoff,:effictive_cutoff] = R1 + R4
                R2_R3[:effictive_cutoff,:effictive_cutoff] = R2 + R3
                f_name = f"FC2DES_{t_step_index}_"
                FULL_ft(R1_R4, R2_R3, t_dat, freqs, param_set.E_adiabatic, f_name)

        
with torch.no_grad():
    def Compute_fc2des_response_function_W_HT(gs_freqs, ex_freqs,J_mat, k_vec, dipole_vec, dipole_derv, t_dat,t_2,Temp,device):
        #driver for computing full response function. will allow GPU and CPU calculations along with t=0 limits... the later taking less of a focus
        num_modes = len(k_vec)
        KbT = Temp * (8.6173303*10.0**(-5.0)/27.211396132)
        dipole_derv = torch.from_numpy(dipole_derv).type(torch.complex128)
        dipole_vec = torch.from_numpy(dipole_vec).type(torch.complex128)
        P_mat = torch.zeros((2*num_modes, 2* num_modes), dtype= torch.complex128)
        P_mat[0:num_modes, 0:num_modes] = P_mat[num_modes:2*num_modes, num_modes:2*num_modes] =torch.diag(2 * torch.sinh(gs_freqs/(2*KbT)))
      
        R_1 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        R_2 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad= False)
        R_3 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        R_4 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)

        R_1_P = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        R_2_P = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad= False)
        R_3_P = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        R_4_P = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        print("PRODUCING RESPONSE FUNCTIONS: ")
        if device == "GPU":
            R_1 = R_1
            R_2 = R_2
            R_3 = R_3
            R_4 = R_4
            J_mat = J_mat.cuda()
            gs_freqs = gs_freqs.cuda()
            ex_freqs = ex_freqs.cuda()
            k_vec = k_vec.cuda()
            P_mat = P_mat.cuda()
            dipole_derv = dipole_derv.cuda()
            dipole_vec = dipole_vec.cuda()
            print("NO BATCH INVERSE CALCULATION!")
            for t_3_ind in range(0, len(t_dat)):#######FIX THIS DONT BE DUMB
                for t_1_ind in range(0, len(t_dat)):
                    t_1 = t_dat[t_1_ind]
                    t_3= t_dat[t_3_ind] 
                    tau_sets = torch.tensor([[-t_1 -1j/KbT, -t_2, -t_3, t_1+t_2 +t_3],[-t_3, t_2 + t_3, t_1 -1j/KbT, -t_1 - t_2],
                                [-t_2 - t_3, t_3, t_1 +t_2 -1j/KbT, -t_1], [-t_1 -t_2 - t_3 -1j/KbT, t_3, t_2, t_1]]).to(torch.complex64)
                    r1p,r1,r2p,r2,r3p,r3,r4p,r4 = compute_response_at_time_GPU_HT_LOWMEM(gs_freqs, ex_freqs, J_mat, k_vec, num_modes, tau_sets, P_mat, dipole_vec, dipole_derv)
                    R_1[t_1_ind, t_3_ind] = r1.detach().cpu()
                    R_2[t_1_ind, t_3_ind] = r2.detach().cpu()
                    R_3[t_1_ind, t_3_ind] = r3.detach().cpu()
                    R_4[t_1_ind, t_3_ind] = r4.detach().cpu()
                    R_1_P[t_1_ind, t_3_ind] = r1p.detach().cpu()
                    R_2_P[t_1_ind, t_3_ind] = r2p.detach().cpu()
                    R_3_P[t_1_ind, t_3_ind] = r3p.detach().cpu()
                    R_4_P[t_1_ind, t_3_ind] = r4p.detach().cpu()
                    torch.cuda.empty_cache()
                print(f"CALCULATING RESPONSE(GPU):  %{np.round(t_3_ind/ len(t_dat) ,1)*100}", end='\r')
        if device == "CPU":
            if torch.equal(J_mat.real, torch.eye(num_modes)):
                print("J=I NOT IMPLEMENTED FOR HT, USING J!=I FORMALISM")
                J_mat[0,0]= 0.00000001
                
            if not  torch.equal(J_mat.real, torch.eye(num_modes)):
                for t_1_ind in range(0, len(t_dat)):
                    for t_3_ind in range(0, len(t_dat)):
                        t_1 = t_dat[t_1_ind]
                        t_3= t_dat[t_3_ind]

                        tau_sets = torch.tensor([[-t_1 -1j/KbT, -t_2, -t_3, t_1+t_2 +t_3],[-t_3, t_2 + t_3, t_1 -1j/KbT, -t_1 - t_2],
                                    [-t_2 - t_3, t_3, t_1 +t_2 -1j/KbT, -t_1], [-t_1 -t_2 - t_3 -1j/KbT, t_3, t_2, t_1]]).to(torch.complex64)
                        r1p,r1,r2p,r2,r3p,r3,r4p,r4 = compute_response_at_time_CPU_HT_LOWMEM(gs_freqs, ex_freqs, J_mat, k_vec, num_modes, tau_sets, P_mat, dipole_vec, dipole_derv)
                        R_1[t_1_ind, t_3_ind] = r1 
                        R_2[t_1_ind, t_3_ind] = r2 
                        R_3[t_1_ind, t_3_ind] = r3 
                        R_4[t_1_ind, t_3_ind] = r4 
                        R_1_P[t_1_ind, t_3_ind] = r1p 
                        R_2_P[t_1_ind, t_3_ind] = r2p 
                        R_3_P[t_1_ind, t_3_ind] = r3p 
                        R_4_P[t_1_ind, t_3_ind] = r4p 
                    print(f"CALCULATING RESPONSE(CPU):  %{np.round(t_1_ind/ len(t_dat) ,1)*100}", end='\r')
        R_1 = R_1.detach().cpu().numpy()
        R_2 = R_2.detach().cpu().numpy()
        R_3 = R_3.detach().cpu().numpy()
        R_4 = R_4.detach().cpu().numpy()
        R_1_P = R_1_P.detach().cpu().numpy()
        R_2_P = R_2_P.detach().cpu().numpy()
        R_3_P = R_3_P.detach().cpu().numpy()
        R_4_P = R_4_P.detach().cpu().numpy()

        return R_1_P, R_1,R_2_P, R_2,R_3_P, R_3,R_4_P, R_4
with torch.no_grad():
    def compute_response_at_time_CPU_HT_LOWMEM(gs_freqs, es_freqs, J_mat, k_vec, num_modes:int, tau_sets,  p_mat, dipole_vec, dipole_derv):
            ###CALCULATES THE 4 RESPONSE PATHWAYS USING THE FRANCK-CONDON 2DES SOLUTION DERIVED BY LUKE ALLAN (MEEEEE)
            ###USING A CPU
            ###NOTE: THE TIME VARIABLES ARE LABELED BASED ON A FIRST DERIVATION THAT MADE THEM MORE AWKWARD, BUT THE RESULT IS THE SAME
            ###ANOTHER NOTE: THE HT ROUTINE RELIES ON VECTOR AND MATRIX OPERATIONS INSTEAD OF RANK 3&4 CONTRACTIONS...SLOWER BUT LESS MEMORY
            D_tensor = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128)
            E_tensor = torch.zeros((4, 4*num_modes), dtype= torch.complex128)
            D3_tensor = torch.zeros((4, 2* num_modes, 2*num_modes), dtype= torch.complex128)
            D_inv_pref = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128)
            A_12_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128)
            A_34_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128)
            Q_ind_vals = torch.zeros(4, dtype= torch.complex128)
            ###HERE WE BUILD ALL THE ELEMENTS NEEDED ON TENSORS FOR THE ABLITY TO RUN A BATCH INVERSE
            count = 0
            while count < 4:
                tau_set = tau_sets[count, :]
                a_1_tensor= torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[0])).to(torch.complex128)
                b_1_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[0])).to(torch.complex128) 
                a_3_tensor = torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[2])).to(torch.complex128) 
                b_3_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[2])).to(torch.complex128)
                a_2_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[1])).to(torch.complex128)
                b_2_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[1])).to(torch.complex128)
                a_4_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[3])).to(torch.complex128)
                b_4_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[3])).to(torch.complex128)
                c_3_tensor = b_3_tensor - a_3_tensor
                c_1_tensor = b_1_tensor - a_1_tensor
                E_tensor[count, :num_modes] = E_tensor[count, 3*num_modes:] = J_mat.t() @(c_3_tensor) @ k_vec
                E_tensor[count, num_modes:2*num_modes] = E_tensor[count, 2*num_modes:3*num_modes] = J_mat.t() @ (c_1_tensor)@k_vec
                D_tensor[count, :num_modes, :num_modes] = b_4_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, num_modes:2*num_modes, num_modes:2*num_modes] = b_4_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 2*num_modes:3*num_modes] = b_2_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 3*num_modes:, 3*num_modes:] = b_2_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, :num_modes, num_modes:2*num_modes] = D_tensor[count, num_modes:2*num_modes, :num_modes] = -a_4_tensor
                D_tensor[count, :num_modes, 3*num_modes:4*num_modes] = D_tensor[count, 3*num_modes: 4*num_modes, :num_modes]=- J_mat.t() @ a_3_tensor @J_mat
                D_tensor[count, num_modes:2*num_modes, 2*num_modes:3*num_modes] =D_tensor[count, 2*num_modes:3*num_modes, num_modes:2*num_modes]= - J_mat.t()@a_1_tensor @ J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 3*num_modes:] = D_tensor[count, 3*num_modes:, 2*num_modes:3*num_modes] = -a_2_tensor

                A_12_pref_tensor[count, :num_modes, :num_modes] = a_1_tensor
                A_12_pref_tensor[count, num_modes:, num_modes:] = a_2_tensor
                A_34_pref_tensor[count, :num_modes, :num_modes] = a_3_tensor
                A_34_pref_tensor[count, num_modes:, num_modes: ] = a_4_tensor
 
                Q_ind_vals[count] = k_vec.t()@(c_1_tensor + c_3_tensor)@k_vec
                D3_tensor[count, :,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]
                count +=1 
            D_inv = D_tensor.inverse()
            D3_inv = D3_tensor.inverse()
            count = 0
            while count < 4:
                D_inv_pref[count,:,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]@(
                     D_tensor[count, :2*num_modes, :2*num_modes] - D_tensor[count, :2*num_modes, 2*num_modes: ] @ D3_inv[count,:,:] @ D_tensor[count, 2*num_modes:, :2*num_modes ])
                count+=1
            ###NOTE: PYTORCH DET FUNCTION IS MORE FUSSY THAN NUMPY'S USING SLOG DET HELPS STABLIZE 
            D_inv_pref = D_inv_pref.inverse()
            pref_1 = A_12_pref_tensor[0,:,:] @ A_34_pref_tensor[0,:,:] @ D_inv_pref[0,:,:] @ p_mat
            s,l = torch.slogdet(pref_1)
            pref_1 = torch.sqrt(s*torch.exp(l))
            pref_2 = A_12_pref_tensor[1,:,:] @ A_34_pref_tensor[1,:,:] @ D_inv_pref[1,:,:] @ p_mat 
            s,l = torch.slogdet(pref_2)
            pref_2 = torch.sqrt(s*torch.exp(l))
            pref_3 = A_12_pref_tensor[2,:,:] @ A_34_pref_tensor[2,:,:] @ D_inv_pref[2,:,:] @ p_mat
            s,l = torch.slogdet(pref_3)
            pref_3 = torch.sqrt(s * torch.exp(l))
            pref_4 = A_12_pref_tensor[3,:,:] @ A_34_pref_tensor[3,:,:] @ D_inv_pref[3,:,:] @ p_mat
            s,l = torch.slogdet(pref_4)
            pref_4 = torch.sqrt(s*torch.exp(l))
            exp_1 = 1j* Q_ind_vals[0] -0.5j*E_tensor[0,:].t()@ D_inv[0,:,:]@ E_tensor[0,:] 
            exp_2 = 1j* Q_ind_vals[1] -0.5j*E_tensor[1,:].t()@ D_inv[1,:,:]@ E_tensor[1,:]  
            exp_3 = 1j* Q_ind_vals[2] -0.5j*E_tensor[2,:].t()@ D_inv[2,:,:]@ E_tensor[2,:]
            exp_4 = 1j*  Q_ind_vals[3] -0.5j*E_tensor[3,:].t()@ D_inv[3,:,:]@ E_tensor[3,:]  
            R1 = pref_1*torch.exp(exp_1)
            R2 = pref_2*torch.exp(exp_2)
            R3 = pref_3*torch.exp(exp_3)
            R4 = pref_4*torch.exp(exp_4)
            ### TELLLER TERMS TO DO: 
            # 1) BUILD TENSORS TO CONTRACT OVER, THESE ARE USED ALONG EACH i,j,k DIRECTION
            # 2) LOOP OVER i,j,k DIRECTIONS W/ R_i_HT_PREF += CONTRIBUTION
            # LOOP OVER i,j,k COMPONETS OF THE 
            ####PART 1##################################################
            ############LINEAR HT########################################
            D_inv_E_Tensor = torch.einsum('ijk,ik->ij', D_inv,E_tensor)
            linear_Tensor = D_inv_E_Tensor[:,:num_modes] + D_inv_E_Tensor[:,num_modes:2*num_modes]+ D_inv_E_Tensor[:,2*num_modes:3*num_modes] + + D_inv_E_Tensor[:,3*num_modes:]
            R1_HT_PREF = 0
            R2_HT_PREF = 0
            R3_HT_PREF = 0
            R4_HT_PREF = 0
            for cart_index in range(0,3):
                eff_derv = dipole_derv[:,cart_index]
                dipole_comp = dipole_vec[cart_index]
                HT_1_Tensor = -torch.einsum('ij,j->i',linear_Tensor, eff_derv)###<-LINEAR HT VALUES
                HT_2_Tensor = torch.zeros(4, dtype= torch.complex128)
                HT_3_Tensor = torch.zeros(4, dtype= torch.complex128)
                HT_4_Tensor = torch.zeros(4, dtype= torch.complex128)
                #FORM 2 DIPOLE CONTRIBUTIONS
                quad_combos = itertools.combinations([0,1,2,3],2)
                for combo in quad_combos:
                    k_start = combo[0]*num_modes
                    j_start = combo[1]* num_modes
                    HT_2_Tensor += torch.einsum('ik,k->i', D_inv_E_Tensor[:,k_start:k_start+num_modes],eff_derv)* torch.einsum('ij,j->i', D_inv_E_Tensor[:,j_start:j_start+num_modes],eff_derv) 
                    HT_2_Tensor +=  1j*torch.einsum('ijk,j,k->i', D_inv[:,k_start:k_start+num_modes, j_start:j_start+num_modes], eff_derv,eff_derv)
                #FORM 3 DIPOLE CONTRIBUTIONS
                cube_combos = itertools.combinations([0,1,2,3], 3)
                for combo in cube_combos:
                    k_start = combo[0]*num_modes
                    j_start = combo[1]*num_modes
                    m_start = combo[2]*num_modes
                    DE_j = torch.einsum('ij,j->i',D_inv_E_Tensor[:,j_start:j_start + num_modes],eff_derv)
                    DE_k = torch.einsum('ik,k->i',D_inv_E_Tensor[:,k_start:k_start + num_modes],eff_derv)
                    DE_m = torch.einsum('im,m->i',D_inv_E_Tensor[:,m_start:m_start + num_modes],eff_derv)
                    HT_3_Tensor += 1j*torch.einsum('ijk,j,k->i', D_inv[:, j_start:j_start+num_modes, k_start:k_start+num_modes], eff_derv, eff_derv)*DE_m
                    HT_3_Tensor += 1j*torch.einsum('ikm,k,m->i', D_inv[:, k_start:k_start+num_modes, m_start:m_start+num_modes], eff_derv, eff_derv)*DE_j
                    HT_3_Tensor += 1j*torch.einsum('ijm,j,m->i', D_inv[:, j_start:j_start+num_modes, m_start:m_start+num_modes], eff_derv, eff_derv)*DE_k
                    HT_3_Tensor += DE_m * DE_j * DE_k
                HT_3_Tensor = -HT_3_Tensor
                #FOR 4 DIPOLE CONTRIBUTION
                k_start = 0
                j_start = num_modes
                m_start = 2*num_modes
                n_start = 3*num_modes
                D_jk = torch.einsum('ijk,j,k->i', D_inv[:, j_start:j_start + num_modes, k_start:k_start+num_modes], eff_derv, eff_derv)
                D_mn = torch.einsum('imn,m,n->i', D_inv[:,m_start: m_start + num_modes, n_start:n_start + num_modes], eff_derv, eff_derv)
                HT_4_Tensor += -D_jk*D_mn
                D_km = torch.einsum('ikm,k,m->i', D_inv[:, k_start:k_start+num_modes, m_start:m_start+num_modes], eff_derv, eff_derv)
                D_jn = torch.einsum('ijn,j,n->i', D_inv[:,j_start:j_start+num_modes, n_start:n_start+num_modes], eff_derv, eff_derv)
                HT_4_Tensor += -D_km*D_jn
                D_kn = torch.einsum('ikn,k,n->i', D_inv[:, k_start:k_start+num_modes, n_start:n_start+num_modes], eff_derv, eff_derv)
                D_jm = torch.einsum('ijm,j,m->i', D_inv[:, j_start:j_start+num_modes, m_start:m_start + num_modes], eff_derv, eff_derv)
                HT_4_Tensor+= - D_kn*D_jm
                DE_m = torch.einsum('im,m->i', D_inv_E_Tensor[:, m_start : m_start +num_modes], eff_derv)
                DE_n = torch.einsum('in,n->i', D_inv_E_Tensor[:, n_start : n_start +num_modes], eff_derv)
                HT_4_Tensor += 1j*D_jk*DE_m*DE_n
                DE_j = torch.einsum('ij,j->i', D_inv_E_Tensor[:, j_start:j_start + num_modes], eff_derv)
                HT_4_Tensor += 1j*D_km*DE_j*DE_n
                DE_k = torch.einsum('ik,k->i', D_inv_E_Tensor[:, k_start: k_start + num_modes], eff_derv)
                HT_4_Tensor += 1j*D_jm*DE_k*DE_n
                HT_4_Tensor += 1j*D_kn* DE_j*DE_m
                HT_4_Tensor += 1j*D_jn*DE_k*DE_m
                HT_4_Tensor += 1j*D_mn*DE_k*DE_j
                HT_4_Tensor += DE_j*DE_k*DE_m*DE_n
                R1_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[0] + dipole_comp**2 * HT_2_Tensor[0] + dipole_comp*HT_3_Tensor[0] + HT_4_Tensor[0]
                R2_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[1] + dipole_comp**2 * HT_2_Tensor[1] + dipole_comp*HT_3_Tensor[1] + HT_4_Tensor[1]
                R3_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[2] + dipole_comp**2 * HT_2_Tensor[2] + dipole_comp*HT_3_Tensor[2] + HT_4_Tensor[2]
                R4_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[3] + dipole_comp**2 * HT_2_Tensor[3] + dipole_comp*HT_3_Tensor[3] + HT_4_Tensor[3]
            return R1_HT_PREF, R1, R2_HT_PREF, R2, R3_HT_PREF, R3, R4_HT_PREF, R4

with torch.no_grad():
    def compute_response_at_time_GPU_HT_LOWMEM(gs_freqs, es_freqs, J_mat, k_vec, num_modes:int, tau_sets, p_mat,dipole_vec,dipole_derv):
            D_tensor = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            D_inv = torch.zeros((4, 4*num_modes, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            E_tensor = torch.zeros((4, 4*num_modes), dtype= torch.complex128, device = 'cuda')
            D3_tensor = torch.zeros((4, 2* num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            D_inv_pref = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device = 'cuda')

            A_12_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')
            A_34_pref_tensor = torch.zeros((4, 2*num_modes, 2*num_modes), dtype= torch.complex128, device= 'cuda')

            Q_ind_vals = torch.zeros(4, dtype= torch.complex64, device= 'cuda')
            ###Build all inital tensors across all response pathways...toss saving all a's and b's for all modes in next go?
            count = 0
            while count < 4:
                tau_set = tau_sets[count, :]
                a_1_tensor= torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[0])).to(torch.complex128).cuda()
                b_1_tensor = torch.diag(gs_freqs/ torch.tan(gs_freqs*tau_set[0])).to(torch.complex128).cuda() 
                a_3_tensor = torch.diag(gs_freqs/torch.sin(gs_freqs*tau_set[2])).to(torch.complex128).cuda() 
                b_3_tensor = torch.diag(gs_freqs/torch.tan(gs_freqs*tau_set[2])).to(torch.complex128).cuda()

                a_2_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[1])).to(torch.complex128).cuda()
                b_2_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[1])).to(torch.complex128).cuda()
                a_4_tensor  = torch.diag(es_freqs/ torch.sin(es_freqs*tau_set[3])).to(torch.complex128).cuda()
                b_4_tensor  = torch.diag(es_freqs/ torch.tan(es_freqs*tau_set[3])).to(torch.complex128).cuda()

                b_1_tensor = torch.nan_to_num(b_1_tensor)
                b_2_tensor = torch.nan_to_num(b_2_tensor)
                b_3_tensor = torch.nan_to_num(b_3_tensor)
                b_4_tensor = torch.nan_to_num(b_4_tensor)

                ###THIS COULD BE HACKY, MAP LARGE VALUES OF B (AND A?) TO SOMETHING SYNTHETICALLY SMALLER TO AVOID ASYMPTOTES?

                c_3_tensor = b_3_tensor - a_3_tensor
                c_1_tensor = b_1_tensor - a_1_tensor

                E_tensor[count, :num_modes] = E_tensor[count, 3*num_modes:] = J_mat.t() @(c_3_tensor) @ k_vec
                E_tensor[count, num_modes:2*num_modes] = E_tensor[count, 2*num_modes:3*num_modes] = J_mat.t() @ (c_1_tensor)@k_vec
                D_tensor[count, :num_modes, :num_modes] = b_4_tensor+ J_mat.t()@b_3_tensor @ J_mat
                D_tensor[count, num_modes:2*num_modes, num_modes:2*num_modes] = b_4_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 2*num_modes:3*num_modes] = b_2_tensor + J_mat.t()@b_1_tensor@J_mat
                D_tensor[count, 3*num_modes:, 3*num_modes:] = b_2_tensor+ J_mat.t()@b_3_tensor @ J_mat

                D_tensor[count, :num_modes, num_modes:2*num_modes] = D_tensor[count, num_modes:2*num_modes, :num_modes] = -a_4_tensor
                D_tensor[count, :num_modes, 3*num_modes:4*num_modes] = D_tensor[count, 3*num_modes: 4*num_modes, :num_modes]=- J_mat.t() @ a_3_tensor @J_mat
                D_tensor[count, num_modes:2*num_modes, 2*num_modes:3*num_modes] =D_tensor[count, 2*num_modes:3*num_modes, num_modes:2*num_modes]= - J_mat.t()@a_1_tensor @ J_mat
                D_tensor[count, 2*num_modes:3*num_modes, 3*num_modes:] = D_tensor[count, 3*num_modes:, 2*num_modes:3*num_modes] = -a_2_tensor

                A_12_pref_tensor[count, :num_modes, :num_modes] = a_1_tensor
                A_12_pref_tensor[count, num_modes:, num_modes:] = a_2_tensor

                A_34_pref_tensor[count, :num_modes, :num_modes] = a_3_tensor
                A_34_pref_tensor[count, num_modes:, num_modes: ] = a_4_tensor

                Q_ind_vals[count] = k_vec.t()@(c_1_tensor + c_3_tensor)@k_vec
                D3_tensor[count, :,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]
                count +=1 
            count = 0
            while count < 4:
                D_inv[count,:,:] = D_tensor[count,:,:].inverse()
                count+=1
            D3_inv = D3_tensor.inverse()
            count = 0
            while count < 4:
                D_inv_pref[count,:,:] = D_tensor[count, 2*num_modes:, 2*num_modes:]@(
                     D_tensor[count, :2*num_modes, :2*num_modes] - D_tensor[count, :2*num_modes, 2*num_modes: ] @ D3_inv[count,:,:] @ D_tensor[count, 2*num_modes:, :2*num_modes ])
                count+=1

            D_inv_pref = D_inv_pref.inverse()

            pref_1 = A_12_pref_tensor[0,:,:] @ A_34_pref_tensor[0,:,:] @ D_inv_pref[0,:,:] @ p_mat
            s,l = torch.slogdet(pref_1)
            pref_1 = torch.sqrt(s*torch.exp(l))
            pref_2 = A_12_pref_tensor[1,:,:] @ A_34_pref_tensor[1,:,:] @ D_inv_pref[1,:,:] @ p_mat
            s,l = torch.slogdet(pref_2)
            pref_2 = torch.sqrt(s*torch.exp(l))
            pref_3 = A_12_pref_tensor[2,:,:] @ A_34_pref_tensor[2,:,:] @ D_inv_pref[2,:,:] @ p_mat
            s,l = torch.slogdet(pref_3)
            pref_3 = torch.sqrt(s * torch.exp(l))
            pref_4 = A_12_pref_tensor[3,:,:] @ A_34_pref_tensor[3,:,:] @ D_inv_pref[3,:,:] @ p_mat
            s,l = torch.slogdet(pref_4)
            pref_4 = torch.sqrt(s * torch.exp(l))
            exp_1 = 1j* Q_ind_vals[0] -0.5j*E_tensor[0,:].t()@ D_inv[0,:,:]@ E_tensor[0,:] 
            exp_2 = 1j* Q_ind_vals[1] -0.5j*E_tensor[1,:].t()@ D_inv[1,:,:]@ E_tensor[1,:]
            exp_3 = 1j* Q_ind_vals[2] -0.5j*E_tensor[2,:].t()@ D_inv[2,:,:]@ E_tensor[2,:] 
            exp_4 = 1j*  Q_ind_vals[3] -0.5j*E_tensor[3,:].t()@ D_inv[3,:,:]@ E_tensor[3,:] 
            R1 = pref_1*torch.exp(exp_1)
            R2 = pref_2*torch.exp(exp_2)
            R3 = pref_3*torch.exp(exp_3)
            R4 = pref_4*torch.exp(exp_4)
            ### TELLLER TERMS TO DO: 
            # 1) BUILD TENSORS TO CONTRACT OVER, THESE ARE USED ALONG EACH i,j,k DIRECTION
            # 2) LOOP OVER i,j,k DIRECTIONS W/ R_i_HT_PREF += CONTRIBUTION
            # LOOP OVER i,j,k COMPONETS OF THE 
            ####PART 1##################################################
            ############LINEAR HT########################################
            D_inv_E_Tensor = torch.einsum('ijk,ik->ij', D_inv,E_tensor)
            linear_Tensor = D_inv_E_Tensor[:,:num_modes] + D_inv_E_Tensor[:,num_modes:2*num_modes]+ D_inv_E_Tensor[:,2*num_modes:3*num_modes] + + D_inv_E_Tensor[:,3*num_modes:]
            R1_HT_PREF = 0
            R2_HT_PREF = 0
            R3_HT_PREF = 0
            R4_HT_PREF = 0
            for cart_index in range(0,3):
                eff_derv = dipole_derv[:,cart_index]
                dipole_comp = dipole_vec[cart_index]
                HT_1_Tensor = -torch.einsum('ij,j->i',linear_Tensor, eff_derv)###<-LINEAR HT VALUES
                HT_2_Tensor = torch.zeros(4, dtype= torch.complex128).cuda()
                HT_3_Tensor = torch.zeros(4, dtype= torch.complex128).cuda()
                HT_4_Tensor = torch.zeros(4, dtype= torch.complex128).cuda()
                #FORM 2 DIPOLE CONTRIBUTIONS
                quad_combos = itertools.combinations([0,1,2,3],2)
                for combo in quad_combos:
                    k_start = combo[0]*num_modes
                    j_start = combo[1]* num_modes
                    HT_2_Tensor += torch.einsum('ik,k->i', D_inv_E_Tensor[:,k_start:k_start+num_modes],eff_derv)* torch.einsum('ij,j->i', D_inv_E_Tensor[:,j_start:j_start+num_modes],eff_derv) 
                    HT_2_Tensor +=  1j*torch.einsum('ijk,j,k->i', D_inv[:,k_start:k_start+num_modes, j_start:j_start+num_modes], eff_derv,eff_derv)
                #FORM 3 DIPOLE CONTRIBUTIONS
                cube_combos = itertools.combinations([0,1,2,3], 3)
                for combo in cube_combos:
                    k_start = combo[0]*num_modes
                    j_start = combo[1]*num_modes
                    m_start = combo[2]*num_modes
                    DE_j = torch.einsum('ij,j->i',D_inv_E_Tensor[:,j_start:j_start + num_modes],eff_derv)
                    DE_k = torch.einsum('ik,k->i',D_inv_E_Tensor[:,k_start:k_start + num_modes],eff_derv)
                    DE_m = torch.einsum('im,m->i',D_inv_E_Tensor[:,m_start:m_start + num_modes],eff_derv)
                    HT_3_Tensor += 1j*torch.einsum('ijk,j,k->i', D_inv[:, j_start:j_start+num_modes, k_start:k_start+num_modes], eff_derv, eff_derv)*DE_m
                    HT_3_Tensor += 1j*torch.einsum('ikm,k,m->i', D_inv[:, k_start:k_start+num_modes, m_start:m_start+num_modes], eff_derv, eff_derv)*DE_j
                    HT_3_Tensor += 1j*torch.einsum('ijm,j,m->i', D_inv[:, j_start:j_start+num_modes, m_start:m_start+num_modes], eff_derv, eff_derv)*DE_k
                    HT_3_Tensor += DE_m * DE_j * DE_k
                HT_3_Tensor = -HT_3_Tensor
                #FOR 4 DIPOLE CONTRIBUTION
                k_start = 0
                j_start = num_modes
                m_start = 2*num_modes
                n_start = 3*num_modes
                D_jk = torch.einsum('ijk,j,k->i', D_inv[:, j_start:j_start + num_modes, k_start:k_start+num_modes], eff_derv, eff_derv)
                D_mn = torch.einsum('imn,m,n->i', D_inv[:,m_start: m_start + num_modes, n_start:n_start + num_modes], eff_derv, eff_derv)
                HT_4_Tensor += -D_jk*D_mn
                D_km = torch.einsum('ikm,k,m->i', D_inv[:, k_start:k_start+num_modes, m_start:m_start+num_modes], eff_derv, eff_derv)
                D_jn = torch.einsum('ijn,j,n->i', D_inv[:,j_start:j_start+num_modes, n_start:n_start+num_modes], eff_derv, eff_derv)
                HT_4_Tensor += -D_km*D_jn
                D_kn = torch.einsum('ikn,k,n->i', D_inv[:, k_start:k_start+num_modes, n_start:n_start+num_modes], eff_derv, eff_derv)
                D_jm = torch.einsum('ijm,j,m->i', D_inv[:, j_start:j_start+num_modes, m_start:m_start + num_modes], eff_derv, eff_derv)
                HT_4_Tensor+= - D_kn*D_jm
                DE_m = torch.einsum('im,m->i', D_inv_E_Tensor[:, m_start : m_start +num_modes], eff_derv)
                DE_n = torch.einsum('in,n->i', D_inv_E_Tensor[:, n_start : n_start +num_modes], eff_derv)
                HT_4_Tensor += 1j*D_jk*DE_m*DE_n
                DE_j = torch.einsum('ij,j->i', D_inv_E_Tensor[:, j_start:j_start + num_modes], eff_derv)
                HT_4_Tensor += 1j*D_km*DE_j*DE_n
                DE_k = torch.einsum('ik,k->i', D_inv_E_Tensor[:, k_start: k_start + num_modes], eff_derv)
                HT_4_Tensor += 1j*D_jm*DE_k*DE_n
                HT_4_Tensor += 1j*D_kn* DE_j*DE_m
                HT_4_Tensor += 1j*D_jn*DE_k*DE_m
                HT_4_Tensor += 1j*D_mn*DE_k*DE_j
                HT_4_Tensor += DE_j*DE_k*DE_m*DE_n
                R1_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[0] + dipole_comp**2 * HT_2_Tensor[0] + dipole_comp*HT_3_Tensor[0] + HT_4_Tensor[0]
                R2_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[1] + dipole_comp**2 * HT_2_Tensor[1] + dipole_comp*HT_3_Tensor[1] + HT_4_Tensor[1]
                R3_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[2] + dipole_comp**2 * HT_2_Tensor[2] + dipole_comp*HT_3_Tensor[2] + HT_4_Tensor[2]
                R4_HT_PREF += dipole_comp**4 + dipole_comp**3 * HT_1_Tensor[3] + dipole_comp**2 * HT_2_Tensor[3] + dipole_comp*HT_3_Tensor[3] + HT_4_Tensor[3]
            return R1_HT_PREF, R1, R2_HT_PREF, R2, R3_HT_PREF, R3, R4_HT_PREF, R4



with torch.no_grad():
    def Compute_fc2des_response_function(gs_freqs, ex_freqs,J_mat, k_vec,P_mat, t_dat,t_2,Temp, device: str): 
        #driver for computing full response function. will allow GPU and CPU calculations along with t=0 limits... the later taking less of a focus
        t_dat[0] = 10 ###FIXES T=0 ISSUE
        num_modes = len(k_vec)
        KbT = Temp * (8.6173303*10.0**(-5.0)/27.211396132)
        R_1 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        R_2 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad= False)
        R_3 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        R_4 = torch.zeros((len(t_dat), len(t_dat)), dtype= torch.complex128,requires_grad=False)
        if device == "GPU":
            R_1 = R_1
            R_2 = R_2
            R_3 = R_3
            R_4 = R_4
            J_mat = J_mat.cuda()
            gs_freqs = gs_freqs.cuda()
            ex_freqs = ex_freqs.cuda()
            k_vec = k_vec.cuda()
            P_mat = P_mat.cuda()
            if num_modes <= 110:
                for t_3_ind in range(0, len(t_dat)):#######FIX THIS DONT BE DUMB
                    for t_1_ind in range(0, len(t_dat)):
                        t_1 = t_dat[t_1_ind]
                        t_3= t_dat[t_3_ind] 
                        tau_sets = torch.tensor([[-t_1 -1j/KbT, -t_2, -t_3, t_1+t_2 +t_3],[-t_3, t_2 + t_3, t_1 -1j/KbT, -t_1 - t_2],
                                    [-t_2 - t_3, t_3, t_1 +t_2 -1j/KbT, -t_1], [-t_1 -t_2 - t_3 -1j/KbT, t_3, t_2, t_1]]).to(torch.complex64)
                        r1,r2,r3,r4 = torch_response_at_time_GPU(gs_freqs, ex_freqs, J_mat, k_vec, num_modes, tau_sets,  P_mat)
                        R_1[t_1_ind, t_3_ind] = r1.detach().cpu()
                        del r1
                        R_2[t_1_ind, t_3_ind] = r2.detach().cpu()
                        del r2 
                        R_3[t_1_ind, t_3_ind] = r3.detach().cpu()
                        del r3
                        R_4[t_1_ind, t_3_ind] = r4.detach().cpu()
                        del r4
                        torch.cuda.empty_cache()
                    print(f"CALCULATING RESPONSE(GPU):  %{np.round(t_3_ind/ len(t_dat) ,1)*100}", end='\r')
            if num_modes > 110:
                print("NO BATCH INVERSE CALCULATION!")
                for t_3_ind in range(0, len(t_dat)):
                    for t_1_ind in range(0, len(t_dat)):
                        t_1 = t_dat[t_1_ind]
                        t_3= t_dat[t_3_ind] 
                        tau_sets = torch.tensor([[-t_1 -1j/KbT, -t_2, -t_3, t_1+t_2 +t_3],[-t_3, t_2 + t_3, t_1 -1j/KbT, -t_1 - t_2],
                                    [-t_2 - t_3, t_3, t_1 +t_2 -1j/KbT, -t_1], [-t_1 -t_2 - t_3 -1j/KbT, t_3, t_2, t_1]]).to(torch.complex64)
                        r1,r2,r3,r4 = torch_response_at_time_GPU_NOBATCH(gs_freqs, ex_freqs, J_mat, k_vec, num_modes, tau_sets,   P_mat)
                        R_1[t_1_ind, t_3_ind] = r1.detach().cpu()
                        del r1
                        R_2[t_1_ind, t_3_ind] = r2.detach().cpu()
                        del r2 
                        R_3[t_1_ind, t_3_ind] = r3.detach().cpu()
                        del r3
                        R_4[t_1_ind, t_3_ind] = r4.detach().cpu()
                        del r4
                        torch.cuda.empty_cache()
                    print(f"CALCULATING RESPONSE(GPU):  %{np.round(t_3_ind/ len(t_dat) ,1)*100}", end='\r')
        if device == "CPU":
            if torch.equal(J_mat.real, torch.eye(num_modes)):
                for t_3_ind in range(0, len(t_dat)):
                    for t_1_ind in range(0, len(t_dat)):
                        t_1 = t_dat[t_1_ind]
                        t_3= t_dat[t_3_ind]    
                        tau_sets = torch.tensor([[-t_1 -1j/KbT, -t_2, -t_3, t_1+t_2 +t_3],[-t_3, t_2 + t_3, t_1 -1j/KbT, -t_1 - t_2],
                                    [-t_2 - t_3, t_3, t_1 +t_2 -1j/KbT, -t_1], [-t_1 -t_2 - t_3 -1j/KbT, t_3, t_2, t_1]]).to(torch.complex64)
                        
                        r1,r2,r3,r4 = compute_response_at_time_NO_ROT(gs_freqs, ex_freqs,  k_vec, num_modes, tau_sets, P_mat)
                        R_1[t_1_ind, t_3_ind] = r1 
                        R_2[t_1_ind, t_3_ind] = r2 
                        R_3[t_1_ind, t_3_ind] = r3 
                        R_4[t_1_ind, t_3_ind] = r4 
                    print(f"CALCULATING RESPONSE(CPU, J-OFF): %{np.round(t_3_ind/ len(t_dat) ,1)*100}", end = '\r')

            if not  torch.equal(J_mat.real, torch.eye(num_modes)):
  
                for t_1_ind in range(0, len(t_dat)):
                    for t_3_ind in range(0, len(t_dat)):
                        t_1 = t_dat[t_1_ind]
                        t_3= t_dat[t_3_ind]
                        tau_sets = torch.tensor([[-t_1 -1j/KbT, -t_2, -t_3, t_1+t_2 +t_3],[-t_3, t_2 + t_3, t_1 -1j/KbT, -t_1 - t_2],
                                    [-t_2 - t_3, t_3, t_1 +t_2 -1j/KbT, -t_1], [-t_1 -t_2 - t_3 -1j/KbT, t_3, t_2, t_1]]).to(torch.complex64)
                        r1,r2,r3,r4 = compute_response_at_time_CPU(gs_freqs, ex_freqs, J_mat, k_vec, num_modes, tau_sets, P_mat)
                        R_1[t_1_ind, t_3_ind] = r1 
                        R_2[t_1_ind, t_3_ind] = r2 
                        R_3[t_1_ind, t_3_ind] = r3 
                        R_4[t_1_ind, t_3_ind] = r4 
                        print(f"CALCULATING RESPONSE(CPU, J-ON):  %{np.round(t_1_ind/ len(t_dat) ,1)*100}", end='\r')
        R_1 = R_1.detach().cpu().numpy()
        R_2 = R_2.detach().cpu().numpy()
        R_3 = R_3.detach().cpu().numpy()
        R_4 = R_4.detach().cpu().numpy()
        return R_1, R_2, R_3, R_4



def Stabilize_phase(R_function):
    response_function = np.copy(R_function)
    phi = np.zeros_like(response_function)
    mag = np.zeros_like(response_function)
    for i in range(0, response_function.shape[0]):
        for j in range(0, response_function.shape[0]):
            mag[i,j], phi[i,j] = cmath.polar(response_function[i,j])
    phi = phi.real
    mag = mag.real
    count = 0
    while count < response_function.shape[0]:
        for i in range(0, response_function.shape[0] -1):
            jump = phi[i+1, count] - phi[i,count]
            if np.abs(jump) > 0.7* np.pi:
                phi[i+1:, count] -= jump
        count +=1
    count = 0
    while count < response_function.shape[0]:
        for i in range(0, response_function.shape[0] -1):
            jump = phi[count, i+1] - phi[count,i]
            if np.abs(jump) >  0.7*np.pi:
                phi[count, i+1:] -= jump
        count +=1
    tamed_response = mag * np.exp(1j * phi)
    return tamed_response
 