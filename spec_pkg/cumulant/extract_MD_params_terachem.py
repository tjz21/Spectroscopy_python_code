#! /usr/bin/env python
  
import os.path
import numpy as np
import math
import sys
import linecache
import numba
from spec_pkg.constants import constants as const

def apply_eckhart_conditions(dipole_mom, geom, ref_dipole,prev_T,prev_T_grad,reference_geom,mass_vec):
    # sign of the dipole moment is arbitrary. Do we have a flip? Compare to previous/reference dipole moment
#    if np.dot(dipole_mom,reference_dipole)<np.dot(-1.0*dipole_mom,reference_dipole):
#        dipole_mom=-1.0*dipole_mom

    com_geom=np.zeros(3)
    com_ref=np.zeros(3)
    total_mass=np.sum(mass_vec)
    for i in range(mass_vec.shape[0]):
        com_geom[:]=com_geom[:]+mass_vec[i]*geom[i,:]
        com_ref[:]=com_ref[:]+mass_vec[i]*reference_geom[i,:]

    com_geom=com_geom/total_mass
    com_ref=com_ref/total_mass

    for i in range(geom.shape[0]):
        geom[i,:]=geom[i,:]-com_geom
        reference_geom[i,:]=reference_geom[i,:]-com_ref

    # successufly shifted both geometries to the COM frame. Now apply eckhart conditions
    # build Eckhard rotation matrix
    amat=np.zeros((3,3))
    for i in range(geom.shape[0]):
        for j in range(3):
            for k in range(3):
                amat[j,k]=amat[j,k]+geom[i,j]*reference_geom[i,k]*mass_vec[i]
    a1=np.dot(amat,np.transpose(amat))
    a2=np.dot(np.transpose(amat),amat)

    evals1,evecs1=np.linalg.eigh(a1)
    evals2,evecs2=np.linalg.eigh(a2)

    # Ensure the correct handedness of the coordinate system: Evec 3 is the cross product of
    # Evect 1x evec 2
    evecs1[2,:]=np.cross(evecs1[0,:],evecs1[1,:])
    evecs2[2,:]=np.cross(evecs2[0,:],evecs2[1,:])

    # now address the sign problem. Ajust evecs1 by keeping evecs2 unchanged. 
    inner_prod=np.zeros(3)
    inner_prod[0]=np.dot(evecs1[0,:],evecs2[0,:])
    inner_prod[1]=np.dot(evecs1[1,:],evecs2[1,:])
    inner_prod[2]=np.dot(evecs1[2,:],evecs2[2,:])

    # find smallest inner product
    smallest_index=0
    smallest_val=inner_prod[smallest_index]
    for i in range(1,3):
        if smallest_val>inner_prod[i]:
            smallest_val=inner_prod[i]
            smallest_index=i

    # flip signs for two largest inner products IF inner products are smaller than zeros
    for i in range(3):
        if i!=smallest_index:
            if inner_prod[i]<0.0:
                evecs1[i,:]=-1.0*evecs1[i,:]

    # now build rotation matrix
    Tmat=np.dot(evecs2,np.transpose(evecs1))

    # Now that we have constructed Tmat, make sure that it is close enough to the previous matrix to 
    # guarantee a smooth dipole mom
    overlap=np.dot(Tmat[0,:],prev_T[0,:])+np.dot(Tmat[1,:],prev_T[1,:])+np.dot(Tmat[2,:],prev_T[2,:])
#    if overlap<2.7:  # if it changes too much from previous iteration:
#        # for now just set to previous Tmat
#        Tmat_temp=prev_T+prev_T_grad
#	Q,R=np.linalg.qr(Tmat_temp)
#	Tmat=Q
	# Watch for sign changes:
        #for i in range(3):
        #    if np.dot(Tmat_temp[i,:],prev_T[i,:])<0.0:
	#	Tmat[i,:]=-1.0*Tmat_temp[i,:]
        #    else:
        #        Tmat[i,:]=Tmat_temp[i,:]

#    print Tmat
    dipole_transformed=np.dot(Tmat,dipole_mom)

    # sign of the dipole moment can still arbitrarily flip due to sign choice in QR
    if np.dot(dipole_transformed,dipole_transformed)>0.0001:  # do not enforce this condition for very small dipole moments 
        for i in range(3):
            if abs(-1.0*dipole_transformed[i]-ref_dipole[i])<abs(dipole_transformed[i]-ref_dipole[i]):
               dipole_transformed[i]=-1.0*dipole_transformed[i]

    return dipole_transformed,Tmat

def get_coors_frame(frame_num,num_atoms,filename):
    coors=np.zeros((num_atoms,3))
#    linefile=open(filename)
#    lines=linefile.readlines()
    line_start=frame_num*(num_atoms+2)+2
    for i in range(num_atoms):
#        current_line=lines[line_start+i].split()
        current_line=linecache.getline(filename,line_start+1+i).split()
        coors[i,0]=float(current_line[1])
        coors[i,1]=float(current_line[2])
        coors[i,2]=float(current_line[3])

    return coors

def get_atomic_masses(num_atoms,filename):
    masses=np.zeros(num_atoms)
#    linefile=open(filename)
#    lines=linefile.readlines()
    linestart=2
    for i in range(num_atoms):
        current_line=linecache.getline(filename,linestart+1+i).split()
#       current_line=lines[linestart+i].split()
        elem=current_line[0]
        if elem=='H':
            masses[i]=const.Mass_list[0]
        elif elem=='He':
            masses[i]=const.Mass_list[1]
        elif elem=='Li':
            masses[i]=const.Mass_list[2]
        elif elem=='Be':
            masses[i]=const.Mass_list[3]
        elif elem=='B':
            masses[i]=const.Mass_list[4]
        elif elem=='C':
            masses[i]=const.Mass_list[5]
        elif elem=='N':
             masses[i]=const.Mass_list[6]
        elif elem=='O':
             masses[i]=const.Mass_list[7]
        elif elem=='F':
             masses[i]=const.Mass_list[8]
        elif elem=='Ne':
             masses[i]=const.Mass_list[9]
        else:
             sys.exit('Error: Could not find atomic mass for element '+elem)


    return masses


def get_energy_osc_frame(frame_num,excitation_index,filename):
    energy_osc=np.zeros(2)
    search_phrase="Final Excited State Results:"
    excit_counter=0
    line_count=0
    searchfile=open(filename,"r")
    keyword_line=99999999999999
    for line in searchfile:
        if search_phrase in line:
            if excit_counter==frame_num:
                keyword_line=line_count+3+excitation_index
                break
            excit_counter=excit_counter+1
        line_count=line_count+1

    searchfile.close()
#    linefile=open(filename)
#    lines=linefile.readlines()		
    if keyword_line != 99999999999999:
        energy_line=linecache.getline(filename,keyword_line+1).split()
#        energy_line=lines[keyword_line].split()
        energy_osc[0]=float(energy_line[2]) # do NOT convert to hartree. This is done internally.
        energy_osc[1]=float(energy_line[3])

    return energy_osc


def get_dipole_mom_frame(frame_num,excitation_index,filename):
    dipole_mom=np.zeros(3)
    energy_osc=np.zeros(2)
    search_phrase="Transition dipole moments:"
    excit_counter=0
    line_count=0
    searchfile=open(filename,"r")
    keyword_line=99999999999999
    for line in searchfile:
        if search_phrase in line:
            if excit_counter==frame_num:
                keyword_line=line_count+3+excitation_index
                break
            excit_counter=excit_counter+1
        line_count=line_count+1

    searchfile.close()
#    linefile=open(filename) 
#    lines=linefile.readlines()
    if keyword_line != 99999999999999:
#        dipole_line=lines[keyword_line].split()
        dipole_line=linecache.getline(filename,keyword_line+1).split()
        dipole_mom[0]=float(dipole_line[1])
        dipole_mom[1]=float(dipole_line[2])
        dipole_mom[2]=float(dipole_line[3])

    return dipole_mom

# main driver routine reading in a terachem MD output file and coors.xyz file and constructs the relevant cumulant input 
# parameters of energy, oscillator strengths, and transition dipole moment. In order to enable HT calculations, the 
# transition dipole moments are rotated to project out rotation and translation
# a total of num_snapshots are read from file, and the first skip_snapshots are skipped to allow for example for 
# the system to equibrate at the beginning of an MD run. 
def get_full_energy_dipole_moms_from_MD(filename_energies,filename_geoms,num_atoms,excitation_index,num_snapshots,skip_snapshots):
    energies_dipoles=np.zeros((num_snapshots,5))
    dipoles_unrotated=np.zeros((num_snapshots,3))

    # first deal with the first snapshot, which is considered the reference snapshot for the Eckart conditions
    masses=get_atomic_masses(num_atoms,filename_geoms)
    energy_osc=get_energy_osc_frame(skip_snapshots,excitation_index,filename_energies)
    ref_dipole=get_dipole_mom_frame(skip_snapshots,excitation_index,filename_energies)
    ref_geom=get_coors_frame(skip_snapshots,num_atoms,filename_geoms)

    # HACK:
#    masses[0]=masses[0]+4.0   # break symmetry for applying the Eckart conditions
#    masses[9]=masses[9]+6.0
#    masses[7]=masses[7]+4.0

    prev_T=np.zeros((3,3))
    prev_T[0,0]=1.0
    prev_T[1,1]=1.0
    prev_T[2,2]=1.0  # Set previous T matrix to identity matrix
    prev_T_grad=np.zeros((3,3))

    energies_dipoles[0,0]=energy_osc[0]
    energies_dipoles[0,1]=energy_osc[1]
    energies_dipoles[0,2]=ref_dipole[0]
    energies_dipoles[0,3]=ref_dipole[1]
    energies_dipoles[0,4]=ref_dipole[2]

    dipoles_unrotated[0,:]=ref_dipole

    for i in range(1,num_snapshots):
        energy_osc=get_energy_osc_frame(i+skip_snapshots,excitation_index,filename_energies)
        dipole=get_dipole_mom_frame(i+skip_snapshots,excitation_index,filename_energies)
        geom=get_coors_frame(i+skip_snapshots,num_atoms,filename_geoms)
        # correct sign of dipole moment
        if np.dot(dipole,ref_dipole)<np.dot(-1.0*dipole,ref_dipole) and np.dot(dipole,dipole)>0.0001:  # do not enforce this condition for very small dipole moments 
            dipole=-1.0*dipole

        # now apply eckhart conditions
        dipole_transformed,Tmat=apply_eckhart_conditions(dipole, geom, ref_dipole,prev_T,prev_T_grad,ref_geom,masses)
        # for now, don't apply eckart conditions
        dipole_transformed=dipole

        # setting prev_T and prev_T_grad for new iteration
        prev_T_grad=Tmat-prev_T
        prev_T=Tmat

        print('Iteration number  '+str(i))
        print(energy_osc, dipole_transformed)

        # update energies dipole list
        energies_dipoles[i,0]=energy_osc[0]
        energies_dipoles[i,1]=energy_osc[1]
        energies_dipoles[i,2]=dipole_transformed[0]
        energies_dipoles[i,3]=dipole_transformed[1]
        energies_dipoles[i,4]=dipole_transformed[2]

        dipoles_unrotated[i,:]=dipole

        # now reset the reference dipole moment
        ref_dipole=dipole

   
    np.savetxt('dipoles_untransformed.dat',dipoles_unrotated)
    return energies_dipoles
