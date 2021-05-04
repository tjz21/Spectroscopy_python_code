#! /usr/bin/env python

import sys  
import os.path
import numpy as np
import math
import numba
from spec_pkg.constants import constants as const


def get_gs_energy(input_path_gs):
	search_phrase='FINAL ENERGY:'
	searchfile = open(input_path_gs,"r")
	line_count=0
	keyword_line=999999999
	for line in searchfile:
		if search_phrase in line and keyword_line==999999999:
			keyword_line=line_count
		line_count=line_count+1
	# if we haven't found reference geometry, break
	if keyword_line==999999999:
		sys.exit('Error: Could not find final ground state energy in Terachem file:  '+input_path)
	linefile=open(input_path_gs,"r")
	lines=linefile.readlines()
	energy_line=lines[keyword_line].split()
	return float(energy_line[2])

# root determines which root is the root of interest. by default this is 1
def get_ex_energy_dipole_mom(input_path_ex,root):
	search_phrase='Final Excited State Results:'
	searchfile = open(input_path_ex,"r")
	line_count=0
	keyword_line=999999999
	for line in searchfile:
		if search_phrase in line and keyword_line==999999999:
			keyword_line=line_count
		line_count=line_count+1
	if keyword_line==999999999:
		sys.exit('Error: Could not find excited state energy and dipole moment in Terachem file:  '+input_path)
	searchfile.close()
	linefile=open(input_path_ex,"r")
	lines=linefile.readlines()
	energy_line=lines[keyword_line+3+root].split()
	energy=float(energy_line[1])
	ex_energy=float(energy_line[2])/const.Ha_to_eV # convert to Ha 

	searchfile = open(input_path_ex,"r")
	search_phrase='Transition dipole moments:'
	line_count=0
	keyword_line=999999999
	for line in searchfile:
		if search_phrase in line and keyword_line==999999999:
			keyword_line=line_count
		line_count=line_count+1
	if keyword_line==999999999:
		sys.exit('Error: Could not find excited state energy and dipole moment in Terachem file:  '+input_path)
	searchfile.close()
	linefile=open(input_path_ex,"r")
	lines=linefile.readlines()	
	dipole_line=lines[keyword_line+3+root].split()

	dipole_mom=np.zeros(3)
	dipole_mom[0]=float(dipole_line[1])
	dipole_mom[1]=float(dipole_line[2])
	dipole_mom[2]=float(dipole_line[3])
	return energy,dipole_mom

def get_e_adiabatic_dipole(input_path_gs,input_path_ex,root):
	ex_energy,dipole_mom=get_ex_energy_dipole_mom(input_path_ex,root)
	gs_energy=get_gs_energy(input_path_gs)
	return dipole_mom,ex_energy-gs_energy

# Test this
def get_specific_dipole(input_path,displacement,root,ref_dipole):
        search_phrase1='*** Displacement '+str(displacement)+' ***'
        search_phrase2='Transition dipole moments:'
        dipole_vec=np.zeros(3)

        searchfile = open(input_path,"r")
        line_count=0
        keyword_line=999999999
        for line in searchfile:
                if search_phrase1 in line and keyword_line==999999999:
                        keyword_line=line_count
                line_count=line_count+1
        total_line_number=line_count-1
        linefile=open(input_path,"r")
        lines=linefile.readlines()
        # if we haven't found specific displacement, break
        if keyword_line==999999999:
                sys.exit('Error: Could not find transition dipole moment for Displacement '+str(displacement))
        line_count=keyword_line
        keyword_line2=999999999
        while line_count<total_line_number:
                if search_phrase2 in lines[line_count] and keyword_line2==999999999:
                        keyword_line2=line_count
                line_count=line_count+1
        # if we havent found the gradient following the displacement, break
        if keyword_line2==999999999:
                sys.exit('Error: Could not find transition dipole moment for Displacement '+str(displacement))
        line_start=keyword_line2+3+root
        current_line=lines[line_start].split()
	 
        dipole_vec[0]=float(current_line[1])
        dipole_vec[1]=float(current_line[2])
        dipole_vec[2]=float(current_line[3])

        # Check sign relative to reference dipole moment
        if np.dot(dipole_vec,ref_dipole)/(np.dot(dipole_vec,dipole_vec)*np.dot(ref_dipole,ref_dipole))<-0.5:
                dipole_vec=-1.0*dipole_vec

        return dipole_vec

# Works
def get_specific_grad(input_path,displacement,num_atoms):
	search_phrase1='*** Displacement '+str(displacement)+' ***'
	search_phrase2='dE/dX            dE/dY            dE/dZ'

	grad_vec=np.zeros(num_atoms*3)

	searchfile = open(input_path,"r")
	line_count=0
	keyword_line=999999999
	for line in searchfile:
		if search_phrase1 in line and keyword_line==999999999:
			keyword_line=line_count
		line_count=line_count+1
	total_line_number=line_count-1
	linefile=open(input_path,"r")
	lines=linefile.readlines()
	# if we haven't found specific displacement, break
	if keyword_line==999999999:
		sys.exit('Error: Could not find gradient for Displacement '+str(displacement))
	line_count=keyword_line
	keyword_line2=999999999
	while line_count<total_line_number:
		if search_phrase2 in lines[line_count] and keyword_line2==999999999:
			keyword_line2=line_count
		line_count=line_count+1
	# if we havent found the gradient following the displacement, break
	if keyword_line2==999999999:
		sys.exit('Error: Could not find gradient for Displacement '+str(displacement))
	line_start=keyword_line2+1
	# loop over number of atoms to extract the gradient vec
	atom_count=0
	while atom_count<num_atoms:
		current_line=lines[line_start+atom_count].split()
		dx=float(current_line[0])
		dy=float(current_line[1])
		dz=float(current_line[2])
		grad_vec[atom_count*3]=dx
		grad_vec[atom_count*3+1]=dy
		grad_vec[atom_count*3+2]=dz
		atom_count=atom_count+1

	return grad_vec	

def get_dipole_deriv_from_terachem(input_path,frozen_atom_list,num_frozen_atoms,root):
	# ref dipole: 
        energy,ref_dipole=get_ex_energy_dipole_mom(input_path,root)


        num_atoms=frozen_atom_list.shape[0]
        dipole_deriv=np.zeros((3,num_atoms*3))
        displacement=0.0
        # find displacement:
        searchphrase='Using displacements of '
        searchfile = open(input_path,"r")
        line_count=0
        keyword_line=999999999
        for line in searchfile:
                if searchphrase in line and keyword_line==999999999:
                        keyword_line=line_count
                line_count=line_count+1
        searchfile.close()
        if keyword_line < 9999999:
                linefile=open(input_path,"r")
                lines=linefile.readlines()
                keyword=lines[keyword_line].split()
                displacement=float(keyword[3])
        else:
                sys.exit('Error: Could not find displacement used in numerical Hessian calculation in Terachem file')	

        print('FD parameter: 2*displacement')
        print(displacement*2)

        # Now construct the dipole_derivative. First displacement is the (x+delta x) displacement, then the (x-delta x) displacement 
        unfrozen_atoms=num_atoms-num_frozen_atoms
        unfrozen_count=0
        total_atom_count=0
        while total_atom_count<num_atoms:
                # check whether this is a frozen or an unfrozen atom:
                if frozen_atom_list[total_atom_count]==0: # unfrozen

                        # loop over xyz coordinates:
                        xyz_count=0
                        while xyz_count<3:
                                print('Getting dipole ', unfrozen_count, ' of ',3*unfrozen_atoms)
                                dipole1=get_specific_dipole(input_path,unfrozen_count*2+1,root,ref_dipole)
                                dipole2=get_specific_dipole(input_path,unfrozen_count*2+2,root,ref_dipole)

                                print(dipole1)
                                print(dipole2)
                                print(dipole1-dipole2)

                                # build the effective row of the Hessian through the finite difference scheme. 
                                fd_dipole=(dipole1-dipole2)/(2.0*displacement)

                                # this seems to be the correct approach. This way, off diagonal Hessian elements are the average of the numerical and the 
                                # analytical force constants
                                dipole_deriv[:,total_atom_count*3+xyz_count]=fd_dipole[:]

                                print(fd_dipole)

                                unfrozen_count=unfrozen_count+1
                                xyz_count=xyz_count+1

                total_atom_count=total_atom_count+1

	# no need to symmetrize like in the Hessian. Just return

        print('Full Dipole derivative. Units should be in a.u')
        print(dipole_deriv)

        return dipole_deriv


#Works
def get_hessian_from_terachem(input_path,frozen_atom_list,num_frozen_atoms):
	num_atoms=frozen_atom_list.shape[0] # frozen atom list is a list Natoms long, where frozen atoms
					    # are labelled by 1 and unfrozen atoms labelled by 0
	hessian=np.zeros((num_atoms*3,num_atoms*3))
	displacement=0.0
	# find displacement:
	searchphrase='Using displacements of '
	searchfile = open(input_path,"r")
	line_count=0
	keyword_line=999999999
	for line in searchfile:
		if searchphrase in line and keyword_line==999999999:
			keyword_line=line_count
		line_count=line_count+1
	searchfile.close()	
	if keyword_line < 9999999:
		linefile=open(input_path,"r")
		lines=linefile.readlines()
		keyword=lines[keyword_line].split()
		displacement=float(keyword[3])
	else:
		sys.exit('Error: Could not find displacement used in numerical Hessian calculation in Terachem file')

	# Now construct the Hessian. First displacement is the (x+delta x) displacement, then the (x-delta x) displacement 
	unfrozen_atoms=num_atoms-num_frozen_atoms
	unfrozen_count=0
	total_atom_count=0
	while total_atom_count<num_atoms: 
		# check whether this is a frozen or an unfrozen atom:
		if frozen_atom_list[total_atom_count]==0: # unfrozen

			# loop over xyz coordinates:
			xyz_count=0
			while xyz_count<3:
				print('Getting gradient ', unfrozen_count, ' of ',3*unfrozen_atoms)
				grad1=get_specific_grad(input_path,unfrozen_count*2+1,num_atoms)
				grad2=get_specific_grad(input_path,unfrozen_count*2+2,num_atoms)

				# build the effective row of the Hessian through the finite difference scheme. 
				fd_grad=(grad1-grad2)/(2.0*displacement)
			
				# this seems to be the correct approach. This way, off diagonal Hessian elements are the average of the numerical and the 
				# analytical force constants
				hessian[total_atom_count*3+xyz_count,:]=fd_grad[:]
		
				unfrozen_count=unfrozen_count+1
				xyz_count=xyz_count+1

		else:  # frozen atom
			# loop over xyz coordinates:
			xyz_count=0
			while xyz_count<3:
				# set block diagonal elements corresponding to purely frozen atoms to 1
				hessian[total_atom_count*3+xyz_count,total_atom_count*3+xyz_count]=1.0
				xyz_count=xyz_count+1

		total_atom_count=total_atom_count+1

	# currently the Hessian has only one off diagonal block if there are frozen atoms. Need to fix this
	i_count=0
	while i_count<num_atoms:
		j_count=i_count
		while j_count<num_atoms:
			if frozen_atom_list[i_count]==0 and frozen_atom_list[j_count]==1: # icount refers to unfrozen and jcount refers to frozen atom
				# make sure the hessian element with the frozen atom as its first index is set.
				for xyz1 in range(3):
					for xyz2 in range(3):
						hessian[j_count*3+xyz2,i_count*3+xyz1]=hessian[i_count*3+xyz1,j_count*3+xyz2]
			j_count=j_count+1
		i_count=i_count+1	


	# symmetrize hessian h_sym=0.5*(H+H.T)
	sym_hessian=0.5*(hessian+np.transpose(hessian))

	return sym_hessian

# extract both list of atomic masses and reference geometry from terachem
# Works
def get_masses_geom_from_terachem(input_path, num_atoms):
	geom=np.zeros((num_atoms,3))
	masses=np.zeros(num_atoms)
	search_phrase='*** Reference Geometry ***'
	searchfile = open(input_path,"r")

	line_count=0
	keyword_line=999999999
	for line in searchfile:
		if search_phrase in line and keyword_line==999999999:
			keyword_line=line_count
		line_count=line_count+1
	linefile=open(input_path,"r")
	lines=linefile.readlines()
	# if we haven't found reference geometry, break
	if keyword_line==999999999:
		sys.exit('Error: Could not find reference geometry in terachem file '+input_path)

	line_start=keyword_line+2
	atom_count=0
	while atom_count<num_atoms:
		current_line=lines[line_start+atom_count].split()
		geom[atom_count,0]=float(current_line[1])
		geom[atom_count,1]=float(current_line[2])
		geom[atom_count,2]=float(current_line[3])
		elem=current_line[0] # match element with mass number then store in masses array
		if elem=='H':
			masses[atom_count]=const.Mass_list[0]
		elif elem=='He':
			masses[atom_count]=const.Mass_list[1]
		elif elem=='Li':
			masses[atom_count]=const.Mass_list[2]
		elif elem=='Be':
			masses[atom_count]=const.Mass_list[3]
		elif elem=='B':
			masses[atom_count]=const.Mass_list[4]
		elif elem=='C':
			masses[atom_count]=const.Mass_list[5]
		elif elem=='N':
			masses[atom_count]=const.Mass_list[6]
		elif elem=='O':
			masses[atom_count]=const.Mass_list[7]
		elif elem=='F':
			masses[atom_count]=const.Mass_list[8]
		elif elem=='Ne':
			masses[atom_count]=const.Mass_list[9]
		elif elem=='Na':
			masses[atom_count]=const.Mass_list[10]
		elif elem=='Mg':
			masses[atom_count]=const.Mass_list[11]	
		elif elem=='Al':
			masses[atom_count]=const.Mass_list[12]
		elif elem=='Si':
			masses[atom_count]=const.Mass_list[13]
		elif elem=='P':
			masses[atom_count]=const.Mass_list[14]
		elif elem=='S':
			masses[atom_count]=const.Mass_list[15]
		elif elem=='Cl':
			masses[atom_count]=const.Mass_list[16]
		elif elem=='Ar':
			masses[atom_count]=const.Mass_list[17]

		else:
			sys.exit('Error: Could not find atomic mass for element '+elem)
	
		atom_count=atom_count+1

	return masses,geom

