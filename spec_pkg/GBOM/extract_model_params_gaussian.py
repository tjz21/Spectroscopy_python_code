#! /usr/bin/env python

import os.path
import numpy as np
import math
import numba
from spec_pkg.constants import constants as const

def extract_transition_dipole(filename,target_state):
	dipole=np.zeros(3)
	searchfile=open(filename,"r") 
	line_count=0
	dipole_line=0
	for line in searchfile:
		searchphrase='Ground to excited state transition electric dipole moments' 
		if searchphrase in line:
			dipole_line=line_count
		line_count=line_count+1
	if dipole_line>0:
		dipole_line=dipole_line+1+target_state
		searchfile.close()
		linefile=open(filename,"r")
		lines=linefile.readlines()
		current_line=lines[dipole_line].split()
		dipole[0]=float(current_line[1])	
		dipole[1]=float(current_line[2])
		dipole[2]=float(current_line[3])
	return dipole      


def construct_vertical_gradient_model(filename_gs,filename_grad,num_modes):
	freq_gs,gs_nm,num_atoms_chromophore=extract_normal_mode_freqs_vertical_gradient(filename_gs,num_modes)
	ex_grad=extract_excited_state_gradient(filename_grad,num_atoms_chromophore)
	Jmat,Kvec,freq_ex=JKwex_from_vertical_gradient_model(gs_nm,freq_gs,ex_grad)

	return freq_gs,freq_ex,Kvec,Jmat

def JKwex_from_vertical_gradient_model(nm_gs,freq_gs,force_gs):
	Kvec=np.zeros(freq_gs.shape[0])
	freq_ex=freq_gs  # in the simplest vertical gradient model, ground and excited state freq are identical
			# we could potentially relax this condition, need to read vertical excitation energy at gs minimum from file



	Jmat=np.zeros((freq_gs.shape[0],freq_gs.shape[0]))
	for i in range(nm_gs.shape[0]):
		Jmat[i,i]=1.0   # Duschinsky matrix is the identity matrix 
		Kvec[i]=np.dot(force_gs[:],nm_gs[i,:])/(freq_ex[i])**2.0

	return Jmat,Kvec,freq_ex


# make sure to mass_weigh the gradient
def extract_excited_state_gradient(filename,num_atoms_chromophore):
	force=np.zeros(3*num_atoms_chromophore)
	searchfile = open(filename,"r")
	line_count=0
	force_line=0
	for line in searchfile:
		searchphrase='Forces (Hartrees/Bohr)'
		if searchphrase in line and force_line==0:
			force_line=line_count
		line_count=line_count+1

	searchfile.close()
	if force_line>0:
		linefile=open(filename,"r")
		lines=linefile.readlines()
		for i in range(num_atoms_chromophore):
			current_line=lines[force_line+3+i].split()
			current_mass=const.Mass_list[int(current_line[1])-1]/const.emass_in_au
			force[i*3]=float(current_line[2])/math.sqrt(current_mass)
			force[i*3+1]=float(current_line[3])/math.sqrt(current_mass)
			force[i*3+2]=float(current_line[4])/math.sqrt(current_mass)

	return force
	

# Extract ground state frequencies and also normal modes for a vertical grad calculation
def extract_normal_mode_freqs_vertical_gradient(filename,num_modes):
        freq_list=np.zeros(num_modes)
        # Start by finding out how many atoms we have in the system:
        searchfile = open(filename,"r")
        line_count=0
        freq_line=0
        num_atoms_chromophore=(num_modes)/3
        frozen_atoms=0
        for line in searchfile:
                searchphrase='Deg. of freedom'
                if searchphrase in line and freq_line==0:
                        freq_line=line_count
                line_count=line_count+1
        if freq_line>0:
                searchfile.close()
                linefile=open(filename,"r")
                lines=linefile.readlines()
                current_line=lines[freq_line].split()
                if int(current_line[3])<num_modes+1: # no frozen atoms
                        num_atoms_chromophore=int((num_modes+6)/3)
                        frozen_atoms=0
                        total_num_atoms=num_atoms_chromophore
                else:   # frozen atoms
                        print('Frozen atoms?')
                        total_num_atoms=int(((current_line[3])+6)/3)
                        frozen_atoms=total_num_atoms-num_atoms_chromophore
                        print(total_num_atoms,frozen_atoms,num_atoms_chromophore)
        # Done
        print(num_atoms_chromophore)

        # also return the normal modes
        nm_vec=np.zeros((num_modes,num_atoms_chromophore*3))

        searchfile = open(filename,"r")
       	line_count=0
        freq_line=0
        for line in searchfile:
                searchphrase='Frequencies'
                if searchphrase in line and freq_line==0:
                        freq_line=line_count
                line_count=line_count+1

        # found first line of frequencies:
        #check number of loops over frequency lines have to be 
        # performed. This requires us to figure out whether it is a HPmodes calculation or not.
        searchfile.close()
        if freq_line>0:
                linefile=open(filename,"r")
                lines=linefile.readlines()

                current_line=lines[freq_line].split()
                if len(current_line) == 5:
                        freqs_per_row=3

                        lines_between_freq_rows=int(num_atoms_chromophore)+7

                        num_freq_loops=int(num_modes/3)

                elif len(current_line) == 7:
                        freqs_per_row=5
                        if frozen_atoms == 0:
                                lines_between_freq_rows=int(num_modes+6)+7
                        else:
                                lines_between_freq_rows=int(num_modes+3*frozen_atoms)+7

                        num_freq_loops=int(num_modes/5)


        freq_counter=0
        if freq_line>0:
                row_counter=0
                searchfile.close()
                linefile=open(filename,"r")
                lines=linefile.readlines()
                while row_counter<num_freq_loops:
                        current_line_start=row_counter*lines_between_freq_rows+freq_line
                        current_line=lines[current_line_start].split()
                        if freqs_per_row==5:
                                freq_list[freq_counter]=float(current_line[2])
                                freq_list[freq_counter+1]=float(current_line[3])
                                freq_list[freq_counter+2]=float(current_line[4])
                                freq_list[freq_counter+3]=float(current_line[5])
                                freq_list[freq_counter+4]=float(current_line[6])

				# Now fill normal mode matrix
                                for i in range(num_atoms_chromophore*3):
                                        current_line=lines[current_line_start+5+i].split()
                                        nm_vec[freq_counter,i]=float(current_line[3])
                                        nm_vec[freq_counter+1,i]=float(current_line[4])
                                        nm_vec[freq_counter+2,i]=float(current_line[5])
                                        nm_vec[freq_counter+3,i]=float(current_line[6])
                                        nm_vec[freq_counter+4,i]=float(current_line[7])
                                freq_counter=freq_counter+5
                        elif freqs_per_row==3:
                                freq_list[freq_counter]=float(current_line[2])
                                freq_list[freq_counter+1]=float(current_line[3])
                                freq_list[freq_counter+2]=float(current_line[4])
                                freq_counter=freq_counter+3

                        row_counter=row_counter+1


        #deal with missing frequencies.
        if freq_counter<num_modes:
                missing_freqs=num_modes-freq_counter
                current_line_start=row_counter*lines_between_freq_rows+freq_line
                current_line=lines[current_line_start].split()
                missing_counter=0
                while missing_counter<missing_freqs:
                        freq_list[freq_counter+missing_counter]=float(current_line[2+missing_counter])
                        # now do missing nm_vec elements
                        for i in range(num_atoms_chromophore*3):
                                current_line_nm=lines[current_line_start+5+i].split()
                                nm_vec[freq_counter+missing_counter,i]=float(current_line_nm[missing_counter+3])

                        missing_counter=missing_counter+1


        return freq_list/const.Ha_to_cm, nm_vec, num_atoms_chromophore

def extract_normal_mode_freqs(filename,num_modes):
        freq_list=np.zeros(num_modes)
        # Start by finding out how many atoms we have in the system:
        searchfile = open(filename,"r")
        line_count=0
        freq_line=0
        num_atoms_chromophore=(num_modes)/3
        frozen_atoms=0
        for line in searchfile:
                searchphrase='Deg. of freedom'
                if searchphrase in line and freq_line==0:
                        freq_line=line_count
                line_count=line_count+1
        if freq_line>0:
                searchfile.close()
                linefile=open(filename,"r")
                lines=linefile.readlines()
                current_line=lines[freq_line].split()
                if int(current_line[3])<num_modes: # no frozen atoms
                        num_atoms_chromophore=(num_modes+6)/3
                        frozen_atoms=0
                        total_num_atoms=num_atoms_chromophore
                else:   # frozen atoms
                        total_num_atoms=(int(current_line[3])+6)/3
                        frozen_atoms=total_num_atoms-num_atoms_chromophore
	# Done

        searchfile = open(filename,"r")
        line_count=0
        freq_line=0
        for line in searchfile:
                searchphrase='Frequencies'
                if searchphrase in line and freq_line==0:
                        freq_line=line_count
                line_count=line_count+1

        # found first line of frequencies:
        #check number of loops over frequency lines have to be 
        # performed. This requires us to figure out whether it is a HPmodes calculation or not.
        searchfile.close()
        if freq_line>0:
                linefile=open(filename,"r")
                lines=linefile.readlines()

                current_line=lines[freq_line].split()
                if len(current_line) == 5:
                        freqs_per_row=3
                        lines_between_freq_rows=int(num_atoms_chromophore)+7
                        num_freq_loops=int(num_modes/3)
                elif len(current_line) == 7:
                        freqs_per_row=5
                        if frozen_atoms == 0:
                                lines_between_freq_rows=int(num_modes+6)+7
                        else:
                                lines_between_freq_rows=int(num_modes+3*frozen_atoms)+7

                        num_freq_loops=int(num_modes/5)

        freq_counter=0
        if freq_line>0:
                row_counter=0
                searchfile.close()
                linefile=open(filename,"r")
                lines=linefile.readlines()
                while row_counter<num_freq_loops:
                        current_line_start=row_counter*lines_between_freq_rows+freq_line
                        current_line=lines[current_line_start].split()
                        if freqs_per_row==5:
                                freq_list[freq_counter]=float(current_line[2])
                                freq_list[freq_counter+1]=float(current_line[3])
                                freq_list[freq_counter+2]=float(current_line[4])
                                freq_list[freq_counter+3]=float(current_line[5])
                                freq_list[freq_counter+4]=float(current_line[6])
                                freq_counter=freq_counter+5
                        elif freqs_per_row==3:
                                freq_list[freq_counter]=float(current_line[2])
                                freq_list[freq_counter+1]=float(current_line[3])
                                freq_list[freq_counter+2]=float(current_line[4])
                                freq_counter=freq_counter+3

                        row_counter=row_counter+1


	#deal with missing frequencies.
        if freq_counter<num_modes:
                missing_freqs=num_modes-freq_counter
                current_line_start=row_counter*lines_between_freq_rows+freq_line
                current_line=lines[current_line_start].split()
                missing_counter=0
                while missing_counter<missing_freqs:
                        freq_list[freq_counter+missing_counter]=float(current_line[2+missing_counter])
                        missing_counter=missing_counter+1


        return freq_list/const.Ha_to_cm


def convert_string_format(string):
	temp_string=''
	for x in string:
		if x=='D':
 			temp_string=temp_string+'E'
		else:
			temp_string=temp_string+x

	return temp_string

def extract_adiabatic_freq(filename):
	searchfile = open(filename,"r")
	line_count=0
	freq_line=0
	for line in searchfile:
		searchphrase='Energy of the 0-0 transition:'
		if searchphrase in line:
			freq_line=line_count
		line_count=line_count+1

	if freq_line>0:
		searchfile.close()
		linefile=open(filename,"r")
		lines=linefile.readlines()
		current_line=lines[freq_line].split()
		temp_string=convert_string_format(current_line[5])
		freq=float(temp_string)
		return freq/const.Ha_to_cm    # units

	else:
		return 0.0

def extract_Kmat(filename,num_modes):
	searchfile = open(filename,"r")
	line_count=0
	k_line=0
	for line in searchfile:
		searchphrase='Shift Vector'
		if searchphrase in line:
			k_line=line_count+3
		line_count=line_count+1

	if k_line>0:
		Kmat=np.zeros(num_modes)
		searchfile.close()
		linefile=open(filename,"r")
		lines=linefile.readlines()
		counter=0
		while counter<num_modes:
			current_line=lines[k_line+counter].split()
			temp_string=convert_string_format(current_line[1])
			Kmat[counter]=float(temp_string)
			counter=counter+1

		return Kmat

	else:
		return np.zeros((1,1))

def extract_duschinsky_mat(filename,num_modes):
	duschinsky_mat=np.zeros((num_modes,num_modes))
	searchfile = open(filename,"r")	
	line_count=0
	duschinsky_line=0
	for line in searchfile:
		searchphrase='Duschinsky matrix'
		if searchphrase in line and duschinsky_line==0:
			duschinsky_line=line_count
		line_count=line_count+1

	if duschinsky_line>0:
		start_line=duschinsky_line+5
		lines_between_rows=num_modes+1
		full_rows=int(num_modes/5)  # 5 columns in each row
		num_leftover_rows=num_modes-full_rows*5  # only if total value of normal modes is divisible by 5 this is zero
		searchfile.close()
		linefile=open(filename,"r")
		lines=linefile.readlines()
		row_counter=0
		# loop over full rows:
		while row_counter<full_rows:
			row_start=start_line+row_counter*lines_between_rows
			mode_counter=0
			while mode_counter<num_modes:
				current_line=lines[row_start+mode_counter].split()
				freq1=float(convert_string_format(current_line[1]))
				freq2=float(convert_string_format(current_line[2]))
				freq3=float(convert_string_format(current_line[3]))
				freq4=float(convert_string_format(current_line[4]))
				freq5=float(convert_string_format(current_line[5]))
				duschinsky_mat[mode_counter,row_counter*5+0]=freq1
				duschinsky_mat[mode_counter,row_counter*5+1]=freq2
				duschinsky_mat[mode_counter,row_counter*5+2]=freq3
				duschinsky_mat[mode_counter,row_counter*5+3]=freq4
				duschinsky_mat[mode_counter,row_counter*5+4]=freq5
				mode_counter=mode_counter+1
			row_counter=row_counter+1

		# get the last missing rows:
		row_start=start_line+row_counter*lines_between_rows
		mode_counter=0
		while mode_counter<num_modes:
			current_line=lines[row_start+mode_counter].split()
			missing_row_count=0
			while missing_row_count<num_leftover_rows:
				freq=float(convert_string_format(current_line[1+missing_row_count]))
				duschinsky_mat[mode_counter,row_counter*5+missing_row_count]=freq
				missing_row_count=missing_row_count+1
			mode_counter=mode_counter+1

		return duschinsky_mat

	else:
		return np.zeros((1,1))

