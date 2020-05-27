#! /usr/bin/env python

import os.path
import numpy as np
import math
import numba
from spec_pkg.constants import constants as const


def extract_normal_mode_freqs(filename,num_modes,frozen_atoms):
	freq_list=np.zeros(num_modes)
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
	# performed
	num_freq_loops=int(num_modes/5)
	# number of lines between successive frequency rows:
	lines_between_freq_rows=0
	if frozen_atoms==0:   # standard case, no frozen atoms
		lines_between_freq_rows=int(num_modes+6)+7             # number of atoms    
	else:      # nonstandard case, frozen atoms
		lines_between_freq_rows=int(num_modes+3*frozen_atoms)+7

	freq_counter=0
	if freq_line>0:
		row_counter=0
		searchfile.close()
		linefile=open(filename,"r")
		lines=linefile.readlines()
		while row_counter<num_freq_loops:
			current_line_start=row_counter*lines_between_freq_rows+freq_line
			current_line=lines[current_line_start].split()
			freq_list[freq_counter]=float(current_line[2])
			freq_list[freq_counter+1]=float(current_line[3])
			freq_list[freq_counter+2]=float(current_line[4])
			freq_list[freq_counter+3]=float(current_line[5])
			freq_list[freq_counter+4]=float(current_line[6])
			freq_counter=freq_counter+5
			row_counter=row_counter+1


	#deal with missing frequencies.
	if freq_counter<num_modes:
		missing_freqs=num_modes-freq_counter
		current_line_start=row_counter*lines_between_freq_rows+freq_line
		current_line=lines[current_line_start].split()
		#print current_line
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

