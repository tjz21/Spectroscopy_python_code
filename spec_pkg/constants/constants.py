#! /usr/bin/env python
import numpy as np
# Module containing all constants used in the code

fine_struct=0.0072973525693
hbar_in_eVfs=0.6582119514
ang_to_bohr=1.889725989
fs_to_Ha=2.418884326505*10.0**(-2.0)
Ha_to_eV=27.211396132
Ha_to_cm=219474.63
kb_in_Ha=8.6173303*10.0**(-5.0)/Ha_to_eV
emass_in_au=5.4857990*10**(-4.0)
hessian_freqs_to_cm=5140.487125351268
debye_in_au=0.3934303
gaussian_to_debye=1.0/6.5005  # Scaling factor translating gaussian output
				# of transition dipole moment derivative (Km/mol)^0.5
				# to Debye/(ang amu*0.5) 

# add all masses
Mass_list=np.zeros(18)
Mass_list[0]=1.0079  # Hydrogen
Mass_list[1]=4.002603 # Helium
Mass_list[2]=7.016004 #lithium
Mass_list[3]=9.012182 # Berylium
Mass_list[4]=11.009306 # Boron
Mass_list[5]=12.0000 # Carbon
Mass_list[6]=14.00307 # Nitrogen 
Mass_list[7]=15.99491 # Oxygen
Mass_list[8]=18.998403 # Fluorine
Mass_list[9]=19.992440 # Neon
Mass_list[10]=22.989769 # Sodium
Mass_list[11]=23.985041 # Magnesium
Mass_list[12]=26.981539 # Aluminium
Mass_list[13]=28.0855   # Silicon
Mass_list[14]=30.973762  # Phosphorus
Mass_list[15]=32.065	# Sulphur
Mass_list[16]=35.453    # Chlorine
Mass_list[17]=39.948    # Argon
