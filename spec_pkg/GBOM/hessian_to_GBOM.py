#! /usr/bin/env python
  
import numpy as np
import math
import numba
from ..constants import constants as const

#######################################################################
#  In this module, we take the ground and excited                     #
#  state optimized structure of a molecule, as well as                #
#  its Hessian at both points, and use this to construct              #
#  the ground and excited state Frequencies, J and K that             #
#  are needed to parameterize the given GBOM. We explicitly           #
#  consider two cases: One where no atoms are frozen in the           #
#  geometry optimization. Then we need to project out rotation        #
#  and translation contributions and end up with 3N-6 normal modes.   #
#  Alternatively, if atoms are frozen, we end up with 3N_unfrozen     #
#  modes.                                                             #
#######################################################################

def construct_freqs_J_K(coords_gs,coords_ex,hessian_gs,hessian_ex,dipole_mom,dipole_deriv,atomic_masses,frozen_atoms,frozen_atom_list):
	gs_freqs,gs_nm=freqs_NM_from_hessian(hessian_gs, coords_gs,atomic_masses,frozen_atoms,frozen_atom_list)
	ex_freqs,ex_nm=freqs_NM_from_hessian(hessian_ex, coords_ex,atomic_masses,frozen_atoms,frozen_atom_list)
	if frozen_atoms>0:
		print(frozen_atoms)
		J,K,dipole_transformed,dipole_deriv_nm=construct_J_K(gs_nm,ex_nm,coords_gs,coords_ex,dipole_mom,dipole_deriv,atomic_masses,True)
	else:
		print(frozen_atoms)
		J,K,dipole_transformed,dipole_deriv_nm=construct_J_K(gs_nm,ex_nm,coords_gs,coords_ex,dipole_mom,dipole_deriv,atomic_masses,False)


	# STUPID TEST: zero out low freqs
#	for i in range(gs_freqs.shape[0]):
#		if gs_freqs[i]<0.00045569: # 100cm-1 in ha
#			print gs_freqs[i]
#			ex_freqs[i]=gs_freqs[i]
#			K[i]=0.0
#			J[i,:]=0.0
#			J[:,i]=0.0
#			J[i,i]=1.0   # remove all coupling to these modes
	# Sanity check. simply get rid of negative freqs:
       
	for i in range(gs_freqs.shape[0]):
		if gs_freqs[i]<0.0:
			print('Warning: Negative ground state frequency detected. Removing:')
			gs_freqs[i]=abs(gs_freqs[i])
		if ex_freqs[i]<0.0:
			print('Warning: Negative excited state frequency detected. Removing:')
			ex_freqs[i]=abs(ex_freqs[i])
	

	return gs_freqs,ex_freqs,J,K,dipole_transformed,dipole_deriv_nm



        # dipole_deriv_cartesian has dimensions of (3,3*natoms) since it is a vector quantity 
def construct_dipole_deriv_nm(dipole_deriv_cartesian,nm_ex,atomic_masses):
	dipole_deriv_nm=np.zeros((dipole_deriv_cartesian.shape[0],nm_ex.shape[1])) # dipole deriv normal mode has dimension of 3,3N-6
	for i in range(atomic_masses.shape[0]):
		dipole_deriv_cartesian[:,i*3]=dipole_deriv_cartesian[:,i*3]/np.sqrt(atomic_masses[i])
		dipole_deriv_cartesian[:,i*3+1]=dipole_deriv_cartesian[:,i*3+1]/np.sqrt(atomic_masses[i])
		dipole_deriv_cartesian[:,i*3+2]=dipole_deriv_cartesian[:,i*3+2]/np.sqrt(atomic_masses[i])			

	dipole_deriv_nm[0,:]=np.sqrt(const.emass_in_au)*np.dot(dipole_deriv_cartesian[0,:],nm_ex)
	dipole_deriv_nm[1,:]=np.sqrt(const.emass_in_au)*np.dot(dipole_deriv_cartesian[1,:],nm_ex)
	dipole_deriv_nm[2,:]=np.sqrt(const.emass_in_au)*np.dot(dipole_deriv_cartesian[2,:],nm_ex) 

	# I think we want the dipole derivative in the format (Nnormal_modes,3), since this is what we 
	# have been assuming in other parts of the code
	return np.transpose(dipole_deriv_nm)

def construct_J_K(nm_gs,nm_ex,coords_gs,coords_ex,dipole_mom,dipole_deriv,atomic_masses,is_atom_constraint):
	# nm matrix is a matrix of Natoms*3,Natoms*3-Nconstraints
	# full Duschinsky matrix acts in normal mode space, thus needs to have 3N-Nconstraints,3N-Nconstraints 
	# dimensions nm^T*nm has the correct dimensions
	# if is_atom_constraint is true, we do NOT rotate and shift ex coordinates to overlap with gs coordinates to 
	# fulfill Eckart conditions. This is because that would lead to moving frozen atoms between the ground
	# and excited state reference geometry, which is wrong.

	if not is_atom_constraint and atomic_masses.shape[0]>2:
		# first make sure that gs and exited state coords have the same COM
		com_gs=np.zeros(3)
		com_ex=np.zeros(3)
		total_mass=np.sum(atomic_masses)
		for i in range(atomic_masses.shape[0]):
			com_gs[0]=com_gs[0]+atomic_masses[i]*coords_gs[i,0]
			com_gs[1]=com_gs[1]+atomic_masses[i]*coords_gs[i,1]
			com_gs[2]=com_gs[2]+atomic_masses[i]*coords_gs[i,2]
			com_ex[0]=com_ex[0]+atomic_masses[i]*coords_ex[i,0]
			com_ex[1]=com_ex[1]+atomic_masses[i]*coords_ex[i,1]
			com_ex[2]=com_ex[2]+atomic_masses[i]*coords_ex[i,2]

		com_gs=com_gs/total_mass
		com_ex=com_ex/total_mass

		for i in range(coords_gs.shape[0]):
			coords_gs[i,:]=coords_gs[i,:]-com_gs[:]
			coords_ex[i,:]=coords_ex[i,:]-com_ex[:]

		# now both coordinates are shifted such that their COM overlaps. 
		# build the Eckart rotation matrix: 
		amat=np.zeros((3,3))
		for i in range(coords_gs.shape[0]):
			# loop over xyz coords
			for j in range(3):
				for k in range(3):
					amat[j,k]=amat[j,k]+atomic_masses[i]*coords_ex[i,j]*coords_gs[i,k]

		sym_a=np.dot(amat,np.transpose(amat))
		inv_a=np.linalg.inv(amat)
		evals,evecs=np.linalg.eig(sym_a)
		# Check for negative eigenvalues. These can happen due to numerical issues if eval is very small
		# Very small eval might mean linear dependence--> should set Eckart rotation matrix to the identity 
		# matrix
		print('Eckart Evals')
		print(evals)
		linear_dependence=False
		for i in range(evals.shape[0]):
			if evals[i]<0.0 or abs(evals[i])<1.0e-10: # Check for small evals/linear dependence  
				linear_dependence=True

		if not linear_dependence:

			# create dmat, the diagonal matrix of sqrts 
			Dmat=np.zeros((3,3))
			for i in range(3):
				Dmat[i,i]=np.sqrt(evals[i])
			# now build rotation matrix U
			temp_mat=np.dot(Dmat,np.transpose(evecs))
			temp_mat2=np.dot(evecs,temp_mat)
			Tmat=np.dot(inv_a,temp_mat2)
		else: # if linear dependence, set Tmat to the identity matrix:
			Tmat=np.zeros((3,3))
			for i in range(Tmat.shape[0]):
				Tmat[i,i]=1.0

		print('Rotation matrix T for Eckart conditions:')
		print(Tmat)

		# successfully built Eckart rotation matrix. Now rotate both coords_ex and nm_ex. 
		coords_ex_rotated=np.zeros((coords_ex.shape[0],coords_ex.shape[1]))	
		nm_ex_rotated=np.zeros((nm_ex.shape[0],nm_ex.shape[1]))	
		# first sort out coordinates. loop over atoms	
		for i in range(coords_ex.shape[0]):
			current_xyz=coords_ex[i,:]
			coords_ex_rotated[i,:]=np.dot(Tmat,current_xyz)

		# now sort out nm vector. loop over normal modes:
		for i in range(nm_ex.shape[1]):
			# loop over atoms
			for j in range(coords_ex.shape[0]):
				# find effective xyz normal mode compnent of normal mode vector i and atom j 
				nm_xyz=np.zeros(3)
				nm_xyz[0]=nm_ex[3*j,i]
				nm_xyz[1]=nm_ex[3*j+1,i]
				nm_xyz[2]=nm_ex[3*j+2,i]
				rot_nm_xyz=np.dot(Tmat,nm_xyz)
				nm_ex_rotated[3*j,i]=rot_nm_xyz[0]
				nm_ex_rotated[3*j+1,i]=rot_nm_xyz[1]
				nm_ex_rotated[3*j+2,i]=rot_nm_xyz[2]
		# all done. Now make sure we overwrite old normal modes and xyz coordinates
		nm_ex=nm_ex_rotated
		coords_ex=coords_ex_rotated

		# if this is not a frozen atom calculation, need to also rotate the transition
		# dipole moment and its derivative into the Eckart frame
		dipole_rotated=np.dot(Tmat,dipole_mom)
		dipole_deriv_rotated=np.zeros((dipole_deriv.shape[0],dipole_deriv.shape[1]))
		for i in range(dipole_deriv.shape[1]):
			current_dipole=dipole_deriv[:,i]
			dipole_deriv_rotated[:,i]=np.dot(Tmat,current_dipole)
		print('Dipole before rotation')
		print(dipole_mom)
		print('Dipole after rotation:')
		print(dipole_rotated)
			
		dipole_mom=dipole_rotated
		dipole_deriv=dipole_deriv_rotated

	Jmat=np.dot(np.transpose(nm_gs),nm_ex)  # This is Jmat as defined in Baiardi,Barone. 	

	coord_diff=np.zeros(3*coords_ex.shape[0])
	# create mass weighted difference vector between ground and excited state geometry
	for i in range(coords_ex.shape[0]):
		coord_diff[3*i]=np.sqrt(atomic_masses[i])*(coords_ex[i,0]-coords_gs[i,0])
		coord_diff[3*i+1]=np.sqrt(atomic_masses[i])*(coords_ex[i,1]-coords_gs[i,1])
		coord_diff[3*i+2]=np.sqrt(atomic_masses[i])*(coords_ex[i,2]-coords_gs[i,2])
		
	# now build Kvector.
	Kvec=np.sqrt(1.0/const.emass_in_au)*np.dot(np.transpose(nm_gs),coord_diff) # This again follows the Barone definition.

	# now just need to construct the dipole derivative in NM coordnates needed for HT calcs
	dipole_deriv_nm=construct_dipole_deriv_nm(dipole_deriv,nm_ex,atomic_masses) 

	return Jmat,Kvec, dipole_mom,dipole_deriv_nm

def moments_of_inertia(coords,atomic_masses):
	natoms=atomic_masses.shape[0]
	total_mass=np.sum(atomic_masses)
	# find center of mass
	com=np.zeros(3)
	for i in range(atomic_masses.shape[0]):
		com[0]=com[0]+atomic_masses[i]*coords[i,0]
		com[1]=com[1]+atomic_masses[i]*coords[i,1]
		com[2]=com[2]+atomic_masses[i]*coords[i,2]
	com=com/total_mass

	# shift all coords by com
	for i in range(coords.shape[0]):
		coords[i,:]=coords[i,:]-com[:]

	# build the 3x3 inertia matix
	inertia_mat=np.zeros((3,3))
	for i in range(atomic_masses.shape[0]):
		inertia_mat[0,0]=inertia_mat[0,0]+atomic_masses[i]*(coords[i,1]**2.0+coords[i,2]**2.0)
		inertia_mat[1,0]=inertia_mat[1,0]-atomic_masses[i]*(coords[i,0]*coords[i,1])
		inertia_mat[2,0]=inertia_mat[2,0]-atomic_masses[i]*(coords[i,0]*coords[i,2])
		inertia_mat[1,1]=inertia_mat[1,1]+atomic_masses[i]*(coords[i,0]**2.0+coords[i,2]**2.0)
		inertia_mat[2,1]=inertia_mat[2,1]-atomic_masses[i]*(coords[i,1]*coords[i,2])
		inertia_mat[2,2]=inertia_mat[2,2]+atomic_masses[i]*(coords[i,0]**2.0+coords[i,1]**2.0)

	inertia_mat[0,1]=inertia_mat[1,0]
	inertia_mat[0,2]=inertia_mat[2,0]
	inertia_mat[1,2]=inertia_mat[2,1]

	# find eigenvalues and eigenvectors of inertia matrix. 
	eigVal,eigVecs = np.linalg.eigh(inertia_mat)

	rotation_vecs=np.zeros((natoms*3,3))

	for i in range(atomic_masses.shape[0]):
		principal_x=coords[i,0]*eigVecs[0,0]+coords[i,1]*eigVecs[1,0]+coords[i,2]*eigVecs[2,0]
		principal_y=coords[i,0]*eigVecs[0,1]+coords[i,1]*eigVecs[1,1]+coords[i,2]*eigVecs[2,1]
		principal_z=coords[i,0]*eigVecs[0,2]+coords[i,1]*eigVecs[1,2]+coords[i,2]*eigVecs[2,2]
		
		rotation_vecs[i*3,0]=principal_y*eigVecs[0,2]-principal_z*eigVecs[0,1]
		rotation_vecs[i*3+1,0]=principal_y*eigVecs[1,2]-principal_z*eigVecs[1,1]
		rotation_vecs[i*3+2,0]=principal_y*eigVecs[2,2]-principal_z*eigVecs[2,1]

		rotation_vecs[i*3,1]=principal_z*eigVecs[0,0]-principal_x*eigVecs[0,2]
		rotation_vecs[i*3+1,1]=principal_z*eigVecs[1,0]-principal_x*eigVecs[1,2]
		rotation_vecs[i*3+2,1]=principal_z*eigVecs[2,0]-principal_x*eigVecs[2,2]

		rotation_vecs[i*3,2]=principal_x*eigVecs[0,1]-principal_y*eigVecs[0,0]
		rotation_vecs[i*3+1,2]=principal_x*eigVecs[1,1]-principal_y*eigVecs[1,0]
		rotation_vecs[i*3+2,2]=principal_x*eigVecs[2,1]-principal_y*eigVecs[2,0]

	# now mass weight vectors:
	for i in range(atomic_masses.shape[0]):
		rotation_vecs[i*3,:]=rotation_vecs[i*3,:]*np.sqrt(atomic_masses[i])
		rotation_vecs[i*3+1,:]=rotation_vecs[i*3+1,:]*np.sqrt(atomic_masses[i])
		rotation_vecs[i*3+2,:]=rotation_vecs[i*3+2,:]*np.sqrt(atomic_masses[i])

	return rotation_vecs

#build the projector for rotation and translation
def build_rot_trans_proj(coords,atomic_masses):
	natoms=atomic_masses.shape[0]
	# get rotation vectors
	rotation_vecs=moments_of_inertia(coords,atomic_masses)

	# now build full 6 projectors
	constraint_vec=np.zeros((natoms*3,6))

	# fill constraint vector. First do translation:
	trans_vec=np.zeros((natoms*3,3))
	for i in range(atomic_masses.shape[0]):
		trans_vec[i*3,0]=np.sqrt(atomic_masses[i])
		trans_vec[i*3+1,1]=np.sqrt(atomic_masses[i])
		trans_vec[i*3+2,2]=np.sqrt(atomic_masses[i])
	# normalize vectors
	trans_vec[:,0]=trans_vec[:,0]/np.linalg.norm(trans_vec[:,0])
	trans_vec[:,1]=trans_vec[:,1]/np.linalg.norm(trans_vec[:,1]) 
	trans_vec[:,2]=trans_vec[:,2]/np.linalg.norm(trans_vec[:,2]) 	

	constraint_vec[:,0]=trans_vec[:,0]
	constraint_vec[:,1]=trans_vec[:,1]
	constraint_vec[:,2]=trans_vec[:,2]

	# now do rotation. Normalize rotation_vecs
	rotation_vecs[:,0]=rotation_vecs[:,0]/np.linalg.norm(rotation_vecs[:,0])
	rotation_vecs[:,1]=rotation_vecs[:,1]/np.linalg.norm(rotation_vecs[:,1])
	rotation_vecs[:,2]=rotation_vecs[:,2]/np.linalg.norm(rotation_vecs[:,2])  

	constraint_vec[:,3]=rotation_vecs[:,0]
	constraint_vec[:,4]=rotation_vecs[:,1]
	constraint_vec[:,5]=rotation_vecs[:,2]

	# now build the projector
	temp_proj=np.dot(constraint_vec,np.transpose(constraint_vec)) # proj=I-const*const^T
	projector=identity_mat(natoms*3)
	projector=projector-temp_proj

	return projector
	

def mass_weigh_hessian(hessian,atomic_masses):
	hessian_mw=hessian
	# loop over atoms i
	for i in range(atomic_masses.shape[0]):

		# loop over atoms j
		for j in range(atomic_masses.shape[0]):
			scale_fac=1.0/(np.sqrt(atomic_masses[i])*np.sqrt(atomic_masses[j]))
			hessian_mw[3*i,3*j]=scale_fac*hessian_mw[3*i,3*j]
			hessian_mw[3*i,3*j+1]=scale_fac*hessian_mw[3*i,3*j+1]
			hessian_mw[3*i,3*j+2]=scale_fac*hessian_mw[3*i,3*j+2]
			hessian_mw[3*i+1,3*j]=scale_fac*hessian_mw[3*i+1,3*j]
			hessian_mw[3*i+1,3*j+1]=scale_fac*hessian_mw[3*i+1,3*j+1]
			hessian_mw[3*i+1,3*j+2]=scale_fac*hessian_mw[3*i+1,3*j+2]
			hessian_mw[3*i+2,3*j]=scale_fac*hessian_mw[3*i+2,3*j]
			hessian_mw[3*i+2,3*j+1]=scale_fac*hessian_mw[3*i+2,3*j+1]
			hessian_mw[3*i+2,3*j+2]=scale_fac*hessian_mw[3*i+2,3*j+2]

	return hessian_mw

# returns an identity matrix of dimension "dimension"
def identity_mat(dimension):
	identity=np.zeros((dimension,dimension))
	for i in range(identity.shape[0]):
		identity[i,i]=1.0

	return identity


# hessian has the format (ix,iy,iz,jx,jy,jz....)
# frozen list contains a list of zeros and 1s, where 0s represent unfrozen atoms and 1s frozen atoms
# if there are no frozen atoms, this is just a dummy list
def freqs_NM_from_hessian(hessian, coords,atomic_masses,frozen_atoms,frozen_list): 
	natoms=coords.shape[0]
	constraints=6

	# check if we have frozen atoms
	if frozen_atoms !=0:
		constraints=frozen_atoms*3  

	# in a first step, build projector for frozen atoms
	constraint_vec=np.zeros((natoms*3,constraints))  	

	if frozen_atoms !=0:  # do not have to deal with projecting out rotation/tranlation
		# Start with the case of actual frozen atoms
		# loop over atoms
		constraint_count=0
		for i in range(natoms):
			if frozen_list[i]==1:  # this is a frozen atom
				for j in range(3): #loop over x,y,z	
					temp_vec=np.zeros(natoms*3)
					temp_vec[i*3+j]=np.sqrt(atomic_masses[i])
					
					constraint_vec[:,constraint_count]=temp_vec/(np.linalg.norm(temp_vec))
					constraint_count=constraint_count+1

		# build the effective projector
		temp_proj=np.dot(constraint_vec,np.transpose(constraint_vec)) # proj=I-const*const^T
		projector=identity_mat(natoms*3)
		projector=projector-temp_proj

	else: # no frozen atoms, but this means we have to build projectors for rotation/translation
		projector=build_rot_trans_proj(coords,atomic_masses)		

	hessian_mw=mass_weigh_hessian(hessian,atomic_masses)   # mass weigh hessian

	hessian_mw_proj=np.dot(projector,np.dot(hessian_mw,projector))   #Proj*Hessian_MW*Proj

	# now compute eigenvalues and eigenvectors of this hessian. 
	eigVal,eigVecs = np.linalg.eigh(hessian_mw_proj)
	freqs=np.zeros(eigVal.shape[0])
	for i in range(freqs.shape[0]):
		freqs[i]=np.sign(eigVal[i])*np.sqrt(np.absolute(eigVal[i])) 
	
	constrained_freqs=np.zeros(freqs.shape[0]-constraints)
	constrained_nm=np.zeros((freqs.shape[0],freqs.shape[0]-constraints)) # this vector contains columns of normalized evecs
									     # corresponding to the non-zero eigenvalues
	constraint_counter=0
	evec_counter=0
	for x in freqs:
		if abs(x)*const.hessian_freqs_to_cm>0.001:
			constrained_freqs[constraint_counter]=x*const.hessian_freqs_to_cm/const.Ha_to_cm  # work out unit conversions here!
			constrained_nm[:,constraint_counter]=eigVecs[:,evec_counter]
			constraint_counter=constraint_counter+1
		evec_counter=evec_counter+1

	# return frequencies of constrained hessian in Ha, normalized unitless Normal mode vectors
	return constrained_freqs,constrained_nm
