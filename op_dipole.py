#!/usr/bin/python

from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *
from itertools import product

import numpy as np
import sympy as sp
import scipy as scp
import sys
import re

from apply_sym_wan import *

# Calculate dipole matrix element between states, and symmetry characters
# Cyrus Dreyer, Flatiron CCQ and Stony Brook University 
# 02/25/20

#*************************************************************************************
# Get expectation values of an operator for all states. Should be used in many of the operations below!!!
def get_exp_vals(ad,spin_names,orb_names,fops,operator,verbose=False):
    '''
    Get the density matricies <c^\dagger_i c_j> 

    Inputs: 
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    operator: Operator to get exp value of

    Outputs:
    exp_val: Expectation values

    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)
    exp_vals=[]
    n_eig=0
    # Loop through subspaces
    for sub in range(0,ad.n_subspaces):
        exp_val=np.zeros(ad.get_subspace_dim(sub),dtype=complex)
        for ind in range(0,ad.get_subspace_dim(sub)):
        
            # Get desired states in eigenvector basis
            state_eig = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig[int(n_eig)]=1.0
        
            exp_val[ind]=np.dot(state_eig.T,act(operator,state_eig,ad))

            n_eig+=1

        exp_vals.append(exp_val)
        
    return exp_vals

#*************************************************************************************
# Get expectation value of an operator for ONE states. Should be used in many of the operations below!!!
def get_exp_val(ad,state,operator):
    '''
    Get the density matricies <c^\dagger_i c_j> 

    Inputs: 
    ad: Solution to atomic problem
    state: Index of the state (in energy ordering)
    operator: Operator to get exp value of

    Outputs:
    exp_val: Expectation value

    '''
        
    state_eig = np.zeros((int(ad.full_hilbert_space_dim)))
    state_eig[int(state)]=1.0        
    exp_val=np.dot(state_eig.T,act(operator,state_eig,ad))

        
    return exp_val

#*************************************************************************************
# Get the one body density matricies
def get_den_mats(ad,spin_names,orb_names,fops,eigensys,n_mbwfs=[0,10],verbose=False):
    '''
    Get the density matricies <c^\dagger_i c_j> 

    Inputs: 
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    eigensys:  states, energies, etc. from ad
    n_orb: Number of orbitals
    n_mbwfs: Number of states to check
    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)
    den_mats=[]
    for state in range(n_mbwfs[0],n_mbwfs[1]):

        if state == len(eigensys):
            break

        # Convert from states in energy order to those in Hilbert space
        n_eig=int(eigensys[state][3])
        
        # Get desired states in eigenvector basis
        state_eig = np.zeros((int(ad.full_hilbert_space_dim)))
        state_eig[int(n_eig)]=1.0
        
        # Construct density matrix
        den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb),dtype='complex128')
        for s1 in range(0,n_spin):
            for s2 in range(0,n_spin):
                for ii in range(0,n_orb):
                    for jj in range(0,n_orb):
                                
                        den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])

                        xx=ii+s1*n_orb
                        yy=jj+s2*n_orb
                    
                        den_mat[xx,yy]=np.dot(state_eig,act(den_op,state_eig,ad))#np.real(np.dot(state_eig,act(den_op,state_eig,ad)))

                        
        den_mats.append(den_mat)

        if verbose:
            print(state)
            print(den_mat)
            print(" ")
            print(np.abs(np.matmul(den_mat,den_mat)))
            print(" ")
            print(np.matmul(den_mat,den_mat)-den_mat)
            print(np.amax(den_mat-np.abs(np.matmul(den_mat,den_mat))))
            print(np.trace(den_mat-np.matmul(den_mat,den_mat)))
            print(" ")

        #quit()


    return den_mats

#*************************************************************************************
# Check the multi-reference nature of single state
def check_multi_ref_state(ad,spin_names,orb_names,fops,n_eig,verbose=False,den_mat=[]):
    '''
    Check the multi-reference nature of the state. From J. Cano: if
     <c^\dagger_i c_j> does not square to itself, then there is no
     basis where the state can be expressed as a single slater
     determinant

    Inputs: 
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    n_eig: Eigenstate number in the Hilbert space
    verbose: print stuff or not
    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)
    
    # Get desired states in eigenvector basis
    state_eig = np.zeros((int(ad.full_hilbert_space_dim)))
    state_eig[int(n_eig)]=1.0        
    
    # Construct density matrix
    if not den_mat:
        den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb),dtype='complex128')
        for s1 in range(0,n_spin):
            for s2 in range(0,n_spin):
                for ii in range(0,n_orb):
                    for jj in range(0,n_orb):
                        
                        den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])
                    
                        xx=ii+s1*n_orb
                        yy=jj+s2*n_orb
                        
                        den_mat[xx,yy]=np.dot(state_eig,act(den_op,state_eig,ad))#np.real(np.dot(state_eig,act(den_op,state_eig,ad)))

                    #print(s1,s2,ii,jj,den_mat[xx,yy])

    multiref=np.trace(den_mat-np.matmul(den_mat,den_mat))


    
    if verbose:
        print("state",n_eig)
        print("Den mat")
        print(den_mat)
        print("Den mat ^ 2 ")
        print(np.abs(np.matmul(den_mat,den_mat)))
        print(" ")
        print(np.matmul(den_mat,den_mat)-den_mat)
        print(np.amax(den_mat-np.abs(np.matmul(den_mat,den_mat))))
        print("Lambda_MR",np.trace(den_mat-np.matmul(den_mat,den_mat)))
        print(" ")

        #quit()


    return den_mat,multiref

#*************************************************************************************
# Read in wannier90_r file
def read_rij(r_wan_files,n_orb,lat_param,diag_vecs=[]):
    '''
    Read in the dipole matrix elements for single particle basis
    
    Inputs:
    r_wan_file: wannier90_r file
    n_orb: Number of orbitals
    lat_param: Lattice parameters of the cell

    Outputs:
    rijs: List with dipole matrix elements. Should work for spin-pol
    '''

    rijs=[]
    for irwf,r_wan_file in enumerate(r_wan_files):
        r_file = open(r_wan_file,"r")
        lines=r_file.readlines()[3:]
        rij=np.zeros((3,n_orb,n_orb))

        for line in lines:
            rij[0][int(line.split()[3])-1][int(line.split()[4])-1]=float(line.split()[5])
            rij[1][int(line.split()[3])-1][int(line.split()[4])-1]=float(line.split()[7])
            rij[2][int(line.split()[3])-1][int(line.split()[4])-1]=float(line.split()[9])
        
        r_file.close()

        
        # I think we should make sure that diagonal elements are zero. Thus, we are shifting. SOMETHING WRONG WITH THIS...
        if np.linalg.norm(lat_param) > 1.0e-10:
            for orb in range(n_orb):
                for idir in range(3):
                    rij[idir,orb,orb]=np.mod(rij[idir,orb,orb],lat_param[idir])
        else:
            print('WARNING: No lattice parameters specified or invalid. Wannier functions not shifted to home cell.')


        #print('DIGAVECS',diag_vecs)
            
        # Convert to band basis
        if diag_vecs:
            for idir in range(3):
                rij[idir,:,:]=np.dot(np.dot(np.matrix(diag_vecs[irwf]).H,rij[idir,:,:]),diag_vecs[irwf])
    
                
        rijs.append(rij) 
        
        
    return rijs


#*************************************************************************************
# Hake the dipole operator
def make_dipole_op(ad,spin_names,orb_names,fops,dipol_file,tij,lat_param,diag_basis=False,velocity=False):
    '''
    Make the many-body dipole operator

    Inputs:
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    dipol_file: wannier90_r file
    tij: hopping matrix elements
    lat_param: lattice parameters of the cell
    velocity: Calculate the velocity operator

    Outputs:
    r_op: Many body dipole/velocity operator in directions x,y,z

    '''

    n_orb=len(orb_names) 
    
    if diag_basis:
        diag_vecs=[]
        for t in tij:
            diag_val,diag_vec=np.linalg.eig(t)
            diag_vecs.append(diag_vec)
        # read in wannier90_r, convert to band basis
        rijs=read_rij(dipol_file,n_orb,lat_param,diag_vecs=diag_vecs)
    else:
        rijs=read_rij(dipol_file,n_orb,lat_param)

                
    if len(rijs) == 1:
        rijs.append(rijs[0])
        
    # Contsruct MB dipole operator
    r_op=[Operator(),Operator(),Operator()]

    for ispin,s in enumerate(spin_names):
        for o1 in orb_names:
            for o2 in orb_names:
                for x in range(0,3):
                    r_op[x] += rijs[ispin][x][int(o1)][int(o2)] * c_dag(s,o1) * c(s,o2)

#    if velocity:
#        for s in spin_names:
#            for i,o1 in enumerate(orb_names):
#                for j,o2 in enumerate(orb_names):
#                    for x in range(0,3):
#                        p_ij=0.0
#                        for k,o3 in enumerate(orb_names):
#                            p_ij+=tij[i,k]*rij[x][k][j]-tij[j,k]*rij[x][k][i]                       
#                        r_op[x] += p_ij * c_dag(s,o1) * c(s,o2)
#                        print('velocity',i,j,p_ij)
                        
#    else:

                        
    return r_op
                        
#*************************************************************************************
# Calculate dipole matrix elements between states
def apply_dipole_op(r_op,ad,n_state_l,n_state_r,eigensys,verbose=False):
    '''
    Get the dipole matrix elements between many-body state n_state_l
    and n_state_r

    Inputs:
    r_op: Dipole/velocity operator
    ad: Solution to atomic problem
    n_state_l: MB state on the left
    n_state_r: MB state on the right
    eigensys: Eigenstates
    verbose: print stuff or not 

    Outputs:
    Rij: Dipole matrix elements between many-body states

    '''                    

    # Convert from states in energy order to those in Hilbert space
    l_hs=int(eigensys[n_state_l][3])
    r_hs=int(eigensys[n_state_r][3])

    # Get desired states in eigenvector basis
    state_r = np.zeros((int(ad.full_hilbert_space_dim)))
    state_r[int(r_hs)]=1

    state_l = np.zeros((int(ad.full_hilbert_space_dim)))
    state_l[int(l_hs)]=1

    # Apply dipole operator to state_r and print out

    Rij=np.zeros(3,dtype='complex128')
    for direc in range(3):
        Rij[direc]=np.dot(state_l,act(r_op[direc],state_r,ad))

    
    if verbose:
        print ("HS l: ",l_hs," HS r: ",r_hs )
        #print ("<",n_state_l,"|r_x|",n_state_r,"> = ",Rij[0])
        #print ("<",n_state_l,"|r_y|",n_state_r,"> = ",Rij[1])
        #print ("<",n_state_l,"|r_z|",n_state_r,"> = ",Rij[2])
        #print (" ")

        #opp=Operator()
        #opp=1
        #print ("<",n_state_l,"|",n_state_r,"> = ",np.dot(state_l,act(opp,state_r)
        #print (" ")

    return Rij


#*************************************************************************************
# Print the dipole matricies
def print_dipole_mat(n_dipol,ad,spin_names,orb_names,fops,dipol_file,eigensys,out_label,lat_param,tij,diag_basis=False):

    '''
    Print dipole/velocity matrix elements

    Inputs:
    n_dipol: Min and max index of many-body states to include in dipole calculations
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    dipol_file: wannier90_r file
    eigensys: Eigenstates
    lat_param: lattice parameters of the cell
    tij: hopping matrix elements

    Outputs:
    None.

    '''

    # Make the dipole operator
    r_op=make_dipole_op(ad,spin_names,orb_names,fops,dipol_file,tij,lat_param,diag_basis=diag_basis)
    
    with open(out_label+"rij.dat","w") as rf:
        rf.write('# state 1  state 2  dir  dipole (real)  dipole (imag) \n')

        for n_state_l in range(n_dipol[0],n_dipol[1]+1):
            for n_state_r in range(n_state_l,n_dipol[1]+1):

                Rij=apply_dipole_op(r_op,ad,n_state_l,n_state_r,eigensys)
                
                # Write out to file (State l, state r, direction)
                rf.write('%f %f %f %E %E\n' % (n_state_l,n_state_r,1,Rij[0].real,Rij[0].imag))
                rf.write('%f %f %f %E %E\n' % (n_state_l,n_state_r,2,Rij[1].real,Rij[1].imag))
                rf.write('%f %f %f %E %E\n' % (n_state_l,n_state_r,3,Rij[2].real,Rij[2].imag))
    return

#*************************************************************************************
# Get the symmetry eigenvalue of a state
def get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,Dij,cmplx):
    '''
    Get symmetry character for given symmetry operation. First we
    express state as raising operators applied to vacuum, then
    trasform the operators using the representation, finally use act
    function to calculate inner product.

    Inputs:
    i_mb_st: Index of MB state (in terms of energy ordering)
    fops: Many-body operators
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to atomic problem
    eigensys: List of eigenstates (see sort_states)
    Dij: n_orb*n_spin x n_orb*n_spin representation matrix 
    cmplx: Are the weights complex?

    Outputs: 
    char: Symmetry character <\psi_i | \hat{R} \vert \psi_i >
    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)
    n_fock= n_orb*n_spin

    op_on_vacuum=Operator()
    for i_state,state in enumerate(eigensys[i_mb_st][0][1]):

        coeff1=eigensys[i_mb_st][0][0][i_state]
        if coeff1 == '':
            continue

        state1=list(map(str,state.replace('|','').replace('>','')))

        # Express state as raising operators to be applied to vacuum
        jj = n_spin*n_orb-1 # We go from right to left
        op_vac=1
        for s in range(0,n_spin):
            for orb in range(0,n_orb):
            
                # If orbital is occupied
                if int(state1[jj]) == 1:
                    
                    # Multiply by row of representation matrix
                    Dj=n_spin*n_orb-1
                    
                    R_op_R= Operator()
                    for Di in Dij[:][jj]:
                        Dspin= 0 if Dj < n_orb else 1
                        Dorb=np.mod(Dj,n_orb)
                        
                        R_op_R += complex(Di)*c_dag(spin_names[Dspin],orb_names[Dorb])
                        
                        Dj -= 1
                    
                    # Construct the operator
                    op_vac *= R_op_R
                
                
                jj -= 1
                
        op_on_vacuum += coeff1*op_vac

    
    # Test to make sure that the vacuum state is in the space
    if np.dot(ad.vacuum_state,ad.vacuum_state) > 1.e-10:
        
        # Calculate character <\Psi | sym_op | \Psi>
        # i_mb_st is in terms of order in eigensys. Get state index in Hilbert space
        state_l = np.zeros((int(ad.full_hilbert_space_dim)))
        state_l[int(eigensys[i_mb_st][3])]=1

        state_r=act(op_on_vacuum,ad.vacuum_state,ad)
        char = np.dot(state_l,state_r)

    # For constrained occupancy we need to do this without the vacuum state
    else:

        # Find the size of the hibert space (WILL NOT BE full_hilbert_space_dim BECAUSE WE ALWAYS LIMIT N)
        hilbert_space_dim=np.sum([len(sub) for sub in ad.fock_states])    
        
        state_l_less=np.zeros((int(hilbert_space_dim)))
        state_l_less[int(eigensys[i_mb_st][3])]=1
    
        # Setup the state on RHS
        state_r_subs=[]
        for sub in range(ad.n_subspaces):
            state_r_subs.append(np.zeros(len(ad.fock_states[sub]),dtype='complex128'))

        for iop,op in enumerate(op_on_vacuum):
            fock=0
            for o in op[0]:
                c_op=o[1]
                bin_add=10**(int(c_op[1]))
                if c_op[0]=='dn': bin_add*=10**(n_orb)

                fock+=bin_add
                
            found_fock=False
            for sub in range(ad.n_subspaces):
            
                state_ind=np.where(ad.fock_states[sub]==int(str(fock),2))
            
                if state_ind[0].size>0:
                    found_fock= True                        
                    state_ind=int(state_ind[0])
                    state_r_subs[sub][state_ind]=op[1]
                    break


                # Make sure we found the state
                if sub==ad.n_subspaces-1 and found_fock==False:
                    print('ERROR: Could not find fock state ',int(str(fock),2))
                    quit()
                
        # Convert into eigenvalue basis
        for sub in range(ad.n_subspaces):
            state_r_subs[sub]=np.dot(ad.unitary_matrices[sub].conj().T,state_r_subs[sub])
        
        # Convert to full Hilbert space
        state_r = np.concatenate(state_r_subs[:])
    
        char=np.dot(state_l_less,state_r)

    return char,state_r
 
#*************************************************************************************
# Get the matricies for symmetry representations
def construct_dij(n_orb,n_spin,repsfile):
    '''
    Read in the representation matricies, generated from the wannier functions
    
    Inputs: 
    n_orb: Number of orbitals r
    n_spin: Number of explicit spin channels. If 1, assumer SOC and first n_orb/2 are spin up
    epsfile: File containing the representations 
    flip: Reverse the representation matricies, since
    Fock states expressed with orbital 0 on the far right.

    Outputs:
    dij: [symmetry operation,orbital character,spin character]
    '''

    # Assume SOC if only one explicit spin channel
    if n_spin == 1:
        _n_orb=int(n_orb/2)
    else:
        _n_orb=n_orb
    
    r_file = open(repsfile,"r")
    lines=r_file.readlines()[2:]
    dij=[]
    
    n_rep = 0

    for line in range(0,len(lines)):
        # Symmetry operation 
        if "Rotation matrix" in lines[line]:
            rot_mat = [] 
            for ii in range(1,4):
                el=lines[line+ii].split()
                el[-1]=el[-1].strip() # Get rid of new line
                for jj in el:
                    rot_mat.append(jj)
                    
            rot_mat = np.reshape(np.array(rot_mat,dtype=float),(3,3))

        # Orbital part of the representaion
        elif  "Representation (Orb)" in lines[line]:
            rep_mat = [] 
            for ii in range(1,_n_orb+1):
                el=lines[line+ii].split()
                for jj in el:
                    rep_mat.append(jj)

            rep_mat = np.reshape(np.array(rep_mat,dtype=float),(_n_orb,_n_orb))
            
            # Flip the orbital reps (since MB states fo from right to left)
          #  if flip:
          #      rep_mat_flip=np.zeros((_n_orb,_n_orb))
          #      for ii in range(0,_n_orb):
          #          for jj in range(0,_n_orb):
          #              rep_mat_flip[ii][jj]=rep_mat[_n_orb-1-ii][_n_orb-1-jj]
          #      rep_mat=rep_mat_flip

                
        # Spin part of the representation
        elif  "Representation (spin)" in lines[line]:
            spin_mat = [] 
            for ii in range(1,3):
                el=lines[line+ii].split()
                for jj in el:
                    spin_mat.append(complex(jj))
            spin_mat = np.reshape(np.array(spin_mat,dtype=complex),(2,2))
            
              
            dij.append([rot_mat,rep_mat,spin_mat])
            
    return dij
#*************************************************************************************
 
#*************************************************************************************
# Calculates and sums the characters for orbitals in a degenerate group 
def mb_degerate_character(orb_names,spin_names,ad,eigensys,counts,cmplx,dij_orb_spin=[],repsfile=[],state_limit=41,verbose=True,spin=True,out_label=''):
    '''
    Sum the characters for a degenerate manifold of many-body states.

    Input:
    dij_orb_spin: Symmetry representations
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to atomic problem
    eigensys: Eigenstates (see sort_states)
    counts: List of degeneracies
    out_label: Label for output files
    state_limit: Number of states to find characters for
    cmplx: Will the weights be real or complex?
    verbose: Write stuff to STOUT
    spin: Spinfull reps?
    
    Output:
    None, output to mb_char.dat.

    '''
    # Extract the rotation matricies and reps for orbitals
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    fops = [(sn,on) for sn, on in product(spin_names,orb_names)]
    
    # dij
    if not dij_orb_spin and repsfile:
        dij_orb_spin=construct_dij(n_orb,n_spin,repsfile,flip=True)

    elif not dij_orb_spin and not repsfile:
        raise Exception('You need to enter a reps filename or a dij list.' )
        

    n_reps=len(dij_orb_spin)
 
    # Write to output file
    with open(out_label+"mb_char.dat","w+") as rf:

        #Counter for rep:
        i_rep=0
        for rep in dij_orb_spin:
     
            # Print symmetry operation
            rf.write("Sym op:\n")
            sym_op = rep[0]

            print(sym_op)
            
            for i in range(0,3):
                rf.write('%3.1f %3.1f %3.1f\n' % (sym_op[i,0],sym_op[i,1],sym_op[i,2]))

            # Loop over degenerate manifolds
            i_state = 0 # counter for state number
            for deg in counts:
            
                # Sum up character for degenerate manifolds
                char=0.0
                for state in range(0,int(deg)):
     
                    # Whether or not to include spin
                    if spin:
                        dij_kron=np.kron(rep[2],rep[1])
                    else:
                        no_spin=np.array([[1,0],[0,1]])
                        dij_kron=np.kron(no_spin,rep[1])

                    i_mb_st=i_state+state
                    char+=get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,dij_kron,cmplx)[0]
                    
                    if verbose:
                        print(i_mb_st,get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,dij_kron,cmplx)[0])
                rf.write(' %s %s %s %10.5f  \n' % \
                             ("Deg: ",deg," Char: ",char.real))
                if verbose:
                    print(' %s %s %s %s  \n' % \
                          ("Deg: ",deg," Char: ",char))

                i_state+=int(deg)
                if i_state > state_limit:
                    break


                #TEST
                #raise

            # Let us know our progress
            if verbose:
                print( ' %s %s %s %s' % ("Done sym_op",i_rep,"/",n_reps))
                
            i_rep += 1

    return

#*************************************************************************************

#*************************************************************************************
# Makes an Sz operator if spin is not explicitly given
def spin_orbit_S_op(fops,proj='z'):
    '''
    Defines a spin operator assuming the first half of the orbitals are spin up and the second half
    are spin down. Of course only for colinear spins.

    Inputs:
    fops: Many-body operators

    Outputs:
    S_SO: S operator

    '''

    n_fops=len(fops)
    
    if proj=='z':
        pauli_mat=0.5*np.array([[1,0],[0,-1]])
    elif proj=='x':
        pauli_mat=0.5*np.array([[0,1],[1,0]])
    elif proj=='y':
        pauli_mat=0.5*np.array([[0,-1j],[1j,0]])
    else:
        raise Exception('INCORRECT SPIN PROJECTION!')
    
    # Lets try and make this general
    up_fops=fops[0:int(n_fops/2)]
    dn_fops=fops[int(n_fops/2):n_fops+1]

    S_SO=Operator()
    for o1 in range(int(n_fops/2)):

        S_SO+=np.dot(np.dot(np.array([c_dag(up_fops[o1][0],up_fops[o1][1]),c_dag(dn_fops[o1][0],dn_fops[o1][1])]),pauli_mat),\
                     np.array([c(up_fops[o1][0],up_fops[o1][1]),c(dn_fops[o1][0],dn_fops[o1][1])]))

    return S_SO

#*************************************************************************************

#*************************************************************************************
# Makes an S2 operator if spin is not explicitly given
def spin_orbit_S2_op(fops):
    '''
    Defines a Sz operator assuming the first half of the orbitals are spin up and the second half
    are spin down.

    Inputs:
    fops: Many-body operators

    Outputs:
    S2_SO: S2 operator
    '''
    Sx=spin_orbit_S_op(fops,proj='x')
    Sy=spin_orbit_S_op(fops,proj='y')
    Sz=spin_orbit_S_op(fops,proj='z')

    Sp=Sx+1j*Sy
    Sm=Sx-1j*Sy
    
    S2_SO=Sz*Sz + 0.5*(Sp*Sm + Sm*Sp)
    
    return S2_SO

#*************************************************************************************

#*************************************************************************************
# Makes an L operator. Only assumption is that first half are spin up,
# second half are spin down. NOTE: ALL OF THESE L OPERATORS ARE WRONG!!!
def spin_orbit_L_op(fops,ml_order,proj='z'):
    '''
    Defines a L operator

    Inputs:
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators

    Outputs:
    L_SO: L operator

    '''

    n_fops=len(fops)

    ll=int((n_fops/2-1)/2)
    if ll < 0 or ll > 3:
         raise Exception('INVALID NUMBER OF ORBITALS! l=',ll)
        
    # Generalized up and down spin
    up_fops=fops[0:int(n_fops/2)]
    dn_fops=fops[int(n_fops/2):n_fops+1]

    # Convert to basis
    ml_basis_up=np.zeros([int(n_fops/2),int(n_fops/2)])
    ml_basis_dn=np.zeros([int(n_fops/2),int(n_fops/2)])
    up_ml_ord=np.array(ml_order[0:int(n_fops/2)])-1
    dn_ml_ord=np.array(ml_order[int(n_fops/2):n_fops+1])-1
    
    up_fops = [up_fops[i] for i in up_ml_ord]
    dn_fops = [dn_fops[i] for i in dn_ml_ord]

    L_SO=Operator()
    if proj=='z':
        for o1 in range(int(n_fops/2)):
            m=o1-ll
            L_SO+=m*1.0j*c_dag(up_fops[o1][0],up_fops[o1][1])*c(up_fops[o1][0],up_fops[o1][1])
            L_SO+=m*1.0j*c_dag(dn_fops[o1][0],dn_fops[o1][1])*c(dn_fops[o1][0],dn_fops[o1][1])

    else: 
        L_plus=Operator()
        L_minus=Operator()
        for ml1 in range(-ll,ll+1):
            for ml2 in range(-ll,ll+1):
                if ml1==ml2+1:
                    L_plus+=np.sqrt(ll*(ll+1)-ml2*(ml2+1))*c_dag(up_fops[ml1+ll][0],up_fops[ml1+ll][1])*c(up_fops[ml2+ll][0],up_fops[ml2+ll][1])
                    L_plus+=np.sqrt(ll*(ll+1)-ml2*(ml2+1))*c_dag(dn_fops[ml1+ll][0],dn_fops[ml1+ll][1])*c(dn_fops[ml2+ll][0],dn_fops[ml2+ll][1])
                elif ml1==ml2-1:
                    L_minus+=np.sqrt(ll*(ll+1)-ml2*(ml2-1))*c_dag(up_fops[ml1+ll][0],up_fops[ml1+ll][1])*c(up_fops[ml2+ll][0],up_fops[ml2+ll][1])
                    L_minus+=np.sqrt(ll*(ll+1)-ml2*(ml2-1))*c_dag(dn_fops[ml1+ll][0],dn_fops[ml1+ll][1])*c(dn_fops[ml2+ll][0],dn_fops[ml2+ll][1])
                    
        if proj=='+':
            L_SO=L_plus
        elif proj=='-':
            L_SO=L_minus
        elif proj=='x':
            L_SO=0.5*(L_plus+L_minus)
        elif proj=='y':
            L_SO=-0.5*1j*(L_plus-L_minus)
        else:
            raise Exception('INCORRECT SPIN PROJECTION!',proj)
            


    return L_SO 
#*************************************************************************************

#*************************************************************************************
# Makes an S2 operator if spin is not explicitly given
def spin_orbit_L2_op(fops,ml_order):
    '''
    Defines a L^2 operator assuming the first half of the orbitals are spin up and the second half
    are spin down.

    Inputs:
    fops: Many-body operators

    Outputs:
    L2_SO: L^2 operator
    '''
    Lz=spin_orbit_L_op(fops,ml_order,proj='z')
    Lp=spin_orbit_L_op(fops,ml_order,proj='+')
    Lm=spin_orbit_L_op(fops,ml_order,proj='-')

    L2_SO=Lz*Lz + 0.5*(Lp*Lm + Lm*Lp)
    
    return L2_SO

#*************************************************************************************

#*************************************************************************************
# Get expectation values of an operator for all states. Should be used in many of the operations below!!!
def get_char_table(pg_symbol):
    '''
    Get information from the character table of point group. For now need to hard code each symmetry

    Inputs:
    pg_symbol: Point group symbol

    Outputs:
    char_table: List with sym names, multiplicities, irrep labels, characters, and one symmetry operation per family

    '''
    
    #ops=get_sym_ops(pg_symbol)
    spin=False
    if pg_symbol=='m-3m' or pg_symbol=='Oh':

        ops=get_sym_ops('m-3m')
        
        sym_names = ['1','4','2_{100}','3','2_{110}','-1','-4','m_{100}','-3','m_{110}']
        sym_mult = [1,6,3,8,6,1,6,3,8,6]
        irrep_labels = ['A1g','A1u','A2g','A2u','Eg','Eu','T2u','T2g','T1u','T1g']
        irrep_deg = [1,1,1,1,2,2,3,3,3,3]
        characters = np.array([[1,1,1,1,1,1,1,1,1,1],\
                               [1,1,1,1,1,-1,-1,-1,-1,-1],\
                               [1,-1,1,1,-1,1,-1,1,1,-1],\
                               [1,-1,1,1,-1,-1,1,-1,-1,1],\
                               [2,0,2,-1,0,2,0,2,-1,0],\
                               [2,0,2,-1,0,-2,0,-2,1,0],\
                               [3,-1,-1,0,1,-3,1,1,0,-1],\
                               [3,-1,-1,0,1,3,-1,-1,0,1],\
                               [3,1,-1,0,-1,-3,-1,1,0,1],\
                               [3,1,-1,0,-1,3,1,-1,0,-1]])

        #ops_one_each=[ops[9],ops[42],ops[28],ops[1],ops[0],ops[13],ops[47],ops[6],ops[16],ops[21]]
        ops_index=[[9],[3,12,32,34,42,46],[20,25,28],[1,11,14,17,22,35,36,43],[0,15,24,27,29,40],[13],[5,7,19,23,38,47],[6,8,31],[2,16,18,33,37,39,41,45],[4,10,21,26,30,44]]

        # With spin
        spin=True
        spin_irrep_labels=['E1/2g','E5/2g','F3/2g','E1/2u','E5/2u','F3/2u']
        spin_chars=np.array([[2,0,1,np.sqrt(2),0,2,0,1,np.sqrt(2),0],
                             [2,0,1,-np.sqrt(2),0,2,0,1,-np.sqrt(2),0],
                             [4,0,-1,0,0,4,0,-1,0,0],
                             [2,0,1,np.sqrt(2),0,-2,0,-1,-np.sqrt(2),0],
                             [2,0,1,-np.sqrt(2),0,-2,0,-1,np.sqrt(2),0],
                             [4,0,-1,0,0,-4,0,1,0,0]])
        
    elif pg_symbol=='-43m' or pg_symbol=='Td':

        ops=get_sym_ops('-43m')
        
        sym_names = ['1','3','2_{100}','-4','m_{110}']
        sym_mult = [1,8,3,6,6]
        irrep_labels = ['A1','A2','E','T1','T2']
        irrep_deg = [1,1,2,3,3]
        characters = np.array([[1,1,1,1,1],
                               [1,1,1,-1,-1],
                               [2,-1,2,0,0],
                               [3,0,-1,1,-1],
                               [3,0,-1,-1,1]])

        ops_index=[[1],[3,7,14,15,16,18,19,22],[0,10,11],[2,5,13,17,20,21],[4,6,8,9,12,23]]
        
    elif pg_symbol=='6/mmm' or pg_symbol=='D6h':

        ops=get_sym_ops('6/mmm')

        sym_names = ['1','6','3','2_z','2_{120}','2_{100}','-1','-6','-3','m_z','m_{120}','m_{100}']
        sym_mult = [1,2,2,1,3,3,1,2,2,1,3,3]
        irrep_labels = ['A1g','A1u','A2g','A2u','B1g','B1u','B2g','B2u','E2u','E2g','E1u','E1g']
        irrep_deg = [1,1,1,1,1,1,1,1,2,2,2,2]
        characters = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],\
                               [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],\
                               [1,1,1,1,-1,-1,1,1,1,1,-1,-1],\
                               [1,1,1,1,-1,-1,-1,-1,-1,-1,1,1],\
                               [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],\
                               [1,-1,1,-1,1,-1,-1,1,-1,1,-1,1],\
                               [1,-1,1,-1,-1,1,1,-1,1,-1,-1,1],\
                               [1,-1,1,-1,-1,1,-1,1,-1,1,1,-1],\
                               [2,-1,-1,2,0,0,-2,1,1,-2,0,0],\
                               [2,-1,-1,2,0,0,2,-1,-1,2,0,0],\
                               [2,1,-1,-2,0,0,-2,-1,1,2,0,0],\
                               [2,1,-1,-2,0,0,2,1,-1,-2,0,0]])
        
        ops_index=[[4],[9,15],[1,6],[7],[13,16,8],[0,11,5],[3],[18,20],[2,22],[10],[17,19,14],[21,23,12]]

        # With spin:
        spin=True
        spin_irrep_labels=['E1/2g','E3/2g','E5/2g','E1/2u','E3/2u','E5/2u']
        spin_chars=np.array([[2,np.sqrt(3),1,0,0,0,2,np.sqrt(3),1,0,0,0],
                             [2,0,-2,0,0,0,2,0,-2,0,0,0],
                             [2,-np.sqrt(3),1,0,0,0,2,-np.sqrt(3),1,0,0,0],
                             [2,np.sqrt(3),1,0,0,0,-2,-np.sqrt(3),-1,0,0,0],
                             [2,0,-2,0,0,0,-2,0,2,0,0,0],
                             [2,-np.sqrt(3),1,0,0,0,-2,np.sqrt(3),-1,0,0,0]])

        
    elif pg_symbol=='3m' or pg_symbol=='C3v':

        ops=get_sym_ops('3m')
        
        sym_names = ['1','3','m_{1-10}']
        sym_mult = [1,2,3]
        irrep_labels = ['A1','A2','E']
        irrep_deg = [1,1,2]
        characters = np.array([[1,1,1],
                              [1,1,-1],
                              [2,-1,0]])
        ops_index=[[3],[2,5],[0,1,4]]

    elif pg_symbol=='mmm' or pg_symbol=='D2h':

        ops=get_sym_ops('mmm')

        sym_names = ['1','2z','2y','2x','-1','mz','my','mx']
        sym_mult = [1,1,1,1,1,1,1,1]
        irrep_labels = ['Ag','B1g','B2g','B3g','Au','B1u','B2u','B3u']
        irrep_deg = [1,1,1,1,1,1,1,1]
        characters = np.array([[1,1,1,1,1,1,1,1],
                              [1,1,-1,-1,1,1,-1,-1],
                               [1,-1,1,-1,1,-1,1,-1],
                               [1,-1,-1,1,1,-1,-1,1],
                               [1,1,1,1,-1,-1,-1,-1],
                               [1,1,-1,-1,-1,-1,1,1],
                               [1,-1,1,-1,-1,1,-1,1],
                               [1,-1,-1,1,-1,1,1,-1]])
        
        ops_index=[[0],[4],[2],[1],[3],[5],[6],[7]]

    elif pg_symbol=='-6m2' or pg_symbol=='D3h':

        ops=get_sym_ops('-6m2')

        sym_names = ['1','m','3','-6','2_{120}','m_{100}']
        sym_mult = [1,1,2,2,3,3]
        irrep_labels = ["A'1","A'2","A''1","A''2","E'","E''"]
        irrep_deg = [1,1,1,1,2,2]
        characters = np.array([[1,1,1,1,1,1],
                               [1,1,1,1,-1,-1],
                               [1,-1,1,-1,1,-1],
                               [1,-1,1,-1,-1,1],
                               [2,2,-1,-1,0,0],
                               [2,-2,-1,1,0,0]])
        
        ops_index=[[2],[6],[5,10],[0,7],[1,3,11],[4,8,9]]

    elif pg_symbol=='mm2' or pg_symbol=='C2v':

        ops=get_sym_ops('mm2')

        sym_names = ['1','2z','my','mx']
        sym_mult = [1,1,1,1]
        irrep_labels = ["A1","A2","B1","B2"]
        irrep_deg = [1,1,1,1]
        characters = np.array([[1,1,1,1],
                               [1,1,-1,-1],
                               [1,-1,1,-1],
                               [1,-1,-1,1]])
        
        ops_index=[[2],[3],[1],[0]]

    elif pg_symbol=='222' or pg_symbol=='D2':

        ops=get_sym_ops('222')

        sym_names = ['1','2z','2y','2x']
        sym_mult = [1,1,1,1]
        irrep_labels = ["A","B1","B2","B3"]
        irrep_deg = [1,1,1,1]
        characters = np.array([[1,1,1,1],
                               [1,1,-1,-1],
                               [1,-1,1,-1],
                               [1,-1,-1,1]])
        
        ops_index=[[2],[3],[1],[0]]

    elif pg_symbol=='m' or pg_symbol=='Cs':

        ops=get_sym_ops('m')

        sym_names = ['1','m']
        sym_mult = [1,1]
        irrep_labels = ["A'","A''"]
        irrep_deg = [1,1,1,1]
        characters = np.array([[1,1],
                               [1,-1]])
        
        ops_index=[[0],[1]]

        
    else:
        raise ValueError('Point group not coded.')

    if spin:
        char_table={'sym_names':sym_names,'sym_mult':sym_mult,'irrep_labels':irrep_labels,'irrep_deg':irrep_deg, \
                    'characters':characters,'ops_index':ops_index,'all_ops':ops,'spin_irrep_labels':spin_irrep_labels,'spin_chars':spin_chars}
    else:
        char_table={'sym_names':sym_names,'sym_mult':sym_mult,'irrep_labels':irrep_labels,'irrep_deg':irrep_deg, \
                'characters':characters,'ops_index':ops_index,'all_ops':ops}

    return char_table
       
#*************************************************************************************

#*************************************************************************************
# Project MB states on irreps
def get_irrep_projection(pg_symbol,orb_names,spin_names,ad,eigensys,n_print,repsfile=[],dij=[],spin=False,cmplx=False,out_label='',verbose=True):
    '''
    Get the irrep projections of the many-body states. VERY slow :(

    Inputs:
    pg_symbol: Symbol of the point group
    orb_names: Names of orbitals
    spin_names: Names of spins
    ad: Solution to atomic problem
    eigensys: Eigenstates (see sort_states)
    n_print: Range of states to consider
    repsfile: Location of the *_reps.dat file (need to specify this or dij)
    dij: Single-particle symmetry representations of the bais
    spin: Spinfull reps?
    cmplx: Will the weights be real or complex?
    out_label: Label for output files
    verbose: Write stuff to STOUT

    Outputs:
    char_table['irrep_labels']: List of the irreps, sometimes useful
    states_proj: MB states projected onto irreps
    
    '''

    # Single particle representations
    if not dij and repsfile:
        dij=construct_dij(n_orb,n_spin,repsfile,flip=False)
    elif not dij and not repsfile:
        raise Exception('Need to pass a dij list or filename for reps.')
    
    # Info from character table
    char_table=get_char_table(pg_symbol)
    
    if spin:

        # Construct double group
        characters=np.concatenate((np.concatenate((char_table['characters'],char_table['characters']),axis=1),np.concatenate((char_table['spin_chars'],-char_table['spin_chars']),axis=1)),axis=0) 
        irrep_labels=char_table['irrep_labels']+char_table['spin_irrep_labels']
        
        h=2*np.sum(np.array(char_table['sym_mult']))
        n_ops=2*len(char_table['sym_names'])

        dij_spin=dij.copy()
        for d in dij:
            dij_spin.append([d[0],d[1],-d[2]]) 

        dij=dij_spin
        
    else:
        characters=char_table['characters']
        irrep_labels=char_table['irrep_labels']
    
        h=np.sum(np.array(char_table['sym_mult']))
        n_ops=len(char_table['sym_names'])
    
    
    fops = [(sn,on) for sn, on in product(spin_names,orb_names)]
    n_orb=len(orb_names)
    n_spin=len(spin_names)
        
    # Loop over state
    states_proj=[]
    for i_mb_st in range(n_print[0],min(n_print[1],len(eigensys))):

        # Project onto irreps
        proj_irreps=[]
        for irrep in range(len(irrep_labels)):
            proj=0
            for sym_el in range(h):

                # Get the single-particle rep for this symmetry element
                rep=dij[sym_el]

                # Index for character table stuff
                for iop,op in enumerate(char_table['ops_index']):
                    if sym_el in op:
                        char_index=iop

                        # For spin, may want to go to \overline{E} sector
                        if spin and sym_el >= h/2:
                            char_index+=int(h/2)
                        break
                
                # Whether or not to include spin
                if spin:
                    dij_kron=np.kron(rep[2],np.flip(rep[1]))
                else:
                    no_spin=np.array([[1,0],[0,1]])
                    dij_kron=np.kron(no_spin,np.flip(rep[1]))

                proj_R=(characters[irrep,0]/h)*characters[irrep,char_index]*get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,dij_kron,cmplx)[1]
                proj+= proj_R

                #if irrep==0 and i_mb_st==0 and (sym_el==1 or sym_el==25):
                #    print(sym_el, get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,dij_kron,cmplx)[1])
                
            proj_irreps.append(proj)
        states_proj.append(proj_irreps)

    # Write out results to file
    if out_label and not out_label.endswith('_'): out_label+='_'
    filename=out_label+'irrep_proj.dat'

    labeled_states=[]
    with open(filename, 'w') as proj_file:
        proj_file.write('# state number   irrep name   weight \n')
        
        for istate,state in enumerate(states_proj):

            nonzero_proj=[]
            for iproj,proj_irreps in enumerate(state):
                norm=np.linalg.norm(proj_irreps)

                if norm > 1e-5:
                    proj_file.write('{:d}  {}  {:6.4f} \n'.format(istate,irrep_labels[iproj],np.linalg.norm(proj_irreps)))

                    # Here we just pick the largest projection
                    if not nonzero_proj:
                        nonzero_proj=[irrep_labels[iproj],np.linalg.norm(proj_irreps)]
                    elif nonzero_proj[1] < np.linalg.norm(proj_irreps):
                        nonzero_proj=[irrep_labels[iproj],np.linalg.norm(proj_irreps)]
                    
                    if verbose:
                        print('{:d}  {}  {:6.4f} '.format(istate,irrep_labels[iproj],np.linalg.norm(proj_irreps)))

            
            labeled_states.append([istate,nonzero_proj[:]])

            proj_file.write('\n')
            if verbose: print('')

    return irrep_labels,states_proj,labeled_states

#*************************************************************************************

#*************************************************************************************
# Use weak proj operator to make symmetrizd tij
def weak_proj_tij(pg_symbol,orb_names,spin_names,dij,cfs=[],spin=False):
    '''
    Use the weak form of the projector: \hat{P}^{\Gamma_n}=\frac{l_n}{h}\sum_R \chi^{\Gamma_n}(R)\hat{P}^R 
    to make a symmetrtic tij matrix with cfs
    
    Inputs:
    pg_symbol: Symbol of the point group
    orb_names: Names of orbitals
    spin_names: Names of spins
    dij: Single-particle symmetry representations of the bais
    cfs: Table for crystal-field splitting energies for each irrep
    spin: Whether to include spin
    
    Outputs:
    tij: Symmetrized tij with cfs
    proj_irreps: Projection of identity on each irrep
    
    '''    
    
    # Info from character table
    char_table=get_char_table(pg_symbol)
    characters=char_table['characters']
    h=np.sum(np.array(char_table['sym_mult']))
    n_ops=len(char_table['sym_names'])

    # Fundamental operators
    fops = [(sn,on) for sn, on in product(spin_names,orb_names)]
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    
    # Generate fully symmetric tij which is big identity matrix
    tij=np.identity(n_orb*n_spin)
    
    # Project onto irreps
    proj_irreps=[]
    for irrep in range(n_ops):
        proj=0
        for sym_el in range(h):

            # Get the single-particle rep for this symmetry element
            rep=dij[sym_el]

            # Index for character table stuff
            for iop,op in enumerate(char_table['ops_index']):
                if sym_el in op:
                    char_index=iop
                    break

            # Whether or not to include spin
            if spin:
                dij_kron=np.kron(rep[2],rep[1])
            else:
                no_spin=np.array([[1,0],[0,1]])
                dij_kron=np.kron(no_spin,rep[1])

            proj_R=(char_table['irrep_deg'][irrep]/h)*characters[irrep,char_index]*np.matmul(dij_kron,tij)
            proj+= proj_R

        proj_irreps.append([char_table['irrep_labels'][irrep],proj])

    # Test the cfs
    if not cfs:
        cfs=[np.ones(n_ops),np.ones(n_ops)]
    elif not isinstance(cfs,list):
        cfs=[cfs]
    if len(cfs)==1:
        cfs=[cfs[0],cfs[0]]
    
    if len(cfs[0]) != n_ops or len(cfs[1]) != n_ops:
        raise ValueError('cfs must length equal to number of irreps. nops:',n_ops,'len(cfs):',len(cfs[0]))
    
    # Now make tij
    tij=[]
    for ispin in range(n_spin):
        tij_spin=np.zeros((n_orb,n_orb),dtype=complex)
        for icf,cf in enumerate(cfs[ispin]):


            tij_spin+=cf*proj_irreps[icf][1][0+n_orb*ispin:n_orb+n_orb*ispin,0+n_orb*ispin:n_orb+n_orb*ispin]
    
        tij.append(tij_spin)
    
    
    return tij,proj_irreps

#*************************************************************************************

#*************************************************************************************
# Strong projection operator
def get_strong_projection_kk(pg,orb_names,spin_names,dij,verbose=True):
    '''
    Based on the orbital representations, project onto the irrep
    partner functions using the diagonal version of the "strong" projection operator:

    \hat{P}_{kk}^{(\Gamma_n)}=\frac{l_n}{h}D^{(\Gamma_n)}(R)^*_{kk} \hat{P}_R

    to project on partner functions of irrep.

    NOTE: THIS WILL NOT PRODUCE CORRECT SIGNS BETWEEEN DEGENERATE STATES!

    Inputs:
    pg: PointGroup object
    orb_names: Names of orbitals
    spin_names: Names of spins
    dij: Single-particle symmetry representations of the bais
    verbose: Write stuff out
    
    Outputs:
    states_proj: List containing the projection of each function onto the irrep partner functions
    
    
    '''

    # Get the point group information
    #pg=ptgp.PointGroup(name=pg_symbol)
    rot_mats=[np.array(i,dtype=complex) for i in pg.rot_mat]
    
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    h=len(dij)
    
    # loop over Wannier states
    states_proj=[]
    for istate in range(n_orb):
        
        proj_Gam_kk={}
        #proj_Gam_kk_norm=[]
        
        for i_irrep,irrep in enumerate(pg.irreps):
            
            proj_kk=np.zeros((int(pg.dims_irreps[i_irrep]),n_orb))
            for sym_el in range(h):

                # Get the single-particle rep for this symmetry element
                sym=dij[sym_el][0]
                rep=dij[sym_el][1]

                # Find this sym el:
                try:
                    sym_ind=np.where([np.allclose(sym,i) for i in rot_mats])[0][0]
                except:
                    raise Exception('Cannot find symmetry operation!')
                    
                D_R=np.array(pg.get_matrices(irrep=irrep, element=pg.elements[sym_ind])[0])
                
                if D_R.shape[0]==1:
                    D_R_kk=D_R[0]
                else:
                    D_R_kk=np.diag(D_R)
                
                for ik,k in enumerate(D_R_kk):
                    # P_kk^(Gamma_n)(R)
                    proj_kk[ik,:]+=np.array((pg.dims_irreps[i_irrep]/h)*float(k)*rep[istate,:])
                   
                
            # Store for irrep
            #proj_Gam_kk_norm=[np.linalg.norm(proj_kk[k,:]) for k in range(pg.dims_irreps[i_irrep])]
            proj_Gam_kk[str(irrep)]=proj_kk#proj_Gam_kk_norm               
        
        states_proj.append(proj_Gam_kk)
        
    if verbose:
        for istate,state in enumerate(states_proj):
            print('State: ',istate)
            for key,val in state.items():
                if np.linalg.norm(np.array(val)) > 1e-10:
                    print(key,np.round(val,4))

            print()


    return states_proj

#*************************************************************************************

#*************************************************************************************
# Get unitary from projections
def get_irrep_unitary_kk(n_orb,pg,states_proj,irrep_basis,verbose=True,mat_format='numpy'):
    '''
    Get the unitary matrix that converts between the orbital basis
    and the irrep basis, taking as input the results from
    get_strong_projection_kk

    NOTE: THIS WILL NOT PRODUCE CORRECT SIGNS BETWEEEN DEGENERATE STATES!

    Inputs:
    n_orb: Number of orbitals
    pg: PointGroup object
    states_proj: Orbital basis projected on irrep partner functions, from get_strong_projection_kk.
    irrep_basis: List of irreps to have in the basis
    verbose: Write stuff out

    Outputs:
    unitary_trans: n_orb x n_orb unitary matrix 
    

    '''
    unitary_trans=[]
    for ibasis,basis in enumerate(irrep_basis): # irrep
        
        # Get the degeneracy
        i_irrep=pg.irrep_names.index(basis)
        dim_irrep=pg.dims_irreps[i_irrep]

        for k in range(dim_irrep): # parnter function number
            
            found=False
            # Search through states
            for istate,state in enumerate(states_proj):
    
                if np.linalg.norm(state[basis][k]) > 1e-10: # Some row has nonzero values
                    unitary_trans.append(state[basis][k]/np.linalg.norm(state[basis][k]))
                    found=True
                    break

    unitary_trans=np.vstack(unitary_trans).T

    if mat_format=='sympy':
        unitary_trans=sympyize_unitary_matrix(unitary_trans)

    if verbose:
        print('Unitary matrix:')
        print(np.round(unitary_trans,4))
        print('Test of unitarity: ', np.linalg.norm(np.linalg.inv(unitary_trans)-unitary_trans.T))
        
        
    return unitary_trans

#*************************************************************************************

#*************************************************************************************
# Strong projection operator
def get_strong_projection_kl(pg,orb_names,spin_names,dij,verbose=True,cart_to_hex=False,rhomb_to_hex=False,pg_symbol=''):
    '''
    Based on the orbital representations, project onto the irrep
    partner functions using the "strong" projection operator:

    \hat{P}_{kl}^{(\Gamma_n)}=\frac{l_n}{h}D^{(\Gamma_n)}(R)^*_{kl} \hat{P}_R

    to project on partner functions of irrep. Off-diagonal elements
    are needed to get correct relative signs for degenerate states.

    Inputs:
    pg: PointGroup object
    orb_names: Names of orbitals
    spin_names: Names of spins
    dij: Single-particle symmetry representations of the bais
    verbose: Write stuff out
    
    Outputs:
    states_proj: List containing the projection of each function onto the irrep partner functions

    '''

    # Get the point group information
    #pg=ptgp.PointGroup(name=pg_symbol)
    rot_mats=[np.array(i,dtype=complex) for i in pg.rot_mat]
    
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    h=len(dij)
    
    # loop over Wannier states
    states_proj=[]
    for istate in range(n_orb):
        
        proj_Gam_kl={}
        
        for i_irrep,irrep in enumerate(pg.irreps):
            
            proj_kl=np.zeros((int(pg.dims_irreps[i_irrep]),int(pg.dims_irreps[i_irrep]),n_orb),dtype=complex)
            for sym_el in range(h):

                # Get the single-particle rep for this symmetry element
                rep=dij[sym_el][1]

                if cart_to_hex:
                    #sym=get_sym_ops(pg_symbol,hex_to_cart=cart_to_hex,hex_to_rhomb=rhomb_to_hex)[sym_el]
                    sym=get_sym_ops(pg_symbol)[sym_el] 
                elif rhomb_to_hex:
                    sym=get_sym_ops(pg_symbol,hex_to_rhomb=True)[sym_el] 
                else:
                    sym=dij[sym_el][0]

                # Find this sym el:
                try:
                    sym_ind=np.where([np.allclose(sym,i) for i in rot_mats])[0][0]
                except:
                    print(sym)
                    raise Exception('Cannot find symmetry operation!')
                    
                D_R=np.conjugate(np.array(pg.get_matrices(irrep=irrep, element=pg.elements[sym_ind])[0],dtype=complex)).T 
     
                for ik in range(D_R.shape[0]):
                    for il in range(D_R.shape[1]):
                        proj_kl[ik,il,:]+=np.array((pg.dims_irreps[i_irrep]/h)*D_R[ik,il]*rep[:,istate])
                                    
            # Store for irrep, here we assume that D is real!
            proj_Gam_kl[str(irrep)]=proj_kl.real               
        
        states_proj.append(proj_Gam_kl)
        
    if verbose:
        for istate,state in enumerate(states_proj):
            print('State: ',istate)
            for key,val in state.items():
                if np.linalg.norm(np.array(val)) > 1e-10:
                    print(key,np.round(val,4))

            print()


    return states_proj

#*************************************************************************************


#*************************************************************************************
# Get unitary from projections
def get_irrep_unitary_kl(n_orb,pg,states_proj,irrep_basis,verbose=True,mat_format='numpy',tol=0.1,clean=True):
    '''
    Get the unitary matrix that converts between the orbital basis and
    the irrep basis, taking as input the results from
    get_strong_projection_kl. Off-diagonal elements are needed to get
    correct relative signs for degenerate states.

    Inputs:
    n_orb: Number of orbitals
    pg: PointGroup object
    states_proj: Orbital basis projected on irrep partner functions, from get_strong_projection_kk.
    irrep_basis: List of irreps to have in the basis
    verbose: Write stuff out

    Outputs:
    unitary_trans: n_orb x n_orb unitary matrix

    '''
    
    unitary_trans=[]
    state_used=[]
    for ibasis,basis in enumerate(irrep_basis): # irrep
        
        # Get the degeneracy
        i_irrep=pg.irrep_names.index(basis)
        dim_irrep=pg.dims_irreps[i_irrep]

        found=False
        for istate,state in enumerate(states_proj):
            if np.linalg.norm(state[basis]) > tol and not (istate in state_used):
                uni_lines=state[basis].reshape(dim_irrep**2,n_orb)
                
                # Remove zeros and normalize
                uni_lines=uni_lines[~np.all(np.abs(uni_lines) < tol, axis=1)]
                row_sums = np.sum(np.abs(uni_lines)**2,axis=-1)**(1./2)
                uni_lines = uni_lines / row_sums[:, np.newaxis]
                
                unitary_trans.append(uni_lines)
                found=True
                state_used.append(istate) # Don't repeat states for repeated irreps
                break

        if found: continue

    unitary_trans=np.vstack(unitary_trans).T

    assert unitary_trans.shape == (n_orb,n_orb), 'Error: Unitary has wrong size. Try changing tol.\n '+str(unitary_trans)
    
    if clean:
        clean_tol=tol
        clean_vals=[0.0,1.0]#,np.sqrt(3)/2,0.5,0.25,1/np.sqrt(2),np.sqrt(3/8),np.sqrt(5/8)]
        for i in range(n_orb):
            for j in range(n_orb):         
                for val in clean_vals:
                    if np.abs(unitary_trans[i,j]-val) < clean_tol:
                        unitary_trans[i,j]=val
                    elif np.abs(unitary_trans[i,j]+val) < clean_tol:
                        unitary_trans[i,j]=-val
    
    if mat_format=='sympy':
        unitary_trans=sympyize_unitary_matrix(unitary_trans)

    if verbose:
        print('Unitary matrix:')
        print(np.round(unitary_trans,4))

        try:
            print('Test of unitarity: ', np.linalg.norm(np.linalg.inv(unitary_trans)-unitary_trans.T))
        except:
            print('Error: Cannot test unitarity.')
            
        
    return unitary_trans
#*************************************************************************************

#*************************************************************************************
# Convert to sympy
def sympyize_unitary_matrix(unitary):
    '''
    Quick and dirty tool to convert from numpy to "exact" sympy for some common bases of f and d orbitals

    Input:
    unitary: numpy array for unitary transformation

    Output:
    sp_unitary: Sympy unitary
    '''
    

    zero=sp.Integer(0)
    one=sp.Integer(1)
    two=sp.Integer(2)
    three=sp.Integer(3)
    four=sp.Integer(4)
    five=sp.Integer(5)
    eight=sp.Integer(8)
    
    sp_unitary=sp.Matrix(np.copy(unitary))
    
    tol=1.0e-2
    vals=[[0.0,zero],[1.0,one],[np.sqrt(3)/2,sp.sqrt(three)/two],[0.5,one/two],[0.25,one/four],
          [1/np.sqrt(2),one/sp.sqrt(two)],[np.sqrt(3/8),sp.sqrt(three/eight)],[np.sqrt(5/8),sp.sqrt(five/eight)],
         [1j/np.sqrt(2),sp.I/sp.sqrt(two)]]
    
    for i in range(unitary.shape[0]):
        for j in range(unitary.shape[0]):
        
            for val in vals:
                if np.abs(unitary[i,j]-val[0]) < tol:
                    sp_unitary[i,j]=val[1]
                elif np.abs(unitary[i,j]+val[0]) < tol:
                    sp_unitary[i,j]=-val[1]

    return sp_unitary

#*************************************************************************************
# Symmetrize the uijkl matrix
def symmetrize_uijkl(uijkls,dij):
    
    # Make sure we have a list                                                                                                                                                                                                                               
    if not isinstance(uijkls, list):
        _uijkls=[uijkls]
    else:
        _uijkls=uijkls

    uijkls_sym=[]
    for uijkl in _uijkls:
        
        uijkl_sym=np.copy(uijkl)
        
        for rep in dij:
            
            # Find multiplicty of symmetry operation
            sym_op=rep[0]
            for mult in range(1,7):
                
                if np.amax(np.abs(sym_op - np.eye(3))) < 1e-5: 
                    break
                else:
                    sym_op = sym_op @ rep[0]
            
            #print(mult)
            
            # Symetrize
            _uijkl_sym=uijkl_sym/mult
            for n in range(0,mult-1):
                term=np.copy(uijkl_sym)
                for m in range(0,n+1):
                    term=transform_U_matrix(term, rep[1].T)

                _uijkl_sym += term/(mult)
                
            uijkl_sym=_uijkl_sym
        
        uijkls_sym.append(uijkl_sym)
                                                                                                                                                                                                                                           
    if not isinstance(uijkls, list):
        return uijkls_sym[0]
        
    else:
        return uijkls_sym
#*************************************************************************************
                    
#*************************************************************************************
# Symmetrize the uijkl matrix
def symmetrize_tij(tijs,dij):
    
    # Make sure we have a list                                                                                                                                                                                                                               
    if not isinstance(tijs, list):
        _tijs=[tijs]
    else:
        _tijs=tijs

    tijs_sym=[]
    for tij in _tijs:
        
        tij_sym=np.copy(tij)
        
        for rep in dij:
            
            # Find multiplicty of symmetry operation
            sym_op=rep[0]
            for mult in range(1,7):
                
                if np.amax(np.abs(sym_op - np.eye(3))) < 1e-5: 
                    break
                else:
                    sym_op = sym_op @ rep[0]
            
            #print(mult)
            
            # Symetrize
            _tij_sym=tij_sym/mult
            for n in range(0,mult-1):
                term=np.copy(tij_sym)
                for m in range(0,n+1):
                    term=rep[1].T @ term @ rep[1]

                _tij_sym += term/(mult)
        
            tij_sym=_tij_sym
            
        tijs_sym.append(tij_sym)
                                                                                                                                                                                                                                           
    if not isinstance(tijs, list):
        return tijs_sym[0]
        
    else:
        return tijs_sym            

#*************************************************************************************
