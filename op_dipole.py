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


# Calculate dipole matrix element between states, and symmetry characters
# Cyrus Dreyer, Flatiron CCQ and Stony Brook University 
# 02/25/20

#*************************************************************************************
# Check the multi-reference nature of the state.
def check_multi_ref(ad,spin_names,orb_names,fops,eigensys,n_states=10,verbose=False):
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
    eigensys:  states, energies, etc. from ad
    n_orb: Number of orbitals
    n_states: Number of states to check
    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)
    den_mats=[]
    for state in range(0,n_states):

        if state == len(eigensys):
            break

        # Convert from states in energy order to those in Hilbert space
        n_eig=int(eigensys[state][3])
        
        # Get desired states in eigenvector basis
        state_eig = np.zeros((int(ad.full_hilbert_space_dim)))
        state_eig[int(n_eig)]=1.0
        
        # Construct density matrix
        den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb))
        for s1 in range(0,n_spin):
            for s2 in range(0,n_spin):
                for ii in range(0,n_orb):
                    for jj in range(0,n_orb):
                                
                        den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])

                        xx=ii+s1*n_orb
                        yy=jj+s2*n_orb
                    
                        den_mat[xx,yy]=np.real(np.dot(state_eig,act(den_op,state_eig,ad)))

                        #print(xx,yy,den_op)
                        
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
def check_multi_ref_state(ad,spin_names,orb_names,fops,n_eig,verbose=False):
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
    den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb))
    for s1 in range(0,n_spin):
        for s2 in range(0,n_spin):
            for ii in range(0,n_orb):
                for jj in range(0,n_orb):
                        
                    den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])
                    
                    xx=ii+s1*n_orb
                    yy=jj+s2*n_orb
                        
                    den_mat[xx,yy]=np.real(np.dot(state_eig,act(den_op,state_eig,ad)))

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
def read_rij(r_wan_file,n_orb):
    '''
    Read in the dipole matrix elements for single particle basis
    
    Inputs:
    r_wan_file: wannier90_r file
    n_orb: Number of orbitals

    Outputs:
    rij: Array with dipole matrix elements
    '''
    
    r_file = open(r_wan_file,"r")
    lines=r_file.readlines()[3:]
    rij=np.zeros((3,n_orb,n_orb))

    for line in lines:
        rij[0][int(line.split()[3])-1][int(line.split()[4])-1]=float(line.split()[5])
        rij[1][int(line.split()[3])-1][int(line.split()[4])-1]=float(line.split()[7])
        rij[2][int(line.split()[3])-1][int(line.split()[4])-1]=float(line.split()[9])
        
    r_file.close()

    return rij


#*************************************************************************************
# Calculate dipole matrix elements
def dipole_op(ad,spin_names,orb_names,fops,r_wan_file,n_state_l,n_state_r,eigensys,verbose=False):
    '''
    Get the dipole matrix elements between many-body state n_state_l
    and n_state_r

    Inputs:
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    r_wan_file: wannier90_r file
    n_state_l: MB state on the left
    n_state_r: MB state on the right
    eigensys: Eigenstates
    verbose: Write stuff to the termial?

    Outputs:
    None.

    '''

    n_orb=len(orb_names) 
    rij=read_rij(r_wan_file,n_orb) # read in wannier90_r

    # Contsruct MB dipole operator
    r_op=[Operator(),Operator(),Operator()]
    for s in spin_names:
        for o1 in orb_names:
            for o2 in orb_names:
                for x in range(0,3):
                    r_op[x] += rij[x][int(o1)][int(o2)] * c_dag(s,o1) * c(s,o2)
    

    # Convert from states in energy order to those in Hilbert space
    l_hs=int(eigensys[n_state_l][3])
    r_hs=int(eigensys[n_state_r][3])

    # Get desired states in eigenvector basis
    state_r = np.zeros((int(ad.full_hilbert_space_dim)))
    state_r[int(r_hs)]=1

    state_l = np.zeros((int(ad.full_hilbert_space_dim)))
    state_l[int(l_hs)]=1

    # Apply dipole operator to state_r and print out

    if verbose:
        print ("HS l: ",l_hs," HS r: ",r_hs )
        print ("<",n_state_l,"|r_x|",n_state_r,"> = ",np.dot(state_l,act(r_op[0],state_r,ad)))
        print ("<",n_state_l,"|r_y|",n_state_r,"> = ",np.dot(state_l,act(r_op[1],state_r,ad)))
        print ("<",n_state_l,"|r_z|",n_state_r,"> = ",np.dot(state_l,act(r_op[2],state_r,ad)))
        print (" ")


    # Write out to file (State l, state r, direction)
    with open("rij.dat","a") as rf: 
        rf.write('%f %f %f %f %f\n' % (n_state_l,n_state_r,1, \
                                        np.dot(state_l,act(r_op[0],state_r,ad)).real,\
                                        np.dot(state_l,act(r_op[0],state_r,ad)).imag))

        rf.write('%f %f %f %f %f\n' % (n_state_l,n_state_r,2, \
                                        np.dot(state_l,act(r_op[1],state_r,ad)).real,\
                                        np.dot(state_l,act(r_op[1],state_r,ad)).imag))

        rf.write('%f %f %f %f %f\n' % (n_state_l,n_state_r,3, \
                                        np.dot(state_l,act(r_op[2],state_r,ad)).real,\
                                        np.dot(state_l,act(r_op[2],state_r,ad)).imag))

    return

#*************************************************************************************
# Get the symmeetry eigenvalue of a state
def get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,Dij):
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

    Outputs: 
    char: Symmetry character <\psi_i | \hat{R} \vert \psi_i >
    '''

    # Test to make sure that the vacuum state is in the space
    if np.dot(ad.vacuum_state,ad.vacuum_state) < 1.e-10:
        print("ERROR: The vacuum state must be included for symmetry analysis (no constrained occ.)")
        sys.exit()

    n_orb=len(orb_names)
    n_spin=len(spin_names)
    n_fock= n_orb*n_spin

    
    i_state=eigensys[i_mb_st][0].split('>')
    
    n_i=len(i_state)

    op_on_vacuum=Operator()
    for ii in range(0,n_i):
        
        coeff1 = i_state[ii].split("*")[0]
        if coeff1 == '':
            continue

        # Remove space in negative numbers
        coeff1=float(str(coeff1).replace(" ",""))

        state1 = list(i_state[ii].split("*")[1])
        state1.remove('|')

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
    

    # TEST
    #print(op_on_vacuum)
    #quit()
    # Calculate character <\Psi | sym_op | \Psi>
    # i_mb_st is in terms of order in eigensys. Get state index in Hilbert space
    state_l = np.zeros((int(ad.full_hilbert_space_dim)))
    state_l[int(eigensys[i_mb_st][3])]=1
    
    char = np.dot(state_l,act(op_on_vacuum,ad.vacuum_state,ad))
    
    return char
 
#*************************************************************************************
# Get the matricies for symmetry representations
def construct_dij(n_orb,repsfile,flip=False):
    '''
    Read in the representation matricies, generated from the wannier functions
    
    Inputs: 
    n_orb: Number of orbitals r
    epsfile: File containing the representations 
    flip: Reverse the representation matricies, since
    Fock states expressed with orbital 0 on the far right.

    Outputs:
    dij: [symmetry operation,orbital character,spin character]
    '''

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
            #TEST
            #print(rot_mat)
            rot_mat = np.reshape(np.array(rot_mat,dtype=float),(3,3))

        # Orbital part of the representaion
        elif  "Representation (Orb)" in lines[line]:
            rep_mat = [] 
            for ii in range(1,n_orb+1):
                el=lines[line+ii].split()
                for jj in el:
                    rep_mat.append(jj)
            rep_mat = np.reshape(np.array(rep_mat,dtype=float),(n_orb,n_orb))
            
            # Flip the orbital reps (since MB states fo from right to left)
            if flip:
                rep_mat_flip=np.zeros((n_orb,n_orb))
                for ii in range(0,n_orb):
                    for jj in range(0,n_orb):
                        rep_mat_flip[ii][jj]=rep_mat[n_orb-1-ii][n_orb-1-jj]
                rep_mat=rep_mat_flip

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
# Calculates and sums the characters for orbitals in a degenerate group 
def mb_degerate_character(repsfile,fops,orb_names,spin_names,ad,eigensys,counts,out_label,state_limit=41,verbose=True,spin=True):
    '''
    Sum the characters for a degenerate manifold of many-body states.

    Input:
    repsfile: File containing symmetry representations
    fops: Many-body operators
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to atomic problem
    eigensys: Eigenstates (see sort_states)
    counts: List of degeneracies
    out_label: Label for output files
    state_limit: Number of states to find characters for
    verbose: Write stuff to STOUT
    spin: Spinfull reps?
    
    Output:
    None, output to mb_char.dat.

    '''
    # Extract the rotation matricies and reps for orbitals
    n_orb=len(orb_names)
    dij_orb_spin=construct_dij(n_orb,repsfile,flip=True)

    n_reps=len(dij_orb_spin)
 
    # Write to output file
    with open(out_label+"mb_char.dat","w+") as rf:

        #Counter for rep:
        i_rep=0
        for rep in dij_orb_spin:
     
            # Print symmetry operation
            rf.write("Sym op:\n")
            sym_op = rep[0]
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
                    char+=get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,dij_kron)
                    
                    #TEST
                    if verbose:
                        print(i_mb_st,get_char(i_mb_st,fops,orb_names,spin_names,ad,eigensys,dij_kron))
                rf.write(' %s %s %s %10.5f  \n' % \
                             ("Deg: ",deg," Char: ",char.real))

                #TEST
                print(' %s %s %s %s  \n' % \
                      ("Deg: ",deg," Char: ",char))

                i_state+=int(deg)
                if i_state > state_limit:
                    break

            # Let us know our progress
            print( ' %s %s %s %s' % ("Done sym_op",i_rep,"/",n_reps))
            i_rep += 1

    return

#*************************************************************************************
