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
# Get expectation values of an operator for all states. Should be used in many of the operations below!!!
def get_exp_vals(ad,spin_names,orb_names,fops,operator,verbose=False):
    '''
    Get the density matricies <c^\dagger_i c_j> 

    Inputs: 
    ad: Solution to atomic problem
    spin_names: List of spins
    orb_names: Orbital names
    fops: Many-body operators
    eigensys:  states, energies, etc. from ad
    state: Index of the state (in energy ordering)
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
        exp_val=np.zeros(ad.get_subspace_dim(sub),dtype='complex128')
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
def read_rij(r_wan_file,n_orb,lat_param):
    '''
    Read in the dipole matrix elements for single particle basis
    
    Inputs:
    r_wan_file: wannier90_r file
    n_orb: Number of orbitals
    lat_param: Lattice parameters of the cell

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

    # I think we should make sure that diagonal elements are zero. Thus, we are shifting
    for orb in range(n_orb):
        for idir in range(3):
            rij[idir,orb,orb]=np.mod(rij[idir,orb,orb],lat_param[idir])

            #print('rii',orb,idir,lat_param[idir], rij[idir,orb,orb])
            
    return rij


#*************************************************************************************
# Hake the dipole operator
def make_dipole_op(ad,spin_names,orb_names,fops,dipol_file,tij,lat_param,velocity=False):
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
    rij=read_rij(dipol_file,n_orb,lat_param) # read in wannier90_r

    # Contsruct MB dipole operator
    r_op=[Operator(),Operator(),Operator()]

    if velocity:
        for s in spin_names:
            for i,o1 in enumerate(orb_names):
                for j,o2 in enumerate(orb_names):
                    for x in range(0,3):
                        p_ij=0.0
                        for k,o3 in enumerate(orb_names):
                            p_ij+=tij[i,k]*rij[x][k][j]-tij[j,k]*rij[x][k][i]
                        
                        r_op[x] += p_ij * c_dag(s,o1) * c(s,o2)
                        print('velocity',i,j,p_ij)
                        
    else:
        for s in spin_names:
            for o1 in orb_names:
                for o2 in orb_names:
                    for x in range(0,3):
                        r_op[x] += rij[x][int(o1)][int(o2)] * c_dag(s,o1) * c(s,o2)

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
def print_dipole_mat(n_dipol,ad,spin_names,orb_names,fops,dipol_file,eigensys,out_label,lat_param,tij):

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
    r_op=make_dipole_op(ad,spin_names,orb_names,fops,dipol_file,tij,lat_param)
    
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
        print('INCORRECT SPIN PROJECTION!')
        raise
    
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
# second half are spin down
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
         print('INVALID NUMBER OF ORBITALS! l=',ll)
         raise
     

    
#    if int(n_fops/2)==1: # s orbitals
#        ll=0
#    elif int(n_fops/2)==3: # p orbitals
#        ll=1
#    elif int(n_fops/2)==5: # d orbitals
#        ll=2
#    elif int(n_fops/2)==7: # f orbitals
#        ll=3
#    else:
#        print('INVALID NUMBER OF ORBITALS!')
#        raise
        
        
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
            print('INCORRECT SPIN PROJECTION!',proj)
            raise


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
