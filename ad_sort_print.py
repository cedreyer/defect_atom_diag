#!/usr/bin/python

from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *
from itertools import product

import numpy as np
import sympy as sp

from op_dipole import * 

# Tools for sorting and printing the output of atom diag
# Cyrus Dreyer, Flatiron CCQ and Stony Brook University 
# 09/10/19

#*************************************************************************************
# Print out some properties of the ground state
def print_occ_ang_mom(orb_names,spin_names,ad,dm,occ_print=True,s2_print=True,l2_print=False):
    '''
    Print to the terminal some properties to check the run
    
    Inputs:
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to the atomic problem
    dm: Density mnatrix
    occ_print: Whether to print occupancies
    s2_print: Whether to print total spin
    l2_print: Whether to print total angular momentum

    Outputs:
    None.
    '''

    if occ_print:
        print("Ground state occupancies")
        for s in spin_names:
            for o1 in orb_names:
                print(o1,s, trace_rho_op(dm, n(s,o1) , ad))

    # Angular momentum
    if s2_print:
        S2=S2_op(spin_names,orb_names,off_diag=True)
        print("s(s+1) = ",trace_rho_op(dm, S2, ad))
    if l2_print:
        L2=L2_op(spin_names,orb_names,off_diag=True)
        print("l(l+1) = ",trace_rho_op(dm, L2, ad))

    return
#*************************************************************************************

#*************************************************************************************
# Get energies, sort and find degeneracies
def get_eng_degen_eigensys(ad,eigensys,out_label,prec=3,out=True):
    '''
    Sort the energies and detect degeneracies
 
    Input:
    ad: Solution to the atomic problem
    eigensys: List of states and energies
    prec: Precision to define degeneracies
    out: Whether to print to terminal
    out_label: To label output file

    Outputs:
    counts: States in a given degeneracy
    '''
    energies = []
    for e in eigensys:
        energies.append(e[1])
    energies = np.round(energies,prec)

    unique_energies, counts = np.unique(energies, return_counts=True)

    # Just print out first 30
    if len(unique_energies) > 30:
        n_print=30
    else:
        n_print=len(unique_energies)
        
    if out:
        f_eng= open(out_label+"energy_deg.dat","w+")

        for u in range(n_print):
            print('E=',unique_energies[u],'#',counts[u])
            #print('{} {}'.format('E =',unique_energies[u]), ' # ', counts[u])
            f_eng.write(' %s  %s\n' % (unique_energies[u], counts[u]))
            
        f_eng.close()

    return counts
#*************************************************************************************

#*************************************************************************************
# Print out information about the states
def sort_states(spin_names,orb_names,ad,fops,n_print,out_label,prt_mrchar=False,prt_state=True,prt_dm=True,target_mu=5):
    '''
    Sort eigenstates and write them in fock basis
    
    Inputs:
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to the atomic problem
    fops: Many-body operators 
    n_print: Number of states to print out
    prt_state: Whether to print to eigensys file
    target_mu: Only keep states with a given occupation
    out_label: To label the output files for multiple runs

    Outputs:
    eigensys: List containing states, energies, and other info
    
    '''

    # Get spin eigenvalues, may be useful
    S2 = S2_op(spin_names, orb_names, off_diag=True)
    S2 = make_operator_real(S2)
    Sz=S_op('z', spin_names, orb_names, off_diag=True)
    Sz = make_operator_real(Sz)
    
    
    S2_states = quantum_number_eigenvalues(S2, ad)
    Sz_states = quantum_number_eigenvalues(Sz, ad)
 
    n_orb=len(orb_names)
    n_spin=len(spin_names)

    # For printing out the density matrix
    if prt_dm:
        with open(out_label+'den_mat.dat','w') as f:
            f.write('Density Matrices\n')
            f.write('\n')
    
    n_eigenvec=0.0
    eigensys=[]
    # Loop through subspaces
    for sub in range(0,ad.n_subspaces):

        skip_sub=False
        # get fock state in nice format
        subspace_fock_state=[]
        for fs in ad.fock_states[sub]:
            state =int(bin(int(fs))[2:])

            
            # Test to make sure particle number is the target_mu
            #if tune_opt and sum(map(int,str(state))) != target_mu: 
            #print(sum(map(int,str(state))))
            if sum(map(int,str(state))) != target_mu: 
                skip_sub=True                
                break
                
            state_leng=n_orb*n_spin
            fmt='{0:0'+str(state_leng)+'d}'
            state_bin="|"+fmt.format(state)+">"
            state_bin_sym=sp.symbols(state_bin)
            subspace_fock_state.append(state_bin_sym)
            
        if skip_sub:
            n_eigenvec += len(ad.fock_states[sub]) 
            continue
        
        # convert to eigenstate 
        kp = sp.kronecker_product
        u_mat=sp.Matrix(np.round(ad.unitary_matrices[sub],3).conj().T)
        st_mat=sp.Matrix(subspace_fock_state)
        eig_state=np.matrix(u_mat*st_mat)
        
        # format eigenstate:
        sub_state=[]
        for row in eig_state:
            sub_state.append(str(row).replace('[','').replace(']',''))    
        
        # store state and energy    
        for ind in range(0,ad.get_subspace_dim(sub)):

            eng=ad.energies[sub][ind]
            spin=round(float(S2_states[sub][ind]),3)
            ms=round(float(Sz_states[sub][ind]),3)

            
            # Get multireference character:
            if prt_mrchar:
                den_mat,multiref=check_multi_ref_state(ad,spin_names,orb_names,fops,n_eigenvec)
                eigensys.append([sub_state[ind],eng,spin,n_eigenvec,ms.real,multiref.real])

                # Print out density matrix
                if prt_dm:
                    with open(out_label+'den_mat.dat','a') as f:
                        f.write('state: '+str(n_eigenvec)+'     eng: '+str(eng)+'     s(s+1): '+str(spin)+'     ms: '+str(ms.real)+'\n')
                        for i in range(0,den_mat.shape[0]):
                            for j in range(0,den_mat.shape[1]):
                                f.write('%10.4f' % (den_mat[i,j]))
                            f.write('\n')
                        f.write('\n')
                        
            else:
                eigensys.append([sub_state[ind],eng,spin,n_eigenvec,ms.real])
                
            # Keep track of the absolute number of eigenstate
            n_eigenvec += 1

    # Sort by energy
    eigensys.sort(key=take_second)

    # print info for given number of states
    f_state= open(out_label+"eigensys.dat","w+")

    # Make sure we are not trying to print more states than there are
    if n_print[1]>len(eigensys):
        n_print[1]=len(eigensys)

    for ii in range(n_print[0],n_print[1]):
        # Write out info
        f_state.write('%10s %10s %10s %10s %10s %10s %10s %10s' % \
                      ("Energy:", np.round(float(eigensys[ii][1]),6), \
                       "HS eig num:",eigensys[ii][3],\
                       "s(s+1):", eigensys[ii][2], \
                       "ms:",np.round(float(eigensys[ii][4]),3)))
        
        # Optional stuff
        if prt_mrchar: # Character of mb state
            f_state.write('%10s %10s\n' % ("MultiRef:",np.round(float(eigensys[ii][5]),4)))
        else:
            f_state.write('\n')
            
            
        if prt_state:
            f_state.write('%10s %s\n' % ("State:",eigensys[ii][0]))
            f_state.write("\n")

    f_state.close()
    return eigensys
#*************************************************************************************

#*************************************************************************************
# For sorting
def take_second(elem):
    return elem[1]
#*************************************************************************************
