#!/usr/bin/python

from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *
from itertools import product

from h5 import HDFArchive

import numpy as np
import sympy as sp

from op_dipole import * 
from apply_sym_wan import *

# Tools for sorting and printing the output of atom diag
# Cyrus Dreyer, Flatiron CCQ and Stony Brook University 
# 09/10/19

#*************************************************************************************
# Print out some properties of the ground state
def print_occ_ang_mom(orb_names,spin_names,fops,ad,ml_order,prt_L):
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

    # Get density matrix
    beta = 1e5
    dm = atomic_density_matrix(ad, beta)
    
    print("Ground state occupancies")
    for s in spin_names:
        for o1 in orb_names:
            print(o1,s, trace_rho_op(dm, n(s,o1) , ad))

    # Spin general version
    S2=spin_orbit_S2_op(fops)                
    Sz=spin_orbit_S_op(fops,proj='z')
    print("s(s+1) = ",trace_rho_op(dm, S2, ad))
    print("Sz = ",trace_rho_op(dm, Sz, ad))

    # Angular momentum
    if prt_L:
        if not ml_order:
            print('NEED TO SPECIFY ORERING OF ORBITALS!')
            raise
        
        # General version
        L2=spin_orbit_L2_op(fops,ml_order)
        Lx=spin_orbit_L_op(fops,ml_order,proj='x')
        Ly=spin_orbit_L_op(fops,ml_order,proj='y')
        Lz=spin_orbit_L_op(fops,ml_order,proj='z')

        print("l(l+1) = ",trace_rho_op(dm, L2, ad))
        print("Lx = ",trace_rho_op(dm, Lx, ad))
        print("Ly = ",trace_rho_op(dm, Ly, ad))
        print("Lz = ",trace_rho_op(dm, Lz, ad))

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
# Save atom_diag object
def print_at_diag(ad,out_label,eigensys,den_mats=[]):
    '''
    Save the atom diag object

    Inputs:
    ad: AtomDiag object
    out_label: label for output file

    Outputs:
    None

    '''
    
    with HDFArchive(out_label+'ad.h5') as ar:
        ar['ad'] = ad        
        ar['eigensys'] = eigensys

        if den_mats:
            ar['den_mats'] = den_mats

    return

#*************************************************************************************


#*************************************************************************************
# Create many-body wavefunctions for plotting/visualization. Based off of code from Gabriel
def print_mb_wfs(ad,wf_files,n_mbwfs,spin_names,orb_names,fops,eigensys,out_label,den_mats=[],verbose=True):#,den_occ_cmplx,mbwfs_frmt,mb_mesh,out_label,center_in_cell=False):
    '''
    Print the many-body wavefunctions by tracing over the density matrix
 
    Inputs:
    ad: Solution to the atomic problem
    wf_files: List of the files for wannier functions in real space, either xsf or cube (NOT TESTED)
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Fundamental operators
    eigensys: List of states and energies
    out_label: Label for output files
    verbose: Print status

    Outputs:
    None for now. Can return list of MB states on the real-space grid if needed.

    '''

    n_spin=len(spin_names)
    n_orb=len(orb_names)
    
    # Determine file type and read them in. Ordering should be the same as in the code
    if '.xsf' in wf_files[0]:
        file_type='xsf'
        wanns, delr, n_mesh, header=load_xsf_files(wf_files,flip_xz=False)

    elif '.cube' in wf_files[0]:
        file_type='cube'
        wanns, delr, n_mesh, header=load_xsf_files(wf_files)
    else:
        print('INVALID FILE FOR REAL SPACE WFs!')
        raise


    # Assume that if we have n_spin*n_orb Wannier functions, they are
    # spinful with the first half spin up, second half spin down. If
    # we have n_spin*n_orb/2, then they are spinless
    if len(wf_files)==n_spin*n_orb:
        wf_type='spinful'
        if verbose: print('NOTE: assuming spinful/spinorbit Wannier functions')
    elif n_spin==2 and len(wf_files)==n_orb:
        wf_type='spinless'
        if verbose:print('NOTE: assuming spinless Wannier functions')
    else:
        print('INCORRECT NUMBER OF WANNIER FUNCTIONS!')
        raise
    
    # Get the density matrix
    if not den_mats:
        print('Generating the density matrix')
        den_mats=get_den_mats(ad,spin_names,orb_names,fops,eigensys,n_mbwfs=n_mbwfs)
        
    mb_wfs=[]
    for iden_mat,den_mat in enumerate(den_mats):

        
        occs=np.diag(den_mat)

        # Make sure 1RDM is positive semi-definite. If so, get rid of imaginary part
        if all(iocc < 1.0E-10 for iocc in np.abs(occs.imag)) and any(iocc < 0 for iocc in occs.real):
            print('1RDM IS NOT POSITIVE SEMIDEFINITE!')
            raise
        elif any(iocc < 0 for iocc in np.abs(occs)):
            print('1RDM IS NOT POSITIVE SEMIDEFINITE!')
            raise
        else:
            occs=np.abs(np.diag(den_mat))
        
        
        mb_wf_spins=[]
        mb_wf_tot=np.zeros([n_mesh[0],n_mesh[1],n_mesh[2]])

        # Note: we always do two spins, only difference is if we have different WF for different spins
        for ispin in range(0,2): 
            mb_wf_spin=np.zeros([n_mesh[0],n_mesh[1],n_mesh[2]])
            if wf_type=='spinless':
                for iorb in range(0,n_orb):
                    mb_wf_tot+=occs[iorb+n_orb*ispin]*wanns[iorb]
                    mb_wf_spin+=occs[iorb+n_orb*ispin]*wanns[iorb]
            elif wf_type=='spinful':
                for iorb in range(0,int(n_orb/2)):
                    mb_wf_tot+=occs[iorb+int(n_orb/2)*ispin]*wanns[iorb+int(n_orb/2)*ispin]
                    mb_wf_spin+=occs[iorb+int(n_orb/2)*ispin]*wanns[iorb+int(n_orb/2)*ispin]
                    
            # Print out real space MB wavefunctions for given spin
            out_file_name='mbwf_'+str(iden_mat+n_mbwfs[0])+'_spin_'+str(ispin)+'.'+file_type
            if out_label:
                out_file_name+=out_label+'_'

            write_wann(out_file_name,mb_wf_spin,n_mesh,header)
            if verbose: print('Wrote mbwf '+str(iden_mat+n_mbwfs[0])+' for spin '+str(ispin))
                
            mb_wf_spins.append(mb_wf_spin)
                      
        # Print out total real space MB wavefunction
        out_file_name='mbwf_'+str(iden_mat+n_mbwfs[0])+'_tot'+'.'+file_type
        if out_label:
            out_file_name+=out_label+'_'

        write_wann(out_file_name,mb_wf_tot,n_mesh,header)
        if verbose:print('Wrote total mbwf '+str(iden_mat+n_mbwfs[0]))

        # Print out total real space MB spin density
        mb_spin_den=mb_wf_spins[0]**2-mb_wf_spins[1]**2
        
        out_file_name='mbwf_'+str(iden_mat+n_mbwfs[0])+'_spden'+'.'+file_type
        if out_label:
            out_file_name+=out_label+'_'

        write_wann(out_file_name,mb_spin_den,n_mesh,header)
        if verbose: print('Wrote mb spin den '+str(iden_mat+n_mbwfs[0]))

        mb_wfs.append([mb_wf_spins[:],mb_wf_tot])

    return #mb_wfs

#*************************************************************************************

#*************************************************************************************
# Print out information about the states
def sort_states(spin_names,orb_names,ad,fops,n_print,out_label,prt_mrchar=False,prt_state=True,prt_dm=True,target_mu=5,prt_ad=False,ml_order=[],verbose=True):
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
    prt_mrchar: Print multiref character
    prt_dm: Print 1rdm
    prt_ad: Print AtomDiag object

    Outputs:
    eigensys: List containing states, energies, and other info
    
    '''
    # General spin operators. Should work with spinful or spinless as
    # long as first half spin up, second half spin down.
    Sz=spin_orbit_S_op(fops,proj='z')
    S2=spin_orbit_S2_op(fops)
    
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    
    den_mats=[]    
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
            if sum(map(int,str(state))) != target_mu: 
                skip_sub=True                
                break
                
            state_leng=n_orb*n_spin
            fmt='{0:0'+str(state_leng)+'d}'
            state_bin="|"+fmt.format(state)+">"
            subspace_fock_state.append(state_bin)
            
        if skip_sub:
            n_eigenvec += len(ad.fock_states[sub]) 
            continue

        u_mat=ad.unitary_matrices[sub].conj().T
        st_mat=subspace_fock_state
            
        # store state and energy    
        for ind in range(0,ad.get_subspace_dim(sub)):

            eng=ad.energies[sub][ind]
            eigensys.append([[u_mat[ind,:],st_mat],eng,n_eigenvec])

            # Keep track of the absolute number of eigenstate
            n_eigenvec += 1

    # Sort by energy
    eigensys.sort(key=take_second)

    if verbose:
        print('Finished sorting states')
        
    # Make sure we are not trying to print more states than there are
    if n_print[1]>=len(eigensys):
        n_print[1]=len(eigensys)-1    

    # Slice eigensys to just states of interest 
    eigensys=eigensys[n_print[0]:n_print[1]+1]
        
    # print info for given number of states
    f_state= open(out_label+"eigensys.dat","w+")

    for ii in range(n_print[0],n_print[1]+1):

        # Only get spin info for states of interest
        spin=get_exp_val(ad,eigensys[ii][2],S2)
        ms=get_exp_val(ad,eigensys[ii][2],Sz)

        # Restore previous order
        eigensys[ii]=[eigensys[ii][0],eigensys[ii][1],spin.real,eigensys[ii][2],ms.real]
        
        # Write out info
        f_state.write('%10s %10s %10s %10s %10s %10s %10s %10s' % \
                      ("Energy:", np.round(float(eigensys[ii][1]),6), \
                       "HS eig num:",eigensys[ii][3],\
                       "s(s+1):", np.round(float(eigensys[ii][2]),4), \
                       "ms:",np.round(float(eigensys[ii][4]),3)))
        
        # Multireference character
        if prt_mrchar: # Character of mb state
            den_mat,multiref=check_multi_ref_state(ad,spin_names,orb_names,fops,int(eigensys[ii][3]))
            den_mats.append(den_mat)
                
            f_state.write('{:^16}'.format("MultiRef:"))
            if np.abs(multiref.imag) < 1.0E-10:
                f_state.write('{:6.4f}\n '.format(multiref.real))
            else:
                f_state.write('{:6.4f}\n '.format(multiref))
        else:
            f_state.write('\n')

        # State eigenvector
        if prt_state:

            eig_state=sp.Matrix(np.round(eigensys[ii][0][0],4)).dot(sp.symbols(eigensys[ii][0][1]))
            
            f_state.write('%10s %s\n' % ("State:",eig_state))
            f_state.write("\n")

    f_state.close()

    if verbose:
        print('Finished printing states')


    # Calculate and print the density matrix
    if prt_dm: 
        if not prt_mrchar:
            den_mats=get_den_mats(ad,spin_names,orb_names,fops,eigensys,n_mbwfs=[n_print[0],n_print[1]+1])
            
        with open(out_label+'den_mat.dat','w') as f:
            f.write('Density Matrices\n')
            f.write('\n')

            for iden,den_mat in enumerate(den_mats):            
                f.write('state: '+str(eigensys[iden][3])\
                        +'     eng: '+str(eigensys[iden][1])\
                        +'     s(s+1): '+str(np.round(float(eigensys[iden][2]),4)) \
                        +'     ms: '+str(np.round(float(eigensys[iden][4]),4))+'\n')
                for i in range(0,den_mat.shape[0]):
                    for j in range(0,den_mat.shape[1]):

                        if np.abs(den_mat[i,j].imag) < 1.0E-10:
                            f.write('{:.6f}\t '.format(den_mat[i,j].real))
                        else:
                            f.write('{:.6f}\t'.format(den_mat[i,j]))

                    f.write('\n')
                f.write('\n')

    
    # Save the AtomDiag object
    if prt_ad:
        print_at_diag(ad,out_label,eigensys,den_mats)

    return eigensys,den_mats
#*************************************************************************************

#*************************************************************************************
# For sorting
def take_second(elem):
    return elem[1]
#*************************************************************************************
