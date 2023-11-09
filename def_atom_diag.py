 #!/usr/OAbin/pythonA

from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *
from itertools import product

from h5 import HDFArchive

import numpy as np
import sympy as sp
import math
import sys
import time
import os

from ad_sort_print import *
from op_dipole import *

# For tprf HF
from triqs_tprf.tight_binding import TBLattice
from triqs_tprf.ParameterCollection import ParameterCollection
from triqs_tprf.hf_solver import HartreeSolver, HartreeFockSolver
from triqs_tprf.wannier90 import parse_hopping_from_wannier90_hr_dat
from triqs_tprf.wannier90 import parse_lattice_vectors_from_wannier90_wout



# Main script to run atom_diag calculations
# Cyrus Dreyer, Stony Brook University, Flatiorn CCQ, cyrus.dreyer@stonybrook.edu
# 09/11/19

# Other contributers:
# Gabriel Lopez-Morales, CUNY
# Malte Rosner, Radboud
# Alexander Hampel, Flatiron CCQ
# Danis Badrtdinov, Radboud

#*************************************************************************************
# Tune chemical, generate ad and dm
def solve_ad(H,spin_names,orb_names,fops,mu_in):
    '''
    Solve the atomic problem and tune the chemical potential to the
    target filling. Very inefficient way of doing it but works for now.

    Inputs:
    H: Hamiltonian operator
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    mu_in: Dictionary for chem pot options
    
    Outputs:
    H: Hamiltonian with chemical potential term added
    ad: Solution to the atomic problem
    N: Number operator
    '''
    
    mu_init=mu_in['mu_init']
    tune_occ=mu_in['tune_occ']
    target_occ=mu_in['target_occ']
    const_occ=mu_in['const_occ']
    step=mu_in['mu_step']
    
    if tune_occ and const_occ:
        print('ERROR: You have to choose between tuning and constraining the occupation.')
        sys.exit(1)
    elif not tune_occ and not const_occ:
        print('WARNING: Assuming tune_occ')

    
    # Setup the particle number operator
    N = Operator() # Number of particles
    for s in spin_names:
        for o in orb_names:
            N += n(s,o)

    filling=0
    mu=mu_init

    # Constrained occupation:
    if const_occ:
        ad = AtomDiagComplex(H, fops, n_min=int(target_occ), n_max=int(target_occ))

        
        
    # Tune the chemical potential to get desired occupancy, solve the
    # atomic problem
    elif tune_occ:
        while True:
            # Add chemical potential
            H += mu * N
            
            # Split the Hilbert space automatically
            try:
                ad = AtomDiagComplex(H, fops,n_min=0, n_max=int(target_occ)) # No need to go above n_max
            except:
                ad = AtomDiagComplex(H, fops) # No const occ for backwards compatability
                
            beta = 1e5
            dm = atomic_density_matrix(ad, beta)
            filling = trace_rho_op(dm, N, ad)

            print("mu:",mu,"filling:",filling)

            if abs(filling.real-target_occ) < 1.0e-4:
                break
            elif filling.real < target_occ:
                H += -mu * N
                mu+=-step
            elif filling.real > target_occ:
                H += -mu * N
                mu+=step
                
    else: # Use input chemical potential
        H += mu * N
        ad = AtomDiagComplex(H, fops)

    beta = 1e10
    dm = atomic_density_matrix(ad, beta)
    filling = trace_rho_op(dm, N, ad)

    if not const_occ:
        print("Chemical potential: ",mu)
    print('# of e-: ', filling)


    return ad
#*************************************************************************************

#*************************************************************************************
# Add the interaction Hamiltonian
def add_interaction(H,n_sites,spin_names,orb_names,fops,int_in,verbose=False):
    '''
    Add the interaction part to the Hamiltonian

    Inputs:
    H: Hamiltonian operator
    tij: Hopping matrix
    n_sites: Number of sites
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    int_in: Parameters from input file
    verbose: write out U matrix

    Outputs:
    H: Hamiltonian with interaction term added

    '''

    # Setup
    n_orb=len(orb_names)
    H_int=Operator()
    int_opt=int_in['int_opt']
    eps_eff=int_in['eps_eff']
    tij=int_in['tij']

    # To include spin polarization, uijkl and vijkl will be lists with either 1 or 4 elements
    uijkl=int_in['uijkl']
    vijkl=int_in['vijkl']
    
    if len(int_in['uijkl']) == 1:        
        spin_pol=False
    else:
        spin_pol=True

        # Only the full interaction for now with spin polarization!
        if int_opt != 0:
            print('ERROR: No use of int_opt other than 0 for spin polarization!')
            quit()

    
    # Need to flip indicies of uijkl coming from VASP
    if int_in['flip']:
        uijkl=flip_u_index(n_orb,uijkl)
        if int_in['vijkl']: vijkl = flip_u_index(n_orb, vijkl)
        print('uijkl -> uikjl')
    else:
#        uijkl=int_in['uijkl']
#        if int_in['vijkl']: vijkl = int_in['vijkl']
        print('uijkl -> uijkl')


    # BASIS CHANGE TO "BAND"
    if int_in['diag_basis']:
        tij,uijkl=wan_diag_basis(n_orb,tij,H_elmt='both',uijkls=uijkl)
        print('CHANGED TO BAND BASIS!')

        # Construct U matrix for d orbitals
        if int_opt == 6:
            print('SLATER U!!!!')
            Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl[0],verbose=False)
            # Basis from wannierization is z2, xz,yz,x2-y2,xy
            # Triqs = xy,yz,z^2,xz,x^2-y^2
            rot_def_to_w90 = np.array([[0, 0, 0, 0, 1],
                                       [0, 0, 1, 0, 0],
                                       [1, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0]])

            for iu,u in enumerate(uijkl):
                uijkl[iu]=U_matrix(2, radial_integrals=None, U_int=Uavg[iu], J_hund=1, basis='spherical', T=rot_def_to_w90.T)


    # Choices to average the uijkl matrix
    if int_opt == 1: # Use average U, U', and J
        Uavg,Uprime,Javg,uijkl=avg_U_sp(n_orb,uijkl,verbose=False)
    elif int_opt == 2: # Use two index Uij and Ji
        uij,jij,uijkl=make_two_ind_U_sp(n_orb,uijkl,verbose=False)
    # Only diagonal
    elif int_opt == 3:
        Uavg,Uprime,Javg,uijkl=avg_U_sp(n_orb,uijkl,verbose=False,U_elem=[True,False,False])
    # Only density-density
    elif int_opt == 4:
        Uavg,Uprime,Javg,uijkl=avg_U_sp(n_orb,uijkl,verbose=False,U_elem=[True,True,False])
    # U and J
    elif int_opt == 5:
        Uavg,Uprime,Javg,uijkl=avg_U_sp(n_orb,uijkl,verbose=False,U_elem=[True,False,True])
    # Construct U matrix for d orbitals
    elif int_opt == 6:
        print('SLATER U, Wan basis!!!!')
        Uavg,Uprime,Javg,uijkl=avg_U_sp(n_orb,uijkl,verbose=False)
        # Basis from wannierization is z2, xz,yz,x2-y2,xy
        # Triqs = xy,yz,z^2,xz,x^2-y^2
        rot_def_to_w90 = np.linalg.inv(np.array([[0, 0, 0, 0, 1],
                                   [0, 0, 1, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0]]))


        for iu,u in enumerate(uijkl):
            uijkl_tmp=U_matrix(2, radial_integrals=None, U_int=Uavg[iu], J_hund=Javg[iu], basis='cubic')
            uijkl[iu]=transform_U_matrix(uijkl_tmp,rot_def_to_w90)

    # Construct U matrix for d orbitals
    elif int_opt == 7:
        print('SLATER U, triqs basis!!!!')
        Uavg,Uprime,Javg,uijkl=avg_U_sp(n_orb,uijkl,verbose=False)
        # Basis from wannierization is z2, xz,yz,x2-y2,xy
        # Triqs = xy,yz,z^2,xz,x^2-y^2

        for iu,u in enumerate(uijkl):
            uijkl[iu]=U_matrix(2, radial_integrals=None, U_int=Uavg[iu], J_hund=Javg[iu], basis='cubic')


    elif int_opt == 8: # Use two index Uij and Ji with effective eps_eff (Danis)

        uij=[]
        jij=[]
        for iu,u in enumerate(uijkl):
            uij_iu,jij_iu,uijkl_iu=effective_screening(n_orb,uijkl[iu],vijkl[iu],eps_eff,verbose=False)
            uij.append(uij_iu)
            jij.append(jij_iu)
            uijkl[iu]=uijkl_iu
            


    #Print U matrix
    #if verbose:
    #    for ii in range(0,n_orb):
    #        for jj in range(0,n_orb):
    #            for kk in range(0,n_orb):
    #                for ll in range(0,n_orb):
    #                    if abs(np.real(uijkl[ii,jj,kk,ll])) > 1e-5:
    #                        print("{:.0f}    {:.0f}     {:.0f}     {:.0f}    {:.5f}"\
    #                              .format(ii,jj,kk,ll,np.real(uijkl[ii,jj,kk,ll])))


    # SPIN POL UP TO HERE
    
    # Check the symmetry of the U matrix
    if int_in['sym']:
        n_spin=len(spin_names)
        check_sym_u_mat(uijkl,n_orb,n_spin)

    # Add the interactions
    if spin_pol:
        H_int= make_spin_pol_H_int(uijkl,spin_names,orb_names,fops)
    else:
        # For backwards compatibility, use either number of orbitals or list
        try:
            H_int = h_int_slater(spin_names, len(orb_names), uijkl[0], off_diag=True,complex=True)
        except:
            H_int = h_int_slater(spin_names, orb_names, uijkl[0], off_diag=True,complex=True)
        
    if verbose:
        print('')
        print("Interaction:")
        print(H_int)
        print('')
    
    H+=H_int
    
    return H
#*************************************************************************************

#*************************************************************************************
# Convert basis of uijkl to wan. diagonal
def wan_diag_basis(n_orb,tijs,H_elmt='both',uijkls=[],verbose=False):
    '''
    Convert Hamiltonian to basis where tij is diagonal

    Inputs:
    H_elmt: 'tij': Just convert hoppings, 'both': convert both
    tijs: Hopping matrix, np array
    uijkls: Interaction matrix, np array

    Outputs:
    tij_conv: diagonalized tij
    uijkl_conv: uijkl matrix in diagonal basis

    '''

    tij_conv=[]
    for tij in tijs:

        print('TIJ',tij)
        
        evals,evecs=np.linalg.eig(tij)
        if verbose:
            print('wannier eval order:',evals)
        tij_conv.append(np.multiply(np.identity(n_orb),evals))
        
    #print(tij_conv)
    #quit()
    # Convert to wannier basis
    #wan_den=np.dot(np.dot(np.linalg.inv(evecs.T),mo_den),np.linalg.inv(evecs))

    if H_elmt == 'both':

        uijkl_conv=[]
        for uijkl in uijkls:
            #uijkl_conv=uijkl
            basis_trans=np.linalg.inv(evecs)#evecs.T????
            uijkl_conv.append(transform_U_matrix(uijkl,basis_trans))
            #print('WARNING: Band basis with spin polarization not tested!') 

        uijkls=uijkl_conv
            
    return tij_conv, uijkls

#*************************************************************************************

#*************************************************************************************
# Check if the  U matrix obeys symmetry of single particle reps
def check_sym_u_mat(uijkls,n_orb,n_spin,dij=[]):
    '''
    Checks to see if U matrix is consistent with symmetry of single
    particle reps

    Inputs:
    uijkl: n_orb x n_orb matrix of U values
    n_orb: Number of orbitals

    Outputs:
    None
    '''

    # Get reps from reps.dat
    if not dij:
        dij=construct_dij(n_orb,n_spin,"reps.dat")

    for uijkl in uijkls:
        i_rep=0
        print("Check if uijkl obeys sym of reps:")
        for rep in dij:

            # Transform U matrix
            Tij=rep[1]
            uijkl_t=transform_U_matrix(uijkl,Tij)

            print('%s %f %s %f' % ("For rep: ",i_rep," max val: ",np.amax(np.real(uijkl-uijkl_t))))
            i_rep+=1

    return
#*************************************************************************************

#*************************************************************************************
# Check if the hopping obeys symmetry
def check_sym_t_mat(tijs,n_orb,n_spin,dij=[]):
    '''
    Checks to see if tij matrix is consistent with symmetry of single
    particle reps

    Inputs:
    tij: n_orb x n_orb matrix of tij values
    n_orb: Number of orbitals
    n_orb: Number of explicit spin states

    Outputs:
    None
    '''

    #TODO: If we have SOC, this will need to be modified to construct the full spinful reps
    
    # Get reps from reps.dat
    if not dij:
        dij=construct_dij(n_orb,n_spin,"reps.dat")

    for tij in tijs:
        print("Check if tij commutes with reps:")
        for i_rep,rep in enumerate(dij):
            print('%s %f %s %f' % ("For rep: ",i_rep," max val: ",\
                                   np.abs(np.amax(np.matmul(rep[1],tij)-np.matmul(tij,rep[1])))))

    return
#*************************************************************************************

#*************************************************************************************
# Add hoppings
def add_hopping(H,spin_names,orb_names,int_in,verbose=False):
    '''
    Add kinetic part to the Hamiltonian.

    Inputs:
    H: Hamiltonian operator
    spin_names: List of spins
    orb_names: List of orbitals
    tij: Hopping matrix elements

    Outputs:
    H: Hamiltonian with kinetic part
    '''

    # Test for spin pol
    if len(int_in['tij']) == 1:
           tij=[int_in['tij'][0],int_in['tij'][0]]
    else:
           tij=[int_in['tij'][0],int_in['tij'][1]]

    # For testing to check the symmetry of the tij matrix
    n_orb=len(orb_names)
    n_spin=len(spin_names)

    # TEST: use diagonal basis
    if int_in['diag_basis']:
        tij,uijkl=wan_diag_basis(n_orb,tij,H_elmt='tij')

    if int_in['sym']: # Check the sym verus provided reps
        check_sym_t_mat(tij,n_orb,n_spin)


    # Make noniteracting operator
    H_kin=Operator()
    for ispin,s in enumerate(spin_names):
        for o1 in orb_names:
            for o2 in orb_names:
                H_kin += 0.5*(tij[ispin][int(o1),int(o2)] * c_dag(s,o1) * c(s,o2)+ np.conj(tij[ispin][int(o1),int(o2)])*c_dag(s,o2) * c(s,o1))
    H += H_kin

    if verbose:
        print('')
        print("Ind prtcl:")
        print(H_kin)
        print('')

    return H
#*************************************************************************************

#*************************************************************************************
# Fix the fact that U_ijkl out of vasp has indicies switched
def flip_u_index(n_orb,uijkls):
    '''
    uijkls -> uikjl for UIJKL read in from VASP

    Input:
    n_orb: Number of orbitals
    uijkls: n_orb x n_orb Uijkl matrix (possibly four if spin polarized)

    Output:
    uijkls_new: Uijkl with indicies switched
    '''

    uijkls_new=[]
    for uijkl in uijkls:
        
        if isinstance(uijkl[0,0,0,0],complex):
            uijkl_new=np.zeros((n_orb,n_orb,n_orb,n_orb),dtype=np.complex128)
        else:
            uijkl_new=np.zeros((n_orb,n_orb,n_orb,n_orb),dtype=np.float64)
        
        for i in range(0,n_orb):
            for j in range(0,n_orb):
                for k in range(0,n_orb):
                    for l in range(0,n_orb):
                        uijkl_new[i,j,k,l]=uijkl[i,k,j,l]

        uijkls_new.append(uijkl_new)
        
    return uijkls_new
#*************************************************************************************


#*************************************************************************************
# Double counting
def add_double_counting(H,spin_names,orb_names,fops,int_in,mo_den,verbose=False):
    '''
    Subtract double counting correction to the Hamiltonian.

    Inputs:
    H: Hamiltonian operator
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    uijkl: n_orb x n_orb Uijkl matrix (screened)
    vijkl: n_orb x n_orb Vijkl matrix (bare)
    tij: Hopping matrix elements
    mo_den: Occupation in band basis
    dc_x_wt: Weight on exchange term in HF
    DC_opt: 0: full uijkl, 1: orbital average, 2: Two index
    flip: Flip uijkl here?
    eps_eff: Effective screening parameter: U1 = V1/eps_eff

    Outputs:
    H: Hamiltonian with DC subtracted
    '''

    # Get variables from int_in. NOTE: NO SPIN POLARIZATION YET!
    tij=int_in['tij']
    dc_x_wt=int_in['dc_x_wt']
    dc_opt=int_in['dc_opt']
    eps_eff=int_in['eps_eff']
    uijkl=int_in['uijkl']
    vijkl= int_in['vijkl']

    if len(int_in['uijkl']) == 1:        
        spin_pol=False
    else:
        spin_pol=True

        # Only the full interaction for now with spin polarization!
        if dc_opt != 0:
            print('ERROR: No use of int_opt other than 0 for spin polarization!')
            quit()
    
    n_orb=len(orb_names)

    if int_in['flip']:
        uijkl=flip_u_index(n_orb,uijkl)
        if int_in['vijkl']: vijkl = flip_u_index(n_orb,vijkl)

    # BASIS CHANGE TO "BAND"
    if int_in['diag_basis']:
        tij,uijkl=wan_diag_basis(n_orb,tij,H_elmt='both',uijkls=uijkl)

    # if dc_opt < 0: # Use averaged density
    #    for mo in mo_den:
    #    avg_den=np.trace(mo_den)/n_orb
    #    wan_den=np.identity(n_orb)*avg_den
    #    dc_opt=abs(dc_opt)

    # Convert DFT density to wannier basis
    # Find matrix that converts between wannier and MO reps

    wan_den=[]
    for it,t in enumerate(tij):
        evals,evecs=np.linalg.eig(t)

        print("Ordering of states",evals)
        
        # Convert to wannier basis
        evecs=np.matrix(evecs)
        wan_den.append(np.dot(np.dot(np.linalg.inv(evecs.H),mo_den[it]),np.linalg.inv(evecs)))
        
    # Mixing of exchange part
    print("DC mix:",dc_x_wt)

    # Choices to average the uijkl matrix
    if dc_opt == 1: # Use average U, U', and J
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False)
    elif dc_opt == 2: # Use two index Uij and Ji
        uij,jij,uijkl=make_two_ind_U(n_orb,uijkl,verbose=False)
    elif dc_opt == 3: # Only diagonal
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,False,False])
    # Only density-density
    elif dc_opt == 4:
        print('Only density-density term will be used for DC')
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,True,False])
    # U and J
    elif dc_opt == 5:
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,False,True])

    elif dc_opt == 8: # Use two index Uij and Ji with effective eps_eff (Danis)

        uij=[]
        jij=[]
        for iu,u in enumerate(uijkl):
            uij_iu,jij_iu,uijkl_iu=effective_screening(n_orb,uijkl[iu],vijkl[iu],eps_eff,verbose=False)
            uij.append(uij_iu)
            jij.append(jij_iu)
            uijkl[iu]=uijkl_iu

        #uij,jij,uijkl=effective_screening(n_orb,uijkl,vijkl,eps_eff,verbose=False)

    if verbose:
        print('')
        print("DC:")


    H_dc=get_DC_from_uijkl(spin_names,orb_names,fops,uijkl,dc_x_wt,wan_den,spin_pol)
#    H_dc=Operator()
#    for s in spin_names:
#        for i in range(0,n_orb):
#            for j in range(0,n_orb):
#                fock=0
#                for k in range(0,n_orb):
#                    for l in range(0,n_orb):
#                        fock += 1.0*(uijkl[i,l,j,k] - dc_x_wt*uijkl[i,l,k,j] ) * wan_den[k,l] # From Szabo and Ostland

#                        # TEST: See what terms are included
#                        if verbose:
#                            if abs(1.0*(uijkl[i,l,j,k] - dc_x_wt*uijkl[i,l,k,j] ) * wan_den[k,l]) > 0.00001:
#                                print(s,i,j,k,l,uijkl[i,l,j,k]-dc_x_wt*uijkl[i,l,k,j]* wan_den[k,l])

#                # For both integer and string orbital names
#                if isinstance(orb_names[0], int):
#                    H_dc += -fock * c_dag(s,i) * c(s,j)
#                else:
#                    H_dc += -fock * c_dag(s,str(i)) * c(s,str(j))

#                if verbose:
#                    if abs(fock) > 1.0e-1:
#                                print(s,i,j,-fock)


    H += H_dc


    return H
#*************************************************************************************

#*************************************************************************************
# Calculate the H_dc from a give uijkl, allow for spin_pol
def get_DC_from_uijkl(spin_names,orb_names,fops,uijkls,dc_x_wt,wan_den,spin_pol,verbose=False):

    '''
    '''

    n_orb=len(orb_names)
    H_dc=Operator()

    if spin_pol:

        wan_del_t=wan_den[0]+wan_den[1]
        
        for is1,s in enumerate(spin_names):
            is2=1-is1
            
            for i in range(0,n_orb):
                for j in range(0,n_orb):
                    fock=0
                    for k in range(0,n_orb):
                        for l in range(0,n_orb):
                            # From Szabo and Ostland Eq. 3.349 pg 214 (unrestricted expression), and Miguel :). Factor of 2 to cancel the default dc_x_wt=0.5
                            fock_term = uijkls[2][i,l,j,k]*wan_den[is2][k,l]+uijkls[is1][i,l,j,k]*wan_den[is1][k,l] - 2.0*dc_x_wt*uijkls[is1][i,l,k,j]*wan_den[is1][k,l] 

                            fock += fock_term
                            # TEST: See what terms are included
                            if verbose:
                                if abs(fock_term) > 0.00001:
                                    print(s,i,j,k,l,fock_term)

                    # For both integer and string orbital names
                    if isinstance(orb_names[0], int):
                        H_dc += -fock * c_dag(s,i) * c(s,j)
                    else:
                        H_dc += -fock * c_dag(s,str(i)) * c(s,str(j))
                        
                    if verbose:
                        if abs(fock) > 1.0e-1:
                            print(s,i,j,-fock)

                    
    else:
        for s in spin_names:
            for i in range(0,n_orb):
                for j in range(0,n_orb):
                    fock=0
                    for k in range(0,n_orb):
                        for l in range(0,n_orb):
                            fock += 1.0*(uijkls[0][i,l,j,k] - dc_x_wt*uijkls[0][i,l,k,j] ) * wan_den[0][k,l] # From Szabo and Ostland

                            # TEST: See what terms are included
                            if verbose:
                                if abs(1.0*(uijkls[0][i,l,j,k] - dc_x_wt*uijkls[0][i,l,k,j] ) * wan_den[0][k,l]) > 0.00001:
                                    print(s,i,j,k,l,uijkls[0][i,l,j,k]-dc_x_wt*uijkls[0][i,l,k,j]* wan_den[0][k,l])

                    # For both integer and string orbital names
                    if isinstance(orb_names[0], int):
                        H_dc += -fock * c_dag(s,i) * c(s,j)
                    else:
                        H_dc += -fock * c_dag(s,str(i)) * c(s,str(j))

                    if verbose:
                        if abs(fock) > 1.0e-1:
                            print(s,i,j,-fock)


    return H_dc

#*************************************************************************************
# Calculate the MF HF coulomb part and use it as DC
def add_hartree_fock_DC(H,spin_names,orb_names,fops,int_in,target_occ):
    '''
    Use TPRF to solve H with Hartree-Fock, to use the Coulomb interaction as a DC
    
    Inputs:
    H: The Hamiltonian
    spin_names: Spin names
    orb_names: Orbital names
    fops: Fundamental operators
    int_in: Input parameters governing interaction
    target_occ: Number of electrons

    Outputs:
    H: Hamiltonian with HF DC subtracted

    '''
    
    if not isinstance(orb_names[0], int):
        print('ERROR: Need to specify n_orb NOT orb_names in input file so names are integers for the HF solver!!')
        quit()
    
    gf_struct = [[spin_names[0], len(orb_names)],
                 [spin_names[1], len(orb_names)]]
    beta_temp=100.0
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    
    orbital_positions = [(0,0,0)]*n_orb*n_spin

    # Use diagonal basis
    tij=int_in['tij']

    if int_in['diag_basis']:
        tij,uijkl=wan_diag_basis(n_orb,int_in['tij'],H_elmt='tij')


    if len(uijkl) > 1:
        print('ERROR: HF solver does not work with spin polarization')
    else:
        tij=tij[0]
        
    h_loc=np.kron(np.identity(2),tij)
    nb=4
    hop={(0,0,0):h_loc}

    units = parse_lattice_vectors_from_wannier90_wout('wannier90.wout')
        
    tbl=TBLattice(units=units,hopping=hop,orbital_positions=orbital_positions)#, orbital_names=nms)
    kmesh=tbl.get_kmesh(n_k=(1,1,1))
    h_k=tbl.fourier(kmesh)
    
    # Interaction part of H:
    H_int=Operator()
    H_int=add_interaction(H_int,1,spin_names,orb_names,fops,int_in)

    print(H_int)
    
    # Solve hartree-fock equation
    HFS = HartreeFockSolver(h_k, beta_temp, H_int=H_int, gf_struct=gf_struct)
    HFS.solve_iter(target_occ)
    #HFS.solve_newton(target_occ)

    # Make noniteracting operator
    H_kin=Operator()
    for s in spin_names:
        for o1 in orb_names:
            for o2 in orb_names:
                H_kin += -0.5*HFS.M[int(o1),int(o2)] * (c_dag(s,o1) * c(s,o2)+ c_dag(s,o2) * c(s,o1))

    H+=H_kin

    return H

#*************************************************************************************

#*************************************************************************************
# DC correction for cbcn using the "DFT" approach from Malte
def add_cbcn_DFT_DC(H,spin_names,orb_names,fops,int_in,mo_den,verbose=True):
    '''
    Dimer double counting from Malte and Van Loon
    
    Inputs:
    H: The Hamiltonian
    spin_names: Spin names
    orb_names: Orbital names
    fops: Fundamental operators
    int_in: Input parameters governing interaction
    mo_den: Density in the band basis
    verbose: Output more to the terminal

    Outputs:
    H: Hamiltonian with dimer DC subtracted

    '''
    

    if len(int_in['uijkl']) > 1:
        print('ERROR: No dimer DC for spin polarized interaction!')
        quit()

    uijkl=int_in['uijkl']
    tij=int_in['tij']
    n_orb=len(orb_names)

        
    # Get evecs to eventually transform back
    if not int_in['diag_basis']:
        evals,evecs=np.linalg.eig(int_in['tij'])

    if int_in['flip']:
        uijkl=flip_u_index(n_orb,uijkl)

    # Basis change to band
    tij,uijkl=wan_diag_basis(n_orb,tij,H_elmt='both',uijkls=uijkl)

    # Assume that our basis is [[b*,0],[0,b]]
    tij_dc=np.array([[uijkl[0][1,0,1,0],0.0],[0.0,uijkl[0][1,1,1,1]]])

    # If in orbital basis, transform back
    if not int_in['diag_basis']:
        tij_dc=np.matmul(np.matmul(np.linalg.inv(evecs.T),tij_dc),evecs.T)

    # Make noniteracting operator
    H_dft_dc=Operator()
    for s in spin_names:
        for o1 in orb_names:
            for o2 in orb_names:
                H_dft_dc += -0.5*tij_dc[int(o1),int(o2)] * (c_dag(s,o1) * c(s,o2)+ c_dag(s,o2) * c(s,o1))

    H += H_dft_dc

    if verbose:
        print("DFT DC:")
        print(H_dft_dc)

    return H

#*************************************************************************************

#*************************************************************************************
# Average U values, allowing for potentially spin-polarized U
def make_two_ind_U_sp(n_orb,uijkls,verbose=False,U_elem=[True,True],spin_pol_avg=False):

    '''

    '''
    
    uijs=[]    
    jijs=[]
    uijkls_avg=[]
    for uijkl in uijkls:
        uij, jij, uijkl_avg=make_two_ind_U(n_orb,uijkl,verbose=verbose,U_elem=U_elem)
        uijs.append(Uavg)
        jijs.append(Uprimes)
        uijkls_avg.append(uijkl_avg)

    if spin_pol_avg:
        uijs=np.average(np.array(uijs))
        jijs=np.average(np.array(jijs))
        uijkls_avg=np.average(np.array(uijkls_avg))

    return uijs, jijs, uijkls_avg

#*************************************************************************************

#*************************************************************************************
# Average U values
def make_two_ind_U(n_orb,uijkl,verbose=False,U_elem=[True,True]):
    '''
    Reduce full interaction to two index.

    Inputs:
    n_orb: Number of orbitals
    uijkl: Full interaction tensor
    verbose: Write out more to the terminal
    U_elem: Whether to include U and J

    Outputs:
    uij: Two index direct interaction
    jij: Two index exchange
    uijkl_avg: Interaction with just uij and jij elements
    
    '''
    
    uij = np.zeros((n_orb, n_orb))
    jij = np.zeros((n_orb, n_orb))
    uijkl_avg = uijkl
    # Get the average values, indicies should be flipped already!!!
    for ii in range(0, n_orb):
        for jj in range(0, n_orb):
            for kk in range(0, n_orb):
                for ll in range(0, n_orb):
                    if ii == kk and jj == ll and U_elem[0]:  # Density density: Uii,ii and Uijij
                        if verbose:
                            print(ii, jj, kk, ll, uijkl[ii, jj, kk, ll], "Added to uij")
                        uij[ii, jj] = uijkl[ii, jj, kk, ll]
                    elif ii == ll and kk == jj and U_elem[1]:  # Hunds: Uij,ji
                        if verbose:
                            print(ii, jj, kk, ll, uijkl[ii, jj, kk, ll], "Added to jij")
                        jij[ii, jj] = uijkl[ii, jj, kk, ll]

                    # MALTE's modification
                    elif ii == jj and kk == ll and U_elem[1]:  # Hunds: Uij,ji ... with pair hopping
                        if verbose:
                            print(ii, jj, kk, ll, uijkl[ii, jj, kk, ll], "Added to jij")
                        jij[ii, jj] = uijkl[ii, jj, kk, ll]

                    else:
                        uijkl_avg[ii, jj, kk, ll] = 0.0


    # For testing
    if verbose:
        print(" ")
        for ii in range(0, n_orb):
            for jj in range(0, n_orb):
                for kk in range(0, n_orb):
                    for ll in range(0, n_orb):
                        print(ii, jj, kk, ll, uijkl_avg[ii, jj, kk, ll])

    return uij, jij, uijkl_avg
#*************************************************************************************

#*************************************************************************************
# Average U values (Danis)
def effective_screening(n_orb,uijkl,vijkl,eps_eff,verbose=False,U_elem=[True,True]):

    print('Effective screening scheme will be used')
    print('Please, make sure that density-density and Hunds exchange give a proper approximation!')

    uij=np.zeros((n_orb,n_orb))
    vij=np.zeros((n_orb, n_orb))
    jij=np.zeros((n_orb,n_orb))
    uijkl_avg=uijkl
    # Get the average values, indicies should be flipped already!!!
    for ii in range(0,n_orb):
        for jj in range(0,n_orb):
            for kk in range(0,n_orb):
                for ll in range(0,n_orb):
                    if ii==kk and jj==ll and U_elem[0]: # Density density: Uii,ii and Uijij
                        if verbose:
                            print(ii,jj,kk,ll,uijkl[ii,jj,kk,ll],"Added to uij")
                        uij[ii,jj]=uijkl[ii,jj,kk,ll]
                        vij[ii,jj]=vijkl[ii,jj,kk,ll]  #bare Coulomb interaction
                    elif ii==ll and kk==jj and U_elem[1]: # Hunds: Uij,ji
                        if verbose:
                            print(ii,jj,kk,ll,uijkl[ii,jj,kk,ll], "Added to jij")
                        jij[ii,jj]=uijkl[ii,jj,kk,ll]


    evals, evecs = np.linalg.eig(uij)
    evals_bare, evecs_bare = np.linalg.eig(vij)
    evals[0] = evals_bare[0]/eps_eff # U1 = V1/eps_eff according to effective screening, eval[0] corresponds to the largest eigenvalue
    print('eps_eff:', eps_eff)
    uij_conv = np.matmul(np.matmul(np.linalg.inv(evecs.T), np.multiply(np.identity(n_orb),evals)), evecs.T).T

    if verbose:
        print('Uij:')
        for ii in range(0, n_orb):
            for jj in range(0, n_orb):
                print(ii, jj, uij[ii, jj])

    if verbose:
        print('\n Uij_conv:')
        for ii in range(0, n_orb):
            for jj in range(0, n_orb):
                print(ii, jj, uij_conv[ii, jj])


#    # Now construct a full uijkl matrix with averaged values
    uijkl_avg=np.zeros((n_orb,n_orb,n_orb,n_orb))
    for ii in range(0,n_orb):
        for jj in range(0,n_orb):
            for kk in range(0,n_orb):
                for ll in range(0,n_orb):
                    if ii==kk and jj==ll: # Indicies should be flipped already
                        uijkl_avg[ii,jj,kk,ll]=uij_conv[ii,jj]
                    elif ii==ll and kk==jj:
                        uijkl_avg[ii,jj,kk,ll]=jij[ii,kk]

    # For testing

    return uij,jij,uijkl_avg

#*************************************************************************************

#*************************************************************************************
# Average U values, accounting for possible spin polarized Wannier functions
def avg_U_sp(n_orb,uijkls,verbose=False,U_elem=[True,True,True],triqs_U=False,spin_pol_avg=False):
    '''

    '''

    Uavgs=[]
    Uprimes=[]
    Javgs=[]
    uijkls_avg=[]
    for uijkl in uijkls:
        Uavg,Uprime,Javg,uijkl_avg=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,True,True],triqs_U=False)
        Uavgs.append(Uavg)
        Uprimes.append(Uprimes)
        Javgs.append(Javgs)
        uijkls_avg.append(uijkl_avg)

    if spin_pol_avg:
        Uavgs=np.average(np.array(Uavgs))
        Uprimes=np.average(np.array(Uprimes))
        Javgs=np.average(np.array(Javgs))
        uijkls_avg=np.average(np.array(uijkls_avg))

    return Uavgs,Uprimes,Javgs,uijkls_avg
        
#*************************************************************************************

#*************************************************************************************
# Average U values
def avg_U(n_orb,uijkl,verbose=False,U_elem=[True,True,True],triqs_U=False):
    '''
    

    '''
    
    Uavg=0.0
    Uprime=0.0
    Javg=0.0
    # Get the averaged values
    for ii in range(0,n_orb):
        for jj in range(0,n_orb):
            for kk in range(0,n_orb):
                for ll in range(0,n_orb):
                    if ii==jj and jj==kk and kk==ll:
                        if verbose:
                            print(ii,jj,kk,ll,uijkl[ii,jj,kk,ll],"Uavg")
                        Uavg+=(1.0/n_orb)*uijkl[ii,jj,kk,ll]
                    elif ii==kk and jj==ll:
                        if verbose:
                            print(ii,jj,kk,ll,uijkl[ii,jj,kk,ll], "Uprime")
                        Uprime+=(1.0/(n_orb**2-n_orb))*uijkl[ii,jj,kk,ll]
                    elif ii==ll and kk==jj: # Hunds: Uij,ji
                        if verbose:
                            print(ii,jj,kk,ll,uijkl[ii,jj,kk,ll],"Javg")
                        Javg+=(1.0/(n_orb**2-n_orb))*uijkl[ii,jj,kk,ll]

    # Now construct a full uijkl matrix with averaged values
    if not triqs_U:
        uijkl_avg=np.zeros((n_orb,n_orb,n_orb,n_orb))
        for ii in range(0,n_orb):
            for jj in range(0,n_orb):
                for kk in range(0,n_orb):
                    for ll in range(0,n_orb):
                        if ii==jj and jj==kk and kk==ll and U_elem[0]:
                            uijkl_avg[ii,jj,kk,ll]=Uavg
                        elif ii==kk and jj==ll and U_elem[1]:
                            uijkl_avg[ii,jj,kk,ll]=Uprime
                        elif ii==ll and kk==jj and U_elem[2]:
                            uijkl_avg[ii,jj,kk,ll]=Javg
#    else:
#        uijkl_avg = U_matrix(l=2, U_int=Uavg, J_hund=Javg, basis='spheric')

        # Need to make this more general
#        rot_def_to_w90 = np.array([[0, 0, 0, 0, 1],
#                               [0, 0, 1, 0, 0],
#                               [1, 0, 0, 0, 0],
#                               [0, 1, 0, 0, 0],
#                               [0, 0, 0, 1, 0]])

#        rot_def_to_w90 = np.array([[1, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 1],
#                               [0, 1, 0, 0, 0],
#                               [0, 0, 0, 1, 0],
#                               [0, 0, 1, 0, 0]])


#        uijkl_avg = transform_U_matrix(uijkl_avg, rot_def_to_w90)

    # For testing
    print(" ")
    print("Uavg",Uavg,"Uprime",Uprime,"Javg",Javg)

    if verbose and 1==2:
        for ii in range(0,n_orb):
            for jj in range(0,n_orb):
                for kk in range(0,n_orb):
                    for ll in range(0,n_orb):
                        print(ii,jj,kk,ll,uijkl_avg[ii,jj,kk,ll])


    return Uavg,Uprime,Javg,uijkl_avg
#*************************************************************************************

#*************************************************************************************
# Setup the Hamiltonian
def setup_H(spin_names,orb_names,fops,comp_H,int_in,mu_in,verbose,mo_den=[]):
    '''
    Add terms to the Hamiltonian depending on the input file.

    Inputs:
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    comp_H: List of components of the Hamiltonian
    int_in: Input parameters
    verbose: Print out parts of H
    mo_den: Occupation in the band basis

    Outputs:
    H: Hamiltonian
    ad: Solution to atomic problem
    dm: Density matrix
    N: Number operator
    '''
    
    # Setup the Hamiltonian
    H=Operator()
    
    # kinetic
    if comp_H['Hkin']:

        H = add_hopping(H,spin_names,orb_names,int_in,verbose=verbose)
        
    # Double counting
    if comp_H['Hdc']:

        if len(int_in['uijkl']) > 1 or len(int_in['tij']) > 1:
            print('WARNING: DC not tested for spin polarized calculations. Good luck!')        
        
        if int_in['dc_typ']==0:
            H = add_double_counting(H,spin_names,orb_names,fops,int_in,mo_den,verbose=verbose)
        elif int_in['dc_typ']==1:
            H = add_hartree_fock_DC(H,spin_names,orb_names,fops,int_in,mu_in['target_occ'])
        elif int_in['dc_typ']==2:
            H = add_cbcn_DFT_DC(H,spin_names,orb_names,fops,int_in,mo_den)
            
    # Interaction
    if comp_H['Hint']:
        H = add_interaction(H,1,spin_names,orb_names,fops,int_in,verbose=verbose)

    # Solve atomic problem
    ad = solve_ad(H,spin_names,orb_names,fops,mu_in)


    return ad
#*************************************************************************************

#*************************************************************************************
# Read in atom_diag object
def read_at_diag(ad_file):
    '''
    Save the atom diag object

    Inputs:
    ad_file: label for saved ad file

    Outputs:
    ad: AtomDiag object read in

    '''

    with HDFArchive(ad_file,'r') as ar:
        ad=ar['ad']
        eigensys=ar['eigensys']

        if 'den_mats' in ar:
            den_mats=ar['den_mats']
        else:
            den_mats=[]
        
        
    return ad,eigensys,den_mats

#*************************************************************************************

#*************************************************************************************
# Read in files for interaction
def read_Uijkl(uijkl_file,cmplx,spin_names,orb_names,fops):
    '''

    Inputs:

    Outputs:

    '''
    n_orb=len(orb_names)

    # Test if we have different U for up and down
#    if os.path.isfile(uijkl_file+'.up.up.1'):
#        spin_pol=True
#        uijkl_files=[uijkl_file+'.up.up.1',uijkl_file+'.dn.dn.1',uijkl_file+'.up.dn.1']
#    else:
#        spin_pol=False
#        uijkl_files=[uijkl_file]
        
    uijkls=[]
    # Read in the uijkl files. Skip lines of vijkl. Ordering: upup, dndn, updn
    for uijkl_comp in uijkl_file:
        u_file = open(uijkl_comp,"r")

        if cmplx:
            uijkl=np.zeros((n_orb,n_orb,n_orb,n_orb),dtype=np.complex128)
        else:
            uijkl=np.zeros((n_orb,n_orb,n_orb,n_orb),dtype=np.float64)
            
        for line in u_file:

            # skip lines for vijkl
            if line.split()[0] == "#":
                continue

            i=int(line.split()[0])
            j=int(line.split()[1])
            k=int(line.split()[2])
            l=int(line.split()[3])
            if cmplx:
                uijkl[i-1,j-1,k-1,l-1]=float(line.split()[4])+1j*float(line.split()[5])
            else:
                uijkl[i-1,j-1,k-1,l-1]=float(line.split()[4])
                
        u_file.close()

        uijkls.append(uijkl)

    return uijkls

#*************************************************************************************
# Construct the interaction in the spin polarized case
def make_spin_pol_H_int(uijkls,spin_names,orb_names,fops):
    '''
    '''
    
    n_orbs=len(orb_names)
    
    # Construct the interaction part of the Hamiltonian
    H_int=Operator()
    for s1 in spin_names:
        for s2 in spin_names:

            if s1=='up' and s2=='up':
                ispin=0
            elif s1=='dn' and s2=='dn':
                ispin=1
            else:
                ispin=2
            
            for i in range(n_orbs):
                for j in range(n_orbs):
                    for k in range(n_orbs):
                        for l in range(n_orbs):
                            H_int+=0.5*uijkls[ispin][i,j,k,l]*c_dag(s1,i)*c_dag(s2,j)*c(s2,l)*c(s1,k)


    # TEST
    #H_int_2 = h_int_slater(spin_names, orb_names, uijkls[0], off_diag=True,complex=True)

    #print(H_int-H_int_2)

    #raise
                            

    return H_int

#*************************************************************************************

#*************************************************************************************
# Read in files for spin polarized interaction
def read_tij(wan_file,cmplx,spin_names,orb_names,fops):
    '''

    Inputs:

    Outputs:

    '''

    n_orb=len(orb_names)

    # Make sure tij is list type
    if not isinstance(wan_file, list):
        wan_file=[wan_file]
    
    tijs=[]
    # Read in tij from wannier_hr file
    for wfile in wan_file:
        t_file = open(wfile,"r")
        lines=t_file.readlines()
        tij=[]
        count=0
        max_inter=0
        line_rwts=99999 # Magic number :(
        for line in lines:
            if count==2:
                nrpts=float(line)
                line_rwts = 2+int(np.ceil(nrpts/15.0))

            elif count > line_rwts:
                # Only extract the intrasite elements
                if int(line.split()[0])==0 and	int(line.split()[1])==0 and int(line.split()[2])==0:

                    if cmplx:
                        tij.append(float(line.split()[5])+1j*float(line.split()[6]))
                    else:
                        tij.append(float(line.split()[5]))
                # Find largest intersite hopping
                elif abs(float(line.split()[5])) > max_inter:
                    max_inter=float(line.split()[5])


            count+=1


        print('Largest intersite hop:', max_inter)
        tij=np.reshape(np.array(tij),(n_orb,n_orb))
        t_file.close()

        tijs.append(tij)

    return tijs
    
    
#*************************************************************************************

#*************************************************************************************
# Read in the input file
def run_at_diag(interactive,file_name='iad.in',uijkl_file='',vijkl_file='',wan_file='',dipol_file='',dft_den_file='',ad_file='',wf_files='',out_label='',mo_den=[],spin_names = ['up','dn'],orb_names = [0,1,2,3,4],lat_param=[0,0,0],ml_order=[], \
                comp_H = {'Hkin':False,'Hint':False,'Hdc':False}, \
                int_in = {'int_opt':0,'U_int':0,'J_int':0,'sym':'','uijkl':[],'vijkl':[],'tij':[],'flip':False,'diag_basis':False,'dc_x_wt':0.5,'dc_opt':0,'dc_typ':0, 'eps_eff':1,'cmplx':False}, \
                mu_in = {'tune_occ':False,'mu_init':-8.5,'target_occ':5.0,'const_occ':False,'mu_step':0.5}, \
                prt_occ = False,prt_state = False,prt_energy = False,prt_eigensys = False,prt_mbchar = False,prt_mrchar=False, prt_ad = False,prt_L=False,\
                mb_char_spin = True,n_print = [0,42],verbose=False,prt_dm=False,prt_dipol=False,prt_mbwfs=False):

    '''
    "Main" program, reads input, constructs Hamiltonian, runs
    atom_diag, and prints output.

    Input:
    interactive: Triggers interactive version for jupyter notebooks
    file_name: Input file (if we are reading in from file)

    Parameters:
    uijkl_file=''
    vijkl_file=''
    wan_file=''
    dft_den_file=''
    out_label=''
    mo_den=[]
    spin_names = ['up','dn']
    orb_names = [0,1,2,3,4]

    comp_H = {'Hkin':False,'Hint':False,'Hdc':False}
    int_in = {'int_opt':0,'U_int':0,'J_int':0,'sym':'','uijkl':[],'vijkl':[],'flip':False,'dc_x_wt':0.5,'dc_opt':0,'eps_eff':1}
    mu_in = {'tune_occ':False,'mu_init':-8.5,'target_occ':5.0}

    prt_occ = False
    prt_state = False
    prt_energy = False
    prt_eigensys = False
    prt_mbchar = False
    mb_char_spin = True
    prt_mrchar = False
    n_print = [0,42]

    To call interactively (no defaults):
    run_at_diag(interactive,uijkl_file=uijkl_file,vijkl_file=vijkl_file,wan_file=uijkl_file,dft_den_file=uijkl_file,out_label=uijkl_file,\
                mo_den=mo_den,spin_names=spin_names,orb_names=orb_names,comp_H=comp_H,int_in=int_in,mu_in=mu_in, \
                prt_occ=prt_occ,prt_state=prt_state,prt_energy=prt_energy,prt_eigensys=prt_eigensys,prt_mbchar=prt_mbchar,\
                mb_char_spin=mb_char_spin,n_print=n_print)


    Outputs:
    ????



    '''

    # Read in the input parameter file if not interactive
    if not interactive:

        in_file = open(file_name,"r")
        for line in in_file:

            # Skip blank lines
            if line=='\n' or line.strip()=='':
                continue
            
            # Skip comments
            if str.startswith(line,'#'):
                continue
            elif '#' in line:
                line=line.split('#')[0].strip()
                
            var=line.split('=')[0].strip()
            val=line.split('=')[1].strip()

            # Input files
            if var=='uijkl_file':
                uijkl_file=[]
                for names in str(val).split():
                    uijkl_file.append(names)
            elif var=='vijkl_file':
                vijkl_file=[]
                for names in str(val).split():
                    vijkl_file.append(names)
            elif var=='wan_file':
                wan_file=[]
                for names in str(val).split():
                    wan_file.append(names)
            elif var=='dipol_file':
                dipol_file=[]
                for names in str(val).split():
                    dipol_file.append(names)
            elif var=='dft_den_file':
                dft_den_file=[]
                for names in str(val).split():
                    dft_den_file.append(names)
            elif var=='ad_file':
                ad_file=val
            elif var=='wf_files':
                wf_files=[]
                for files in str(val).split():
                    wf_files.append(files)

            # Operators
            elif var=='spin_names':
                spin_names=[]
                for names in str(val).split():
                    spin_names.append(names)
            elif var=='orb_names':
                orb_names=[]
                for names in str(val).split():
                    orb_names.append(int(names))
            elif var=='n_orbs':
                orb_names=[]
                for i in range(0,int(val)):
                    orb_names.append(i)

            # Parts of the Hamiltonian
            elif var=='Hkin':
                if val=='True' or val=='T' or val=='true':
                    comp_H['Hkin']=True
            elif var=='Hint':
                if val=='True' or val=='T' or val=='true':
                    comp_H['Hint']=True
            elif var=='Hdc':
                if val=='True' or val=='T' or val=='true':
                    comp_H['Hdc']=True
                    
            # Interaction parameters
            elif var=='int_opt':
                int_in['int_opt']=int(val)
            elif var=='dc_opt':
                int_in['dc_opt']=int(val)
            elif var=='dc_typ':
                int_in['dc_typ']=int(val)
            elif var=='U_int':
                int_in['U_int']=float(val)
            elif var=='J_int':
                int_in['J_int']=float(val)
            elif var=='dc_x_wt':
                int_in['dc_x_wt']=float(val)
            elif var=='eps_eff':
                int_in['eps_eff']=float(val)
            elif var=='flip':
                if val=='True' or val=='T' or val=='true':
                    int_in['flip']=True
            elif var=='sym':
                if val=='True' or val=='T' or val=='true':
                    int_in['sym']=True
            elif var=='cmplx':
                if val=='True' or val=='T' or val=='true':
                    int_in['cmplx']=True
                    
            # Change basis
            elif var=='diag_basis':
                if val=='True' or val=='T' or val=='true':
                    int_in['diag_basis']=True

            # Chemical potential, etc
            elif var=='tune_occ':
                if val=='True' or val=='T' or val=='true':
                    mu_in['tune_occ']=True
            elif var=='mu_init':
                mu_in['mu_init']=float(val)
            elif var=='target_occ':
                mu_in['target_occ']=float(val)
            elif var=='const_occ':
                if val=='True' or val=='T' or val=='true':
                    mu_in['const_occ']=True
            elif var=='mu_step':
                mu_in['mu_step']=float(val)

                    
            # Options for printing and output
            elif var=='out_label':
                out_label=val+'_'
            elif var=='prt_occ':
                if val=='True' or val=='T' or val=='true':
                    prt_occ=True
            elif var=='prt_L':
                if val=='True' or val=='T' or val=='true':
                    prt_L=True
            elif var=='prt_energy':
                if val=='True' or val=='T' or val=='true':
                    prt_energy=True
            elif var=='prt_eigensys':
                if val=='True' or val=='T' or val=='true':
                    prt_eigensys=True
            elif var=='prt_state':
                if val=='True' or val=='T' or val=='true':
                    prt_state=True
            elif var=='prt_mrchar':
                if val=='True' or val=='T' or val=='true':
                    prt_mrchar=True
            elif var=='prt_dm':
                if val=='True' or val=='T' or val=='true':
                    prt_dm=True

            elif var=='n_print':
                n_print[0]=int(val.split()[0].strip())
                n_print[1]=int(val.split()[1].strip())

            elif var=='prt_mbchar':
                if val=='True' or val=='T' or val=='true':
                    prt_mbchar=True
            elif var=='mb_char_spin':
                if val=='False' or val=='F' or val=='false':
                    mb_char_spin=False

            elif var=='prt_ad':
                if val=='True' or val=='T' or val=='true':
                    prt_ad=True
                    
            elif var=='verbose':
                if val=='True' or val=='T' or val=='true':
                    verbose=True

            elif var=='ml_order':
                for ml in str(val).split():
                    ml_order.append(int(ml))

            # For dipole calculations
            elif var=='prt_dipol':
                if val=='True' or val=='T' or val=='true':
                    prt_dipol=True
                    
            #elif var=='n_dipol':
            #    n_dipol[0]=int(val.split()[0].strip())
            #    n_dipol[1]=int(val.split()[1].strip())

            elif var=='lat_param':
                lat_param[0]=float(val.split()[0].strip())
                lat_param[1]=float(val.split()[1].strip())
                lat_param[2]=float(val.split()[2].strip())

            # For real-space MB wavefunctions
            elif var=='prt_mbwfs':
                if val=='True' or val=='T' or val=='true':
                    prt_mbwfs=True

                
            # UNKNOWN PARAMETER
            else:
                print('ERROR: Unknown input parameter:',var)
                quit()
                
        in_file.close()

    
    # Setup the operators
    n_orb=len(orb_names)
    fops = [(sn,on) for sn, on in product(spin_names,orb_names)]
    
    # Read in the uijkl files. Skip lines of uijkl
    if uijkl_file:
        int_in['uijkl']=read_Uijkl(uijkl_file,int_in['cmplx'],spin_names,orb_names,fops)

    # Read in the vijkl files. Skip lines of vijkl
    if vijkl_file:
        int_in['vijkl']=read_Uijkl(vijkl_file,int_in['cmplx'],spin_names,orb_names,fops)

    if wan_file:
        int_in['tij']=read_tij(wan_file,int_in['cmplx'],spin_names,orb_names,fops)

    if dft_den_file:
        # Read in DFT densities
        mo_den=[]
        for dfile in dft_den_file:
            mo=[]
            den_file = open(dfile,"r")
            for lines in den_file:
                for num in lines.split():
                    mo.append(float(num))
            mo_den.append(np.reshape(np.array(mo),(n_orb,n_orb)))
            den_file.close()
            
    elif comp_H['Hdc']==True:
        print('ERROR: Must specify DFT den to use DC!')
        quit()

    # Read in ad and eigensys?
    den_mats=[]
    if ad_file:
        print('Reading in ad, eigensys, den_mats')
        ad,eigensys,den_mats=read_at_diag(ad_file)
    else:
        
        # Setup and solve the Hamiltonian
        start = time.time()
        ad=setup_H(spin_names,orb_names,fops,comp_H,int_in,mu_in,verbose,mo_den=mo_den)
        end = time.time()
        print("Time to setup and solve H:",end-start)

    # print out occupations and angular momentum
    if prt_occ:
        print_occ_ang_mom(orb_names,spin_names,fops,ad,ml_order,prt_L)
        
    # Get eigenstates sorted by energies
    if prt_eigensys:
        start = time.time()
        eigensys,den_mats=sort_states(spin_names,orb_names,ad,fops,n_print,out_label,\
                                      prt_mrchar=prt_mrchar,prt_state=prt_state,prt_dm=prt_dm,\
                                      target_mu=mu_in['target_occ'],prt_ad=prt_ad,ml_order=ml_order)
        end = time.time()
        print("Time to get eigenstates sorted by energies:",end-start)
        
    # Print out energies and degeneracies
    if prt_energy:
        start = time.time()
        counts=get_eng_degen_eigensys(ad,eigensys,out_label,prec=10)
        end = time.time()
        print("Time to print out energies and degeneracies:",end-start)

    # Print out characters of degeneracies
    if prt_mbchar:
        start = time.time()
        if int_in['cmplx']:
            print('WARNING: NOT TESTED MB characters for complex inputs at the moment')

        dij_orb_spin=construct_dij(len(orb_names),len(spin_names),'reps.dat',flip=True)
        
        mb_degerate_character(fops,orb_names,spin_names,ad,eigensys,counts,out_label,int_in['cmplx'],dij_orb_spin=dij_orb_spin,state_limit=n_print[1],spin=mb_char_spin)
        end = time.time()
        print("Time to print out characters of degeneracies:",end-start)


    # Print Dipole matrix elements
    if prt_dipol:

        # Check for spin pol
        if len(int_in['tij']) > 1:
            print('WARNING: Dipoles not tested for spin polarization!!!')

        if dipol_file:
            start = time.time()
            print_dipole_mat(n_print,ad,spin_names,orb_names,fops,dipol_file,eigensys,out_label,lat_param,int_in['tij'],diag_basis=int_in['diag_basis'])
            end = time.time()
            print("Time to print dipole matrix elements:",end-start)
        else:
            print("ERROR: No dipole file, cannot calculate rij")
            

    # Print out real-space MB wavefunctions
    if prt_mbwfs:
        start = time.time()
        print_mb_wfs(ad,wf_files,n_print,spin_names,orb_names,fops,eigensys,out_label,den_mats=den_mats)#,den_occ_cmplx,mbwfs_frmt,mb_mesh,out_label,center_in_cell=False,verbose=True):
        end = time.time()
        print("Time to print many-body wavefunctions:",end-start)
    
        
    print('')
    print('Calculation completed.')        
    

    return eigensys,ad,fops
#*************************************************************************************

# End of functions
# ___________________

if __name__ == '__main__':
    eigensys=run_at_diag(False)
