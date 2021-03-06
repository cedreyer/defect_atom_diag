#!/usr/bin/python

from pytriqs.operators.util.hamiltonians import *
from pytriqs.operators.util import *
from pytriqs.operators import *
from pytriqs.gf import *
from pytriqs.archive import HDFArchive
from pytriqs.atom_diag import *
from itertools import product

import numpy as np
import sympy as sp
import math

from ad_sort_print import *
from op_dipole import *

# Main script to run atom_diag calculations
# Cyrus Dreyer, Stony Brook University, Flatiorn CCQ 
# 09/11/19 

#*************************************************************************************
# Tune chemical, generate ad and dm
def add_chem_pot(H,spin_names,orb_names,init_mu,fops,tune_opt,step=0.5,target_occ=5.0):
    '''
    Solve the atomic problem and tune the chemical potential to the
    target filling. Very inefficient way of doing it but works for now.

    Inputs:
    H: Hamiltonian operator
    spin_names: List of spins
    orb_names: List of orbitals
    init_mu: Initial chemical potential
    fops: Many-body operators
    tune_opt: Logical, to tune chemical potential
    step: Steps to increment chemical potential
    targer_occ: Occupation we are shooting for

    Outputs:
    H: Hamiltonian with chemical potential term added
    ad: Solution to the atomic problem
    dm: Density matrix
    N: Number operator
    '''


    # Setup the particle number operator
    N = Operator() # Number of particles
    for s in spin_names:
        for o in orb_names:        
            N += n(s,o)

    filling=0
    mu=init_mu

    # Tune the chemical potential to get desired occupancy, solve the
    # atomic problem
    if tune_opt:
        while True:
            # Add chemical potential
            H += mu * N
            
            # Split the Hilbert space automatically
            ad = AtomDiagComplex(H, fops)
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
    
    else: # No tuning
        H += mu * N
        ad = AtomDiagComplex(H, fops)
        beta = 1e10
        dm = atomic_density_matrix(ad, beta)
        filling = trace_rho_op(dm, N, ad)
        
    print("Chemical potential: ",mu)
    print('# of e-: ', filling)
    return H,ad,dm,N
#*************************************************************************************

#*************************************************************************************
# Add the interaction Hamiltonian
def add_interaction(H,tij,n_sites,spin_names,orb_names,fops,int_in):
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
    
    Outputs:
    H: Hamiltonian with interaction term added

    '''

    # Setup
    n_orb=len(orb_names)
    H_int=Operator()
    int_opt=int_in['int_opt']
    
    # Need to flip indicies of uijkl coming from VASP
    if int_in['flip']:
        uijkl=flip_u_index(n_orb,int_in['uijkl'])
        print('uijkl -> uikjl')
    else:
        uijkl=int_in['uijkl']
        print('uijkl -> uijkl')


    #TEST: BASIS CHANGE TO "BAND"
    if int_in['diag_basis']:
        tij,uijkl=wan_diag_basis(n_orb,tij,H_elmt='both',uijkl=uijkl)

        # Construct U matrix for d orbitals
        if int_opt == 6:
            print('SLATER U!!!!')
            Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False)
            # Basis from wannierization is z2, xz,yz,x2-y2,xy
            # Triqs = xy,yz,z^2,xz,x^2-y^2
            rot_def_to_w90 = np.array([[0, 0, 0, 0, 1],
                                       [0, 0, 1, 0, 0],
                                       [1, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0]])

            uijkl=U_matrix(2, radial_integrals=None, U_int=Uavg, J_hund=1, basis='spherical', T=rot_def_to_w90.T)
        
        
    # Choices to average the uijkl matrix
    if int_opt == 1: # Use average U, U', and J
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False) 
    elif int_opt == 2: # Use two index Uij and Ji
        uij,jij,uijkl=make_two_ind_U(n_orb,uijkl,verbose=False)    
    # Only diagonal
    elif int_opt == 3:
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,False,False])
    # Only density-density
    elif int_opt == 4:
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,True,False])
    # U and J
    elif int_opt == 5:
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,False,True])
    # Construct U matrix for d orbitals
    elif int_opt == 6:
        print('SLATER U!!!!')
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False)
        # Basis from wannierization is z2, xz,yz,x2-y2,xy
        # Triqs = xy,yz,z^2,xz,x^2-y^2
        rot_def_to_w90 = np.array([[0, 0, 0, 0, 1],
                                   [0, 0, 1, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0]])

        uijkl=U_matrix(2, radial_integrals=None, U_int=Uavg, J_hund=1, basis='spherical', T=rot_def_to_w90.T)

        

        
    # Check the symmetry of the U matrix
    if int_in['sym']:
        check_sym_u_mat(uijkl,n_orb)         
            
    # Add the interactions
    H += h_int_slater(spin_names, orb_names, uijkl, off_diag=True,complex=True)

    return H
#*************************************************************************************

#*************************************************************************************
# Convert basis of uijkl to wan. diagonal
def wan_diag_basis(n_orb,tij,H_elmt='both',uijkl=[]):
    '''
    Convert Hamiltonian to basis where tij is diagonal 
    
    Inputs: 
    H_elmt: 'tij': Just convert hoppings, 'both': convert both
    tij: Hopping matrix, np array
    uijkl: Interaction matrix, np array
    
    Outputs:
    tij_conv: diagonalized tij
    uijkl_conv: uijkl matrix in diagonal basis

    '''

    evals,evecs=np.linalg.eig(tij)
    print('wannier eval order:',evals)
    tij_conv=np.multiply(np.identity(n_orb),evals)
    #print(tij_conv)
    #quit()
    # Convert to wannier basis 
    #wan_den=np.dot(np.dot(np.linalg.inv(evecs.T),mo_den),np.linalg.inv(evecs))

    uijkl_conv=uijkl
    if H_elmt == 'both':
        basis_trans=np.linalg.inv(evecs)#evecs.T????
        uijkl_conv=transform_U_matrix(uijkl,basis_trans)
        #print(uijkl)
        #print(uijkl_t)
        #quit()
    

    return tij_conv, uijkl_conv
    
#*************************************************************************************
 
#*************************************************************************************
# Check if the  U matrix obeys symmetry of single particle reps
def check_sym_u_mat(uijkl,n_orb):
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
    dij=construct_dij(n_orb,"reps.dat")
    i_rep=0
    print("Check if uijkl obeys sym of reps:")
    for rep in dij:

        # Transform U matrix
        Tij=rep[1]
        uijkl_t=transform_U_matrix(uijkl,Tij)

        print('%s %f %s %f' % ("For rep: ",i_rep," max val: ",np.amax(uijkl-uijkl_t)))
        i_rep+=1

    return
#*************************************************************************************

#*************************************************************************************
# Check if the hopping obeys symmetry 
def check_sym_t_mat(tij,n_orb):
    '''
    Checks to see if tij matrix is consistent with symmetry of single
    particle reps
    
    Inputs:
    tij: n_orb x n_orb matrix of tij values
    n_orb: Number of orbitals

    Outputs:
    None
    '''

    # Get reps from reps.dat
    dij=construct_dij(n_orb,"reps.dat")
    i_rep=0
    print("Check if tij commutes with reps:")
    for rep in dij:
        print('%s %f %s %f' % ("For rep: ",i_rep," max val: ",\
                               np.amax(np.matmul(rep[1],tij)-np.matmul(tij,rep[1]))))
        i_rep+=1

    return
#*************************************************************************************

#*************************************************************************************
# Add hoppings
def add_hopping(H,spin_names,orb_names,tij,sym,diag_basis):
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

    # For testing to check the symmetry of the tij matrix
    n_orb=len(orb_names)
    if sym: # Check the sym verus provided reps
        check_sym_t_mat(tij,n_orb)         

    # TEST: use diagonal basis
    if diag_basis:
        tij,uijkl=wan_diag_basis(n_orb,tij,H_elmt='tij')

        
    # Make noniteracting operator
    H_kin=Operator()
    for s in spin_names:
        for o1 in orb_names:
            for o2 in orb_names:
                H_kin += 0.5*tij[int(o1),int(o2)] * (c_dag(s,o1) * c(s,o2)+ c_dag(s,o2) * c(s,o1))
    H += H_kin
    return H
#*************************************************************************************

#*************************************************************************************
# Fix the fact that U_ijkl out of vasp has indicies switched
def flip_u_index(n_orb,uijkl):
    '''
    uijkl -> uikjl for UIJKL read in from VASP
    
    Input:
    n_orb: Number of orbitals
    uijkl: n_orb x n_orb Uijkl matrix

    Output:
    uijkl_new: Uijkl with indicies switched
    '''

    uijkl_new=np.zeros((n_orb,n_orb,n_orb,n_orb),dtype=np.float)
    for i in range(0,n_orb):
        for j in range(0,n_orb):
            for k in range(0,n_orb):
                for l in range(0,n_orb):
                    uijkl_new[i,j,k,l]=uijkl[i,k,j,l]
                    
    return uijkl_new
#*************************************************************************************


#*************************************************************************************
# Double counting
def add_double_counting(H,spin_names,orb_names,fops,uijkl,tij,mo_den,dc_x_wt,flip=False,dc_opt=0):
    '''
    Subtract double counting correction to the Hamiltonian. Very much
    in the testing phase.

    Inputs:
    H: Hamiltonian operator
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    uijkl: n_orb x n_orb Uijkl matrix
    tij: Hopping matrix elements
    mo_den: Occupation in band basis
    dc_x_wt: Weight on exchange term in HF
    DC_opt: 0: full uijkl, 1: orbital average, 2: Two index
    flip: Flip uijkl here?
    
    Outputs:
    H: Hamiltonian with DC subtracted
    '''

    n_orb=len(orb_names)

    if flip:
        uijkl=flip_u_index(n_orb,uijkl)

    
    if dc_opt < 0: # Use averaged density
        avg_den=np.trace(mo_den)/n_orb
        wan_den=np.identity(n_orb)*avg_den
        dc_opt=abs(dc_opt)

    else: # Convert DFT density to wannier basis
        # Find matrix that converts between wannier and MO reps
        evals,evecs=np.linalg.eig(tij)

        print("Ordering of states",evals)
        #quit()

        # Convert to wannier basis 
        wan_den=np.dot(np.dot(np.linalg.inv(evecs.T),mo_den),np.linalg.inv(evecs))

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
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,True,False])
    # U and J
    elif dc_opt == 5:
        Uavg,Uprime,Javg,uijkl=avg_U(n_orb,uijkl,verbose=False,U_elem=[True,False,True])

        
    H_dc=Operator()
    for s in spin_names:
        for i in range(0,n_orb):
            for j in range(0,n_orb):
                fock=0
                for k in range(0,n_orb):
                    for l in range(0,n_orb):
                            fock += 1.0*(uijkl[i,l,j,k] - dc_x_wt*uijkl[i,l,k,j] ) * wan_den[k,l] # From Szabo and Ostland
                            

                H_dc += -fock * c_dag(s,str(i)) * c(s,str(j))
                
    # From Karsten and Flensberg Eq. 4.22 and 4.23
#    for s in spin_names:  
#        for i in range(0,n_orb):
#            for j in range(0,n_orb):
#                for k in range(0,n_orb):
#                    for l in range(0,n_orb):
#                        # First three terms: Hartree, last three: Fock
#                        H_dc += -0.5*uijkl[i,j,k,l]*( \
#                                                      wan_den[j,l]*c_dag(s,str(i))*c(s,str(k)) \
#                                                      + wan_den[i,k]*c_dag(s,str(j))*c(s,str(l)) \
#                                                      - wan_den[i,l]*c_dag(s,str(j))*c(s,str(k)) \
#                                                      - wan_den[j,k]*c_dag(s,str(i))*c(s,str(l)) )

    H += H_dc

    #print(H_dc)
    #quit()

    return H
#*************************************************************************************

#*************************************************************************************
# Average U values
def make_two_ind_U(n_orb,uijkl,verbose=False,U_elem=[True,True]):

    uij=np.zeros((n_orb,n_orb))
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
                    elif ii==ll and kk==jj and U_elem[1]: # Hunds: Uij,ji
                        if verbose:
                            print(ii,jj,kk,ll,uijkl[ii,jj,kk,ll], "Added to jij")
                        jij[ii,jj]=uijkl[ii,jj,kk,ll]                        
                    else:
                        uijkl_avg[ii,jj,kk,ll]=0.0
                        

#    # Now construct a full uijkl matrix with averaged values
#    uijkl_avg=np.zeros((n_orb,n_orb,n_orb,n_orb))
#    for ii in range(0,n_orb):
#        for jj in range(0,n_orb):
#            for kk in range(0,n_orb):
#                for ll in range(0,n_orb):
#                    if ii==kk and jj==ll: # Indicies should be flipped already
#                        uijkl_avg[ii,jj,kk,ll]=uij[ii,jj]
#                    elif ii==ll and kk==jj:
#                        uijkl_avg[ii,jj,kk,ll]=jij[ii,kk]

    # For testing
    if verbose:
        print(" ")
        for ii in range(0,n_orb):
            for jj in range(0,n_orb):
                for kk in range(0,n_orb):
                    for ll in range(0,n_orb):
                        print(ii,jj,kk,ll,uijkl_avg[ii,jj,kk,ll])

    
    return uij,jij,uijkl_avg
#*************************************************************************************

#*************************************************************************************
# Average U values
def avg_U(n_orb,uijkl,verbose=False,U_elem=[True,True,True],triqs_U=False):
    
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
def setup_H(spin_names,orb_names,fops,comp_H,int_in,mu_in,tij=[],mo_den=[]):
    '''
    Add terms to the Hamiltonian depending on the input file.
    
    Inputs:
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    comp_H: List of components of the Hamiltonian
    int_in: Input parameters
    tij: Hopping matrix elements
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
        H = add_hopping(H,spin_names,orb_names,tij,int_in['sym'],int_in['diag_basis'])

    # Double counting
    if comp_H['Hdc']: 
        H = add_double_counting(H,spin_names,orb_names,\
                                fops,int_in['uijkl'],tij,mo_den,int_in['dc_x_wt'],flip=int_in['flip'],dc_opt=int_in['dc_opt'])

    # Interaction
    if comp_H['Hint']:
        H = add_interaction(H,tij,1,spin_names,orb_names,fops,int_in)

            #H,1,int_in['int_op'],spin_names,orb_names,fops,uijkl=int_in['uijkl'],U=int_in['U_int'],J=int_in['J_int'],sym=int_in['sym']) 

    # Chemical potential
    H,ad,dm,N = add_chem_pot(H,spin_names,orb_names,mu_in['mu_int'],fops,mu_in['tune_opt'],target_occ=mu_in['target_occ']) 


    return H,ad,dm,N
#*************************************************************************************

#*************************************************************************************
# Read in the input file
def run_at_diag(interactive,file_name='iad.in',uijkl_file='',wan_file='',dft_den_file='',out_label='',mo_den=[],spin_names = ['up','dn'],orb_names = [0,1,2,3,4], \
                comp_H = {'Hkin':False,'Hint':False,'Hdc':False}, \
                int_in = {'int_opt':0,'U_int':0,'J_int':0,'sym':'','uijkl':[],'flip':False,'diag_basis':False,'dc_x_wt':0.5,'dc_opt':0}, \
                mu_in = {'tune_opt':False,'mu_int':-8.5,'target_occ':5.0}, \
                prt_occ = False,prt_state = False,prt_energy = False,prt_eigensys = False,prt_mbchar = False,prt_mrchar=False,\
                mb_char_spin = True,n_print = [0,42]):

    '''
    "Main" program, reads input, constructs Hamiltonian, runs
    atom_diag, and prints output.
    
    Input:
    interactive: Triggers interactive version for jupyter notebooks
    file_name: Input file (if we are reading in from file)

    Parameters:
    uijkl_file=''
    wan_file=''
    dft_den_file=''
    out_label=''
    mo_den=[]
    spin_names = ['up','dn']
    orb_names = [0,1,2,3,4]

    comp_H = {'Hkin':False,'Hint':False,'Hdc':False}
    int_in = {'int_opt':0,'U_int':0,'J_int':0,'sym':'','uijkl':[],'flip':False,'dc_x_wt':0.5,'dc_opt':0}
    mu_in = {'tune_opt':False,'mu_int':-8.5,'target_occ':5.0}

    prt_occ = False
    prt_state = False
    prt_energy = False
    prt_eigensys = False
    prt_mbchar = False
    mb_char_spin = True
    prt_mrchar = False
    n_print = [0,42]

    To call interactively (no defaults):
    run_at_diag(interactive,uijkl_file=uijkl_file,wan_file=uijkl_file,dft_den_file=uijkl_file,out_label=uijkl_file,\
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

            var=line.split()[0]        

            # Skip comments
            if str.startswith(var,'#'):
                continue
        
            val=line.split()[2]
        
            # Input files
            if var=='uijkl_file':
                uijkl_file=val
            elif var=='wan_file':
                wan_file=val
            elif var=='dft_den_file':
                dft_den_file=val

            # Operators
            elif var=='spin_names':
                spin_names=[]
                for names in line.split()[2:]:
                    spin_names.append(names)
            elif var=='orb_names':
                orb_names=[]
                for names in line.split()[2:]:
                    orb_names.append(names)
            elif var=='n_orbs':
                orb_names=[]
                for i in range(0,val):
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
                
            # Interaction paramters
            elif var=='int_opt':
                int_in['int_opt']=int(val)
            elif var=='dc_opt':
                int_in['dc_opt']=int(val)
            elif var=='U_int':
                int_in['U_int']=float(val)
            elif var=='J_int':
                int_in['J_int']=float(val)
            elif var=='dc_x_wt':
                int_in['dc_x_wt']=float(val)
            elif var=='flip':
                if val=='True' or val=='T' or val=='true':
                    int_in['flip']=True
            elif var=='sym':
                if val=='True' or val=='T' or val=='true':
                    int_in['sym']=True

            # TEST: change basis
            elif var=='diag_basis':
                if val=='True' or val=='T' or val=='true':
                    int_in['diag_basis']=True

            # Chemical potential, etc
            elif var=='tune_opt':
                if val=='True' or val=='T' or val=='true':
                    mu_in['tune_opt']=True
            elif var=='mu_int':
                mu_in['mu_int']=float(val)
            elif var=='target_occ':
                mu_in['target_occ']=float(val)
            
            # Options for printing and output
            elif var=='out_label':
                out_label=val+'_'
            elif var=='prt_occ':
                if val=='True' or val=='T' or val=='true':
                    prt_occ=True
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

            elif var=='n_print':
                n_print[0]=int(val)
                n_print[1]=int(line.split()[3])

            elif var=='prt_mbchar':
                if val=='True' or val=='T' or val=='true':
                    prt_mbchar=True
            elif var=='mb_char_spin':
                if val=='False' or val=='F' or val=='false':
                    mb_char_spin=False

        in_file.close()

    # Setup the operators
    n_orb=len(orb_names)
    fops = [(sn,on) for sn, on in product(spin_names,orb_names)]
    
    # Read in the uijkl files. Skip lines of vijkl
    if uijkl_file:
        u_file = open(uijkl_file,"r")
        
        uijkl=np.zeros((n_orb,n_orb,n_orb,n_orb),dtype=np.float)
        for line in u_file:
            
            # skip lines for vijkl
            if line.split()[0] == "#":
                continue

            i=int(line.split()[0])
            j=int(line.split()[1])
            k=int(line.split()[2])
            l=int(line.split()[3])
            uijkl[i-1,j-1,k-1,l-1]=float(line.split()[4])

        u_file.close()        

        int_in['uijkl']=uijkl
                
    if wan_file:
        # Read in tij from wannier_hr file
        t_file = open(wan_file,"r")
        lines=t_file.readlines()
        tij=[]
        count=0
        max_inter=0
        line_rwts=99999
        for line in lines:
            if count==2:
                nrpts=float(line)
                line_rwts = 2+int(np.ceil(nrpts/15.0))
        
            elif count > line_rwts:
                # Only extract the intrasite elements
                if int(line.split()[0])==0 and	int(line.split()[1])==0 and int(line.split()[2])==0:
                    tij.append(float(line.split()[5]))

                # Find largest intersite hopping
                elif abs(float(line.split()[5])) > max_inter:
                    max_inter=float(line.split()[5])
                    

            count+=1


        print('Largest intersite hop:', max_inter)
        tij=np.reshape(np.array(tij),(n_orb,n_orb))
        t_file.close()
        
    if dft_den_file:
        # Read in DFT densities
        mo_den=[]
        den_file = open(dft_den_file,"r")
        for lines in den_file:
            for num in lines.split():
                mo_den.append(float(num))
        mo_den=np.reshape(np.array(mo_den),(n_orb,n_orb))
        den_file.close()

    # Setup and solve the Hamiltonian
    H,ad,dm,N=setup_H(spin_names,orb_names,fops,comp_H,int_in,mu_in,tij=tij,mo_den=mo_den)

    #TEST
    #print(H)

    # print out occupations and angular momentum 
    if prt_occ:
        print_occ_ang_mom(orb_names,spin_names,ad,dm)

    # Get eigenstates sorted by energies
    if prt_eigensys:
        eigensys=sort_states(spin_names,orb_names,ad,fops,n_print,out_label,prt_mrchar=prt_mrchar,prt_state=prt_state,target_mu=mu_in['target_occ'])

    # Print out energies and degeneracies
    if prt_energy:
        counts=get_eng_degen_eigensys(ad,eigensys,out_label,prec=10)
    
    # Print out characters of degeneracies
    if prt_mbchar:
        mb_degerate_character("reps.dat",fops,orb_names,spin_names,ad,eigensys,counts,out_label,state_limit=n_print[1],spin=mb_char_spin)



    # TEST: Check multi-reference
    check_multi_ref(ad,spin_names,orb_names,fops,eigensys,n_states=10)

    # TEST: Dipole matrix elements 
#    for ii in range(0,12):
#        for jj in range(ii,12):
#            dipole_op(ad,spin_names,orb_names,fops,"wannier90_r.dat",ii,jj,eigensys)




    return eigensys,ad,fops
#*************************************************************************************

# End of functions
# ___________________

#eigensys=run_at_diag(False)
