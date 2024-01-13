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

# Coraline's code 
sys.path.append('/Users/cdreyer/Dropbox/CCQ_SB_Research/NV/atom_diag/coulomb_symmetries/code')

import pointgroup as ptgp
import basis as basis
import invariants_pointgroup as inv_ptgp
import invariants_permutations as inv_perm

import slater

import plot_invariants as pltinv
import plot_Umatrices as pltmat



# Symmetry tools
# Cyrus Dreyer, Flatiron CCQ and Stony Brook University 
# 12/15/23

#*************************************************************************************
def get_strong_projection_kk(pg_symbol,orb_names,spin_names,dij,verbose=True,hex_to_rhomb=False,hex_to_cart=False):
    '''
    Use the strong projection operator to project the basis onto irreps
    
    Inputs:
    pg_symbol: Point group symbol (international)
    spin_names: List of spins
    orb_names: List of orbitals
    dij: Single-particle reps of the basis
    verbose: Write stuff out
    hex_to_rhomb: Convert hexagonal setting to rhomahedral in order to compare symmetry operations with Coraline's code
    hex_to_cart: Convert symmetry operations expressed in terms of hexagonal latice vectors to cartesian axes
    
    Outputs:
    
    pg: PointGroup object
    states_proj: Contains the projection onto irreps for each state.
    
    '''

    # Get the point group information
    pg=ptgp.PointGroup(name=pg_symbol)
    rot_mats=[np.array(i,dtype=complex) for i in pg.rot_mat]
    
    # Need this because Coraline used rhombahedral setting for C3v
    if hex_to_rhomb or hex_to_cart:
        sym_ops=get_sym_ops(pg_symbol,hex_to_cart=hex_to_cart,hex_to_rhomb=hex_to_rhomb)

    
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    h=len(dij)
    
    # loop over Wannier states
    states_proj=[]
    for istate in range(n_orb):
        
        proj_Gam_kk={}
        
        for i_irrep,irrep in enumerate(pg.irreps):
            
            proj_kk=np.zeros((int(pg.dims_irreps[i_irrep]),n_orb))
            for sym_el in range(h):

                # Get the single-particle rep for this symmetry element
                if hex_to_rhomb or hex_to_cart:
                    sym=sym_ops[sym_el]
                else:
                    sym=dij[sym_el][0]

                rep=dij[sym_el][1]

                # Find this sym el:
                try:
                    sym_ind=np.where([np.allclose(sym,i) for i in rot_mats])[0][0]
                except:
                    raise Exception('Cannot find symmetry operation!',sym)
                    
                D_R=np.array(pg.get_matrices(irrep=irrep, element=pg.elements[sym_ind])[0])
                
                if D_R.shape[0]==1:
                    D_R_kk=D_R[0]
                else:
                    D_R_kk=np.diag(D_R)
                
                for ik,k in enumerate(D_R_kk):
                    # P_kk^(Gamma_n)(R)
                    proj_kk[ik,:]+=np.array((pg.dims_irreps[i_irrep]/h)*float(k)*rep[istate,:])
                   
                
            # Store for irrep
            proj_Gam_kk_norm=[np.linalg.norm(proj_kk[k,:]) for k in range(pg.dims_irreps[i_irrep])]
            proj_Gam_kk[str(irrep)]=proj_kk#proj_Gam_kk_norm               
        
        states_proj.append(proj_Gam_kk)
        
    if verbose:
        for istate,state in enumerate(states_proj):
            print('State: ',istate)
            for key,val in state.items():
                if np.linalg.norm(np.array(val)) > 1e-10:
                    print(key,np.round(val,4))

            print()


    return pg,states_proj

#*************************************************************************************

#*************************************************************************************
def get_irrep_unitary_kk(n_orb,pg,states_proj,irrep_basis):
    '''
    Get the unitary matrix that converts between the orbital and irrep basis 
    
    Inputs:
    n_orb: Number of orbitals
    pg: PointGroup object
    states_proj: List of irrep projections of states from get_strong_projection_kk
    irrep_basis: List of irreps for the basis, e.g., =['A1','E','E']
    
    Outputs:
    unitary_trans: n_orb x n_orb unitary matrix

    '''
    
    _sp=states_proj.copy()
    
    unitary_trans=[]
    for ibasis,basis in enumerate(irrep_basis): # irrep
        
        # Get the degeneracy
        i_irrep=pg.irrep_names.index(basis)
        dim_irrep=pg.dims_irreps[i_irrep]

        for k in range(dim_irrep): # parnter function number
            
            found=False
            # Search through states
            for istate,state in enumerate(_sp):
        
                try:
                    if np.linalg.norm(state[basis][k]) > 1e-10: # Some row has nonzero values
                        unitary_trans.append(state[basis][k]/np.linalg.norm(state[basis][k]))
                        found=True
                        _sp[istate][basis]=np.delete(_sp[istate][basis],k,axis=0)
                        break
                except IndexError:
                    pass


    unitary_trans=np.vstack(unitary_trans).T

    # Test if it is unitary
    assert np.linalg.norm(np.linalg.inv(unitary_trans) -np.conjugate(unitary_trans).T) < 1.0e-10, 'Matrix is not unitary!'
    return np.vstack(unitary_trans).T
#*************************************************************************************

