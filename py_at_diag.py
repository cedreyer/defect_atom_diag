#!/usr/bin/python

# Triqs stuff
from pytriqs.operators.util.hamiltonians import *
from pytriqs.operators.util import *
from pytriqs.operators import *
from pytriqs.gf import *
from pytriqs.archive import HDFArchive
from pytriqs.atom_diag import *
from itertools import product

# General stuff
import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

# Get functions from my triqs scripts
import sys
sys.path.append('/Users/cdreyer/Dropbox/CCQ_SB_Research/NV/atom_diag') # Where my python files are 
from CED_atom_diag import * # Make sure to comment run_at_diag at the bottom of CED_atom_diag
from op_dipole import *
from ad_sort_print import *

DC_both_full_uijkl,ad,fops,=run_at_diag(False)#interactive,uijkl_file=uijkl_file,wan_file=wan_file,dft_den_file=dft_den_file,out_label=out_label,\
                                        #spin_names=spin_names,orb_names=orb_names,comp_H=comp_H,int_in=int_in,mu_in=mu_in, \
                                        #prt_occ=prt_occ,prt_state=prt_state,prt_energy=prt_energy,prt_eigensys=prt_eigensys,prt_mbchar=prt_mbchar,prt_mrchar=prt_mrchar,\
                                        #mb_char_spin=mb_char_spin,n_print=n_print)

