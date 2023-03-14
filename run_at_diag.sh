#!/bin/bash

# For cluster:
module --force purge
module load triqs triqs-tprf

source /mnt/home/cdreyer/apps/venvs/at_diag/bin/activate

mpirun -np 1 python /mnt/home/cdreyer/apps/defect_atom_diag/def_atom_diag.py

#python /Users/cdreyer/Dropbox/CCQ_SB_Research/NV/atom_diag/defect_atom_diag/def_atom_diag.py
