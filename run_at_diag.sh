#!/bin/bash

module --force purge
module load triqs triqs-tprf

mpirun -np 1 python /mnt/home/cdreyer/apps/defect_atom_diag/def_atom_diag.py
