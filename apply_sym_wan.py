#!/usr/bin/python

import numpy as np
import scipy.interpolate as inter
import scipy.ndimage as sni
import scipy as scp
import math 

# pymatgen symmetry stuff
import pymatgen as pmg
from  pymatgen.symmetry.groups import *

#*************************************************************************************
# Load the cube files
def load_cube_files(wf_file_names):
    '''
    Load a wannier functions in cube file format. Note that the option
    wannier_plot_mode = molecule should be used to center the wannier
    function in the cell (which is very important as you see below).

    Inputs: 
    wf_file_names: Names of the wannier function files
    
    Outputs:
    wanns: list of wannier functions on 3D real-space mesh
    delr: 1D 3 element array containing mesh spacing
    n_mesh: 1D 3 element array containing number of mesh points

    '''
    wanns=[]
    for ifile,filename1 in enumerate(wf_file_names):

        # Open file
        f1=open(filename1)
        lines1=f1.readlines()

        # Make sure headers are the same
        if ifile==0:
            # Get basic info, should be the same for all files
            n_at1=lines1[2].split()[0]
            delr=[float(lines1[3].split()[1]),float(lines1[4].split()[2]),float(lines1[5].split()[3])]
            n_mesh=[int(lines1[3].split()[0]),int(lines1[4].split()[0]),int(lines1[5].split()[0])]

            header=lines1[0:6]

        else:
            _header=lines1[0:6]

            if header != _header:
                print('CUBE files are not compatible!!')
                raise
            

        # Read in wannier functions on the mesh
        wann1=[]
        for grd_lines in lines1[6+int(n_at1):]:
            grd_pts=grd_lines.split()
            for grds in grd_pts:
                wann1.append(float(grds))
            
        wanns.append(np.reshape(wann1,(n_mesh[0],n_mesh[1],n_mesh[2])))

    return wanns, delr, n_mesh, header
#*************************************************************************************

#*************************************************************************************
# Load an xsf files
def load_xsf_files(wf_file_names,convert_lat=False,flip_xz=True):
    '''
    Load a wannier function in xsf file format. wannier_plot_mode =
    molecule is not available in this case

    Inputs: 
    wf_file_names: Names of the wannier function files 
    
    Outputs:
    wanns: List of wannier functions on 3D real-space mesh
    delr: 1D 3 element array containing mesh spacing
    n_mesh: 1D 3 element array containing number of mesh points

    '''

    wanns=[]
    for ifile,filename1 in enumerate(wf_file_names):

        # Open file
        f1=open(filename1)
        lines1=f1.readlines()
        f1.close()

        # Make sure all headers are the same
        if ifile==0:
            
            # Get basic info
            n_at1=int(lines1[14].split()[0])
            n_mesh=[int(lines1[n_at1+20].split()[0]),int(lines1[n_at1+20].split()[1]),int(lines1[n_at1+20].split()[2])]
        
            # Get the header (makes it easier for output). Assume all are the same
            header=lines1[0:n_at1+25]

            # Since its not a cube, need lattice vector info. So far, not
            # tested for nonorthorhombic cells, likely will not work.
            lat_vec=[]
            for i in range(0,3):
                ln=lines1[n_at1+22+i].split()
                for l in ln:
                    lat_vec.append(float(l))

            lat_vec=np.reshape(lat_vec,(3,3))
        
            delr=lat_vec.dot(np.reciprocal(np.array([float(n_mesh[0]),float(n_mesh[1]),float(n_mesh[2])])))


        else:
            # Get the header (makes it easier for output). Assume all are the same
            _header=lines1[0:n_at1+25]

            if header != _header:
                print('XSF files are not compatible!!')
                raise

            
        # Read in wannier functions on the mesh
        wann1=[]
        for grd_lines in lines1[25+int(n_at1):]:
            if 'END_DATAGRID_3D' in grd_lines:
                break
                
            grd_pts=grd_lines.split()
            for grds in grd_pts:
                wann1.append(float(grds))
                    
                
        wann1=np.reshape(wann1,(n_mesh[0],n_mesh[1],n_mesh[2]))
        
        
        # Convert to cartesian coordinates.
        # TODO: NEED THIS FOR NON-CUBIC CELLS
        if convert_lat:
        
            # normalize the lattice vectors using the first vector
            lat_vec_norm=lat_vec/np.linalg.norm(lat_vec[0,:])
            
            # Is this enough padding?
            padding=5
            wann1=np.pad(wann1,(padding,padding),mode='constant')
            n_mesh+=np.array([padding*2,padding*2,padding*2])

            #Move back to the center
            center=0.5*np.array(wann1.shape)
            offset=(center-np.matmul(np.transpose(lat_vec),center))
    
            # Apply symmetry element
            wann1=sni.affine_transform(wann1,np.transpose(lat_vec_norm),offset=offset)
            
            #print(wann1.shape)
            #print(n_mesh)
        

        # Seems like for xsf, the x and z axes are switched
        # compared to cube:
        if flip_xz:
            wann1=wann1.transpose(2,1,0)

        # Add wannier function
        wanns.append(wann1)

    return wanns, delr, n_mesh, header
#*************************************************************************************

#*************************************************************************************
# Load an xsf file, get mesh points in cartesian coordinates
def load_xsf_get_mesh(wf_file_name, mesh_shape='3D'):
    '''
    Get the mesh on which the xsf file is plotted

    Inputs: 
    wf_file_name: Name of the wannier function XSF file 
    mesh_shape: Output shape of mesh, '1D' or '3D'
    
    Outputs:
    mesh_pts: Cartesian coordinates of 

    '''

    # Open file
    f1=open(wf_file_name)
    lines1=f1.readlines()
    f1.close()
            
    # Get basic info
    n_at1=int(lines1[14].split()[0])
    n_mesh=[int(lines1[n_at1+20].split()[0]),int(lines1[n_at1+20].split()[1]),int(lines1[n_at1+20].split()[2])]
         
    # Since its not a cube, need lattice vector info.
    lat_vec=[]
    for i in range(0,3):
        ln=lines1[n_at1+22+i].split()
        for l in ln:
            lat_vec.append(float(l))

    lat_vec=np.reshape(lat_vec,(3,3))

    # spacing in units of lattice constants
    delr=lat_vec.dot(np.reciprocal(np.array([float(n_mesh[0]),float(n_mesh[1]),float(n_mesh[2])])))
       
    # Get the header (makes it easier for output).
    header=lines1[0:n_at1+25]

    # Read in wannier functions on the mesh
    wann1=[]
    for grd_lines in lines1[25+int(n_at1):]:
        if 'END_DATAGRID_3D' in grd_lines:
            break

        grd_pts=grd_lines.split()
        for grds in grd_pts:
            wann1.append(float(grds))
                           
    wann1=np.reshape(wann1,(n_mesh[0],n_mesh[1],n_mesh[2]))        
        
    # Get cartesian grid:

    _x1=np.linspace(0,1,n_mesh[0])
    _y1=np.linspace(0,1,n_mesh[1])
    _z1=np.linspace(0,1,n_mesh[2])            
            
    x1,y1,z1=np.meshgrid(_x1,_y1,_z1)
    positions1 = np.vstack([x1.ravel(), y1.ravel(),z1.ravel()]).T

    if mesh_shape=='1D':
        mesh_pts=np.dot(positions1,lat_vec)
    elif mesh_shape=='3D':
        mesh_pts=np.reshape(np.dot(positions1,lat_vec),(n_mesh[0],n_mesh[1],n_mesh[2],3))
    else:
        print('INCORRECT MESH_SHAPE')
        raise
            
    return wann1, delr, n_mesh, header, mesh_pts
#*************************************************************************************


#*************************************************************************************
# write xsf file
def write_wann(file_name,wann,n_mesh,header):
    '''
    '''

    with open(file_name, 'w') as out_file:

        for line in header:
            out_file.write(line)

        wann_out=np.reshape(wann,(n_mesh[0]*n_mesh[1]*n_mesh[2])) 

        limit=int(np.ceil(float(n_mesh[0]*n_mesh[1]*n_mesh[2])/6.0))

        wann_out=np.pad(wann_out,(0,limit*6-len(wann_out)),mode='empty')
        wann_out=np.reshape(wann_out,(limit,6))

        np.savetxt(out_file,wann_out,fmt='%.8e',delimiter='\t')

        if 'xsf' in file_name:
            out_file.write("END_DATAGRID_3D\n")
            out_file.write("END_BLOCK_DATAGRID_3D\n")
        
    
    return 

    
#*************************************************************************************

#*************************************************************************************
# Normalizer
def normalize(wann,delr):
    '''
    Normalize the wannier function on the grid.

    Input:
    wann: Wannier function on 3D grid in real space
    delr: 1D 3 element array containing mesh spacing

    Output:
    wann1: Normalized wannier function on the grid
    
    '''

    wann1_squared =np.multiply(wann,wann)
    wan_int=int_3d(wann1_squared,delr)
    wann1=wann/np.sqrt(wan_int)
    
    return wann1
#*************************************************************************************

#************************************************************************************* 
# Do 3D numerical integration using the trapezoid rule
def int_3d(wann,delr):
    '''
    3D integration via trapezoid rule.
    
    Input: 
    wann: Wannier function on 3D grid in real space
    delr: 1D 3 element array containing mesh spacing
    
    Output:
    wann_int: 3D integral of wannier function

    '''

    wann_int=np.trapz(np.trapz(np.trapz(wann,dx=delr[0], axis=0),dx=delr[1], axis=0),dx=delr[2], axis=0)

    return wann_int
#*************************************************************************************

#*************************************************************************************
# Extract the rotation matricies from the pymatgen symmetry object
def get_sym_ops(pg_symbol,verbose=True,rhom=False,around_z=True):
    '''
    Get point symmetry operations from pymatgen.
    
    Input: 
    pg_symbol: Hermann-Mauguin notation of point group
    verbose: Write out to the terminal or not
    rhom: Should we convert from hex to rhom?
    around_z: Convert 3-fold axis from 111 to 001

    Outputs:
    point_sym_ops: List of 2D 3x3 arrays for symmetry operations
    '''

    #if rhom:
    #    trans_latt=JonesFaithfulTransformation.from_transformation_string("0.333333333(-a+b+c),0.333333333(2a+b+c),0.333333333(-a-2b+c);0,0,0")
        #trans_axis=JonesFaithfulTransformation.from_transformation_string("2.449489742783178(a-b),4.242640687119286(a+b-2c),0.333333333(a+b+c);0,0,0")
        #trans_axis=JonesFaithfulTransformation.from_transformation_string("1.4142135623730951(a-b),2.449489742783178(a+b-2c),1.7320508075688772(a+b+c);0,0,0")
        #trans_axis=JonesFaithfulTransformation.from_transformation_string("2.449489742783178(a-b),0.33333333(a+b-1.4142135623730951c),0.33333333(a+b+c);0,0,0")
    #    if around_z:
    #        trans_axis_mat=np.reshape(np.array([1.0/math.sqrt(2.0),1.0/math.sqrt(6.0),1.0/math.sqrt(3.0),\
    #                                            -1.0/math.sqrt(2.0),1.0/math.sqrt(6.0),1.0/math.sqrt(3.0),\
    #                                            0.0,-math.sqrt(2.0/3.0),1.0/math.sqrt(3.0)]),(3,3))
        
    # From pymatgen
    point=PointGroup(pg_symbol)
    point_sym_ops=[]
    for sym in point.symmetry_ops:

        # Convert from hex to rhom
        if rhom:
            # Transform to rhomahedral setting
            rot_mat=trans_latt.transform_symmop(sym).rotation_matrix

            if around_z:
                # Transform 3-fold axis to around z axis. Can't figure out how to do this with JFT
                rot_mat=np.matmul(np.matmul(np.linalg.inv(trans_axis_mat),rot_mat),trans_axis_mat)
                
        else:
            rot_mat=sym.rotation_matrix

        point_sym_ops.append(rot_mat)

    if verbose:
        for op in point_sym_ops:
            print (op)
        

    return point_sym_ops
#*************************************************************************************

#*************************************************************************************
# For a given symmetry operation, calculate the representation
def representation_fast(sym_op,wanns,delr,n_mesh,file_type,com_tot=(0.0,0.0,0.0),center_in_cell=False,cheat=True,cht_rnd=0.05):
    '''Calculate the single-particle symmetry representation matrix for
    symmerty operation given by sym_op.

    Inputs:
    sym_op: 2D, 3x3 array for the symmetry operation
    wf_file_names: Names of the wannier function files.
    file_type: cube or xsf
    cheat: To help get pretty looking representation matricies, do
    some strategic rounding :).

    Outputs:
    rep: 2D n_orb x n_orb representation matrix
    '''

    # Initialize rep
    n_wf=len(wanns)
    rep=np.zeros([n_wf,n_wf],dtype=float)

    # Loop through wannier functions
    for i in range(0,n_wf):

        wann_i=wanns[i]
        
        for j in range(0,n_wf):            
            
            wann_j=wanns[j]
                
            # Apply affine transformation around center of cell or
            # total wannier function COM for xsf
            if file_type=='xsf' and not center_in_cell:
                # Around COM
                center=com_tot
            else:
                # Around center of cell
                center=0.5*np.array(wann_j.shape)
            
            offset=-(center-center.dot(sym_op)).dot(np.linalg.inv(sym_op))

            # Apply symmetry element
            wann_j_sym=sni.affine_transform(wann_j,sym_op,offset=offset)

            #TEST: Recenter yet again
            if file_type=='xsf':
                if center_in_cell:
                    wann_j_sym,com_sym=center_wan_func(wann_j_sym,n_mesh,wrt_tot_com=False,com_tot=com_tot,mod_wan=True)
                else:
                    wann_j_sym,com_sym=center_wan_func(wann_j_sym,n_mesh,wrt_tot_com=True,com_tot=com_tot,mod_wan=True)

            sym_rep=int_3d(np.multiply(wann_i,wann_j_sym),delr)
 
            #TEST: Do some cheating to get higher accuracy
            if cheat:
                if abs(sym_rep-1.0) < cht_rnd:
                    sym_rep=1.0
                elif abs(sym_rep+1.0) < cht_rnd:
                    sym_rep=-1.0
                elif abs(sym_rep-0.5) < cht_rnd/10.0:
                    sym_rep =0.5
                elif abs(sym_rep+0.5) < cht_rnd/10.0:
                    sym_rep =-0.5
                elif abs(sym_rep-math.sqrt(3)/2) < cht_rnd/10.0:
                    sym_rep=math.sqrt(3)/2
                elif abs(sym_rep+math.sqrt(3)/2) < cht_rnd/10.0:
                    sym_rep=-math.sqrt(3)/2
                elif abs(sym_rep) < cht_rnd:
                    sym_rep=0.0

            rep[i,j]=sym_rep

    return rep
#*************************************************************************************

#*************************************************************************************
# Generate the symmetry representations of spin 1/2
def get_spin_rep(sym_op,verbose=False):
    '''
    From a given symmerty operation, automatically generate the
    representation for spin 1/2 under that symmetry
    operation. Algorithm is from J. Cano (SB/Flatiron). What we want
    to calculate is exp(-i\nu\hat{n}\sigma/2) where \nu is the angle
    of rotation and \sigma is the normalized sum of Pauli matrices
    corresponding to the axis of the symmetry operation, \hat{n}. All
    of this information can be obtained from the diagonalization of
    the symmetry operation.

    Inputs:
    sym_op: 2D, 3x3 array for the symmetry operation
    verbose: What to write out to the terminal

    Outputs:
    spin_d: 2D 2x2 matrix for spin 1/2 rep

    '''


    # Pauli matricies:
    sigma_x = np.array([[0, 1],[1, 0]])
    sigma_y = np.array([[0, -1j],[1j, 0]])
    sigma_z = np.array([[1, 0],[0, -1]])
    sig=np.array([sigma_x,sigma_y,sigma_z])
    
    if verbose:
        for i in range(0,3):
            print('%3.1f %3.1f %3.1f' % (sym_op[i,0],sym_op[i,1],sym_op[i,2]))

    # Diagonalize sym_op
    evals,evecs=np.linalg.eig(sym_op)
    det=np.linalg.det(sym_op)
        
    # Now lets get n_hat and the angle:
    n_hat=0.0
    ang=0.0
        
    # Proper rotations should have determinant of 1
    if det > 0:
        # Axis is the eigenvector with eigenvalue 1
        for eig in range(0,3):
            if abs(evals[eig]-1.0) < 1.0e-10:
                n_hat=evecs.transpose()[eig]
            else:
                # Angle is obtained from non-unity eigenvalue
                ang=np.angle(evals[eig])
                    
    # If det < 0 either mirror or improper rotation
    else:
        # Axis is the eigenvector with eigenvalue -1
        for eig in range(0,3):
            if abs(evals[eig]+1.0) < 1.0e-10:
                n_hat=evecs.transpose()[eig]
            else:
                # Angle is obtained from non -1 eigenvalue (note the
                # minus sign)
                ang=np.angle(-evals[eig])

    # The spin rep is exp[-i \phi \hat{n}\cdot\sigma/2]
    spin_d=scp.linalg.expm(-1j*0.5*ang*(sig[0]*n_hat[0]+sig[1]*n_hat[1]+sig[2]*n_hat[2]))

    # Write out some stuff
    if verbose:
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=1)
        print("n_hat and angle")
        print('%3.1f %3.1f %3.1f' % (n_hat[0].real,n_hat[1].real,n_hat[2].real))
        print(ang)
        print(spin_d[0,0],spin_d[0,1])
        print(spin_d[1,0],spin_d[1,1])
        print(" ")

    return spin_d
#*************************************************************************************

#*************************************************************************************
# Print represeantations
def print_reps(wan_files,file_type='cube',center_in_cell=False,cht_rnd=0.01):
    '''
    Create file 'reps.dat' that includes: The symmetry operations, the
    orbital representations and characters, and the spin 1/2
    representations and characters.

    Inputs:
    wan_files: File containing names of the wannier function files
    point_grp: Hermann-Mauguin notation of point group
    file_type: Type of wannier function files, either xsf or cube
    cht_rnd: USed to round elements of reps

    Outputs: 
    None.

    Example of output format:
    Rotation matrix
     -1.0 0.0 0.0
      0.0 1.0 0.0
      0.0 0.0 -1.0
    Representation (Orb)
      1.00000000  0.00000000  0.00000000  0.00000000  0.00000000
      0.00000000  1.00000000  0.00000000  0.00000000  0.00000000
      0.00000000  0.00000000 -1.00000000  0.00000000  0.00000000
      0.00000000  0.00000000  0.00000000  1.00000000  0.00000000
      0.00000000  0.00000000  0.00000000  0.00000000 -1.00000000
    Character
      1.000
    Representation (spin)
      0.0000000000+0.0000000000j   -1.0000000000+0.0000000000j
      1.0000000000+0.0000000000j   0.0000000000+0.0000000000j
    Spin character
      0.00000+0.00000j
    '''

    # Get Wannier function file names
    wf_file_names=[]
    with open(wan_files,'r') as f:
        lines=f.readlines()
        point_grp=lines[0].strip()
        for line in lines[1:]:
            wf_file_names.append(line.strip())
    
    # Get symmetry operations from pymatgen
    if point_grp=='3m' or point_grp=='-3m':
        pt_sym_ops=get_sym_ops(point_grp,verbose=False,rhom=True)
    else:
        pt_sym_ops=get_sym_ops(point_grp,verbose=False)

    # Number of orbitals
    n_orb=len(wf_file_names)

    # Read in wannier functions
    if file_type=='cube':
        wanns,delr,n_mesh,header = load_cube_files(wf_file_names)
    elif file_type=='xsf':
        wanns,delr,n_mesh,header = load_xsf_files(wf_file_names)

    
    # For xsf, get total center of mass and align all wannier functions to it
    if file_type=='xsf':
        shift_tot,com_tot=get_wan_cen_mass_shift(wanns,n_mesh)
    
    # Prepare the wannier functions
    for ii in range(0,n_orb):
        # Normalize the wannier functions
        wanns[ii]=normalize(wanns[ii],delr)
        if file_type=='xsf':
            # Center INDIVIDUAL wannier function with respect to TOTAL com
            # This is only legit if wannier functions should have same origin
            if center_in_cell:
                # Step 1: Center wannier functions WRT total COM
                wanns[ii],com=center_wan_func(wanns[ii],n_mesh,wrt_tot_com=True,com_tot=com_tot,mod_wan=True)
                # Step 2: Center the wannier functions in the cell
                wanns[ii],com=center_wan_func(wanns[ii],n_mesh,wrt_tot_com=False,com_tot=com_tot,mod_wan=True)
            else:
                # Just Step 1
                wanns[ii],com=center_wan_func(wanns[ii],n_mesh,wrt_tot_com=True,com_tot=com_tot,mod_wan=True) 

    # Write out info to file:
    f=open("reps.dat","w+")
    f.write("Representations for the symmetry operations:\n")
    f.write("\n")
 
    for sym_op in pt_sym_ops:

        # Write symmetry operation matrix
        f.write("Rotation matrix\n")
        for i in range(0,3):
            #f.write('%3.1f %3.1f %3.1f\n' % (sym_op[i,0],sym_op[i,1],sym_op[i,2]))
            f.write('{:.10f}   {:.10f}   {:.10f}\n'.format(sym_op[i,0],sym_op[i,1],sym_op[i,2]))

        # Calculate orbital part of representation
        if file_type=='xsf':
            if center_in_cell:
                rep=representation_fast(sym_op,wanns,delr,n_mesh,file_type,com_tot=com_tot,center_in_cell=True,cht_rnd=cht_rnd)
            else:
                rep=representation_fast(sym_op,wanns,delr,n_mesh,file_type,com_tot=com_tot,cht_rnd=cht_rnd)
            
        else:
            rep=representation_fast(sym_op,wanns,delr,n_mesh,file_type,cht_rnd=cht_rnd)

        # Get the spin part
        spin_mat=get_spin_rep(sym_op)
                
        # Write out orbital part
        f.write("Representation (Orb)\n")
        for i in range(0,n_orb):
            for j in range(0,n_orb):
                f.write('%12.8f' % (rep[i,j]))
            f.write('\n')

        f.write("Character\n")
        f.write('%5.3f\n' % np.trace(rep))
        print("Orb Character: ",str(np.trace(rep)))

        # Write out spin matrix
        f.write("Representation (spin)\n")
        f.write('{:.10f}   {:.10f}\n'.format(spin_mat[0,0],spin_mat[0,1]))
        f.write('{:.10f}   {:.10f}\n'.format(spin_mat[1,0],spin_mat[1,1]))
        f.write("Spin character\n")
        f.write('{:.5f}'.format(np.trace(spin_mat)))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        print("Spin Character: ",str(np.trace(spin_mat)))

    f.close()

    return
#*************************************************************************************

#*************************************************************************************
# Make new transformed xsf file
def transform_xsf(in_wan_file,sym_op,shift,out_wan_file,center_in_cell=True):
    '''
    Useful for testing. Takes in an xsf file, then in THIS order:
    1. Centers the wannier function (optional), 2. Translates the
    wannier function, 3. Applies sym_op. Outputs the modified xsf file
    for plotting.

    Inputs:    
    in_wan_file: String, name of input xsf file
    sym_op: 2D, 3x3 array for the symmetry operation
    shift: 1D 3 element array, amount to shift (in terms of grid points)
    out_wan_file: String, output file name
    center: Whether or not to center wannier function in cell

    Outputs:
    None.

    '''

    # This code will be the same as load_xsf_file, but need to to it
    # again to write to new file
    f1=open(in_wan_file)
    lines1=f1.readlines()
    f1.close()

    # Get basic info, should be the same for both files
    n_at1=int(lines1[14].split()[0])
    n_mesh=[int(lines1[n_at1+20].split()[0]),int(lines1[n_at1+20].split()[1]),int(lines1[n_at1+20].split()[2])]

    lat_vec=[]
    for i in range(0,3):
        ln=lines1[n_at1+22+i].split()
        for l in ln:
            lat_vec.append(float(l))

    lat_vec=np.reshape(lat_vec,(3,3))
    delr=lat_vec.dot(np.reciprocal(np.array([float(n_mesh[0]),float(n_mesh[1]),float(n_mesh[2])])))    

    # Read in wannier functions on the mesh
    wann1=[]
    for grd_lines in lines1[25+int(n_at1):]:
        if 'END_DATAGRID_3D' in grd_lines:
            break

        grd_pts=grd_lines.split()
        for grds in grd_pts:
            wann1.append(float(grds))
                    
    wann1=np.reshape(wann1,(n_mesh[0],n_mesh[1],n_mesh[2]))
    
    # First center wannier function
    if center_in_cell:
        # TEST: Center the wannier function in the cell
        wann1,com=center_wan_func(wann1,n_mesh)
    else:
        # Do not center WF
        wann1,com=center_wan_func(wann1,n_mesh,mod_wan=False)

    # Then apply shift:
    trans_mat=np.identity(4)
    trans_mat[0][3]=shift[0]
    trans_mat[1][3]=shift[1]
    trans_mat[2][3]=shift[2]    
    wann1=sni.affine_transform(wann1,trans_mat)

    # Now apply sym operation:
    
    # Use COM for offset to perform symmetry operation
    if center_in_cell:
        center=0.5*np.array(wann1.shape)
    else:
        center=com
    
    # Apply symmetry operation
    offset=-(center-center.dot(sym_op)).dot(np.linalg.inv(sym_op))    
    wann1=sni.affine_transform(wann1,sym_op,offset=offset)

    # To prepare for output, we switch indicies back
    # Write out the atom information to the new file
    with open(out_wan_file,'w') as outf:
        for line in range(0,n_at1+25):
            outf.write(lines1[line])
            
        wann1_out=np.reshape(wann1,(n_mesh[0]*n_mesh[1]*n_mesh[2]))
        count=0
        limit=np.ceil(float(n_mesh[0]*n_mesh[1]*n_mesh[2])/6.0)

        for ii in range(0,int(limit)):
            for jj in range(0,6):
                if count+jj < n_mesh[0]*n_mesh[1]*n_mesh[2]:
                    outf.write('{:20.12e} '.format(wann1_out[count+jj]))
                else:
                    break
            outf.write("\n")

            count +=6

        outf.write("END_DATAGRID_3D\n")
        outf.write("END_BLOCK_DATAGRID_3D\n")

    return
#*************************************************************************************

#*************************************************************************************    
# Get offset to shift COM of wannier centers to the middle
def get_wan_cen_mass_shift(wanns,n_mesh):
    '''Gets the center of mass of the wannier function (squared), for
    centering purposes.
    
    Inputs:
    wanns: List of wannier functions on 3D real-space mesh 

    Outputs: 
    shift: 1D 3 element array indicating the shift required to
    put the COM of the sum of wannier functions in the middle of the
    cell.
    com: Center of mass of sum of wannier functions
    '''
    
    # Load/sum all of the wannier functions
    wann_sum=0.0
    for wann1 in wanns:
        wann1=np.multiply(wann1,wann1)
        wann_sum=np.add(wann1,wann_sum)
        
    # Find center of mass for sum of wannier function
    com=np.array(scp.ndimage.center_of_mass(wann_sum))
    shift=com-np.array([float(n_mesh[0])/2.0,float(n_mesh[1])/2.0,float(n_mesh[2])/2.0])
     

    return shift,com
#*************************************************************************************

#*************************************************************************************
# Center the wannier function in the cell
def center_wan_func(wann1,n_mesh,wrt_tot_com=False,com_tot=0.0,mod_wan=True):
    '''
    Center a wannier function in the cell. We can either center it in
    terms of its COM, or COM of the sum of wannier functions.

    Inputs:
    wann1: SINGLE wannier function on 3D real-space mesh
    n_mesh: 1D 3 element array containing number of mesh points  
    tot_com: True means center WRT TOTAL center of mass
    mod_wan: True centers the wannier function, false just gets COM
    wf_file_names: If doing the total COM, need all wannier functions
    file_type: For now just xsf
    
    Outputs:
    wann1: Centered wannier function on 3D real-space mesh
    com: Center of mass of wannier function 
    '''

    # Get center of mass and shift for individual wannier function
    wann1_sq=np.multiply(wann1,wann1)
    com=np.array(scp.ndimage.center_of_mass(wann1_sq))
    shift=com-np.array([float(n_mesh[0])/2.0,float(n_mesh[1])/2.0,float(n_mesh[2])/2.0])

    # Center Wannier function
    if mod_wan:
        # For the case where we center WRT the total COM
        if wrt_tot_com:
            shift=com-com_tot
            
        trans_mat=np.identity(4)
        trans_mat[0][3]=shift[0]
        trans_mat[1][3]=shift[1]
        trans_mat[2][3]=shift[2]
        
        wann1=sni.affine_transform(wann1,trans_mat)


    return wann1,com
#*************************************************************************************            
 
# END OF FUNCTIONS
# ---------

if __name__ == '__main__':
    print_reps('wan_files.dat',file_type='cube')
    


