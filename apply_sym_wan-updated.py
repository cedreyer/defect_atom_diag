#!/usr/bin/pytAAhonA

import numpy as np
import scipy.interpolate as inter
from scipy.interpolate import interpn
import scipy.ndimage as sni
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation
from scipy import integrate
import scipy as scp
import math 
import time

# pymatgen symmetry stuff
from  pymatgen.symmetry.groups import PointGroup
from settings_CED import JonesFaithfulTransformation_CED



#*************************************************************************************
# Load an xsf files
def load_xsf_files(wf_file_names, path="./", flip_xz=True, convert_lat=False, npad=0, center=np.array([0.5, 0.5, 0.5])):
    """
    Read XSF scalar grids without resampling them into a different
    coordinate frame.
    Returns
    -------
    wanns : list of arrays, each shape (nx, ny, nz)
    grid_steps : (3, 3) array
        One physical grid-step vector per array axis, as rows.
    n_mesh : (3,) array
        Number of grid points along each array axis.
    header : list of strings
        Header of the first input file.
    Important:
        The second return value is now a full matrix, not a 3-element
        vector of scalar spacings.
    """
    wanns = []
    meshes = []
    origins = []
    step_matrices = []
    headers = []
    if convert_lat:
        raise ValueError("This loader preserves the original XSF grid basis. Use convert_lat=False and transform symmetry operations into the grid basis instead.")
    if npad != 0:
        raise NotImplementedError("Padding is disabled here to avoid changing the grid origin without explicitly updating the symmetry-center coordinates.")
    for filename in wf_file_names:
        with open(path + filename, "r") as f:
            lines = f.readlines()
        starts = [i for i, line in enumerate(lines) if "BEGIN_DATAGRID_3D_UNKNOWN" in line]
        if len(starts) != 1:
            raise ValueError(f"{filename}: expected one 3D datagrid block, found {len(starts)}.")
        start = starts[0]
        n_mesh = np.array([int(v) for v in lines[start + 1].split()], dtype=int)
        if n_mesh.shape != (3,) or np.any(n_mesh < 2):
            raise ValueError(f"{filename}: invalid mesh dimensions.")
        origin = np.array([float(v) for v in lines[start + 2].split()], dtype=float)
        # The three XSF spanning vectors, as rows.
        spans = np.array([[float(v) for v in lines[start + 3 + i].split()] for i in range(3)])
        if origin.shape != (3,) or spans.shape != (3, 3):
            raise ValueError(f"{filename}: invalid grid geometry.")
        # Standard XSF convention: the spanning vectors extend from the first grid point to the last grid point.
        grid_steps = spans / (n_mesh - 1)[:, None]
        if not np.all(np.isfinite(grid_steps)):
            raise ValueError(f"{filename}: non-finite grid geometry.")
        if abs(np.linalg.det(grid_steps)) < 1e-14:
            raise ValueError(f"{filename}: singular grid-step matrix.")
        values = []
        for line in lines[start + 6:]:
            if "END_DATAGRID_3D" in line:
                break
            values.extend(float(v) for v in line.split())
        expected_size = int(np.prod(n_mesh))
        if len(values) != expected_size:
            raise ValueError(f"{filename}: expected {expected_size} data values, read {len(values)}.")
        # XSF data are conventionally written with x varying fastest, convert to array axes (x, y, z).
        wann = np.asarray(values).reshape((n_mesh[2], n_mesh[1], n_mesh[0])).transpose(2, 1, 0)
        wanns.append(wann)
        meshes.append(n_mesh)
        origins.append(origin)
        step_matrices.append(grid_steps)
        headers.append(lines[:start + 6])
    n_mesh = meshes[0]
    grid_steps = step_matrices[0]
    origin = origins[0]
    # All functions must refer to the same physical grid.
    for i in range(1, len(wanns)):
        if not np.array_equal(meshes[i], n_mesh):
            raise ValueError("XSF files have different mesh dimensions. Interpolate them onto a common physical grid first.")
        if not np.allclose(origins[i], origin, atol=1e-8, rtol=0):
            raise ValueError("XSF files have different origins. Align their physical coordinate grids before comparison.")
        if not np.allclose(step_matrices[i], grid_steps, atol=1e-8, rtol=0):
            raise ValueError("XSF files have different grid-step vectors. Align their physical coordinate grids first.")
    return wanns, grid_steps, n_mesh, headers[0]
#*************************************************************************************

#*************************************************************************************
# Write xsf file
def write_wann(file_name,wann,n_mesh,header,lat_vec,transpose_wann=True):
    '''
    Write real-space function to xsf file. WILL NOT WORK IF LATTICE IS CONVERTED :(.
    Input:
    file_name: File name
    wann: function on 3D grid
    n_mesh: Mesh size in x, y, z
    header: XSF header
    Output:
    None
    '''
    # Flip wannierfunctions back
    _wann=np.copy(wann)
    _n_mesh=np.copy(n_mesh)
    _wann=_wann.transpose(2,1,0)
    _n_mesh[0], _n_mesh[2] = _n_mesh[2], _n_mesh[0]
    with open(file_name, 'w') as out_file:
        for line in header:
            out_file.write(line)
        wann_out=np.reshape(_wann,(_n_mesh[0]*_n_mesh[1]*_n_mesh[2])) 
        limit=int(np.ceil(float(_n_mesh[0]*_n_mesh[1]*_n_mesh[2])/6.0))
        wann_out = np.pad(wann_out, (0, limit * 6 - len(wann_out)), mode="constant", constant_values=0.0)
        wann_out=np.reshape(wann_out,(limit,6))
        np.savetxt(out_file,wann_out,fmt='%.8e',delimiter='\t')
        if 'xsf' in file_name:
            out_file.write("END_DATAGRID_3D\n")
            out_file.write("END_BLOCK_DATAGRID_3D\n")
    return 
#*************************************************************************************

#*************************************************************************************
# Normalizer
def normalize(wann, delr):
    """
    Normalize a real- or complex-valued wavefunction so that
    integral |wann|**2 dV = 1.
    """
    wann = np.asarray(wann)
    norm_squared = np.real(int_3d(np.abs(wann)**2, delr))
    if not np.isfinite(norm_squared) or norm_squared <= 0:
        raise ValueError(f"Cannot normalize wavefunction with norm squared {norm_squared}.")
    return wann / np.sqrt(norm_squared)
#*************************************************************************************

#************************************************************************************* 
# Do 3D numerical integration using the trapezoid rule
def int_3d(wann, delr):
    """
    Integrate an array of shape (nx, ny, nz).
    Parameters
    ----------
    wann : ndarray
        Real- or complex-valued function on a 3D grid.
    delr : array-like
        Either:
          * (dx, dy, dz): scalar orthogonal-grid spacings, or
          * (3, 3): grid-step vectors, one vector per array axis.
        The second form supports nonorthogonal grids using the
        determinant of the grid-step matrix.
    Returns
    -------
    scalar
        Integral using the tensor-product trapezoidal rule.
    """
    wann = np.asarray(wann)
    delr = np.asarray(delr, dtype=float)
    if wann.ndim != 3:
        raise ValueError("wann must be a 3D array.")
    if delr.shape == (3,):
        dx, dy, dz = delr
        if not np.all(np.isfinite(delr)) or np.any(delr <= 0):
            raise ValueError("Grid spacings must be finite and positive.")
        result = np.trapz(wann, dx=dz, axis=2)
        result = np.trapz(result, dx=dy, axis=1)
        result = np.trapz(result, dx=dx, axis=0)
    elif delr.shape == (3, 3):
        # Rows are the three grid-step vectors.
        volume_element = abs(np.linalg.det(delr))
        if not np.isfinite(volume_element) or volume_element == 0:
            raise ValueError("Grid-step vectors must span a nonzero volume.")
        result = np.trapz(wann, axis=2)
        result = np.trapz(result, axis=1)
        result = np.trapz(result, axis=0)
        result *= volume_element
    else:
        raise ValueError("delr must have shape (3,) or (3, 3).")
    return result
#*************************************************************************************

#************************************************************************************* 
# Interpolate wannier functions onto consistent grid
def interpolate_wann(wanns, delr, n_mesh_fine):
    """
    Interpolate wavefunctions onto a common mesh.
    Assumptions:
      * All input grids cover the same physical region.
      * All input grids have the same orientation and origin.
      * The supplied delr describes the TARGET grid.
      * The grids are orthogonal if delr has shape (3,).
    Parameters
    ----------
    wanns : sequence of 3D arrays
    delr : array-like, shape (3,) or (3, 3)
        Target grid spacing or grid-step vectors.
    n_mesh_fine : sequence of 3 positive integers
    Returns
    -------
    list of 3D arrays
    """
    n_mesh_fine = np.asarray(n_mesh_fine, dtype=int)
    if n_mesh_fine.shape != (3,) or np.any(n_mesh_fine < 2):
        raise ValueError("n_mesh_fine must contain three integers >= 2.")
    target_shape = tuple(n_mesh_fine)
    result = []
    for wann in wanns:
        wann = np.asarray(wann)
        if wann.ndim != 3 or any(n < 2 for n in wann.shape):
            raise ValueError("Each wavefunction must be 3D with >= 2 points per axis.")
        # Fractional coordinates preserve the same domain endpoints.
        axes = tuple(np.linspace(0.0, 1.0, n, dtype=float) for n in wann.shape)
        target_axes = tuple(np.linspace(0.0, 1.0, n, dtype=float) for n in target_shape)
        query = np.stack(np.meshgrid(*target_axes, indexing="ij"), axis=-1)
        interpolated = interpn(axes, wann, query, method="linear", bounds_error=True)
        interpolated = np.asarray(interpolated).reshape(target_shape)
        result.append(normalize(interpolated, delr))
    return result
#*************************************************************************************

#*************************************************************************************
# Calculate overlaps between sets of Wannier functions
def wann_overlap_unitary(wanns_1, delr_1, n_mesh_1, wanns_2, delr_2, n_mesh_2, round_overlap=0,verbose=True):
    '''
    For systems with similar wannier functions with potentially different ordering, 
    use overlaps of functions to determine the unitary between them.
    Input:                                                   
    wanns_1, wanns_2: List of wannier function on 3D grid in real space 
    delr_1, delr_2: 1D 3 element array containing mesh spacing         
    n_mesh_1, n_mesh_2: 1D 3 element array containing number of mesh points
    round_overlap: Number of decimal places
    verbose: Write stuff out                                                      
    Output:                                                  
    overlap: Unitary matrix of overlaps       

    '''
    _wanns_1=wanns_1.copy()
    _wanns_2=wanns_2.copy()
    # Interpolate onto mesh of wanns_2 
    _wanns_1_inter=interpolate_wann(_wanns_1,delr_1,n_mesh_2)
    overlap=np.zeros((len(_wanns_1_inter),len(_wanns_2)))
    for iwann1,wann1 in enumerate(_wanns_1_inter):
        for iwann2,wann2 in enumerate(_wanns_2):
            # Make sure they are normalized
            wann1=normalize(wann1,delr_2)
            wann2=normalize(wann2,delr_2)
            wann_x_wann = np.multiply(wann1,wann2)
            overlap[iwann1,iwann2]=np.trapz(np.trapz(np.trapz(wann_x_wann,dx=delr_2[0], axis=0),dx=delr_2[1], axis=0),dx=delr_2[2], axis=0)
    overlap=np.round(overlap,round_overlap)
    if verbose:
        print(overlap)
        print('Unitary test:',np.linalg.norm(overlap.T - np.linalg.inv(overlap)))
    return overlap
#*************************************************************************************

#*************************************************************************************
# Extract the rotation matricies from the pymatgen symmetry object
def get_sym_ops_xsf_frame(point_grp, verbose=False):
    """
    Return symmetry operations in the coordinate frame before the
    final [111]-to-z axis alignment in get_sym_ops().
    Use this when the XSF coordinate frame matches that pre-alignment
    frame. It reverses the final trans_axis_mat conjugation, but not
    any other change of lattice setting.
    """
    Q = np.array([[1 / np.sqrt(2),  1 / np.sqrt(6), 1 / np.sqrt(3)],
        [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)],
        [0.0,             -np.sqrt(2 / 3), 1 / np.sqrt(3)]])
    # Existing routine returns operations with the threefold axis aligned along z.
    ops_z = get_sym_ops(point_grp, verbose=False, hex_to_cart=True)
    # Undo the final axis alignment.
    ops_xsf = [Q @ R @ Q.T for R in ops_z]
    for R in ops_xsf:
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0):
            raise ValueError("Transformed symmetry operation is not orthogonal. Check the lattice-setting transformation and the actual XSF Cartesian frame.")
    if verbose:
        for R in ops_xsf:
            print(R)
    return ops_xsf


def construct_spinful_reps(orbital_reps, spin_reps, basis_order="spin-major", spin_component_order="up-down", verbose=False):
    """
    Construct combined spin-orbital symmetry matrices.
    Parameters
    ----------
    orbital_reps : sequence of (n_orb, n_orb) arrays
        Orbital representation matrices, one per symmetry operation.
    spin_reps : sequence of (2, 2) arrays
        Spin-1/2 matrices in the standard (up, down) spin basis.
    basis_order : {"spin-major", "orbital-major"}
        "spin-major":
            [all orbitals for first spin, all orbitals for second spin]
            D_full = kron(D_spin, D_orb)
        "orbital-major":
            [orbital 1 up, orbital 1 down, orbital 2 up, ...]
            D_full = kron(D_orb, D_spin)
    spin_component_order : {"up-down", "down-up"}
        Ordering of the two spin components in the Hamiltonian basis.
        get_spin_rep() supplies matrices in its standard up-down basis.
    verbose : bool
        Print dimensions and unitarity residuals.
    Returns
    -------
    spinful_reps : list of complex arrays
        Combined representation matrices, each of shape
        (2*n_orb, 2*n_orb).
    """
    if len(orbital_reps) != len(spin_reps):
        raise ValueError("The number of orbital and spin representations differs.")
    if basis_order not in ("spin-major", "orbital-major"):
        raise ValueError("basis_order must be 'spin-major' or 'orbital-major'.")
    if spin_component_order not in ("up-down", "down-up"):
        raise ValueError("spin_component_order must be 'up-down' or 'down-up'.")
    # Swap matrix for converting the spin basis from (up, down) to (down, up).
    P = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    spinful_reps = []
    for ig, (D_orb, D_spin) in enumerate(zip(orbital_reps, spin_reps)):
        D_orb = np.asarray(D_orb, dtype=np.complex128)
        D_spin = np.asarray(D_spin, dtype=np.complex128)
        if (D_orb.ndim != 2 or D_orb.shape[0] != D_orb.shape[1]):
            raise ValueError(f"Orbital representation {ig} is not square.")
        if D_spin.shape != (2, 2):
            raise ValueError(f"Spin representation {ig} must have shape (2, 2).")
        if (not np.all(np.isfinite(D_orb)) or not np.all(np.isfinite(D_spin))):
            raise ValueError(f"Non-finite entries in symmetry operation {ig}.")
        # Convert spin matrices into the ordering used by the
        # Hamiltonian basis, if necessary.
        if spin_component_order == "down-up":
            D_spin = P @ D_spin @ P
        if basis_order == "spin-major":
            D_full = np.kron(D_spin, D_orb)
        else:
            D_full = np.kron(D_orb, D_spin)
        spinful_reps.append(D_full)
        if verbose:
            err_orb = np.linalg.norm(D_orb.conj().T @ D_orb - np.eye(D_orb.shape[0]))
            err_spin = np.linalg.norm(D_spin.conj().T @ D_spin - np.eye(2))
            err_full = np.linalg.norm(D_full.conj().T @ D_full - np.eye(D_full.shape[0]))
            print("Full representation shape:", D_full.shape)
            print(f"Orbital unitarity residual: {err_orb:.3e}")
            print(f"Spin unitarity residual: {err_spin:.3e}")
            print(f"Full unitarity residual: {err_full:.3e}")
    return spinful_reps


def get_sym_ops(pg_symbol,verbose=False,hex_to_cart=False,hex_to_rhomb=False):
    '''
    Get point symmetry operations from pymatgen.
    Input: 
    pg_symbol: Hermann-Mauguin notation of point group
    verbose: Write out to the terminal or not
    hex_to_cart: Should we convert from symmetry operations WRT hexagonal lattice vectors to cartesian?
    around_z: Convert 3-fold axis from 111 to 001
    Outputs:
    point_sym_ops: List of 2D 3x3 arrays for symmetry operations
    '''
    if hex_to_cart or hex_to_rhomb:
        trans_latt=JonesFaithfulTransformation_CED.from_transformation_string("0.333333333(-a+b+c),0.333333333(2a+b+c),0.333333333(-a-2b+c);0,0,0")
        #if around_z:     
        trans_axis_mat=np.reshape(np.array([1.0/math.sqrt(2.0),1.0/math.sqrt(6.0),1.0/math.sqrt(3.0),-1.0/math.sqrt(2.0),1.0/math.sqrt(6.0),1.0/math.sqrt(3.0), \
                0.0,-math.sqrt(2.0/3.0),1.0/math.sqrt(3.0)]),(3,3))
    # From pymatgen
    point=PointGroup(pg_symbol)
    point_sym_ops=[]
    for sym in point.symmetry_ops:
        # Convert from hex to cartesian symmetry operations
        if hex_to_cart or hex_to_rhomb:
            # Transform to rhomahedral setting
            rot_mat=trans_latt.transform_symmop(sym).rotation_matrix
            if hex_to_cart:
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
def representation_fast(sym_op, wanns, delr, n_mesh, header=None, centering_type="fixed", center=np.array([0.5, 0.5, 0.5]), clean=True, clean_tol=5.0e-3, clean_vals=(0.0, 1.0, np.sqrt(3)/2, 0.5, 0.25,
    1/np.sqrt(2), np.sqrt(3/8), np.sqrt(5/8)), remove_rad=-1, center_index=None):
    """
    Calculate a spatial symmetry representation on an orthogonal or
    nonorthogonal uniform grid.
    delr may be either:
      * (dx, dy, dz), for an axis-aligned Cartesian grid;
      * (3, 3) grid-step vectors stored as rows, for a general grid.
    center_index gives the common symmetry-center position in
    array-index coordinates.
    """
    R = np.asarray(sym_op, dtype=float)
    n_mesh = np.asarray(n_mesh, dtype=int)
    grid = np.asarray(delr, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("sym_op must have shape (3, 3).")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("sym_op must be orthogonal.")
    if centering_type != "fixed":
        raise ValueError("Only fixed common-coordinate centering is supported.")
    if grid.shape == (3,):
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
            raise ValueError("Grid spacings must be positive and finite.")
        # Physical coordinate = A @ array_index.
        A = np.diag(grid)
    elif grid.shape == (3, 3):
        if not np.all(np.isfinite(grid)):
            raise ValueError("Grid-step vectors must be finite.")
        # grid rows are the step vectors of array axes.
        # Physical coordinate = A @ array_index.
        A = grid.T
        if abs(np.linalg.det(A)) < 1e-14:
            raise ValueError("Grid-step vectors are linearly dependent.")
    else:
        raise ValueError("delr must have shape (3,) or (3, 3).")
    if center_index is None:
        # A centered grid spans indices 0 through n_mesh - 1.
        center_index = (n_mesh - 1) * np.asarray(center, dtype=float)
    else:
        center_index = np.asarray(center_index, dtype=float)
    if center_index.shape != (3,):
        raise ValueError("center_index must have shape (3,).")
    # Pull transformation for psi'(r) = psi(R^{-1} r).
    matrix = np.linalg.solve(A, R.T @ A)
    offset = center_index - matrix @ center_index
    wf = [normalize(np.asarray(w), grid) for w in wanns]
    if len(wf) == 0:
        raise ValueError("No wavefunctions were provided.")
    if any(w.shape != tuple(n_mesh) for w in wf):
        raise ValueError("Wavefunction shape does not match n_mesh.")
    if remove_rad > 0:
        coms, _ = get_coms(wf)
        wf = [normalize(remove_outside_radius(w, n_mesh, com, radius=remove_rad), grid) for w, com in zip(wf, coms)]
    rep = np.zeros((len(wf), len(wf)), dtype=np.complex128)
    for j, wj in enumerate(wf):
        wj_sym = sni.affine_transform(wj, matrix=matrix, offset=offset, output_shape=tuple(n_mesh), order=1, mode="constant", cval=0.0, prefilter=False)
        for i, wi in enumerate(wf):
            rep[i, j] = int_3d(np.conjugate(wi) * wj_sym, grid)
    if clean:
        values = np.asarray(clean_vals, dtype=float)
        candidates = np.concatenate((values, -values))
        for i in range(len(wf)):
            for j in range(len(wf)):
                value = rep[i, j]
                if abs(value.imag) > clean_tol:
                    continue
                nearest = candidates[np.argmin(np.abs(candidates - value.real))]
                if abs(value.real - nearest) < clean_tol:
                    rep[i, j] = nearest
    return rep
#*************************************************************************************

#*************************************************************************************
# Generate the symmetry representations of spin 1/2
def get_spin_rep(sym_op, verbose=False, improper="axial"):
    """
    Calculate a 2x2 spin-1/2 matrix from a 3x3 orthogonal matrix.
    Parameters
    ----------
    sym_op : array-like, shape (3, 3)
        Cartesian orthogonal spatial symmetry operation.
    verbose : bool
        Print the rotation axis and angle.
    improper : {"axial", "reject"}
        "axial": for an improper operation, use the associated
                 proper rotation det(R) * R. This represents the
                 transformation of an axial vector such as spin.
                 It does not independently encode orbital parity.
        "reject": reject improper operations rather than assuming
                  a convention for their spin representation.
    Returns
    -------
    ndarray, shape (2, 2), complex
        Spin-1/2 representation.
    """
    R = np.asarray(sym_op, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("sym_op must have shape (3, 3).")
    if not np.all(np.isfinite(R)):
        raise ValueError("sym_op contains non-finite values.")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("sym_op must be an orthogonal matrix.")
    det_R = np.linalg.det(R)
    if not np.isclose(abs(det_R), 1.0, atol=1e-8):
        raise ValueError("sym_op must have determinant +1 or -1.")
    if det_R < 0:
        if improper == "reject":
            raise ValueError("Improper operation: specify a spin convention or decompose inversion and rotation explicitly.")
        elif improper == "axial":
            R = -R
        else:
            raise ValueError("improper must be 'axial' or 'reject'.")
    # Rotation vector = angle * axis, in radians.
    rotvec = Rotation.from_matrix(R).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle > 1e-14:
        axis = rotvec / angle
    else:
        axis = np.array([0.0, 0.0, 1.0])
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sigma_n = (axis[0] * sigma_x + axis[1] * sigma_y + axis[2] * sigma_z)
    # Equivalent to exp(-i * angle * sigma_n / 2).
    spin_d = (np.cos(angle / 2.0) * np.eye(2, dtype=np.complex128) - 1j * np.sin(angle / 2.0) * sigma_n)
    if verbose:
        print("Rotation axis:", axis)
        print("Rotation angle (rad):", angle)
        print("Spin representation:\n", spin_d)
        print("Unitarity error:", np.linalg.norm(spin_d.conj().T @ spin_d - np.eye(2)))
    return spin_d
#*************************************************************************************

#*************************************************************************************
# Print spinful represeantations
def write_spinful_reps(filename, sym_ops, spinful_reps, basis_order="spin-major", spin_component_order="up-down"):
    """
    Write one combined spin-orbital matrix per spatial operation.
    Complex entries are written explicitly as a+bi.
    """
    if len(sym_ops) != len(spinful_reps):
        raise ValueError("Symmetry-operation and spinful-representation counts differ.")
    with open(filename, "w") as f:
        f.write("Spin-orbital symmetry representations\n")
        f.write(f"Basis order: {basis_order}\n")
        f.write(f"Spin component order: {spin_component_order}\n\n")
        for ig, (R, D) in enumerate(zip(sym_ops, spinful_reps)):
            R = np.asarray(R)
            D = np.asarray(D, dtype=complex)
            f.write(f"Operation {ig}\n")
            f.write("Rotation matrix\n")
            for row in R:
                f.write("   ".join(f"{v: .10f}" for v in row) + "\n")
            f.write("Representation (spinful)\n")
            for row in D:
                entries = [f"{z.real:+.10f}{z.imag:+.10f}j" for z in row]
                f.write("   ".join(entries) + "\n")
            f.write("Spinful character\n")
            f.write(f"{np.trace(D).real:+.10f}{np.trace(D).imag:+.10f}j\n")
            f.write("Unitarity residual\n")
            f.write(f"{np.linalg.norm(D.conj().T @ D - np.eye(D.shape[0])):.6e}\n\n")
#*************************************************************************************

#*************************************************************************************
# Print represeantations
def print_reps(wan_files,center_in_cell=False,verbose=True,remove_rad=-1):
    '''
    Create file 'reps.dat' that includes: The symmetry operations, the
    orbital representations and characters, and the spin 1/2
    representations and characters.
    Inputs:
    wan_files: File containing names of the wannier function files
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
    if point_grp == "3m":
        pt_sym_ops = get_sym_ops_xsf_frame(point_grp, verbose=verbose)
    else:
        pt_sym_ops = get_sym_ops(point_grp, verbose=False)
    if verbose: print('Obtained sym ops')
    # Number of orbitals
    n_orb=len(wf_file_names)
    wanns,delr,n_mesh,header = load_xsf_files(wf_file_names)
    # Normalize wannier functions
    for iwann,wann in enumerate(wanns):
        wanns[iwann]=normalize(wann,delr)
    if verbose: print('Wannier functions read in and normalized.')
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
        rep=representation_fast(sym_op,wanns,delr,n_mesh,header,remove_rad=remove_rad)
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
        if verbose: print("Orb Character: ",str(np.trace(rep)))
        # Write out spin matrix
        f.write("Representation (spin)\n")
        f.write('{:.10f}   {:.10f}\n'.format(spin_mat[0,0],spin_mat[0,1]))
        f.write('{:.10f}   {:.10f}\n'.format(spin_mat[1,0],spin_mat[1,1]))
        f.write("Spin character\n")
        f.write('{:.5f}'.format(np.trace(spin_mat)))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        if verbose: print("Spin Character: ",str(np.trace(spin_mat)))
    f.close()
    return
#*************************************************************************************

#*************************************************************************************
# Print represeantations
def print_reps_wann_in(point_grp,n_mesh,wanns=None,delr=None,out_label='',centering_type='fixed',verbose=True,sym_op_lim=None,use_ylm=-1,rad_scale=4.0,clean=True,clean_tol=5.0e-3,
        clean_vals=[0.0,1.0,np.sqrt(3)/2,0.5,0.25,1/np.sqrt(2),np.sqrt(3/8),np.sqrt(5/8)],remove_rad=-1):
    '''
    Different version of print_reps that takes as input wanns instead of reading them in from file.
    Creates file out_label+'reps.dat' that includes: The symmetry operations, the
    orbital representations and characters, and the spin 1/2
    representations and characters.
    Inputs:
    wanns : List of real-space wannier functions on cartesian grid
    delr : Three element list of mesh spacing in x, y, z
    n_mesh : Three element list of number of mesh points in x, y, z
    point_grp : symbol for point group
    out_label : Label for reps file
    verbose : Write out stuff to stdout
    sym_op_lim : Limit to just a subset of symmetry operations 
    use_ylm : Directly generate reps from ylms if set to >= 0
    rad_scale : Scale for radial exponential
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
    # Get symmetry operations from pymatgen
    if point_grp in ['3m','-3m', '6/mmm', '-6m2']:
        pt_sym_ops=get_sym_ops(point_grp,verbose=False,hex_to_cart=True)
    else:
        pt_sym_ops=get_sym_ops(point_grp,verbose=False)
    if sym_op_lim:
        pt_sym_ops=pt_sym_ops[sym_op_lim[0]:sym_op_lim[1]]
    # If we provided wanns
    if wanns:
        # Number of orbitals
        n_orb=len(wanns)
        # Normalize wannier functions
        for iwann,wann in enumerate(wanns):
            wanns[iwann]=normalize(wann,delr)
    # If using ylms directly
    elif use_ylm >= 0:
        n_orb=int(2*use_ylm+1)
    else:
        raise Exception('Must input wanns ore use_ylm')
    if verbose: print('Wannier functions read in and normalized.')
    # Write out info to file:
    f=open(out_label+"reps.dat","w+")
    f.write("Representations for the symmetry operations:\n")
    f.write("\n")
    for sym_op in pt_sym_ops:
        # Write symmetry operation matrix
        f.write("Rotation matrix\n")
        for i in range(0,3):
            f.write('{:.10f}   {:.10f}   {:.10f}\n'.format(sym_op[i,0],sym_op[i,1],sym_op[i,2]))
        # Calculate orbital part of representation
        if wanns is not None and len(wanns) > 0:
            rep=representation_fast(sym_op,wanns,delr,n_mesh,[],clean=clean,clean_tol=clean_tol,clean_vals=clean_vals,remove_rad=remove_rad,centering_type=centering_type)
        else:    
            rep=get_rep_ylms(use_ylm,n_mesh,rad_scale,sym_op,clean=clean,clean_tol=clean_tol,clean_vals=clean_vals,remove_rad=remove_rad,centering_type=centering_type)
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
        if verbose: print("Orb Character: ",str(np.trace(rep)))
        # Write out spin matrix
        f.write("Representation (spin)\n")
        f.write('{:.10f}   {:.10f}\n'.format(spin_mat[0,0],spin_mat[0,1]))
        f.write('{:.10f}   {:.10f}\n'.format(spin_mat[1,0],spin_mat[1,1]))
        f.write("Spin character\n")
        f.write('{:.5f}'.format(np.trace(spin_mat)))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        if verbose: print("Spin Character: ",str(np.trace(spin_mat)))
    f.close()
    return


def print_reps(wan_files, center_in_cell=False, verbose=True, remove_rad=-1, clean=True, clean_tol=5.0e-3, print_spinful=True, spinful_out_file=None, spinful_basis_order="spin-major",
        spin_component_order="up-down", center_index=None):
    """
    Write orbital/spin representations to reps.dat.
    Optionally construct and write combined spin-orbital representations
    to spinful_reps.dat.
    Returns a dictionary containing all generated matrices.
    """
    with open(wan_files, "r") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Empty Wannier-function file list.")
    point_grp = lines[0].strip()
    wf_file_names = [line.strip() for line in lines[1:] if line.strip()]
    if not wf_file_names:
        raise ValueError("No Wannier-function filenames were provided.")
    file_type = wf_file_names[0].rsplit(".", 1)[-1].lower()
    if file_type not in ("xsf", "cube"):
        raise ValueError("Expected .xsf or .cube Wannier-function files.")
    # Preserve the symmetry-operation convention of the current script.
    if point_grp == "3m":
        pt_sym_ops = get_sym_ops_xsf_frame(point_grp, verbose=False)
    else:
        pt_sym_ops = get_sym_ops(point_grp, verbose=False, hex_to_cart=point_grp in ("-3m", "6/mmm", "-6m2"))
    if verbose:
        print("Obtained sym ops")
    if file_type == "xsf":
        wanns, delr, n_mesh, header = load_xsf_files(wf_file_names)
    else:
        wanns, delr, n_mesh, header = load_cube_files(wf_file_names)
    # Work with normalized copies rather than changing the caller's arrays.
    wanns = [normalize(np.asarray(w), delr) for w in wanns]
    n_orb = len(wanns)
    if verbose:
        print("Wannier functions read in and normalized.")
    orbital_reps = []
    spin_reps = []
    with open("reps.dat", "w") as f:
        f.write("Representations for the symmetry operations:\n\n")
        for ig, sym_op in enumerate(pt_sym_ops):
            sym_op = np.asarray(sym_op, dtype=float)
            f.write(f"Operation {ig}\n")
            f.write("Rotation matrix\n")
            for row in sym_op:
                f.write("{:.10f}   {:.10f}   {:.10f}\n".format(*row))
            rep = representation_fast(sym_op, wanns, delr, n_mesh, header, centering_type="fixed", clean=clean, clean_tol=clean_tol, remove_rad=remove_rad, center_index=center_index)
            # This file's orbital representation is expected to be real.
            rep = np.asarray(rep, dtype=complex)
            if np.max(np.abs(rep.imag)) > 1e-7:
                raise ValueError(f"Orbital representation {ig} has significant imaginary components. Check the orbital basis and transformation convention.")
            rep = rep.real
            spin_mat = get_spin_rep(sym_op)
            # Save the matrices in precisely the same operation order.
            orbital_reps.append(rep)
            spin_reps.append(spin_mat)
            f.write("Representation (Orb)\n")
            for row in rep:
                f.write("".join(f"{v:12.8f}" for v in row) + "\n")
            f.write("Character\n")
            f.write(f"{np.trace(rep):.8f}\n")
            f.write("Representation (spin)\n")
            for row in spin_mat:
                f.write("   ".join(f"{z.real:+.10f}{z.imag:+.10f}j" for z in row) + "\n")
            f.write("Spin character\n")
            spin_char = np.trace(spin_mat)
            f.write(f"{spin_char.real:+.8f}{spin_char.imag:+.8f}j\n\n")
            if verbose:
                print(f"Operation {ig}: orbital character = {np.trace(rep):.8f}, spin character = {spin_char}")
    # Construct combined representations entirely within this function.
    spinful_reps = None
    if print_spinful:
        spinful_reps = construct_spinful_reps(orbital_reps, spin_reps, basis_order=spinful_basis_order, spin_component_order=spin_component_order, verbose=verbose)
        if spinful_out_file is None:
            spinful_out_file = "spinful_reps.dat"
        write_spinful_reps(spinful_out_file, pt_sym_ops, spinful_reps, basis_order=spinful_basis_order, spin_component_order=spin_component_order)
        if verbose:
            print(f"Spinful representations written to {spinful_out_file}")
            for ig, D in enumerate(spinful_reps):
                print(f"Spinful character, operation {ig}: {np.trace(D)}")
    return {"sym_ops": pt_sym_ops, "orbital_reps": orbital_reps, "spin_reps": spin_reps, "spinful_reps": spinful_reps}
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
    # This code will be the same as load_xsf_file, but need to to it again to write to new file
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
# Get COMs of wannier functions
def get_coms(wanns):
    """
    Calculate probability-density-weighted centers on a Cartesian grid.
    Returns centers in array-index coordinates, not physical units.
    """
    coms = []
    for wann in wanns:
        rho = np.abs(np.asarray(wann))**2
        total = rho.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Cannot calculate COM of a zero/non-finite orbital.")
        coms.append(np.array(sni.center_of_mass(rho), dtype=float))
    rho_total = sum(np.abs(np.asarray(w))**2 for w in wanns)
    total = rho_total.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot calculate total COM.")
    com_tot = np.array(sni.center_of_mass(rho_total), dtype=float)
    return coms, com_tot
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
    com=np.array(sni.center_of_mass(np.abs(wann1)))
    shift=com-np.array([float(n_mesh[0])/2.0,float(n_mesh[1])/2.0,float(n_mesh[2])/2.0])
    # Center Wannier function
    if mod_wan:
        # For the case where we center WRT the total COM
        if wrt_tot_com:
            shift=(com-com_tot)
        trans_mat=np.identity(4)
        trans_mat[0][3]=shift[0]
        trans_mat[1][3]=shift[1]
        trans_mat[2][3]=shift[2]
        wann1=sni.affine_transform(wann1,trans_mat)
    return wann1,com
#*************************************************************************************            

#*************************************************************************************
# Shift Wannier center (REPEAT OF ABOVE!!!)
def shift_wann_func(wann, new_com):
    """
    Shift a wavefunction so its probability-density COM reaches new_com.
    Coordinates are in array-index units.
    """
    wann = np.asarray(wann)
    com = np.asarray(sni.center_of_mass(np.abs(wann)**2), dtype=float) 
    new_com = np.asarray(new_com, dtype=float)
    offset = com - new_com
    return sni.affine_transform(wann, matrix=np.eye(3), offset=offset, order=1, mode="constant", cval=0.0, prefilter=False)
#*************************************************************************************            

#*************************************************************************************
# Remove parts of Wannier functions outside of some radius
def remove_outside_radius(wann,n_mesh,com,radius=20.0):
    '''
    Remove parts of Wannier functions outside of some radius
    Input:
    wann: Wannier function on 3D grid
    n_mesh: Mesh size in x,y,z
    com: Center of mass of the wannier function
    radius: Rqdius to remove contribtuins outside of
    Output:
    wann: Modified Wannier function on 3D grid
    '''
    _wann=np.copy(wann)
    for i in range(n_mesh[0]):
        for j in range(n_mesh[1]):
            for k in range(n_mesh[2]):
                norm= np.linalg.norm(np.array([i-com[0],j-com[1],k-com[2]]))
                if norm > radius:
                    _wann[i,j,k] = 0.0
    return _wann
#*************************************************************************************

#*************************************************************************************
def make_ylm_wanns(ll, n_mesh, basis, rad_scale):
    """
    Generate normalized spherical or real (cubic-style) harmonics
    multiplied by exp(-rad_scale * r).
    The mesh covers [-1, 1] along each Cartesian axis.
    Array axes are (x, y, z).
    """
    if not isinstance(ll, (int, np.integer)) or ll < 0:
        raise ValueError("ll must be a nonnegative integer.")
    n_mesh = np.asarray(n_mesh, dtype=int)
    if n_mesh.shape != (3,) or np.any(n_mesh < 2):
        raise ValueError("n_mesh must contain three integers >= 2.")
    if rad_scale < 0:
        raise ValueError("rad_scale must be nonnegative.")
    size = 1.0
    axes = [np.linspace(-size, size, n) for n in n_mesh]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(x*x + y*y + z*z)
    # Safe spherical coordinates at the origin
    theta = np.zeros_like(r)
    phi = np.arctan2(y, x)
    nonzero = r > 0
    theta[nonzero] = np.arccos(np.clip(z[nonzero] / r[nonzero], -1.0, 1.0))
    # Avoid the undefined direction at r = 0; for l > 0, all spherical harmonics vanish at the origin under the usual solid-harmonic extension
    radial = np.exp(-rad_scale * r)
    # Compatibility with the legacy scipy.special.sph_harm
    def Y(m, l):
        return sph_harm(m, l, phi, theta)
    wanns = []
    if basis == "spherical":
        for m in range(-ll, ll + 1):
            wann = Y(m, ll).copy()
            if ll > 0:
                wann[~nonzero] = 0.0
            wanns.append(wann)
    elif basis == "cubic":
        for m in range(-ll, ll + 1):
            if m < 0:
                wann = np.sqrt(2.0) * ((-1)**m) * Y(abs(m), ll).imag
            elif m > 0:
                wann = np.sqrt(2.0) * ((-1)**m) * Y(abs(m), ll).real
            else:
                wann = Y(0, ll).real
            wann = np.asarray(wann).copy()
            if ll > 0:
                wann[~nonzero] = 0.0
            wanns.append(wann)
        # Preserve TRIQS ordering just in case
        if ll == 3:
            wanns.reverse()
    else:
        raise ValueError("basis must be 'spherical' or 'cubic'.")
    delr = np.array([axes[0][1] - axes[0][0], axes[1][1] - axes[1][0], axes[2][1] - axes[2][0]])
    wanns = [normalize(wann * radial, delr) for wann in wanns]
    return wanns, delr
#*************************************************************************************

#*************************************************************************************
def get_rep_ylms(ll, n_mesh, rad_scale, sym_op, clean=True, clean_tol=5.0e-3, clean_vals=None):
    """
    Calculate the representation matrix for a real spherical-harmonic
    basis by numerical integration.
    Uses Cartesian coordinates and supports arbitrary orthogonal
    Cartesian symmetry matrices.
    Basis ordering:
        m = -ll, ..., ll
    Note:
        This uses a real-harmonic basis and the legacy SciPy sph_harm
        convention. Verify basis signs and ordering!!
    """
    n_mesh = np.asarray(n_mesh, dtype=int)
    R = np.asarray(sym_op, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("sym_op must have shape (3, 3).")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("sym_op must be orthogonal.")
    axes = [np.linspace(-1.0, 1.0, n) for n in n_mesh]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    coords = np.stack((x, y, z), axis=-1)
    # Use the inverse map for the scalar field: f_transformed(r) = f(R^{-1} r).
    coords_rot = coords @ R
    def angles(c):
        xx, yy, zz = c[..., 0], c[..., 1], c[..., 2]
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        theta = np.zeros_like(rr)
        nonzero = rr > 0
        theta[nonzero] = np.arccos( np.clip(zz[nonzero] / rr[nonzero], -1.0, 1.0))
        phi = np.arctan2(yy, xx)
        return theta, phi, nonzero
    theta, phi, nonzero = angles(coords)
    theta_r, phi_r, nonzero_r = angles(coords_rot)
    def real_ylm(m, theta, phi, nonzero):
        if m == 0:
            val = sph_harm(0, ll, phi, theta).real
        elif m > 0:
            val = (np.sqrt(2.0) * ((-1)**m) * sph_harm(m, ll, phi, theta).real)
        else:
            val = (np.sqrt(2.0) * ((-1)**m) * sph_harm(-m, ll, phi, theta).imag)
        val = np.asarray(val).copy()
        if ll > 0:
            val[~nonzero] = 0.0
        return val
    radial = np.exp(-rad_scale * np.sqrt(x**2 + y**2 + z**2))
    delr = np.array([axes[i][1] - axes[i][0] for i in range(3)])
    basis = []
    basis_rot = []
    for m in range(-ll, ll + 1):
        basis.append(normalize(real_ylm(m, theta, phi, nonzero) * radial, delr))
        basis_rot.append(normalize(real_ylm(m, theta_r, phi_r, nonzero_r) * radial, delr))
    rep = np.zeros((2*ll + 1, 2*ll + 1))
    for i, wi in enumerate(basis):
        for j, wj in enumerate(basis_rot):
            overlap = int_3d(wi * wj, delr)
            if clean:
                if clean_vals is None:
                    clean_vals = [0.0, 1.0, -1.0, np.sqrt(3)/2, -np.sqrt(3)/2, 0.5, -0.5, 1/np.sqrt(2), -1/np.sqrt(2)]
                candidates = np.asarray(clean_vals, dtype=float)
                candidates = np.concatenate((candidates, -candidates))
                nearest = candidates[np.argmin(abs(candidates - overlap))]
                if abs(overlap - nearest) < clean_tol:
                    overlap = nearest
            rep[i, j] = overlap
    return rep
#*************************************************************************************



#*************************************************************************************
# END OF FUNCTIONS
#*************************************************************************************
if __name__ == '__main__':
    print_reps('wan_files.dat')
