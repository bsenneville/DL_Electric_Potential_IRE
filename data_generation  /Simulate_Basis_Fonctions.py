"""
This script implements a 3D numerical solver for the Irreversible Electroporation (IRE) problem,
using both implicit and iterative (BiCGSTAB) methods. The code models the electric potential distribution
in a 3D domain with embedded needle electrodes, accounting for spatial discretization, boundary conditions,
and material conductivity. The solution is obtained via sparse matrix algebra and iterative solvers.
"""

import numpy as np
import time
from compute_gradient import compute_gradient
from Bresenham3D import Bresenham3D
from scipy.sparse.linalg import spsolve
import scipy.sparse as scsp
from fill_matrix_sparsekronecker import Lap3DNeu
import random
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom

"""
Global Parameters:
- nx, ny, nz: Number of spatial discretization steps in x, y, and z directions.
- dx, dy, dz: Spatial step sizes in x, y, and z directions.
- epsilon_needle: Tolerance thresholds for convergence and needle modeling.
- active_needle_size: Length of the active region of the needle electrode.
- tolerence: Convergence tolerance for iterative solvers.
- dx2inv, dy2inv, dz2inv: Precomputed inverse square step sizes for efficiency.
"""
nx = 32  # Number of steps in space(x)
ny = nx  # Number of steps in space(y)
nz = nx  # Number of steps in space(z)

dx = 2 / (nx - 1)  # Width of space step(x)
dy = 2 / (ny - 1)  # Width of space step(y)
dz = 2 / (nz - 1)  # Width of space step(z)

epsilon_needle = (1e-5)**2  # Tolerance for needle modeling
active_needle_size = 40  # 4cm of active needle region
tolerence = 1e-30  # Convergence tolerance for iterative solvers

# Precalculation of inverse square step sizes for efficiency
dx2inv = 1.0 / (dx**2)
dy2inv = 1.0 / (dy**2)
dz2inv = 1.0 / (dz**2)

def global_computation():
    """
    Recalculates global spatial parameters (dx, dy, dz) and their inverse squares.
    Useful for dynamic resizing or adaptive meshing.
    """
    global dx, dy, dz, dx2inv, dy2inv, dz2inv
    dx = 2 / (nx - 1)  # Recompute width of space step(x)
    dy = 2 / (ny - 1)  # Recompute width of space step(y)
    dz = 2 / (nz - 1)  # Recompute width of space step(z)
    dx2inv = 1.0 / (dx**2)  # Recompute inverse square step size(x)
    dy2inv = 1.0 / (dy**2)  # Recompute inverse square step size(y)
    dz2inv = 1.0 / (dz**2)  # Recompute inverse square step size(z)

def random_needle_coord(nom_fichier, number_needle, x_max, y_max, z_max):
    """
    Generates random coordinates for needle electrodes and saves them to a file.
    Ensures the active region of each needle is of fixed length (active_needle_size).
    Uses rejection sampling to avoid invalid coordinates.
    """
    try:
        with open(nom_fichier, 'w') as f:
            for i in range(number_needle):
                x_tail = x_max
                while (x_tail >= x_max or y_tail >= y_max or z_tail >= z_max or
                       x_tail < 0 or y_tail < 0 or z_tail < 0):
                    # Randomly sample tip and tail coordinates
                    x_tip = x_max * random.random()
                    y_tip = y_max * random.random()
                    z_tip = z_max * random.random()
                    x_tail = x_max * random.random()
                    y_tail = y_max * random.random()
                    z_tail = z_max * random.random()

                    # Compute needle length
                    needle_size = np.sqrt((x_tip - x_tail)**2 +
                                          (y_tip - y_tail)**2 +
                                          (z_tip - z_tail)**2)

                    # Adjust tail coordinates to ensure active region length
                    x_tail = x_tip + (x_tail - x_tip) * (active_needle_size/needle_size)
                    y_tail = y_tip + (y_tail - y_tip) * (active_needle_size/needle_size)
                    z_tail = z_tip + (z_tail - z_tip) * (active_needle_size/needle_size)
                f.write('{} {} {} {} {} {}\n'.format(x_tip, y_tip, z_tip, x_tail, y_tail, z_tail))
    except FileNotFoundError:
        print(f"File {nom_fichier} not found.")

def read_needle_coord(nom_fichier):
    """
    Reads needle coordinates from a file and returns them as numpy arrays.
    Returns the number of needles, tip coordinates, and tail coordinates.
    """
    needle_coord = []
    try:
        with open(nom_fichier, 'r') as fichier:
            for ligne in fichier:
                nombres = ligne.strip().split()
                ligne_de_nombres = [float(nombre) for nombre in nombres]
                needle_coord.append(ligne_de_nombres)
    except FileNotFoundError:
        print(f"File {nom_fichier} not found.")
    except ValueError:
        print("Conversion error: Ensure the file contains only floating-point numbers.")

    needle_coord = np.array(needle_coord)
    nb_needles = needle_coord.shape[0]
    tip_coord = needle_coord[:,0:3]
    tail_coord = needle_coord[:,3:6]

    return nb_needles, tip_coord, tail_coord

def compute_basis_function(tip_coord, tail_coord, num_basis_function, nb_needles, nx, ny, nz):
    """
    Computes the basis function for each needle electrode in the 3D domain.
    Uses Bresenham's 3D line algorithm to trace the path of each needle.
    Returns a potential map (each needle is assigned a unique potential) and a binary needle mask.
    """
    # Scale coordinates to grid indices
    tip_coord = np.floor(np.multiply(tip_coord,[nx/100, ny/100, nz/100])).astype(int)
    tail_coord = np.floor(np.multiply(tail_coord,[nx/100, ny/100, nz/100])).astype(int)

    # Initialize potential and needle maps
    potential_map = np.zeros((nx, ny, nz))
    for num_needle in range(nb_needles):
        # Trace the needle path using Bresenham's algorithm
        ListOfPoints = Bresenham3D(tip_coord[num_needle, 0], tip_coord[num_needle, 1], tip_coord[num_needle, 2],
                                   tail_coord[num_needle, 0], tail_coord[num_needle, 1], tail_coord[num_needle, 2])
        for i in range(0,len(ListOfPoints)):
            voxel = ListOfPoints[i]
            potential_map[voxel[0],voxel[1],voxel[2]] = num_needle+1

    # Create binary needle mask
    needle_map = np.zeros((nx, ny, nz))
    needle_map[potential_map>0] = 1

    # Isolate the selected basis function
    potential_map[potential_map!=num_basis_function] = 0
    potential_map[potential_map==num_basis_function] = 1

    return potential_map, needle_map

def compute_implicit_matrix_fast(g, f, conductivity):
    """
    Constructs the sparse system matrix A and right-hand side vector b for the implicit method.
    Incorporates Neumann boundary conditions and conductivity gradients.
    Returns the system matrix A and the right-hand side vector b.
    """
    start_time = time.time()
    (nx, ny, nz) = g.shape

    # Recompute spatial step sizes
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dz = 2 / (nz - 1)
    dx2 = dx**2

    # Precompute scaled terms for efficiency
    f_div_eps = f*dx2/epsilon_needle/conductivity
    gf_div_eps = np.multiply(g, f_div_eps)
    gf_div_eps = np.reshape(gf_div_eps, (nx * ny * nz, 1))

    # Compute gradients of log-conductivity
    [gradx_logconductivity, grady_logconductivity, gradz_logconductivity] = compute_gradient(np.log(conductivity), dx, dy, dz)
    gradx_logconductivity = np.reshape(gradx_logconductivity, (nx * ny * nz))
    grady_logconductivity = np.reshape(grady_logconductivity, (nx * ny * nz))
    gradz_logconductivity = np.reshape(gradz_logconductivity, (nx * ny * nz))

    # Reshape conductivity for matrix operations
    conductivity = np.reshape(conductivity, (nx * ny * nz, 1))

    # Apply boundary conditions to gradients
    gradx_logconductivity[::nx] = 0
    gradx_logconductivity[nx-1::nx] = 0
    for i in range(nx*ny*nz, nx*(ny-1)):
        grady_logconductivity[i:i+nx] = 0
    for i in range(nx*ny*nz, nx*ny*(nz-1)):
        gradz_logconductivity[i:i+nx*ny] = 0

    print('Filling implicit matrix')

    # Initialize right-hand side vector
    b = gf_div_eps

    # Construct system matrix A using sparse operations
    A = -Lap3DNeu(nx, ny, nz, alpha1=dy/dx, alpha2=dz/dx)
    A -= scsp.diags(gradx_logconductivity[:-1]*dx/2, offsets=1) + scsp.diags(-gradx_logconductivity[1:]*dx/2, offsets=-1)
    A -= scsp.diags(grady_logconductivity[:-nx]*dx2/2/dy, offsets=nx) + scsp.diags(-grady_logconductivity[nx:]*dx2/dy/2, offsets=-nx)
    A -= scsp.diags(gradz_logconductivity[:-nx*ny]*dx2/2/dz, offsets=nx*ny) + scsp.diags(-gradz_logconductivity[nx*ny:]*dx2/dz/2, offsets=-nx*ny)
    A += scsp.diags(np.reshape(f_div_eps, (nx * ny * nz)))

    # Apply Neumann boundary conditions to b
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                index = k * nx * ny + j * nx + i
                if i == 0 or j == 0 or k == 0 or i == nx-1 or j == ny-1 or k == nz-1:
                    b[index] = 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Implicit matrix fill elapsed time = ', elapsed_time)

    return A, b

def solve_IRE_Implicit(g, f, conductivity):
    """
    Solves the IRE problem using the implicit method (direct sparse solver).
    Returns the electric potential distribution u.
    """
    (nx, ny, nz) = g.shape
    [A, b] = compute_implicit_matrix_fast(g, f, conductivity)
    print('Solving linear system')
    u = spsolve(A, b)
    u = np.array(u.reshape((nx, ny, nz)))
    return u

def solve_IRE_bicgstab(g, f, conductivity, u0=None):
    """
    Solves the IRE problem using the BiCGSTAB iterative method.
    Tracks convergence via a callback function and returns the solution, residuals, and intermediate solutions.
    """
    num_iters = 0
    residu = []
    list_u = []

    def callback(uk):
        nonlocal num_iters, b, A, residu, list_u
        num_iters += 1
        res = b - A@scsp.csr_matrix(uk.reshape((-1,1)))
        residu.append(np.linalg.norm(res)/np.sqrt(len(res)))
        list_u.append(uk.copy().reshape((nx,ny,nz)))

    (nx, ny, nz) = g.shape
    start_time = time.time()
    [A, b] = compute_implicit_matrix_fast(g, f, conductivity)
    u, info = scsp.linalg.bicgstab(A, b, x0=u0, rtol=tolerence, callback=callback, maxiter=10000)
    print(num_iters)
    u = np.array(u.reshape((nx, ny, nz)))
    end_time = time.time() - start_time
    print("Implicit BiCGSTAB time elapsed =", end_time)
    return u, residu, list_u, end_time

def solve_IRE_bicgstab_fast(g, f, conductivity, u0=None):
    """
    Solves the IRE problem using the BiCGSTAB iterative method without tracking intermediate solutions.
    Returns the electric potential distribution u.
    """
    (nx, ny, nz) = g.shape
    [A, b] = compute_implicit_matrix_fast(g, f, conductivity)
    u, info = scsp.linalg.bicgstab(A, b, x0=u0, rtol=tolerence, maxiter=10000)
    u = np.array(u.reshape((nx, ny, nz)))
    return u
