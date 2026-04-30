"""
This script automates the generation of synthetic 3D Irreversible Electroporation (IRE) simulation datasets
for varying grid resolutions (25x25x25, 50x50x50, 100x100x100). It simulates the electric potential distribution
around randomly positioned needle electrodes, using both direct and iterative solvers. The results are saved
as NIfTI files for further analysis or machine learning applications.
"""

import Simulate_Basis_Fonctions as SBF  # Custom module for IRE basis function and solver routines
import nibabel as nib  # Neuroimaging library for handling NIfTI file format
import numpy as np  # Numerical computing library
import time  # Time measurement for performance benchmarking
from scipy.ndimage import zoom  # For upscaling low-resolution solutions as initial guesses

"""
Global Simulation Parameters:
- nb_simulation: Total number of independent simulations to generate.
- epsilon_needle: Regularization parameter for needle modeling (avoids singularities).
- tolerence: Convergence threshold for iterative solvers (BiCGSTAB).
- data_folder: Directory path for saving simulation outputs (NIfTI files).
"""
nb_simulation = 100  # Total number of simulations
epsilon_needle = 1e-12  # Needle regularization parameter
tolerence = 1e-20  # Iterative solver convergence tolerance
SBF.epsilon_needle = epsilon_needle  # Pass parameters to the simulation module
SBF.tolerence = tolerence
number_needle = 3  # Number of electrodes per simulation
    
# Output directory for simulation data
data_folder = "C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\data\\"

"""
Main Simulation Loop:
Generates nb_simulation independent cases, each with:
- Randomly positioned needle electrodes.
- Basis functions (g, f) and potential (u) computed for three grid resolutions.
- Results saved as NIfTI files for each resolution.
"""
for i in range(0, nb_simulation):
    print("Generation of case {}".format(i))

    # --- Needle Configuration ---
    # Generate random needle coordinates and save to file
    SBF.random_needle_coord(
        data_folder+'needles_coord_{}_{}.txt'.format(number_needle, i),
        number_needle, 100, 100, 100
    )
    # Read needle coordinates from file
    [nb_needles, tip_coord, tail_coord] = SBF.read_needle_coord(
        data_folder+'needles_coord_{}_{}.txt'.format(number_needle, i)
    )

    # --- 25x25x25 Grid Simulation ---
    """
    Low-resolution simulation (25^3 grid):
    - Computes basis functions g (needle mask) and f (source term).
    - Solves for potential u using a direct implicit solver.
    - Saves g, f, and u as NIfTI files.
    """
    start_time = time.time()
    [g, f] = SBF.compute_basis_function(
        tip_coord, tail_coord,
        num_basis_function=2,  # Basis function index (arbitrary choice)
        nb_needles=nb_needles,
        nx=25, ny=25, nz=25
    )
    # Save basis function g (needle mask)
    image_nifti = nib.Nifti1Image(g, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'g_25_{}.nii.gz'.format(i))
    # Save source term f
    image_nifti = nib.Nifti1Image(f, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'f_25_{}.nii.gz'.format(i))

    # Homogeneous conductivity (simplified model)
    conductivity = np.ones((25, 25, 25))
    # Direct implicit solver for low-resolution potential
    u = SBF.solve_IRE_Implicit(g, f, conductivity)
    # Save potential u
    image_nifti = nib.Nifti1Image(u, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'u_25_{}.nii.gz'.format(i))
    print('Time for generation 25 = ', time.time()- start_time)

    # --- 50x50x50 Grid Simulation ---
    """
    Medium-resolution simulation (50^3 grid):
    - Computes basis functions g and f at higher resolution.
    - Uses the low-resolution solution (upscaled) as initial guess for BiCGSTAB.
    - Solves for potential u using an iterative solver.
    - Saves g, f, and u as NIfTI files.
    """
    [g, f] = SBF.compute_basis_function(
        tip_coord, tail_coord,
        num_basis_function=2,
        nb_needles=nb_needles,
        nx=50, ny=50, nz=50
    )
    # Save basis function g
    image_nifti = nib.Nifti1Image(g, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'g_50_{}.nii.gz'.format(i))
    # Save source term f
    image_nifti = nib.Nifti1Image(f, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'f_50_{}.nii.gz'.format(i))

    # Homogeneous conductivity
    conductivity = np.ones((50, 50, 50))
    # Upscale low-resolution potential as initial guess
    u_lowres = zoom(u, 2, mode="nearest")
    # Iterative solver (BiCGSTAB) with multigrid initialization
    u = SBF.solve_IRE_bicgstab_fast(g, f, conductivity, u_lowres.flatten())
    # Save potential u
    image_nifti = nib.Nifti1Image(u, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'u_50_{}.nii.gz'.format(i))
    print('Time for generation 50 = ', time.time()- start_time)

    # --- 100x100x100 Grid Simulation ---
    """
    High-resolution simulation (100^3 grid):
    - Computes basis functions g and f at highest resolution.
    - Uses the medium-resolution solution (upscaled) as initial guess for BiCGSTAB.
    - Solves for potential u using an iterative solver.
    - Saves g, f, and u as NIfTI files.
    """
    [g, f] = SBF.compute_basis_function(
        tip_coord, tail_coord,
        num_basis_function=2,
        nb_needles=nb_needles,
        nx=100, ny=100, nz=100
    )
    # Save basis function g
    image_nifti = nib.Nifti1Image(g, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'g_100_{}.nii.gz'.format(i))
    # Save source term f
    image_nifti = nib.Nifti1Image(f, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'f_100_{}.nii.gz'.format(i))

    # Homogeneous conductivity
    conductivity = np.ones((100, 100, 100))
    # Upscale medium-resolution potential as initial guess
    u_lowres = zoom(u, 2, mode="nearest")
    # Iterative solver (BiCGSTAB) with multigrid initialization
    u = SBF.solve_IRE_bicgstab_fast(g, f, conductivity, u_lowres.flatten())
    # Save potential u
    image_nifti = nib.Nifti1Image(u, affine=np.eye(4))
    nib.save(image_nifti, data_folder+'u_100_{}.nii.gz'.format(i))
    print('Time for generation 100 = ', time.time()- start_time)