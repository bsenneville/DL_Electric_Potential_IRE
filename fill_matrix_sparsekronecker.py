import numpy as np
import scipy.sparse as scsp
import time

def Lap1DNeu(N, ordre):
    """
    Constructs the 1D discrete Laplacian matrix with Neumann boundary conditions.

    Parameters:
    -----------
    N : int
        Number of grid points in the 1D domain.
    ordre : int (1 or 2)
        Order of accuracy for the Neumann boundary condition approximation.
        - ordre=1: First-order approximation (standard finite difference).
        - ordre=2: Second-order approximation (higher accuracy).

    Returns:
    --------
    res : scipy.sparse.lil_matrix
        Sparse matrix representing the 1D discrete Laplacian with Neumann boundary conditions.
    """
    # Initialize the Laplacian matrix as a sparse matrix in LIL format for efficient construction
    # The standard Laplacian is constructed using the identity matrix shifted by ±1 and scaled
    res = scsp.lil_matrix(scsp.eye(N, k=-1) + scsp.eye(N, k=1) - 2 * scsp.eye(N))

    # Apply Neumann boundary conditions based on the specified order of accuracy
    if ordre == 2:
        # Second-order approximation: Adjust the first and last rows for Neumann BCs
        res[0, 0] = -2. / 3.
        res[0, 1] = 2. / 3.
        res[-1, -2] = 2. / 3.
        res[-1, -1] = -2. / 3.
    if ordre == 1:
        # First-order approximation: Adjust the first and last rows for Neumann BCs
        res[0, 0] = -1
        res[0, 1] = 1
        res[-1, -2] = 1
        res[-1, -1] = -1

    return res

def Lap2DNeu(Nx, Ny, alpha=1):
    """
    Constructs the 2D discrete Laplacian matrix with Neumann boundary conditions.

    Parameters:
    -----------
    Nx, Ny : int
        Number of grid points in the x and y directions, respectively.
    alpha : float, optional (default=1)
        Scaling factor for the y-direction Laplacian term.

    Returns:
    --------
    res : scipy.sparse.lil_matrix
        Sparse matrix representing the 2D discrete Laplacian with Neumann boundary conditions.
    """
    # Construct identity matrices for Kronecker product operations
    Idx = scsp.lil_matrix(scsp.eye(Nx))
    Idy = scsp.lil_matrix(scsp.eye(Ny))

    # Construct 1D Laplacians for x and y directions
    Lx = Lap1DNeu(Nx, ordre=1)
    Ly = Lap1DNeu(Ny, ordre=1)

    # Compute the 2D Laplacian using Kronecker products to combine 1D Laplacians
    resx = scsp.kron(Lx, Idy)
    resy = (1 / alpha**2) * scsp.kron(Idx, Ly)

    # Sum the x and y contributions to form the 2D Laplacian
    return resx + resy

def Lap3DNeu(Nx, Ny, Nz, alpha1=1, alpha2=1, ordre=1):
    """
    Constructs the 3D discrete Laplacian matrix with Neumann boundary conditions.

    Parameters:
    -----------
    Nx, Ny, Nz : int
        Number of grid points in the x, y, and z directions, respectively.
    alpha1, alpha2 : float, optional (default=1)
        Scaling factors for the y and z-direction Laplacian terms, respectively.
    ordre : int (1 or 2), optional (default=1)
        Order of accuracy for the Neumann boundary condition approximation.

    Returns:
    --------
    res : scipy.sparse.lil_matrix
        Sparse matrix representing the 3D discrete Laplacian with Neumann boundary conditions.
    """
    # Construct identity matrices for Kronecker product operations
    Idx = scsp.lil_matrix(scsp.eye(Nx))
    Idy = scsp.lil_matrix(scsp.eye(Ny))
    Idz = scsp.lil_matrix(scsp.eye(Nz))

    # Construct 1D Laplacians for x, y, and z directions
    Lx = Lap1DNeu(Nx, ordre)
    Ly = Lap1DNeu(Ny, ordre)
    Lz = Lap1DNeu(Nz, ordre)

    # Compute the 3D Laplacian using Kronecker products to combine 1D Laplacians
    resx = scsp.kron(scsp.kron(Lx, Idy), Idz)
    resy = (1 / alpha1**2) * scsp.kron(scsp.kron(Idx, Ly), Idz)
    resz = (1 / alpha2**2) * scsp.kron(Idx, scsp.kron(Idy, Lz))

    # Sum the x, y, and z contributions to form the 3D Laplacian
    return resx + resy + resz