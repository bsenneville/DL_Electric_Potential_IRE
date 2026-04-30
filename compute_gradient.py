import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(image, dx=1, dy=1, dz=1):
    """
    Computes the gradient of a 2D or 3D image using central differences for interior points
    and forward/backward differences for boundary points.

    The gradient is a fundamental operation in image processing and computer vision,
    used for edge detection, feature extraction, and other applications. This function
    calculates the partial derivatives (Ix, Iy, Iz) of the image intensity function.

    Parameters:
        image (numpy.ndarray): Input image (2D or 3D).
        dx, dy, dz (float): Spatial step sizes for each dimension (default: 1).

    Returns:
        tuple: Gradient components (Ix, Iy) for 2D images or (Ix, Iy, Iz) for 3D images.
    """

    shape = image.shape
    is_2d = len(shape) == 2  # Check if the image is 2D or 3D

    # Initialize gradient components with zeros
    Ix = np.zeros(shape, dtype='float32', order='F')
    Iy = np.zeros(shape, dtype='float32', order='F')

    if is_2d:
        # Central difference for interior points (x-direction)
        Ix[1:-1, :] = (image[2:, :] - image[:-2, :]) / (2.0 * dx)
        # Central difference for interior points (y-direction)
        Iy[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / (2.0 * dy)

        # Forward/backward difference for boundary points (x-direction)
        Ix[0, :] = (4 * image[1, :] - 3 * image[0, :] - image[2, :]) / (2 * dx)
        Ix[-1, :] = (-4 * image[-2, :] + 3 * image[-1, :] + image[-3, :]) / (2 * dx)
        # Forward/backward difference for boundary points (y-direction)
        Iy[:, 0] = (4 * image[:, 1] - 3 * image[:, 0] - image[:, 2]) / (2 * dy)
        Iy[:, -1] = (-4 * image[:, -2] + 3 * image[:, -1] + image[:, -3]) / (2 * dy)

        return Ix, Iy

    else:
        # Initialize gradient component for z-direction (3D case)
        Iz = np.zeros(shape, dtype='float32', order='F')

        # Central difference for interior points (x, y, z directions)
        Ix[1:-1, :, :] = (image[2:, :, :] - image[:-2, :, :]) / (2.0 * dx)
        Iy[:, 1:-1, :] = (image[:, 2:, :] - image[:, :-2, :]) / (2.0 * dy)
        Iz[:, :, 1:-1] = (image[:, :, 2:] - image[:, :, :-2]) / (2.0 * dz)

        # Forward/backward difference for boundary points (x, y, z directions)
        Ix[0, :, :] = (4 * image[1, :, :] - 3 * image[0, :, :] - image[2, :, :]) / (2 * dx)
        Ix[-1, :, :] = (-4 * image[-2, :, :] + 3 * image[-1, :, :] + image[-3, :, :]) / (2 * dx)
        Iy[:, 0, :] = (4 * image[:, 1, :] - 3 * image[:, 0, :] - image[:, 2, :]) / (2 * dy)
        Iy[:, -1, :] = (-4 * image[:, -2, :] + 3 * image[:, -1, :] + image[:, -3, :]) / (2 * dy)
        Iz[:, :, 0] = (4 * image[:, :, 1] - 3 * image[:, :, 0] - image[:, :, 2]) / (2 * dz)
        Iz[:, :, -1] = (-4 * image[:, :, -2] + 3 * image[:, :, -1] + image[:, :, -3]) / (2 * dz)

        return Ix, Iy, Iz

def neumann_boundary_2d(I):
    """
    Applies Neumann boundary conditions to a 2D image.
    Neumann boundary conditions assume that the derivative at the boundary is zero,
    which is equivalent to mirroring the values at the boundary.

    Parameters:
        I (numpy.ndarray): Input 2D image.

    Returns:
        numpy.ndarray: Image with Neumann boundary conditions applied.
    """

    # Mirror the boundary values for Neumann boundary conditions
    I[0, :] = I[1, :]
    I[-1, :] = I[-2, :]
    I[:, 0] = I[:, 1]
    I[:, -1] = I[:, -2]

    # Handle corner points explicitly
    I[0, 0] = I[1, 1]
    I[-1, 0] = I[-2, 1]
    I[0, -1] = I[1, -2]
    I[-1, -1] = I[-2, -2]

    return I

def dirichlet_boundary_2d(I):
    """
    Applies Dirichlet boundary conditions to a 2D image.
    Dirichlet boundary conditions set the boundary values to zero,
    which is useful for certain numerical simulations and image processing tasks.

    Parameters:
        I (numpy.ndarray): Input 2D image.

    Returns:
        numpy.ndarray: Image with Dirichlet boundary conditions applied.
    """

    # Set boundary values to zero for Dirichlet boundary conditions
    I[0, :] = 0
    I[-1, :] = 0
    I[:, 0] = 0
    I[:, -1] = 0

    return I

def neumann_boundary_3d(I):
    """
    Applies Neumann boundary conditions to a 3D image.
    Neumann boundary conditions assume that the derivative at the boundary is zero,
    which is equivalent to mirroring the values at the boundary.

    Parameters:
        I (numpy.ndarray): Input 3D image.

    Returns:
        numpy.ndarray: Image with Neumann boundary conditions applied.
    """

    # Mirror the boundary values for Neumann boundary conditions (3D)
    I[0, :, :] = I[1, :, :]
    I[-1, :, :] = I[-2, :, :]
    I[:, 0, :] = I[:, 1, :]
    I[:, -1, :] = I[:, -2, :]
    I[:, :, 0] = I[:, :, 1]
    I[:, :, -1] = I[:, :, -2]

    # Handle corner points explicitly (3D)
    I[0, 0, 0] = I[1, 1, 1]
    I[-1, 0, 0] = I[-2, 1, 1]
    I[0, -1, 0] = I[1, -2, 1]
    I[-1, -1, 0] = I[-2, -2, 1]

    I[0, 0, -1] = I[1, 1, -2]
    I[-1, 0, -1] = I[-2, 1, -2]
    I[0, -1, -1] = I[1, -2, -2]
    I[-1, -1, -1] = I[-2, -2, -2]

    return I

def dirichlet_boundary_3d(I):
    """
    Applies Dirichlet boundary conditions to a 3D image.
    Dirichlet boundary conditions set the boundary values to zero,
    which is useful for certain numerical simulations and image processing tasks.

    Parameters:
        I (numpy.ndarray): Input 3D image.

    Returns:
        numpy.ndarray: Image with Dirichlet boundary conditions applied.
    """

    # Set boundary values to zero for Dirichlet boundary conditions (3D)
    I[0, :, :] = 0
    I[-1, :, :] = 0
    I[:, 0, :] = 0
    I[:, -1, :] = 0
    I[:, :, 0] = 0
    I[:, :, -1] = 0

    return I