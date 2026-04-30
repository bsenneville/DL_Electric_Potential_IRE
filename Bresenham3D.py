import numpy as np
import random
import time
import os

def Bresenham3D(x1, y1, z1, x2, y2, z2):
    """
    Implements the 3D Bresenham's line algorithm to generate a sequence of points
    approximating a straight line between two 3D points (x1, y1, z1) and (x2, y2, z2).

    The algorithm extends the 2D Bresenham's line algorithm to 3D space by selecting
    the dominant axis (the axis with the largest difference) and incrementally adjusting
    the other axes based on decision parameters.

    Parameters:
        x1, y1, z1 (int): Starting coordinates of the line.
        x2, y2, z2 (int): Ending coordinates of the line.

    Returns:
        list: A list of 3D points (tuples) representing the line.
    """

    # List to store the sequence of 3D points approximating the line
    ListOfPoints = []
    # Add the starting point to the list
    ListOfPoints.append((x1, y1, z1))

    # Calculate the absolute differences between the coordinates
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    # Determine the direction of movement for each axis
    # xs, ys, zs are step directions: +1 for increment, -1 for decrement
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Case 1: X-axis is the dominant axis (dx is the largest difference)
    if (dx >= dy and dx >= dz):
        # Initialize decision parameters for Y and Z axes
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        # Loop until x1 reaches x2
        while (x1 != x2):
            x1 += xs  # Increment or decrement x1
            if (p1 >= 0):
                y1 += ys  # Update y1 if necessary
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs  # Update z1 if necessary
                p2 -= 2 * dx
            # Update decision parameters
            p1 += 2 * dy
            p2 += 2 * dz
            # Add the new point to the list
            ListOfPoints.append((x1, y1, z1))

    # Case 2: Y-axis is the dominant axis (dy is the largest difference)
    elif (dy >= dx and dy >= dz):
        # Initialize decision parameters for X and Z axes
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        # Loop until y1 reaches y2
        while (y1 != y2):
            y1 += ys  # Increment or decrement y1
            if (p1 >= 0):
                x1 += xs  # Update x1 if necessary
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs  # Update z1 if necessary
                p2 -= 2 * dy
            # Update decision parameters
            p1 += 2 * dx
            p2 += 2 * dz
            # Add the new point to the list
            ListOfPoints.append((x1, y1, z1))

    # Case 3: Z-axis is the dominant axis (dz is the largest difference)
    else:
        # Initialize decision parameters for X and Y axes
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        # Loop until z1 reaches z2
        while (z1 != z2):
            z1 += zs  # Increment or decrement z1
            if (p1 >= 0):
                y1 += ys  # Update y1 if necessary
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs  # Update x1 if necessary
                p2 -= 2 * dz
            # Update decision parameters
            p1 += 2 * dy
            p2 += 2 * dx
            # Add the new point to the list
            ListOfPoints.append((x1, y1, z1))

    # Return the list of 3D points approximating the line
    return ListOfPoints