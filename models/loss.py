import torch.nn as nn
from torch.nn.functional import interpolate
import torch
import scipy.sparse as scsp
import torch.nn.functional as F
import time
import SimpleITK as sitk
import numpy as np

class MSELossDS(nn.Module):
    def __init__(self, size = 5, sigma = 3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.MSELoss(reduction='sum')
        self.size = size
        self.sigma = sigma

    def forward(self, outputs, gt, mask=None):
        factor = 1
        loss = 0
        mask_given = False
        if mask != None:
            mask[mask != 0] = 1
            mask = apply_gaussian_blur_3d(mask, self.size, self.sigma)
            mask_given = True
        for output in outputs:
            if not mask_given:
                mask = torch.ones_like(output)
            else :       
                mask = interpolate(mask, size=output.shape[-3:])
                if torch.sum(mask) == 0:
                    mask = torch.ones_like(output)
            gt = interpolate(gt, size=output.shape[-3:])
            output1 = mask*output
            loss += factor*self.loss(output1, mask*gt)/torch.sum(mask)
            factor /= 2
        return loss
    
class MSELoss(nn.Module):
    def __init__(self, size=5, sigma=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.MSELoss(reduction='sum')
        self.size = size
        self.sigma = sigma

    def forward(self, output, gt, mask=None):
        if mask != None:
            mask[mask != 0] = 1
            mask = apply_gaussian_blur_3d(mask, self.size, self.sigma)
        else :
            mask = torch.ones_like(output)

        
        loss = self.loss(mask*output, mask*gt)/torch.sum(mask)
        return loss


class LossUResidu(nn.Module):
    def __init__(self, weight_residu, device, ordre=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight_residu = weight_residu
        self.device = device
        self.init_A = []
        self.ordre = ordre
        self.epsilon_needle = 1e-6
        self.A = []

    def forward(self, output, gt, input):
        if gt.shape[-3:] not in self.init_A:
            self.A.append(self.initialize_A(gt.shape[-3:], self.ordre))
            self.init_A.append(gt.shape[-3:])
        
        loss = torch.zeros(input.shape, device=self.device)
        for i, image in enumerate(input):
            
            g = (image[0]==1).float()
            f = ((image[0]==-1) + g).float()
            
            conductivity = torch.ones_like(g)
            A, b = self.compute_A_b(g, f, self.A, conductivity, self.epsilon_needle, gt.shape[-3:])
            loss_residu = torch.abs((A@torch.reshape(output[i,0],(-1, 1)) - b))*(1-torch.reshape(f, (-1, 1)))
            loss[i] += (self.weight_residu*loss_residu).reshape(loss[i].shape)
        return torch.mean(loss)
    
    def initialize_A(self, size, ordre):
        (nx, ny, nz) = size

        # Precalculation
        dx = 2 / (nx - 1)  # Width of space step(x)
        dy = 2 / (ny - 1)  # Width of space step(y)
        dz = 2 / (nz - 1)  # Width of space step(z)

        A = -self.Lap3DNeu(nx, ny, nz, alpha1=dy/dx, alpha2 = dz/dx, ordre=ordre)
        crow_indices = torch.tensor(A.indptr, dtype=torch.int64)
        col_indices = torch.tensor(A.indices, dtype=torch.int64)
        values = torch.tensor(A.data, dtype=torch.float32)

        # Step 3: Create a PyTorch sparse tensor in CSR format
        A_torch = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=A.shape).to(self.device)
        
        return A_torch
    
    def compute_A_b(self, g, f, A, conductivity, epsilon_needle, size):
        (nx, ny, nz) = size

        # Precalculation
        dx = 2 / (nx - 1)  # Width of space step(x)

        dx2 = dx**2

        f_div_eps = f*dx2/epsilon_needle/conductivity
        gf_div_eps = torch.multiply(g, f_div_eps)    
        gf_div_eps = torch.reshape(gf_div_eps, (nx * ny * nz, 1))
        
        b = gf_div_eps.to(self.device)
        diagonal_elements = torch.flatten(f_div_eps)
        crow_indices = torch.tensor([i for i in range(nx*ny*nz+1)], dtype=torch.int64).to(self.device)
        col_indices = torch.tensor([i for i in range(nx*ny*nz)], dtype=torch.int64).to(self.device)
        index = self.init_A.index(g.shape[-3:])
        A_full = A[index] + torch.sparse_csr_tensor(crow_indices, col_indices, diagonal_elements).to(self.device)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):

                    index = k * nx * ny + j * nx + i
        
                    # Neumann boundary conditions
                    if i == 0:
                        b[index] = 0
                    elif j==0:
                        b[index] = 0
                    elif k==0:
                        b[index] = 0
                    elif i==nx-1:
                        b[index] = 0
                    elif j==ny-1:
                        b[index] = 0
                    elif k==nz-1:
                        b[index] = 0
        max = torch.max(A_full.to_dense())
        return (A_full.to_dense()/max).to_sparse_csr(), b/max
    
    def Lap3DNeu(self, Nx,Ny,Nz,alpha1=1,alpha2=1, ordre=1):
        Idx=scsp.eye(Nx)
        Idy=scsp.eye(Ny)
        Idz=scsp.eye(Nz)
        Lx=self.Lap1DNeu(Nx, ordre)
        Ly=self.Lap1DNeu(Ny, ordre)
        Lz=self.Lap1DNeu(Nz, ordre)
        resx=scsp.kron(scsp.kron(Lx,Idy),Idz)
        resy=1/alpha1**2*scsp.kron(scsp.kron(Idx,Ly),Idz)
        resz=1/alpha2**2*scsp.kron(Idx,scsp.kron(Idy,Lz))
        return resx+resy+resz
    
    def Lap1DNeu(self, N, ordre):
        res=scsp.eye(N,k=-1)+scsp.eye(N,k=1)-2*scsp.eye(N)
        if ordre == 2:
            res[0,0]=-2./3.
            res[0,1]=2./3.
            res[-1,-2]=2./3.
            res[-1,-1]=-2./3.
        if ordre == 1: 
            res[0,0]=-1
            res[0,1]=1
            res[-1,-2]=1
            res[-1,-1]=-1
        return res
    
class LossUResiduDS(LossUResidu):
    def forward(self, outputs, gt, input):
        factor = 1
        loss = 0
        for output in outputs:
            gt = interpolate(gt, size=output.shape[-3:])
            input = interpolate(input, size=output.shape[-3:])
            loss += factor*super().forward(output, gt, input)
            factor /= 2
        return loss

    
class L1LossDS(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.L1Loss()

    def forward(self, outputs, gt):
        factor = 1
        loss = 0
        for output in outputs:
            gt = interpolate(gt, size=output.shape[-3:])
            loss += factor*self.loss(output, gt)
            factor /= 2
        return loss
    
class LossGradientU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.MSELoss()
    
    def forward(self, outputs, gt):
        if type(outputs) is list:
            outputs = outputs[0]
        return self.loss(grad(outputs), grad(gt))

class LnLoss(nn.Module):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.L1Loss(reduction='none')
        self.n = n
    
    def forward(self, outputs, gt, mask):
        if type(outputs) is list:
            error = self.loss(outputs[0], gt)
        else:
            error = self.loss(outputs, gt)
        error = torch.pow(error, self.n)
        return torch.mean(error)

def grad(u):
    shape = u.shape
    Ix = torch.zeros(shape)
    Iy = torch.zeros(shape)
    Iz = torch.zeros(shape)

    Ix[:,:,1:-1,:,:] = (u[:,:,2:,:,:] - u[:,:,:-2,:,:]) / 2.0
    Iy[:,:,:,1:-1,:] = (u[:,:,:,2:,:] - u[:,:,:,:-2,:]) / 2.0
    Iz[:,:,:,:,1:-1] = (u[:,:,:,:,2:] - u[:,:,:,:,0:-2]) / 2.0
    
    Ix[:,:,0, :, :] = (4*u[:,:,1, :, :]-3*u[:,:,0, :, :]-u[:,:,2, :, :])/2
    Ix[:,:,-1, :, :] = (-4*u[:,:,-2, :, :]+3*u[:,:,-1, :, :]+u[:,:,-3, :, :])/2
    Iy[:,:,:, 0, :] = (4*u[:,:,:, 1, :]-3*u[:,:,:, 0, :]-u[:,:,:, 2, :])/2
    Iy[:,:,:, -1, :] = (-4*u[:,:,:, -2, :]+3*u[:,:,:, -1, :]+u[:,:,:, -3, :])/2
    Iz[:,:,:, :, 0] = (4*u[:,:,:, :, 1]-3*u[:,:,:, :, 0]-u[:,:,:, :, 2])/2
    Iz[:,:,:, :, -1] = (-4*u[:,:,:, :, -2]+3*u[:,:,:, :, -1]+u[:,:,:, :, -3])/2

    return torch.concat([Ix,Iy,Iz], dim=1)

def compute_implicit_matrix(g, f):
    nx = g.shape[-1]  # Number of steps in space(x)
    ny = nx  # Number of steps in space(y)
    nz = nx  # Number of steps in space(z)

    dx = 2 / (nx - 1)  # Width of space step(x)
    dy = 2 / (ny - 1)  # Width of space step(y)
    dz = 2 / (nz - 1)  # Width of space step(z)

    epsilon_needle = (1e-6)**2

    # Precalculation
    dx2inv = 1.0 / (dx**2)
    dy2inv = 1.0 / (dy**2)
    dz2inv = 1.0 / (dz**2)
    
    dx2dy2dz2inv=-2.0*(dx2inv + dy2inv + dz2inv)    
    f_div_eps = f/epsilon_needle
    gf_div_eps = g *f_div_eps
    
    f_div_eps = torch.flatten(f_div_eps, start_dim=2)
    gf_div_eps = torch.flatten(gf_div_eps, start_dim=2)
    
    A = torch.zeros((g.shape[0], g.shape[1],nx * ny * nz, nx * ny * nz))

    # Remplissage du vecteur b
    b = -gf_div_eps

    # Remplissage de la matrice A
    for k in range(1, nz-1):
        for j in range(1, nz-1):
            for i in range(1, nz-1):
                index = k * nx * ny + j * nx + i
                A[:,:,index, index] = dx2dy2dz2inv - f_div_eps[:,:,index]
                
                if i - 1 >= 0:
                    A[:,:, index, index - 1] = dx2inv
                if i + 1 <= nx - 1:
                    A[:,:, index, index + 1] = dx2inv
                if j - 1 >= 0:
                    A[:,:,index, index - nx] = dy2inv
                if j + 1 <= ny - 1:
                    A[:,:,index, index + nx] = dy2inv
                if k - 1 >= 0:
                    A[:,:,index, index - nx * ny] = dz2inv
                if k + 1 <= nz - 1:
                    A[:,:,index, index + nx * ny] = dz2inv

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):

                index = k * nx * ny + j * nx + i
    
                # Neumann boundary conditions
                if i == 0:
                    A[:,:,index, index ] = 1
                    if i + 1 <= nx - 1:
                        A[:,:,index, index + 1 ] = -1
                    b[:,:,index] = 0
                elif j==0:
                    A[:,:,index, index ] = 1
                    if j + 1 <= ny - 1:
                        A[:,:,index, index + nx ] = -1
                    b[:,:,index] = 0
                elif k==0:
                    A[:,:,index, index ] = 1
                    if k + 1 <= nz - 1:
                        A[:,:,index, index + nx * ny ] = -1
                    b[:,:,index] = 0
                elif i==nx-1:
                    A[:,:,index, index ] = 1
                    if i - 1 >= 0:
                        A[:,:,index, index - 1 ] = -1
                    b[:,:,index] = 0
                elif j==ny-1:
                    A[:,:,index, index ] = 1
                    if j - 1 >= 0:
                        A[:,:,index, index - nx ] = -1
                    b[:,:,index] = 0
                elif k==nz-1:
                    A[:,:,index, index ] = 1
                    if k - 1 >= 0:
                        A[:,:,index, index - nx * ny ] = -1
                    b[:,:,index] = 0

    return A, b

def create_gaussian_kernel_3d(size: int, sigma: float, dtype=torch.float32, device='cpu'):
    """
    Creates a 3D Gaussian kernel with the specified size, sigma, data type, and device.
    Args:
        size (int): The size of the kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian.
        dtype (torch.dtype): The desired data type for the kernel (float32 or float16).
        device (str): The device on which to create the kernel (e.g., 'cpu' or 'cuda').
    Returns:
        kernel (torch.Tensor): The 3D Gaussian kernel of shape (1, 1, size, size, size).
    """
    # Create a 3D grid of (x, y, z) coordinates
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    
    # Compute the Gaussian function
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    
    # Normalize the kernel to sum to 1
    kernel = kernel / kernel.sum()
    
    # Reshape to (1, 1, size, size, size) for convolution
    kernel = kernel.view(1, 1, size, size, size)
    return kernel

def apply_gaussian_blur_3d(input_tensor: torch.Tensor, kernel_size: int, sigma: float):
    """
    Applies a 3D Gaussian blur to a 5D input tensor using the specified kernel size and sigma.
    Supports both full and half precision (float16).
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, C, D, H, W).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian kernel.
    Returns:
        torch.Tensor: The blurred 3D tensor.
    """
    # Determine the data type (float32 or float16) based on input_tensor
    dtype = input_tensor.dtype
    device = input_tensor.device

    # Create Gaussian kernel with the same dtype and device as the input tensor
    kernel = create_gaussian_kernel_3d(kernel_size, sigma, dtype=dtype, device=device)
    
    # Expand the kernel to apply to all input channels
    kernel = kernel.repeat(input_tensor.size(1), 1, 1, 1, 1)
    
    # Apply the kernel using 3D convolution (groups=input_tensor.size(1) to apply to each channel independently)
    blurred_tensor = F.conv3d(input_tensor, kernel, padding=kernel_size // 2, groups=input_tensor.size(1))
    
    return blurred_tensor
