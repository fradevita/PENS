import os
import sys
import math
import numpy as np
import scipy.fft

sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Cartesian
import Field

# Solver function for 2-dimensional Poisson equation with periodic/periodic boundary condition
# on grid Grid
def solve_PP(rhs: Field.scalar):

    # Set problem size
    Nx, Ny, dx = rhs.G.Nx, rhs.G.Ny, rhs.G.dx

    # Wavenumber and modified wave number in x
    kx = np.linspace(0, Nx - 1, Nx)
    mwn_x = 2.*(np.cos(2.*np.pi*kx/Nx) - 1.)/dx**2

    # Wavenumber and Modified wave number in y
    ky = np.linspace(0, Ny - 1, Ny)
    mwn_y = 2.*(np.cos(2.*np.pi*ky/Ny) - 1.)/dx**2

    # Perform FFT of the RHS
    fft_rhs = np.fft.fft2(rhs.f[rhs.gl:rhs.sy - rhs.gl, rhs.gl: rhs.sx - rhs.gl])

    # Poisson Equtaion
    MWN_X, MWN_Y = np.meshgrid(mwn_x, mwn_y)
    fft_rhs[1:,1:] = fft_rhs[1:,1:]/(MWN_X[1:,1:] + MWN_Y[1:,1:])
    fft_rhs[0,0] = 0.0 # avoid division by zero

    # Transform back to find the solution
    return np.fft.ifft2(fft_rhs)

# Solver function for 2-dimensional Poisson equation with neumann boundary condition
# in all direction
def solve_NN(rhs: Field.scalar):

    # Set problem size
    Nx, Ny, dx = rhs.G.Nx, rhs.G.Ny, rhs.G.dx

    # Wavenumber and modified wave number in x
    kx = np.linspace(0, Nx - 1, Nx)
    mwn_x = 2.*(np.cos(1.*np.pi*kx/Nx) - 1.)/dx**2

    # Wavenumber and Modified wave number in y
    ky = np.linspace(0, Ny - 1, Ny)
    mwn_y = 2.*(np.cos(1.*np.pi*ky/Ny) - 1.)/dx**2

    # Perform DCT of the RHS
    out = scipy.fft.dctn(rhs.f[rhs.gl:rhs.sy - rhs.gl, rhs.gl: rhs.sx - rhs.gl])

    # Divide rhs by modified wave numbers
    MWN_X, MWN_Y = np.meshgrid(mwn_x, mwn_y)
    out[1:,1:] = out[1:,1:]/(MWN_X[1:,1:] + MWN_Y[1:,1:])
    out[0,0] = 0.0 # avoid difision by zero

    return scipy.fft.idctn(out)

def solve_PN(rhs: Field.scalar):

    # Set problem size
    Nx, Ny, dx = rhs.G.Nx, rhs.G.Ny, rhs.G.dx

    # Wavenumber and modified wave number in x
    kx = np.linspace(0, Nx - 1, Nx)
    mwn_x = 2.*(np.cos(2.*np.pi*kx/Nx) - 1.)/dx**2

    # Wavenumber and Modified wave number in y
    ky = np.linspace(0, Ny - 1, Ny)
    mwn_y = 2.*(np.cos(1.*np.pi*ky/Ny) - 1.)/dx**2

    # Perform FFT of the RHS
    rfft_rhs = scipy.fft.dctn(rhs.f[rhs.gl:rhs.sy - rhs.gl, rhs.gl: rhs.sx - rhs.gl], axes = 0)
    cfft_rhs = scipy.fft.fftn(rfft_rhs, axes = 1)
    
    # Poisson Equtaion
    MWN_X, MWN_Y = np.meshgrid(mwn_x, mwn_y)
    cfft_rhs[1:,1:] = cfft_rhs[1:,1:]/(MWN_X[1:,1:] + MWN_Y[1:,1:])
    cfft_rhs[0,0] = 0.0 # avoid division by zero

    # Transform back to find the solution
    ifft_rhs = scipy.fft.ifftn(cfft_rhs, axes = 1)
    ifft_rhs = scipy.fft.idctn(ifft_rhs, axes = 0)
    return ifft_rhs