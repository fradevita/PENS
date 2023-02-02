import math
import numpy as np

# Solver function for 2-dimensional Poisson equation with double periodic boundary condition
def solve(rhs, dx):

    # Set problem size
    Ny, Nx = np.shape(rhs)

    # Modified wave number in x
    mwn_x = np.zeros(Nx)
    for i in range(Nx):
        mwn_x[i] = 2.*(math.cos(2.*math.pi*i/Nx) - 1.)/dx**2

    # Modified wave number in x
    mwn_y = np.zeros(Ny)
    for i in range(Ny):
        mwn_y[i] = 2.*(math.cos(2.*math.pi*i/Ny) - 1.)/dx**2

    # Perform FFT of the RHS
    fft_rhs = np.fft.fft2(rhs)

    # Poisson Equtaion    
    fft_rhs[1:,1:] = fft_rhs[1:,1:]/(mwn_x[1:,None] + mwn_y[None,1:])
    fft_rhs[0,0] = 0.0

    # Transform back to find the solution
    return np.fft.ifft2(fft_rhs)