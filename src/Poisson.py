import os
import sys
import numpy as np
import scipy.fft

sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Cartesian
import Field
import Thomas

# Wrapper function for the Poisson solver
def solve(rhs: Field.scalar):

    if (rhs.G.bc == ['Periodic', 'Periodic', 'Periodic', 'Periodic']):
        return solve_PP(rhs)
    elif (rhs.G.bc == ['Periodic', 'Periodic', 'Wall', 'Wall']):
        return solve_PN(rhs)
    elif (rhs.G.bc == ['Wall', 'Wall', 'Wall', 'Wall']):
        return solve_NN(rhs)
    else:
        sys.exit('ERROR: cannot find proper Poisson solver for the given computational grid.')

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
    fft_rhs = scipy.fft.fft2(rhs.f[rhs.gl:rhs.sx - rhs.gl, rhs.gl: rhs.sy -rhs.gl])

    # Poisson Equtaion
    factor = mwn_x[:,None] + mwn_y[None,:]
    mfft_rhs = np.where(factor == 0., 0., fft_rhs/factor)

    # Transform back to find the solution
    return scipy.fft.ifftn(mfft_rhs).real

# Solver function for 2-dimensional Poisson equation with neumann boundary condition
# in all direction
def solve_NN(rhs: Field.scalar):

    # Set problem size
    Nx, Ny, dx = rhs.G.Nx, rhs.G.Ny, rhs.G.dx

    # Wavenumber and modified wave number in x
    kx = np.linspace(0, Nx - 1, Nx)
    mwn_x = 2.*(np.cos(1.*np.pi*kx/Nx) - 1.)/dx**2

    # Perform DCT of the RHS in x
    fft_rhs = np.zeros((Nx,Ny))
    for j in range(Ny):
        fft_rhs[:,j] = scipy.fft.dct(rhs.f[1:-1, j+1])

    # Tridiagonal system coefficients
    a = np.ones(Ny)/dx**2
    b = -2.*np.ones(Ny)/dx**2
    c = np.ones(Ny)/dx**2
    b[0] = b[0] + a[0]
    b[-1] = b[-1] + c[-1]
    a[0] = 0.
    c[-1] = 0.
    c1 = np.zeros((Nx,Ny))
    d1 = np.zeros((Nx,Ny))
    out=np.zeros((Nx,Ny))
    
    # For every wavenumber kx solve a tridiagonal system in y
    # for i in range(Nx):
    #     c1[i,0] = c[0]/(b[0] + mwn_x[i]) 
    #     d1[i,0] = fft_rhs[i,0]/(b[0] + mwn_x[i])
    # for j in range(1,Ny-1):
    #     for i in range(Nx):
    #         c1[i,j] = c[j]/(b[j] + mwn_x[i] - a[j]*c1[i,j-1])
    #         d1[i,j] = (fft_rhs[i,j] - a[j]*d1[i,j-1])/(b[j] + mwn_x[i] - a[j]*c1[i,j-1])
    # for i in range(Nx):
    #     frac = b[-1] + mwn_x[i] - a[-1]*c1[i,-2]
    #     if (frac == 0.):
    #         d1[i,-1] = 0.
    #     else:
    #         d1[i,-1] = (fft_rhs[i,-1] - a[-1]*d1[i,-2])/(b[-1] + mwn_x[i] - a[-1]*c1[i,-2])
    # for i in range(Nx):
    #     out[i,-1] = d1[i,-1]
    # for j in range(Ny-2,-1,-1):
    #     for i in range(Nx):
    #         out[i,j] = d1[i,j] - c1[i,j]*out[i,j+1]
    
    c1[:,0] = c[0]/(b[0] + mwn_x) 
    d1[:,0] = fft_rhs[:,0]/(b[0] + mwn_x)
    for j in range(1,Ny-1):
        c1[:,j] = c[j]/(b[j] + mwn_x - a[j]*c1[:,j-1])
        d1[:,j] = (fft_rhs[:,j] - a[j]*d1[:,j-1])/(b[j] + mwn_x - a[j]*c1[:,j-1])
    for i in range(Nx):
        frac = b[-1] + mwn_x[i] - a[-1]*c1[i,-2]
        if (frac == 0.):
            d1[i,-1] = 0.
        else:
            d1[i,-1] = (fft_rhs[i,-1] - a[-1]*d1[i,-2])/(b[-1] + mwn_x[i] - a[-1]*c1[i,-2])
    out[:,-1] = d1[:,-1]
    for j in range(Ny-2,-1,-1):
        out[:,j] = d1[:,j] - c1[:,j]*out[:,j+1]
    
    sol = np.zeros((Nx,Ny)) 
    for j in range(Ny):
        sol[:,j] = scipy.fft.idct(out[:,j])
    return sol - np.mean(sol)

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
    rfft_rhs = scipy.fft.dctn(rhs.f[rhs.gl:rhs.sx - rhs.gl, rhs.gl: rhs.sy - rhs.gl], axes = 0)
    cfft_rhs = scipy.fft.fftn(rfft_rhs, axes = 1)
    
    # Poisson Equtaion
    # MWN_X, MWN_Y = np.meshgrid(mwn_x, mwn_y)
    # cfft_rhs[1:,1:] = cfft_rhs[1:,1:]/(MWN_X[1:,1:] + MWN_Y[1:,1:])
    # cfft_rhs[0,0] = 0.0 # avoid division by zero

    # Poisson Equtaion
    factor = mwn_x[:,None] + mwn_y[None,:]
    mfft_rhs = np.where(factor == 0., 0., cfft_rhs/factor)

    # Transform back to find the solution
    ifft_rhs = scipy.fft.ifftn(mfft_rhs, axes = 1)
    ifft_rhs = scipy.fft.idctn(ifft_rhs, axes = 0)
    return np.real(ifft_rhs)
