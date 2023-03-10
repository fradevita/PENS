import sys
import numpy as np
import scipy.fft
import Field

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

    # For every wavenumber kx solve a tridiagonal system in y
    # Tridiagonal system coefficients
    a = np.ones(Ny)/dx**2
    b = -2.*np.ones(Ny)/dx**2
    c = np.ones(Ny)/dx**2

    # Boundary conditions
    b[0] = b[0] + a[0]
    b[-1] = b[-1] + c[-1]
    a[0] = 0.
    c[-1] = 0.

    # Forward step
    c1 = np.zeros((Nx,Ny))
    d1 = np.zeros((Nx,Ny))
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

    # Backward step
    out=np.zeros((Nx,Ny))
    out[:,-1] = d1[:,-1]
    for j in range(Ny-2,-1,-1):
        out[:,j] = d1[:,j] - c1[:,j]*out[:,j+1]
    
    # Inverse DCT in x to find final solution
    sol = np.zeros((Nx,Ny)) 
    for j in range(Ny):
        sol[:,j] = scipy.fft.idct(out[:,j])
    return sol - np.mean(sol)

# Solver function for 2-dimensional Poisson equation with mixed periodic/neumann boundary 
# conditions
def solve_PN(rhs: Field.scalar):

    # Set problem size
    Nx, Ny, dx = rhs.G.Nx, rhs.G.Ny, rhs.G.dx

    # Wavenumber and modified wave number in x
    kx = np.linspace(0, Nx - 1, Nx)
    mwn_x = 2.*(np.cos(2.*np.pi*kx/Nx) - 1.)/dx**2

    # Perform FFT of the RHS in x direction
    fft_rhs = np.zeros((Nx,Ny), dtype = np.complex128)
    for j in range(Ny):
        fft_rhs[:,j] = scipy.fft.fft(rhs.f[1:-1, j+1])

    # For every wavenumber kx solve a tridiagonal system in y
    # Tridiagonal system coefficients
    a = np.ones(Ny)/dx**2
    b = -2.*np.ones(Ny)/dx**2
    c = np.ones(Ny)/dx**2

    # Boundary conditions (Neumann = 0 for now)
    b[0] = b[0] + a[0]
    b[-1] = b[-1] + c[-1]
    a[0] = 0.
    c[-1] = 0.

    # Forward step
    c1 = np.zeros_like(fft_rhs)
    d1 = np.zeros_like(fft_rhs)
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
    
    # Backward step
    out = np.zeros_like(fft_rhs)
    out[:,-1] = d1[:,-1]
    for j in range(Ny-2,-1,-1):
        out[:,j] = d1[:,j] - c1[:,j]*out[:,j+1]
    
    # Inverse fft in x direction to find solution
    sol = np.zeros_like(fft_rhs) 
    for j in range(Ny):
        sol[:,j] = scipy.fft.ifft(out[:,j])

    return sol.real