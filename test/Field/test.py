import os
import sys
import math
import numpy as np

# Load PENS 
sys.path.insert(0, os.path.expandvars(os.environ['PENS']))

# Check gradient and laplacian computation
resolutions = np.array([16, 32, 64, 128])
Linf_gradx = np.zeros(len(resolutions))
Linf_grady = np.zeros(len(resolutions))
Linf_lapl = np.zeros(len(resolutions))
for count, N in enumerate(resolutions):

    # Create a Cartesian Grid
    import Cartesian
    Grid = Cartesian.Grid(int(N), int(N), 1.0, 1.0, [0.0, 0.0], ['Periodic', 'Periodic', 'Periodic', 'Periodic'])

    # Create a Scalar field
    import Field
    S = Field.scalar({'name': 'S', 'Grid': Grid, 'gl': 1})
    
    # Set S values
    S0 = np.zeros((Grid.Nx,Grid.Ny))
    for j in range(Grid.Ny):
        for i in range(Grid.Nx):
            S0[i,j] = math.sin(2.*math.pi*Grid.x[i]) - math.cos(2.*math.pi*Grid.y[j])
    S.set_field(S0)
    S.update_ghost_nodes()

    # Evalute S gradient
    gradS = Field.Gradient(S)

    # Compare with analytical expression
    dS0dx = np.zeros((Grid.Nx,Grid.Ny), order = 'F')
    dS0dy = np.zeros((Grid.Nx,Grid.Ny), order = 'F')
    for j in range(Grid.Ny):
        for i in range(Grid.Nx):
            dS0dx[i,j] = 2.*math.pi*math.cos(2.*math.pi*(Grid.x[i] + Grid.dx/2))
            dS0dy[i,j] = 2.*math.pi*math.sin(2.*math.pi*(Grid.y[j] + Grid.dx/2))

    Linf_gradx[count] = np.amax(abs(dS0dx - gradS.x.f[1:-1,1:-1]))
    Linf_grady[count] = np.amax(abs(dS0dy - gradS.y.f[1:-1,1:-1]))

    # Evaluate laplacian of S
    lapS = Field.Laplacian(S)

    # Compare with analytical expression
    lapS0 = np.zeros((Grid.Nx,Grid.Ny), order ='F')
    for j in range(Grid.Ny):
        for i in range(Grid.Nx):
            lapS0[i,j] = -(2.*math.pi)**2*math.sin(2.*math.pi*(Grid.x[i])) + \
                            (2.*math.pi)**2*math.cos(2.*math.pi*Grid.y[j])
    Linf_lapl[count] = np.amax(abs(lapS0 - lapS.f))

# Evaluate scaling
from scipy.optimize import curve_fit

# Fit the error with a line in loglog plot
def objective(x, a, b):
  return a*x + b

pars, cov = curve_fit(f = objective, xdata = np.log(resolutions), ydata = np.log(Linf_gradx))
a, b = pars
pars, cov = curve_fit(f = objective, xdata = np.log(resolutions), ydata = np.log(Linf_grady))
c, d = pars
pars, cov = curve_fit(f = objective, xdata = np.log(resolutions), ydata = np.log(Linf_lapl))
e, f = pars

print('Infinity norm of error vector:')
print('gradient_x: ', -a)
print('gradient_x: ', -c)
print('laplacian : ', -e)

scaling1 = 1./resolutions
scaling2 = 10./resolutions**2

import matplotlib.pyplot as plt
plt.figure()
plt.loglog(resolutions, Linf_gradx, 'x')
plt.loglog(resolutions, Linf_grady, '+')
plt.loglog(resolutions, Linf_lapl , 'D', fillstyle = 'none')
plt.loglog(resolutions, scaling1,  '-', color = 'black', label = r'$1/N$')
plt.loglog(resolutions, scaling2, '--', color = 'black', label = r'$1/N^2$')
plt.grid()
plt.legend()
plt.show()
plt.close()