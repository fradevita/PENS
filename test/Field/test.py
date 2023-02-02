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
    Grid = Cartesian.Grid(int(N), int(N), 1.0, 1.0, [0.0, 0.0])

    # Create a Scalar field
    import Field
    S = Field.scalar({'name': 'S', 'Grid': Grid, 'gl': 1})

    # Set S values
    S0 = np.zeros((Grid.Ny,Grid.Nx))
    for j in range(Grid.Ny):
        for i in range(Grid.Nx):
            S0[j,i] = math.sin(2.*math.pi*Grid.x[i]) - math.cos(2.*math.pi*Grid.y[j])
    S.set_field(S0)

    # Evalute S gradient
    gradS = Field.vector({'name': 'gradS', 'Grid': Grid, 'gl': 0})
    gradS.x.f, gradS.y.f = Field.Gradient(S)

    # Compare with analytical expression
    dS0dx = np.zeros((Grid.Ny,Grid.Nx))
    dS0dy = np.zeros((Grid.Ny,Grid.Nx))
    for j in range(Grid.Ny):
        for i in range(Grid.Nx):
            dS0dx[j,i] = 2.*math.pi*math.cos(2.*math.pi*(Grid.x[i] + 0.5*Grid.dx))
            dS0dy[j,i] = 2.*math.pi*math.sin(2.*math.pi*(Grid.y[j] + 0.5*Grid.dx))

    Linf_gradx[count] = np.amax(abs(dS0dx - gradS.x.f))
    Linf_grady[count] = np.amax(abs(dS0dy - gradS.y.f))

    # Evaluate laplacian of S
    lapS = Field.Laplacian(S)

    # Compare with analytical expression
    lapS0 = np.zeros((Grid.Ny,Grid.Nx))
    for j in range(Grid.Ny):
        for i in range(Grid.Nx):
            lapS0[j,i] = -(2.*math.pi)**2*math.sin(2.*math.pi*(Grid.x[i])) + \
                            (2.*math.pi)**2*math.cos(2.*math.pi*Grid.y[j])
    Linf_lapl[count] = np.amax(abs(lapS0 - lapS))

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