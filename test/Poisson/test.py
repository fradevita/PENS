import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Load PENS 
sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Poisson_2D

# Perform grid convergence over the following resolution values
resolutions = np.array([8, 16, 32, 64, 128, 256])

Linf = np.zeros(len(resolutions))
L2 = np.zeros(len(resolutions))
c = 0
for N in resolutions:

    # Define the numerical grid
    L = 1.0
    dx = L/N
    dy = L/N
    x = np.linspace(0, L - dx, N)
    y = np.linspace(0, L - dy, N)
    
    # Define target solution and equation RHS
    f = np.zeros((N,N))
    rhs = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            f[j,i] = math.sin(2.*math.pi*x[i])*math.cos(2.*math.pi*y[j])
            rhs[j,i] = -8.*math.pi*math.pi*math.sin(2.*math.pi*x[i])*math.cos(2.*math.pi*y[j])

    # Solve Poisson equation
    sol = Poisson_2D.solve(rhs, dx)

    # Evaluate errors
    e = abs(f - sol)
    Linf[c] = e.max()
    L2[c] =  e.std()
    c = c + 1

# Scaling lines for the plot
scaling1 = 0.5/resolutions
scaling2 = 5/resolutions**2

plt.figure()
plt.xlabel(r'$N$')
plt.ylabel(r'$|e|$')
plt.title('2D FFT Poisson solver convergence rate') 
plt.loglog(resolutions,     Linf,                   'o', label = r'$L_{\infty}$')
plt.loglog(resolutions,       L2,                   's', label = r'$L_{2}$')
plt.loglog(resolutions, scaling1,  '-', color = 'black', label = r'$1/N$')
plt.loglog(resolutions, scaling2, '--', color = 'black', label = r'$1/N^2$')
plt.grid()
plt.legend()
plt.show()
plt.close()