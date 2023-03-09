"""
    Test for the compute_diffusion_fluxes function in the Navier Stokes module.
    We use the following velocity field

        u = 2*x**3 + y**4
        v = x**4 + 2*y**3

    Diffusion fluxes are:

        d2u/dx2 = 12*x
        d2u/dy2 = 12*y**2
        d2v/dx2 = 12*x**2
        d2v/dy2 = 12*y

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Load PENS modules
sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Cartesian
import Field
import Navier_Stokes

resolutions = np.array([8, 16, 32, 64, 128, 256, 512])

ex = []
ey = []

for N in resolutions:

    # Create a grid G
    G = Cartesian.Grid(N, N, 1., 1., [0.0, 0.0], ['Periodic', 'Periodic', 'Periodic', 'Periodic'])

    # Create a vector field V defined on G with one level of ghost nodes
    V = Field.vector({'name': 'V', 'Grid': G, 'gl': 1})

    # Initialize the vector field (remember the staggering)
    V.x.f = 2.*(G.xc + G.dx/2)**3 + G.yc**4
    V.y.f = 2.*(G.yc + G.dy/2)**3 + G.xc**4

    # Evaluate the advection term
    Diffusion = Navier_Stokes.compute_diffusion_fluxes(V)

    # Evaluate the analytical advection fluxes
    sol = Field.vector({'name': 'Va', 'Grid': G, 'gl': 0})
    sol.x.f = 12.*(G.xc[1:-1,1:-1] + G.dx/2) + 12.*G.yc[1:-1,1:-1]**2
    sol.y.f = 12.*(G.yc[1:-1,1:-1] + G.dy/2) + 12.*G.xc[1:-1,1:-1]**2

    ex.append(np.amax(np.abs(sol.x.f - Diffusion.x.f)))
    ey.append(np.amax(np.abs(sol.y.f - Diffusion.y.f)))

scaling_1st = 0.1/resolutions
scaling_2nd = 1./resolutions**2

plt.figure()
plt.xlabel(r'$N$')
plt.ylabel(r'$L_{\infty}(e)$')
plt.title('Convergence rate of the advection fluxes function')
plt.loglog(resolutions, ex, 'o', fillstyle = 'none')
plt.loglog(resolutions, ey, 'x', fillstyle = 'none')
plt.loglog(resolutions, scaling_1st,  '-k', label = r'$1/N$')
plt.loglog(resolutions, scaling_2nd, '-.k', label = r'$1/N^2$')
plt.grid()
plt.legend()
plt.show()
plt.close()