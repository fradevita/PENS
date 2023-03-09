"""
    Test for the compute_advection_fluxes function in the Navier Stokes module.
    We use the following velocity field

        u = 2*x + y**2
        v = x**2 + 2*y

    The terms uu, uv and vv are

        uu = 4*x**2 + y**4 + 4*x*y**2
        uv = 2*x**3 + 4*x*y + y**2*x**2 + 2*y**3
        vv = x**4 + 4*x**2*y + 4*y**2

    Advection fluxes are:

        d(uu)/dx = 8*x + 4*y**2
        d(uv)/dy = 4*x + 2*x**2*y + 6*y**2
        d(vu)/dx = 6*x**2 + 4*y + 2*y**2*x
        d(vv)/dy = 4*x**2 + 8*y

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
    G = Cartesian.Grid(N, N, 2.*np.pi, 2.*np.pi, [0.0, 0.0], ['Periodic', 'Periodic', 'Periodic', 'Periodic'])

    # Create a vector field V defined on G with one level of ghost nodes
    V = Field.vector({'name': 'V', 'Grid': G, 'gl': 1})

    # Initialize the vector field (remember the staggering)
    V.x.f = 2.*(G.xc + G.dx/2) + G.yc**2
    V.y.f = 2.*(G.yc + G.dy/2) + G.xc**2

    # Evaluate the advection term
    Advection = Navier_Stokes.compute_advection_fluxes(V)

    # Evaluate the analytical advection fluxes
    sol = Field.vector({'name': 'Va', 'Grid': G, 'gl': 0})
    sol.x.f = -(12.*(G.xc[1:-1,1:-1] + G.dx/2.) + 10.*G.yc[1:-1,1:-1]**2 + \
              2.*(G.xc[1:-1,1:-1] + G.dx/2.)**2*G.yc[1:-1,1:-1])
    sol.y.f = -(10.*G.xc[1:-1,1:-1]**2 + 12.*(G.yc[1:-1,1:-1] + G.dy/2.) + \
              2.*(G.yc[1:-1,1:-1] + G.dy/2.)**2*G.xc[1:-1,1:-1])

    ex.append(np.amax(np.abs(sol.x.f - Advection.x.f)))
    ey.append(np.amax(np.abs(sol.y.f - Advection.y.f)))

scaling_1st = 10./resolutions
scaling_2nd = 100./resolutions**2

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