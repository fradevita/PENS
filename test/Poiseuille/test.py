'''
    We wish to solve the flow in a channel for which the analytical solution is available.
'''

###################################################################################################
# Import all requested modules
###################################################################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# Load PENS modules
sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Cartesian
import Field
import Navier_Stokes

resolutions = np.array([4, 8, 16, 32])
e = []

for N in resolutions:

    ###################################################################################################
    # Define the computational domain and the numerical setup
    ###################################################################################################
    Ly = 1.
    Nx = 4
    Lx = Ly*float(Nx/N)
    Grid = Cartesian.Grid(int(Nx), int(N), Lx, Ly, [0.0, 0.0], ['Periodic', 'Periodic', 'Wall', 'Wall'])

    # Fluid material properties
    density = 1.
    viscosity = 1.0e-2

    # External body force
    dpdx = 1.0

    # Set timstep due to viscous limit
    dt = 0.125*(Grid.dx**2)/viscosity

    ###################################################################################################
    # Init all requested fields
    ###################################################################################################
    P = Field.scalar({'name': 'Pn', 'Grid': Grid, 'gl': 1})
    V = Field.vector({'name': 'Vn', 'Grid': Grid, 'gl': 1})
    RHS = Field.vector({'name': 'RHSnm1', 'Grid': Grid, 'gl': 0}) # no needs of BC for this field

    fields = {'Pn': P, 'Vn': V, 'RHSnm1': RHS}
    parameters = {'dt': dt, 'rho': density, 'mu': viscosity, 'g': [dpdx, 0.]}

    # Analytical solution
    u = np.zeros(N)
    for j in range(N):
        u[j] = -0.5/viscosity*dpdx*(Grid.y[j]**2 - Grid.y[j]*Grid.Ly)

    ###################################################################################################
    # Solve
    ###################################################################################################
    Vs_old = Field.scalar({'name': 'Uo', 'Grid': Grid, 'gl': 1})

    not_steady = True
    s = 1
    while (not_steady):
        Vs_old.f = V.x.f.copy()
        P, V, RHS = Navier_Stokes.advance(fields, parameters)
        diff = np.amax(np.abs(Vs_old.f - V.x.f))
        print('step: %d, max div: %e, diff: %g' % (s, np.amax(Field.Divergence(V)), diff))
        # Check for steady state
        if (diff < 1.0e-6): not_steady = False
        s = s + 1

    ###################################################################################################
    # Evaluate error
    ###################################################################################################
    e.append(np.amax(np.abs(V.x.f[2,1:-1] - u)))

###################################################################################################
# Plot
###################################################################################################
scaling_1st = 5./resolutions
scaling_2nd = 10./resolutions**2
fig, ax = plt.subplots()
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$L_{\infty}(e)$')
ax.set_title('Channel flow')
ax.loglog(resolutions, e, 'o')
ax.loglog(resolutions, scaling_1st,  '-k', label = r'$1/N$')
ax.loglog(resolutions, scaling_2nd, '-.k', label = r'$1/N^2$')
ax.legend()
ax.grid()
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
plt.tight_layout()
plt.show()
plt.close()