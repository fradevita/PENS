'''
    We wish to solve the Taylor-Green Vortex test case for which analytical solution is available.
    See https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex
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

resolutions = np.array([8, 16, 32, 64])
eu = []
ev = []

for N in resolutions:

    ###################################################################################################
    # Define the computational domain and the numerical setup
    ###################################################################################################
    L = 2.*np.pi
    Grid = Cartesian.Grid(int(N), int(N), L, L, [0.0, 0.0], ['Periodic', 'Periodic', 'Periodic', 'Periodic'])

    # Fluid material properties
    density = 1.
    viscosity = 1.

    # Set timstep due to viscous limit
    dt = 0.125*(Grid.dx)**2/viscosity
    Tend = 0.1
    smax = int(Tend/dt) + 1
    dt = Tend/(smax)

    ###################################################################################################
    # Init all requested fields
    ###################################################################################################
    P = Field.scalar({'name': 'Pn', 'Grid': Grid, 'gl': 1})
    V = Field.vector({'name': 'Vn', 'Grid': Grid, 'gl': 1})
    RHS = Field.vector({'name': 'RHSnm1', 'Grid': Grid, 'gl': 0}) # no needs of BC for this field

    # Initial fields
    V.x.f = -np.cos(Grid.xc + Grid.dx/2.)*np.sin(Grid.yc)
    V.y.f =  np.sin(Grid.xc)*np.cos(Grid.yc + Grid.dy/2.)
    P.f = -0.25*(np.cos(2.*Grid.xc) + np.cos(2.*Grid.yc))

    print('initial divergence', np.amax(Field.Divergence(V)))
    fields = {'Pn': P, 'Vn': V, 'RHSnm1': RHS}
    parameters = {'dt': dt, 'rho': density, 'mu': viscosity, 'g': [0., 0.]}

    ###################################################################################################
    # Solve equation
    ###################################################################################################
    time = 0.
    for s in range(1,smax+1):
        P, V, RHS = Navier_Stokes.advance(fields, parameters)
        time = time + dt
        print('step: %d time: %f, maximum divergence: %g' % (s, time, np.amax(Field.Divergence(V))))

    ###################################################################################################
    # Define analytical solution
    ###################################################################################################
    Ft = np.exp(-2.*viscosity*Tend)
    Vxa = -np.cos(Grid.xc + Grid.dx/2)*np.sin(Grid.yc)*Ft
    Vya = np.sin(Grid.xc)*np.cos(Grid.yc + Grid.dy/2.)*Ft

    ###################################################################################################
    # Evaluate error
    ###################################################################################################
    eu.append(np.max(np.abs(Vxa - V.x.f)))
    ev.append(np.max(np.abs(Vya - V.y.f)))

###################################################################################################
# Plot
####################################################################################################
scaling_1st = 0.5/resolutions
scaling_2nd = 5./resolutions**2
fig, ax = plt.subplots()
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$L_{\infty}(e)$')
ax.set_title('2D Poisson solver convergence rate')
ax.loglog(resolutions, eu, 'o', fillstyle = 'none', label = 'e_u')
ax.loglog(resolutions, ev, 'x', fillstyle = 'none', label = 'e_v')
ax.loglog(resolutions, scaling_1st,  '-', color = 'black', label = r'$1/N$')
ax.loglog(resolutions, scaling_2nd, '-.', color = 'black', label = r'$1/N^2$')
ax.legend()
ax.grid()
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
plt.tight_layout()
plt.show()
plt.close()