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

###################################################################################################
# Define the computational domain and the numerical setup
###################################################################################################
N = 64
L = 1.
Grid = Cartesian.Grid(int(N), int(N), L, L, [0.0, 0.0], ['Wall', 'Wall', 'Wall', 'Wall'])

# Fluid material properties
density = 1.
viscosity = 1.0e-3

# Set timstep due to viscous limit
dt = 0.125*(Grid.dx**2)/viscosity
dt = dt/2.
###################################################################################################
# Init all requested fields
###################################################################################################
P = Field.scalar({'name': 'Pn', 'Grid': Grid, 'gl': 1})
V = Field.vector({'name': 'Vn', 'Grid': Grid, 'gl': 1})
RHS = Field.vector({'name': 'RHSnm1', 'Grid': Grid, 'gl': 0}) # no needs of BC for this field

# Set bc on top wall
V.x.t[:] = 1.

fields = {'Pn': P, 'Vn': V, 'RHSnm1': RHS}
parameters = {'dt': dt, 'rho': density, 'mu': viscosity, 'g': [0., 0.]}

###################################################################################################
# Solve
###################################################################################################
Vs_old = Field.scalar({'name': 'Uo', 'Grid': Grid, 'gl': 1})

not_steady = True
s = 1
while (not_steady):
    Vs_old.f = V.x.f.copy()
    P, V, RHS = Navier_Stokes.advance(fields, parameters)

    # # Enforce bc on top wall
    V.x.t[:] = 1.

    diff = np.amax(np.abs(Vs_old.f - V.x.f))
    
    if (s%500 == 0):
        plt.figure()
        plt.contourf(np.transpose(np.sqrt(V.x.f**2 + V.y.f**2)))
        plt.show()
        plt.close()

    print('step: ', s, 'max div: ', np.amax(Field.Divergence(V)), 'diff: ', diff)
    
    # Check for steady state
    if (diff < 1.0e-8 and s > 3): not_steady = False
    s = s + 1

V.plot()

###################################################################################################
# Plot
###################################################################################################
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$N$')
# ax.set_ylabel(r'$L_{\infty}(e)$')
# ax.set_title('Channel flow')
# plt.plot(Grid.y, u, label = 'Analytical Solution')
# plt.plot(Grid.y, V.x.f[int(N/2),1:-1], 'o', label = 'Numerical Solution')
# ax.legend()
# ax.grid()
# fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
# plt.tight_layout()
# plt.show()
# plt.close()