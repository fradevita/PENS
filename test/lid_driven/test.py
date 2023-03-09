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
dt = (0.125*(Grid.dx**2)/viscosity)/2

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
uref = np.genfromtxt('uref')
vref = np.genfromtxt('vref')
not_steady = True
s = 1
while (not_steady):
    Vs_old.f = V.x.f.copy()
    P, V, RHS = Navier_Stokes.advance(fields, parameters)

    # # Enforce bc on top wall
    V.x.t[:] = 1.

    diff = np.amax(np.abs(Vs_old.f - V.x.f))
    print('step: ', s, 'max div: ', np.amax(Field.Divergence(V)), 'diff: ', diff)
    
    # Check for steady state
    if (diff < 1.0e-6 and s > 3): not_steady = False
    s = s + 1

###################################################################################################
# Plot
###################################################################################################

fig, ax = plt.subplots()
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$u$')
plt.plot(uref[:,0] + 0.5, uref[:,1], 'o', label = 'reference solution')
plt.plot(Grid.y, V.x.f[int(N/2),1:-1], '-', label = 'numerical solution')
ax.legend()
ax.grid()
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$v$')
plt.plot(vref[:,0] + 0.5, vref[:,1], 'o', label = 'reference solution')
plt.plot(Grid.x, V.y.f[1:-1,int(N/2)], '-', label = 'numerical solution')
ax.legend()
ax.grid()
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
plt.tight_layout()
plt.show()
plt.close()