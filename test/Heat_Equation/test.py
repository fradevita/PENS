'''
    We wish to solve the Heat equation in a rectangular box.
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
import Heat
import Utils

###################################################################################################
# Define the computational domain and the numerical setup
###################################################################################################
N = 32
L = 1.
Grid = Cartesian.Grid(int(N), int(N), L, L, [-0.5, -0.5], ['Periodic', 'Periodic', 'Periodic', 'Periodic'])

# Material properties
alpha = 1.

# Set timstep due to diffusion limit
dt = 0.25*(L/float(N))**2/alpha

###################################################################################################
# Init all requested fields
###################################################################################################
T = Field.scalar({'name': 'T', 'Grid': Grid, 'gl': 1})

for j in range(N):
    for i in range(N):
        T.f[i+1,j+1] = Utils.Gaussian(Grid.x[i], Grid.y[j], 0.1)       
T.update_ghost_nodes()
integral_T0 = T.integral()
fields = {'T': T}
parameters = {'dt': dt, 'alpha': alpha}

###################################################################################################
# Solve the equation
###################################################################################################

for s in range(1000):
    T = Heat.advance(fields, parameters)
    if (s%100 == 0): T.plot()
    print('step: ', s, 'Delta T/T(0):', abs(T.integral() - integral_T0)/integral_T0)

# ###################################################################################################
# # Plot
# ####################################################################################################
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$N$')
# ax.set_ylabel(r'$L_{\infty}(e)$')
# ax.set_title('2D Poisson solver convergence rate')
# ax.loglog(resolutions, e, 'o')
# ax.loglog(resolutions, scaling_1st,  '-', color = 'black', label = r'$1/N$')
# ax.loglog(resolutions, scaling_2nd, '-.', color = 'black', label = r'$1/N^2$')
# ax.legend()
# ax.grid()
# fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
# plt.tight_layout()
# plt.show()
# plt.close()