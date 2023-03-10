import numpy as np
import sys
import os
import matplotlib.pyplot as plt
# Load PENS modules
sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Cartesian
import Field
import Navier_Stokes

Lx = 1.
Ly = 1.
Nx = 8
Ny = 8
Grid = Cartesian.Grid(Nx, Ny, Lx, Ly, [0.0, 0.0,], ['Periodic','Periodic','Wall','Wall' ])

density = 1.
viscosity = 1.
U_w = 0.5

dt = 0.125*Grid.dx**2/viscosity # Viscous timestep limit

P = Field.scalar({'name': 'P', 'Grid': Grid, 'gl': 1})
V = Field.vector({'name': 'V', 'Grid': Grid, 'gl': 1})
RHS = Field.vector({'name': 'RHS', 'Grid': Grid, 'gl': 0})

fields = {'Pn': P, 'Vn': V, 'RHSnm1': RHS}
parameters = {'dt': dt, 'rho': density, 'mu': viscosity, 'g': [0., -1.]}

diff = 19999
tol = 1.0e-12
Vold = Field.vector({'name': 'V', 'Grid': Grid, 'gl': 1})
while (diff > tol):
    Vold.x.f = V.x.f.copy()
    P, V, RHS = Navier_Stokes.advance(fields, parameters)

    V.x.t[:] =  U_w
    V.x.b[:] = -U_w

    diff = np.amax( np.abs(V.x.f - Vold.x.f))
    print(diff, np.amax(Field.Divergence(V)))

# Analytical solution
u = 2.*U_w/Ly*Grid.y - U_w

V.y.plot('surface')
