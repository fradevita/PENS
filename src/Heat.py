# Module to solve Heat equation in 2D
import sys
import os
sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Field
import numpy as np

#########################################################################################
# Advance function
#########################################################################################
# Advance one timestep of the incompressible Navier-Stokes equations from timestep n to 
# timestep n + 1
def advance(fields, parameters):

    # Select input
    T = fields.get('T')
    dt = parameters.get('dt')
    alpha = parameters.get('alpha')

    # Evaluate RHS at timestep n
    RHS = Field.Laplacian(T)

    # Update Temperature field 
    T.f[1:-1,1:-1] = T.f[1:-1,1:-1] + dt*alpha*RHS.f

    # Update ghost nodes
    T.update_ghost_nodes()

    return T  