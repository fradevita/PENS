'''
    We wish to solve the Poisson equation

        d^2f
        ---- = phi
        dx^2

    with f(x,y) = cos(pi x) * cos(pi y), x,y = [-1 , 1]
    BCs are Neumann = 0 (df/dx = 0) on all boundaries.
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
import Poisson

###################################################################################################
# Evaluate grid convergence
###################################################################################################
resolutions = [8]
for i in range(6):
    resolutions.append(resolutions[i]*2)
resolutions = np.array(resolutions)

e = []
for N in resolutions:

    # Create the Cartesian Grid
    Grid = Cartesian.Grid(int(N), int(N), 4.0, 4.0, [-2.0, -2.0], ['Periodic', 'Periodic', 'Periodic', 'Periodic'])
    
    # Create a Scalar field (no need of ghost nodes)
    phi = Field.scalar({'name': 'phi', 'Grid': Grid, 'gl': 1})
    # And set it to the RHS of poisson equation
    phi.f = -2.*np.pi**2*np.cos(np.pi*Grid.xc)*np.cos(np.pi*Grid.yc)

    # Solve Poisson equation
    sol = Poisson.solve_NN(phi)
    #sol = Poisson.solve_PP(phi)
    #sol = Poisson.solve_PN(phi)
    
    # The solution is 
    f = np.cos(np.pi*Grid.xc)*np.cos(np.pi*Grid.yc)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')
    # # surf = ax.plot_surface(Y, X, self.f[self.gl:self.sy - self.gl, self.gl: self.sx -self.gl], cmap=cm.coolwarm,
    # #             linewidth=0, antialiased=False)
    # surf = ax.plot_wireframe(Grid.xc[1:-1,1:-1], Grid.yc[1:-1,1:-1], sol)
    # surf = ax.plot_wireframe(Grid.xc, Grid.yc, f, color = 'red')
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    e.append(np.amax(np.abs(sol - f[1:-1,1:-1])))

# Scaling lines for the plot
scaling_1st = 1./resolutions
scaling_2nd = 10/resolutions**2

###################################################################################################
# Plot
####################################################################################################
fig, ax = plt.subplots()
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$L_{\infty}(e)$')
ax.set_title('2D Poisson solver convergence rate')
ax.loglog(resolutions, e, 'o')
ax.loglog(resolutions, scaling_1st,  '-', color = 'black', label = r'$1/N$')
ax.loglog(resolutions, scaling_2nd, '-.', color = 'black', label = r'$1/N^2$')
ax.legend()
ax.grid()
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
plt.tight_layout()
plt.show()
plt.close()