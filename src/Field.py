import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

#########################################################################################
# Scalar class
#########################################################################################
# List of avialables arguments for the scalar field class
scalar_args = ['name', 'Grid', 'gl']
class scalar:
    # Class constructor
    def __init__(self, args: dict):
        # By default the scalar has zero ghost level
        self.gl = 0
        # Read input arguments
        if (args):
            # First check that all the provided arguments are valid
            for key in args:
                if key not in scalar_args:
                    sys.exit(key + ' is not a valid argument for scalar field.')

            # Init the provided arguments
            if 'name' in args:
                self.name = args.get('name')
            if 'gl' in args:
                self.gl = args.get('gl')
            if 'Grid' in args:
                self.G = args.get('Grid')
                self.f = np.zeros((self.G.Ny + 2*self.gl, self.G.Nx + 2*self.gl))
        
        # Size of the scalar in each dimension, used to indexing
        self.sy = len(self.f[:,0])
        self.sx = len(self.f[0,:])

    def set_field(self, S: np.ndarray):
        self.f[self.gl:self.sy - self.gl, self.gl: self.sx -self.gl] = S[:,:]
        self.update_ghost_nodes()

    def plot(self, plot_type = 'contour'):
        if (plot_type == 'contour'): 
            plt.figure()
            plt.contourf(self.G.y, self.G.x, self.f[self.gl:self.sy - self.gl, self.gl: self.sx -self.gl])
            plt.colorbar()
            plt.show()
            plt.close()
        elif (plot_type == 'surface'):
            X, Y = np.meshgrid(self.G.x, self.G.y)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            surf = ax.plot_surface(Y, X, self.f[self.gl:self.sy - self.gl, self.gl: self.sx -self.gl], cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
            fig.colorbar(surf, label = self.name, orientation = 'horizontal', 
                            shrink = 0.6)
            plt.tight_layout()
            plt.show()
            plt.close()

    def update_ghost_nodes(self):
        self.f[ 0, :] = self.f[-2, :]
        self.f[-1, :] = self.f[ 1, :]
        self.f[ :, 0] = self.f[ :,-2]
        self.f[ :,-1] = self.f[ :, 1]

    def integral(self):
        return np.sum(self.f[1:-1,1:-1])*self.G.dx**2

#########################################################################################
# Vector class
#########################################################################################
# List of avialables arguments for the vector field class
vector_args = ['name', 'Grid', 'gl']
class vector:
    # Class constructor
    def __init__(self, args: dict):
        if (args):
            # First check that all the provided arguments are valid
            for key in args:
                if key not in vector_args:
                    print(key + ' is not a valid argument for vector field.')

            # Init the provided arguments
            if 'name' in args:
                self.name = args.get('name')
            if 'Grid' in args:
                self.G = args.get('Grid')
                self.x = scalar(args)
                self.x.name = args.get('name')+'.x'
                self.y = scalar(args)
                self.y.name = args.get('name')+'.y'

    def set_field(self, Sx: np.ndarray, Sy: np.ndarray):
        self.x.f[1:-1,1:-1] = Sx[:,:]
        self.y.f[1:-1,1:-1] = Sy[:,:]
        self.update_ghost_nodes()

    def plot(self, plot_type = 'quiver'):
        if (plot_type == 'quiver'): 
            plt.figure()
            plt.quiver(self.G.y, self.G.x, self.x.f[1:-1,1:-1], self.y.f[1:-1,1:-1])
            plt.show()
            plt.close()

    def update_ghost_nodes(self):
        self.x.update_ghost_nodes()
        self.y.update_ghost_nodes()

#########################################################################################
# Function operating on fields
#########################################################################################
from functools import singledispatch

# Define divergence function
@singledispatch
def Divergence():
    return
@Divergence.register
def _(F: scalar):
    sys.exit('ERROR: cannot compute divergence of a scalar field.')
@Divergence.register
def _(F: vector):
    dFxdx = np.diff(F.x.f[1:-1,:-1], axis = 1)/F.G.dx
    dFydy = np.diff(F.y.f[:-1,1:-1], axis = 0)/F.G.dy
    return dFxdx + dFydy

# Define gradient function
@singledispatch
def Gradient():
    return
@Gradient.register
def _(F: scalar):
    # First check that the scalar has ghost nodes
    if (F.gl == 0):
        sys.exit('Scalar field ' + F.name + 'must have ghost nodes in order to evaluate its gradient.')
    dFdx = np.zeros((F.G.Ny,F.G.Nx))
    dFdy = np.zeros((F.G.Ny,F.G.Nx))
    dFdx[:, :] = (F.f[1:-1, 2:] - F.f[1:-1, 1:-1])/F.G.dx
    dFdy[:, :] = (F.f[2:, 1:-1] - F.f[1:-1, 1:-1])/F.G.dy
    return dFdx, dFdy
@Gradient.register
def _(F: vector):
    sys.exit('ERROR: cannot compute gradient of a vector field (tensors do not exist yet).')

# Define curl function
@singledispatch
def Curl():
    return
@Curl.register
def _(F: scalar):
    sys.exit('ERROR: the curl of a scalar field is not defined.')
@Curl.register
def _(F: vector):
    # Since the grid is 2D the output is a scalar
    return F

# Define Laplacian function
@singledispatch
def Laplacian():
    return
@Laplacian.register
def _(F: scalar):
    LapF = np.zeros((F.G.Ny,F.G.Nx))
    LapF[:,:] = (F.f[2:,1:-1] + F.f[:-2,1:-1] + F.f[1:-1,2:] + F.f[1:-1,:-2] - 
                    4.*F.f[1:-1,1:-1])/F.G.dx**2
    return LapF
@Laplacian.register
def _(F: vector):
    sys.exit('ERROR: cannot call Laplacian with a vector field input')
