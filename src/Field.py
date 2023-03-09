import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

#########################################################################################
# Scalar class
#########################################################################################
# List of avialables arguments for the scalar field class
scalar_args = ['name', 'Grid', 'gl', 'boundary_conditions']
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
                self.f = np.zeros((self.G.Nx + 2*self.gl, self.G.Ny + 2*self.gl), order = 'F')
            if 'boundary_conditions' in args:
                self.bc = args.get('boundary_conditions')
        
        # Size of the scalar in each dimension, used to indexing
        self.sx = len(self.f[:,0])
        self.sy = len(self.f[0,:])

        # By default set boundary valuse to zero
        if (self.gl > 0):    
            self.l = np.zeros(self.sy)
            self.r = np.zeros(self.sy)
            self.t = np.zeros(self.sx)
            self.b = np.zeros(self.sx)

    def set_field(self, S: np.ndarray):
        self.f[self.gl:self.sx - self.gl, self.gl: self.sy -self.gl] = S[:,:]
        self.update_ghost_nodes()

    def plot(self, plot_type = 'contour'):
        if (plot_type == 'contour'): 
            fig, ax = plt.subplots()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            cf = ax.contourf(self.G.x, self.G.y, self.f[self.gl:self.sx - self.gl, self.gl: self.sy -self.gl])
            plt.colorbar(cf, label = self.name)
            plt.show()
            plt.close()
        elif (plot_type == 'surface'):
            X, Y = np.meshgrid(self.G.x, self.G.y)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            # surf = ax.plot_surface(Y, X, self.f[self.gl:self.sy - self.gl, self.gl: self.sx -self.gl], cmap=cm.coolwarm,
            #             linewidth=0, antialiased=False)
            surf = ax.plot_wireframe(X, Y, self.f[self.gl:self.sx - self.gl, self.gl: self.sy -self.gl])
            fig.colorbar(surf, label = self.name, orientation = 'horizontal', 
                            shrink = 0.6)
            plt.tight_layout()
            plt.show()
            plt.close()

    def update_ghost_nodes(self):
        if (self.G.bc[0] == 'Periodic'):
            self.f[ 0, :] = self.f[-2, :]
        elif (self.G.bc[0] == 'Wall'):
            self.f[ 0, :] = self.f[ 1, :]
        else:
            sys.exit('Error: wrong bc[0]')
        
        if (self.G.bc[1] == 'Periodic'):
            self.f[-1, :] = self.f[ 1, :]
        elif (self.G.bc[1] == 'Wall'):
            self.f[-1, :] = self.f[-2, :]
        else:
            sys.exit('Error: wrong bc[1]')
        
        if (self.G.bc[2] == 'Periodic'):
            self.f[ :, 0] = self.f[ :,-2]
        elif (self.G.bc[2] == 'Wall'):
            self.f[ :, 0] = self.f[ :, 1]
        else:
            sys.exit('Error: wrong bc[2]')
        
        if (self.G.bc[3] == 'Periodic'):
            self.f[ :,-1] = self.f[ :, 1]
        elif (self.G.bc[3] == 'Wall'):
            self.f[ :,-1] = self.f[ :, -2]
        else:
            sys.exit('Error: wrong bc[3]')

    def integral(self):
        return np.sum(self.f[1:-1,1:-1])*self.G.dx**2

#########################################################################################
# Vector class
#########################################################################################
# List of avialables arguments for the vector field class
vector_args = ['name', 'Grid', 'gl', 'boundary_conditions']
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
            else:
                self.name = 'unset'
            if 'Grid' in args:
                self.G = args.get('Grid')
                self.x = scalar(args)
                self.x.name = args.get('name')+'.x'
                self.y = scalar(args)
                self.y.name = args.get('name')+'.y'
            if 'boundary_conditions' in args:
                self.bc = args.get('boundary_conditions')

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
        if (self.G.bc[0] == 'Periodic'):
            self.x.f[ 0, :] = self.x.f[-2, :]
            self.y.f[ 0, :] = self.y.f[-2, :]
        elif (self.G.bc[0] == 'Wall'):
            self.x.f[ 0, :] = self.x.l
            self.y.f[ 0, :] = 2.*self.y.l - self.y.f[ 1, :] 
        else:
            sys.exit('Error: wrong bc[0]')
        
        if (self.G.bc[1] == 'Periodic'):
            self.x.f[-1, :] = self.x.f[ 1, :]
            self.y.f[-1, :] = self.y.f[ 1, :]
        elif (self.G.bc[1] == 'Wall'):
            self.x.f[-1, :] = self.x.r
            self.x.f[-2, :] = self.x.r
            self.y.f[-1, :] = 2.*self.y.r - self.y.f[-2, :]
        else:
            sys.exit('Error: wrong bc[1]')
        
        if (self.G.bc[2] == 'Periodic'):
            self.x.f[ :, 0] = self.x.f[ :,-2]
            self.y.f[ :, 0] = self.y.f[ :,-2]
        elif (self.G.bc[2] == 'Wall'):
            self.x.f[ :, 0] = 2.*self.x.b - self.x.f[ :, 1]
            self.y.f[ :, 0] = self.y.b
        else:
            sys.exit('Error: wrong bc[2]')
        
        if (self.G.bc[3] == 'Periodic'):
            self.x.f[ :,-1] = self.x.f[ :, 1]
            self.y.f[ :,-1] = self.y.f[ :, 1]
        elif (self.G.bc[3] == 'Wall'):
            self.x.f[ :,-1] = 2.*self.x.t - self.x.f[ :,-2]
            self.y.f[ :,-1] = self.y.t
            self.y.f[ :,-2] = self.y.t
        else:
            sys.exit('Error: wrong bc[3]')

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
    div = np.zeros((F.G.Nx,F.G.Ny), order = 'F')
    # for j in range(1,F.G.Ny+1):
    #     for i in range(1,F.G.Nx+1):
    #         div[i-1,j-1] = (F.x.f[i,j] - F.x.f[i-1,j])/F.G.dx + (F.y.f[i,j] - F.y.f[i,j-1])/F.G.dx
    div = (F.x.f[1:-1,1:-1] - F.x.f[:-2,1:-1])/F.G.dx + (F.y.f[1:-1,1:-1] - F.y.f[1:-1,:-2])/F.G.dx
    return div

# Define gradient function
@singledispatch
def Gradient():
    return
@Gradient.register
def _(F: scalar):
    # First check that the scalar has ghost nodes
    if (F.gl == 0):
        sys.exit('Scalar field ' + F.name + 'must have ghost nodes in order to evaluate its gradient.')
    gradF = vector({'name': 'grad_of_'+ F.name, 'Grid': F.G, 'gl': 0})
    # for j in range(1,F.G.Ny+1):
    #     for i in range(1,F.G.Nx+1):
    #         gradF.x.f[i-1,j-1] = (F.f[i+1,j] - F.f[i,j])/F.G.dx
    #         gradF.y.f[i-1,j-1] = (F.f[i,j+1] - F.f[i,j])/F.G.dx
    gradF.x.f = (F.f[2:,1:-1] - F.f[1:-1,1:-1])/F.G.dx
    gradF.y.f = (F.f[1:-1,2:] - F.f[1:-1,1:-1])/F.G.dx
    return gradF
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
    if (F.gl == 0):
        sys.exit('Scalar field ' + F.name + 'must have ghost nodes in order to evaluate its laplacian.')
    LapF = scalar({'name': 'grad_of_'+ F.name, 'Grid': F.G, 'gl': 0})
    # for j in range(1,F.G.Ny+1):
    #     for i in range(1,F.G.Nx+1):
    #         LapF.f[i-1,j-1] = (F.f[i+1,j] + F.f[i-1,j] + F.f[i,j+1] + F.f[i,j-1] - 4.*F.f[i,j])/F.G.dx**2
    LapF.f = (F.f[2:,1:-1] + F.f[:-2,1:-1] + F.f[1:-1,2:] + F.f[1:-1,:-2] - 4.*F.f[1:-1,1:-1])/F.G.dx**2
    return LapF

@Laplacian.register
def _(F: vector):
    sys.exit('ERROR: cannot call Laplacian with a vector field input')