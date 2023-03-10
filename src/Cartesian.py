import numpy as np

#########################################################################################
# Grid class definition
#########################################################################################
class Grid:
    # Class members:
    # Nx = the number of cell in the x-direction
    # Ny = the number of cell in the y-direction
    # Lx = the phyisical length in the x-direction
    # Ly = the phyisical length in the y-direction
    # origin = an array of size 2 which gives the coordinates of the 
    #          lower-left corner of the grid.
    # [x,y] = coordiantes of the center of the grid cells.

    # Class Constructor
    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float, origin: np.ndarray, bc: list):
        self.Nx = Nx
        self.Lx = Lx
        self.dx = Lx/Nx
        self.origin = origin
        self.x = np.zeros(Nx)
        for i in range(Nx):
            self.x[i] = origin[0] + (i + 0.5)*self.dx

        self.Ny = Ny
        self.Ly = Ly
        self.dy = Ly/Ny
        assert(self.dx==self.dy)
        self.y = np.zeros(Ny)
        for j in range(Ny):
            self.y[j] = origin[1] + (j + 0.5)*self.dy
        self.xc = np.zeros((Nx+2,Ny+2))
        self.yc = np.zeros((Nx+2,Ny+2))
        for j in range(Ny+2):
            for i in range(Nx+2):
                self.xc[i,j] = origin[0] + (i - 0.5)*self.dx
                self.yc[i,j] = origin[1] + (j - 0.5)*self.dx

        self.bc = bc

def CreateGridFromFile(setup: dict):
    # Create the grid from the json file stored in the input variable setup
    return Grid(setup["Grid"]["Nx"], setup["Grid"]["Ny"],
                setup["Grid"]["Lx"], setup["Grid"]["Ly"], setup["Grid"]["origin"])
