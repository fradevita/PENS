# Module to solve Navier-Stokes equations in 2D
import json
import sys
import os

sys.path.insert(0, os.path.expandvars('./'))
import Cartesian
import Field

def solve(setup_file):

    # Load setup file
    setup = json.load(open(setup_file))
    print(setup["Parameters"])

    # Create the Grid
    Grid = Cartesian.CreateGridFromFile(setup_file)

    # Create all necessary fields
    

    return