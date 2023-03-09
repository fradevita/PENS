# Module to solve Navier-Stokes equations in 2D
import sys
import os

sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Field
import Poisson
import numpy as np

#########################################################################################
# Advance function
#########################################################################################
# Advance one timestep of the incompressible Navier-Stokes equations from timestep n to 
# timestep n + 1
def advance(fields, parameters):

    # Select input
    Pn = fields.get('Pn')
    Vn = fields.get('Vn')
    RHSnm1 = fields.get('RHSnm1')
    dt = parameters.get('dt')
    rho = parameters.get('rho')
    mu = parameters.get('mu')
    g = parameters.get('g')
    
    # Evaluate RHS at timestep n
    RHS = Field.vector({'name': 'RHS', 'Grid': Vn.G, 'gl': 0})
    gradP = Field.Gradient(Pn)
    Advection = compute_advection_fluxes(Vn)
    Diffusion = compute_diffusion_fluxes(Vn)
    RHS.x.f = mu*Diffusion.x.f/rho + Advection.x.f
    RHS.y.f = mu*Diffusion.y.f/rho + Advection.y.f

    # Compute predicted velocity field
    Vs = Field.vector({'name': 'velocity*', 'Grid': Vn.G, 'gl': 1})
    Vs.x.f[1:-1, 1:-1] = Vn.x.f[1:-1, 1:-1] + dt*(-gradP.x.f/rho + 1.5*RHS.x.f - 0.5*RHSnm1.x.f + g[0])
    Vs.y.f[1:-1, 1:-1] = Vn.y.f[1:-1, 1:-1] + dt*(-gradP.y.f/rho + 1.5*RHS.y.f - 0.5*RHSnm1.y.f + g[1])
    
    # Set bc of predicted velocity field equal to that of V
    Vs.x.l = Vn.x.l
    Vs.x.r = Vn.x.r
    Vs.x.b = Vn.x.b
    Vs.x.t = Vn.x.t
    Vs.y.l = Vn.y.l
    Vs.y.r = Vn.y.r
    Vs.y.b = Vn.y.b
    Vs.y.t = Vn.y.t
    Vs.update_ghost_nodes()

    # Solve Poisson equation
    phi = Field.scalar({'name': 'phi', 'Grid': Vn.G, 'gl': 1})
    phi.f[1:-1,1:-1] = Field.Divergence(Vs)*rho/dt
    phi.f[1:-1,1:-1] = Poisson.solve(phi)
    phi.update_ghost_nodes()

    # Update velocity
    gradphi = Field.Gradient(phi)
    Vn.x.f[1:-1,1:-1] = Vs.x.f[1:-1,1:-1] - gradphi.x.f*dt/rho
    Vn.y.f[1:-1,1:-1] = Vs.y.f[1:-1,1:-1] - gradphi.y.f*dt/rho
    Vn.update_ghost_nodes()

    # Update pressure
    Pn.f[1:-1,1:-1] = Pn.f[1:-1,1:-1] + phi.f[1:-1,1:-1]
    Pn.update_ghost_nodes()
    
    RHSnm1.x.f = RHS.x.f.copy()
    RHSnm1.y.f = RHS.y.f.copy()

    del RHS, phi, Vs, gradP, gradphi, Diffusion, Advection
    return Pn, Vn, RHSnm1

#########################################################################################
# Advection function
#########################################################################################
def compute_advection_fluxes(V: Field.vector):
    Advection = Field.vector({'name': 'Advection', 'Grid': V.G, 'gl': 0})
    idelta = 1./V.G.dx
    for j in range(1,V.G.Ny+1):
        jp = j+1
        jm = j-1
        for i in range(1,V.G.Nx+1):
            ip = i+1
            im = i-1
            # Advection.x: -d(uu)/dx - d(uv)/dy
            # d(uu)/dx = (uuip - uuim)/dx
            uuip = 0.25*(V.x.f[ip,j] + V.x.f[i,j])**2
            uuim = 0.25*(V.x.f[im,j] + V.x.f[i,j])**2

            # d(uv)/dy = (uvjp - uvjm)/dy
            uvjp = (V.x.f[i,jp] + V.x.f[i,j ])*(V.y.f[ip,j ] + V.y.f[i,j ])*0.25
            uvjm = (V.x.f[i,j ] + V.x.f[i,jm])*(V.y.f[ip,jm] + V.y.f[i,jm])*0.25

            Advection.x.f[i-1,j-1] = - (uuip - uuim)*idelta - (uvjp - uvjm)*idelta
            
            # Advection.y: -d(vu)/dx - d(vv)/dy
            # d(vu)/dx = (vuip - vuim)/delta
            vuip = (V.y.f[ip,j] + V.y.f[i ,j])*(V.x.f[i ,jp] + V.x.f[i ,j])*0.25
            vuim = (V.y.f[i ,j] + V.y.f[im,j])*(V.x.f[im,jp] + V.x.f[im,j])*0.25

            # d(vv)/dy = (vvjp - vvjm)/delta
            vvjp = 0.25*(V.y.f[i,jp] + V.y.f[i,j])**2
            vvjm = 0.25*(V.y.f[i,jm] + V.y.f[i,j])**2

            Advection.y.f[i-1,j-1] = - (vuip - vuim)*idelta - (vvjp - vvjm)*idelta
    return Advection

#########################################################################################
# Diffusion function
#########################################################################################
def compute_diffusion_fluxes(V: Field.vector):
    Diffusion = Field.vector({'name': 'Diffusion', 'Grid': V.G, 'gl': 0})
    Diffusion.x = Field.Laplacian(V.x)
    Diffusion.y = Field.Laplacian(V.y)
    return Diffusion