# Module to solve Navier-Stokes equations in 2D
import sys
import os

sys.path.insert(0, os.path.expandvars(os.environ['PENS']))
import Field
import Poisson_2D

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

    # Evaluate RHS at timestep n
    RHS = Field.vector({'Grid': Vn.G})
    gradP = Field.Gradient(Pn)
    Advection = compute_advection_fluxes(Vn)
    Diffusion = compute_diffusion_fluxes(Vn, mu)
    RHS.x = -gradP.x + (1.5*Diffusion.x + 1.5*Advection.x - 0.5*RHSnm1.x)
    RHS.y = -gradP.y + (1.5*Diffusion.y + 1.5*Advection.y - 0.5*RHSnm1.y)

    # Compute predicted velocity field
    Vs = Field.vector({'name': 'velocity*', 'Grid': Vn.G, 'gl': 1})
    Vs.x.f[1:-1, 1:-1] = Vn.x.f[1:-1, 1:-1] + dt*RHS.x.f/rho
    Vs.y.f[1:-1, 1:-1] = Vn.y.f[1:-1, 1:-1] + dt*RHS.y.f/rho
    Vs.update_ghost_nodes()

    # Solve Poisson equation
    phi = Field.scalar({'name': 'phi', 'Grid': Vn.G, 'gl': 1})
    phi.x.f = Poisson_2D.FFT_solver(Field.Divergence(Vs)*rho/dt)
    phi.update_ghost_nodes()

    # Update velocity
    dphidx, dphidy = Field.Gradient(phi)
    Vn.x = Vn.x - dphidx*dt/rho
    Vn.y = Vn.y - dphidy*dt/rho

    # Update pressure
    Pn.x = Pn.x + phi

    return Pn, Vn, RHS

#########################################################################################
# Advection function
#########################################################################################
def compute_advection_fluxes(V: Field.vector):
    Advection = Field.vector({'Grid': V.G})
    idelta = 1./V.G.dx
    for j in range(V.G.Ny):
        jp = j + 1
        jm = j - 1
        for i in range(V.G.Nx):
            ip = i + 1
            im = i - 1
            # Advection.x: -d(uu)/dx - d(uv)/dy
            # d(uu)/dx = (uuip - uuim)/dx
            uuip = 0.25*(V.x.f[j,ip] + V.x.f[j,i])**2
            uuim = 0.25*(V.x.f[j,im] + V.x.f[j,i])**2

            # d(uv)/dy = (uvjp - uvjm)/dy
            uvjp = (V.x.f[jp,i] + V.x.f[ j,i])*(V.y.f[ j,ip] + V.y.f[ j,i])*0.25
            uvjm = (V.x.f[ j,i] + V.x.f[jm,i])*(V.y.f[jm,ip] + V.y.f[jm,i])*0.25

            Advection.x.f[j,i] = - (uuip - uuim)*idelta - (uvjp - uvjm)*idelta

            # Advection.y: -d(vu)/dx - d(vv)/dy
            # d(vu)/dx = (vuip - vuim)/delta
            vuip = (V.y.f[j,ip] + V.y.f[j,i ])*(V.x.f[jp,i ] + V.x.f[j, i])*0.25
            vuim = (V.y.f[j,i ] + V.y.f[j,im])*(V.x.f[jp,im] + V.x.f[j,im])*0.25

            # d(vv)/dy = (vvjp - vvjm)/delta
            vvjp = 0.25*(V.y.f[jp,i] + V.y.f[j,i])**2
            vvjm = 0.25*(V.y.f[jm,i] + V.y.f[j,i])**2

            Advection.y.f[j,i] = (vuip - vuim)*idelta - (vvjp - vvjm)*idelta
    return Advection

#########################################################################################
# Diffusion function
#########################################################################################
def compute_diffusion_fluxes(V: Field.vector, mu):
    Diffusion = Field.vector({'Grid': V.G})
    Diffusion.x.f = mu*Field.Laplacian(V.x)
    Diffusion.y.f = mu*Field.Laplacian(V.y)
    return Diffusion