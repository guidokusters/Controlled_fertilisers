import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from scipy.sparse import diags
from scipy import integrate
import math
import h5py
import sys

#input parameters
experiment = 'Agrium_Deff_1_8' #experiment name
real_time = 90 #time in days
real_radius = 1.5e-3 #radius in m
real_thickness = 6.4e-5 #coating thickness in m
real_D0 = 1.380e-9 #diffusivity in water in m^2/s
real_Dp = 1.8e-14 #diffusivity in coating in m^2/s

#definition of numerical parameters
N = 2201 #number of grid points
dt = 1e-4 #time step
L = 22. #size of grid
tf = math.ceil(real_time/((real_radius)**2/(real_D0)/(3600*24))) #total integration time
nsteps = int(tf/dt) #number of time steps
dx = L/(N-1) #grid spacing
S = 545/1320 #saturation concentration/solid density urea
interval = math.ceil(nsteps/real_time) #index interval for updating progress

#time array for plotting
tau = np.linspace(0,tf,nsteps+1)*(real_radius)**2/(real_D0)/(3600*24)
#initialize grid
x = np.linspace(0,L,N)
#initial condition
R0 = 1.0
index0 = round(R0/dx)
index1 = round(1.0/dx)
eps = real_thickness/real_radius
index2 = round((1.+eps)/dx)
u = np.asarray([1.0 if xx<=R0+dx*0.5 else 0.0 for xx in x])
edge = np.ones(nsteps+1)
edge[0] = R0
release4 = np.zeros(nsteps+1)
f2 = real_Dp/real_D0
f3 = 1.0

alpha = dt/(2*dx**2.)

def loopsolve(nsteps,index0,alpha,S,u,index2,f3,index1,f2,dx):

    writefile = open('progress'+experiment+'.txt', 'w')
    writefile.close()

    start_time = time.time()
    
    days = 0
    writefile = open('progress'+experiment+'.txt', 'a+')
    writefile.write('Real time to simulate: '+str(real_time)+'\n')
    writefile.write('Real time: '+str(days)+' days\n')
    writefile.write('Simulation time: '+str(time.time()-start_time)+' seconds\n')
    writefile.write('\n')
    writefile.close()
    days += 1

    # Regime 2
    diag2 = np.full(index2-index1-1,1.+2.*alpha*f2)
    diagplus2 = np.zeros(index2-index1-2)
    diagmin2 = np.zeros(index2-index1-2)
    diagplus2[:] = [-alpha*f2/(i+1+index1)-alpha*f2 for i in range(index2-index1-2)]
    diagmin2[:] = [alpha*f2/(i+2+index1)-alpha*f2 for i in range(index2-index1-2)]

    # Regime 3
    diag3 = np.full(N-1-index2,1.+2.*alpha*f3)
    diagplus3 = np.zeros(N-2-index2)
    diagmin3 = np.zeros(N-2-index2)
    diagplus3[:] = [-alpha*f3/(i+1+index2)-alpha*f3 for i in range(N-2-index2)]
    diagmin3[:-1] = [alpha*f3/(i+2+index2)-alpha*f3 for i in range(N-3-index2)]
    diagmin3[-1] = -2.*alpha*f3

    A = diags([np.append(diag2,diag3),np.append(np.append(diagplus2,0),diagplus3),np.append(np.append(diagmin2,0),diagmin3)],[0,1,-1]).toarray()

    A[index2-index1-2,index2-index1-3] = (alpha*f2/(index2-1)-alpha*f2) + (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f2+f3))
    A[index2-index1-2,index2-index1-2] = 1.+2.*alpha*f2 - (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f2+f3))
    A[index2-index1-2,index2-index1-1] = -(alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f2+f3))
    A[index2-index1-2,index2-index1] = (alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f2+f3))

    A[index2-index1-1,index2-index1-3] = -(alpha*f3/(index2+1)-alpha*f3)*f2/(3.*(f2+f3))
    A[index2-index1-1,index2-index1-2] = (alpha*f3/(index2+1)-alpha*f3)*4.*f2/(3.*(f2+f3))
    A[index2-index1-1,index2-index1-1] = 1.+2.*alpha*f3 + (alpha*f3/(index2+1)-alpha*f3)*4.*f3/(3.*(f2+f3))
    A[index2-index1-1,index2-index1] = -(alpha*f3/(index2+1)+alpha*f3) - (alpha*f3/(index2+1)-alpha*f3)*f3/(3.*(f2+f3))

    # Regime 2
    Bd2 = np.full(index2-index1-1,1.-2.*alpha*f2)
    Bplus2 = np.zeros(index2-index1-2)
    Bmin2 = np.zeros(index2-index1-2)
    Bplus2[:] = [alpha*f2/(i+1+index1)+alpha*f2 for i in range(index2-index1-2)]
    Bmin2[:] = [-alpha*f2/(i+2+index1)+alpha*f2 for i in range(index2-index1-2)]

    # Regime 3
    Bd3 = np.full(N-1-index2,1.-2.*alpha*f3)
    Bplus3 = np.zeros(N-2-index2)
    Bmin3 = np.zeros(N-2-index2)
    Bplus3[:] = [alpha*f3/(i+1+index2)+alpha*f3 for i in range(N-2-index2)]
    Bmin3[:-1] = [-alpha*f3/(i+2+index2)+alpha*f3 for i in range(N-3-index2)]
    Bmin3[-1] = 2.*alpha*f3

    B = diags([np.append(Bd2,Bd3),np.append(np.append(Bplus2,0),Bplus3),np.append(np.append(Bmin2,0),Bmin3)],[0,1,-1]).toarray()

    B[index2-index1-2,index2-index1-3] = (-alpha*f2/(index2-1)+alpha*f2) - (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f2+f3))
    B[index2-index1-2,index2-index1-2] = 1.-2.*alpha*f2 + (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f2+f3))
    B[index2-index1-2,index2-index1-1] = (alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f2+f3))
    B[index2-index1-2,index2-index1] = -(alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f2+f3))

    B[index2-index1-1,index2-index1-3] = -(-alpha*f3/(index2+1)+alpha*f3)*f2/(3.*(f2+f3))
    B[index2-index1-1,index2-index1-2] = (-alpha*f3/(index2+1)+alpha*f3)*4.*f2/(3.*(f2+f3))
    B[index2-index1-1,index2-index1-1] = 1.-2.*alpha*f3 + (-alpha*f3/(index2+1)+alpha*f3)*4.*f3/(3.*(f2+f3))
    B[index2-index1-1,index2-index1] = (alpha*f3/(index2+1)+alpha*f3) - (-alpha*f3/(index2+1)+alpha*f3)*f3/(3.*(f2+f3))

    b = np.zeros((N-index1-2))
    b[0] = -2.*alpha*f2/(1+index1)+2.*alpha*f2
    #evaluate right hand side at t=0
    bb = B.dot(np.append(u[index1+1:index2],u[index2+1:])) + b
    
    Ainv = np.linalg.inv(A)
    
    kstart = 0

    # In case the sphere and the coating touch (interface at same lattice site)
    # No regime 1
    while (index0==index1) and (kstart<nsteps):
        
        # Do the loop (as before)
        edge[kstart+1] = max(edge[kstart] + S*dt/(2*dx)*(4*u[index0+1]-u[index0+2]-3*u[index0])/2,0)
        
        #find solution inside domain
        temp = Ainv@bb
        u[index0+1:index2] = temp[0:index2-index0-1]
        u[index2+1:] = temp[index2-index0-1:]
        u[index2] = (f2*(4.*u[index2-1]-u[index2-2]) + f3*(4.*u[index2+1]-u[index2+2]))/(3.*(f2+f3))
        
        edge[kstart+1] = max(edge[kstart+1] + S*dt/(2*dx)*(4*u[index0+1]-u[index0+2]-3*u[index0])/2,0)
        release4[kstart+1] = 3.*S*integrate.simps(u[index2+1:]*(x[index2+1:]**2.), x[index2+1:])
        
        #update right hand side
        bb = B.dot(np.append(u[index0+1:index2],u[index2+1:])) + b
        
        kstart += 1
        
        if round(edge[kstart]/dx)<round(edge[kstart-1]/dx):
            
            edge[kstart] = max(edge[kstart]**3.-3.*S*edge[kstart]**2*dx*int(round(edge[kstart-1]/dx)-round(edge[kstart]/dx)),0)**(1./3.)
            index0 = int(round(edge[kstart]/dx))
            
        if kstart % interval == 0:
            writefile = open('progress'+experiment+'.txt', 'a+')
            writefile.write('Real time: '+str(days)+' days\n')
            writefile.write('Simulation time: '+str(time.time()-start_time)+' seconds\n')
            writefile.write('Remaining time: '+str((time.time()-start_time)*(real_time-days)/days)+' seconds\n')
            writefile.write('\n')
            writefile.close()
            days += 1
            
    # Construct matrices for next scenario
    A[0][0] = 1.+2.*f2*alpha+(alpha*f2/(1+index1)-alpha*f2)*f2/(1.+f2)
    B[0][0] = 1.-2.*f2*alpha-(alpha*f2/(1+index1)-alpha*f2)*f2/(1.+f2)
    b[0] = (-2.*alpha*f2/(1+index1)+2.*alpha*f2)/(1.+f2)
    #evaluate right hand side at t=0
    bb = B.dot(np.append(u[index1+1:index2],u[index2+1:])) + b
    
    Ainv = np.linalg.inv(A)
        
    # In case the sphere and the coating touch (interface at neighbouring lattice site)
    # No regime 1
    while (index0+1==index1) and (kstart<nsteps):
        
        # Do the other loop (with lower order discretisation at inner boundary by necessity)
        edge[kstart+1] = max(edge[kstart] + S*dt/(2*dx)*(u[index0+1]-u[index0]),0)
        
        #find solution inside domain
        temp = Ainv@bb
        u[index1+1:index2] = temp[0:index2-index1-1]
        u[index2+1:] = temp[index2-index1-1:]
        u[index1] = (1.+f2*u[index1+1])/(1.+f2)
        u[index2] = (f2*(4.*u[index2-1]-u[index2-2]) + f3*(4.*u[index2+1]-u[index2+2]))/(3.*(f2+f3))
        
        # profile[kstart+1] = u
        
        edge[kstart+1] = max(edge[kstart+1] + S*dt/(2*dx)*(u[index0+1]-u[index0]),0)
        release4[kstart+1] = 3.*S*integrate.simps(u[index2+1:]*(x[index2+1:]**2.), x[index2+1:])
        
        #update right hand side
        bb = B.dot(np.append(u[index1+1:index2],u[index2+1:])) + b
        
        kstart += 1
        
        if round(edge[kstart]/dx)<round(edge[kstart-1]/dx):
            
            edge[kstart] = max(edge[kstart]**3.-3.*S*edge[kstart]**2*dx*int(round(edge[kstart-1]/dx)-round(edge[kstart]/dx)),0)**(1./3.)
            index0 = int(round(edge[kstart]/dx))

        if kstart % interval == 0:
            writefile = open('progress'+experiment+'.txt', 'a+')
            writefile.write('Real time: '+str(days)+' days\n')
            writefile.write('Simulation time: '+str(time.time()-start_time)+' seconds\n')
            writefile.write('Remaining time: '+str((time.time()-start_time)*(real_time-days)/days)+' seconds\n')
            writefile.write('\n')
            writefile.close()
            days += 1

    if (kstart<nsteps):
        # Construct matrices for next scenario

        # Regime 1
        diag = np.full(index1-index0-1,1.+2.*alpha)
        diagplus = np.zeros(index1-index0-2)
        diagmin = np.zeros(index1-index0-2)
        diagplus[:] = [-alpha/(i+1+index0)-alpha for i in range(index1-index0-2)]
        diagmin[:] = [alpha/(i+2+index0)-alpha for i in range(index1-index0-2)]

        # Regime 2
        diag2 = np.full(index2-index1-1,1.+2.*alpha*f2)
        diagplus2 = np.zeros(index2-index1-2)
        diagmin2 = np.zeros(index2-index1-2)
        diagplus2[:] = [-alpha*f2/(i+1+index1)-alpha*f2 for i in range(index2-index1-2)]
        diagmin2[:] = [alpha*f2/(i+2+index1)-alpha*f2 for i in range(index2-index1-2)]

        # Regime 3
        diag3 = np.full(N-1-index2,1.+2.*alpha*f3)
        diagplus3 = np.zeros(N-2-index2)
        diagmin3 = np.zeros(N-2-index2)
        diagplus3[:] = [-alpha*f3/(i+1+index2)-alpha*f3 for i in range(N-2-index2)]
        diagmin3[:-1] = [alpha*f3/(i+2+index2)-alpha*f3 for i in range(N-3-index2)]
        diagmin3[-1] = -2.*alpha*f3
        
        A = diags([np.append(np.append(diag,diag2),diag3),np.append(np.append(np.append(np.append(diagplus,0),diagplus2),0),diagplus3),np.append(np.append(np.append(np.append(diagmin,0),diagmin2),0),diagmin3)],[0,1,-1]).toarray()

        # Fix boundary condition first interface from below
        A[index1-index0-2,index1-index0-2] = 1.+2.*alpha - (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
        A[index1-index0-2,index1-index0-1] = -(alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
        A[index1-index0-2,index1-index0] = (alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

        # Fix boundary condition first interface from above
        A[index1-index0-1,index1-index0-2] = (alpha*f2/(index1+1)-alpha*f2)*4./(3.*(1.+f2))
        A[index1-index0-1,index1-index0-1] = 1.+2.*alpha*f2 + (alpha*f2/(index1+1)-alpha*f2)*4.*f2/(3.*(1.+f2))
        A[index1-index0-1,index1-index0] = -(alpha*f2/(index1+1)+alpha*f2) - (alpha*f2/(index1+1)-alpha*f2)*f2/(3.*(1.+f2))

        # Fix boundary condition second interface from below
        A[index2-index0-3,index2-index0-4] = (alpha*f2/(index2-1)-alpha*f2) + (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f3+f2))
        A[index2-index0-3,index2-index0-3] = 1.+2.*alpha*f2 - (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f3+f2))
        A[index2-index0-3,index2-index0-2] = -(alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f3+f2))
        A[index2-index0-3,index2-index0-1] = (alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f3+f2))

        # Fix boundary condition second interface from above
        A[index2-index0-2,index2-index0-4] = -(alpha*f3/(index2+1)-alpha*f3)*f2/(3.*(f3+f2))
        A[index2-index0-2,index2-index0-3] = (alpha*f3/(index2+1)-alpha*f3)*4.*f2/(3.*(f3+f2))
        A[index2-index0-2,index2-index0-2] = 1.+2.*alpha*f3 + (alpha*f3/(index2+1)-alpha*f3)*4.*f3/(3.*(f3+f2))
        A[index2-index0-2,index2-index0-1] = -(alpha*f3/(index2+1)+alpha*f3) - (alpha*f3/(index2+1)-alpha*f3)*f3/(3.*(f3+f2))
        
        # Regime 1
        Bd = np.full(index1-index0-1,1.-2.*alpha)
        Bplus = np.zeros(index1-index0-2)
        Bmin = np.zeros(index1-index0-2)
        Bplus[:] = [alpha/(i+1+index0)+alpha for i in range(index1-index0-2)]
        Bmin[:] = [-alpha/(i+2+index0)+alpha for i in range(index1-index0-2)]

        # Regime 2
        Bd2 = np.full(index2-index1-1,1.-2.*alpha*f2)
        Bplus2 = np.zeros(index2-index1-2)
        Bmin2 = np.zeros(index2-index1-2)
        Bplus2[:] = [alpha*f2/(i+1+index1)+alpha*f2 for i in range(index2-index1-2)]
        Bmin2[:] = [-alpha*f2/(i+2+index1)+alpha*f2 for i in range(index2-index1-2)]

        # Regime 3
        Bd3 = np.full(N-1-index2,1.-2.*alpha*f3)
        Bplus3 = np.zeros(N-2-index2)
        Bmin3 = np.zeros(N-2-index2)
        Bplus3[:] = [alpha*f3/(i+1+index2)+alpha*f3 for i in range(N-2-index2)]
        Bmin3[:-1] = [-alpha*f3/(i+2+index2)+alpha*f3 for i in range(N-3-index2)]
        Bmin3[-1] = 2.*alpha*f3

        B = diags([np.append(np.append(Bd,Bd2),Bd3),np.append(np.append(np.append(np.append(Bplus,0),Bplus2),0),Bplus3),np.append(np.append(np.append(np.append(Bmin,0),Bmin2),0),Bmin3)],[0,1,-1]).toarray()

        # Fix boundary condition first interface from below
        B[index1-index0-2,index1-index0-2] = 1.-2.*alpha + (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
        B[index1-index0-2,index1-index0-1] = (alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
        B[index1-index0-2,index1-index0] = -(alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

        # Fix boundary condition first interface from above
        B[index1-index0-1,index1-index0-2] = (-alpha*f2/(index1+1)+alpha*f2)*4./(3.*(1.+f2))
        B[index1-index0-1,index1-index0-1] = 1.-2.*alpha*f2 + (-alpha*f2/(index1+1)+alpha*f2)*4.*f2/(3.*(1.+f2))
        B[index1-index0-1,index1-index0] = (alpha*f2/(index1+1)+alpha*f2) - (-alpha*f2/(index1+1)+alpha*f2)*f2/(3.*(1.+f2))

        # Fix boundary condition second interface from below
        B[index2-index0-3,index2-index0-4] = (-alpha*f2/(index2-1)+alpha*f2) - (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f2+f3))
        B[index2-index0-3,index2-index0-3] = 1.-2.*alpha*f2 + (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f2+f3))
        B[index2-index0-3,index2-index0-2] = (alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f2+f3))
        B[index2-index0-3,index2-index0-1] = -(alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f2+f3))

        # Fix boundary condition second interface from above
        B[index2-index0-2,index2-index0-4] = -(-alpha*f3/(index2+1)+alpha*f3)*f2/(3.*(f2+f3))
        B[index2-index0-2,index2-index0-3] = (-alpha*f3/(index2+1)+alpha*f3)*4.*f2/(3.*(f2+f3))
        B[index2-index0-2,index2-index0-2] = 1.-2.*alpha*f3 + (-alpha*f3/(index2+1)+alpha*f3)*4.*f3/(3.*(f2+f3))
        B[index2-index0-2,index2-index0-1] = (alpha*f3/(index2+1)+alpha*f3) - (-alpha*f3/(index2+1)+alpha*f3)*f3/(3.*(f2+f3))
        
        # Boundary condition at the lattice site next to the solid sphere
        b = np.zeros((N-index0-3))
        b[0] = 2.*(alpha-alpha/(index0+1)-(alpha/(index0+1)+alpha)/(3.*(1.+f2)))
        # CHANGED INDEX0+2 TO INDEX1+1 IN LINE BELOW
        b[1] = 2.*((alpha*f2/(index1+1)-alpha*f2)/(3.*(1.+f2)))

        # Update right hand side
        bb = B.dot(np.append(np.append(u[index0+1:index1],u[index1+1:index2]),u[index2+1:])) + b
        
        Ainv = np.linalg.inv(A)
               
    # In case the sphere and the coating almost touch (one lattice site between interfaces)
    # One lattice site in regime 1, but multiple sites feel the boundary at index0 through the interface
    while (index0+2==index1) and (kstart<nsteps):
               
        edge[kstart+1] = max(edge[kstart] + S*dt/(2*dx)*(4*u[index0+1]-u[index0+2]-3*u[index0])/2,0)
        
        #find solution inside domain
        temp = Ainv@bb
        u[index0+1:index1] = temp[:index1-index0-1]
        u[index1+1:index2] = temp[index1-index0-1:index2-index0-2]
        u[index2+1:] = temp[index2-index0-2:]
        u[index1] = ((4.*u[index1-1]-u[index1-2]) + f2*(4.*u[index1+1]-u[index1+2]))/(3.*(1+f2))
        u[index2] = (f2*(4.*u[index2-1]-u[index2-2]) + f3*(4.*u[index2+1]-u[index2+2]))/(3.*(f2+f3))
        
        # profile[kstart+1] = u
        
        edge[kstart+1] = max(edge[kstart+1] + S*dt/(2*dx)*(4*u[index0+1]-u[index0+2]-3*u[index0])/2,0)
        release4[kstart+1] = 3.*S*integrate.simps(u[index2+1:]*(x[index2+1:]**2.), x[index2+1:])
        
        # Update right hand side
        bb = B.dot(np.append(np.append(u[index0+1:index1],u[index1+1:index2]),u[index2+1:])) + b
               
        kstart += 1
        
        if round(edge[kstart]/dx)<round(edge[kstart-1]/dx):
            edge[kstart] = max(edge[kstart]**3.-3.*S*edge[kstart]**2*dx*int(round(edge[kstart-1]/dx)-round(edge[kstart]/dx)),0)**(1./3.)
            index0 = int(round(edge[kstart]/dx))
            
        if kstart % interval == 0:
            writefile = open('progress'+experiment+'.txt', 'a+')
            writefile.write('Real time: '+str(days)+' days\n')
            writefile.write('Simulation time: '+str(time.time()-start_time)+' seconds\n')
            writefile.write('Remaining time: '+str((time.time()-start_time)*(real_time-days)/days)+' seconds\n')
            writefile.write('\n')
            writefile.close()
            days += 1
    
    if (kstart<nsteps):
        
        # Regime 1
        diag = np.full(index1-index0-1,1.+2.*alpha)
        diagplus = np.zeros(index1-index0-2)
        diagmin = np.zeros(index1-index0-2)
        diagplus[:] = [-alpha/(i+1+index0)-alpha for i in range(index1-index0-2)]
        diagmin[:] = [alpha/(i+2+index0)-alpha for i in range(index1-index0-2)]

        # Regime 2
        diag2 = np.full(index2-index1-1,1.+2.*alpha*f2)
        diagplus2 = np.zeros(index2-index1-2)
        diagmin2 = np.zeros(index2-index1-2)
        diagplus2[:] = [-alpha*f2/(i+1+index1)-alpha*f2 for i in range(index2-index1-2)]
        diagmin2[:] = [alpha*f2/(i+2+index1)-alpha*f2 for i in range(index2-index1-2)]

        # Regime 3
        diag3 = np.full(N-1-index2,1.+2.*alpha*f3)
        diagplus3 = np.zeros(N-2-index2)
        diagmin3 = np.zeros(N-2-index2)
        diagplus3[:] = [-alpha*f3/(i+1+index2)-alpha*f3 for i in range(N-2-index2)]
        diagmin3[:-1] = [alpha*f3/(i+2+index2)-alpha*f3 for i in range(N-3-index2)]
        diagmin3[-1] = -2.*alpha*f3

        A = diags([np.append(np.append(diag,diag2),diag3),np.append(np.append(np.append(np.append(diagplus,0),diagplus2),0),diagplus3),np.append(np.append(np.append(np.append(diagmin,0),diagmin2),0),diagmin3)],[0,1,-1]).toarray()

        # Fix boundary condition first interface from below
        A[index1-index0-2,index1-index0-3] = (alpha/(index1-1)-alpha) + (alpha/(index1-1)+alpha)/(3.*(1.+f2))
        A[index1-index0-2,index1-index0-2] = 1.+2.*alpha - (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
        A[index1-index0-2,index1-index0-1] = -(alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
        A[index1-index0-2,index1-index0] = (alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

        # Fix boundary condition first interface from above
        A[index1-index0-1,index1-index0-3] = -(alpha*f2/(index1+1)-alpha*f2)/(3.*(1.+f2))
        A[index1-index0-1,index1-index0-2] = (alpha*f2/(index1+1)-alpha*f2)*4./(3.*(1.+f2))
        A[index1-index0-1,index1-index0-1] = 1.+2.*alpha*f2 + (alpha*f2/(index1+1)-alpha*f2)*4.*f2/(3.*(1.+f2))
        A[index1-index0-1,index1-index0] = -(alpha*f2/(index1+1)+alpha*f2) - (alpha*f2/(index1+1)-alpha*f2)*f2/(3.*(1.+f2))

        # Fix boundary condition second interface from below
        A[index2-index0-3,index2-index0-4] = (alpha*f2/(index2-1)-alpha*f2) + (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f3+f2))
        A[index2-index0-3,index2-index0-3] = 1.+2.*alpha*f2 - (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f3+f2))
        A[index2-index0-3,index2-index0-2] = -(alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f3+f2))
        A[index2-index0-3,index2-index0-1] = (alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f3+f2))

        # Fix boundary condition second interface from above
        A[index2-index0-2,index2-index0-4] = -(alpha*f3/(index2+1)-alpha*f3)*f2/(3.*(f3+f2))
        A[index2-index0-2,index2-index0-3] = (alpha*f3/(index2+1)-alpha*f3)*4.*f2/(3.*(f3+f2))
        A[index2-index0-2,index2-index0-2] = 1.+2.*alpha*f3 + (alpha*f3/(index2+1)-alpha*f3)*4.*f3/(3.*(f3+f2))
        A[index2-index0-2,index2-index0-1] = -(alpha*f3/(index2+1)+alpha*f3) - (alpha*f3/(index2+1)-alpha*f3)*f3/(3.*(f3+f2))
        
        # Regime 1
        Bd = np.full(index1-index0-1,1.-2.*alpha)
        Bplus = np.zeros(index1-index0-2)
        Bmin = np.zeros(index1-index0-2)
        Bplus[:] = [alpha/(i+1+index0)+alpha for i in range(index1-index0-2)]
        Bmin[:] = [-alpha/(i+2+index0)+alpha for i in range(index1-index0-2)]

        # Regime 2
        Bd2 = np.full(index2-index1-1,1.-2.*alpha*f2)
        Bplus2 = np.zeros(index2-index1-2)
        Bmin2 = np.zeros(index2-index1-2)
        Bplus2[:] = [alpha*f2/(i+1+index1)+alpha*f2 for i in range(index2-index1-2)]
        Bmin2[:] = [-alpha*f2/(i+2+index1)+alpha*f2 for i in range(index2-index1-2)]

        # Regime 3
        Bd3 = np.full(N-1-index2,1.-2.*alpha*f3)
        Bplus3 = np.zeros(N-2-index2)
        Bmin3 = np.zeros(N-2-index2)
        Bplus3[:] = [alpha*f3/(i+1+index2)+alpha*f3 for i in range(N-2-index2)]
        Bmin3[:-1] = [-alpha*f3/(i+2+index2)+alpha*f3 for i in range(N-3-index2)]
        Bmin3[-1] = 2.*alpha*f3

        B = diags([np.append(np.append(Bd,Bd2),Bd3),np.append(np.append(np.append(np.append(Bplus,0),Bplus2),0),Bplus3),np.append(np.append(np.append(np.append(Bmin,0),Bmin2),0),Bmin3)],[0,1,-1]).toarray()

        # Fix boundary condition first interface from below
        B[index1-index0-2,index1-index0-3] = (-alpha/(index1-1)+alpha) - (alpha/(index0-1)+alpha)/(3.*(1.+f2))
        B[index1-index0-2,index1-index0-2] = 1.-2.*alpha + (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
        B[index1-index0-2,index1-index0-1] = (alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
        B[index1-index0-2,index1-index0] = -(alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

        # Fix boundary condition first interface from above
        B[index1-index0-1,index1-index0-3] = -(-alpha*f2/(index1+1)+alpha*f2)/(3.*(1.+f2))
        B[index1-index0-1,index1-index0-2] = (-alpha*f2/(index1+1)+alpha*f2)*4./(3.*(1.+f2))
        B[index1-index0-1,index1-index0-1] = 1.-2.*alpha*f2 + (-alpha*f2/(index1+1)+alpha*f2)*4.*f2/(3.*(1.+f2))
        B[index1-index0-1,index1-index0] = (alpha*f2/(index1+1)+alpha*f2) - (-alpha*f2/(index1+1)+alpha*f2)*f2/(3.*(1.+f2))

        # Fix boundary condition second interface from below
        B[index2-index0-3,index2-index0-4] = (-alpha*f2/(index2-1)+alpha*f2) - (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f2+f3))
        B[index2-index0-3,index2-index0-3] = 1.-2.*alpha*f2 + (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f2+f3))
        B[index2-index0-3,index2-index0-2] = (alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f2+f3))
        B[index2-index0-3,index2-index0-1] = -(alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f2+f3))

        # Fix boundary condition second interface from above
        B[index2-index0-2,index2-index0-4] = -(-alpha*f3/(index2+1)+alpha*f3)*f2/(3.*(f2+f3))
        B[index2-index0-2,index2-index0-3] = (-alpha*f3/(index2+1)+alpha*f3)*4.*f2/(3.*(f2+f3))
        B[index2-index0-2,index2-index0-2] = 1.-2.*alpha*f3 + (-alpha*f3/(index2+1)+alpha*f3)*4.*f3/(3.*(f2+f3))
        B[index2-index0-2,index2-index0-1] = (alpha*f3/(index2+1)+alpha*f3) - (-alpha*f3/(index2+1)+alpha*f3)*f3/(3.*(f2+f3))
        
        # Boundary condition at the lattice site next to the solid sphere
        b = np.zeros((N-index0-3))
        b[0] = -2.*alpha/(1+index0)+2.*alpha

        # Update right hand side
        bb = B.dot(np.append(np.append(u[index0+1:index1],u[index1+1:index2]),u[index2+1:])) + b

        # Create variable that checks if the sphere has fully dissolved
        var = 0
        
        Ainv = np.linalg.inv(A)
    
    # Main loop
    for k in range(kstart,nsteps):
        
        # Update radius of sphere, in two steps according to C-N stencil
        edge[k+1] = (1-var)*max(edge[k] + S*dt/(2*dx)*(4*u[index0+1]-u[index0+2]-3*u[index0])/2,0)
        
        #find solution inside domain
        temp = Ainv@bb
        u[index0+1-var:index1] = temp[:index1-index0-1+var]
        u[index1+1:index2] = temp[index1-index0-1+var:index2-index0-2+var]
        u[index2+1:] = temp[index2-index0-2+var:]
        u[index1] = ((4.*u[index1-1]-u[index1-2]) + f2*(4.*u[index1+1]-u[index1+2]))/(3.*(1+f2))
        u[index2] = (f2*(4.*u[index2-1]-u[index2-2]) + f3*(4.*u[index2+1]-u[index2+2]))/(3.*(f2+f3))
        
        # profile[k+1] = u
        
        # Second step in updating sphere radius
        edge[k+1] = (1-var)*max(edge[k+1] + S*dt/(2*dx)*(4*u[index0+1]-u[index0+2]-3*u[index0])/2,0)
        release4[k+1] = 3.*S*integrate.simps(u[index2+1:]*(x[index2+1:]**2.), x[index2+1:])
        
        # Consider the case that the sphere has fully dissolved
        # No flux boundary condition at origin
        if (edge[k+1]==0) and (var==0):
            
            edge[k+1] = max(edge[k+1]**3.-3.*S*edge[k+1]**2*dx*int(round(edge[k]/dx)-round(edge[k+1]/dx)),0)**(1./3.)
            index0 = 0

            # Regime 1
            diag = np.full(index1-index0,1.+2.*alpha)
            diagplus = np.zeros(index1-index0-1)
            diagmin = np.zeros(index1-index0-1)
            diagplus[1:] = [-alpha/(i+index0)-alpha for i in range(1,index1-index0-1)]
            diagmin[:] = [alpha/(i+1+index0)-alpha for i in range(index1-index0-1)]
            diagplus[0] = -2.*alpha

            # Regime 2
            diag2 = np.full(index2-index1-1,1.+2.*alpha*f2)
            diagplus2 = np.zeros(index2-index1-2)
            diagmin2 = np.zeros(index2-index1-2)
            diagplus2[:] = [-alpha*f2/(i+1+index1)-alpha*f2 for i in range(index2-index1-2)]
            diagmin2[:] = [alpha*f2/(i+2+index1)-alpha*f2 for i in range(index2-index1-2)]

            # Regime 3
            diag3 = np.full(N-1-index2,1.+2.*alpha*f3)
            diagplus3 = np.zeros(N-2-index2)
            diagmin3 = np.zeros(N-2-index2)
            diagplus3[:] = [-alpha*f3/(i+1+index2)-alpha*f3 for i in range(N-2-index2)]
            diagmin3[:-1] = [alpha*f3/(i+2+index2)-alpha*f3 for i in range(N-3-index2)]
            diagmin3[-1] = -2.*alpha*f3

            A = diags([np.append(np.append(diag,diag2),diag3),np.append(np.append(np.append(np.append(diagplus,0),diagplus2),0),diagplus3),np.append(np.append(np.append(np.append(diagmin,0),diagmin2),0),diagmin3)],[0,1,-1]).toarray()

            # No flux boundary condition at origion
            A[0][1] = -2.*alpha
            
            # Fix boundary condition first interface from below
            A[index1-index0-1,index1-index0-2] = (alpha/(index1-1)-alpha) + (alpha/(index1-1)+alpha)/(3.*(1.+f2))
            A[index1-index0-1,index1-index0-1] = 1.+2.*alpha - (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
            A[index1-index0-1,index1-index0] = -(alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
            A[index1-index0-1,index1-index0+1] = (alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

            # Fix boundary condition first interface from above
            A[index1-index0,index1-index0-2] = -(alpha*f2/(index1+1)-alpha*f2)/(3.*(1.+f2))
            A[index1-index0,index1-index0-1] = (alpha*f2/(index1+1)-alpha*f2)*4./(3.*(1.+f2))
            A[index1-index0,index1-index0] = 1.+2.*alpha*f2 + (alpha*f2/(index1+1)-alpha*f2)*4.*f2/(3.*(1.+f2))
            A[index1-index0,index1-index0+1] = -(alpha*f2/(index1+1)+alpha*f2) - (alpha*f2/(index1+1)-alpha*f2)*f2/(3.*(1.+f2))

            # Fix boundary condition second interface from below
            A[index2-index0-2,index2-index0-3] = (alpha*f2/(index2-1)-alpha*f2) + (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f3+f2))
            A[index2-index0-2,index2-index0-2] = 1.+2.*alpha*f2 - (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f3+f2))
            A[index2-index0-2,index2-index0-1] = -(alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f3+f2))
            A[index2-index0-2,index2-index0] = (alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f3+f2))

            # Fix boundary condition second interface from above
            A[index2-index0-1,index2-index0-3] = -(alpha*f3/(index2+1)-alpha*f3)*f2/(3.*(f3+f2))
            A[index2-index0-1,index2-index0-2] = (alpha*f3/(index2+1)-alpha*f3)*4.*f2/(3.*(f3+f2))
            A[index2-index0-1,index2-index0-1] = 1.+2.*alpha*f3 + (alpha*f3/(index2+1)-alpha*f3)*4.*f3/(3.*(f3+f2))
            A[index2-index0-1,index2-index0] = -(alpha*f3/(index2+1)+alpha*f3) - (alpha*f3/(index2+1)-alpha*f3)*f3/(3.*(f3+f2))
            
            # Regime 1
            Bd = np.full(index1-index0,1.-2.*alpha)
            Bplus = np.zeros(index1-index0-1)
            Bmin = np.zeros(index1-index0-1)
            Bplus[1:] = [alpha/(i+index0)+alpha for i in range(1,index1-index0-1)]
            Bmin[:] = [-alpha/(i+1+index0)+alpha for i in range(index1-index0-1)]
            Bplus[0] = 2.*alpha

            # Regime 2
            Bd2 = np.full(index2-index1-1,1.-2.*alpha*f2)
            Bplus2 = np.zeros(index2-index1-2)
            Bmin2 = np.zeros(index2-index1-2)
            Bplus2[:] = [alpha*f2/(i+1+index1)+alpha*f2 for i in range(index2-index1-2)]
            Bmin2[:] = [-alpha*f2/(i+2+index1)+alpha*f2 for i in range(index2-index1-2)]

            # Regime 3
            Bd3 = np.full(N-1-index2,1.-2.*alpha*f3)
            Bplus3 = np.zeros(N-2-index2)
            Bmin3 = np.zeros(N-2-index2)
            Bplus3[:] = [alpha*f3/(i+1+index2)+alpha*f3 for i in range(N-2-index2)]
            Bmin3[:-1] = [-alpha*f3/(i+2+index2)+alpha*f3 for i in range(N-3-index2)]
            Bmin3[-1] = 2.*alpha*f3

            B = diags([np.append(np.append(Bd,Bd2),Bd3),np.append(np.append(np.append(np.append(Bplus,0),Bplus2),0),Bplus3),np.append(np.append(np.append(np.append(Bmin,0),Bmin2),0),Bmin3)],[0,1,-1]).toarray()

            # No flux boundary condition at origin
            B[0][1] = 2.*alpha
            
            # Fix boundary condition first interface from below
            B[index1-index0-1,index1-index0-2] = (-alpha/(index1-1)+alpha) - (alpha/(index1-1)+alpha)/(3.*(1.+f2))
            B[index1-index0-1,index1-index0-1] = 1.-2.*alpha + (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
            B[index1-index0-1,index1-index0] = (alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
            B[index1-index0-1,index1-index0+1] = -(alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

            # Fix boundary condition first interface from above
            B[index1-index0,index1-index0-2] = -(-alpha*f2/(index1+1)+alpha*f2)/(3.*(1.+f2))
            B[index1-index0,index1-index0-1] = (-alpha*f2/(index1+1)+alpha*f2)*4./(3.*(1.+f2))
            B[index1-index0,index1-index0] = 1.-2.*alpha*f2 + (-alpha*f2/(index1+1)+alpha*f2)*4.*f2/(3.*(1.+f2))
            B[index1-index0,index1-index0+1] = (alpha*f2/(index1+1)+alpha*f2) - (-alpha*f2/(index1+1)+alpha*f2)*f2/(3.*(1.+f2))

            # Fix boundary condition second interface from below
            B[index2-index0-2,index2-index0-3] = (-alpha*f2/(index2-1)+alpha*f2) - (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f2+f3))
            B[index2-index0-2,index2-index0-2] = 1.-2.*alpha*f2 + (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f2+f3))
            B[index2-index0-2,index2-index0-1] = (alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f2+f3))
            B[index2-index0-2,index2-index0] = -(alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f2+f3))

            # Fix boundary condition second interface from above
            B[index2-index0-1,index2-index0-3] = -(-alpha*f3/(index2+1)+alpha*f3)*f2/(3.*(f2+f3))
            B[index2-index0-1,index2-index0-2] = (-alpha*f3/(index2+1)+alpha*f3)*4.*f2/(3.*(f2+f3))
            B[index2-index0-1,index2-index0-1] = 1.-2.*alpha*f3 + (-alpha*f3/(index2+1)+alpha*f3)*4.*f3/(3.*(f2+f3))
            B[index2-index0-1,index2-index0] = (alpha*f3/(index2+1)+alpha*f3) - (-alpha*f3/(index2+1)+alpha*f3)*f3/(3.*(f2+f3))
            
            # Boundary condition at the lattice site next to the solid sphere
            b = np.zeros((N-index0-2))
            
            var = 1
            
            Ainv = np.linalg.inv(A)
        
        elif (round(edge[k+1]/dx)<round(edge[k]/dx)) and (var==0):
            
            edge[k+1] = max(edge[k+1]**3.-3.*S*edge[k+1]**2*dx*int(round(edge[k]/dx)-round(edge[k+1]/dx)),0)**(1./3.)
            index0 = int(round(edge[k+1]/dx))

            # Regime 1
            diag = np.full(index1-index0-1,1.+2.*alpha)
            diagplus = np.zeros(index1-index0-2)
            diagmin = np.zeros(index1-index0-2)
            diagplus[:] = [-alpha/(i+1+index0)-alpha for i in range(index1-index0-2)]
            diagmin[:] = [alpha/(i+2+index0)-alpha for i in range(index1-index0-2)]

            # Regime 2
            diag2 = np.full(index2-index1-1,1.+2.*alpha*f2)
            diagplus2 = np.zeros(index2-index1-2)
            diagmin2 = np.zeros(index2-index1-2)
            diagplus2[:] = [-alpha*f2/(i+1+index1)-alpha*f2 for i in range(index2-index1-2)]
            diagmin2[:] = [alpha*f2/(i+2+index1)-alpha*f2 for i in range(index2-index1-2)]

            # Regime 3
            diag3 = np.full(N-1-index2,1.+2.*alpha*f3)
            diagplus3 = np.zeros(N-2-index2)
            diagmin3 = np.zeros(N-2-index2)
            diagplus3[:] = [-alpha*f3/(i+1+index2)-alpha*f3 for i in range(N-2-index2)]
            diagmin3[:-1] = [alpha*f3/(i+2+index2)-alpha*f3 for i in range(N-3-index2)]
            diagmin3[-1] = -2.*alpha*f3

            A = diags([np.append(np.append(diag,diag2),diag3),np.append(np.append(np.append(np.append(diagplus,0),diagplus2),0),diagplus3),np.append(np.append(np.append(np.append(diagmin,0),diagmin2),0),diagmin3)],[0,1,-1]).toarray()

            # Fix boundary condition first interface from below
            A[index1-index0-2,index1-index0-3] = (alpha/(index1-1)-alpha) + (alpha/(index1-1)+alpha)/(3.*(1.+f2))
            A[index1-index0-2,index1-index0-2] = 1.+2.*alpha - (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
            A[index1-index0-2,index1-index0-1] = -(alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
            A[index1-index0-2,index1-index0] = (alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

            # Fix boundary condition first interface from above
            A[index1-index0-1,index1-index0-3] = -(alpha*f2/(index1+1)-alpha*f2)/(3.*(1.+f2))
            A[index1-index0-1,index1-index0-2] = (alpha*f2/(index1+1)-alpha*f2)*4./(3.*(1.+f2))
            A[index1-index0-1,index1-index0-1] = 1.+2.*alpha*f2 + (alpha*f2/(index1+1)-alpha*f2)*4.*f2/(3.*(1.+f2))
            A[index1-index0-1,index1-index0] = -(alpha*f2/(index1+1)+alpha*f2) - (alpha*f2/(index1+1)-alpha*f2)*f2/(3.*(1.+f2))

            # Fix boundary condition second interface from below
            A[index2-index0-3,index2-index0-4] = (alpha*f2/(index2-1)-alpha*f2) + (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f3+f2))
            A[index2-index0-3,index2-index0-3] = 1.+2.*alpha*f2 - (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f3+f2))
            A[index2-index0-3,index2-index0-2] = -(alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f3+f2))
            A[index2-index0-3,index2-index0-1] = (alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f3+f2))

            # Fix boundary condition second interface from above
            A[index2-index0-2,index2-index0-4] = -(alpha*f3/(index2+1)-alpha*f3)*f2/(3.*(f3+f2))
            A[index2-index0-2,index2-index0-3] = (alpha*f3/(index2+1)-alpha*f3)*4.*f2/(3.*(f3+f2))
            A[index2-index0-2,index2-index0-2] = 1.+2.*alpha*f3 + (alpha*f3/(index2+1)-alpha*f3)*4.*f3/(3.*(f3+f2))
            A[index2-index0-2,index2-index0-1] = -(alpha*f3/(index2+1)+alpha*f3) - (alpha*f3/(index2+1)-alpha*f3)*f3/(3.*(f3+f2))
            
            # Regime 1
            Bd = np.full(index1-index0-1,1.-2.*alpha)
            Bplus = np.zeros(index1-index0-2)
            Bmin = np.zeros(index1-index0-2)
            Bplus[:] = [alpha/(i+1+index0)+alpha for i in range(index1-index0-2)]
            Bmin[:] = [-alpha/(i+2+index0)+alpha for i in range(index1-index0-2)]

            # Regime 2
            Bd2 = np.full(index2-index1-1,1.-2.*alpha*f2)
            Bplus2 = np.zeros(index2-index1-2)
            Bmin2 = np.zeros(index2-index1-2)
            Bplus2[:] = [alpha*f2/(i+1+index1)+alpha*f2 for i in range(index2-index1-2)]
            Bmin2[:] = [-alpha*f2/(i+2+index1)+alpha*f2 for i in range(index2-index1-2)]

            # Regime 3
            Bd3 = np.full(N-1-index2,1.-2.*alpha*f3)
            Bplus3 = np.zeros(N-2-index2)
            Bmin3 = np.zeros(N-2-index2)
            Bplus3[:] = [alpha*f3/(i+1+index2)+alpha*f3 for i in range(N-2-index2)]
            Bmin3[:-1] = [-alpha*f3/(i+2+index2)+alpha*f3 for i in range(N-3-index2)]
            Bmin3[-1] = 2.*alpha*f3

            B = diags([np.append(np.append(Bd,Bd2),Bd3),np.append(np.append(np.append(np.append(Bplus,0),Bplus2),0),Bplus3),np.append(np.append(np.append(np.append(Bmin,0),Bmin2),0),Bmin3)],[0,1,-1]).toarray()

            # Fix boundary condition first interface from below
            B[index1-index0-2,index1-index0-3] = (-alpha/(index1-1)+alpha) - (alpha/(index1-1)+alpha)/(3.*(1.+f2))
            B[index1-index0-2,index1-index0-2] = 1.-2.*alpha + (alpha/(index1-1)+alpha)*4./(3.*(1.+f2))
            B[index1-index0-2,index1-index0-1] = (alpha/(index1-1)+alpha)*4.*f2/(3.*(1.+f2))
            B[index1-index0-2,index1-index0] = -(alpha/(index1-1)+alpha)*f2/(3.*(1.+f2))

            # Fix boundary condition first interface from above
            B[index1-index0-1,index1-index0-3] = -(-alpha*f2/(index1+1)+alpha*f2)/(3.*(1.+f2))
            B[index1-index0-1,index1-index0-2] = (-alpha*f2/(index1+1)+alpha*f2)*4./(3.*(1.+f2))
            B[index1-index0-1,index1-index0-1] = 1.-2.*alpha*f2 + (-alpha*f2/(index1+1)+alpha*f2)*4.*f2/(3.*(1.+f2))
            B[index1-index0-1,index1-index0] = (alpha*f2/(index1+1)+alpha*f2) - (-alpha*f2/(index1+1)+alpha*f2)*f2/(3.*(1.+f2))

            # Fix boundary condition second interface from below
            B[index2-index0-3,index2-index0-4] = (-alpha*f2/(index2-1)+alpha*f2) - (alpha*f2/(index2-1)+alpha*f2)*f2/(3.*(f2+f3))
            B[index2-index0-3,index2-index0-3] = 1.-2.*alpha*f2 + (alpha*f2/(index2-1)+alpha*f2)*4.*f2/(3.*(f2+f3))
            B[index2-index0-3,index2-index0-2] = (alpha*f2/(index2-1)+alpha*f2)*4.*f3/(3.*(f2+f3))
            B[index2-index0-3,index2-index0-1] = -(alpha*f2/(index2-1)+alpha*f2)*f3/(3.*(f2+f3))

            # Fix boundary condition second interface from above
            B[index2-index0-2,index2-index0-4] = -(-alpha*f3/(index2+1)+alpha*f3)*f2/(3.*(f2+f3))
            B[index2-index0-2,index2-index0-3] = (-alpha*f3/(index2+1)+alpha*f3)*4.*f2/(3.*(f2+f3))
            B[index2-index0-2,index2-index0-2] = 1.-2.*alpha*f3 + (-alpha*f3/(index2+1)+alpha*f3)*4.*f3/(3.*(f2+f3))
            B[index2-index0-2,index2-index0-1] = (alpha*f3/(index2+1)+alpha*f3) - (-alpha*f3/(index2+1)+alpha*f3)*f3/(3.*(f2+f3))
            
            # Boundary condition at the lattice site next to the solid sphere
            b = np.zeros((N-index0-3))
            b[0] = -2.*alpha/(1+index0)+2.*alpha
            
            Ainv = np.linalg.inv(A)

        # Update right hand side
        bb = B.dot(np.append(np.append(u[index0+1-var:index1],u[index1+1:index2]),u[index2+1:])) + b
        
        if k % interval == 0:
            writefile = open('progress'+experiment+'.txt', 'a+')
            writefile.write('Real time: '+str(days)+' days\n')
            writefile.write('Simulation time: '+str(time.time()-start_time)+' seconds\n')
            writefile.write('Remaining time: '+str((time.time()-start_time)*(real_time-days)/days)+' seconds\n')
            writefile.write('\n')
            writefile.close()
            days += 1
            
    return release4, edge

release4, edge = loopsolve(nsteps,index0,alpha,S,u,index2,f3,index1,f2,dx)

# Open file to write data
hfwrite = h5py.File('Corbion_data_'+experiment+'.h5', 'w')

# Write data
# Parameters to check
hfwrite.create_dataset('real_radius',data=real_radius)
hfwrite.create_dataset('real_thickness',data=real_thickness)
hfwrite.create_dataset('real_Dp',data=real_Dp)
# Actual data
hfwrite.create_dataset('release',data=release4)
hfwrite.create_dataset('edge',data=edge)
hfwrite.create_dataset('tau',data=tau)

#Close file
hfwrite.close()