
import numpy as np
import math

def IC_Riemann_1D(x,crhoL,cuL,cpL,crhoR,cuR,cpR):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              
    u_init = np.zeros((x.shape[0]))                                                
    p_init = np.zeros((x.shape[0]))                                                

    for i in range(N):
        if (x[i,1] <= 0.5):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR
            u_init[i] = cuR
            p_init[i] = cpR

    return rho_init, u_init, p_init

def IC_2Blast(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition

    # rho, p - initial condition
    for i in range(N):
        if (x[i,1] <= 0.1):
            rho_init[i] = 1
            p_init[i] =  1
        else: #if (x[i,1] <= 0.9):
            rho_init[i] = 1
            p_init[i] =  1e-4
        #else:
        #    rho_init[i] = 1
        #    p_init[i] =  1

    return rho_init, u_init, p_init
def IC_Blast(x):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              
    u_init = np.zeros((x.shape[0]))                                                
    p_init = np.zeros((x.shape[0]))                                                

    crhoL = 1
    cuL = 0
    cpL = 1
    #e = 1
    p = 1
    crhoR =1
    cuR = 0
    cpR = 0.01
    # rho, p - initial condition
    for i in range(N):
        if (abs(x[i,1] -0.5) < 0.1):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR
            u_init[i] = cuR
            p_init[i] = cpR

    return rho_init, u_init, p_init

def IC_123(x):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              
    u_init = np.zeros((x.shape[0]))                                                
    p_init = np.zeros((x.shape[0]))                                                

    crhoL = 1
    cuL = -2
    cpL = 0.4
    
    crhoR =1
    cuR = 2
    cpR = 0.4
    # rho, p - initial condition
    for i in range(N):
        if (x[i,1] <= 0.5):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR
            u_init[i] = cuR
            p_init[i] = cpR

    return rho_init, u_init, p_init
def IC_Sin(x):
    N = x.shape[0]
    u = np.zeros((N))                                                
    for i in range(N):
        u[i] = np.sin(np.pi*(x[i,1]))
    return u
def IC_Sin_Case1(x):
    N = x.shape[0]
    u = np.zeros((N))                                                
    for i in range(N):
        u[i] = np.sin(2*np.pi*(x[i,1])) +1
    return u
def IC_Sin_Case2(x):
    N = x.shape[0]
    u = np.zeros((N))                                                
    for i in range(N):
      xa = x[i,1]
      if (xa < 1 and xa > 0):
        u[i] = np.sin(2*np.pi*(xa)) +1
      else:
        u[i] = 1
    return u

def IC_ShuOsher(x):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              
    u_init = np.zeros((x.shape[0]))                                                
    p_init = np.zeros((x.shape[0]))                                                

    crhoL = 1.0
    cuL = 0.83219*2
    cpL = 1.0
    
    crhoR = 0.259259
    cuR = 0
    cpR = 0.096774

   # crhoL = 3.857143
   # cuL = 2.629369
   # cpL = 10.333333

   # crhoR = 1
   # cuR = 0
   # cpR = 1
    # rho, p - initial condition
    for i in range(N):
        if (x[i,1] <= 0.2):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR*(1+0.6*np.sin(10*x[i,1]))
            u_init[i] = cuR
            p_init[i] = cpR

    return rho_init, u_init, p_init

def IC_Combination_Wave(X):
    N = X.shape[0]
    u = np.zeros((X.shape[0])) 
    for i in range(N):
        x = X[i,1]
        u[i] = Combination_wave(x)
        #u[i] = np.sin(np.pi*(x))
    return u
def Combination_wave(x):
    
    delta = 0.005
    a = 0.5
    z = -0.7
    alpha = 10
    beta = np.log(2)/(36*delta**2)
    
    if (x<-0.6 and x>=-0.8):
        u = 1/6*(G(x,beta,z-delta) + G(x,beta,z+delta)+4*G(x,beta,z))
    elif (x<-0.2 and x>=-0.4):
        u = 1
    elif (x>0 and x<0.2):
        u = 1-np.abs(10*(x-0.1))
    elif (x > 0.4 and x<0.6):
        u = 1/6*(F(x,alpha,a-delta)+F(x,alpha,a+delta)+4*F(x,alpha,a))
    else:
        u = 0
    return u

    
def Con_F(T,N):
    L = 1
    u = 0
    x = np.zeros((N,2)) 
    for i in range(N):
        x[i,0] = T
        x[i,1] = -L + 2*L*i/N
        u = u + Combination_wave(x[i,1])
        
    u=u/N
    
    return x,u
        


def G(x,beta,z):
    return np.exp(-beta*(x-z)**2)
def F(x,alpha,a):
    return np.sqrt(np.maximum(1-alpha**2*(x-a)**2,0))

def Mesh_Data(num_x,num_t,Tstart,Tend, Xstart,Xend):
    x_ic = np.zeros((num_x,2))
    x_int = np.zeros((num_x*(num_t-1),2))
    
    x_bc =np.zeros((2*(num_t-1),2)) 
    
    dt = (Tend - Tstart)/num_t
    x =   np.linspace(Xstart, Xend, num_x) 
    x_ic[:,0] = 0
    x_ic[:,1] = x
    t = np.linspace(Tstart+dt, Tend, num_t-1)                                     
    x_bc[:num_t-1,0] = t
    x_bc[:num_t-1,1] = Xstart 
    x_bc[num_t-1:,0] = t
    x_bc[num_t-1:,1] = Xend

    
    t_grid, x_grid = np.meshgrid(t, x)                                 
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]                                      
    x_int = X[:, 0][:,None]                                        
    t_int = T[:, 0][:,None]                                        

    x_int = np.hstack((t_int, x_int))                            
    
    return x_ic,x_bc,x_int
 