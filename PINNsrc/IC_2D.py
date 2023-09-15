
import numpy as np
import math
def IC_Vortex(x):
  N =x.shape[0]
  rho_init = np.zeros((x.shape[0]))
  u_init = np.zeros((x.shape[0])) 
  v_init = np.zeros((x.shape[0]))
  p_init = np.zeros((x.shape[0]))
  x0 = 2.0
  y0 = 2.0
  GAMMA = 1.4
  b = 5.0
  pi = math.pi
  u_inf = 1.0
  p_inf = 1.0
  rho_inf = 1.0
  v_inf = 1.0



  for i in range(N):
    rx = x[i,1] - x0
    ry = x[i,2] - y0
    #if rx < -5:
    #    rx += 10
    #elif rx > 5:
    #    rx -= 10
    rsq = rx * rx + ry * ry
    rho_init[i] = math.pow(1.0 - ((GAMMA - 1.0) * b * b) / (8.0 * GAMMA * pi * pi) * math.exp(1.0 - rsq),
                   1.0 / (GAMMA - 1.0))
    p_init[i] = math.pow(rho_init[i], GAMMA)
    du = -b / (2.0 * pi) * math.exp(0.5 * (1.0 - rsq)) * ry
    dv = b / (2.0 * pi) * math.exp(0.5 * (1.0 - rsq)) * rx
    u_init[i] = u_inf + du
    v_init[i] = v_inf + dv
     # u0[p] = rho
     # u1[p] = rho * u
     # u2[p] = rho * v
     # u3[p] = P / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
  return rho_init, u_init, v_init,p_init

def IC_Constant(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    rho1 = 1.0
    p1 =  1.16
   # p1 =  1.458
    v1 = 0.0
    u1 = 1.0
    # rho, p - initial condition
    for i in range(N):
        #if x[i,1] < 0.5:
          rho_init[i] = rho1
          u_init[i] =   u1
          v_init[i] =  v1
          p_init[i] =  p1
        #else:
        #    rho_init[i] = rho2
        #    u_init[i] =   u2
        #    v_init[i] =  v2
        #    p_init[i] =  p2

    return rho_init, u_init, v_init,p_init

def IC_circle(t,xc,yc,r,r2,n):
    x = np.zeros((n,3)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        rr = np.random.rand()*(r2-r) + r
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*rr
        x[i,2] = yc  + yd*rr
    return x

def IC_circle_init(xc,yc,r,r2,n):
    x = np.zeros((n,3)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        rr = np.random.rand()*(r2-r) + r
        x[i,0] = 0 #np.random.rand()*t
        x[i,1] = xc  + xd*rr
        x[i,2] = yc  + yd*rr
    return x

def IC_Circle_Stretch(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    
    x0 = 0.5
    y0 = 0.75
    radius = 0.15
    
    for i in range(N):
        x1 = x[i,1]
        y1 = x[i,2]
        d = dist(x1,y1,x0,y0)
        if (d -radius < 0):
            u_init[i] =  1
        else:
            u_init[i] =  0
        

    return  u_init 
def IC_Zalesak(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    
    x0 = 0.5
    y0 = 0.5
    radius = 0.4
    
    for i in range(N):
        x1 = x[i,1]
        y1 = x[i,2]
        d = dist(x1,y1,x0,y0)
        if (d -radius >= 0 or ((y1< 0.6 and y1> 0.4) and x1>=0.5) ):
            u_init[i] =  0
        else:
            u_init[i] =  1
        

    return  u_init 


def dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def Velocity_Circle_Stretch(x,T):
    N =x.shape[0]
    ax = np.zeros((x.shape[0]))                                             
    ay = np.zeros((x.shape[0]))                                             
    x0 = 0.5*np.pi
    y0 = 0.7
    radius = 0.2
    for i in range(N):
        t = x[i,0]
        x1 = x[i,1]
        y1 = x[i,2]
        
        if (t < T/2):
          ax[i] =  (np.sin(x1*np.pi))**2*np.sin(2*y1*np.pi)
          ay[i] = -(np.sin(y1*np.pi))**2*np.sin(2*x1*np.pi)
        else:
          ax[i] = -(np.sin(x1*np.pi))**2*np.sin(2*y1*np.pi)
          ay[i] =  (np.sin(y1*np.pi))**2*np.sin(2*x1*np.pi)
    return ax,ay
        
def Velocity_Zalesak(x,T):
    N =x.shape[0]
    ax = np.zeros((x.shape[0]))                                             
    ay = np.zeros((x.shape[0]))                                             
    x0 = 0.5
    y0 = 0.5
    radius = 0.2
    for i in range(N):
        t = x[i,0]
        x1 = x[i,1]
        y1 = x[i,2]
        ax[i] =  2*np.pi*(y1-y0) #(np.sin(x1*np.pi))**2*np.sin(2*y1*np.pi)
        ay[i] = -2*np.pi*(x1-x0) #-(np.sin(y1*np.pi))**2*np.sin(2*x1*np.pi)
    return ax,ay

def Velocity_Circle_Stretch2(t,x):
    N =x.shape[0]
    ax = np.zeros((x.shape[0]))                                             
    ay = np.zeros((x.shape[0]))                                             
    x0 = 0.5
    y0 = 0.5
    radius = 0.15
    for i in range(N):
        x1 = x[i,0]
        y1 = x[i,1]
        T = 4
        
        ax[i] = -np.sin(4*np.pi*x1+2*np.pi)*np.sin(4*np.pi*y1+2*np.pi)*np.cos(np.pi*t/T)
        ay[i] = -np.cos(4*np.pi*x1+2*np.pi)*np.cos(4*np.pi*y1+2*np.pi)*np.cos(np.pi*t/T)
        
    return ax,ay