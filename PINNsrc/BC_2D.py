
import numpy as np
import math
def Naca0012data(x):
    a = 0.594689181
    b = 0.298222773  
    c = 0.127125232 
    d = 0.357907906 
    e = 0.291984971 
    f = 0.105174606 
    y1 = a*(b*np.sqrt(x) - c*x-d*x**2+e*x**3 - f*x**4)
    y2 = -y1
    dy1 =  a*(0.5*b/np.sqrt(x) - c - 2*d*x +3*e*x**2 - 4*f*x**3)
    dy2 = -a*(0.5*b/np.sqrt(x) - c - 2*d*x +3*e*x**2 - 4*f*x**3)
    return y1,y2,dy1,dy2

def BD_naca0012(t,xb,yb,n):
    x = np.zeros((2*n,3)) 
    sin = np.zeros((2*n,1)) 
    cos = np.zeros((2*n,1)) 

    for i in range(n):
        xd = np.random.rand()
        yd1,yd2,dy1,dy2 = Naca0012data(xd)
        
        x[i,0] = np.random.rand()*t
        x[i,1] = xb + xd
        x[i,2] = yb  + yd1
        cos[i,0] = -dy1/np.sqrt(dy1**2 + 1)
        sin[i,0] =   1/np.sqrt(dy1**2 + 1)
    for i in range(n):
        xd = np.random.rand()
        yd1,yd2,dy1,dy2 = Naca0012data(xd)
        
        x[i+n,0] = np.random.rand()*t
        x[i+n,1] = xb + xd
        x[i+n,2] = yb  + yd2
        cos[i+n,0] = -dy2/np.sqrt(dy2**2 + 1)
        sin[i+n,0] =  1/np.sqrt(dy2**2 + 1)
    return x, sin,cos
 
def Data_move(x,dt, dx,dy):
    N =x.shape[0]
    xL = np.zeros((N,3))
    
    for i in range(N):
        xL[i,0] = x[i,0] +dt
        xL[i,1] = x[i,1] +dx
        xL[i,2] = x[i,2] +dy
        
    return xL
  
def Pertur(x, dx):
    N =x.shape[0]
    xL = np.zeros((N,3))
    xR = np.zeros((N,3))
    xU = np.zeros((N,3))
    xD = np.zeros((N,3))
    
    for i in range(N):
        xL[i,0] = x[i,0]
        xR[i,0] = x[i,0]
        xU[i,0] = x[i,0]
        xD[i,0] = x[i,0]
        
        
        xL[i,1] = x[i,1] - dx
        xR[i,1] = x[i,1] + dx
        xU[i,1] = x[i,1]
        xD[i,1] = x[i,1]
        
        xL[i,2] = x[i,2] 
        xR[i,2] = x[i,2]
        xU[i,2] = x[i,2] + dx
        xD[i,2] = x[i,2] - dx
        
    return xL,xR,xU,xD
 
def BD_BackCorner(t,n):
    
    x = np.zeros((n,3)) 
    x2 = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    sin2 = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 
    cos2 = np.zeros((n,1)) 
    
    for i in range(n):
        x[i,0] = np.random.rand()*t
        x[i,1] = np.random.rand()*0.3 + 0.2
        x[i,2] = 1.5
        sin[i] = 1
        cos[i] = 0
    for i in range(n):
        x2[i,0] = np.random.rand()*t
        x2[i,1] = np.random.rand()*0.5
        x2[i,2] = 1.5
        sin2[i] = 1
        cos2[i] = 0
    x = np.vstack((x,x2))
    sin = np.vstack((sin,sin2))
    cos = np.vstack((cos,cos2))
    
    for i in range(n):
        x2[i,0] = np.random.rand()*t
        x2[i,1] = 0.5
        x2[i,2] = np.random.rand()*1.5
        sin2[i] = 0
        cos2[i] = 1
        
    x = np.vstack((x,x2))
    sin = np.vstack((sin,sin2))
    cos = np.vstack((cos,cos2))
    
    for i in range(n):
        x2[i,0] = np.random.rand()*t
        x2[i,1] = 0.5
        x2[i,2] = np.random.rand()*0.3 + 1.2
        sin2[i] = 0
        cos2[i] = 1
        
    x = np.vstack((x,x2))
    sin = np.vstack((sin,sin2))
    cos = np.vstack((cos,cos2))
        
    return x,sin,cos