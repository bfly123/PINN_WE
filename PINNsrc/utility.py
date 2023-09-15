import numpy as np
def l2_relative_error(arr1, arr2):
    diff = arr1 - arr2
    squared_diff = np.square(diff)
    sum_squared_diff = np.sum(squared_diff)
    l2_norm_arr1 = np.linalg.norm(arr1)
    relative_error = np.sqrt(sum_squared_diff) / l2_norm_arr1 * 100
    return relative_error


def Unit_var(rhoL,uL,pL,rhoR,uR,pR,t):
  rhoref = max(rhoL,rhoR)
  pmax = max(pL,pR)
  umin = (uL+uR)/2
  uref = max(np.sqrt(pmax/rhoref),abs(uL-uR)/2)
  pref = uref**2*rhoref

  
  uLn = (uL-umin)/uref
  uRn = (uR-umin)/uref
  pLn = pL/pref
  pRn = pR/pref
  rhoLn = rhoL/rhoref
  rhoRn = rhoR/rhoref
  
  tn = t*uref
  
  return rhoLn,uLn,pLn,rhoRn,uRn,pRn, tn,rhoref,uref,pref

  
def Pertur_1D(x,T,dt,dx):
  N=x.shape[0]
  xs   = np.zeros((N,2)) 
  xsL  = np.zeros((N,2)) 
  xsR  = np.zeros((N,2)) 
  xsP  = np.zeros((N,2)) 
  xsPL = np.zeros((N,2)) 
  xsPR = np.zeros((N,2)) 
  
  for i in range(N):
      xs[i,1] = x[i,1]
      xs[i,0] = x[i,0] + T
      xsL[i,1] = xs[i,1] - dx
      xsL[i,0] = xs[i,0]
      xsR[i,1] = xs[i,1] + dx
      xsR[i,0] = xs[i,0]
      xsP[i,0] = xs[i,0] + dt
      xsP[i,1] = xs[i,1]
      xsPL[i,0] = xsP[i,0]
      xsPL[i,1] = xsP[i,1]+ dx
      xsPR[i,0] = xsP[i,0]
      xsPR[i,1] = xsP[i,1]- dx
      
  return xs,xsL,xsR,xsP,xsPL,xsPR

def Move_Time_1D(x,dt):
    N=x.shape[0]
    xen =np.zeros((N,2)) 
    
    for i in range(N):
        xen[i,1] = x[i,1]
        xen[i,0] = x[i,0] + dt
    return xen