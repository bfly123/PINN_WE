import numpy as np

def BC_Sin_L(x):
  N = x.shape[0]
  u = np.zeros((N))                                                
  for i in range(N):
    u[i] = np.sin(2*np.pi*(x[i,1]-x[i,0]))+1
  return u
def BC_Outflow1(x):
  N = x.shape[0]
  u = np.zeros((N))                                                
  for i in range(N):
    u[i] = np.sin(2*np.pi*(x[i,0]-x[i,1]))
  return u


def BC_R1(x):
  N = x.shape[0]
  u = np.zeros((N))
  for i in range(N):
    u[i] = 1
  return u

def BC_Constant(x):
  N = x.shape[0]
  u = np.zeros((N))
  for i in range(N):
    u[i] = 1
  return u