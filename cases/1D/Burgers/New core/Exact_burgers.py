import math
import numpy as np
#import matplotlib.pyplot as plt

# input position 
x = np.linspace(0, 2, 51)
print(x)
def f(x_0,xt):
      return math.sin(np.pi*x_0) - xt + x_0

def Exact_Burgers(x):

  tolerance = 1e-6
  y = np.zeros(np.size(x)) 
  for j in range(np.size(x)):
    xt = x[j]
    if abs(xt)< tolerance: # (0,0)
      yt = 0
    elif abs(xt-2) < tolerance: # (2,0)
      yt = 0
    elif (abs(xt-1)< tolerance):
      yt = 0   
    elif xt < 1: # (0,1)
      a = 0
      b = math.pi/2
      max_iterations = 100

      for i in range(max_iterations):
          c = (a + b) / 2
          if abs(f(c,xt)) < tolerance:
              print("Solution found:", c)
              break
          elif f(a,xt) * f(c,xt) < 0:
              b = c
          else:
              a = c
      yt = math.sin(np.pi*c)
    elif xt > 1: # (1,2)
      xt = 2- xt
      a = 0
      b = math.pi/2
      tolerance = 1e-6
      max_iterations = 100

      for i in range(max_iterations):
          c = (a + b) / 2
          if abs(f(c,xt)) < tolerance:
              print("Solution found:", c)
              break
          elif f(a,xt) * f(c,xt) < 0:
              b = c
          else:
              a = c
      yt =  -math.sin(np.pi*c)
    y[j] = yt
  return y

#y =  Exact_Burgers(x)
#plt.figure()
#plt.scatter(x,y)
#plt.show()
