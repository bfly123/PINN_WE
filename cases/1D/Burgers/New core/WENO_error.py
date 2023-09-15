import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import math
import Exact_burgers

import numpy
a = numpy.loadtxt('WENO_200.dat')
a[:,0]
n = np.size(a,0)
y_e = Exact_burgers.Exact_Burgers(a[:,0])
L2=np.sqrt(np.sum((a[:,1]-y_e)**2)/n)
print('L2 error %e' %L2)
num = 0 
sum = 0 
sum_shock = 0
num_shock = 0
for i in range(n):
  if abs(a[i,0]-1)> 0.05:
      sum = sum +  np.sum((a[i,1]-y_e[i])**2)
      num = num + 1
      #np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/100)
  else:
      sum_shock = sum_shock +  np.sum((a[i,1]-y_e[i])**2)
      num_shock = num_shock + 1
sum = sum/num
sum_shock = sum_shock/num_shock
L2_error_smooth = np.sqrt(sum)
L2_error_shock = np.sqrt(sum_shock)
print('L2 smooth %e' %L2_error_smooth)
print('L2 error shock %e' %L2_error_shock)

  