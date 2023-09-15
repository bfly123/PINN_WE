import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import math
import Exact_burgers
#import matplotlib

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
  #  random.seed(seed)
    torch.backends.cudnn.deterministic = True
WENO = np.loadtxt('1D/Burgers/result50WENO.dat')

num_seed = 50

f= open("result.dat", "w+") 
for nnode in [26,51,101,201]:
#for nnode in [25,51]:
  f.write('###########nnode: %d ########### \n' % nnode)
#nnode = 25 
  L2_smooth_ave = [] 
  L2_shock_ave = [] 
  L2_ave = []
  L2_max_ave = []
  loss_ave = []
  starttime = time.time()
  for i in range(num_seed):  
    seed = i 
    setup_seed(seed)
    f.write('****seed: %d ***** \n' % seed)
    #How to count the computing time


    def train(epoch):
        model.it = epoch
        def closure():
            optimizer.zero_grad()                                              
            loss_pde = model.loss_pde(x_int)                                   
            loss_ic = model.loss_ic(x_ic, u_ic)  
            loss_symetry = model.loss_symetry(x_int,x_int_mirror)  
            loss_zero = model.loss_zero(x_zero)
            loss = loss_pde + 10*loss_ic + loss_zero                                      

            print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f},loss_symetry:{loss_symetry:.8f},loss:{loss:.8f},loss_zero:{loss_zero:.8f}')
            #model.it = model.it + 1
            #outputfile = open('loss_history_burgers.dat','a+')
            #print(f'{model.it}  {loss_pde:.6f}  {loss_ic:.6f}  {loss:.6f}',file=outputfile)
            #outputfile.close() 
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item()
        print(f'epoch {epoch}: loss {loss_value:.6f}')
        return loss

    def gradients(outputs, inputs):
        return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

    def to_numpy(input):
        if isinstance(input, torch.Tensor):
            return input.detach().cpu().numpy()
        elif isinstance(input, np.ndarray):
            return input
        else:
            raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                            'np.ndarray, but got {}'.format(type(input)))

    def IC(x):
        N = len(x)
        u_init = np.zeros((x.shape[0]))                                                
        for i in range(N):
            u_init[i] = -np.sin(np.pi*(x[i,1]-1))
        return u_init

    class DNN(nn.Module):
        def __init__(self):
            super(DNN, self).__init__()
            self.net = nn.Sequential()                                                  
            self.net.add_module('Linear_layer_1', nn.Linear(2, 30))                     
            self.net.add_module('Tanh_layer_1', nn.Tanh())                              

            for num in range(2, 4):                                                     
                self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))       
                self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 
            self.net.add_module('Linear_layer_final', nn.Linear(30, 1))                 

        def forward(self, x):
            return self.net(x)
        def loss_symetry(self, x,x_mirror):
            y = self.net(x)                                                
            y_mirror = self.net(x_mirror)                                                
            f = (((y + y_mirror))**2).mean() 
            return f

        def loss_zero(self, x_zero):
            y = self.net(x_zero)                                                
            f = ((y)**2).mean() 
            return f
        def loss_pde(self, x):
            y = self.net(x)                                                
            u = y[:, 0:1]

            U = u**2/2

            dU_g = gradients(U, x)[0]                                  
            U_x = dU_g[:, 1:]
            du_g = gradients(u, x)[0]                                 
            u_t,u_x = du_g[:, :1],du_g[:,1:]
            d = 0.1*(abs(u_x)-u_x) + 1
            #d = 1

            f = (((u_t + U_x)/d)**2).mean() 

            return f
        def res_pde(self,x):
            y = self.net(x)
            Res = np.zeros((x.shape[0]))                                  

            u = y[:, 0:1]
            U = u**2/2
            dU_g = gradients(U, x)[0]                                 
            U_x = dU_g[:, 1:]
            du_g = gradients(u, x)[0]                                  
            u_t,u_x = du_g[:, :1],du_g[:,1:]
            Res = (u_t + U_x)**2 
            return Res 

        def lambda_pde(self,x):
            y = self.net(x)
            Res = np.zeros((x.shape[0]))                                  

            u = y[:, 0:1]
            du_g = gradients(u, x)[0]                                  
            u_t,u_x = du_g[:, :1],du_g[:,1:]
            d = 0.1*(abs(u_x)-u_x) + 1
            return  d


        def loss_ic(self, x_ic, u_ic):
            y_ic = self.net(x_ic)                                                      
            u_ic_nn = y_ic[:, 0]
            loss_ics = ((u_ic_nn - u_ic) ** 2).mean()
            return loss_ics

    device = torch.device('cuda')                                      
    #device = torch.device('cpu')                                      
    num_x = nnode                                                        
    num_t =  nnode                                                       
    num_i_train = nnode                                          
    num_f_train =  nnode*nnode                                         
    x = np.linspace(0, 2, num_x)                                   
    t = np.linspace(0, 1, num_t)                                   
    t_grid, x_grid = np.meshgrid(t, x)                             
    T = t_grid.flatten()[:, None]                                  
    X = x_grid.flatten()[:, None]                                  

    x_zero = np.zeros((num_t, 2))
    x_zero[:,0] = t
    x_zero[:,1] = 1

    id_ic = np.random.choice(num_x, num_i_train, replace=False)    
    id_f = np.random.choice(num_x*num_t, num_f_train, replace=False)

    x_ic = x_grid[id_ic, 0][:, None]                               
    t_ic = t_grid[id_ic, 0][:, None]                               
    x_ic_train = np.hstack((t_ic, x_ic))                               
    u_ic_train = IC(x_ic_train)                 

    x_int = X[:, 0][id_f, None]                                        
    t_int = T[:, 0][id_f, None]                                        
    x_int_train = np.hstack((t_int, x_int))                            
    x_int_mirror = np.hstack((t_int, 2-x_int))                            

    u_ic_train = IC(x_ic_train)                 

    x_ic = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
    x_int = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
    x_int_mirror = torch.tensor(x_int_mirror, requires_grad=True, dtype=torch.float32).to(device)
    x_zero = torch.tensor(x_zero, requires_grad=True, dtype=torch.float32).to(device)
    u_ic = torch.tensor(u_ic_train, dtype=torch.float32).to(device)

    model = DNN().to(device)

    #lr = 0.001

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, 
                                  max_iter = 20, 
                                  max_eval = None, 
                                  tolerance_grad = 1e-05, 
                                  tolerance_change = 1e-09, 
                                  history_size = 100, 
                                  line_search_fn = 'strong_wolfe')
    epochs = 1000
    tic = time.time()
    
    loss_test = 100.0
    num_epoch = 0
    for epoch in range(1, epochs+1):
      #if int(epoch/10) > num_epoch:
      #  loss_test = loss
      #  abs(loss - loss_test)
      loss=train(epoch)


    x = np.linspace(0.0, 2.0, nnode)                                  
    t = np.linspace(1.0, 1.0, 1)                                        
    t_grid, x_grid = np.meshgrid(t, x)                              
    T = t_grid.flatten()[:, None]                                   
    X = x_grid.flatten()[:, None]                                   
    x_test = np.hstack((T, X))                                      
    x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float32).to(device)
    u_pred = to_numpy(model(x_test))
    res = to_numpy(model.res_pde(x_test))
    d   = to_numpy(model.lambda_pde(x_test))


    y_e = Exact_burgers.Exact_Burgers(x)
    L2_error = np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/nnode)
    f.write('L2 error: %e\n' % L2_error)
    num = 0
    num_shock = 0
    sum = 0.0
    sum_shock = 0
    for i in range(nnode):
      if abs(x[i]-1)> 0.05:
        sum = sum +  np.sum((u_pred[i,0]-y_e[i])**2)
        num = num + 1
        #np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/100)
      else:
        sum_shock = sum_shock +  np.sum((u_pred[i,0]-y_e[i])**2)
        num_shock = num_shock + 1
    sum = sum/num
    sum_shock = sum_shock/num_shock
    L2_error_smooth = np.sqrt(sum)
    L2_error_shock = np.sqrt(sum_shock)
    #f.write('L2_smooth error: %e\n' % L2_error_smooth)
    #f.write('L2_shock  error: %e\n' % L2_error_shock)
    L2_Max =  (np.max(np.abs(u_pred[:,0]-y_e)))
    #f.write('Max error: %e\n' % L2_Max )
    #f.write('Loss: %e\n' % loss)
    L2_max_ave.append(L2_Max)
    L2_ave.append(L2_error)
    L2_smooth_ave.append(L2_error_smooth)
    L2_shock_ave.append(L2_error_shock)
    loss_ave.append(to_numpy(loss))
    #loss_ave = loss_ave + loss

  f.write('L2_max_ave: %e, L2_max_top : %e, L2_max_bottom %e\n' %(np.mean(L2_max_ave),np.max(L2_max_ave),np.min(L2_max_ave)))
  f.write('L2_ave: %e, L2_ave_top: %e, L2_ave_bottom: %e \n' %(np.mean(L2_ave), np.max(L2_ave), np.min(L2_ave)))
  f.write('L2_smooth_ave: %e, L2_smooth_ave_top: %e, L2_smooth_bottom: %e \n' % (np.mean(L2_smooth_ave), np.max(L2_smooth_ave), np.min(L2_smooth_ave)))
  f.write('L2_shock_ave: %e, L2_shock_top: %e, L2_shock_bottom: %e \n' % (np.mean(L2_shock_ave),np.max(L2_shock_ave),np.min(L2_shock_ave)))
  f.write('loss_ave: %e , loss_top: %e, loss_bottom: %e \n' % (np.mean(loss_ave),np.max(loss_ave),np.min(loss_ave)))
  #f.write('L2_max_ave_array: \n')
  #f.write(str(L2_max_ave))
  #f.write('\n L2_ave_array: \n')
  #f.write(str(L2_ave))
  #f.write('\n loss_ave_array: \n')
  #f.write(str(loss_ave))
  #L2_ave = L2_ave.mean()
  #L2_max_ave = L2_max_ave/num_seed  
  #loss_ave = loss_ave/num_seed
  endtime = time.time()
  train_time = (endtime - starttime)/num_seed
  f.write('\n train_time: %e\n' % (train_time/num_seed))
  #y_e_WENO = Exact_burgers.Exact_Burgers(WENO[:,0])
  #f.write('L2 error WENO: %e\n' % (np.sqrt(np.sum((WENO[:,1]-y_e_WENO)**2)/nnode)))
  #num = 0
  #sum = 0.0
  #for i in range(nnode):
  #  if abs(WENO[i,0]-1)> 0.05:
  #    sum = sum +  np.sum((WENO[i,1]-y_e_WENO)**2)
  #    num = num + 1
      #np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/100)
  #sum = sum/num
  #L2_error_smooth = np.sqrt(sum)
  #f.write('L2 error smmoth WENO: %e\n' % L2_error_smooth)
#for i in range(nnode):
#  if abs(x[i]-1)> 0.01:
#    sum = sum +  np.sum((red[i,0]-y_e_WENO)**2)
#    num = num + 1
#    #np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/100)
#sum = sum/num
#f.write('L2_smooth error: %e\n' % np.sqrt(sum))
#f.write('Max error: %e\n' % (np.max(np.abs(u_pred[:,0]-y_e))))
#f.write('Loss: %e\n' % loss)
f.close()

