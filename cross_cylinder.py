import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from numpy import arange, meshgrid
#
from koila import LazyTensor, lazy
from smt.sampling_methods import LHS
# Seeds
torch.manual_seed(123)
np.random.seed(123)
def train(epoch):
    model.train()
    def closure():
        optimizer.zero_grad()                                                     # Optimizer
       # (x_int_train,x_int_train) = lazy(x_int_train,x_int_train ,batch=0) 
        loss_pde = model.loss_pde(x_int_train)                                    # Loss function of PDE
        loss_ic = model.loss_ic(x_ic_train, rho_ic_train,u_ic_train,v_ic_train,p_ic_train)   # Loss function of IC
        #loss_cut = model.loss_bc1(x_cut_train,rho_cut_train,u_cut_train,v_cut_train,p_cut_train) 
        loss_bdI = model.bd_B(x_bcI_train, sin_bcI_train,cos_bcI_train)  

        loss_ib = loss_ic  +  loss_bdI 
        loss = loss_pde + 10*loss_ib

        # Print iteration, loss of PDE and ICs
        print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ib:{loss_ib:.8f}')
        loss.backward()
        return loss

    # Optimize loss function
    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss, float) else loss
    # Print total loss
    print(f'epoch {epoch}: loss {loss_value:.6f}')
    
# Calculate gradients using torch.autograd.grad
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))
def IC(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    rho1 = 2.112
    p1 =  3.011
    v1 = 0.0
    u1 = np.sqrt(1.4*p1/rho1)*0.728
    
    rho2 = 1.0
    p2 = 1.
    v2 = 0.0
    #u1 = ms*npsqrt(gamma)
    u2 = 0.0
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

def BC_L(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    #u1 = ms*npsqrt(gamma)
    # rho, p - initial condition
    rho1 = 2.112
    p1 =  3.011
    v1 = 0.0
    u1 = np.sqrt(1.4*p1/rho1)*0.728
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] =  u1
        v_init[i] =  v1
        p_init[i] =  p1
    return rho_init, u_init, v_init,p_init
def BC_R(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    ms = 2.0
    rho1 = 1.0
    p1 = 1.0
    v1 = 0.0
    u1 = 0
    # rho, p - initial condition
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] = u1
        v_init[i] = v1
        p_init[i] = p1

    return rho_init, u_init, v_init,p_init
def BC_Cut(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))
    v_init = np.zeros((x.shape[0]))
    p_init = np.zeros((x.shape[0]))
    
    gamma = 1.4
    ms = 2.0
    rho1 = 1.0
    p1 = 1.0
    v1 = 0.0
    u1 = 0
    # rho, p - initial condition
    for i in range(N):
        rho_init[i] = 10.01
        u_init[i] =  0
        v_init[i] = 0
        p_init[i] = 10.01

    return rho_init, u_init, v_init,p_init
    
class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()                                                  # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(3, 90))                     # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())                              # First activation Layer

        for num in range(2, 6):                                                     # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(90, 90))       # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(90, 4))                 # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    def bd_B(self,x,sin,cos):
        yb = self.net(x)
        rhob,pb,ub,vb = yb[:, 0:1], yb[:, 1:2], yb[:, 2:3],yb[:,3:]
        drhob_g = gradients(rhob, x)[0]                                      # Gradient [u_t, u_x]
        rhob_x, rhob_y = drhob_g[:, 1:2], drhob_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dub_g = gradients(ub, x)[0]                                      # Gradient [u_t, u_x]
        ub_x, ub_y = dub_g[:, 1:2], dub_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dvb_g = gradients(vb, x)[0]                                      # Gradient [u_t, u_x]
        vb_x, vb_y = dvb_g[:, 1:2], dvb_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dpb_g = gradients(pb, x)[0]                                      # Gradient [p_t, p_x]
        pb_x, pb_y = dpb_g[:, 1:2], dpb_g[:, 2:3]                            # Partial derivatives p_t, p_x
        
        deltau = ub_x + vb_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        #lam = (deltau) - deltau) + 1
        
        fb = (((ub*cos + vb*sin)/lam)**2).mean() +\
            (((pb_x*cos + pb_y*sin)/lam)**2).mean() +\
            (((rhob_x*cos + rhob_y*sin)/lam)**2).mean()
        return fb
    def bd_OY(self,x):
        y = self.net(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                            # Partial derivatives p_t, p_x
        
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        
        f = ((( u_y)/lam)**2).mean() +\
            ((( v_y)/lam)**2).mean() +\
            ((( p_y)/lam)**2).mean() +\
            ((( rho_y)/lam)**2).mean()
        return f
    
    def bd_OX(self,x):
        y = self.net(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                            # Partial derivatives p_t, p_x
        
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        
        f = ((( u_x)/lam)**2).mean() +\
            ((( v_x)/lam)**2).mean() +\
            ((( p_x)/lam)**2).mean() +\
            ((( rho_x)/lam)**2).mean()
        return f
     
    # Loss function for PDE
#    def loss_pde(self, x):
#        
#       # yL = self.net(x_intL_train)
#       # yR = self.net(x_intR_train)
#       # yU = self.net(x_intU_train)
#       # yD = self.net(x_intD_train)
#       # rhoL,pL,uL,vL = yL[:, 0:1], yL[:, 1:2], yL[:, 2:3],yL[:,3:]
#       # rhoR,pR,uR,vR = yR[:, 0:1], yR[:, 1:2], yR[:, 2:3],yR[:,3:]
#       # rhoU,pU,uU,vU = yU[:, 0:1], yU[:, 1:2], yU[:, 2:3],yU[:,3:]
#       # rhoD,pD,uD,vD = yD[:, 0:1], yD[:, 1:2], yD[:, 2:3],yD[:,3:]
#        y = self.net(x)
#        gamma = 1.4                                                    # Heat Capacity Ratio
#        epsilon = 1e-5
#        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
#        
#        rhoE = p/(gamma - 1) +0.5*rho*(u**2+v**2)
#        
#        f1 = rho*u
#        f2 = rho*u*u+p
#        f3 = rho*u*v
#        f4 = (rhoE+p)*u
#        
#        g1 = rho*v
#        g2 = rho*v*u
#        g3 = rho*v*v + p
#        g4 = (rhoE+p)*v
#        
#        drho_g = gradients(rho,x)[0]
#        U1_t = drho_g[:, :1]
#        dU2_g = gradients(f1,x)[0]
#        U2_t = dU2_g[:, :1]
#        dU3_g = gradients(g1,x)[0]
#        U3_t = dU3_g[:, :1]
#        dU4_g = gradients(rhoE,x)[0]
#        U4_t = dU4_g[:, :1]
#        
#        df1_g = gradients(f1, x)[0]                                  # Gradient [rho_t, rho_x]
#        f1_x = df1_g[:, 1:2]
#        df2_g = gradients(f2, x)[0]                                      # Gradient [u_t, u_x]
#        f2_x = df2_g[:, 1:2]
#        df3_g = gradients(f3, x)[0]                                      # Gradient [u_t, u_x]
#        f3_x = df3_g[:, 1:2]
#        df4_g = gradients(f4, x)[0]                                      # Gradient [u_t, u_x]
#        f4_x = df4_g[:, 1:2]
#        
#        dg1_g = gradients(g1, x)[0]                                  # Gradient [rho_t, rho_x]
#        g1_y = dg1_g[:, 2:3]
#        dg2_g = gradients(g2, x)[0]                                      # Gradient [u_t, u_x]
#        g2_y = dg2_g[:, 2:3]
#        dg3_g = gradients(g3, x)[0]                                      # Gradient [u_t, u_x]
#        g3_y = dg3_g[:, 2:3]
#        dg4_g = gradients(g4, x)[0]                                      # Gradient [u_t, u_x]
#        g4_y = dg4_g[:, 2:3]
#        
#        
#        du_g = gradients(u, x)[0]                                
#        u_x = du_g[:, 1:2]         
#        dv_g = gradients(v, x)[0]                    
#        v_y = dv_g[:, 2:3]         
#        
#      #  rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
#      #  gamma = 1.4                                                    # Heat Capacity Ratio
#      #  epsilon = 1e-5
#      #  s = torch.log((abs(p)+epsilon)/(abs(rho)+epsilon)**1.4)
#      #  eta = -rho*s
#      #  phi1 = -rho*u*s
#      #  phi2 = -rho*v*s
#      #  
#      #  drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
#      #  rho_t, rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2],drho_g[:,2:]
#      #  du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
#      #  u_t, u_x, u_y = du_g[:, :1], du_g[:, 1:2], du_g[:,2:]                            # Partial derivatives u_t, u_x
#      #  dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
#      #  v_t, v_x, v_y = dv_g[:, :1], dv_g[:, 1:2], dv_g[:,2:]                            # Partial derivatives u_t, u_x
#      #  
#      #  E = p/0.4 + 0.5*rho*(u**2+v**2)
#      #  EL = pL/0.4 + 0.5*rhoL*(uL**2+vL**2)
#      #  ER = pR/0.4 + 0.5*rhoR*(uR**2+vR**2)
#      #  EU = pU/0.4 + 0.5*rhoU*(uU**2+vU**2)
#      #  ED = pD/0.4 + 0.5*rhoD*(uD**2+vD**2)
#      #  dE_g = gradients(E, x)[0]                                      # Gradient [u_t, u_x]
#      #  E_t = dE_g[:, :1]
#      #  
#      #  
#      #  deta_g = gradients(eta, x)[0]                                      # Gradient [p_t, p_x]
#      #  eta_t, eta_x,eta_y = deta_g[:, :1], deta_g[:, 1:2],deta_g[2:3]                            # Partial derivatives p_t, p_x
#      #  dphi1_g = gradients(phi1, x)[0]                                      # Gradient [p_t, p_x]
#      #  dphi2_g = gradients(phi2, x)[0]                                      # Gradient [p_t, p_x]
#      #  phi1_t, phi1_x,phi1_y = dphi1_g[:, :1], dphi1_g[:, 1:2],dphi1_g[:,2:3]                           # Partial derivatives p_t, p_x
#      #  phi2_t, phi2_x,phi2_y = dphi2_g[:, :1], dphi2_g[:, 1:2],dphi2_g[:,2:3]                           # Partial derivatives p_t, p_x
#        
#        d = np.random.rand()
#        deltau = u_x + v_y
#        nab = abs(deltau) - deltau
#        
#        #a = np.sqrt(1.4*p/rho)
#       # q = 0.01*(rho*deltau**2)
#        
#        d = 1.0
#        lam = d*(0.1*nab) + 1
#        #lam = d + 1
#       # lam = 1/lam
#        
#        f = (((U1_t + f1_x+g1_y )/lam)**2).mean() +\
#            (((U2_t + f2_x+g2_y )/lam)**2).mean() +\
#            (((U3_t + f3_x+g3_y )/lam)**2).mean() +\
#            (((U4_t + f4_x+g4_y )/lam)**2).mean()
#
#      #  p = p+q
#        
#      #  dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
#      #  p_t, p_x, p_y = dp_g[:, :1], dp_g[:, 1:2], dp_g[:,2:]                            # Partial derivatives p_t, p_x
#      #  
#      #  s1 = rho_t + (rhoR*uR - rhoL*uL)/0.02 +  (rhoU*uU - rhoD*uD)/0.02 
#      #  s2 = u*rho_t + u_t*rho + (rhoR*uR*uR +pR - rhoL*uL*uL-pL)/0.02 \
#      #       +(rhoU*uU*vU - rhoD*uD*vD)/0.02 
#      #  s3 = v*rho_t + v_t*rho + (rhoU*vU*vU +pU - rhoD*vD*vD-pD)/0.02 \
#      #       +(rhoR*uR*vR - rhoL*uL*vL)/0.02 
#      #  s4 = E_t + ((ER+pR)*uR - (EL+pL)*uL)/0.02 + ((EU+pU)*vU - (ED+pD)*vD)/0.02 
#      #  
#      #  du_gg = gradients(u_x, x)[0]                                      # Gradient [u_t, u_x]
#      #  u_xx, u_xy = du_gg[:, :1], du_gg[:, 1:2]                            # Partial derivatives u_t, u_x
#      #  
#      #  dv_gg = gradients(v_y, x)[0]                                      # Gradient [u_t, u_x]
#      #  v_yx, v_yy = dv_gg[:, :1], dv_gg[:, 1:2]                            # Partial derivatives u_t, u_x
#      #  
#      #  vis = -0.1*(u_xx + v_yy)
##
#      #  f = (((rho_t+rho*deltau+u*rho_x + v*rho_y)/lam)**2).mean() +\
#      #      (((rho*u_t+rho*u*u_x+rho*v*u_y+p_x +rho*vis)/lam)**2).mean() +\
#      #      (((rho*v_t+rho*u*v_x+rho*v*v_y+p_y +rho*vis)/lam)**2).mean() +\
#      #      (((p_t+u*p_x+v*p_y+1.4*p*deltau +rho*vis)/lam)**2).mean() + \
#      #      ((abs(s1)+s1)**2).mean() +\
#      #      ((abs(s3)+s3)**2).mean() +\
#      #      ((abs(s2)+s2)**2).mean() +\
#      #      ((abs(s4)+s4)**2).mean()
#      #      #(((abs(eta_t+phi1_x + phi2_y)+eta_t+phi1_x+ phi2_y))**2).mean()
#    #
#      #      #((abs(rho-1) - (rho-1))**2).mean()   + \
#      #      #((abs(p-0.7) - (p-0.7))**2).mean() +\
#        return f

    def loss_pde(self, x):
        y = self.net(x)                                                # Neural network
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        gamma = 1.4                                                    # Heat Capacity Ratio
        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_t, rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2],drho_g[:,2:]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_t, u_x, u_y = du_g[:, :1], du_g[:, 1:2], du_g[:,2:]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_t, v_x, v_y = dv_g[:, :1], dv_g[:, 1:2], dv_g[:,2:]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_t, p_x, p_y = dp_g[:, :1], dp_g[:, 1:2], dp_g[:,2:]                            # Partial derivatives p_t, p_x
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        f = (((rho_t+rho*deltau+u*rho_x + v*rho_y)/lam)**2).mean() +\
            (((rho*u_t+rho*u*u_x+rho*v*u_y+p_x)/lam)**2).mean() +\
            (((rho*v_t+rho*u*v_x+rho*v*v_y+p_y)/lam)**2).mean() +\
            (((p_t+u*p_x+v*p_y+1.4*p*deltau)/lam)**2).mean()
        return f
      
      
    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics

    def loss_bc(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics
    def loss_bc1(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() 

        return loss_ics

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


def BD_circle(t,xc,yc,r,n):
    x = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*r
        x[i,2] = yc  + yd*r
        cos[i,0] = xd 
        sin[i,0] = yd
        #cos[i,0] = 1
        #sin[i,0] = 0
    return x, sin,cos

def BD_Star(t,xc,yc,r,n):
    x = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*r
        x[i,2] = yc  + yd*r
        cos[i,0] = xd 
        sin[i,0] = yd
        #cos[i,0] = 1
        #sin[i,0] = 0
    return x, sin,cos


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

# Solve Euler equations using PINNs
# def main():
  # Initialization
#device = torch.device('cuda')                                          # Run on CPU
device = torch.device('cuda')                                          # Run on CPU
lr = 0.001                                                           # Learning rate
num_ib = 30000                                                # Random sampled points from IC0
num_int = 200000                                                # Random sampled points in interior
Tend = 0.4
Lx = 2.0
Ly = 2.0
rx = 1.0
ry = 1.0
rd = 0.25
rd2 = 0.4


xlimits = np.array([[0.,Tend],[0.0, Lx], [0,Ly]])  #interal
sampling = LHS(xlimits=xlimits)
x_int_train = sampling(num_int)
x_int_train =  IC_circle(Tend,rx,ry,rd,rd2,num_int)

#xlimits = np.array([[0.,Tend],[0.5, 2], [0.5,2]])  #interal
#sampling = LHS(xlimits=xlimits)
#x_int1_train = sampling(num_int)
#x_int_train =  np.vstack((x_int_train,x_int1_train))

#A = []
#for i in range(num_int):
#    x = x_int_train[i,1]
#    y = x_int_train[i,2]
#    if ((x - rx)**2 +(y-ry)**2< rd**2):
#        A.append(i)
#x_int_train = np.delete(x_int_train,A,axis=0)

#xlimits = np.array([[0.0, Tend], [1.0, 4.0], [0.2,1.0]])
#sampling = LHS(xlimits=xlimits)
#x_int_train_add = sampling(3*num_int)
#x_int_train = np.vstack((x_int_train,x_int_train_add))

#x_intL_train,x_intR_train,x_intU_train,x_intD_train = Pertur(x_int_train, 0.01)

#xlimits = np.array([[0.,0.0],[0.0,Lx], [0.0,Ly]])  #interal
#sampling = LHS(xlimits=xlimits)
#x_ic_train = sampling(num_ib)
#A = []
#for i in range(num_ib):
#    x = x_ic_train[i,1]
#    y = x_ic_train[i,2]
#    if ((x - rx)**2 +(y-ry)**2< rd**2):
#        A.append(i)
#x_ic_train = np.delete(x_ic_train,A,axis=0)

x_ic_train =  IC_circle_init(rx,ry,rd,rd2,num_int)

#xlimits = np.array([[0.0, 0.0], [1.0, 4.0], [0.2,1.0]])
#sampling = LHS(xlimits=xlimits)
#x_ic_train_add = sampling(3*num_int)
#x_ic_train = np.vstack((x_ic_train,x_ic_train_add))

#xlimits = np.array([[0.0,0.0],[0.0, Lx], [0.0,Ly]])
#sampling = LHS(xlimits=xlimits)
#x_ic_train =  sampling(num_ib)
#A = []
#for i in range(num_ib):
#    x = x_ic_train[i,1]
#    y = x_ic_train[i,2]
#    if ((x-rx)**2 + (y-ry)**2 ) < rd**2:
#        A.append(i)
#x_ic_train = np.delete(x_ic_train,A,axis=0)


x_bcI_train,sin_bcI_train,cos_bcI_train = BD_circle(Tend,rx,ry,rd,num_ib)
#x_bcI_train,sin_bcI_train,cos_bcI_train = BD_BackCorner(Tend,num_ib)

rho_ic_train, u_ic_train,v_ic_train, p_ic_train = IC(x_ic_train)  


x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
#x_intL_train = torch.tensor(x_intL_train,dtype=torch.float32).to(device)
#x_intR_train = torch.tensor(x_intR_train,dtype=torch.float32).to(device)
#x_intU_train = torch.tensor(x_intU_train,dtype=torch.float32).to(device)
#x_intD_train = torch.tensor(x_intD_train,dtype=torch.float32).to(device)

x_bcI_train = torch.tensor(x_bcI_train, requires_grad=True, dtype=torch.float32).to(device)
sin_bcI_train = torch.tensor(sin_bcI_train, dtype=torch.float32).to(device)
cos_bcI_train = torch.tensor(cos_bcI_train, dtype=torch.float32).to(device)

#rho_cut_train = torch.tensor(rho_cut_train, dtype=torch.float32).to(device)
#u_cut_train = torch.tensor(u_cut_train, dtype=torch.float32).to(device)
#v_cut_train = torch.tensor(v_cut_train, dtype=torch.float32).to(device)
#p_cut_train = torch.tensor(p_cut_train, dtype=torch.float32).to(device)
#x_cut_train = torch.tensor(x_cut_train, dtype=torch.float32).to(device)

rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
v_ic_train = torch.tensor(v_ic_train, dtype=torch.float32).to(device)
p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)
x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)


model = DNN().to(device)

#optimizer = torch.optim.LBFGS(model.parameters(),lr=lr,max_iter=500)
#rho_ic_train.clone().detach().requires_grad_(True)
#u_ic_train.clone().detach().requires_grad_(True)
# Initialize neural network

print('Start training...')
model_path = 'Withoushock.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model = model.to(device)

lr = 0.001
#optimizer = torch.optim.LBFGS(model.parameters(),lr=lr,max_iter=500)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 10000
tic = time.time()
for epoch in range(1, epochs+1):
    train(epoch)
toc = time.time()
print(f'Total training time: {toc - tic}')

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_Adam,gamma=0.995)
optimizer = torch.optim.LBFGS(model.parameters(),lr=0.5,max_iter=20)
#optimizer_LBFGS = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=50000)

epochs = 1000
tic = time.time()
for epoch in range(1, epochs+1):
    train(epoch)
toc = time.time()
print(f'Total training time: {toc - tic}')

model_path = 'Withoushock2D.pth'
torch.save(model.to('cpu'), model_path)