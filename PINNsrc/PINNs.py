import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
from numpy import arange, meshgrid
import math
from BC_2D import *
from IC_2D import *
from IC_1D import *
from BC_1D import *
from utility import *


cuda = torch.device('cuda')
cpu = torch.device('cuda')
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class PINNs_WE_Euler_1D(nn.Module):

    def __init__(self,Nl,Nn):
        super(PINNs_WE_Euler_1D, self).__init__()
        self.net = nn.Sequential()                                                 
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))                    
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             

        for num in range(2, Nl):                                                    
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))      
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 3))                

    # Forward Feed
    def forward(self, x):
        u = self.net(x)
       # y = -torch.sin(20*torch.pi*x[:,0:1])*u
        return u

    # Loss function for PDE
    def loss_pde(self, x):
        y = self.net(x)                                                
        rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        
        
        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/0.4
        
        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)
        
        gamma = 1.4                                                    

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        
        dp_g = gradients(p, x)[0]                                      
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            

       # dp_g = gradients(p, x)[0]                                     
       # p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                           
        
        dU2_g = gradients(U2, x)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = gradients(U3, x)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = gradients(F2, x)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = gradients(F3, x)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]

        d = 1/(0.2*(abs(u_x)-(u_x) )+1)#/(abs(p)+0.001)#/(rho + 0.001) #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        #d2 = 0.2*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        #d3 = 0.2*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        
        #d = 0.1*(abs(uR-uL)-(uR-uL))/Dx + 1
        #d = torch.exp(-10*u_x)+1
        #d1 = torch.clamp(d/5,min=1)
        #d = 1.0
        #d = 1/(abs(rho)+0.0001) + 1/(abs(p)+0.0001)
     
        f = ((d*(rho_t + U2_x))**2).mean() + \
            ((d*(U2_t  + F2_x))**2).mean() + \
            ((d*(U3_t  + F3_x))**2).mean() #+\
            #((rho_t).mean())**2 +\
            #((U3_t).mean())**2 
    
        return f

    def loss_ic(self, x, rho, u, p):
        y = self.net(x)                                                      
        rho_nn, p_nn,u_nn = y[:, 0], y[:, 1], y[:, 2]            

        loss_ics = ((u_nn - u) ** 2).mean() + \
               ((rho_nn- rho) ** 2).mean()  + \
               ((p_nn - p) ** 2).mean()

        return loss_ics
    
    # Loss function for conservation
    def loss_con(self, x_en,x_in,crhoL,cuL,cpL,crhoR,cuR,cpR,t):
        y_en = self.net(x_en)                                       
        y_in = self.net(x_in)                                       
        rhoen, pen,uen = y_en[:, 0], y_en[:, 1], y_en[:, 2]         
        rhoin, pin,uin = y_in[:, 0], y_in[:, 1], y_in[:, 2]         

        U3en = 0.5*rhoen*uen**2 + pen/0.4
        U3in = 0.5*rhoin*uin**2 + pin/0.4
        gamma = 0.4
        cU3L = 0.5*crhoL*cuL**2 + cpL/0.4 
        cU3R = 0.5*crhoR*cuR**2 + cpR/0.4 
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() - t*(crhoL*cuL-crhoR*cuR))**2+ \
            ((-U3en+ U3in).mean() + t*(cU3L*cuL - cU3R*cuR) + (cpL*cuL - cpR*cuR)*t )**2 +\
            ((-rhoen*uen + rhoin*uin).mean()+(cpL-cpR)*t +(crhoL*cuL*cuL-crhoR*cuR*cuR)*t)**2
        return loss_en    
    def loss_rh(self, x,x_l):
        y = self.net(x)                                    
        y_l = self.net(x_l)                                    
        rho, p,u = y[:, 0], y[:, 1], y[:, 2]          
        rhol, pl,ul = y_l[:, 0], y_l[:, 1], y_l[:, 2]          

      #  du_g = gradients(u, x)[-1]                                      
      #  u_t, u_x = du_g[:, -1], du_g[:, 1]                            
      #  d = 0/(0.1*(abs(u_x)-u_x)  + 1)
        #eta =  torch.clamp(d-1.1,max=0)*torch.clamp(abs(pr-pl)-0.1,min=0)#*torch.clamp(abs(ur-ul)-0.1,min=0)
        eta =  torch.clamp(abs(p-pl)-0.2,min=0)*torch.clamp(abs(u-ul)-0.2,min=0)
        #eta = 0
        
      #  #loss_rh =  (((rho/rhol - (5*p+pl)/(6*pl+p))*eta)**2).mean()+\
      #  loss_rh = (((rhor/rhol - (5*pr+pl)/(6*pl+pr))*(ur-ul)*eta)**2).mean()+\
      #             ((((ur-ul)**1 -2/rhor*(pr-pl)**2/(0.4*pr+2.4*pl))*eta)**2).mean()
           #        ((((ur-u)**1 -2/rho*(pr-p)**2/(0.4*pr+2.4*p))*eta)**2).max()
            
        loss_rh = ((rho*rhol*(u-ul)**2 -(pl-p)*(rhol - rho))**2*eta).mean()+\
                   (((rho*pl/0.4-rhol*p/0.4) - 0.5*(pl+p)*(rhol-rho))**2*eta).mean()#+\
       #             (((rhor/rhol - (6*pr+pl)/(6*pl+pr))*(ur-ul))**2*eta).mean()+\
       #            ((((ur-ul)**2 -2/rhor*(pr-pl)**2/(0.4*pr+2.4*pl)))**2*eta).mean()
        #loss_rh =  (((pr/pl - (5*rhor-rhol)/(6*rhol-rhor))*(pr-pl)*eta)**2).max()+\
                   #((((u-ul)**1 -2/rho*(p-pl)**2/(0.4*p+2.4*pl))*eta)**2).max()+\
        return loss_rh
    
    def loss_character(self, x_l,x_r):
        y_r = self.net(x_r)                                                      # Initial condition
        y_l = self.net(x_l)                                                      # Initial condition
        rhol, pl,ul = y_l[:, -1], y_l[:, 1], y_l[:, 2]            # rho, u, p - initial condition
        rhor, pr,ur = y_r[:, -1], y_r[:, 1], y_r[:, 2]            # rho, u, p - initial condition

        #du_g = gradients(ul, x_l)[-1]                                      
        #u_t, u_x = du_g[:, :0], du_g[:, 1:]                            
        #d = 0/(0.1*(abs(u_x)-u_x)  + 1)
        #eta =  torch.clamp(d-1.1,max=0)*torch.clamp(abs(pr-pl)-0.01,min=0)*torch.clamp(abs(ur-ul)-0.01,min=0)
        eta =  torch.clamp(abs(pr-pl)-1.01,min=0)*torch.clamp(abs(ur-ul)-0.01,min=0)
       # eta = 0
        # Loss function for the initial condition
        gamma = 0.4
        ss = 0.0e-10
        cL = torch.sqrt(gamma*abs(pl)/(abs(rhol)+ss))
        cR = torch.sqrt(gamma*abs(pr)/(abs(rhor)+ss))
        sR = torch.max(ul+cL,ur+cR)* (rhol-rhor)
        sL = torch.min(ul-cL,ur-cR)*(rhol-rhor)
        
        s = rhol*ul - rhor*ur
       # if (s.max() > 999):
       #     print(rhol-rhor)
       #     print(s)
        #print(torch.clamp(s-sR,min=-1))
       # print(eta)
       # sm = exp(-101*(s-sR))
        loss_s = (((s-sR)*(s-sL)*eta)**1).mean()  #torch.min((((,torch.tensor(1.0))  #+ ((torch.clamp(sL-s,min=0))**2).max()
        return loss_s
 

class PINNs_Euler_1D(nn.Module):

    def __init__(self,Nl,Nn):
        super(PINNs_Euler_1D, self).__init__()
        self.net = nn.Sequential()                                                 
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))                    
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             

        for num in range(2, Nl):                                                    
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))      
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 3))                

    # Forward Feed
    def forward(self, x):
        u = self.net(x)
       # y = -torch.sin(20*torch.pi*x[:,0:1])*u
        return u

    # Loss function for PDE
    def loss_pde(self, x):
        y = self.net(x)                                                
        rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        
        
        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/0.4
        
        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)
        
        gamma = 1.4                                                    

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        
        dp_g = gradients(p, x)[0]                                      
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            

       # dp_g = gradients(p, x)[0]                                     
       # p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                           
        
        dU2_g = gradients(U2, x)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = gradients(U3, x)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = gradients(F2, x)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = gradients(F3, x)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]

        d1 = 0.2*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        d2 = 0.2*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        d3 = 0.2*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        
        #d = 0.1*(abs(uR-uL)-(uR-uL))/Dx + 1
        #d = torch.exp(-10*u_x)+1
        #d1 = torch.clamp(d/5,min=1)
        d = 1.0
        #d = 1/(abs(rho)+0.001) #+ 1/(abs(p)+0.0001)
       # d1 = 1/(abs(rho)+0.0001)
        #d2 = 1/(abs(U2) + 0.0001)
        #d3 = 1/(abs(U3) + 0.0001)
     
        f = ((d*(rho_t + U2_x))**2).mean() + \
            ((d*(U2_t  + F2_x))**2).mean() + \
            ((d*(U3_t  + F3_x))**2).mean() #+\
            #((rho_t).mean())**2 +\
            #((U3_t).mean())**2 
    
        return f

    def loss_ic(self, x, rho, u, p):
        y = self.net(x)                                                      
        rho_nn, p_nn,u_nn = y[:, 0], y[:, 1], y[:, 2]            

        loss_ics = ((u_nn - u) ** 2).mean() + \
               ((rho_nn- rho) ** 2).mean()  + \
               ((p_nn - p) ** 2).mean()

        return loss_ics
    
    # Loss function for conservation
    def loss_con(self, x_en,x_in,t):
        y_en = self.net(x_en)                                       
        y_in = self.net(x_in)                                       
        rhoen, pen,uen = y_en[:, -1], y_en[:, 1], y_en[:, 2]         
        rhoin, pin,uin = y_in[:, -1], y_in[:, 1], y_in[:, 2]         

        U2en = 0.5*rhoen*uen**2 + pen/0.4
        U2in = 0.5*rhoin*uin**2 + pin/0.4
        gamma = 0.4
        cU2L = 0.5*crhoL*cuL**2 + cpL/0.4 
        cU2R = 0.5*crhoR*cuR**2 + cpR/0.4 
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() - t*(crhoL*cuL-crhoR*cuR))**1+ \
            ((-U2en+ U3in).mean() + t*(cU3L*cuL - cU3R*cuR) + (cpL*cuL - cpR*cuR)*t )**2 +\
            ((-rhoen*uen + rhoin*uin).mean()+(cpL-cpR)*t +(crhoL*cuL*cuL-crhoR*cuR*cuR)*t)**1
        return loss_en
    
    def loss_rh(self, x,x_l,x_r):
        y = self.net(x)                                    
        y_r = self.net(x_r)                                    
        y_l = self.net(x_l)                                    
        rho, p,u = y[:, -1], y[:, 1], y[:, 2]          
        rhol, pl,ul = y_l[:, -1], y_l[:, 1], y_l[:, 2]          
        rhor, pr,ur = y_r[:, -1], y_r[:, 1], y_r[:, 2]          

        du_g = gradients(u, x)[-1]                                      
        u_t, u_x = du_g[:, -1], du_g[:, 1]                            
        d = 0/(0.1*(abs(u_x)-u_x)  + 1)
        eta =  torch.clamp(d-1.1,max=0)*torch.clamp(abs(pr-pl)-0.1,min=0)*torch.clamp(abs(ur-ul)-0.1,min=0)
       # eta =  torch.clamp(abs(pr-pl)-1.1,min=0)*torch.clamp(abs(ur-ul)-0.1,min=0)
        #eta = 0
        
        #loss_rh =  (((rho/rhol - (5*p+pl)/(6*pl+p))*eta)**2).mean()+\
        loss_rh = (((rhor/rhol - (5*pr+pl)/(6*pl+pr))*(ur-ul)*eta)**2).mean()+\
                   ((((ur-ul)**1 -2/rhor*(pr-pl)**2/(0.4*pr+2.4*pl))*eta)**2).mean()
           #        ((((ur-u)**1 -2/rho*(pr-p)**2/(0.4*pr+2.4*p))*eta)**2).max()
            
        #loss_rh =  (((pr/pl - (5*rhor-rhol)/(6*rhol-rhor))*(pr-pl)*eta)**2).max()+\
                   #((((u-ul)**1 -2/rho*(p-pl)**2/(0.4*p+2.4*pl))*eta)**2).max()+\
        return loss_rh
    
    def loss_character(self, x_l,x_r):
        y_r = self.net(x_r)                                                      # Initial condition
        y_l = self.net(x_l)                                                      # Initial condition
        rhol, pl,ul = y_l[:, -1], y_l[:, 1], y_l[:, 2]            # rho, u, p - initial condition
        rhor, pr,ur = y_r[:, -1], y_r[:, 1], y_r[:, 2]            # rho, u, p - initial condition

        #du_g = gradients(ul, x_l)[-1]                                      
        #u_t, u_x = du_g[:, :0], du_g[:, 1:]                            
        #d = 0/(0.1*(abs(u_x)-u_x)  + 1)
        #eta =  torch.clamp(d-1.1,max=0)*torch.clamp(abs(pr-pl)-0.01,min=0)*torch.clamp(abs(ur-ul)-0.01,min=0)
        eta =  torch.clamp(abs(pr-pl)-1.01,min=0)*torch.clamp(abs(ur-ul)-0.01,min=0)
       # eta = 0
        # Loss function for the initial condition
        gamma = 0.4
        ss = 0.0e-10
        cL = torch.sqrt(gamma*abs(pl)/(abs(rhol)+ss))
        cR = torch.sqrt(gamma*abs(pr)/(abs(rhor)+ss))
        sR = torch.max(ul+cL,ur+cR)* (rhol-rhor)
        sL = torch.min(ul-cL,ur-cR)*(rhol-rhor)
        
        s = rhol*ul - rhor*ur
       # if (s.max() > 999):
       #     print(rhol-rhor)
       #     print(s)
        #print(torch.clamp(s-sR,min=-1))
       # print(eta)
       # sm = exp(-101*(s-sR))
        loss_s = (((s-sR)*(s-sL)*eta)**1).mean()  #torch.min((((,torch.tensor(1.0))  #+ ((torch.clamp(sL-s,min=0))**2).max()
        return loss_s
 

class PINNs_WE_Euler_2D(nn.Module):

    def __init__(self,Nl,Nn):
        super(PINNs_WE_Euler_2D, self).__init__()
        self.net = nn.Sequential()                                                 
        self.net.add_module('Linear_layer_1', nn.Linear(3, Nn))                    
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             

        for num in range(2, Nl):                                                    
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))      
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 4))                

    # Forward Feed
    def forward(self, x):
        u = self.net(x)
       # y = -torch.sin(20*torch.pi*x[:,0:1])*u
        return u

    def bd_B(self,x,sin,cos):
        yb = self.net(x)
        rhob,pb,ub,vb = yb[:, 0:1], yb[:, 1:2], yb[:, 2:3],yb[:,3:]
        drhob_g = gradients(rhob, x)[0]                                           
        rhob_x, rhob_y = drhob_g[:, 1:2], drhob_g[:, 2:3] 
        dub_g = gradients(ub, x)[0]
        ub_x, ub_y = dub_g[:, 1:2], dub_g[:, 2:3]
        dvb_g = gradients(vb, x)[0]
        vb_x, vb_y = dvb_g[:, 1:2], dvb_g[:, 2:3]
        dpb_g = gradients(pb, x)[0]
        pb_x, pb_y = dpb_g[:, 1:2], dpb_g[:, 2:3]
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
        
        drho_g = gradients(rho, x)[0]                                  
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    
        du_g = gradients(u, x)[0]                                      
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                           
        dv_g = gradients(v, x)[0]                                      
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                           
        dp_g = gradients(p, x)[0]                                      
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                           
        
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
        
        drho_g = gradients(rho, x)[0]                                
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                  
        du_g = gradients(u, x)[0]                                    
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                         
        dv_g = gradients(v, x)[0]                                    
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                         
        dp_g = gradients(p, x)[0]                                    
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                         
        
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
#        df1_g = gradients(f1, x)[0]     G                             # Gradient [rho_t, rho_x]
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
    def loss_con(self, x_en,x_in):
        y_en = self.net(x_en)                                       
        y_in = self.net(x_in)                                       
        rhoen, pen,uen = y_en[:, 0], y_en[:, 1], y_en[:, 2]         
        rhoin, pin,uin = y_in[:, 0], y_in[:, 1], y_in[:, 2]         

        U3en = 0.5*rhoen*uen**2 + pen/0.4
        U3in = 0.5*rhoin*uin**2 + pin/0.4
        gamma = 1.4
        #UcU3L = 0.5*crhoL*cuL**2 + cpL/0.4 
        #cU3R = 0.5*crhoR*cuR**2 + cpR/0.4 
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() )**2+ \
            ((-U3en+ U3in).mean()   )**2 +\
            ((-rhoen*uen + rhoin*uin).mean())**2
        return loss_en

    def loss_pde(self, x):
        y = self.net(x) 
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        gamma = 1.4                                                   
        drho_g = gradients(rho, x)[0]                                 
        rho_t, rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2],drho_g[:,2:]
        du_g = gradients(u, x)[0]                                      
        u_t, u_x, u_y = du_g[:, :1], du_g[:, 1:2], du_g[:,2:]          
        dv_g = gradients(v, x)[0]                                      
        v_t, v_x, v_y = dv_g[:, :1], dv_g[:, 1:2], dv_g[:,2:]          
        dp_g = gradients(p, x)[0]                                      
        p_t, p_x, p_y = dp_g[:, :1], dp_g[:, 1:2], dp_g[:,2:]          
        deltau = u_x + v_y
        lam = 0.2*(abs(deltau) - deltau) + 1
        f = (((rho_t+rho*deltau+u*rho_x + v*rho_y)/lam)**2).mean() +\
            (((rho*u_t+rho*u*u_x+rho*v*u_y+p_x)/lam)**2).mean() +\
            (((rho*v_t+rho*u*v_x+rho*v*v_y+p_y)/lam)**2).mean() +\
            (((p_t+u*p_x+v*p_y+1.4*p*deltau)/lam)**2).mean()
            
        return f
      
      
    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics

    def loss_bc(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics
    def loss_bc_cons(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()
        return loss_ics

    def loss_bc1(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        loss_ics = ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() 

        return loss_ics
    def loss_period(self, x_l, x_r):
        U_L = self.net(x_l)                                                      # Initial condition
        U_R = self.net(x_r)                                                      # Initial condition
        rho_l, p_l,u_l,v_l = U_L[:, 0], U_L[:, 1], U_L[:, 2],U_L[:,3]            # rho, u, p - initial condition
        rho_r, p_r,u_r,v_r = U_R[:, 0], U_R[:, 1], U_R[:, 2],U_R[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((rho_l - rho_r) ** 2).mean() + \
               ((u_l- u_r) ** 2).mean()  + \
               ((p_l - p_r) ** 2).mean() 
        return loss_ics
    def loss_bc1(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() 

        return loss_ics
    
    def loss_rh(self, x,x_l):
        y = self.net(x)                                    
        y_l = self.net(x_l)                                    
        rho, p,u,v = y[:, 0], y[:, 1], y[:, 2],y[:,3]          
        rhol, pl,ul,vl = y_l[:, 0], y_l[:, 1], y_l[:, 2],y_l[:,3 ]          

      #  du_g = gradients(u, x)[-1]                                      
      #  u_t, u_x = du_g[:, -1], du_g[:, 1]                            
      #  d = 0/(0.1*(abs(u_x)-u_x)  + 1)
        #eta =  torch.clamp(d-1.1,max=0)*torch.clamp(abs(pr-pl)-0.1,min=0)#*torch.clamp(abs(ur-ul)-0.1,min=0)
        eta =  torch.clamp(abs(p-pl)-0.2,min=0)*torch.clamp((u-ul)**2+(v-vl)**2-0.04,min=0)
            
        loss_rh = ((rho*rhol*((u-ul)**2+ (v-vl)**2)-(pl-p)*(rhol - rho))**2*eta).mean()+\
                   (((rho*pl/0.4-rhol*p/0.4) - 0.5*(pl+p)*(rhol-rho))**2*eta).mean()#+\
        return loss_rh

    def loss_con(self, x_en,x_in,crhoL,cuL,cpL,crhoR,cuR,cpR,t):
        y_en = self.net(x_en)                                       
        y_in = self.net(x_in)                                       
        rhoen, pen,uen = y_en[:, 0], y_en[:, 1], y_en[:, 2]         
        rhoin, pin,uin = y_in[:, 0], y_in[:, 1], y_in[:, 2]         

        U3en = 0.5*rhoen*uen**2 + pen/0.4
        U3in = 0.5*rhoin*uin**2 + pin/0.4
        gamma = 0.4
        cU3L = 0.5*crhoL*cuL**2 + cpL/0.4 
        cU3R = 0.5*crhoR*cuR**2 + cpR/0.4 
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() - t*(crhoL*cuL-crhoR*cuR))**2+ \
            ((-U3en+ U3in).mean() + t*(cU3L*cuL - cU3R*cuR) + (cpL*cuL - cpR*cuR)*t )**2 +\
            ((-rhoen*uen + rhoin*uin).mean()+(cpL-cpR)*t +(crhoL*cuL*cuL-crhoR*cuR*cuR)*t)**2
        return loss_en    


class PINNs_scalar_2D(nn.Module):

    def __init__(self,Nl,Nn):
        super(PINNs_scalar_2D, self).__init__()
        self.net = nn.Sequential()                                                 
        self.net.add_module('Linear_layer_1', nn.Linear(3, Nn))                    
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             

        for num in range(2, Nl):                                                    
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))      
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 1))                

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x,ax,ay):
        
   #     u0 = u_ic_train
   #     if (iter > 0):
   #         y0 = model0(x)                                                # Neural network
   #         u0 = y0[:,0]
        y = self.net(x)
        u = y[:,0]
        du_g = gradients(u, x)[0]                                
        u_t,u_x,u_y = du_g[:, 0],du_g[:,1],du_g[:,2]
        #d = 1+abs(u_x)+abs(u_x)
      #   = gradients(u_x, x)[0]                                
      #  duy_g = gradients(u_y, x)[0]                                
      #  u_xx = dux_g[:, 1]
      #  u_yy = duy_g[:, 2]
      #  print(u.shape)
      #  print(u0.shape)
      #  print(ax.shape)
      #  print(u_x.shape)
        
        f =  ((u_t + ax*u_x+ay*u_y)**2).mean()
        
        return f
    
    def res_pde(self,x):
        y = self.net(x)
        Res = np.zeros((x.shape[0]))                                  
        u = y[:, 0:1]
        du_g = gradients(u, x)[0]                                  
        u_t,u_x = du_g[:, :1],du_g[:,1:]
        Res = (u_t - u_x)**2 
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
        u_ic_in = y_ic[:,0]
        loss_ics = ((u_ic_in - u_ic) ** 2).mean()
        if (len(x_ic) == 0):
            loss_ics = 0
        return loss_ics
    def loss_mm(self, x_ic, u_ic):
        y_ic = self.net(x_ic)                                                      
        u_ic_in = y_ic[:,0]
        loss_ics = ((u_ic_in - u_ic[:,0]) ** 2).mean()
        if (len(x_ic) == 0):
            loss_ics = 0
        return loss_ics
    def loss_bd(self, x,u_in):
        y = self.net(x)                                                      
        loss_bds = ((y[:,0] - u_in[:]) ** 2).mean()
        if (len(x) == 0):
            loss_bds = 0
        return loss_bds
    def loss_bd1(self, xL,xR):
        yL = self.net(xL)                                                      
        yR = self.net(xR)                                                      
        loss_bds = ((yL[:,0] - yR[:,0])**2).mean()
        if (len(xL) == 0):
            loss_bds = 0
        return loss_bds

class PINNs_scalar_1D(nn.Module):

    def __init__(self,Nl,Nn):
        super(PINNs_scalar_1D, self).__init__()
        self.net = nn.Sequential()                                                 
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))                    
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             

        for num in range(2, Nl):                                                    
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))      
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 1))                

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)                                                
        du_g = gradients(u, x)[0]                                 
        u_t,u_x = du_g[:, :1],du_g[:,1:]
    #d = 0.01*abs(u_x)+1
        f = (((u_t +u_x))**2).mean() 
        if (len(x) == 0) :
            f = 0 
        #f = ((u_t +u_x)**2).mean() 
        return f
    
    def loss_con(self,x,u_con):
        u = self.net(x)                                                
    #d = 0.01*abs(u_x)+1
        loss = ((u - u_con).mean())**2
        #f = ((u_t +u_x)**2).mean() 
        return loss
  
    
    def res_pde(self,x):
        y = self.net(x)
        Res = np.zeros((x.shape[0]))                                  
        u = y[:, 0:1]
        du_g = gradients(u, x)[0]                                  
        u_t,u_x = du_g[:, :1],du_g[:,1:]
        Res = (u_t - u_x)**2 
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
        u_ic_in = y_ic[:,0]
        loss_ics = ((u_ic_in - u_ic) ** 2).mean()
        if (len(x_ic) == 0):
            loss_ics = 0
        return loss_ics
    def loss_mm(self, x_ic, u_ic):
        y_ic = self.net(x_ic)                                                      
        u_ic_in = y_ic[:,0]
        loss_ics = ((u_ic_in - u_ic[:,0]) ** 2).mean()
        if (len(x_ic) == 0):
            loss_ics = 0
        return loss_ics
    def loss_bd(self, x,u_in):
        y = self.net(x)                                                      
        loss_bds = ((y[:,0] - u_in[:]) ** 2).mean()
        if (len(x) == 0):
            loss_bds = 0
        return loss_bds
    def loss_outflow1(self, x,xr):
        y = self.net(x)                                                      
        yr = self.net(xr)                                                      
        loss_bds = ((yr - y) ** 2).mean()
        return loss_bds

    def loss_outflow2(self, x,xr):
        y = self.net(x)                                                      
        yr = self.net(xr)                                                      
        dy_g = gradients(y, x)[0]                                  
        dyp_g = gradients(yr, xr)[0]                                  
        loss_bds = ((dy_g- dyp_g) ** 2).mean()
        return loss_bds


    def loss_bd1(self, xL,xR):
        yL = self.net(xL)                                                      
        yR = self.net(xR)                                                      
        loss_bds = ((yL[:,0] - yR[:,0])**2).mean()
        if (len(xL) == 0):
            loss_bds = 0
        return loss_bds