from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from math import *
import numpy as np

layerdict = ['module.conv1.0.weight', 'module.conv2.0.weight', 'module.conv3.0.weight','module.conv4.0.weight','module.conv5.0.weight','module.conv6.0.weight','module.conv7.0.weight']
class admm_op():

    # ADMM opertions to do weight quantization


    def __init__ (self, model, b, admm_iter=10, rho=1e-4, mu=10, tau_incr=1.5, tau_decr=1.5):
        super(admm_op, self).__init__()
        self.model = model

        self.W = []
        self.U = []
        self.Z = []
        self.preW = []
        self.storeW = []
        self.r = []
        self.s = []

        for key, value in model.named_parameters():
            if key in layerdict:
                print(key)
                self.W.append(value)
                self.preW.append(value.clone())
                self.storeW.append(value.clone())
                #print(value.data.view(1,-1))
                self.Z.append(value.data.clone())
                self.U.append(value.data.clone().zero_())
                self.r.append(value.data.clone().zero_())
                self.s.append(value.data.clone().zero_())

        self.b = b #bits of each layer
        self.a = torch.zeros((len(self.W)),dtype=torch.float) # a is the scale factor of quatized data
        self.rho = torch.zeros((len(self.W)),dtype=torch.float).fill_(rho) # rho parameter

        for i in range(len(self.W)):
            self.a[i] = 0.5 / (pow(2,b[i])-1)

        self.admm_iter = admm_iter
        self.mu = mu
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr


        # parameters for stop criteria
        # epsilon^abs > 0, absolute tolerance
        #self.eabs = eabs
        # epsilon^rel > 0, relative tolerance
        #self.erel = erel 

    '''
    def check_if_stop(self,):

        epri = sqrt(rho) * erel + erel * torch.norm(W, p=2) 
        edual = sqrt(224*224) * eabs + erel * torch.norm(U, p=2)
        #r^{k+1} = W^{k+1} - Z^{k+1}
        r = W - Z
        #s^{k+1} = rho (Z^{k+1} - Z^k)
    '''

    def update(self, epoch):

        if epoch % self.admm_iter != 0:
            return
        #with torch.no_grad():
        for i in range(len(self.W)):
            Vi = (self.W[i] + self.U[i])#.view(-1,1)
            Qi = (self.Z[i] / self.a[i])#.view(-1,1)

            # update a
            #self.a[i] = torch.matmul(torch.t(Vi),Qi) / (torch.matmul(torch.t(Qi),Qi))
            self.a[i] = torch.sum(torch.mul(Vi,Qi)) / torch.sum(torch.mul(Qi,Qi))

            #update Z
            # quantized values are uniform
            
            Qi = Vi/self.a[i]
            Qi = torch.round((Qi-1)/2)*2+1
            Qi = torch.clamp(Qi,-(pow(2,self.b[i])-1),pow(2,self.b[i])-1)
            

            '''
            Qi = Vi/self.a[i]
            pos = (Qi >= 0).type(torch.float)
            #neg = Qi < 0
            Qi = torch.round(torch.log2(Qi*(2*pos-1)))
            Qi = torch.clamp(Qi, 0, pow(2, self.b[i]-1))
            Qi = torch.pow(2, Qi) * (2*pos-1)
            '''
            
            # quantized values are 2^n
            # tbc

            self.Z[i] = Qi * self.a[i] 


            # update U
            if epoch > 0:
                # For epoch = 0, U is fixed to all zero.
                self.U[i] = self.U[i] + self.W[i].data - self.Z[i]

            # update rho
            self.r[i] = self.W[i].data - self.Z[i]
            self.s[i] = self.rho[i] * (self.W[i].data - self.preW[i].data)
            norm_r = torch.norm(self.r[i].view(1,-1))#np.linalg.norm(r)
            norm_s = torch.norm(self.s[i].view(1,-1))#np.linalg.norm(s)
            #print('w val:',self.W[i].data.view(1,-1))
            #print('prew val:', self.preW[i].data.view(1,-1))
            #print('norm_r:',norm_r)
            #print('norm_s:',norm_s)
            if norm_r > self.mu * norm_s:
                # set upper bound of rho
                if self.rho[i] < 0.4:
                    self.rho[i] = self.rho[i] * self.tau_incr
                    # the scaled dual variable u = (1/ρ)y must also be rescaled
                    #after updating ρ
                    self.U[i] = self.U[i] / self.tau_incr
            elif norm_s > self.mu * norm_r:
                self.rho[i] = self.rho[i] / self.tau_decr
                # the scaled dual variable u = (1/ρ)y must also be rescaled
                #after updating ρ
                self.U[i] = self.U[i] * self.tau_decr
            #else:
                # do not updata rho

            #epri = sqrt(rho[i]) * erel + erel * np.linalg.norm(W[i].data) 
            #edual =  * eabs + erel * np.linalg.norm(U[i])
            self.preW[i] = self.W[i].clone()



    def loss_grad(self):
    # updata W grad, \partial_W L = \partial_W f + \rho (W-Z^K+U^K)

        for i in range(len(self.W)):
            grad = self.W[i].data - (self.Z[i]) + (self.U[i])
            grad = grad * self.rho[i]
            self.W[i].grad.data += grad


    def print_info(self, epoch):
        if epoch % self.admm_iter !=0:
            return
        print('\n' + '-' * 30)
        i = 0
        for key, value in self.model.named_parameters():
            if key in layerdict:
                print(key)
                print('W val:',self.W[i].data.view(1,-1))
                print('Z val:',self.Z[i].data.view(1,-1))
                print('a val:',self.a[i])
                print('2-norm of W and Z:',torch.norm(self.r[i].view(1,-1)))
                print('L2 norm of prime residual:',torch.norm(self.s[i].view(1,-1)))
                print('rho val:',self.rho[i])
                i = i+1
        '''
        for i in range(len(W)):
            print(self.model.named_parameters()[i][0])
            print('W val:',self.W[i].data.view(1,-1))
            print('Z val:',self.Z[i].data.view(1,-1))
            print('a val:',self.a[i])
            print('2-norm of W and Z:',torch.norm(self.r[i].view(1,-1)))
            print('rho val:',self.rho[i])
        '''
        print('\n' + '-' * 30)

    def apply_quantval(self):

        i = 0
        for key, value in self.model.named_parameters():
            if key in layerdict:
                self.model.state_dict()[key].copy_(self.Z[i])
                i = i + 1

    def applyquantW(self):
        for i in range(len(self.W)):
            self.storeW[i] = self.W[i].clone()
            self.W[i].data.copy_(self.Z[i])

    def restoreW(self):
        for i in range(len(self.W)):
            self.W[i].data.copy_(self.storeW[i])
