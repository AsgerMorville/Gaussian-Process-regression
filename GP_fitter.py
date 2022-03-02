# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:01:21 2022

@author: Asger
"""

import torch

def GP_fitter(train_x,train_y):
    w_est = torch.tensor([1.0,1],requires_grad=True)
    n_iters = 500
    learning_rate = 0.1
    optimizer = torch.optim.Adam([w_est],lr=learning_rate)

    for epoch in range(n_iters):
        #y_pred = forward(train_x)
        
        K = Kmat(train_x)
        l =  -loglik(train_x,train_y,K)
        l.backward()
        
        optimizer.step()
        #with torch.no_grad():
        #    w_est -= learning_rate*w_est.grad
        #print(w_est.grad)
        
        optimizer.zero_grad()
        #if epoch % 100 == 0: 
        #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #    epoch + 1, n_iters, l.item(),
        #    w_est[0],
        #    w_est[1]
        #))
        return w_est
    
def transforml(x):
    return 1/(1+torch.exp(-x))

    
def transforms(par):
    return torch.log(1+torch.exp(par))+0.001


def Kmat(x):
    sz = x.size(dim=0)
    t1 = torch.cat((x, x.repeat(sz-1))).reshape(sz,sz)
    t2 = t1-t1.transpose(0,1)
    lscale = transforml(w_est[0])
    t3 = -1.0/(2*lscale**2)*torch.square(t2)
    t4 = torch.exp(t3)
    return t4

def loglik(X,y,K):
    sz =X.size(dim=0)
    sigma = transforms(w_est[1])
    L = torch.linalg.cholesky(K+(sigma**2)*torch.eye(sz))
    l1 = torch.triangular_solve(torch.unsqueeze(y,1),L,upper=False)
    #print(l1)
    alpha = torch.triangular_solve(l1.solution,torch.transpose(L,0,1),upper=True)
    #print(alpha)
    #print(-0.5*torch.dot(y,torch.squeeze(alpha.solution)))
    return -0.5*torch.dot(y,torch.squeeze(alpha.solution))-torch.log(torch.diagonal(L)).sum()
    #return torch.log(torch.diagonal(L)).sum()