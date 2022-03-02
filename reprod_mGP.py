# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:40:19 2022

@author: Asger
"""

# First, we sample the function points and plot them

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(10, 6), dpi=80)
import torch

train_n = 50
x = torch.normal(mean=torch.linspace(0, 0, train_n), std=torch.ones(train_n))
w = torch.normal(mean=torch.linspace(0, 0, train_n), std=torch.ones(train_n)*0.01)

def F(x):
    n = x.size(dim=0)
    return torch.ge(x,torch.zeros(n))


train_y = F(x)+w

# This is how data looks
plt.plot(x,train_y,'o')

# Lets try and fit our normal GP
import GP_fitter

def GP_fitter(train_x,train_y):
    #w_est = torch.tensor([2.0,1],requires_grad=True)
    n_iters = 500
    learning_rate = 0.1
    optimizer = torch.optim.Adam([w_est],lr=learning_rate)

    for epoch in range(n_iters):
        K = Kmat(train_x)
        l =  -loglik(train_x,train_y,K)
        l.backward()        
        optimizer.step()
        optimizer.zero_grad()
        #if epoch % 100 == 0: 
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            epoch + 1, n_iters, l.item(),
            w_est[0],
            w_est[1]
        ))
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
    alpha = torch.triangular_solve(l1.solution,torch.transpose(L,0,1),upper=True)
    return -0.5*torch.dot(y,torch.squeeze(alpha.solution))-torch.log(torch.diagonal(L)).sum()


def Kmat2(x1,x2):
    sz1 = x1.size(dim=0)
    sz2 = x2.size(dim=0)
    first = torch.cat((x1 ,x1.repeat(sz2-1))).reshape(sz2,sz1)
    second = torch.cat((x2 ,x2.repeat(sz1-1))).reshape(sz1,sz2)
    t2 = first.transpose(0,1)-second
    t3 = -1.0/(2*transforml(w_est[0])**2)*torch.square(t2)
    t4 = torch.exp(t3)
    return t4

w_est = torch.tensor([1.0,1],requires_grad=True)
w_est = GP_fitter(x,train_y)

test_x = torch.linspace(-4, 4, 200)
kmat2 = Kmat2(test_x,x)

sz = x.size(dim=0)
K = Kmat(x)
L = torch.linalg.cholesky(K+(transforms(w_est[1])**2)*torch.eye(sz))
l1 = torch.triangular_solve(torch.unsqueeze(train_y,1),L,upper=False)
alpha = torch.triangular_solve(l1.solution,torch.transpose(L,0,1))


f_pred = torch.matmul(kmat2,alpha.solution)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), f_pred.numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])


def mGP_fitter(train_x,train_y):
    #w_est = torch.tensor([2.0,1],requires_grad=True)
    n_iters = 500
    learning_rate = 0.1
    optimizer = torch.optim.Adam([w_est],lr=learning_rate)

    for epoch in range(n_iters):
        K = Kmat(train_x)
        l =  -loglik(train_x,train_y,K)
        l.backward()        
        optimizer.step()
        optimizer.zero_grad()
        #if epoch % 100 == 0: 
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            epoch + 1, n_iters, l.item(),
            w_est[0],
            w_est[1]
        ))
    return w_est