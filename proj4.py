# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:47:48 2022

@author: 45303
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:38:26 2022

@author: 45303
"""

import numpy as np
import math
import torch
from matplotlib import pyplot as plt

# Implementing manual gaussian process regression
# We wish to find the hyper parameters of the Gaussian process

train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
train_y =  train_y-torch.mean(train_y)

def transforml(par):
    return torch.log(1+torch.exp(par))
def transforms(par):
    return torch.log(1+torch.exp(par))+0.001

# Our loss function will be the likelihood.


w_est = torch.tensor([1,1.0],requires_grad=True)

def Kmat(x):
    sz = x.size(dim=0)
    t1 = torch.cat((x, x.repeat(sz-1))).reshape(sz,sz)
    t2 = t1-t1.transpose(0,1)
    lscale = transforml(w_est[0])
    t3 = -lscale*torch.square(t2)
    t4 = torch.exp(t3)
    return t4

def loglik(X,y,K):
    sz =X.size(dim=0)
    sigma = transforms(w_est[1])
    L = torch.linalg.cholesky(K+sigma*torch.eye(sz))
    l1 = torch.triangular_solve(torch.unsqueeze(y,1),L)
    alpha = torch.triangular_solve(l1.solution,torch.transpose(L,0,1))
    return -0.5*torch.dot(y,torch.squeeze(alpha.solution))-torch.log(torch.diagonal(L)).sum()
    
# We try by first implementing 



#loglik(train_x,train_y)

n_iters = 1000
learning_rate = 0.1
optimizer = torch.optim.Adam([w_est],lr=learning_rate)
#optimizer = torch.optim.SGD([w_est],lr=learning_rate)
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
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        epoch + 1, n_iters, l.item(),
        w_est[0],
        w_est[1]
    ))


def Kmat2(x1,x2):
    sz1 = x1.size(dim=0)
    sz2 = x2.size(dim=0)
    first = torch.cat((x1 ,x1.repeat(sz2-1))).reshape(sz2,sz1)
    #print(first.transpose(0,1))
    second = torch.cat((x2 ,x2.repeat(sz1-1))).reshape(sz1,sz2)
    #print(second)
    t2 = first.transpose(0,1)-second
    #print(t2)
    t3 = -1.0/(2*transforml(w_est[0])**2)*torch.square(t2)
    t4 = torch.exp(t3)
    return t4

w_est = torch.tensor([1,1])
test_x = torch.linspace(0, 1, 51)
kmat2 = Kmat2(test_x,train_x)

sz = train_x.size(dim=0)
K = Kmat(train_x)
L = torch.linalg.cholesky(K+(transforms(w_est[1])**2)*torch.eye(sz))
l1 = torch.triangular_solve(torch.unsqueeze(train_y,1),L)
alpha = torch.triangular_solve(l1.solution,torch.transpose(L,0,1))


Wf_pred = torch.matmul(kmat2,alpha.solution)
#f_pred1 = torch.matmul(torch.inverse(K+(transforms(w_est[1])**2)*torch.eye(sz)),train_y)
#f_pred = torch.matmul(kmat2,f_pred1)
#w_est = torch.tensor()



with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), f_pred.numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    
    

####################################################################################
    ####################################################################################
    ####################################################################################
    ####################################################################################
    ####################################################################################
# tests  https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Hyperparameters.html
#Calculate loglik with two differ ent hyper pars
def Kmat(x):
    sz = x.size(dim=0)
    t1 = torch.cat((x, x.repeat(sz-1))).reshape(sz,sz)
    t2 = t1-t1.transpose(0,1)
    t3 = -1.0/(2*w_est[0]**2)*torch.square(t2)
    t4 = torch.exp(t3)
    return t4

def loglik(X,y,K):
    sz =X.size(dim=0)
    L = torch.linalg.cholesky(K+(w_est[1]**2)*torch.eye(sz))
    l1 = torch.triangular_solve(torch.unsqueeze(y,1),L)
    alpha = torch.triangular_solve(l1.solution,torch.transpose(L,0,1))
    return -0.5*torch.dot(y,torch.squeeze(alpha.solution))-torch.log(torch.diagonal(L)).sum()
with torch.no_grad():
    w_est = torch.tensor([1.408,0.356])
    K = Kmat(train_x)
    loglik1 = -loglik(train_x,train_y,K)
    
    w_est = torch.tensor([0.3,0.04])
    K = Kmat(train_x)
    loglik2 = -loglik(train_x,train_y,K)
    """
#b = 2

def test_func(a):
    c = e[0]
    return a+c
e = torch.tensor([19,4])
test_func(2)




# functionality
test = torch.tensor([[4.0,1],[1,4]])
test2 = test+2
torch.exp(test)
test3 = torch.linalg.cholesky(test)

torch.matmul(test3,torch.transpose(test3,0,1))

test4 = torch.tensor([1, 2, 3])
test5 =torch.tensor([test4,test4])

test6 = torch.cat((test4, test4.repeat(2))).reshape(3,3)
test7 = test6-test6.transpose(0,1)
test6 = torch.cat((test4, test4.repeat(2))).reshape(3,3)
test8 = torch.square(test6)


train_x
l = 3
t1 = torch.cat((train_x, train_x.repeat(n-1))).reshape(n,n)
t2 = t1-t1.transpose(0,1)
t3 = -1/(2*l^2)*torch.square(t2)
t4 = torch.exp(t3)


test11 = torch.tensor([1,2])
t122  = torch.tensor([1,2,3])

Kmat2(test11,t122)

tt1 = torch.cat((test11 ,test11 .repeat(3-1))).reshape(3,2)
#first = torch.cat((test11 ,test11.repeat(sz2-1))).reshape(sz2,sz1)
#second = torch.cat((t2 ,t2.repeat(sz1-1))).reshape(sz1,sz2)
"""