# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:12:21 2022

@author: 45303
"""

import numpy as np
import math
import torch
from matplotlib import pyplot as plt

# First we do a linear regression
n = 200

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 10, n)
# True function is sin(2*pi*x) with Gaussian noise
w = torch.tensor([2,-2])
b = -2
train_y = w[0]*train_x+w[1]+torch.normal(mean=torch.linspace(0, 0, n), std=torch.ones(n))

plt.scatter(train_x,train_y)

#print(train_y)

n_iters = 5000

w_est = torch.tensor([0.0,0.0],requires_grad=True)

# Find the true coefficients
npy = train_y.numpy()
npx = train_x.numpy()
meanx =np.mean(npx)
meany = np.mean(npy)
top = sum((npx-meanx)*(npy-meany))
bot = sum((npx-meanx)**2)
beta = top/bot
alpha = meany-beta*meanx


#Model prediction
def forward(x):
    return w_est[0]*x+w_est[1]

def loss(y,y_pred):
    return ((y-y_pred)**2).mean()

# tests
forward(train_x)
loss(train_y,torch.ones(n))

learning_rate=0.01
"""
optimizer = torch.optim.SGD([w_est],lr=learning_rate)

for epoch in range(n_iters):
    y_pred = forward(train_x)
    
    l = loss(train_y,y_pred)
    l.backward()
    
    optimizer.step()
    #with torch.no_grad():
    #    w_est -= learning_rate*w_est.grad
    #print(w_est.grad)
    
    w_est.grad.zero_()
    if epoch % 100 == 0: 
        print(w_est)
print(alpha,beta)
"""
w_est = torch.tensor([0.0,0.0],requires_grad=True)
optimizer = torch.optim.Adam([w_est],lr=learning_rate)
for epoch in range(n_iters):
    y_pred = forward(train_x)
    
    l = loss(train_y,y_pred)
    l.backward()
    
    optimizer.step()
    #with torch.no_grad():
    #    w_est -= learning_rate*w_est.grad
    #print(w_est.grad)
    
    optimizer.zero_grad()
    if epoch % 100 == 0: 
        print(w_est)
print(alpha,beta)

# SGD seems to be doing better than Adam
