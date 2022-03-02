# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:55:09 2022

@author: Asger
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(10, 6), dpi=80)
import math

def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

class Net(nn.Module):
      def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)
      def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

criterion = nn.BCELoss()
net = Net(1)
optimizer = optim.Adam(net.parameters(), lr=0.001)
train_n = 100
x = torch.linspace(0,3,train_n)
y = torch.sin(x * (2 * math.pi))

x = x.unsqueeze(1)
#testinput = torch.stack((x,y))

for epoch in range(1000):
    y_pred = net(x)
    y_pred = torch.squeeze(y_pred)
    print(y_pred)
    train_loss = criterion(y_pred, y)
    #if epoch % 100 == 0:
    #  train_acc = calculate_accuracy(y, y_pred)
    #  y_test_pred = net(X_test)
    #  y_test_pred = torch.squeeze(y_test_pred)
    #  test_loss = criterion(y_test_pred, y_test)
    #  test_acc = calculate_accuracy(y_test, y_test_pred)
    #  print(
    #    f'''epoch {epoch}
    #    Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
    #    Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
    #    ''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

#plt.plot(testinput,testoutput.detach()[:,:])
plt.plot(x,y_pred.detach())





"""


testinput  = torch.tensor([torch.linspace(0, 1,train_n),

                           
net = Net(testinput.shape[1])

testoutput = m(testinput)
plt.plot(testinput,testoutput.detach()[:,:])

to2 = nn.BatchNorm1d(4)
to22 = to2(testoutput)

to22
plt.plot(testinput,to22.detach()[:,:])
to22
"""