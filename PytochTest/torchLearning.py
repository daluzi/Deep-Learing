# -*- coding: utf-8 -*-
'''
@Time    : 2020/10/9 15:51
@Author  : daluzi
@File    : torchLearning.py
'''

from __future__ import print_function
import torch


'''
    pytorch的基础
'''
# x = torch.empty(5, 3)
# print(x)
# x = torch.rand(5, 3)
# print(x)
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
# x = torch.tensor([5.5, 3])
# print(x)
# x = x.new_ones(5, 3, dtype=torch.double)
# # new_* methods take in sizes
# print(x)
# x = torch.randn_like(x, dtype=torch.float)
#
# # override dtype!
# print(x)
#
# # result has the same size
#
# print(x.size())
# y = torch.rand(5, 3)
# print(x + y)
# print(torch.add(x, y))
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
#
# y.add_(x)
# print(y)
# print(x[:, 1])
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())
#
#
# x = torch.randn(1)
# print(x)
# print(x.item())


'''
    一个神经网络
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss.grad_fn)
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update