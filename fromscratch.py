# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:38:37 2019

@author: Hans
"""
#%reset -f

import numpy as np
import torch

x = torch.tensor(3)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

print(x)
print(w)
print(b)

#%% we can use regular algebra

y =  w * x + b

print(y)

#%% we can calculate the gradients of y in function of y in function of w and b

print('test')
# Compute gradients
y.backward(retain_graph=True)

print('dy/dw:', w.grad)
print('dy/db:', b.grad)

#%% we attempt to find the weights describing the relationship between temperature, rain and humidity
# on the yield of apples and oranges


# training in and output
# av temp (F), av rain (mm) av humidity (%)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

#convert the numpy arrays to torch arrays

# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)

#%% initiate the random weights. These we need to be able to update, hence they need to be derivable


# Weights and biases
W = torch.randn(2, 3, requires_grad=True)
B = torch.randn(2, requires_grad=True)
print(W)
print(B)



# define the model of the linear regression

def model(x, w, b):
    return torch.addmm(b,inputs, w.t())

preds = model(inputs, W, B)
print(preds)
