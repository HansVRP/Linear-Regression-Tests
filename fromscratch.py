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

#%% Compare with targets MSE

def MSE(A,B):
    return torch.mean(torch.pow((A - B), 2))

loss = MSE(preds, targets)
print(loss)

loss.backward()

#%% Compare with targets MSE

print(w)
print(w.grad)

print(b)
print(b.grad)

# after calculating eacht gradient loss, the gradients need to be set to 0
#pytorch accumulates the loss elsewise
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)

#%% We can now set up a first training epoch

W = torch.randn(2, 3, requires_grad=True)
B = torch.randn(2, requires_grad=True)
# we make an estimate for the output
preds = model(inputs, W, B)

# we calculate the loss
loss = MSE(preds, targets)
print (loss)
# backpropagate 

loss.backward()

# update W and b
# we don't want PyTorch to calculate the gradients of the new defined variables w and b
# since  want to update their values.

lr = 1e-5

with torch.no_grad():
    W -= W.grad * lr
    B -= B.grad * lr
    W.grad.zero_()
    B.grad.zero_()
    
# If a gradient element is negative, increasing the element's value slightly will decrease the loss.
    
preds = model(inputs, W, B)

# we calculate the loss
loss = MSE(preds, targets)
print (loss)

#%% Remeat the training multiple epochs
import matplotlib.pyplot as plt

W = torch.randn(2, 3, requires_grad=True)
B = torch.randn(2, requires_grad=True)

lr = 1e-5

d = []

for i in range(100):
    
    preds = model(inputs, W, B)
    loss = MSE(preds, targets)
    d.append(loss.data)
    loss.backward()
    with torch.no_grad():
        W -= W.grad * lr
        B -= B.grad * lr
        W.grad.zero_()
        B.grad.zero_()
        
fig, ax = plt.subplots()
ax.plot(d)
ax.set_xlabel("iteration")
ax.set_ylabel("distance")
plt.show()


