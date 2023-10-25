################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyMLP import MyMLP

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)
#####################
# ADD YOUR CODE BELOW
#####################
# print(torch.has_mps)
# print(aa)
device = torch.device("mps")
lr = [1e-5,1e-4,1e-3,1e-2,1e-1]
device = torch.device("cpu")

print("training with Adam\n")
for i in lr:
    mlp = MyMLP(784,128,10,i,5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=i)
    mlp.train()
    mlp.fit(train_loader,criterion,optimizer)
    mlp.eval()
    mlp.predict(test_loader,criterion)



print("training with SGD\n")
for i in lr:
    mlp = MyMLP(784,128,10,i,5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=i,momentum=0.9)
    mlp.fit(train_loader,criterion,optimizer)
    mlp.eval()
    mlp.predict(test_loader,criterion)

print("training with adagrad\n")
for i in lr:
    mlp = MyMLP(784,128,10,i,5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(mlp.parameters(), lr=i)
    mlp.fit(train_loader,criterion,optimizer)
    mlp.eval()
    mlp.predict(test_loader,criterion)

print("training with RMSprop\n")
for i in lr:
    mlp = MyMLP(784,128,10,i,5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(mlp.parameters(), lr=i,momentum=0.9)
    mlp.fit(train_loader,criterion,optimizer)
    mlp.eval()
    mlp.predict(test_loader,criterion)