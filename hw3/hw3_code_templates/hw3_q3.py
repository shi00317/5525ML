################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyCNN import MyCNN

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
# lr = [1e-5,1e-4,1e-3,1e-2,1e-1]
lr = [1e-2]
dev = torch.device("cpu")
myCNN = MyCNN(128,10,10).to(dev)
criterion = torch.nn.CrossEntropyLoss()
for i in lr:
    optimizer = torch.optim.SGD(myCNN.parameters(), lr=i,momentum=0.9)
    myCNN.train()
    myCNN.fit(train_loader,criterion,optimizer)
myCNN.to("cpu")
myCNN.predict(test_loader,criterion)


