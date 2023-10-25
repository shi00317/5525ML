import numpy as np

import torch
import torch.nn as nn
dev = torch.device("cpu")

class MyCNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_epochs):
        super().__init__()

        self.cnn = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=3,stride=1,dilation=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop_out1 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3380,hidden_size)
        self.drop_out2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.soft_max = nn.Softmax(dim=1)
        self.max_epochs = max_epochs
    def forward(self,x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.drop_out1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop_out2(x)
        x = self.fc2(x)
        x = self.soft_max(x)
        return x
    
    def fit(self, train_loader, criterion, optimizer):
        for i in range(self.max_epochs):
            total_loss = 0
            error_rate = 0
            total = 0
            for j, (images,labels) in enumerate(train_loader):
                # print(images.shape,images.reshape(-1,784).shape)
                y_hat = self.forward(images.to(dev))
                loss = criterion(y_hat,labels.to(dev))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss+=loss.item()
                error_rate += (y_hat.argmax(dim=1)==labels.to(dev)).sum().item()
                total+=labels.shape[0]
            print(f"Epoch: {i}, Loss: {total_loss}, Error_Rate: {1-(error_rate/total)}")
            
    def predict(self, test_loader, criterion):
        total_loss = 0
        error_rate = 0
        total = 0
        error = []
        self.eval()
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):
                y_hat = self.forward(images)
                # Compute prediction output and loss
                loss = criterion(y_hat,labels)
                # Measure loss and error rate and record
                total_loss+=loss.item()
                error_rate += (y_hat.argmax(dim=1)==labels).sum().item()
                # if (y_hat.argmax(dim=1)!=labels).sum().item()!=0:
                #     print(labels,y_hat.argmax(dim=1))
                #     temp = [i for i, val in enumerate(y_hat.argmax(dim=1)!=labels) if val]
                #     cur = images[temp]
                #     import matplotlib.pyplot as plt
                #     for i in cur:
                #         plt.imshow(i.reshape(28,28))
                #         plt.show()
                total+=labels.shape[0]
        # Print/return test loss and error rate
        print(f"Loss: {total_loss}, Error_Rate: {1-(error_rate/total)}")
        # return error

