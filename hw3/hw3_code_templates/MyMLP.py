import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MyMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)  
        super(MyMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size),
            nn.Softmax(dim=1)
        )
        self.lr = learning_rate
        self.max_epochs =  max_epochs
        self.input_size = input_size

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output
        return self.mlp(x)

    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''
        device = torch.device("cpu")
        loss_list = []
        error_rate_list = []
        # Epoch loop
        for i in range(self.max_epochs):
            error_rate = 0
            total = 0
            total_loss = 0
            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):

                # Forward pass (consider the recommmended functions in homework writeup)
                # print(images.reshape(-1,self.input_size).shape)
                y_hat = self.forward(images.reshape(-1,self.input_size).to(device))
                # print(y_hat[0],labels)
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                loss = criterion(y_hat,labels.to(device))
                loss.backward()
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                optimizer.step()
                optimizer.zero_grad()
                # Track the loss and error rate
                total+=labels.shape[0]
                total_loss += loss.item()                
                error_rate += (y_hat.argmax(dim=1)==labels.to(device)).sum().item()
            # Print/return training loss and error rate in each epoch
            # print(f"Epoch: {i}, Loss: {loss.item()}, Error_Rate: {1-(error_rate/total)}")
            loss_list.append(total_loss)
            error_rate_list.append(1-(error_rate/total))
        return loss_list,error_rate_list
    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''
        total_loss = 0
        error_rate = 0
        total = 0
        self.eval()
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):
                y_hat = self.forward(images.reshape(-1,self.input_size))
                # Compute prediction output and loss
                loss = criterion(y_hat,labels)
                # Measure loss and error rate and record
                total_loss+=loss.item()
                error_rate += (y_hat.argmax(dim=1)==labels).sum().item()
                total+=labels.shape[0]
        # Print/return test loss and error rate
        print(f"Loss: {total_loss}, Error_Rate: {1-(error_rate/total)}")
        return total_loss,1-(error_rate/total)
