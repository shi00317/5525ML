################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MySVM import MySVM

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

# change labels from 0 and 1 to -1 and 1 for SVM
y[y == 0] = -1

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))
num_data, num_features = X.shape

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################

# Import your CV package here (either your my_cross_val or sci-kit learn )
from my_cross_val import my_cross_val
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [1, 10, 100]

# SVM
for eta_val in eta_vals:
    for c_val in C_vals:

        # instantiate svm object
        cur_svm = MySVM(3, 100000, eta_val, c_val)
        # call to your CV function to compute error rates for each fold
        res =my_cross_val(cur_svm, 'mse', X_train,y_train , k=10)
        # print error rates from CV
        print(f"========================={eta_val} with {c_val}=============================")
        print("SVM")
        print(f"the 10-fold with learning rate {eta_val} loss value: {res}")
        print(f"the mean loss value is: {np.mean(res)}")
        print(f"the std for loss value is: {np.std(res)}")
        print("")
        # print error rates from CV

# instantiate svm object for best value of eta and C
best_svm = MySVM(3,50000,0.00001,100)
# fit model using all training data
best_svm.fit(X_train,y_train)
# predict on test data
y_hat = best_svm.predict(X_test)
# compute error rate on test data
mse = mean_squared_error(y_hat, y_test)
zero_one = 1-accuracy_score(y_hat, y_test)
# print error rate on test data
print("Using the best learning rate 0.00001 and C 100")
print(f"Mean Squared Error: {mse}")
print(f"Accuracy Score: {zero_one}")
plt.plot(best_svm._getLoss_())
plt.ylabel('Loss')
plt.show()
