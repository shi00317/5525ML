################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

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

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################
ridge_res_list = []
ridge_mean = []
ridge_val = []
lasso_res_list = []
lasso_mean = []
lasso_val = []
for lambda_val in lambda_vals:

    # instantiate ridge regression object
    ridge = MyRidgeRegression(lambda_val)

    # call to your CV function to compute mse for each fold
    res = my_cross_val(ridge, 'mse', X_train, y_train, k=10)

    # print mse from CV
    print(f"========================={lambda_val}=============================")
    print("ridge")
    print(f"the 10-fold with lambda {lambda_val} loss value: {res}")
    print(f"the mean loss value is: {np.mean(res)}")
    print(f"the std for loss value is: {np.std(res)}")
    print("")
    ridge_res_list.append(res)
    ridge_mean.append(np.mean(res))
    ridge_val.append(np.std(res))


    # instantiate lasso object
    lasso = Lasso(lambda_val)

    # call to your CV function to compute mse for each fold
    res = my_cross_val(lasso, 'mse', X_train, y_train, k=10)

    # print mse from CV
    print("lasso")
    print(f"the 10-fold with lambda {lambda_val} loss value: {res}")
    print(f"the mean loss value is: {np.mean(res)}")
    print(f"the std for loss value is: {np.std(res)}")
    print("")
    lasso_res_list.append(res)
    lasso_mean.append(np.mean(res))
    lasso_val.append(np.std(res))



# instantiate ridge regression and lasso objects for best values of lambda
best_ridge = MyRidgeRegression(0.01)
best_lasso = Lasso(0.01)
# fit models using all training data
best_ridge.fit(X_train,y_train)
best_lasso.fit(X_train,y_train)
# predict on test data
ridgeY_hat = best_ridge.predict(X_test)
lassoY_hat = best_lasso.predict(X_test)
# compute mse on test data
from my_cross_val import mse
res_ridge = mse(y_test,ridgeY_hat)
res_lasso = mse(y_test,lassoY_hat)

# print mse on test data
print(f"mse for ridge with lambda 0.01: {res_ridge}")
print(f"mse for lasso with lambda 0.01: {res_lasso}")


# import pandas as pd
# data = {
#     'Lambda': [0.01, 0.10, 1.00, 10.00, 100.00],
#     'k-fold CV (10)': ridge_res_list
# ,
#     'Mean': ridge_mean,
#     'Std': ridge_val
# }

# df = pd.DataFrame(data)
# for i in range(10):
#     df[f'Fold {i + 1}'] = [x[i] for x in df['k-fold CV (10)']]
# df.drop(columns=['k-fold CV (10)'], inplace=True)
# df = df[['Lambda'] + [f'Fold {i + 1}' for i in range(10)] + ['Mean', 'Std']]
# markdown_table = df.to_markdown(index=False)
# print(markdown_table)
