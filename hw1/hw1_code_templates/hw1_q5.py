################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('hw1_q5_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

# plt.scatter(X[:1000, 0], X[:1000, 1])
# plt.scatter(X[1000:, 0], X[1000:, 1])
# plt.show()

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

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
res_list = []
mean = []
val = []
lambda_vals = np.arange(-2,1,0.5)
for lambda_val in lambda_vals:
    # instantiate LDA object
    lda = MyLDA(lambda_val)
    # call to your CV function to compute error rates for each fold
    res = my_cross_val(lda, 'err_rate', X_train, y_train, k=10)
    mean.append(np.mean(res))
    res_list.append(res)
    val.append(np.std(res))
    # print mse from CV
    print(f"========================={lambda_val}=============================")
    print("LDA")
    print(f"the 10-fold with lambda {lambda_val} loss value: {res}")
    print(f"the mean loss value is: {np.mean(res)}")
    print(f"the std for loss value is: {np.std(res)}")
    print("")


best_lda = MyLDA(-1)
from my_cross_val import zero_one
best_lda.fit(X_train,y_train)
best_y_hat = best_lda.predict(X_test)
score = zero_one(y_test,best_y_hat)
print(f"the optimal value for lambda is {-1} and loss value is: {score}")


# import pandas as pd
# data = {
#     'Lambda': lambda_vals,
#     'k-fold CV (10)': res_list
# ,
#     'Mean': mean,
#     'Std': val
# }

# df = pd.DataFrame(data)
# for i in range(10):
#     df[f'Fold {i + 1}'] = [x[i] for x in df['k-fold CV (10)']]
# df.drop(columns=['k-fold CV (10)'], inplace=True)
# df = df[['Lambda'] + [f'Fold {i + 1}' for i in range(10)] + ['Mean', 'Std']]
# markdown_table = df.to_markdown(index=False)
# print(markdown_table)


