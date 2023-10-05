import numpy as np

def my_cross_val(model, loss_func, X, y, k=10):
    # print(X.shape,y.shape)
    # temp = np.array(np.array_split(X,k))
    # print(temp.shape)
    
    # X_splite = np.array(np.array_split(X,k,axis=0))
    # y_splite = np.array(np.array_split(y,k))
    fold_size = X.shape[0] // k
    res = []

    for i in range(k):
        begin = i*fold_size
        end = (i+1)*fold_size
        x_train = np.concatenate([X[:begin],X[end:]])
        x_val = X[begin:end]
        y_train = np.concatenate([y[:begin],y[end:]])
        y_val = y[begin:end]
        # print(x_train.shape,y_train.reshape([-1]).shape)
        model.fit(x_train,y_train)

        y_hat = model.predict(x_val)


        if loss_func == "mse":
            res.append(mse(y_val,y_hat))
        
        elif loss_func == "err_rate":
            res.append(zero_one(y_val,y_hat))

    return res


def mse(y,y_hat):
    '''
    y:      the ground truth
    y_hat:  the prediction 

    return the mean square error loss
    '''
    return np.sum((y-y_hat)**2)/y.shape[0]
def zero_one(y,y_hat):
    '''
    y:      the ground truth
    y_hat:  the prediction 

    return the 0 - 1 loss
    '''
    return np.sum(y!=y_hat)/y.shape[0]