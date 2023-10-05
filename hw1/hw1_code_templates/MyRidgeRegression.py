import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
        self.w = 0

    def fit(self, X, y):
        '''
            Solve similarly as least squares (minimizer is stationary point)
            ppt: lecture2;page24
        '''
        d = X.shape[1]
        I = np.identity(d)
        self.w = np.linalg.inv(X.T@X+self.lambda_val*I)@X.T@y
    def predict(self, X):
        y_hat = X@self.w
        return y_hat

