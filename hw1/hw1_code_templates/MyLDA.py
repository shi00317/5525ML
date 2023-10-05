import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
        self.w = 0

    def fit(self, X, y):
        # m1 = np.mean(X[y==0],axis=0).reshape([2,1])
        # m2 = np.mean(X[y==1],axis=0).reshape([2,1])
        m1 = np.mean(X[y==0],axis=0)
        m2 = np.mean(X[y==1],axis=0)
        S_B = (m2-m1)@(m2-m1).T
        s1 = np.cov(X[y==0].T)
        s2 = np.cov(X[y==0].T)
        S_w = (X[y==0]-m1.reshape([-1])).T@(X[y==0]-m1.reshape([-1])) \
        +(X[y==1]-m2.reshape([-1])).T@(X[y==1]-m2.reshape([-1]))
        S_w = s1+s2
        self.w = np.dot(np.linalg.inv(S_w),m2-m1)

    def predict(self, X):
        y_hat = X@self.w
        return 1*(y_hat>self.lambda_val)


