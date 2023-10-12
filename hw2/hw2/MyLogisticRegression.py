import numpy as np
class MyLogisticRegression:

    def __init__(self, d, max_iters, eta_val):
        self.d  = d
        self.max_iters = max_iters
        self.eta_val = eta_val
        self.w = np.random.uniform(-0.01,0.01,d)
        self.loss = []

    def sigmoid(self,data):
        return np.exp(data)/(1+ np.exp(data))

    def fit(self, X, y):
        magnitude = 0
        for i in range(self.max_iters):
            # for j in range(X.shape[0]):
            j = np.random.randint(0, len(X))
            # print(X[j].shape, self.w.shape)
            y_hat =self.sigmoid(np.dot(X[j], self.w))
            grad = X[j]*(y[j]-y_hat)
            self.w -=self.eta_val*(-grad)
            magnitude+=grad
            average_loss = np.linalg.norm(magnitude)/(i+1)
            self.loss.append(average_loss)
            if average_loss<=1e-6:
                break
    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.w))>0.5).astype(int).reshape(-1,)
    
    def _getLoss_(self):
        return np.array(self.loss)
    

