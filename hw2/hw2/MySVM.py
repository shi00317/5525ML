import numpy as np
from tqdm import tqdm
class MySVM:

    def __init__(self, d, max_iters, eta_val, c):
        self.d  = d
        self.max_iters = max_iters
        self.eta_val = eta_val
        self.w = np.random.uniform(-0.01,0.01,d)
        self.c = c
        self.loss = []
    def fit(self, X, y):
        magnitude = 0
        for i in range(self.max_iters):
            j = np.random.randint(0, len(X))
            
            hinge_loss = 1-y[j]*np.dot(X[j], self.w)
            self.loss.append(hinge_loss)
            if hinge_loss<=0:
                grad = self.w
            else:
                grad = self.w-self.c*(X[j]*y[j])
            self.w -=self.eta_val*(grad)
            magnitude+=grad
            average_loss = np.linalg.norm(magnitude)/(i+1)
            self.loss.append(average_loss)
            if average_loss<=1e-3:
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.w))


    def _getLoss_(self):
        return np.array(self.loss)
