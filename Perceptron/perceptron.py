import numpy as np


class Perceptron:
    def __init__(self, eps=0.05, max_iters=1000):
        self.eps = eps
        self.best_params = None
        self.loss = None
        self.n_iters = None
        self.max_iters = max_iters

    
    def activation(self, X):
        return np.where(X >= 0, 1, -1)

    def compute_loss(self, X, y, w, b):
        n = X.shape[0]
        S = 0
        for i in range(n):
            linear_output = np.dot(X[i], w) + b
            if self.activation(linear_output) != y[i]:
                S += 1
        Ls = S / n
        return Ls

    def init_params(self, dim):
        w_0 = np.zeros(dim)
        b_0 = 0
        return w_0, b_0

    def fit(self, X, y):
        n = X.shape[0]
        w, b = self.init_params(X.shape[1])
        loss = self.compute_loss(X, y, w, b)
        n_iters = 0
        losses = [loss]
        while loss >= self.eps:
            for i in range(n):
                linear_output = np.dot(X[i], w) + b
                if self.activation(linear_output) * y[i] < 0:
                    w = w + y[i] * X[i]
                    b = b + y[i]
            loss = self.compute_loss(X, y, w, b)
            losses.append(loss)
            n_iters = n_iters + 1
            if n_iters >= self.max_iters:
                break
        self.best_params = {
            'w': w,
            'b': b
        }
        self.loss = loss
        self.n_iters = n_iters
        return losses

    def predict(self, X):
        w = self.best_params.get('w')
        b = self.best_params.get('b')
        linear_output = np.dot(X, w) + b
        y_predicted = self.activation(linear_output)
        return y_predicted
