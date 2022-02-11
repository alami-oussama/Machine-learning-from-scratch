import numpy as np


class Perceptron:
    def __init__(self, max_iters=1000):
        self.params = None
        self.max_iters = max_iters

    
    def sign(self, X):
        return np.where(X >= 0, 1, -1)

    def compute_loss(self, X, y, w, b):
        n_samples = X.shape[0]
        linear_output = np.dot(X, w) + b
        y_pred = self.sign(linear_output)
        Ls = 1/n_samples * (np.abs(y - y_pred)).sum()
        return Ls

    def init_params(self, dim):
        w_0 = np.zeros(dim)
        b_0 = 0
        return w_0, b_0

    def fit(self, X, y):
        n = X.shape[0]
        w, b = self.init_params(X.shape[1])
        loss = self.compute_loss(X, y, w, b)
        losses = [loss]
        for _ in range(self.max_iters):
            for i in range(n):
                linear_output = np.dot(X[i], w) + b
                if self.sign(linear_output) * y[i] < 0:
                    w = w + y[i] * X[i]
                    b = b + y[i]
            loss = self.compute_loss(X, y, w, b)
            losses.append(loss)
        self.params = {
            'w': w,
            'b': b
        }
        return losses

    def get_params(self):
        return self.params

    def set_params(self, **params):
        w = params.get('w')
        b = params.get('b')
        self.params = {
            'w': w,
            'b': b
        }
        return self

    def predict(self, X):
        w = self.params.get('w')
        b = self.params.get('b')
        linear_output = np.dot(X, w) + b
        y_pred = self.sign(linear_output)
        return y_pred

    def score(self, X, y):
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        score = 1 - 1/n_samples * (np.abs(y - y_pred)).sum()
        return score

