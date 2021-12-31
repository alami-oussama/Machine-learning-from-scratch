import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.0001, max_iters=1000):
        self.best_params = None
        self.learning_rate = learning_rate
        self.loss = None
        self.max_iters = max_iters

    
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def activation(self, X):
        return np.where(X >= 0.5, 1, 0)

    def compute_loss(self, X, y, w, b):
        n_samples = X.shape[0]
        linear_output = np.dot(X, w) + b
        A = self.sigmoid(linear_output)
        Ls = -1 / n_samples * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        return Ls

    def init_params(self, n_features):
        w_0 = np.zeros(n_features)
        b_0 = 0
        return w_0, b_0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w, b = self.init_params(n_features)
        loss = self.compute_loss(X, y, w, b)
        losses = [loss]
        for _ in range(self.max_iters):
            linear_output = np.dot(X, w) + b
            A = self.sigmoid(linear_output)
            dw = 1 / n_samples * np.dot(X.T, A - y)
            db = 1 / n_samples * np.sum(A - y)
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            loss = self.compute_loss(X, y, w, b)
            losses.append(loss)
        self.best_params = {
            'w': w,
            'b': b
        }
        self.loss = loss
        return losses

    def predict(self, X):
        w = self.best_params.get('w')
        b = self.best_params.get('b')
        linear_output = np.dot(X, w) + b
        A = self.activation(self.sigmoid(linear_output))
        return A
