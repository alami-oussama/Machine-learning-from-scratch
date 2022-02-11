import numpy as np


class PolynomailRegression:
    def __init__(self, degree, learning_rate=0.01, max_iters=1000):
        self.degree = degree
        self.learning_rate = learning_rate  
        self.max_iters = max_iters


    def transform(self, X):
        n_samples = X.shape[0]
        X_transform = np.ones((n_samples, 1))
        j = 0
        for j in range(self.degree + 1) :
            if j != 0 : 
                x_pow = np.power(X, j)
                X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)
        return X_transform

    def normalize(self, X) :
        X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis = 0)) / np.std(X[:, 1:], axis = 0)
        return X

    def init_params(self):
        w = np.zeros(self.degree + 1)
        return w
        
    def fit(self, X, y) :
        n_samples = X.shape[0]
        self.W = self.init_params()
        losses = []
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)     
        for _ in range(self.max_iters):
            y_pred = self.predict(X)
            error = y_pred - y
            loss = np.square(-np.sum(error))
            losses.append(loss)
            dw = (1 / n_samples) * np.dot(X_normalize.T, error)
            self.W = self.W - self.learning_rate * dw
        return losses

    def predict(self, X) :
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)
        return np.dot(X_normalize, self.W)
