import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000, random_state=None):
        self.params = None
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.random_state = random_state


    def compute_loss(self, X, y, w, b):
        n_samples = X.shape[0]
        y_predicted = np.dot(X, w) + b
        Ls = 1 / n_samples * np.sum(np.square(y_predicted - y))
        return Ls

    def init_params(self, n_features):
        np.random.seed(self.random_state)
        w_0 = np.random.rand(n_features)
        b_0 = np.random.rand()
        return w_0, b_0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w, b = self.init_params(n_features)
        loss = self.compute_loss(X, y, w, b)
        losses = [loss]
        for _ in range(self.max_iters):
            y_predicted = np.dot(X, w) + b
            dw = 1 / n_samples * np.dot(X.T, y_predicted - y)
            db = 1 / n_samples * np.sum(y_predicted - y)
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
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
        y_predicted = np.dot(X, w) + b
        return y_predicted

    def score(self, X, y):
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        score = 1/n_samples * ((y - y_pred)** 2).sum()
        return score
