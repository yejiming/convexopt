import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


class NewtonLogisticRegression:

    def __init__(self, alpha, beta, threshold):
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        self.w = None

    def _calc_cost(self, X, y, w):
        XW = np.dot(X, w)
        cost = -np.sum(y * XW) + np.log(1 + np.exp(XW)).sum()
        return cost

    def fit(self, X, y):

        def _linesearch(w_old, gradient, hessian):
            learning_rate = 1.0

            v = -np.dot(np.linalg.inv(hessian), gradient)

            while self._calc_cost(X, y, w_old + learning_rate * v) > self._calc_cost(X, y, w_old) + \
                self.alpha * learning_rate * np.dot(gradient, v):
                learning_rate *= self.beta

            return learning_rate

        X = np.hstack((np.ones((X.shape[0], 1)), X)).astype(np.float64)
        y = y.flatten()

        self.w = np.zeros(X.shape[1], dtype=np.float64)
        costs = []
        while len(costs) < 2 or np.abs(costs[-1] - costs[-2]) > self.threshold:
            p = 1 - 1 / (1 + np.exp(np.dot(X, self.w)))
            gradient = np.dot(X.T, p - y)
            hessian = np.dot(np.dot(X.T, np.diag(p * (1 - p))), X)
            t = _linesearch(self.w, gradient, hessian)
            self.w -= t * np.dot(np.linalg.inv(hessian), gradient)
            costs.append(self._calc_cost(X, y, self.w))

        return np.array(costs)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        p = 1 - 1 / (1 + np.exp(np.dot(X, self.w)))
        return p

    def score(self, X, y):
        pred = self.predict(X)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        return np.sum(pred == y) / float(y.size)


if __name__ == "__main__":
    best_cost = 186.637

    train = sio.loadmat("data/hwk4_moviesTrain.mat")
    X_train = train["trainRatings"]
    y_train = train["trainLabels"].flatten()

    test = sio.loadmat("data/hwk4_moviesTest.mat")
    X_test = test["testRatings"]
    y_test = test["testLabels"].flatten()

    model = NewtonLogisticRegression(alpha=0.01, beta=0.9, threshold=1e-6)
    costs = model.fit(X_train, y_train)
    print "Train accuracy: {}".format(model.score(X_train, y_train))
    print "Test accuracy: {}".format(model.score(X_test, y_test))

    plt.plot(costs - best_cost)
    plt.yscale("log")

    plt.show()
