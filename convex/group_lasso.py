import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def soft_threshold(x, threshold):
    return np.sign(x) * np.fmax(np.abs(x) - threshold, 0)


class GroupLasso:

    def __init__(self, learning_rate, lambd, max_iters):
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.max_iters = max_iters

        self.w = None

    def _calc_cost(self, X, y, groups):
        cost = np.square(np.dot(X, self.w) - y).sum()
        for group in groups:
            cost += self.lambd * np.square(len(group)) * np.linalg.norm(self.w[group])
        return cost

    def fit_normal(self, X, y, groups):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.flatten()
        groups = [0] + groups
        groups = [np.where(groups == i)[0] for i in np.unique(groups)]

        self.w = np.zeros(X.shape[1], dtype=np.float32)

        costs = [self._calc_cost(X, y, groups)]
        for k in range(self.max_iters):
            w = self.w - self.learning_rate * np.dot(X.T, np.dot(X, self.w) - y)
            for group in groups:
                w[group] = soft_threshold(w[group], self.lambd * self.learning_rate * np.square(len(group)))
            self.w = np.array(w)
            costs.append(self._calc_cost(X, y, groups))

        return np.array(costs)

    def fit_accelerate(self, X, y, groups):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.flatten()
        groups = [0] + groups
        groups = [np.where(groups == i)[0] for i in np.unique(groups)]

        self.w = np.zeros(X.shape[1], dtype=np.float32)
        prev_w = np.zeros(X.shape[1], dtype=np.float32)

        costs = [self._calc_cost(X, y, groups)]
        for k in range(1, self.max_iters + 1):
            v = self.w + (k - 2.0) / (k + 1.0) * (self.w - prev_w)
            prev_w = self.w
            w = v - self.learning_rate * np.dot(X.T, np.dot(X, v) - y)
            for group in groups:
                w[group] = soft_threshold(w[group], self.lambd * self.learning_rate * np.square(len(group)))
            self.w = np.array(w)
            costs.append(self._calc_cost(X, y, groups))

        return np.array(costs)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.w)

    def score(self, X, y):
        pred = self.predict(X)
        return mean_squared_error(pred, y)


if __name__ == "__main__":
    best_cost = 84.6952
    max_iters = 1000

    X = pd.read_csv("data/birthweight/X.csv").values
    y = pd.read_csv("data/birthweight/y.csv").Birthwt.values

    # group lasso
    model = GroupLasso(learning_rate=0.002, lambd=4.0, max_iters=max_iters)
    costs = model.fit_normal(X, y, groups=[1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 8])
    w1 = model.w
    print "Group lasso MSE: {}".format(model.score(X, y))
    plt.plot(costs - best_cost)
    plt.yscale("log")

    # accelerated group lasso
    model = GroupLasso(learning_rate=0.002, lambd=4.0, max_iters=max_iters)
    costs = model.fit_normal(X, y, groups=[1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 8])
    w2 = model.w
    print "Accelerated group lasso MSE: {}".format(model.score(X, y))
    plt.plot(costs - best_cost)
    plt.yscale("log")

    # lasso
    model = GroupLasso(learning_rate=0.002, lambd=0.35, max_iters=max_iters)
    costs = model.fit_normal(X, y, groups=[i+1 for i in range(X.shape[1])])
    w3 = model.w
    print "Lasso MSE: {}".format(model.score(X, y))

    print "solution1:", np.round(w1, 3)
    print "solution2:", np.round(w2, 3)
    print "solution3:", np.round(w3, 3)

    plt.show()
