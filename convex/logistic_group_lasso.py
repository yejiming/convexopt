import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


def soft_threshold(x, threshold):
    return np.sign(x) * np.fmax(np.abs(x) - threshold, 0)


class LogisticGroupLasso:

    def __init__(self, learning_rate, lambd, max_iters):
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.max_iters = max_iters

        self.w = None

    def _g(self, w, X, y):
        XW = np.dot(X, w)
        cost = -np.sum(y * XW) + np.log(1 + np.exp(XW)).sum()
        return cost

    def _h(self, groups):
        cost = 0.
        for group in groups:
            cost += self.lambd * np.square(len(group)) * np.linalg.norm(self.w[group])
        return cost

    def _calc_cost(self, X, y, groups):
        return self._g(self.w, X, y) + self._h(groups)

    def fit_normal(self, X, y, groups):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.flatten()
        groups = [0] + groups
        groups = [np.where(groups == i)[0] for i in np.unique(groups)]

        self.w = np.zeros(X.shape[1], dtype=np.float32)

        costs = [self._calc_cost(X, y, groups)]
        for k in range(self.max_iters):
            XW = np.dot(X, self.w)
            g = np.dot(X.T, np.exp(XW) / (1 + np.exp(XW)) - y)
            w = self.w - self.learning_rate * g
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
            XW = np.dot(X, v)
            g = np.dot(X.T, np.exp(XW) / (1 + np.exp(XW)) - y)
            w = self.w - self.learning_rate * g
            for group in groups:
                w[group] = soft_threshold(w[group], self.lambd * self.learning_rate * np.square(len(group)))
            self.w = np.array(w)
            costs.append(self._calc_cost(X, y, groups))

        return np.array(costs)

    def fit_backtrack(self, X, y, groups):

        def _linesearch(w_old, t, costs):
            beta = 0.1

            XW = np.dot(X, w_old)
            g = np.dot(X.T, np.exp(XW) / (1 + np.exp(XW)) - y)

            w_new = w_old - t * g
            for group in groups:
                w_new[group] = soft_threshold(w_new[group], self.lambd * t * np.square(len(group)))

            G_t = (self.w - w_new) / t
            while self._g(w_new, X, y) > self._g(w_old, X, y) - t * np.dot(g, G_t) + 0.5 * t * np.square(G_t).sum():
                t *= beta
                w_new = w_old - t * g
                for group in groups:
                    w_new[group] = soft_threshold(w_new[group], self.lambd * t * np.square(len(group)))
                G_t = (self.w - w_new) / t
                costs.append(self._calc_cost(X, y, groups))

            return t

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.flatten()
        groups = [0] + groups
        groups = [np.where(groups == i)[0] for i in np.unique(groups)]

        self.w = np.zeros(X.shape[1], dtype=np.float32)
        t = self.learning_rate

        costs = [self._calc_cost(X, y, groups)]
        for k in range(self.max_iters):
            t = _linesearch(self.w, t, costs)

            XW = np.dot(X, self.w)
            g = np.dot(X.T, np.exp(XW) / (1 + np.exp(XW)) - y)
            w = self.w - t * g

            for group in groups:
                w[group] = soft_threshold(w[group], self.lambd * self.learning_rate * np.square(len(group)))
            self.w = np.array(w)
            costs.append(self._calc_cost(X, y, groups))

        return np.array(costs)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        XW = np.dot(X, self.w)
        return np.exp(XW) / (1 + np.exp(XW))

    def score(self, X, y):
        pred = self.predict(X)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        return np.sum(pred == y) / float(y.size)


if __name__ == "__main__":
    best_cost = 336.207

    train = sio.loadmat("data/Q4c_movies/moviesTrain.mat")
    X_train = train["trainRatings"]
    y_train = train["trainLabels"].flatten()

    test = sio.loadmat("data/Q4c_movies/moviesTest.mat")
    X_test = test["testRatings"]
    y_test = test["testLabels"].flatten()

    groups = sio.loadmat("data/Q4c_movies/moviesGroups.mat")["groupLabelsPerRating"].flatten()

    # logistic group lasso
    model1 = LogisticGroupLasso(learning_rate=10e-4, lambd=5.0, max_iters=1000)
    costs1 = model1.fit_normal(X_train, y_train, groups=groups)
    print "Logistic group lasso accuracy: {}".format(model1.score(X_test, y_test))

    # accelerated logistic group lasso
    model2 = LogisticGroupLasso(learning_rate=10e-4, lambd=5.0, max_iters=1000)
    costs2 = model2.fit_accelerate(X_train, y_train, groups=groups)
    print "Accelerated logistic group lasso accuracy: {}".format(model2.score(X_test, y_test))

    # logistic group lasso with backtracking linesearch
    model3 = LogisticGroupLasso(learning_rate=10e-4, lambd=5.0, max_iters=40)
    costs3 = model3.fit_backtrack(X_train, y_train, groups=groups)
    print "Backtracking logistic group lasso accuracy: {}".format(model3.score(X_test, y_test))

    plt.plot(costs1 - best_cost, label="normal")
    plt.plot(costs2 - best_cost, label="accelerated")
    plt.plot(costs3 - best_cost, label="backtracking")
    plt.legend(loc="upper right")
    plt.yscale("log")

    plt.show()
