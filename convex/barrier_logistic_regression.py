import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


def barrier_function(w, c):
    if np.any(c < w[1:]):
        return np.inf
    elif np.any(w[1:] < -c):
        return np.inf
    return -np.log(c - w[1:]).sum() - np.log(c + w[1:]).sum()


class BarrierLogisticRegression:

    def __init__(self, alpha, beta, lambd, threshold, u, init_t):
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.u = u
        self.init_t = init_t

        self.lambd = lambd

        self.w = None

    def _calc_cost(self, X, y, w, c, t):
        XW = np.dot(X, w)
        cost = (-np.sum(y * XW) + np.log(1 + np.exp(XW)).sum() + self.lambd * np.sum(c)) * t
        cost += barrier_function(w, c)
        return cost

    def _calc_obj(self, X, y, w):
        XW = np.dot(X, w)
        cost = -np.sum(y * XW) + np.log(1 + np.exp(XW)).sum()
        cost += self.lambd * np.linalg.norm(w[1:], ord=1)
        return cost

    def fit(self, X, y):

        def _linesearch(w_old, c_old, gradient, hessian, t):
            learning_rate = 1.0

            v = -np.dot(np.linalg.inv(hessian), gradient)
            w_v = v[:X.shape[1]]
            c_v = v[X.shape[1]:]

            while self._calc_cost(X, y, w_old + learning_rate * w_v, c_old + learning_rate * c_v, t) > \
                            self._calc_cost(X, y, w_old, c_old, t) + self.alpha * learning_rate * np.dot(gradient, v):
                learning_rate *= self.beta

            return learning_rate

        X = np.hstack((np.ones((X.shape[0], 1)), X)).astype(np.float64)
        y = y.flatten()

        t = self.init_t
        self.w = np.zeros(X.shape[1], dtype=np.float64)
        c = np.ones(X.shape[1] - 1, dtype=np.float64)
        m = 2 * (X.shape[1] - 1)
        costs = []
        while float(m) / t > self.threshold:
            newton_costs = [self._calc_cost(X, y, self.w, c, t)]
            while len(newton_costs) < 2 or np.abs(newton_costs[-1] - newton_costs[-2]) > self.threshold:
                p = 1 - 1 / (1 + np.exp(np.dot(X, self.w)))

                g_dem1 = 1 / (c - self.w[1:])
                g_dem2 = 1 / (c + self.w[1:])

                h_dem1 = np.diag(g_dem1 ** 2)
                h_dem2 = np.diag(g_dem2 ** 2)

                gradient = np.zeros(2 * X.shape[1] - 1)
                gradient[:X.shape[1]] += t * np.dot(X.T, p - y)
                gradient[1:X.shape[1]] += g_dem1 - g_dem2
                gradient[X.shape[1]:] += t * self.lambd - g_dem1 - g_dem2

                hessian = np.zeros((2 * X.shape[1] - 1, 2 * X.shape[1] - 1))
                hessian[:X.shape[1], :X.shape[1]] += t * np.dot(np.dot(X.T, np.diag(p * (1 - p))), X)

                hessian[1:X.shape[1], 1:X.shape[1]] += h_dem1 + h_dem2
                hessian[X.shape[1]:, X.shape[1]:] += h_dem1 + h_dem2

                hessian[1:X.shape[1], X.shape[1]:] += h_dem2 - h_dem1
                hessian[X.shape[1]:, 1:X.shape[1]] += h_dem2 - h_dem1

                v = -np.dot(np.linalg.inv(hessian), gradient)

                learning_rate = _linesearch(self.w, c, gradient, hessian, t)
                self.w += learning_rate * v[:X.shape[1]]
                c += learning_rate * v[X.shape[1]:]

                newton_costs.append(self._calc_cost(X, y, self.w, c, t))

            costs.append(self._calc_obj(X, y, self.w))
            t *= self.u

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
    best_cost = 306.476

    train = sio.loadmat("data/hwk4_moviesTrain.mat")
    X_train = train["trainRatings"]
    y_train = train["trainLabels"].flatten()

    test = sio.loadmat("data/hwk4_moviesTest.mat")
    X_test = test["testRatings"]
    y_test = test["testLabels"].flatten()

    model = BarrierLogisticRegression(alpha=0.2, beta=0.9, lambd=15.0, threshold=1e-9, u=20, init_t=5)
    costs = model.fit(X_train, y_train)
    print "Train accuracy: {}".format(model.score(X_train, y_train))
    print "Test accuracy: {}".format(model.score(X_test, y_test))
    print "Number of zeros at the weights:", np.sum(np.abs(model.w) <= 1e-10)

    plt.plot(costs - best_cost)
    plt.yscale("log")

    plt.show()
