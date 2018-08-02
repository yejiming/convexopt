import numpy as np
import cvxopt as cvx
import scipy.io as sio
from matplotlib import pyplot as plt


def rbf_kernel(x1, x2):
    return np.exp(-np.square(x1 - x2).sum() / 2000000)


def barrier_function(w, C):
    if np.any(w < 0):
        return np.inf
    elif np.any(w > C):
        return np.inf
    return -np.log(w).sum() - np.log(C - w).sum()


class DualKernelSVM:

    def __init__(self, C):
        self.C = C

        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def _calc_obj(self, K_hat, w):
        return 0.5 * np.dot(np.dot(w, K_hat), w) - w.sum()

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        K = np.array([[rbf_kernel(x_i, x_j) for x_j in X] for x_i in X])
        K_hat = np.array([[y[i] * y[j] * K[i][j] for j in range(K.shape[1])] for i in range(K.shape[0])])

        P = cvx.matrix(K_hat)
        q = cvx.matrix(-np.ones(y.size))

        G = cvx.matrix(np.vstack((np.eye(y.size), -np.eye(y.size))))
        h = cvx.matrix(np.hstack((np.ones(y.size) * self.C, np.zeros(y.size))))

        A = cvx.matrix(np.reshape(y, (1, -1)))
        b = cvx.matrix(0.0)

        self.w = np.array(cvx.solvers.qp(P, q, G, h, A, b, options={"show_progress": False})["x"]).flatten()

        sv = (self.w > 1e-6)
        self.b = np.mean(y[sv] - np.dot(self.w * y, K[sv, :].T))

        return self._calc_obj(K_hat, self.w)

    def predict(self, X):
        K = np.array([[rbf_kernel(x_i, x_j) for x_j in self.X_train] for x_i in X])
        preds = np.sign(np.dot(self.w * self.y_train, K.T) + self.b)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y) / float(np.size(y))


class BarrierKernelSVM:

    def __init__(self, C, alpha, beta, u, init_t, threshold):
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.u = u
        self.init_t = init_t
        self.threshold = threshold

        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def _calc_cost(self, K_hat, w, t):
        cost = t * (0.5 * np.dot(np.dot(w, K_hat), w) - w.sum())
        cost += barrier_function(w, self.C)
        return cost

    def _calc_obj(self, K_hat, w):
        return 0.5 * np.dot(np.dot(w, K_hat), w) - w.sum()

    def fit(self, X, y):

        def _init_w():
            ratio = np.sum(y > 0) / float(np.sum(y < 0))
            w = np.array([ratio if y[i] < 0 else 1.0 for i in range(y.size)])
            return w

        def _newton_step(gradient, hessian):
            P = cvx.matrix(hessian)
            q = cvx.matrix(gradient)

            A = cvx.matrix(np.reshape(y, (1, -1)))
            b = cvx.matrix(0.0)

            v = np.array(cvx.solvers.qp(P, q, A=A, b=b, options={"show_progress": False})["x"]).flatten()
            return v

        def _linesearch(K_hat, w_old, t, gradient, v):
            learning_rate = 1.0
            while self._calc_cost(K_hat, w_old + learning_rate * v, t) > self._calc_cost(K_hat, w_old, t) + \
                                    self.alpha * learning_rate * np.dot(gradient, v):
                learning_rate *= self.beta
            return learning_rate

        self.X_train = X
        self.y_train = y

        K = np.array([[rbf_kernel(x_i, x_j) for x_j in X] for x_i in X])
        K_hat = np.array([[y[i] * y[j] * K[i][j] for j in range(K.shape[1])] for i in range(K.shape[0])])

        t = self.init_t
        m = len(y) * 2
        costs = []

        w = _init_w()

        while float(m) / t > self.threshold:
            newton_costs = [self._calc_cost(K_hat, w, t)]
            while len(newton_costs) < 2 or np.abs(newton_costs[-1] - newton_costs[-2]) > self.threshold:
                gradient = t * (np.dot(K_hat, w) - 1) - 1 / w - 1 / (w - self.C)
                hessian = t * K_hat + np.diag(1 / (w ** 2)) + np.diag(1 / ((w - self.C) ** 2))

                v = _newton_step(gradient, hessian)
                learning_rate = _linesearch(K_hat, w, t, gradient, v)

                w += learning_rate * v[:X.shape[1]]

                newton_costs.append(self._calc_cost(K_hat, w, t))

            costs.append(self._calc_obj(K_hat, w))
            t *= self.u

        sv = (w > 1e-6)
        self.w = w
        self.b = np.mean(y[sv] - np.dot(w * y, K[sv, :].T))
        return np.array(costs)

    def predict(self, X):
        K = np.array([[rbf_kernel(x_i, x_j) for x_j in self.X_train] for x_i in X])
        preds = np.sign(np.dot(self.w * self.y_train, K.T) + self.b)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y) / float(np.size(y))


class InteriorPointKernelSVM:

    def __init__(self, C, alpha, beta, u):
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.u = u

        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def _calc_obj(self, K_hat, w):
        return 0.5 * np.dot(np.dot(w, K_hat), w) - w.sum()

    def fit(self, X, y):

        def _init_w():
            ratio = np.sum(y > 0) / float(np.sum(y < 0))
            w = np.array([ratio if y[i] < 0 else 1.0 for i in range(y.size)])
            return w

        def _calc_t(w, u, v):
            return -2 * self.u * y.size / np.dot(np.hstack((-w, w-self.C)), np.hstack((u, v)))

        def _calc_residuals(K_hat, w, u, v, lambd):
            r_dual = np.dot(K_hat, w) - 1 - u + v + lambd * y
            r_cent = np.hstack((np.dot(np.diag(u), -w), np.dot(np.diag(v), w - self.C))) + 1.0 / _calc_t(w, u, v)
            r_prim = np.dot(w, y)
            return r_dual, r_cent, r_prim

        def _linesearch(K_hat, w, u, v, lambd, delta_z):
            r_dual, r_cent, r_prim = _calc_residuals(K_hat, w, u, v, lambd)

            delta_w = delta_z[:n]
            delta_u = delta_z[n: 2 * n]
            delta_v = delta_z[2 * n: 3 * n]
            delta_l = delta_z[-1]

            delta = np.hstack((delta_u, delta_v))
            ratio = -np.hstack((u, v)) / delta
            theta_max = np.min([1.0, np.min(ratio[delta < 0])])

            theta = 0.99 * theta_max

            while np.any(w + theta * delta_w) <= 0 or np.any(w + theta * delta_w) >= self.C:
                theta *= self.beta

            while np.linalg.norm(np.hstack(_calc_residuals(
                K_hat, w + theta * delta_w, u + theta * delta_u,
                v + theta * delta_v, lambd + theta * delta_l)
            )) > (1 - self.alpha * theta) * np.linalg.norm(np.hstack((r_dual, r_cent, r_prim))):
                theta *= self.beta

            return theta

        self.X_train = X
        self.y_train = y

        K = np.array([[rbf_kernel(x_i, x_j) for x_j in X] for x_i in X])
        K_hat = np.array([[y[i] * y[j] * K[i][j] for j in range(K.shape[1])] for i in range(K.shape[0])])

        n = y.size
        w = _init_w()
        u = np.ones_like(w)
        v = np.ones_like(w)
        lambd = 0.

        r_dual, r_cent, r_prim = _calc_residuals(K_hat, w, u, v, lambd)

        costs = []
        while np.sqrt(np.sum(r_prim ** 2) + np.sum(r_dual ** 2)) > 1e-5 or \
                                -np.dot(-w, u) - np.dot(w - self.C, v) > 2e-5:
            A = np.vstack((
                np.hstack((K_hat, -np.eye(n), np.eye(n), np.reshape(y, (-1, 1)))),
                np.hstack((-np.diag(u), -np.diag(w), np.zeros((n, n + 1)))),
                np.hstack((np.diag(v), np.zeros((n, n)), np.diag(w - self.C), np.zeros((n, 1)))),
                np.hstack((np.reshape(y, (1, -1)), np.zeros((1, 2 * n + 1))))
            ))
            b = np.hstack((r_dual, r_cent, r_prim))
            delta_z = np.linalg.solve(A, -b)

            delta_w = delta_z[:n]
            delta_u = delta_z[n: 2 * n]
            delta_v = delta_z[2 * n: 3 * n]
            delta_l = delta_z[-1]

            theta = _linesearch(K_hat, w, u, v, lambd, delta_z)

            w += theta * delta_w
            u += theta * delta_u
            v += theta * delta_v
            lambd += theta * delta_l

            r_dual, r_cent, r_prim = _calc_residuals(K_hat, w, u, v, lambd)
            costs.append(self._calc_obj(K_hat, w))

        sv = (w > 1e-6)
        self.w = w
        self.b = np.mean(y[sv] - np.dot(w * y, K[sv, :].T))
        return np.array(costs)

    def predict(self, X):
        K = np.array([[rbf_kernel(x_i, x_j) for x_j in self.X_train] for x_i in X])
        preds = np.sign(np.dot(self.w * self.y_train, K.T) + self.b)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y) / float(np.size(y))


if __name__ == "__main__":
    train = sio.loadmat("data/Q4c_movies/moviesTrain.mat")
    X_train = train["trainRatings"].astype(np.float64)
    y_train = train["trainLabels"].flatten().astype(np.float64)
    y_train[y_train == 0] = -1

    test = sio.loadmat("data/Q4c_movies/moviesTest.mat")
    X_test = test["testRatings"].astype(np.float64)
    y_test = test["testLabels"].flatten().astype(np.float64)
    y_test[y_test == 0] = -1

    model = DualKernelSVM(C=1000)
    cost = model.fit(X_train, y_train)

    print "CVX cost: {}".format(cost)
    print "CVX Train accuracy: {}".format(model.score(X_train, y_train))
    print "CVX Test accuracy: {}".format(model.score(X_test, y_test))

    print "============================================================="

    model = BarrierKernelSVM(C=1000, alpha=0.01, beta=0.5, u=1.5, init_t=1000, threshold=1e-8)
    costs = model.fit(X_train, y_train)

    print "Barrier cost: {}".format(costs[-1])
    print "Barrier train accuracy: {}".format(model.score(X_train, y_train))
    print "Barrier test accuracy: {}".format(model.score(X_test, y_test))

    plt.figure()
    plt.plot(costs)
    plt.title("Barrier Method Costs", fontsize="large", fontweight="bold")

    print "============================================================="

    model = InteriorPointKernelSVM(C=1000, alpha=0.01, beta=0.5, u=2.0)
    costs = model.fit(X_train, y_train)

    print "Interior point cost: {}".format(costs[-1])
    print "Interior point train accuracy: {}".format(model.score(X_train, y_train))
    print "Interior point test accuracy: {}".format(model.score(X_test, y_test))

    plt.figure()
    plt.plot(costs)
    plt.title("Interior Point Method Costs", fontsize="large", fontweight="bold")

    plt.show()
