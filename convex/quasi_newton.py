import time

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


def random_positive_definite_matrix(n):
    D = np.diag([1 for _ in range(n-1)] + [10])
    P = np.linalg.qr(np.random.normal(size=(n, n)))[0]
    Q = P.T.dot(D).dot(P)
    return Q


def linesearch(f, df, x, v, alpha=0.0001, beta=0.9):
    t = 1.0
    g = df(x)
    while f(x + t * v) > f(x) + alpha * t * np.dot(g, v):
        t *= beta
    return t


def steepest_update(H, s, y):
    return H


def dfp_update(H, s, y):
    return H - H.dot(y).dot(y.T).dot(H) / y.T.dot(H).dot(y) + s.dot(s.T) / (y.T.dot(s) + 1e-15)


def bfgs_update(H, s, y):
    I = np.eye(H.shape[0])
    return (I - s.dot(y.T) / (y.T.dot(s) + 1e-15)).dot(H).dot(I - y.dot(s.T) / (y.T.dot(s)) + 1e-15) + \
           s.dot(s.T) / (y.T.dot(s) + 1e-15)


def lbfgs_update(H_0, g, S, Y, m):
    k = len(S)
    alpha = [0 for _ in range(k)]

    q = -g.reshape((-1, 1))
    for i in range(max(k - m, 0), k)[::-1]:
        alpha[i] = S[i].T.dot(q) / (Y[i].T.dot(S[i]) + 1e-15)
        q -= alpha[i] * Y[i]

    p = H_0.dot(q)
    for i in range(max(k - m, 0), k):
        beta = Y[i].T.dot(p) / (Y[i].T.dot(S[i]) + 1e-15)
        p += (alpha[i] - beta) * S[i]

    return np.ravel(p)


def quadratic_program(n, method):

    Q = random_positive_definite_matrix(n)
    b = np.zeros(n)

    def f(x):
        return 0.5 * x.T.dot(Q).dot(x) - b.T.dot(x)

    def df(x):
        return np.dot(Q, x) - b

    x = np.random.normal(size=n)
    H = np.eye(n)

    for k in range(100):
        if abs(f(x)) < 1e-8:
            break
        g = df(x)
        p = -np.dot(H, g)
        t = linesearch(f, df, x, p)
        s = t * p

        x_new = x + s
        y = df(x_new) - df(x)
        if method == "dfp":
            H = dfp_update(H, s.reshape((-1, 1)), y.reshape((-1, 1)))
        elif method == "bfgs":
            H = bfgs_update(H, s.reshape((-1, 1)), y.reshape((-1, 1)))
        else:
            raise ValueError("Quasi Newton update method is not applicable.")

        x = x_new

    return k, f(x), np.linalg.norm(np.linalg.inv(H) - Q, ord="fro")


def rosenbrock(method, **kwargs):

    def f(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def df(x):
        return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

    x = np.array([-1.2, 1])
    if method == "exact":
        H = np.array([[-400 * x[0] * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]])
    else:
        H = np.eye(2)

    obj, path, S, Y = [f(x)], [x], [], []
    for k in range(100):
        g = df(x)
        if method == "lbfgs" and k > 0:
            p = lbfgs_update(H, g, S, Y, kwargs["m"])
        else:
            p = -np.dot(H, g)
        t = linesearch(f, df, x, p)
        s = t * p

        if method == "lbfgs":
            if np.all(np.abs(df(x)) * np.maximum(np.abs(x), 1) / max(f(x), 1) < 1e-8):
                break
            if np.all(np.abs(s) / np.maximum(np.abs(x), 1) < 4 * 1e-15):
                break

        x_new = x + s
        y = df(x_new) - df(x)

        s = s.reshape((-1, 1))
        y = y.reshape((-1, 1))
        S.append(s)
        Y.append(y)

        if method == "steepest":
            H = np.eye(2)
        elif method == "dfp":
            H = dfp_update(H, s, y)
        elif method == "bfgs":
            H = bfgs_update(H, s, y)
        elif method == "lbfgs":
            H = y.T.dot(s) / (y.T.dot(y) + 1e-15) * np.eye(2)
        else:
            H = H

        x = x_new
        obj.append(f(x))
        path.append(x)

    return obj, path


def draw_rosenbrock_contour():
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2

    plt.figure()
    plt.contour(X, Y, Z, 50, alpha=0.75)


def draw_rosenbrock_path(path, color, label):
    x1 = np.array([i[0] for i in path])
    x2 = np.array([i[1] for i in path])
    plt.plot(x1, x2, color=color, label=label)
    plt.scatter(x1[-1], x2[-1], marker="o")


class RidgeLogisticRegression:

    def __init__(self, m, alpha, beta, lambd, max_iters, optimizer="lbfgs"):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.max_iters = max_iters

        self.optimizer = optimizer

        self.w = None

    def fit(self, X, y):

        def f(w):
            XW = np.dot(X, w)
            cost = -np.sum(y * XW) + np.log(1 + np.exp(XW)).sum()
            cost += self.lambd * np.linalg.norm(w, ord=2)
            return cost

        def df(w):
            XW = np.dot(X, w)
            g = np.dot(X.T, np.exp(XW) / (1 + np.exp(XW)) - y)
            g += 2 * self.lambd * w
            return g

        X = np.hstack((np.ones((X.shape[0], 1)), X))

        w = np.zeros(X.shape[1], dtype=np.float32)
        H = np.eye(X.shape[1])
        S, Z = [], []
        for k in range(self.max_iters):
            g = df(w)
            if self.optimizer == "lbfgs" and k > 0:
                p = lbfgs_update(H, g, S, Z, self.m)
            else:
                p = -np.dot(H, g)
            t = linesearch(f, df, w, p)
            s = t * p

            if np.all(np.abs(df(w)) * np.maximum(np.abs(w), 1) / max(f(w), 1) < 1e-8):
                break
            if np.all(np.abs(s) / np.maximum(np.abs(w), 1) < 4 * 1e-15):
                break

            w_new = w + s
            z = df(w_new) - df(w)

            s = s.reshape((-1, 1))
            z = z.reshape((-1, 1))
            S.append(s)
            Z.append(z)

            if self.optimizer == "lbfgs":
                H = z.T.dot(s) / (z.T.dot(z) + 1e-15) * np.eye(X.shape[1])
            elif self.optimizer == "bfgs":
                H = bfgs_update(H, s, z)
            else:
                raise ValueError("optimization method is not applicable.")

            w = w_new

        self.w = w

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
    ##################################
    # Quadratic programming problem  #
    ##################################
    k, obj, norm = quadratic_program(5, "bfgs")
    print "Iterations: {}, objective value: {}, norm of difference: {}".format(k, obj, norm)

    ##################################
    # Rosenbrock problem             #
    ##################################
    obj1, path1 = rosenbrock("steepest")
    obj2, path2 = rosenbrock("dfp")
    obj3, path3 = rosenbrock("bfgs")
    obj4, path4 = rosenbrock("exact")
    obj5, path5 = rosenbrock("lbfgs", m=3)

    plt.figure()
    plt.plot(obj1, label="steepest")
    plt.plot(obj2, label="dfp")
    plt.plot(obj3, label="bfgs")
    plt.plot(obj4, label="exact")
    plt.plot(obj5, label="lbfgs")
    plt.legend(loc="lower right")
    plt.yscale("log")

    draw_rosenbrock_contour()
    draw_rosenbrock_path(path1, "red", label="steepest")
    draw_rosenbrock_path(path2, "blue", label="dfp")
    draw_rosenbrock_path(path3, "green", label="bfgs")
    draw_rosenbrock_path(path4, "purple", label="exact")
    draw_rosenbrock_path(path5, "gray", label="lbfgs")

    plt.show()

    #############################################
    # L2-penalized logistic regression problem  #
    #############################################
    train = sio.loadmat("data/Q4c_movies/moviesTrain.mat")
    X_train = train["trainRatings"]
    y_train = train["trainLabels"].flatten()

    test = sio.loadmat("data/Q4c_movies/moviesTest.mat")
    X_test = test["testRatings"]
    y_test = test["testLabels"].flatten()

    start = time.time()
    model1 = RidgeLogisticRegression(m=7, alpha=0.0001, beta=0.9, lambd=100, max_iters=10000, optimizer="lbfgs")
    model1.fit(X_train, y_train)
    print "L-BFGS    accuracy: {}, cost time: {}".format(model1.score(X_test, y_test), time.time() - start)

    start = time.time()
    model2 = RidgeLogisticRegression(m=7, alpha=0.0001, beta=0.9, lambd=100, max_iters=10000, optimizer="bfgs")
    model2.fit(X_train, y_train)
    print "BFGS      accuracy: {}, cost time: {}".format(model2.score(X_test, y_test), time.time() - start)
