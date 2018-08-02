import h5py

import numpy as np
import cvxpy as cp
import cvxopt as cvx
import matplotlib.pyplot as plt


def plot_dual_variables(X, y, primal, dual):
    distances = (np.matmul(X, primal.w) + primal.b) * y
    plt.figure()
    plt.scatter(distances, dual.a, color="b", marker=".")
    plt.title("Dual Variables", fontsize="large", fontweight="bold")


class WeightedSVM(object):

    def __init__(self, C1, C2):
        self.C1 = C1
        self.C2 = C2

        self.a = None
        self.w = None
        self.b = None

    def predict(self, X):
        pred = np.matmul(X, self.w) + self.b
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        score = np.sum((y == 1) * (pred == -1)) * self.C1
        score += np.sum((y == -1) * (pred == 1)) * self.C2
        return score

    def plot_boundries(self, X, y, title=None):
        X1 = X[y == 1]
        X2 = X[y == -1]

        boundry = np.array([[X[:, 0].min(), (-self.b - X[:, 0].min() * self.w[0]) / self.w[1]],
                            [X[:, 0].max(), (-self.b - X[:, 0].max() * self.w[0]) / self.w[1]]])
        margin1 = np.array([[X[:, 0].min(), (1 -self.b - X[:, 0].min() * self.w[0]) / self.w[1]],
                            [X[:, 0].max(), (1 -self.b - X[:, 0].max() * self.w[0]) / self.w[1]]])
        margin2 = np.array([[X[:, 0].min(), (-1 -self.b - X[:, 0].min() * self.w[0]) / self.w[1]],
                            [X[:, 0].max(), (-1 -self.b - X[:, 0].max() * self.w[0]) / self.w[1]]])

        sv1 = X1[np.matmul(X1, self.w) + self.b <= 1]
        sv2 = X2[np.matmul(X2, self.w) + self.b >= -1]

        plt.figure()

        plt.plot(boundry[:, 0], boundry[:, 1], "g-")
        plt.plot(margin1[:, 0], margin1[:, 1], "g--")
        plt.plot(margin2[:, 0], margin2[:, 1], "g--")

        plt.scatter(X1[:, 0], X1[:, 1], color="b", marker=".")
        plt.scatter(X2[:, 0], X2[:, 1], color="b", marker="x")

        plt.scatter(sv1[:, 0], sv1[:, 1], color="r", marker=".")
        plt.scatter(sv2[:, 0], sv2[:, 1], color="r", marker="x")

        if title is not None:
            plt.title(title, fontsize="large", fontweight="bold")


class PrimalWeightedSVM(WeightedSVM):

    def __init__(self, C1, C2):
        super(PrimalWeightedSVM, self).__init__(C1, C2)

    def fit(self, X, y):
        weights = np.array([self.C1 if y_i == 1 else self.C2 for y_i in y])

        beta = cp.Variable(X.shape[1], name="beta")
        beta_0 = cp.Variable(name="beta_0")
        phi = cp.Variable(X.shape[0], name="phi")

        objective = 0.5 * cp.sum_squares(beta) + cp.sum(cp.multiply(phi, weights))
        constraints = [phi >= 0, cp.multiply(y, X * beta + beta_0) >= 1 - phi]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        prob.solve()

        self.w = beta.value
        self.b = beta_0.value

    def predict(self, X):
        pred = np.matmul(X, self.w) + self.b
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        score = np.sum((y == 1) * (pred == -1)) * self.C1
        score += np.sum((y == -1) * (pred == 1)) * self.C2
        return score


class DualWeightedSVM(WeightedSVM):

    def __init__(self, C1, C2):
        super(DualWeightedSVM, self).__init__(C1, C2)

    def fit(self, X, y):
        weights = np.array([self.C1 if y_i == 1 else self.C2 for y_i in y])

        P = cvx.matrix(np.outer(y, y) * np.dot(X, X.T))
        q = cvx.matrix(-np.ones(y.size))

        A = cvx.matrix(np.reshape(y, (1, -1)))
        b = cvx.matrix(0.0)

        G = cvx.matrix(np.vstack((np.eye(y.size), -np.eye(y.size))))
        h = cvx.matrix(np.hstack((weights, np.zeros(y.size))))

        a = np.array(cvx.solvers.qp(P, q, G, h, A, b, options={"show_progress": False})["x"]).flatten()
        sv = (a > 1e-8)

        self.a = a
        self.w = np.dot(X.T, y * a)
        self.b = np.mean(y[sv] - np.dot(X[sv, :], self.w))

    def predict(self, X):
        pred = np.matmul(X, self.w) + self.b
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        score = np.sum((y == 1) * (pred == -1)) * self.C1
        score += np.sum((y == -1) * (pred == 1)) * self.C2
        return score


if __name__ == "__main__":
    f = h5py.File("data/toy.hdf5")
    X = f["X"][:].T
    y = f["y"][:]

    primal = PrimalWeightedSVM(C1=1.0, C2=1.0)
    primal.fit(X, y)
    primal.plot_boundries(X, y, "Primal SVM")

    dual = DualWeightedSVM(C1=1.0, C2=1.0)
    dual.fit(X, y)
    dual.plot_boundries(X, y, "Dual SVM")

    plot_dual_variables(X, y, primal, dual)

    plt.show()
