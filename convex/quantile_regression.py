import numpy as np
from matplotlib import pyplot as plt


class QuantileRegression:

    def __init__(self, alpha, max_iters):
        self.alpha = alpha
        self.max_iters = max_iters

        self.B = None

    def fit(self, X, y):

        def prox(lambd, A):
            ind = A - lambd * np.array(self.alpha)
            result1 = A - lambd * np.array(self.alpha)
            result2 = A - lambd * (np.array(self.alpha) - 1)

            result = np.zeros_like(A)
            result[ind > 0] = result1[ind > 0]
            result[ind < -lambd] = result2[ind < -lambd]

            return result

        D = np.hstack((-np.eye(len(self.alpha)-1), np.zeros((len(self.alpha)-1, 1)))) + \
            np.hstack((np.zeros((len(self.alpha)-1, 1)), np.eye(len(self.alpha)-1)))

        B = np.zeros((X.shape[1], len(self.alpha)))
        Z1 = np.zeros((X.shape[0], len(self.alpha)))
        Z2 = np.zeros((X.shape[0], len(self.alpha)))
        Z3 = np.zeros((X.shape[0], len(self.alpha) - 1))
        U1 = np.zeros_like(Z1)
        U2 = np.zeros_like(Z2)
        U3 = np.zeros_like(Z3)

        rau = 1.0
        Y = np.dot(y.reshape(-1, 1), np.ones_like(self.alpha).reshape(1, -1))

        for i in range(self.max_iters):
            B = 0.5 * np.dot(np.linalg.inv(X.T.dot(X)), X.T.dot(Y + U1 - U2 + Z2 - Z1))
            Z1 = prox(1 / rau, Y - X.dot(B) + U1)
            Z2 = np.dot(X.dot(B) + U2 + Z3.dot(D) - U3.dot(D), np.linalg.inv(D.T.dot(D) + np.eye(len(self.alpha))))
            Z3 = np.maximum(Z2.dot(D.T) + U3, 0)
            U1 = Y - X.dot(B) - Z1 + U1
            U2 = X.dot(B) - Z2 + U2
            U3 = Z2.dot(D.T) - Z3 + U3

        self.B = B

    def predict(self, X):
        Y = X.dot(self.B)
        return Y


def draw_quantile_lines(model, X, y):
    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.scatter(X[:, 0], y, label="samples")
    for i in range(len(model.alpha)):
        w = model.B[:, i]
        line = np.array([[X[:, 0].min(), w[0] * X[:, 0].min() + w[1]],
                         [X[:, 0].max(), w[0] * X[:, 0].max() + w[1]]])
        plt.plot(line[:, 0], line[:, 1], label="a={}".format(model.alpha[i]))
    plt.legend(loc="upper left")


if __name__ == "__main__":
    alpha = [0.1, 0.5, 0.9]
    X = np.loadtxt("data/hw4q4/X.csv", delimiter=",")
    y = np.loadtxt("data/hw4q4/y.csv", delimiter=",")

    model = QuantileRegression(alpha=alpha, max_iters=50)
    model.fit(X, y)

    draw_quantile_lines(model, X, y)
    plt.show()
