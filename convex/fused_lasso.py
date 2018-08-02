import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt


def calc_distance(labels, pred):
    return 0.5 * cvx.sum_squares(labels - pred)


def calc_norm(pred):
    hnorm = cvx.abs(pred[:, :-1] - pred[:, 1:])
    vnorm = cvx.abs(pred[:-1] - pred[1:])
    return cvx.sum(hnorm) + cvx.sum(vnorm)


def show_image(result):
    plt.imshow(result, cmap="gray")
    plt.show()


def show_histogram(result):
    plt.hist(result, bins=100)
    plt.xlim(0, 1)
    plt.show()


if __name__ == "__main__":
    k = 6
    LAMBDA = 10 ** (-k / 4.0)

    labels = np.loadtxt("data/lenna_64.csv", delimiter=",")
    result = cvx.Variable(labels.shape)

    cost = calc_distance(labels, result) + LAMBDA * calc_norm(result)
    prob = cvx.Problem(cvx.Minimize(cost))
    prob.solve()

    show_image(labels)
    show_image(result.value)
    show_histogram(result.value)
