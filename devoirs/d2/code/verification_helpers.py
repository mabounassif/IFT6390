import numpy as np
from code.mlp_helpers import fprop


def check_grad_w2(L, W1, W2, b1, b2, x, y, epsilon):
    grad_diff = np.zeros(W2.shape)

    for i in list(range(W2.shape[0])):
        for j in list(range(W2.shape[1])):
            W2[i, j] += epsilon

            fprop_r_diff = fprop(W1, W2, b1, b2, x, y)

            L_prime = fprop_r_diff['loss']

            W2[i, j] -= epsilon

            grad_diff[i, j] = (L_prime - L) / epsilon

    return grad_diff


def check_grad_b2(L, W1, W2, b1, b2, x, y, epsilon):
    grad_diff = np.zeros(b2.shape)

    for i in list(range(b2.shape[0])):
        b2[i, 0] += epsilon

        fprop_r_diff = fprop(W1, W2, b1, b2, x, y)

        L_prime = fprop_r_diff['loss']

        b2[i, 0] -= epsilon

        grad_diff[i, 0] = (L_prime - L) / epsilon

    return grad_diff


def check_grad_w1(L, W1, W2, b1, b2, x, y, epsilon):
    grad_diff = np.zeros(W1.shape)

    for i in list(range(W1.shape[0])):
        for j in list(range(W1.shape[1])):
            W1[i, j] += epsilon

            fprop_r_diff = fprop(W1, W2, b1, b2, x, y)

            L_prime = fprop_r_diff['loss']

            W1[i, j] -= epsilon

            grad_diff[i,j] = (L_prime - L) / epsilon

    return grad_diff


def check_grad_b1(L, W1, W2, b1, b2, x, y, epsilon):
    grad_diff = np.zeros(b1.shape)

    for i in list(range(b1.shape[0])):
        b1[i, 0] += epsilon

        fprop_r_diff = fprop(W1, W2, b1, b2, x, y)

        L_prime = np.sum(fprop_r_diff['loss'])

        b1[i, 0] -= epsilon

        grad_diff[i, 0] = (L_prime - L) / epsilon

    return grad_diff