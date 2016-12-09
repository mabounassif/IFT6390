import math
import numpy as np
import time

import matplotlib.pyplot as plt
import pylab

from code.verification_helpers import check_grad_w2, check_grad_b2, check_grad_w1, check_grad_b1
from code.mlp_helpers import fprop, bprop


class MLP:
    def __init__(self, d, m, dh, epsilon, show_epoch=False):
        self.total_grad = 0
        self.m = m
        self.epsilon = epsilon
        self.d = d
        self.dh = dh
        self.show_epoch = show_epoch

        # Initialisation des paramètres
        self.W1 = np.random.uniform(
            -1 / math.sqrt(d),
            1 / math.sqrt(d),
            (dh, d))
        self.b1 = np.zeros((dh, 1))

        self.W2 = np.random.uniform(
            -1 / math.sqrt(dh),
            1 / math.sqrt(dh),
            (m, dh))
        self.b2 = np.zeros((m, 1))

    def verify_gradient(self, train, target, k):
        for i in list(range(k)):
            x = train[i]
            y = target[i]

            fprop_r = fprop(self.W1, self.W2, self.b1, self.b2, x, y)
            bprop_r = bprop(fprop_r, self.W1, self.W2, self.b1, self.b2, x, y, self.m)
            L = fprop_r['loss']

            grad_w2_diff = check_grad_w2(L, self.W1, self.W2, self.b1, self.b2, x, y, self.epsilon)
            grad_w1_diff = check_grad_w1(L, self.W1, self.W2, self.b1, self.b2, x, y, self.epsilon)
            grad_b2_diff = check_grad_b2(L, self.W1, self.W2, self.b1, self.b2, x, y, self.epsilon)
            grad_b1_diff = check_grad_b1(L, self.W1, self.W2, self.b1, self.b2, x, y, self.epsilon)

            grad_ratio_b1 = (bprop_r['grad_b1'] + self.epsilon) / (grad_b1_diff + self.epsilon)
            grad_ratio_w1 = (bprop_r['grad_w1'] + self.epsilon) / (grad_w1_diff + self.epsilon)
            grad_ratio_b2 = (bprop_r['grad_b2'] + self.epsilon) / (grad_b2_diff + self.epsilon)
            grad_ratio_w2 = (bprop_r['grad_w2'] + self.epsilon) / (grad_w2_diff + self.epsilon)

            def check_grad_ratio(ratio):
                return (ratio > 0.99).all() and (ratio < 1.01).all()

            if check_grad_ratio(grad_ratio_b2) and check_grad_ratio(grad_ratio_w2) and check_grad_ratio(
                    grad_ratio_b1) and check_grad_ratio(grad_ratio_w1):
                print('Gradient verified for element {0} ✓'.format(i))
            else:
                print('Gradient error for element {0} X'.format(i))

    def calculate_and_show_errors(self, train, train_target, valid, valid_target, test, test_target):
        pass

    def train(self, train, target, lamdas, learning_rate, k=None, iterations=100):
        cursor = 0
        self.total_grad = 0
        t = time.process_time()

        if k is None:
            batch_size = train.shape[0]
        else:
            batch_size = k

        for _ in range(iterations):
            total_grad_w1 = 0
            total_grad_w2 = 0
            total_grad_b1 = 0
            total_grad_b2 = 0
            total_grad_oa = 0

            for _ in range(batch_size):
                x = train[cursor]
                y = target[cursor]

                fprop_r = fprop(self.W1, self.W2, self.b1, self.b2, x, y)
                bprop_r = bprop(fprop_r, self.W1, self.W2, self.b1, self.b2, x, y, self.m)

                self.total_grad += np.sum(bprop_r['grad_oa'])
                total_grad_w1 += bprop_r['grad_w1']
                total_grad_w2 += bprop_r['grad_w2']
                total_grad_b1 += bprop_r['grad_b1']
                total_grad_b2 += bprop_r['grad_b2']
                total_grad_oa += bprop_r['grad_oa']

                cursor += 1
                if cursor >= train.shape[0] and self.show_epoch:
                    elapsed_time = time.process_time() - t
                    print('1 epoch time: ~{0} s'.format(elapsed_time))

                cursor = (cursor%train.shape[0])

            self.total_grad += np.sum(total_grad_oa)

            regularization = lamdas[0, 0] * self.W1.sum() + \
                         lamdas[0, 1] * np.square(self.W1).sum() + \
                         lamdas[1, 0] * self.W2.sum() + \
                         lamdas[1, 1] * np.square(self.W2).sum()

            self.W1 -= (learning_rate * (total_grad_w1 + regularization))
            self.W2 -= (learning_rate * (total_grad_w2 + regularization))

            self.b1 -= np.sum((learning_rate * total_grad_b1), axis=1)
            self.b2 -= np.sum((learning_rate * total_grad_b2), axis=1)

    def show_decision_regions(self, train_data, title='region de décision'):
        def combine(*seqin):
            '''returns a list of all combinations of argument sequences.
            for example: combine((1,2),(3,4)) returns
            [[1, 3], [1, 4], [2, 3], [2, 4]]'''

            def rloop(seqin, listout, comb):
                '''recursive looping function'''
                if seqin:  # any more sequences to process?
                    for item in seqin[0]:
                        newcomb = comb + [item]  # add next item to current comb
                        # call rloop w/ rem seqs, newcomb
                        rloop(seqin[1:], listout, newcomb)
                else:  # processing last sequence
                    listout.append(comb)  # comb finished, add to list

            listout = []  # listout initialization
            rloop(seqin, listout, [])  # start recursive process
            return listout

        d1 = train_data[train_data[:, -1].getA1() > 0]
        d2 = train_data[train_data[:, -1].getA1() == 0]

        plt.figure()

        plt.scatter(d1[:, 0], d1[:, 1], c='b', label='classe 1')
        plt.scatter(d2[:, 0], d2[:, 1], c='g', label='classe 0')

        xgrid = np.linspace(np.min(train_data[:, 0]) - 0.5,
                            np.max(train_data[:, 0]) + 0.5,
                            100)
        ygrid = np.linspace(np.min(train_data[:, 1]) - 0.5,
                            np.max(train_data[:, 1]) + 0.5,
                            100)

        # calcule le produit cartesien entre deux listes
        # et met les resultats dans un array
        thegrid = np.matrix(combine(xgrid, ygrid))

        classesPred = []
        for x in thegrid:
            os = fprop(self.W1, self.W2, self.b1, self.b2, x)['os']
            classesPred.append(np.argmax(os, axis=0) + 1)

        pylab.pcolormesh(xgrid, ygrid, np.array(classesPred).reshape((100, 100)).T, alpha=.3)
        pylab.pcolormesh(xgrid, ygrid, np.array(classesPred).reshape((100, 100)).T, alpha=.3)
        plt.xlim(np.min(train_data[:, 0]) - 0.5, np.max(train_data[:, 0]) + 0.5)
        plt.ylim(np.min(train_data[:, 1]) - 0.5, np.max(train_data[:, 1]) + 0.5)
        plt.grid()
        plt.legend(loc='lower right')
        plt.title(title)