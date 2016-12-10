import numpy as np
import time

from code.mlp_helpers import fprop, bprop
from code.verification_helpers import check_grad_b1, check_grad_w1, check_grad_b2, check_grad_w2
from code.mlp import MLP

import json


class MLPBatch(MLP):
    def train(self, train, target, lamdas, learning_rate, k=None, iterations=100, valid=None, valid_target=None, test=None, test_target=None):
        t = time.process_time()
        self.total_grad = 0
        cursor = 0
        axis = 1

        if k is None:
            batch_size = train.shape[0]
        else:
            batch_size = k

        for _ in range(iterations):
            x = np.roll(train, -1*cursor*batch_size, axis=0)[:batch_size]
            y = np.roll(target, -1*cursor*batch_size, axis=0)[:batch_size]

            fprop_r = fprop(self.W1, self.W2, self.b1, self.b2, x, y)
            bprop_r = bprop(fprop_r, self.W1, self.W2, self.b1, self.b2, x, y, self.m)

            self.total_grad += np.sum(bprop_r['grad_oa'])

            regularization = lamdas[0, 0] * self.W1.sum() + \
                             lamdas[0, 1] * np.square(self.W1).sum() + \
                             lamdas[1, 0] * self.W2.sum() + \
                             lamdas[1, 1] * np.square(self.W2).sum()

            self.W1 -= (learning_rate * (bprop_r['grad_w1'] + regularization))
            self.W2 -= (learning_rate * (bprop_r['grad_w2'] + regularization))

            self.b1 -= np.sum((learning_rate * bprop_r['grad_b1']), axis=axis)
            self.b2 -= np.sum((learning_rate * bprop_r['grad_b2']), axis=axis)

            cursor += 1
            if cursor*batch_size >= train.shape[0]:
                if self.show_epoch:
                    elapsed_time = time.process_time() - t
                    print('1 epoch time: ~{0} s'.format(elapsed_time))

                if self.save_datapoints:
                    self.calculate_and_show_errors(train, target, valid, valid_target, test, test_target)

                cursor = 0

        if self.save_datapoints:
            f = open('datapoints.json', 'w+')
            f.write(json.dumps(self.data_points))
            f.close()

    def verify_gradient(self, train, target, k):
        x = train[:k]
        y = target[:k]

        fprop_r = fprop(self.W1, self.W2, self.b1, self.b2, x, y)
        bprop_r = bprop(fprop_r, self.W1, self.W2, self.b1, self.b2, x, y, self.m)

        L = np.sum(fprop_r['loss'])

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
            print('Gradient verified for k={0} ✓'.format(k))
        else:
            print('Gradient error for k={0} X'.format(k))
