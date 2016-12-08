import numpy as np
import math

from devoirs.d2.code.mlp_helpers import fprop, bprop
from devoirs.d2.code.verification_helpers import check_grad_w1, check_grad_b1, check_grad_w2, check_grad_b2

np.random.seed(1234)

data = np.matrix([[12, 22, 0]])

x = data[:,:-1]
y = data[:,-1].getA1()

dh = 2
d = 2
m = 2
epsilon = 1e-5

# Initialisation des paramètres
W1 = np.random.uniform(
    -1 / math.sqrt(d),
    1 / math.sqrt(d),
    (dh, d))
b1 = np.zeros((dh, 1))

W2 = np.random.uniform(
    -1 / math.sqrt(dh),
    1 / math.sqrt(dh),
    (m, dh))
b2 = np.zeros((m, 1))

fprop_r = fprop(W1, W2, b1, b2, x, y)
bprop_r = bprop(fprop_r, W1, W2, b1, b2, x, y, m)
L = fprop_r['loss']

# Check W2 grad
grad_w2_diff = check_grad_w2(L, W1, W2, b1, b2, x, y, epsilon)

print('Check gradient of w2: \n')
print(grad_w2_diff)
print((bprop_r['grad_w2'] + epsilon) / (grad_w2_diff + epsilon))

# Check b2 grad
grad_b2_diff = check_grad_b2(L, W1, W2, b1, b2, x, y, epsilon)

print('Check gradient of b2: \n')
print(grad_b2_diff)
print((bprop_r['grad_b2'] + epsilon) / (grad_b2_diff + epsilon))

# Check W1 grad
grad_w1_diff = check_grad_w1(L, W1, W2, b1, b2, x, y, epsilon)

print('Check gradient of w1: \n')
print(grad_w1_diff)
print((bprop_r['grad_w1'] + epsilon) / (grad_w1_diff + epsilon))

# Check b1 grad
grad_b1_diff = check_grad_b1(L, W1, W2, b1, b2, x, y, epsilon)


print('Check gradient of b1: \n')
print(grad_b1_diff)
print((bprop_r['grad_b1'] + epsilon) / (grad_b1_diff + epsilon))