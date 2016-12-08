dh = 2
d = 2
m = 2
k = 1
epsilon = 1e-5

import numpy as np
from devoirs.d2.code.mlp import MLP

np.random.seed(1234)

data = np.matrix([[12, 22, 0]])

x = data[:,:-1]
y = data[:,-1].getA1()

nnet = MLP(d, m, dh, epsilon)
nnet.verify_gradient(x, y, k)