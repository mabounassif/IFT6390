dh = 2
d = 2
m = 2
k = 10
epsilon = 1e-5

import numpy as np
from devoirs.d2.code.mlp import MLP

data = np.loadtxt(open('../2moons.txt', 'r'))

np.random.seed(1234)

train = np.matrix(data[:k, :-1])
target = data[:k, -1]

nnet = MLP(d, m, dh, epsilon)
nnet.verify_gradient(train, target, k)