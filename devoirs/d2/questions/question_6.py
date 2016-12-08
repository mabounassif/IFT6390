import numpy as np
from devoirs.d2.code.mlp_batch import MLPBatch
from devoirs.d2.code.mlp import MLP


dh = 300
d = 2
m = 2
k = 10
epsilon = 1e-5

lamdas = np.matrix([[0.001, 0.0001],[0.0001, 0.00006]])
learning_rate = 0.26

data = np.matrix(np.loadtxt(open('../2moons.txt', 'r')))

np.random.seed(1234)

train = data[:, :-1]
target = data[:, -1].getA1()

nnet_batch = MLPBatch(d, m, dh, epsilon)
nnet_batch.train(train, target, lamdas, learning_rate, k=k)
nnet_batch.show_decision_regions(data, title='Question6_batch_k_10')

nnet = MLP(d, m, dh, epsilon)
nnet.batch_train(train, target, lamdas, learning_rate, k=k)
nnet.show_decision_regions(data, title='Question6_boucle_k_10')