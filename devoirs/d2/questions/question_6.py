import numpy as np
from devoirs.d2.code.mlp_batch import MLPBatch
from devoirs.d2.code.mlp import MLP


dh = 300
d = 2
m = 2
k = 100
iterations = 1000
epsilon = 1e-5

lamdas = np.matrix([[0.001, 0.0001],[0.001, 0.00006]])
learning_rate = 0.019

data = np.matrix(np.loadtxt(open('../2moons.txt', 'r')))

np.random.seed(1234)

train = data[:, :-1]
target = data[:, -1].getA1()

nnet_batch = MLPBatch(d, m, dh, epsilon)
nnet = MLP(d, m, dh, epsilon)

nnet.W1 = nnet_batch.W1.copy()
nnet.W2 = nnet_batch.W2.copy()
nnet.b1 = nnet_batch.b1.copy()
nnet.b2 = nnet_batch.b2.copy()

nnet_batch.train(train, target, lamdas, learning_rate , k, iterations=iterations)
nnet_batch.show_decision_regions(data, title='Question6_batch_k_10')

nnet.train(train, target, lamdas, learning_rate, k, iterations=iterations)
nnet.show_decision_regions(data, title='Question6_boucle_k_10')

print(nnet_batch.total_grad)
print(nnet.total_grad)