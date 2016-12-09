import numpy as np
from devoirs.d2.code.mlp_batch import MLPBatch
from devoirs.d2.code.mlp import MLP


dh = 300
d = 2
m = 2
k = 10
iterations = 1000
epsilon = 1e-5

lamdas = np.matrix([[0.001, 0.0001],[0.001, 0.00006]])
learning_rate = 0.019

data = np.matrix(np.loadtxt(open('../2moons.txt', 'r')))

np.random.seed(1234)

train = data[:, :-1]
target = data[:, -1].getA1()

nnet_batch_k_10 = MLPBatch(d, m, dh, epsilon)
nnet_k_10 = MLP(d, m, dh, epsilon)

nnet_k_10.W1 = nnet_batch_k_10.W1.copy()
nnet_k_10.W2 = nnet_batch_k_10.W2.copy()
nnet_k_10.b1 = nnet_batch_k_10.b1.copy()
nnet_k_10.b2 = nnet_batch_k_10.b2.copy()

nnet_batch_k_10.train(train, target, lamdas, learning_rate, k, iterations=iterations)
nnet_batch_k_10.show_decision_regions(data, title='Question7_batch_k_10')

nnet_k_10.train(train, target, lamdas, learning_rate, k, iterations=iterations)
nnet_k_10.show_decision_regions(data, title='Question7_boucle_k_10')

print('======> K 10: \n')
print(nnet_batch_k_10.total_grad)
print(nnet_k_10.total_grad)

iterations=5000

nnet_batch_k_1 = MLPBatch(d, m, dh, epsilon)
nnet_k_1 = MLP(d, m, dh, epsilon)

nnet_k_1.W1 = nnet_batch_k_1.W1.copy()
nnet_k_1.W2 = nnet_batch_k_1.W2.copy()
nnet_k_1.b1 = nnet_batch_k_1.b1.copy()
nnet_k_1.b2 = nnet_batch_k_1.b2.copy()

nnet_batch_k_1.train(train, target, lamdas, learning_rate, 1, iterations=iterations)
nnet_batch_k_1.show_decision_regions(data, title='Question7_batch_k_1')

nnet_k_1.train(train, target, lamdas, learning_rate, 1, iterations=iterations)
nnet_k_1.show_decision_regions(data, title='Question7_boucle_k_1')

print('======> K 1: \n')
print(nnet_batch_k_1.total_grad)
print(nnet_k_1.total_grad)