import numpy as np
from devoirs.d2.code.mlp import MLP


dh = 300
d = 2
m = 2
k = 1
epsilon = 1e-5

lamdas = np.matrix([[0.001, 0.0001],[0.0001, 0.00006]])
learning_rate = 0.26

data = np.matrix(np.loadtxt(open('../2moons.txt', 'r')))

np.random.seed(1234)

train = data[:, :-1]
target = data[:, -1].getA1()


def model_fit(d, m, dh, epsilon, train, target, lamdas, learning_rate, data, iterations=None):
    nnet = MLP(d, m, dh, epsilon)
    nnet.train(train, target, lamdas, learning_rate, iterations=iterations)
    nnet.show_decision_regions(data, title='Question5_iterations{9}-d-{0}_m-{1}_dh-{2}_epsilon-{3}_learning_rate-{4}_lamdas-{5}-{6}-{7}-{8}'.format(d,m,dh,epsilon,learning_rate,lamdas[0,0], lamdas[0,1], lamdas[1,0], lamdas[1,1], iterations).replace('.', '~'))

model_fit(d, m, dh, epsilon, train, target, lamdas, learning_rate, data)
model_fit(d, m, dh, epsilon, train, target, lamdas, learning_rate, data, iterations=50)

# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.0001, 0.0001],[0.0001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.0001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.01, 0.0001],[0.0001, 0.00006]]), learning_rate, data)
#
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.001],[0.0001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.0001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.00001],[0.0001, 0.00006]]), learning_rate, data)
#
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.0001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.00001, 0.00006]]), learning_rate, data)
#
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.0001, 0.000006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.0001, 0.00006]]), learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, np.matrix([[0.001, 0.0001],[0.0001, 0.00001]]), learning_rate, data)

# model_fit(d, m, 200, epsilon, train, target, lamdas, learning_rate, data)
# model_fit(d, m, 140, epsilon, train, target, lamdas, learning_rate, data)
# model_fit(d, m, 500, epsilon, train, target, lamdas, learning_rate, data)
# model_fit(d, m, 1000, epsilon, train, target, lamdas, learning_rate, data)



