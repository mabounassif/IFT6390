import numpy as np

import gzip,pickle
from devoirs.d2.code.mlp import MLP
from devoirs.d2.code.mlp_batch import MLPBatch

f=gzip.open('../mnist.pkl.gz')
data=pickle.load(f, encoding='latin')

"""data[0][0]: matrice de train data
data[0][1]: vecteur des train labels

data[1][0]: matrice de valid data
data[1][0]: vecteur des valid labels

data[2][0]: matrice de test data
data[2][0]: vecteur des test labels"""

train = np.matrix(data[0][0])
valid = np.matrix(data[1][0])
test = np.matrix(data[2][0])
train_target = np.matrix(data[0][1]).getA1()
valid_target = np.matrix(data[1][1]).getA1()
test_target = np.matrix(data[2][1]).getA1()

dh = train.shape[1]
d = train.shape[1]
m = 10
k = 100
epsilon = 1e-5

lamdas = np.matrix([[0.0001, 0.00001],[0.0001, 0.000006]])
learning_rate = 0.0019

iterations = 500

nnet = MLP(d, m, dh, epsilon, show_epoch=True)
nnet.train(train, train_target, lamdas, learning_rate , k, iterations=iterations)

nnet_batch = MLPBatch(d, m, dh, epsilon, show_epoch=True)
nnet_batch.train(train, train_target, lamdas, learning_rate , k, iterations=iterations)