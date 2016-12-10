import numpy as np


def softmax(v):
    e_x = np.exp(v - np.amax(v, axis=0))
    return e_x / e_x.sum(axis=0)


def fprop(W1, W2, b1, b2, x, y=None):
    ha = np.dot(W1, np.transpose(x)) + b1
    hs = np.maximum(np.zeros(ha.shape), ha)

    oa = np.dot(W2, hs) + b2
    os = softmax(oa)

    if y is None:
        loss = None
    else:
        y = y.astype(int)
        loss = -1 * np.log(np.sum(np.multiply(os,np.matrix(np.eye(os.shape[0]))[:, y]), axis=0))

    return {
        'ha': ha,
        'hs': hs,
        'oa': oa,
        'os': os,
        'loss': loss
    }


def bprop(fprop, W1, W2, b1, b2, x, y, m):
    y = y.astype(int)
    grad_oa = fprop['os'] - np.matrix(np.eye(m))[:, y]

    grad_b2 = np.matrix(grad_oa)
    # devrait être grad_w2 = np.dot(grad_oa,np.transpose(fprop['hs']))
    # grad_w2 = np.dot(np.transpose(grad_oa), fprop['hs']) #original eq de Mahmoud
    grad_w2 = np.dot(grad_oa, np.transpose(fprop['hs']))
    # devrait être grad_hs = np.dot(np.transpose(W2),grad_oa) MAIS NE FONCTIONNE PAS
    grad_hs = np.dot(np.transpose(W2), grad_oa)  # original eq de Mahmoud
    # devrait être grad_ha = np.where(fprop['ha'])>=0,grad_hs,np.zeros(fprop['ha'].shape))
    # grad_ha = np.where(np.sign(fprop['ha'])+1,grad_hs,np.zeros(fprop['ha'].shape)) #original eq de Mahmoud
    grad_ha = np.where(fprop['ha'] >= 0, grad_hs, np.zeros(fprop['ha'].shape))
    grad_b1 = np.matrix(grad_ha)

    # devrait être grad_w1 = np.dot(grad_ha,np.transpose(x))
    grad_w1 = np.dot(grad_ha, x) #original eq de Mahmoud
    # grad_w1 = np.transpose(np.dot(np.transpose(x), grad_ha))  # doit ajouter np.transpose sur toute l'éq
    return {
        'grad_w1': grad_w1,
        'grad_b1': grad_b1,
        'grad_ha': grad_ha,
        'grad_hs': grad_hs,
        'grad_w2': grad_w2,
        'grad_b2': grad_b2,
        'grad_oa': grad_oa
    }