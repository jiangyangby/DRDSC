from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.io as sio
from models import RSCConvAE
from utils import thrC, post_proC, err_rate, get_ar, get_fpr, get_nmi, get_purity


def train(iteration, X, y, CAE, lr, alpha, max_step):
    CAE.initlization()
    CAE.restore()  # restore from pre-trained model

    # fine-tune network
    # last_cost = 0
    for epoch in range(max_step):
        cost, Coef, z_diff, x_diff = CAE.partial_fit(X, lr)
        cost = cost / X.shape[0]
        if epoch % 5 == 0:
            print("epoch: %d" % epoch, "cost: %.8f" % cost)
        # last_cost = cost
        # if cost < 10 and abs(cost - last_cost) < last_cost * 1e-5:  # early stopping
        #     break

    Coef = thrC(Coef, alpha)
    d, a = 11, 10
    y_pred, _ = post_proC(Coef, y.max(), d, a)
    err, y_new = err_rate(y, y_pred)
    ar = get_ar(y, y_pred)
    nmi = get_nmi(y, y_pred)
    f, p, r = get_fpr(y, y_pred)
    purity = get_purity(y, y_pred)
    print('metrics: %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%' %
          (err * 100, ar * 100, nmi * 100, f * 100, p * 100, r * 100, purity * 100))

    return Coef


if __name__ == '__main__':
    data = sio.loadmat('./data/COIL20_gnoise0.4.mat')
    X = data['fea'].astype(float)
    y = data['gnd']
    X = np.reshape(X, (X.shape[0], 32, 32, 1))
    y = np.squeeze(y)

    n_input = [32, 32]
    kernel_size = [3]
    n_hidden = [15]

    save_path = './models/model-COIL20.ckpt'
    restore_path = './models/model-COIL20.ckpt'
    logs_path = './logs/'

    num_class = 20  # how many class we sample
    num_sa = 72
    batch_size = num_sa * num_class
    z_dim = 3840

    max_step = 34
    alpha = 0.04
    lr = 5.5e-4

    reg1 = 1.0
    reg2 = 150.0

    CAE = RSCConvAE(n_input=n_input, n_hidden=n_hidden, z_dim=z_dim, lamda1=reg1,
                    lamda2=reg2, eta1=10, eta2=10, kernel_size=kernel_size,
                    batch_size=batch_size, save_path=save_path,
                    restore_path=restore_path, logs_path=logs_path)

    train(0, X, y, CAE, lr, alpha, max_step)
