##
# """This file constains all the necessary classes and functions
#    for 2-d data/convolution
# """

import os
import sys
import pickle
import time
import datetime
import wave
import bisect
import pdb

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.io as sio
from sklearn import metrics
# import spams
# import scipy.sparse as sparse

tt = datetime.datetime.now
# torch.set_default_dtype(torch.double)
np.set_printoptions(linewidth=160)
torch.set_printoptions(linewidth=160)
torch.backends.cudnn.deterministic = True
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class OPT:
    """initial c the number of classes, k0 the size of shared dictionary atoms
    mu is the coeff of low-rank term,
    lamb is the coeff of sparsity
     nu is the coeff of cross-entropy loss
     """
    def __init__(self, C=4, K0=1, K=1, M=30, mu=0.1, eta=0.1, lamb=0.1, delta=0.9, maxiter=500, silent=False):
        self.C, self.K, self.K0, self.M = C, K, K0, M
        self.mu, self.eta, self.lamb, self.delta, self.lamb2 = mu, eta, lamb, delta, 0.01
        self.maxiter, self.plot, self.snr = maxiter, False, 20
        self.dataset, self.show_details, self.save_results = 0, True, True
        self.seed, self.n, self.shuffle, self.transpose = 0, 50, True, True  # n is number of examples per combination for toy data
        self.common_term = True*K0  # if common term exist
        self.shape = '1d' # input data is 1d or 2d, 1d could be vectorized 2d data
        if torch.cuda.is_available():
            self.dev = 'cuda'
            if not silent: print('\nRunning on GPU')
        else:
            self.dev = 'cpu'
            if not silent: print('\nRunning on CPU')


def load_data(opts, data='train'):
    """
    This function will load the preprocessed AASP dataset, train and val are in one set, test is the other dataset
    :param opts: only need teh cpu or gpu info
    :return: training, validation or testing data
    """
    route = '../../data/'
    if data == 'test':  # x, y are numpy double arrays
        # x, y = torch.load(route+'aasp_test_80by150.pt')
        mat = sio.loadmat(route+'test_256by200.mat')
        x, y = mat['rs'], mat['labels']
    else:
        # x, y = torch.load(route + 'aasp_train_80by150.pt')
        mat = sio.loadmat(route+'train_256by200.mat')
        x, y = mat['rs'], mat['labels']
    n, f, t = x.shape
    if opts.shuffle:
        nn = np.arange(x.shape[0])
        np.random.shuffle(nn)
        x, y = x[nn], y[nn]
    X = torch.from_numpy(x).float().to(opts.dev)
    Y = torch.from_numpy(y).float().to(opts.dev)

    # standardization
    X = (X - X.mean())/X.var().sqrt()

    if opts.transpose:  X = X.permute(0, 2, 1)

    indx = torch.arange(n)
    ind, ind2 = indx[indx%4 !=0], indx[indx%4 ==0]
    xtr, ytr = X[ind, :], Y[ind, :]
    xval, yval = X[::4, :], Y[::4, :]
    if data == 'train' : return xtr, ytr
    if data == 'val' : return xval, yval   # validation
    if data == 'test': return  X, Y  # testing


def init(X, opts):
    """
    This function will generate the initial value for D D0 S S0 and W
    :param X: training data with shape of [N, F,T]
    :param Y: training labels with shape of [N, C]
    :param opts: an object with hyper-parameters
        S is 4-d tensor [N,C,K,F,T] [samples,classes, num of atoms, Freq, time series,]
        D is 3-d tensor [C,K,F,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, F, T]pyth
        D0 is a matrix [K0, F, M]
        X is a matrix [N, F, T], training Data, could be in GPU
        Y is a matrix [N, C] \in {0,1}, training labels
        W is a matrix [C, K+1], where K is per-class atoms
    :return: D, D0, S, S0, W
    """
    if opts.transpose:
        N, T, F = X.shape
    else:
        N, F, T = X.shape
    D = torch.rand(opts.C, opts.K, opts.M,1, device=opts.dev)
    D = D/(D*D).sum().sqrt()  # normalization
    D0 = torch.rand(opts.K0, opts.M,1, device=opts.dev)
    D0 = D0/(D0*D0).sum().sqrt()  # normalization
    S = torch.zeros(N, opts.C, opts.K, F, T, device=opts.dev)
    S0 = torch.zeros(N, opts.K0, F, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K +1, device=opts.dev)
    return D, D0, S, S0, W


def train(D, D0, S, S0, W, X, Y, opts):
    """
    This function is the main training body of the algorithm, with showing a lot of details
    :param D: initial value, D, shape of [C, K, F, M]
    :param D0: pre-trained D0,  shape of [C0, K0, F, M]
    :param S: initial value, shape of [N,C,K, F, T]
    :param S0: initial value, shape of [N,K0,F,T]
    :param W: The pre-trained projection, shape of [C, K+1]
    :param X: testing data, shape of [N, F, T]
    :param Y: testing Lable, ground truth, shape of [N, C]
    :param opts: options of hyper-parameters
    :return: D, D0, S, S0, W, loss
    """
    loss, threshold = torch.tensor([], device=opts.dev), 5e-4
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('The initial loss function value is :%3.4e' % loss[-1])
    t, t1 = time.time(), time.time()
    S_numel, S0_numel = S.numel(), S0.numel()
    for i in range(opts.maxiter):
        t0 = time.time()
        S = updateS([D, D0, S, S0, W], X, Y, opts)
        if opts.show_details:
            loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
            print('pass S, time is %3.2f' % (time.time() - t)); t = time.time()
            print('loss function value is %3.4e:' %loss[-1])
            print('check sparsity, None-zero percentage is : %1.4f' % (1-(S==0).sum().item()/S_numel))

        S0 = updateS0([D, D0, S, S0], X, Y, opts) if opts.common_term else S0
        if opts.show_details:
            loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
            print('pass S0, time is %3.2f' % (time.time() - t)); t = time.time()
            print('loss function value is %3.4e:' %loss[-1])
            print('check sparsity, None-zero percentage is : %1.4f' % (1-(S0==0).sum().item()/S0_numel))

        D = updateD([D, D0, S, S0, W], X, Y, opts)
        if opts.show_details:
            loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
            print('pass D, time is %3.2f' % (time.time() - t)); t = time.time()
            print('loss function value is %3.4e:' %loss[-1])

        D0 = updateD0([D, D0, S, S0], X, Y, opts) if opts.common_term else D0
        if opts.show_details:
            loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
            print('pass D0, time is %3.2f' % (time.time() - t)); t = time.time()
            print('loss function value is %3.4e:' %loss[-1])

        W = updateW([S, W], Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        if opts.show_details:
            print('pass W, time is %3.2f' % (time.time() - t)); t = time.time()
            print('loss function value is %3.4e:' %loss[-1])

        if opts.show_details:
            if i > 3 and abs((loss[-1] - loss[-6]) / loss[-6]) < threshold: break
            print('In the %1.0f epoch, the training time is :%3.2f \n' % (i, time.time() - t0))
        else:
            if i > 3 and abs((loss[-1] - loss[-2]) / loss[-2]) < threshold: break
            if i%3 == 0 : print('In the %1.0f epoch, the training time is :%3.2f' % (i, time.time() - t0))

    print('After %1.0f epochs, the loss function value is %3.4e:' % (i, loss[-1]))
    print('All done, the total running time is :%3.2f \n' % (time.time() - t1))
    return D, D0, S, S0, W, loss


def loss_fun(X, Y, D, D0, S, S0, W, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, F, T]
    :param Y: the input label with shape of [N, C]
    :param D: the discriminative dictionary, [C,K,F,M]
    :param D0: the common dictionary, [K0,F,M]
    :param S: the sparse coefficients, shape of [N,C,K,F,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0,F,T]
    :param W: the projection for labels, shape of [C, K+1]
    :param opts: the hyper-parameters
    :return: cost, the value of loss function
    """
    N, K0, F, T = S0.shape
    M = D0.shape[-1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    C, K, _ = D.shape
    ycDcconvSc = S[:, :, 0, :, :].clone()  # initialization
    ycpDcconvSc = S[:, :, 0, :, :].clone()  # initialization
    Dcopy = D.clone().flip(2,3).unsqueeze(2)  # D shape is [C,K,1,F, M]
    DconvS = S[:, :, 0, :, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv2d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]  # shape of [N, C, T]
        ycpDcconvSc[:, c, :] = (1-Y[:, c].reshape(N, 1)) * DconvS[:, c, :]  # shape of [N, C, T]
        torch.cuda.empty_cache()
    R = F.conv2d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    _1_Y_hat = 1 - Y_hat
    fisher1 = torch.norm(X - R - DconvS.sum(1))**2
    fisher2 = torch.norm(X - R - ycDcconvSc.sum(1)) ** 2
    fisher = fisher1 + fisher2 + torch.norm(ycpDcconvSc.sum(1)) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum() * opts.eta
    # label = -1 * opts.eta * (Y * (Y_hat + 3e-38).log() + (1 - Y) * (_1_Y_hat + 3e-38).log()).sum()
    low_rank = N * opts.mu * D0.norm(p='nuc')
    cost = fisher + sparse + label + low_rank
    return cost



