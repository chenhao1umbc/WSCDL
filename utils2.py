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
        W is a matrix [C, K], where K is per-class atoms
    :return: D, D0, S, S0, W
    """
    if opts.transpose:
        N, T, F = X.shape
    else:
        N, F, T = X.shape
    D = torch.rand(opts.C, opts.K, opts.M, device=opts.dev)
    D = D/(D*D).sum().sqrt()  # normalization
    D0 = torch.rand(opts.K0, opts.M, device=opts.dev)
    D0 = D0/(D0*D0).sum().sqrt()  # normalization
    S = torch.zeros(N, opts.C, opts.K, T, device=opts.dev)
    S0 = torch.zeros(N, opts.K0, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K +1, device=opts.dev)

    return D, D0, S, S0, W









