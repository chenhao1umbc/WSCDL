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
    if opts.transpose:  # true means stacking over the column
        X = X.permute(0, 2, 1).reshape(X.shape[0], -1)  # learn atom of over time
    indx = torch.arange(n)
    ind, ind2 = indx[indx%4 !=0], indx[indx%4 ==0]
    xtr, ytr = l2norm(X[ind, :].reshape(ind.shape[0], -1)).reshape(ind.shape[0], f, t), Y[ind, :]
    xval, yval = l2norm(X[::4, :].reshape(ind2.shape[0], -1)).reshape(ind2.shape[0], f, t), Y[::4, :]
    if data == 'train' : return xtr, ytr
    if data == 'val' : return xval, yval   # validation
    if data == 'test': return  l2norm(X), Y  # testing












