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
import torch.nn.functional as Func
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import metrics

import scipy.signal as sg
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
    def __init__(self, C=4, K0=1, K=1, Dh=256, Dw=3,\
                 mu=0.1, eta=0.1, lamb=0.1, delta=0.9, maxiter=500, silent=False):
        self.C, self.K, self.K0, self.Dh, self.Dw = C, K, K0, Dh, Dw
        self.mu, self.eta, self.lamb, self.delta, self.lamb2 = mu, eta, lamb, delta, 0.0
        self.lamb0, self.init = 0.1, 0  # seperate lamb for the common term
        self.maxiter, self.plot, self.snr = maxiter, False, 20
        self.dataset, self.show_details, self.save_results = 0, True, True
        self.seed, self.n, self.shuffle, self.transpose = 0, 50, False, False  # n is number of examples per combination for toy data
        self.common_term = True*K0  # if common term exist
        self.shape, self.batch_size = '1d', 100 # input data is 1d or 2d, 1d could be vectorized 2d data
        if torch.cuda.is_available():
            self.dev = 'cuda'
            if not silent: print('\nRunning on GPU')
        else:
            self.dev = 'cpu'
            if not silent: print('\nRunning on CPU')


def load_data(opts, data='train', fold=0):
    """
    This function will load the preprocessed AASP dataset, train and val are in one set, test is the other dataset
    :param opts: only need teh cpu or gpu info
    :return: training, validation or testing data
    """
    route = '../data/'
    if data == 'test':  # x, y are numpy double arrays
        # x, y = torch.load(route+'aasp_test_80by150.pt')
        mat = sio.loadmat(route+'test_256by200.mat')
        x, y = mat['rs'], mat['labels']

    if data =='train' or data == 'val':
        # x, y = torch.load(route + 'aasp_train_80by150.pt')
        mat = sio.loadmat(route+'train_256by200.mat')
        x, y = mat['rs'], mat['labels']

    "This part will make all the mixture data into one pool"
    if data == 'mix_train' or data == 'mix_val' or data == 'mix_test' :
        mat = sio.loadmat(route + 'test_256by200.mat')
        x, y = mat['rs'], mat['labels']
        mat = sio.loadmat(route + 'train_256by200.mat')
        xx, yy = mat['rs'], mat['labels']
        x = np.concatenate((x, xx))
        y = np.concatenate((y, yy))

    n, f, t = x.shape
    if opts.shuffle:
        np.random.seed(opts.seed)
        nn = np.arange(x.shape[0])
        np.random.shuffle(nn)
        x, y = x[nn], y[nn]
    X = torch.from_numpy(x).float().to(opts.dev)  # to GPU or CPU
    Y = torch.from_numpy(y).float().to(opts.dev)  # to GPU or CPU

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

    _08N = int(0.8*n)
    xtr, ytr, xval, yval= dataloader(X[:_08N], Y[:_08N], fold)
    if data == 'mix_train': return xtr, ytr
    if data == 'mix_val': return xval, yval
    if data == 'mix_test': return X[_08N:], Y[_08N:]


def dataloader(X, Y, fold):
    """
    This function will shuffle the data with given random seed
    :param X: 882 sample, 80% for training, 20% for validation
    :param Y: labels
    :param fold: random seed number
    :return: xtr, ytr, xval, yval
    """
    N = X.shape[0]
    _08N = int(0.8*N)
    np.random.seed(fold)
    nn = np.arange(N)
    np.random.shuffle(nn)
    X, Y = X[nn], Y[nn]
    xtr, ytr = X[:_08N], Y[:_08N]
    xval, yval = X[_08N:], Y[_08N:]
    return xtr, ytr, xval, yval



def init(X, opts, init='good'):
    """
    This function will generate the initial value for D D0 S S0 and W
    :param X: training data with shape of [N, F,T]
    :param Y: training labels with shape of [N, C]
    :param opts: an object with hyper-parameters
        S is 5-d tensor [N,C,K,F,T] [samples,classes, num of atoms, Freq, time series,]
        D is 4-d tensor [C,K,F,M] [num of atoms, classes, atom size]
        S0 is 4-d tensor [N, K0, F, T]
        D0 is 3-d tensor [K0, F, M]
        X is 3-d tensor [N, F, T], training Data, could be in GPU
        init is a indicator using different initializations
        W is a matrix [C, K+1], where K is per-class atoms
    :return: D, D0, S, S0, Wsudo apt install firefox
    """
    if opts.transpose:
        N, T, F = X.shape
    else:
        N, F, T = X.shape
    D = torch.rand(opts.C, opts.K, opts.Dh,opts.Dw, device=opts.dev)
    D0 = torch.rand(opts.K0, opts.Dh, opts.Dw, device=opts.dev)
    if init == 'good' :
        "d is the shape of [16,1,256, 15]"
        d = torch.load('good_intialization.pt').to(opts.dev)
        for i in range(opts.K): D[:,i] = d[:,0]
        for i in range(opts.K0): D0[i] = X[0][:,:opts.Dw]
    D, D0 = D/D.norm(), D0/D0.norm() # D0.norm() = (D0*D0).sum().sqrt()

    S = torch.zeros(N, opts.C, opts.K, 1, T, device=opts.dev)
    S0 = torch.zeros(N, opts.K0, 1, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K +1, device=opts.dev)
    return D, D0, S, S0, W


def train(X, Y, opts):
    """
    This function is the main training body of the algorithm, with showing a lot of details
    :param D: initial value, D, shape of [C, K, Dh, Dw]
    :param D0: pre-trained D0,  shape of [C0, K0, Dh, Dw]
    :param S: initial value, shape of [N,C,K, 1, T]
    :param S0: initial value, shape of [N,K0,1,T]
    :param W: The pre-trained projection, shape of [C, K+1]
    :param X: testing data, shape of [N, F, T]
    :param Y: testing Lable, ground truth, shape of [N, C]
    :param opts: options of hyper-parameters
    :return: D, D0, S, S0, W, loss
    """
    loss, threshold, opts.offset = torch.tensor([], device=opts.dev), 5e-4, (opts.Dw-1)//2
    "if batch_size == -1, it means using all data"
    batch_size = opts.batch_size if opts.batch_size > 0 else X.shape[0]
    t, t1 = time.time(), time.time()
    for i in range(opts.maxiter):
        for indx in range(X.shape[0] // batch_size + 1):
            x, y = X[batch_size * indx:batch_size * (indx + 1)], \
                   Y[batch_size * indx:batch_size * (indx + 1)]
            if x.nelement() == 0: continue  # opts.batch_size==N, x is null
            if i == 0 and indx == 0:
                D, D0, S, S0, W = init(x, opts, opts.init)
                S_numel, S0_numel = S.numel(), S0.numel()
                if opts.init == 'good':
                    print('good intialization')
            if i > 0 and S.shape[0] != x.shape[0]:  # if the last batch size is small
                _, _, S, S0, _ = init(x, opts, opts.init)
                S_numel, S0_numel = S.numel(), S0.numel()

            loss = torch.cat((loss, loss_fun(x, y, D, D0, S, S0, W, opts).reshape(1)))
            if i == 0 and indx ==0: print('The initial loss function value is :%3.4e' % loss[-1])

            t0 = time.time()
            S = updateS([D, D0, S, S0, W], x, y, opts)
            if opts.show_details:
                loss = torch.cat((loss, loss_fun(x, y, D, D0, S, S0, W, opts).reshape(1)))
                print('pass S, time is %3.2f' % (time.time() - t)); t = time.time()
                print('loss function value is %3.4e:' %loss[-1])
                print('check sparsity, None-zero percentage is : %1.4f' % (1-(S==0).sum().item()/S_numel))

            S0 = updateS0([D, D0, S, S0], x, y, opts) if opts.common_term else S0
            if opts.show_details:
                loss = torch.cat((loss, loss_fun(x, y, D, D0, S, S0, W, opts).reshape(1)))
                print('pass S0, time is %3.2f' % (time.time() - t)); t = time.time()
                print('loss function value is %3.4e:' %loss[-1])
                print('check sparsity, None-zero percentage is : %1.4f' % (1-(S0==0).sum().item()/S0_numel))

            D = updateD([D, D0, S, S0, W], x, y, opts)
            if opts.show_details:
                loss = torch.cat((loss, loss_fun(x, y, D, D0, S, S0, W, opts).reshape(1)))
                print('pass D, time is %3.2f' % (time.time() - t)); t = time.time()
                print('loss function value is %3.4e:' %loss[-1])

            D0 = updateD0([D, D0, S, S0], x, y, opts) if opts.common_term else D0
            if opts.show_details:
                loss = torch.cat((loss, loss_fun(x, y, D, D0, S, S0, W, opts).reshape(1)))
                print('pass D0, time is %3.2f' % (time.time() - t)); t = time.time()
                print('loss function value is %3.4e:' %loss[-1])

            W = updateW([S, W], y, opts)
            loss = torch.cat((loss, loss_fun(x, y, D, D0, S, S0, W, opts).reshape(1)))
            if opts.show_details:
                print('pass W, time is %3.2f' % (time.time() - t)); t = time.time()
                print('loss function value is %3.4e:' %loss[-1])
            torch.cuda.empty_cache()

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
    :param D: the discriminative dictionary, [C,K,Dh,Dw]
    :param D0: the common dictionary, [K0,Dh,Dw]
    :param S: the sparse coefficients, shape of [N,C,K,1,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0,1,T]
    :param W: the projection for labels, shape of [C, K+1]
    :param opts: the hyper-parameters
    :return: cost, the value of loss function
    """
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K*C, N*C

    # "DconvS should be the shape of (N, CK, F,T)  not fast on GPU"
    # DconvS = Func.conv2d(S.reshape(N, CK, 1, T) ,D.reshape(CK, 1, Dh, Dw).flip(2,3), padding=(255,1), groups=CK)
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)  # faster on GPU
    Dr = D.reshape(CK, 1, Dh, Dw)
    for ck in range(CK):
        DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck],Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
    DconvS = DconvS0.reshape(N, C, K, F, T)   # shape of [N,C,K,F,T]

    ycDcconvSc = (Y.reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N,CK,F,T).sum(1)  # output shape of (N, F, T)
    ycpDcconvSc =((1-Y).reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N,CK,F,T).sum(1)  # output shape of (N, F, T)
    DconvS = DconvS0.sum(1)  # using the same name to save memory
    "shape of (N, F, T), R is the common recon."
    # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2,3),  padding=(255,opts.offset), groups=K0).sum(1)
    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for kk0 in range(K0):
        R0[:,kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
    R = R0.sum(1)
    torch.cuda.empty_cache()

    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    S_tik = torch.cat((S.squeeze(3).mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    _1_Y_hat = 1 - Y_hat
    fidelity1 = torch.norm(X - R - DconvS)**2
    fidelity2 = torch.norm(X - R - ycDcconvSc) ** 2
    fidelity = fidelity1 + fidelity2 + torch.norm(ycpDcconvSc) ** 2
    sparse = opts.lamb * S.abs().sum() + opts.lamb0*S0.abs().sum()
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum() * opts.eta
    # label = -1 * opts.eta * (Y * (Y_hat + 3e-38).log() + (1 - Y) * (_1_Y_hat + 3e-38).log()).sum()
    low_rank = N * opts.mu * D0.reshape(-1, K0).norm(p='nuc')
    cost = fidelity + sparse + label + low_rank
    return cost


def loss_fun_special(X, Y, D, D0, S, S0, W, opts):
    """
     This function will calculate the cost function by each term value
     :param X: the input data with shape of [N, F, T]
     :param Y: the input label with shape of [N, C]
     :param D: the discriminative dictionary, [C,K,Dh,Dw]
     :param D0: the common dictionary, [K0,Dh,Dw]
     :param S: the sparse coefficients, shape of [N,C,K,1,T] [samples, classes, num of atoms, time series,]
     :param S0: the common coefficients, 3-d tensor [N, K0,1,T]
     :param W: the projection for labels, shape of [C, K+1]
     :param opts: the hyper-parameters
     :return: cost, the value of loss function
     """
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C

    # DconvS should be the shape of (N, CK, F,T)
    # DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3), padding=(255, opts.offset), groups=CK)
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev) # faster on GPU
    Dr = D.reshape(CK, 1, Dh, Dw)
    for ck in range(CK):
        DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck],Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
    DconvS = DconvS0.reshape(N, C, K, F, T)   # shape of [N,CK,F,T]

    ycDcconvSc = (Y.reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N, CK, F, T).sum(1)  # output shape of (N, F, T)
    ycpDcconvSc = ((1 - Y).reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N, CK, F, T).sum(1)  # output shape of (N, F, T)
    DconvS = DconvS0.sum(1)  # using the same name to save memory
    "shape of (N, F, T), R is the common recon."
    # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3), padding=(255, opts.offset), groups=K0).sum(1)
    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for kk0 in range(K0):
        R0[:,kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
    R = R0.sum(1)
    torch.cuda.empty_cache()

    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    S_tik = torch.cat((S.squeeze(3).mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    _1_Y_hat = 1 - Y_hat
    fidelity1 = torch.norm(X - R - DconvS) ** 2
    fidelity2 = torch.norm(X - R - ycDcconvSc) ** 2
    fidelity = fidelity1 + fidelity2 + torch.norm(ycpDcconvSc) ** 2
    sparse = opts.lamb * S.abs().sum() + opts.lamb0 * S0.abs().sum()
    label = (-1 * (1 - Y) * (exp_PtSnW + 1e-38).log() + (exp_PtSnW + 1).log()).sum() * opts.eta
    # label = -1 * opts.eta * (Y * (Y_hat + 3e-38).log() + (1 - Y) * (_1_Y_hat + 3e-38).log()).sum()
    low_rank = N * opts.mu * D0.reshape(-1, K0).norm(p='nuc')

    return fidelity.item(), sparse.item(), label.item(), low_rank.item()


def save_results(D, D0, S, S0, W, opts, loss):
    """
    This function will save the training results
    :param D: initial value, D, shape of [C, K, M]
    :param D0: pre-trained D0,  shape of [C0, K0, M]
    :param S: initial value, shape of [N,C,K,T]
    :param S0: initial value, shape of [N,K0,T]
    :param W: The pre-trained projection, shape of [C, K]
    """
    param = str([opts.K, opts.K0, opts.Dw, opts.lamb, opts.eta , opts.mu])
    torch.save([D, D0, S, S0, W, opts, loss], '../'+param+'DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')


def updateS(DD0SS0W, X, Y, opts):
    """this function is to update the sparse coefficients for dictionary D using BPG-M, updating each S_n,k^(c)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,Dh,Dw] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, Dh, Dw]
        W is a matrix [C, K+1], where K is per-class atoms
        X is a matrix [N, F, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0, W = DD0SS0W  # where DD0SS0 is a list
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C
    "shape of (N, F, T), R is the common recon."
    # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2,3),  padding=(255,opts.offset), groups=K0).sum(1)
    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for kk0 in range(K0):
        R0[:,kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
    R = R0.sum(1)

    Crange = torch.tensor(range(C))
    NC_1, FT = N * (C - 1), F*T
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)

    # '''update the current s_n,k^(c) '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        # "DconvS is the shape of (N,CK, F, T) to (N, C, F, T) ----> there is bug in cuda version, it is too slow"
        # DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
        #                      padding=(255, 1), groups=CK).reshape(N,C,K, F, T).sum(2)
        for ck in range(CK):
            DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
        DconvS = DconvS0.reshape(N, C, K, F, T).sum(2)

        dck = D[c, k, :]  # shape of [Dh, Dw]
        sck = S[:, c, k, :]  # shape of [N, 1, T]
        wc = W[c, :]  # shape of [K+1], including bias
        yc = Y[:, c]  # shape of [N]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes

        Tdck = toeplitz_dck(dck, [Dh, Dw, T])  # shape of [FT, T]
        # dck_conv_sck = Func.conv2d(sck.unsqueeze(1), dck.reshape(1,1,Dh,Dw).flip(2, 3), padding=(255, opts.offset)).squeeze()  # shape of [N,F,T]
        dck_conv_sck = (Tdck @ sck.permute(0, 2, 1)).reshape(N, F, T)
        Dcp_conv_Sncp = DconvS[:, c, :] - dck_conv_sck  # shape of [N, F, T]

        # term 1, 2, 3 should be in the shape of [N, F, T] or [N, F*T]
        term1 = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        term2 = Y[:, c].reshape(N, 1) * (X.reshape(N, -1) - R.reshape(N, -1) - Y[:, c].reshape(N, 1) * Dcp_conv_Sncp.reshape(N, -1) -
                        (Y[:, c_prime].reshape(NC_1, 1) * DconvS[:, c_prime, :].reshape(NC_1, -1)).reshape(N,C-1,-1).sum(1))
        term3 = -(1 - Y[:, c]).reshape(N, 1) * ((1 - Y[:, c]).reshape(N, 1) * Dcp_conv_Sncp.reshape(N,-1) +
                        ((1 - Y[:, c_prime].reshape(NC_1, 1)) * DconvS[:, c_prime, :].reshape(NC_1, -1)).reshape(N,C-1,-1).sum(1))
        b = (term1.reshape(N, FT) + term2 + term3) / 2  # shape of [N, F*T]
        torch.cuda.empty_cache()

        # l00 = loss_fun(X, Y, D, D0, S, S0, W, opts)
        # l0 = loss_fun_special(X, Y, D, D0, S, S0, W, opts)
        # l1 = loss_Sck_special(Tdck, b, S[:, c, :].squeeze(), sck.squeeze(), wc, wc[k], yc, opts)
        S[:, c, k, :] = solv_sck(S[:, c, :].squeeze(2), wc, yc, Tdck, b, k, opts)
        # ll0 = loss_fun_special(X, Y, D, D0, S, S0, W, opts)
        # ll1 = loss_Sck_special(Tdck, b, S[:, c, :].squeeze(), sck.squeeze(), wc, wc[k], yc, opts)
        # print('Overall loss for fidelity, sparse, label, differences: %1.7f, %1.7f, %1.7f' %(l0[0]-ll0[0], l0[1]-ll0[1], l0[2]-ll0[2]))
        # print('Local loss for fidelity, sparse, label, differences: %1.7f, %1.7f, %1.7f' % (l1[0]-ll1[0], l1[1]-ll1[1], l1[2]-ll1[2]))
        # print('Main loss after bpgm the diff is: %1.9e' %(l00 - loss_fun(X, Y, D, D0, S, S0, W, opts)))
        # if (l00 - loss_fun(X, Y, D, D0, S, S0, W, opts)) <0 : print(bug)
        # if torch.isnan(S).sum() + torch.isinf(S).sum() > 0: print('inf_nan_happenned')
    return S


def toeplitz_dck(dck, Dh_Dw_T):
    """This is a the toepliz matrx for 2d-convolution for make dck into toeplitz matrix
    input dck  shape of [Dh, Dw], is to be made in toeplitz format
            Dh_Dw_T is the shape of atoms and sparse vector
    output tx has the shape of [Dh*T, T] (after truncation)
    """
    dev = dck.device
    Dh, Dw, T = Dh_Dw_T
    offset = (Dw - 1) // 2
    ind1, ind2 = T+Dw-1-offset, 2*T+Dw-1-offset
    'dck_padded0 is the shape of [Dh, 2*T+Dw]'
    dck_padded0 = torch.cat([torch.zeros(Dh, T, device=dev), dck.flip(1), torch.zeros(Dh, T, device=dev)], dim=1)
    indx = torch.zeros(T, T).long()
    for i in range(T):
        indx[i, :] = torch.arange(ind1-i, ind2-i)
    indx = indx.repeat(Dh,1)  #shape of [Dh*T, T]
    ind0 = torch.arange(0, Dh*dck_padded0.shape[1], dck_padded0.shape[1]).repeat(200,1).t().reshape(Dh*T,1)
    indx = indx + ind0
    tx = dck_padded0.view(-1)[indx]
    return tx


def solv_sck(sc, wc, yc, Tdck, b, k, opts):
    """
    This function solves snck for all N, using BPGM
    :param sc: shape of [N, K, T]
    :param wc: shape of [K+1], with bias
    :param yc: shape of [N]
    :param Tdck: shape of [FT, T]
    :param b: shape of [N, FT]
    :param k: integer, which atom to update
    :return: sck
    """
    # for the synthetic data correction = 0.7
    maxiter, correction, threshold = 500, 1, 1e-4 # correction is help to make the loss monotonically decreasing
    Mw = opts.delta
    lamb = opts.lamb
    dev = opts.dev
    T = sc.shape[2]
    P = torch.ones(1, T, device=dev)/T  # shape of [1, T]
    # 'skc update will lead sc change'
    sck = sc[:, k, :].clone()  # shape of [N, T]
    sck_old = sck.clone()
    wkc = wc[k]  # scaler
    Tdck_t_Tdck = Tdck.t() @ Tdck  # shape of [T, T]
    abs_Tdck = abs(Tdck)
    eta_wkc_square = opts.eta * wkc**2  # scaler
    _4_Tdckt_bt = 4*Tdck.t() @ b.t()  # shape of [T, N]
    term0 = (yc-1).unsqueeze(1) @ P * wkc * opts.eta  # shape of [N, T]
    term1 = (4 * abs_Tdck.t()@abs_Tdck).sum(1)  # shape of [T]
    M = (term1 + P*eta_wkc_square + 1e-38).squeeze() # M is the diagonal of majorization matrix, shape of [T]
    sc_til = sc.clone()  # shape of [N, K, T]
    marker = 0

    # loss = torch.cat((torch.tensor([], device=opts.dev), loss_Sck(Tdck, b, sc, sck, wc, wkc, yc, opts).reshape(1)))
    for i in range(maxiter):
        sck_til = sck + correction * Mw * (sck - sck_old)  # shape of [N, T]
        sc_til[:, k, :] = sck_til
        exp_PtSnc_tilWc = (sc_til.mean(2) @ wc[:-1] + wc[-1]).exp()  # exp_PtSnc_tilWc should change due to sck_til changing
        exp_PtSnc_tilWc[torch.isinf(exp_PtSnc_tilWc)] = 1e38
        term = term0 + (exp_PtSnc_tilWc / (1 + exp_PtSnc_tilWc)*opts.eta*wkc ).unsqueeze(1) @ P
        nu = sck_til - (4*Tdck_t_Tdck@sck_til.t() - _4_Tdckt_bt + term.t()).t()/M  # shape of [N, T]
        sck_new = shrink(M, nu, lamb)  # shape of [N, T]
        sck_old[:], sck[:] = sck[:], sck_new[:]  # make sure sc is updated in each loop
        if exp_PtSnc_tilWc[exp_PtSnc_tilWc == 1e38].shape[0] > 0: marker = 1
        if torch.norm(sck - sck_old) / (sck.norm() + 1e-38) < threshold: break
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_Sck(Tdck, b, sc, sck, wc, wkc, yc, opts).reshape(1)))

    # print('M max', M.max())
    # if marker == 1 :
    #     print('--inf to 1e38 happend within the loop')
    #     plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    #     print('How many inf to 1e38 happend finally', exp_PtSnc_tilWc[exp_PtSnc_tilWc == 1e38].shape[0])
    # if (loss[0] - loss[-1]) < 0 :
    #     wait = input("Loss Increases, PRESS ENTER TO CONTINUE.")
    # print('sck loss after bpgm the diff is :%1.9e' %(loss[0] - loss[-1]))
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    # wait = input(" PRESS ENTER TO CONTINUE.")
    return sck_old.unsqueeze(-2)


def loss_Sck(Tdck, b, sc, sck, wc, wkc, yc, opts):
    """
    This function calculates the loss func of sck
    :param Tdck: shape of [FT, T]
    :param b: shape of [N, FT]
    :param sc: shape of [N, K, T]
    :param sck: shape of [N, T]
    :param wc: shape of [K+1]
    :param wkc: a scaler
    :param yc: shape [N]
    :param opts: for hyper parameters
    :return:
    """
    epx_PtScWc = (sc.mean(2) @ wc[:-1] + wc[-1]).exp()  # shape of N
    epx_PtScWc[torch.isinf(epx_PtScWc)] = 1e38
    epx_PtSckWck = (sck.mean(1) * wkc).exp()
    epx_PtSckWck[torch.isinf(epx_PtSckWck)] = 1e38
    y_hat = 1 / (1 + epx_PtScWc)
    _1_y_hat = 1- y_hat
    # g_sck_wc = (-(1-yc)*((epx_PtSckWck+1e-38).log()) + (1+epx_PtScWc).log()).sum()
    g_sck_wc = -((yc * (y_hat+ 1e-38).log()) + (1 - yc) * (1e-38 + _1_y_hat).log()).sum()
    # print(g_sck_wc.item()))
    fidelity = 2*(Tdck@sck.t() - b.t()).norm()**2
    sparse = opts.lamb * sck.abs().sum()
    label = opts.eta * g_sck_wc
    loss = fidelity + sparse + label
    if label < 0 or torch.isnan(label).sum() > 0: print(stop)
    return loss


def loss_Sck_special(Tdck, b, sc, sck, wc, wkc, yc, opts):
    """
    This function calculates each term in the loss func of sck
    :param Tdck: shape of [FT, T]
    :param b: shape of [N, FT]
    :param sc: shape of [N, K, T]
    :param sck: shape of [N, T]
    :param wc: shape of [K+1]
    :param wkc: a scaler
    :param yc: shape [N]
    :param opts: for hyper parameters
    :return:
    """
    epx_PtScWc = (sc.mean(2) @ wc[:-1] + wc[-1]).exp()  # shape of N
    epx_PtScWc[torch.isinf(epx_PtScWc)] = 1e38
    epx_PtSckWck = (sck.mean(1) * wkc).exp()
    epx_PtSckWck[torch.isinf(epx_PtSckWck)] = 1e38
    y_hat = 1 / (1 + epx_PtScWc)
    _1_y_hat = 1- y_hat
    # g_sck_wc = (-(1-yc)*((epx_PtSckWck+1e-38).log()) + (1+epx_PtScWc).log()).sum()
    g_sck_wc = -((yc * (y_hat+ 1e-38).log()) + (1 - yc) * (1e-38 + _1_y_hat).log()).sum()
    # print(g_sck_wc.item())
    fidelity = 2*(Tdck@sck.t() - b.t()).norm()**2
    sparse = opts.lamb * sck.abs().sum()
    label = opts.eta * g_sck_wc
    if label <0 or torch.isnan(label).sum()>0 :print(stop)
    return fidelity.item(), sparse.item(), label.item()


def shrink(M, nu, lamb):
    """
    This function is to implement the shrinkage operator, solving the following
    min_p lamb||p||_1 + 1/2||\nu-p||_M^2, p is a vector
    :param M: is used for matrix norm, shape of [T]
    :param lamb: is coefficient of L-1 norm, scaler
    :param nu: the the matrix to be pruned, shape of [N, T]
    :return: P the matrix after thresholding
    """
    b = lamb / M  # shape of [T]
    P = torch.sign(nu) * Func.relu(abs(nu) -b)
    return P


def updateS0(DD0SS0, X, Y, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,Dh,Dw] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, Dh, Dw]
        D0 is a matrix [K0, M]
        W is a matrix [C, K+1], where K is per-class atoms
        X is a matrix [N, F, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C
    Crange = torch.tensor(range(C))
    NC_1, FT = N * (C - 1), F*T

    # "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
    # DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
    #                      padding=(255, 1), groups=CK).reshape(N, C, K, F, T).sum(2)
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)
    for ck in range(CK):
        DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
    DconvS_NFT = DconvS0.sum(1)
    ycDcconvSc = (Y.reshape(NC, 1) * DconvS0.reshape(NC, -1)).reshape(N, CK, F, T).sum(1)  # output shape of (N, F, T)

    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, 1, T]
        "shape of (N, F, T), R is the common recon."
        # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3), padding=(255, opts.offset), groups=K0).sum(1)
        for kk0 in range(K0):
            R0[:, kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
        R = R0.sum(1)
        Tdk0 = toeplitz_dck(dk0, [Dh, Dw, T])  # shape of [FT, T]

        # dk0convsnk0 = Func.conv2d(snk0.unsqueeze(1), dk0.reshape(1,1,Dh,Dw).flip(2, 3), padding=(255,1)).squeeze()
        dk0convsnk0 = (Tdk0 @ snk0.permute(0,2,1)).reshape(N, F, T)
        abs_Tdk0 = abs(Tdk0)
        MS0_diag = (4*abs_Tdk0.t() @ abs_Tdk0).sum(1)  # in the shape of [T]
        MS0_diag = MS0_diag + 1e-38 # make it robust for inverse
        MS0_inv = (1/MS0_diag).diag()
        b = (2*X - DconvS_NFT - ycDcconvSc - 2*R + 2*dk0convsnk0).reshape(N, FT)

        torch.cuda.empty_cache()
        # print(loss_S0(Tdk0, snk0.squeeze(), b, opts.lamb))
        S0[:, k0, :] = solv_snk0(snk0.squeeze(), MS0_diag, MS0_inv, opts.delta, 2*Tdk0, b, opts.lamb0)
        # print(loss_S0(Tdk0, S0[:, k0, :].squeeze(), b, opts.lamb))
    return S0


def solv_snk0(x, M, Minv, Mw, Tdk0, b, lamb):
    """
    :param x: is the snk0, shape of [N, T]
    :param M: is MD, the majorizer matrix with shape of [T], diagonal of matrix
    :param Minv: is MD^(-1)
    :param Mw: is a number, not diagonal matrix
    :param Tdk0: is truncated toeplitz matrix of dk0 with shape of [M, T], already *2
    :param b: bn with all N, with shape of [N, F*T]
    :param lamb: sparsity hyper parameter
    :return: dck0
    """
    # for the synthetic data correction = 0.7
    maxiter, correction,  threshold = 500, 1, 1e-4
    snk0_old, snk0 = x.clone(), x.clone()
    coef = Minv @ Tdk0.t() @ Tdk0  # shape of [T, T]
    term = (Minv @ Tdk0.t() @b.t()).t()  # shape of [N, T]

    # loss = torch.cat((torch.tensor([], device=x.device), loss_S0(Tdk0, snk0, b, lamb).reshape(1)))
    for i in range(maxiter):
        snk0_til = snk0 + correction*Mw*(snk0 - snk0_old)  # Mw is just a number for calc purpose
        nu = snk0_til - (coef@snk0_til.t()).t() + term  # nu is [N, T]
        snk0_new = shrink(M, nu, lamb)  # shape of [N, T]
        snk0, snk0_old = snk0_new, snk0
        if torch.norm(snk0 - snk0_old)/(snk0_old.norm() +1e-38) < threshold: break
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_S0(Tdk0, snk0, b, lamb).reshape(1)))
    # plt.figure();plt.plot(loss.cpu().numpy(), '-x')
    # ll = loss[:-1] - loss[1:]
    # if ll[ll<0].shape[0] > 0: print(something_wrong)
    return snk0.unsqueeze(1)


def loss_S0(_2Tdk0, snk0, b, lamb):
    """
    This function calculates the sub-loss function for S0
    :param _2Tdk0: shape of [FT, T]
    :param snk0: shape of [N, T]
    :param b: shape of [N, FT]
    :param lamb: scaler
    :return: loss
    """
    return ((_2Tdk0 @ snk0.t() - b.t())**2).sum()/2 + lamb * abs(snk0).sum()


def updateD(DD0SS0W, X, Y, opts):
    """this function is to update the distinctive D using BPG-M, updating each d_k^(c)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,1,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,Dh,Dw] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, 1, T]
        D0 is a matrix [K0, Dh, Dw]
        X is a matrix [N, F, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0, W = DD0SS0W  # where DD0SS0 is a list
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C
    # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2,3),padding=(255,opts.offset), groups=K0).sum(1)  #shape of (N, F, T)
    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for kk0 in range(K0):
        R0[:, kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
    R = R0.sum(1)

    Crange = torch.tensor(range(C))
    NC_1, FT = N * (C - 1), F*T
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        # "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
        # DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
        #                      padding=(255, 1), groups=CK).reshape(N, C, K, F, T).sum(2)
        for ck in range(CK):
            DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2), padding=opts.offset).squeeze()
        DconvS = DconvS0.reshape(N, C, K, F, T).sum(2)

        dck = D[c, k, :]  # shape of [Dh, Dw]
        sck = S[:, c, k, :]  # shape of [N, 1, T]
        Tsck_core = toeplitz_sck_core(sck.squeeze(), [Dh, Dw, T])  # shape of [N, T, Dw], supposed to be [N, T*Dh, Dw*Dh]
        Md_core = (Tsck_core.abs().permute(0,2,1) @ Tsck_core.abs()).sum(2).sum(0)  # shape of [Dw]
        Md = Md_core.repeat(Dh)  # shape of [Dh * Dw]
        Md_inv = (Md + 1e-38) ** (-1)  # shape of [Dh * Dw]

        # dck_conv_sck = Func.conv2d(sck.unsqueeze(1),
        #                            dck.reshape(1, 1, Dh, Dw).flip(2, 3), padding=(255, 1)).squeeze()  # shape of [N,F,T]
        dck_conv_sck = (dck @ Tsck_core.permute(0, 2, 1)).reshape(N, F, T)
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        Dcp_conv_Sncp = DconvS[:, c, :] - dck_conv_sck
        # term 1, 2, 3 should be in the shape of [N, F, T] or [N, FT]
        term1 = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        term2 = Y[:, c].reshape(N, 1) * (X.reshape(N, -1) - R.reshape(N, -1) - Y[:, c].reshape(N, 1) * Dcp_conv_Sncp.reshape(N, -1) -
                        (Y[:, c_prime].reshape(NC_1, 1) * DconvS[:, c_prime, :].reshape(NC_1, -1)).reshape(N,C-1,-1).sum(1))
        term3 = -(1 - Y[:, c]).reshape(N, 1) * ((1 - Y[:, c]).reshape(N, 1) * Dcp_conv_Sncp.reshape(N,-1) +
                        ((1 - Y[:, c_prime].reshape(NC_1, 1)) * DconvS[:, c_prime, :].reshape(NC_1, -1)).reshape(N,C-1,-1).sum(1))
        b = (term1+ term2.reshape(N, F, T)  + term3.reshape(N, F, T) ) / 2
        torch.cuda.empty_cache()

        # l00 = loss_fun(X, Y, D, D0, S, S0, W, opts)
        # l0 = loss_fun_special(X, Y, D, D0, S, S0, W, opts)
        D[c, k, :] = solv_dck(dck, Md, Md_inv, opts.delta, Tsck_core, b)
        # ll0 = loss_fun_special(X, Y, D, D0, S, S0, W, opts)
        # print('Overall loss for fidelity, sparse, label, differences: %1.7f, %1.7f, %1.7f' %(l0[0]-ll0[0], l0[1]-ll0[1], l0[2]-ll0[2]))
        # print('Main loss after bpgm the diff is: %1.9e' %(l00 - loss_fun(X, Y, D, D0, S, S0, W, opts)))
        # if (l00 - loss_fun(X, Y, D, D0, S, S0, W, opts)) <0 : print(bug)
        # if torch.isinf(D).sum() > 0: print('inf_nan_happenned')
    return D


def toeplitz_sck_core(sck, Dh_Dw_T):
    """This is a the toepliz matrx for 2d-convolution for make dck into toeplitz matrix
    input sck  shape of [N, T], is to be made in toeplitz format
            Dh_Dw_T is the shape of atoms and sparse vector
    output tx has the shape of [N, T, Dw]
        supposed to be [N, F*T, Dh*Dw] (after truncation, too large)
    """
    dev = sck.device
    N = sck.shape[0]
    Dh, Dw, T = Dh_Dw_T
    'sck_append0 is shape of [N, 2Dw+T]'
    sck_append0 = torch.cat([torch.zeros(N, Dw, device=dev), sck.flip(1), torch.zeros(N, Dw, device=dev)], dim=1)
    indx = torch.zeros(T, Dw).long()
    offset = (Dw-1) //2
    for i in range(T):  # this part including truncation
        indx[i,:] = torch.arange(T+Dw-1-i-offset, T+2*Dw-i-1-offset)
    tx = sck_append0[:, indx]  # this is core
    """  # this part of the code is not practical, though it is explixitly the math expression
        # the following code will give out of memory issue
    tx = torch.cat([sck_append0[:, indx], torch.zeros(N, T, (Dh-1)*Dw, device=dev) ],dim=2) # shape of [N, T, Dw*Dh]
    ind = torch.zeros(Dh*T, Dh*Dw).long()
    ind0 = np.arange(T* Dh*Dw).reshape(T, Dh*Dw)
    for i in range(Dh):
        ind[T*i:T*(i+1), :] = torch.from_numpy(np.roll(ind0, Dw*i))
    return tx.view(sck.shape[0], -1)[ind]
    """
    return tx


def solv_dck(x, Md, Md_inv, Mw, Tsck_core, b):
    """x, is the dck, shape of [Dh, Dw]
        Md is with shape of [Dh * Dw], diagonal of the majorized matrix
        Md_inv, is Md^(-1), shape of [Dh * Dw]
        Mw, is a number, == opts.delta
        Tsck_core, is shape of [N, T, Dw]
                    is the core of Tsck, which is  [N, T*Dh, Dw*Dh]
        b is bn with all N, with shape of [N, F, T]
        """
    Dh, Dw = x.shape
    N, F, T = b.shape
    maxiter, correction, threshold = 500, 0.1, 1e-4  # correction is help to make the loss monotonically decreasing
    d_til, d_old, d = x.view(-1).clone(), x.view(-1).clone(), x.view(-1).clone()
    "coef is the shape of [N, Dw*Dh, Dw*Dh], with block diagnal structure of Dw*Dw small blocks"
    coef_core = (Tsck_core.permute(0, 2, 1) @ Tsck_core)  # shape of [N, Dw, Dw]
    term =(b@Tsck_core).reshape(N, -1)# shape of [N, DhDw],permute before reshape is a must

    # loss = torch.cat((torch.tensor([], device=x.device), loss_D(Tsck_core, d.view(Dh, Dw), b).reshape(1)))
    # dn = []
    for i in range(maxiter):
        d_til = d + correction*Mw*(d - d_old)  # shape of [M]
        nu = d_til - ((d_til.view(Dh,Dw) @ coef_core).reshape(N,-1) - term).sum(0) * Md_inv  # shape of [Dh * Dw]
        if torch.norm(nu) <= 1:
            d_new = nu
        else:
            d_new = acc_newton(Md, -Md*nu)  # QCQP(P, q)
        d, d_old = d_new, d
        if (d - d_old).norm() / d_old.norm() < threshold: break
        if torch.isnan(d).sum() > 0: print('inf_nan_happenned')
        torch.cuda.empty_cache()

    #     loss = torch.cat((loss, loss_D(Tsck_core, d.view(Dh, Dw), b).reshape(1)))
    #     dn.append(d.norm())
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return d.reshape(Dh, Dw)



def loss_D(Tsck_t, dck, b):
    """
    calculate the loss function value for updating D, sum( norm(Tsnck*dck - bn)**2 ) , s.t. norm(dck) <=1
    :param Tsck_t: is shape of [N, T, Dw]
    :param dck: cth, kth, atom of D, shape of [Dh, Dw]
    :param b: the definiation is long in the algorithm, shape of [N, F, T]
    :return: loss fucntion value
    """
    return 2*((dck@Tsck_t.permute(0, 2, 1) - b)**2 ).sum()


def acc_newton(P, q):  # both shape of [M]
    """
    proximal operator for majorized form using Accelorated Newton's method
    follow the solutions of `convex optimization` by boyd, exercise 4.22, solving QCQP
    where P is a square diagonal matrix, q is a vector
    :param P: for update D, P = MD          for update D0, P = Mdk0 + rho*eye(M)
    :param q: for update D, q = -MD@nu,     for update D0, q = -(MD@nu + rho*Zk0 + Yk0),
    :return: dck or dck0
    """
    psi, maxiter= 0, 500
    qq = q*q
    if (qq == 0).sum() > 1:
        psi_new = 0  # q is too small
        print('acc_newton happenend')
        # input()
    else:
        for i in range(maxiter):
            f_grad = -2 * ((P+psi)**(-3) * qq).sum()
            f = ((P+psi)**(-2)*qq).sum()
            psi_new = psi - 2 * f/f_grad * (f.sqrt() - 1)
            if (psi_new - psi).item() < 1e-5:  # psi_new should always larger than psi
                break
            else:
                psi = psi_new.clone()
    dck = -((P + psi_new)**(-1)) * q
    if torch.isnan(dck).sum() > 0: print(inf_nan_happenned)
    return dck



def updateD0(DD0SS0, X, Y, opts):
    """this function is to update the common dictionary D0 using BPG-M, updating each d_k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,1,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,Dh,Dw] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, 1, T]
        D0 is a matrix [K0, Dh, Dw]
        X is a matrix [N, F, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C
    Crange = torch.tensor(range(C))
    NC_1, FT = N * (C - 1), F*T

    # "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
    # DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
    #                      padding=(255, 1), groups=CK).reshape(N, C, K, F, T).sum(2)
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)
    for ck in range(CK):
        DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
    DconvS_NFT = DconvS0.sum(1)
    ycDcconvSc = (Y.reshape(NC, 1) * DconvS0.reshape(NC, -1)).reshape(N, CK, F, T).sum(1)  # output shape of (N, F, T)

    # '''update the current dk0'''
    for k0 in range(K0):
        dk0 = D0[k0, :]  # shape [Dh, Dw]
        snk0 = S0[:, k0, :]  # shape of [N, 1, T]
        "shape of (N, F, T), R is the common recon."
        # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3), padding=(255,opts.offset), groups=K0).sum(1)
        R0 = torch.zeros(N, K0, F, T, device=opts.dev)
        D0r = D0.reshape(K0, 1, Dh, Dw)
        for kk0 in range(K0):
            R0[:, kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
        R = R0.sum(1)
        Tdk0 = toeplitz_dck(dk0, [Dh, Dw, T])  # shape of [FT, T]

        Tsnk0_core = toeplitz_sck_core(snk0.squeeze(), [Dh, Dw, T])  # shape of [N, T, Dw]
        # dk0convsnk0=Func.conv2d(snk0.unsqueeze(1),dk0.reshape(1,1,Dh,Dw).flip(2,3),padding=(255,opts.offset)).squeeze()
        dk0convsnk0 = (dk0 @ Tsnk0_core.permute(0, 2, 1)).reshape(N, F, T)
        b = 2*X - DconvS_NFT - ycDcconvSc - 2*R + 2*dk0convsnk0  #shape of [N, F, T]

        Md_core = 4*(Tsnk0_core.abs().permute(0,2,1) @ Tsnk0_core.abs()).sum(2).sum(0)  # shape of [Dw]
        Md = Md_core.repeat(Dh)  # shape of [Dh * Dw]
        Md_inv = (Md + 1e-38) ** (-1)  # shape of [Dh * Dw]
        torch.cuda.empty_cache()

        # print('D0 loss function value before update is %3.2e:' %loss_D0(2*Tsnk0_t, dk0, b, D0, opts.mu*N))
        D0[k0, :] = solv_dck0(dk0, Md, Md_inv, opts.delta, 2*Tsnk0_core, b, D0, opts.mu * N, k0)
        # print('D0 loss function value after update is %3.2e:' % loss_D0(2*Tsnk0_t, dk0, b, D0, opts.mu*N))
        if torch.isnan(D0).sum() + torch.isinf(D0).sum() > 0: print(inf_nan_happenned)
    return D0


def solv_dck0(x, M, Minv, Mw, Tsnk0_core, b, D0, mu, k0):
    """
    :param x: is the dck, shape of [Dh, Dw]
    :param M: is MD, the majorizer matrix with shape of [Dh*Dw], diagonal matrix
    :param Minv: is MD^(-1), shape of [Dh*Dw]
    :param Mw: is a number not diagonal matrix
    :param Tsnk0_core: is truncated toeplitz matrix of sck with shape of [N,T,Dw], already *2
    :param b: bn with all N, with shape of [N,F,T]
    :param D0: is the shared dictionary, shape of [K0, Dh, Dw]
    :param mu: is the coefficient fo low-rank term, mu = N*mu
    :param k0: the current index of for loop of K0
    :return: dck0: is shape of [Dh, Dw]
    """
    # for the synthetic data correction = 0.1
    [K0, Dh, Dw]= D0.shape
    N, F, T = b.shape
    DhDw = Dh * Dw
    maxiter, correction, threshold = 500, 0.1, 1e-4  # correction is help to make the loss monotonically decreasing
    d_til, d_old, d = x.view(-1).clone(), x.view(-1).clone(), x.view(-1).clone()
    "coef is the shape of [N, Dw*Dh, Dw*Dh], with block diagnal structure of Dw*Dw small blocks"
    coef_core = (Tsnk0_core.permute(0, 2, 1) @ Tsnk0_core)  # shape of [N, Dw, Dw]
    term =(b@Tsnk0_core).reshape(N, -1)# shape of [N, DhDw],permute before reshape is a must

    # loss = torch.cat((torch.tensor([], device=x.device), loss_D0(Tsck0_t, d, b, D0, mu).reshape(1)))
    for i in range(maxiter):
        d_til = d + correction*Mw*(d - d_old)  # shape of [DhDw],  Mw is just a number for calc purpose
        nu = d_til - ((d_til.view(Dh, Dw) @ coef_core).reshape(N, -1) - term).sum(0) * Minv # shape of [DhDw]
        d_new = argmin_lowrank(M, nu, mu, D0.view(K0, DhDw), k0)  # D0 will be changed, because dk0 is in D0
        d, d_old = d_new, d
        if (d - d_old).norm()/d_old.norm() < threshold:break
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_D0(Tsck0_t, d, b, D0, mu).reshape(1)))
    # ll = loss[:-1] - loss[1:]
    # if ll[ll<0].shape[0] > 0: print(something_wrong)
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return d.reshape(Dh, Dw)


def argmin_lowrank(M, nu, mu, D0, k0):
    """
    Solving the QCQP with low rank panelty term. This function is using ADMM to solve dck0
    :param M: majorizer matrix
    :param nu: make d close to ||d-nu||_M^2
    :param mu: hyper-param of ||D0||_*
    :param D0: common dict contains all the dk0, shape of [K0, Dh*Dw]
    :return: dk0
    """
    (K0, m), threshold = D0.shape, 5e-4
    rho = 10 * mu +1e-38 # agrangian coefficients
    dev = D0.device
    Z = torch.eye(K0, m, device=dev)
    Y = torch.eye(K0, m, device=dev)  # lagrangian coefficients
    P = M + rho
    Mbynu = M * nu
    maxiter = 200
    cr = torch.tensor([], device=dev)
    # begin of ADMM
    for i in range(maxiter):
        Z = svt(D0-1/rho*Y, mu/rho)
        q = -(Mbynu + rho * Z[k0, :] + Y[k0, :])
        dk0 = D0[k0, :] = acc_newton(P, q)
        Z_minus_D0 = Z- D0
        Y = Y + rho*Z_minus_D0
        cr = torch.cat((cr, Z_minus_D0.norm().reshape(1)))
        if i>10 and abs(cr[-1] - cr[-2])/cr[i-1] < threshold: break
        if cr[-1] <1e-6 : break
    return dk0


def svt(L, tau):
    """
    This function is to implement the signular value thresholding, solving the following
    min_P tau||P||_* + 1/2||P-L||_F^2
    :param L: low rank matrix to proximate
    :param tau: the threshold
    :return: P the matrix after singular value thresholding
    """
    dev = L.device
    # L = L.cpu()  ##########  in version 1.2 the torch.svd for GPU could be much slower than CPU
    l, h = L.shape
    try:
        u, s, v = torch.svd(L)
    except:                     ########## and torch.svd may have convergence issues for GPU and CPU.
        u, s, v = torch.svd(L + 1e-4*L.mean()*torch.rand(l, h))
        print('unstable svd happened')
    s = s - tau
    s[s<0] = 0
    P = u @ s.diag() @ v.t()
    return P.to(dev)


def updateW(SW, Y, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        SW is a list of [S, W]
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        W is a matrix [C, K+1], where K is per-class atoms
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    S, W = SW
    N, C, K, _, T = S.shape
    # print('the loss_W for updating W %1.3e:' %loss_W(S, W, Y))
    for c in range(C):
        # print('Before bpgm wc loss is : %1.3e'
        #       % loss_W(S[:, c, :, :].clone().unsqueeze(1), W[c, :].reshape(1, -1), Y[:, c].reshape(N, -1)))
        W[c, :] = solv_wc(W[c, :].clone(), S[:, c, :].squeeze(2), Y[:, c], opts.delta)
        # print('After bpgm wc loss is : %1.3e'
        #       % loss_W(S[:, c, :, :].clone().unsqueeze(1), W[c, :].reshape(1, -1), Y[:, c].reshape(N, -1)))
        # print('the loss_W for updating W %1.3e' %loss_W(S, W, Y))
    if torch.isnan(W).sum() + torch.isinf(W).sum() > 0: print(inf_nan_happenned)
    return W


def solv_wc(x, snc, yc, Mw):
    """
    This fuction is using bpgm to update wc
    :param x: shape of [K+1], init value of wc
    :param snc: shape of [N, K, T]
    :param yc: shape of [N]
    :param Mw: real number, is delta
    :return: wc
    """
    # for the synthetic data correction = 0.1
    N, threshold = yc.shape[0], 1e-4
    maxiter, correction = 500, 0.1  # correction is help to make the loss monotonically decreasing
    wc_old, wc, wc_til = x.clone(), x.clone(), x.clone()
    pt_snc = torch.cat((snc.mean(2) , torch.ones(N, 1, device=x.device)), dim=1) # shape of [N, K+1]
    abs_pt_snc = abs(pt_snc)  # shape of [N, K+1]
    const = abs_pt_snc.t() * abs_pt_snc.sum(1)  # shape of [K+1, N]
    M = const.sum(1)/4 + 1e-38  # shape of [K], 1e-38 for robustness
    one_min_ync = 1 - yc  # shape of [N]
    # M_old = M.clone()
    # print('before bpgm wc loss is : %1.3e' %loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)))

    # loss = torch.cat((torch.tensor([], device=x.device),
    #           loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)).reshape(1)))
    for i in range(maxiter):
        wc_til = wc + correction*Mw*(wc - wc_old)  # Mw is just a number for calc purpose
        exp_pt_snc_wc_til = (pt_snc @ wc_til).exp()  # shape of [N]
        exp_pt_snc_wc_til[torch.isinf(exp_pt_snc_wc_til)] = 1e38
        nu = wc_til + M**(-1) * ((one_min_ync - exp_pt_snc_wc_til/(1+exp_pt_snc_wc_til))*pt_snc.t()).sum(1)  # nu is [K]
        wc, wc_old = nu.clone(), wc[:]  # gradient is not needed, nu is the best solution
        # loss = torch.cat((loss, loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)).reshape(1)))
        if torch.norm(wc - wc_old)/wc.norm() < threshold: break
        torch.cuda.empty_cache()
    # ll = loss[:-1] - loss[1:]
    # if ll[ll<0].shape[0] > 0: print(something_wrong)
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return wc


def loss_W(S, W, Y):
    """
    calculating the loss function value for subproblem of W
    :param S: shape of [N, C, K, T]
    :param W: shape of [C, K+1]
    :param Y: shape of [N, C]
    :return:
    """
    N, C = Y.shape
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1/ (1+ exp_PtSnW)
    # loss0 = -1 * (Y * Y_hat.log() + (1 - Y) * (1 - Y_hat + 3e-38).log()).sum()  # the same as below
    loss = (-1 * (1 - Y) * (exp_PtSnW + 1e-38).log() + (exp_PtSnW + 1).log()).sum()

    # loss = (-1 * (1 - Y) * PtSnW + (exp_PtSnW + 1).log()).sum()
    # this one is not stable due inf by setting the threshold 1e38, which means
    # if PtSnW = 40, then exp_PtSnW = inf, but set to exp_PtSnW = 1e38, log(exp_PtSnW) = 38, not 40
    return loss


def plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts):
    """
    This function is the main training body of the algorithm, with showing a lot of details
    :param D: initial value, D, shape of [C, K, Dh, Dw]
    :param D0: pre-trained D0,  shape of [C0, K0, Dh, Dw]
    :param S: initial value, shape of [N,C,K, 1, T]
    :param S0: initial value, shape of [N,K0,1,T]
    :param W: The pre-trained projection, shape of [C, K+1]
    :param X: testing data, shape of [N, F, T]
    :param Y: testing Lable, ground truth, shape of [N, C]
    :param opts: options of hyper-parameters
    :return: D, D0, S, S0, W, loss
    """
    N, C = Y.shape
    S_tik = torch.cat((S.squeeze(3).mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)

    "plot the input data"
    plt.imshow(X[0].cpu())



def test(D, D0, S, S0, W, X, Y, opts):
    """
    This function is made to see the test accuracy by checking the reconstrunction label, with details
    :param D: The pre-trained D, shape of [C, K, M]
    :param D0: pre-trained D0,  shape of [C0, K0, M]
    :param S: initial value, shape of [N,C,K,T]
    :param S0: initial value, shape of [N,K0,T]
    :param W: The pre-trained projection, shape of [C, K]
    :param X: testing data, shape of [N, T]
    :param Y: testing Lable, ground truth, shape of [N, C]
    :param opts: options of hyper-parameters
    :return: acc, Y_hat
    """
    loss, threshold = torch.tensor([], device=opts.dev), 5e-4
    loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
    print('The initial loss function value is %3.4e:' % loss[-1])
    S_numel, S0_numel = S.numel(), S0.numel()
    S, S0 = S.clone(), S0.clone()
    for i in range(opts.maxiter):
        t0 = time.time()
        Sold = S.clone()
        S = updateS_test([D, D0, S, S0], X, opts)
        loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
        if opts.show_details:
            print('check sparsity, None-zero percentage is : %1.4f' % (1-(S==0).sum().item()/S_numel))
            print('In the %1.0f epoch, the sparse coding time is :'
                  '%3.2f, loss function value is :%3.4e'% (i, time.time() - t0, loss[-1]))
            t0 = time.time()
        S0 = updateS0_test([D, D0, S, S0], X, opts)
        if opts.show_details:
            loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
            print('In the %1.0f epoch, the sparse0 coding time is :'
                  '%3.2f, loss function value is :%3.4e'% (i, time.time() - t0, loss[-1]))
        if opts.show_details:
            if i > 3 and abs((loss[-1] - loss[-3]) / loss[-3]) < threshold:
                print('break condition loss value diff satisfied')
                break
            if support_diff(S, Sold) < 0.003:
                print('break condition support diff satisfied')
                break
        else:
            if i > 3 and abs((loss[-1] - loss[-2]) / loss[-2]) < threshold:
                print('break condition loss value diff satisfied')
                break
            if support_diff(S, Sold) < 0.003:
                print('break condition support diff satisfied')
                break
            if i%3 == 0 : print('In the %1.0f epoch, the sparse coding time is :%3.2f' % ( i, time.time() - t0 ))
    N, C = Y.shape
    S_tik = torch.cat((S.squeeze(3).mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    y_hat = 1 / (1 + exp_PtSnW)
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat <= 0.5] = 0
    label_diff = Y - y_hat
    acc = label_diff[label_diff==0].shape[0]/label_diff.numel()
    acc_all = OPT(silent=True)
    acc_all.acc = acc
    acc_all.recall = recall(Y, y_hat)
    acc_all.f1 = f1(Y, y_hat)
    return acc_all, 1/(1+exp_PtSnW), S, S0, loss


def loss_fun_test(X, D, D0, S, S0, opts):
    """
        This function will calculate the costfunction value
        :param X: the input data with shape of [N, F, T]
        :param Y: the input label with shape of [N, C]
        :param D: the discriminative dictionary, [C,K,Dh,Dw]
        :param D0: the common dictionary, [K0,Dh,Dw]
        :param S: the sparse coefficients, shape of [N,C,K,1,T] [samples, classes, num of atoms, time series,]
        :param S0: the common coefficients, 3-d tensor [N, K0,1,T]
        :param W: the projection for labels, shape of [C, K+1]
        :param opts: the hyper-parameters
        :return: cost, the value of loss function
        """
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C

    # DconvS should be the shape of (N, CK, F,T), R is the common reconstruction
    # DconvS = Func.conv2d(S.reshape(N, CK, 1, T) ,D.reshape(CK, 1, Dh, Dw).flip(2,3), padding=(255,1), groups=CK)
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)
    for ck in range(CK):
        DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
    DconvS_NFT = DconvS0.sum(1)

    # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3), padding=(255,opts.offset), groups=K0).sum(1)
    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for kk0 in range(K0):
        R0[:,kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
    R = R0.sum(1)
    torch.cuda.empty_cache()

    fidelity = torch.norm(X - R - DconvS_NFT) ** 2
    sparse = opts.lamb * S.abs().sum() + opts.lamb0 * S0.abs().sum()
    l2 = opts.lamb2 * (S.norm() ** 2 + S0.norm() ** 2)
    cost = fidelity + sparse + l2
    return cost


def updateS_test(DD0SS0, X, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,1,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,Dh,Dw] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, 1,T]
        D0 is a tensor [K0, Dh,Dw]
        X is a tensor [N, F, T], training Data
        Y are the labels, not given
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C
    NC_1, FT = N * (C - 1), F * T

    "shape of (N, F, T), R is the common recon."
    # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3), padding=(255,opts.offset), groups=K0).sum(1)
    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for kk0 in range(K0):
        R0[:,kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
    R = R0.sum(1)

    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        # "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
        for ck in range(CK):
            DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
        DconvS_NFT = DconvS0.sum(1)

        dck = D[c, k, :]  # shape of [Dh, Dw]
        sck = S[:, c, k, :]  # shape of [N, 1, T]
        # dck_conv_sck = Func.conv2d(sck.unsqueeze(1),
        #                            dck.reshape(1, 1, Dh, Dw).flip(2, 3),padding=(255, 1)).squeeze()  # shape of [N,F,T]
        Tdck = toeplitz_dck(dck, [Dh, Dw, T])  # shape of [FT, T]
        dck_conv_sck = (Tdck @ sck.permute(0, 2, 1)).reshape(N, F, T)
        b = X - R - (DconvS_NFT - dck_conv_sck)  # shape of [N, F, T]
        torch.cuda.empty_cache()

        S[:, c, k, :] = solv_sck_test(S[:, c, :].squeeze(2), Tdck, b.reshape(N, FT), k, opts.delta, opts.lamb, opts.lamb2)
        if torch.isnan(S).sum() + torch.isinf(S).sum() >0 : print(inf_nan_happenned)
    return S


def solv_sck_test(sc, Tdck, b, k, delta, lamb, lamb2):
    """
    This function solves snck for all N, using BPGM
    :param sc: shape of [N, K, T]
    :param Tdck: shape of [FT, m=T]
    :param b: shape of [N, FT]
    :param k: integer, which atom to update
    :param lamb2: default as 0, adjustment sparse coding, if needed
    :return: sck
    """
    maxiter, correction, threshold = 500, 0.7, 1e-5
    Mw = delta * correction # correction is help to make the loss monotonically decreasing
    dev = sc.device
    T = sc.shape[2]
    # 'skc update will lead sc change'
    sck = sc[:, k, :].clone()  # shape of [N, T]
    sck_old = sck.clone()
    abs_Tdck = abs(Tdck)
    Tdck_t_Tdck = Tdck.t() @ Tdck  # shape of [T, T]
    Tdckt_bt = Tdck.t() @ b.t()  # shape of [T, N]
    M = (abs_Tdck.t() @ abs_Tdck + lamb2*torch.eye(T, device=dev)  + 1e-38).sum(1)  # M is the diagonal, shape of [T]
    sc_til, sc_old, marker = sc.clone(), sc.clone(), 0 # shape of [N, K, T]

    # loss = torch.cat((torch.tensor([], device=opts.dev), loss_Sck_test(Tdck, b, sc, sck, opts).reshape(1)))
    for i in range(maxiter):
        sck_til = sck + Mw * (sck - sck_old)  # shape of [N, T]
        sc_til[:, k, :] = sck_til
        nu = sck_til - (Tdck_t_Tdck@sck_til.t() - Tdckt_bt + lamb2 *sck_til.t()).t()/M  # shape of [N, T]
        sck_new = shrink(M, nu, lamb/2)  # shape of [N, T]
        sck_old[:], sck[:] = sck[:], sck_new[:]  # make sure sc is updated in each loop
        if torch.norm(sck - sck_old) / (sck.norm() + 1e-38) < threshold: break
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_Sck_test(Tdck, b, sc, sck, opts).reshape(1)))
    # print('M max', M.max())
    # if marker == 1 :
    #     print('--inf to 1e38 happend within the loop')
    #     plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    #     print('How many inf to 1e38 happend finally', exp_PtSnc_tilWc[exp_PtSnc_tilWc == 1e38].shape[0])
    # if (loss[0] - loss[-1]) < 0 :
    #     wait = input("Loss Increases, PRESS ENTER TO CONTINUE.")
    # print('sck loss after bpgm the diff is :%1.9e' %(loss[0] - loss[-1]))
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return sck_old.unsqueeze(-2)


def updateS0_test(DD0SS0, X, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
        S is 4-d tensor [N,C,K,1,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,Dh,Dw] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, 1,T]
        D0 is a tensor [K0, Dh,Dw]
        X is a tensor [N, F, T], training Data
        Y are the labels, not given
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, F, T = X.shape
    K0, Dh, Dw = D0.shape
    C, K, *_ = D.shape
    CK, NC = K * C, N * C
    NC_1, FT = N * (C - 1), F * T

    "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
    # DconvS = Func.conv2d(S.reshape(N, CK, 1, T),
    #                      D.reshape(CK, 1, Dh, Dw).flip(2, 3),padding=(255, 1), groups=CK).reshape(N, C, K, F, T).sum(2)
    DconvS0 = torch.zeros(N, CK, F, T, device=opts.dev)
    Dr = D.reshape(CK, 1, Dh, Dw)
    for ck in range(CK):
        DconvS0[:, ck] = Func.conv1d(S.reshape(N,CK,1,T)[:,ck], Dr[ck].permute(1,0,2).flip(2),padding=opts.offset).squeeze()
    DconvS_NFT = DconvS0.sum(1)  # shape of [N, F, T]

    R0 = torch.zeros(N, K0, F, T, device=opts.dev)
    D0r = D0.reshape(K0, 1, Dh, Dw)
    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, 1, T]
        # R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3),
        #                 padding=(255, opts.offset), groups=K0).sum(1)  # shape of (N, F, T), R is the common recon.
        for kk0 in range(K0):
            R0[:, kk0] = Func.conv1d(S0[:, kk0], D0r[kk0].permute(1, 0, 2).flip(2), padding=opts.offset).squeeze()
        R = R0.sum(1)

        Tdk0 = toeplitz_dck(dk0, [Dh, Dw, T])  # shape of [FT, T]
        # dk0convsck0=Func.conv2d(snk0.unsqueeze(1),dk0.reshape(1,1,Dh,Dw).flip(2,3),padding=(255,opts.offset)).squeeze()
        dk0convsck0 = (Tdk0 @ snk0.permute(0, 2, 1)).reshape(N, F, T)
        b = X - DconvS_NFT - R + dk0convsck0
        torch.cuda.empty_cache()
        # print(loss_S0(2*Tdk0_t.t(), snk0, b, opts.lamb))
        S0[:, k0, :] = solv_sck_test(S0.squeeze(2), Tdk0, b.reshape(N, FT), k0, opts.delta, opts.lamb0, opts.lamb2)
        # print(loss_S0(2*Tdk0_t.t(), S0[:, k0, :], b, opts.lamb))
    return S0


def support_diff(S, Sold):
    """
    This function will return the percentage of the difference for the non-zero locations of the sparse coeffients
    :param S: sparse coeffients
    :param Sold: sparse coeffients
    :return: percentage of different
    """
    sf, ssf = S.view(-1), Sold.view(-1)
    a, b = torch.zeros(sf.numel()), torch.zeros(ssf.numel())
    a[sf != 0] = 1
    b[ssf != 0] = 1
    return (a - b).abs().sum().item() / (b.sum().item()+ 1e-38)


def recall(y, yh):
    """
    calculate the recall score for a matrix
    :param y: N by C
    :param yh: predicted N by C
    :return: recall_score
    """
    # N, C = y.shape
    # s = 0
    # for i in range(C):
    #     s = s + metrics.recall_score(yc[:,i], yhc[:,i])
    # res =s/C
    yc = np.array(y.cpu())
    yhc = np.array(yh.cpu())
    res = metrics.recall_score(yc.flatten(), yhc.flatten())
    return res


def precision(y, yh):
    """
    calculate the recall score for a matrix
    :param y: N by C
    :param yh: predicted N by C
    :return: precision_score
    """
    yc = np.array(y.cpu())
    yhc = np.array(yh.cpu())
    res = metrics.precision_score(yc.flatten(), yhc.flatten())
    return res

def f1(y, yh):
    """
    calculate the recall score for a matrix
    :param y: N by C
    :param yh: predicted N by C
    :return: precision_score
    """
    yc = np.array(y.cpu())
    yhc = np.array(yh.cpu())
    res = metrics.f1_score(yc.flatten(), yhc.flatten())
    return res