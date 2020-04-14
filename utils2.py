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
    def __init__(self, C=4, K0=1, K=1, Dh=256, Dw=3,\
                 mu=0.1, eta=0.1, lamb=0.1, delta=0.9, maxiter=500, silent=False):
        self.C, self.K, self.K0, self.Dh, self.Dw = C, K, K0, Dh, Dw
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
        S is 5-d tensor [N,C,K,F,T] [samples,classes, num of atoms, Freq, time series,]
        D is 4-d tensor [C,K,F,M] [num of atoms, classes, atom size]
        S0 is 4-d tensor [N, K0, F, T]
        D0 is 3-d tensor [K0, F, M]
        X is 3-d tensor [N, F, T], training Data, could be in GPU
        Y is a matrix [N, C] \in {0,1}, training labels
        W is a matrix [C, K+1], where K is per-class atoms
    :return: D, D0, S, S0, W
    """
    if opts.transpose:
        N, T, F = X.shape
    else:
        N, F, T = X.shape
    D = torch.rand(opts.C, opts.K, opts.Dh,opts.Dw, device=opts.dev)
    D = D/(D*D).sum().sqrt()  # normalization
    D0 = torch.rand(opts.K0, opts.Dh, opts.Dw, device=opts.dev)
    D0 = D0/(D0*D0).sum().sqrt()  # normalization
    S = torch.zeros(N, opts.C, opts.K, 1, T, device=opts.dev)
    S0 = torch.zeros(N, opts.K0, 1, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K +1, device=opts.dev)
    return D, D0, S, S0, W


def train(D, D0, S, S0, W, X, Y, opts):
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
    loss, threshold = torch.tensor([], device=opts.dev), 5e-4
    # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    # print('The initial loss function value is :%3.4e' % loss[-1])
    t, t1 = time.time(), time.time()
    S_numel, S0_numel = S.numel(), S0.numel()
    for i in range(opts.maxiter):
        t0 = time.time()
        # S = updateS([D, D0, S, S0, W], X, Y, opts)
        if opts.show_details:
            # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
            print('pass S, time is %3.2f' % (time.time() - t)); t = time.time()
            # print('loss function value is %3.4e:' %loss[-1])
            print('check sparsity, None-zero percentage is : %1.4f' % (1-(S==0).sum().item()/S_numel))

        # S0 = updateS0([D, D0, S, S0], X, Y, opts) if opts.common_term else S0
        if opts.show_details:
            # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
            print('pass S0, time is %3.2f' % (time.time() - t)); t = time.time()
            # print('loss function value is %3.4e:' %loss[-1])
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
    # DconvS should be the shape of (N, CK, F,T)
    DconvS = Func.conv2d(S.reshape(N, CK, 1, T) ,D.reshape(CK, 1, Dh, Dw).flip(2,3), padding=(255,1), groups=CK)
    ycDcconvSc = (Y.reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N,CK,F,T).sum(1)  # output shape of (N, F, T)
    ycpDcconvSc =((1-Y).reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N,CK,F,T).sum(1)  # output shape of (N, F, T)
    DconvS = DconvS.sum(1)  # using the same name to save memory
    R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2,3),  padding=(255,1), groups=K0).sum(1)  # shape of (N, F, T), R is the common recon.
    torch.cuda.empty_cache()

    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    S_tik = torch.cat((S.squeeze().mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    _1_Y_hat = 1 - Y_hat
    fisher1 = torch.norm(X - R - DconvS)**2
    fisher2 = torch.norm(X - R - ycDcconvSc) ** 2
    fisher = fisher1 + fisher2 + torch.norm(ycpDcconvSc) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum() * opts.eta
    # label = -1 * opts.eta * (Y * (Y_hat + 3e-38).log() + (1 - Y) * (_1_Y_hat + 3e-38).log()).sum()
    low_rank = N * opts.mu * D0.reshape(D0.shape[-2], D0.shape[-1]*K0).norm(p='nuc')
    cost = fisher + sparse + label + low_rank
    return cost


def save_results(D, D0, S, S0, W, opts, loss):
    """
    This function will save the training results
    :param D: initial value, D, shape of [C, K, M]
    :param D0: pre-trained D0,  shape of [C0, K0, M]
    :param S: initial value, shape of [N,C,K,T]
    :param S0: initial value, shape of [N,K0,T]
    :param W: The pre-trained projection, shape of [C, K]
    """
    param = str([opts.K, opts.K0, opts.M, opts.lamb, opts.eta , opts.mu])
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
    R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2,3),  padding=(255,1), groups=K0).sum(1)  # shape of (N, F, T), R is the common recon.
    Crange = torch.tensor(range(C))
    NC_1, FT = N * (C - 1), F*T

    # '''update the current s_n,k^(c) '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
        DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
                             padding=(255, 1), groups=CK).reshape(N,C,K, F, T).sum(2)
        dck = D[c, k, :]  # shape of [Dh, Dw]
        sck = S[:, c, k, :]  # shape of [N, 1, T]
        wc = W[c, :]  # shape of [K+1], including bias
        yc = Y[:, c]  # shape of [N]
        dck_conv_sck = Func.conv2d(sck.unsqueeze(1), dck.reshape(1,1,Dh,Dw).flip(2, 3), padding=(255, 1)).squeeze()  # shape of [N,F,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        Dcp_conv_Sncp = DconvS[:, c, :] - dck_conv_sck  # shape of [N, F, T]
        Tdck = toeplitz_dck(dck, [Dh, Dw, T]) # shape of [T, m=T]
        # term 1, 2, 3 should be in the shape of [N, F, T] or [N, F*T]
        term1 = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        term2 = Y[:, c].reshape(N, 1) * (X.reshape(N, -1) - R.reshape(N, -1) - Y[:, c].reshape(N, 1) * Dcp_conv_Sncp.reshape(N, -1) -
                        (Y[:, c_prime].reshape(NC_1, 1) * DconvS[:, c_prime, :].reshape(NC_1, -1)).reshape(N,C-1,-1).sum(1))
        term3 = -(1 - Y[:, c]).reshape(N, 1) * ((1 - Y[:, c]).reshape(N, 1) * Dcp_conv_Sncp.reshape(N,-1) +
                        ((1 - Y[:, c_prime].reshape(NC_1, 1)) * DconvS[:, c_prime, :].reshape(NC_1, -1)).reshape(N,C-1,-1).sum(1))
        b = (term1.reshape(N, FT) + term2 + term3) / 2
        torch.cuda.empty_cache()

        S[:, c, k, :] = solv_sck(S[:, c, :].squeeze(), wc, yc, Tdck, b, k, opts)
        if torch.isnan(S).sum() + torch.isinf(S).sum() > 0: print('inf_nan_happenned')
    return S



def toeplitz_dck(dck, Dh_Dw_T):
    """This is a the toepliz matrx for 2d-convolution for make dck into toeplitz matrix
    input dck  shape of [Dh, Dw], is to be made in toeplitz format
            Dh_Dw_T is the shape of atoms and sparse vector
    output tx has the shape of [Dh*T, T] (after truncation)
    """
    dev = dck.device
    Dh, Dw, T = Dh_Dw_T
    m, m1, m2= Dh*(T+Dw-1), T+Dw-1, 2*T+Dw
    'dck_padded0 is the shape of [Dh, m2]'
    dck_padded0 = torch.cat([torch.zeros(Dh, T, device=dev), dck.flip(1), torch.zeros(Dh, T, device=dev)], dim=1)
    indx = torch.zeros(m, T).long()
    for i in range(m):
        ii = i % m2
        iii = i // m1
        indx[i, :] = torch.arange(m1-ii+iii*m2, m2-1-ii+iii*m2)
    tx = dck_padded0.view(-1)[indx]

    # this part is for truncation
    ind1 = (Dw-1) //2
    ind2 = T + ind1
    rr = torch.arange(ind1, ind2)
    ind = rr
    for i in range(1, Dh):
        ind = torch.cat([ind, m1*i + rr])
    return tx[ind]



def solv_sck(sc, wc, yc, Tdck, b, k, opts):
    """
    This function solves snck for all N, using BPGM
    :param sc: shape of [N, K, T]
    :param wc: shape of [K+1], with bias
    :param yc: shape of [N]
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
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
        # loss = torch.cat((loss, loss_Sck(Tdck, b, sc, sck, wc, wkc, yc, opts).reshape(1)))
        torch.cuda.empty_cache()

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

    "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
    DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
                         padding=(255, 1), groups=CK).reshape(N, C, K, F, T).sum(2)
    ycDcconvSc = (Y.reshape(NC, 1) * DconvS.reshape(NC, -1)).reshape(N, C, F, T).sum(1)  # output shape of (N, F, T)
    R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2, 3), padding=(255, 1), groups=K0).sum(1)  # shape of (N, F, T), R is the common recon.
    alpha_plus_dk0 = DconvS.sum(1) + R
    beta_plus_dk0 = ycDcconvSc + R

    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, 1, T]
        dk0convsck0 = Func.conv2d(snk0.unsqueeze(1), dk0.reshape(1,1,Dh,Dw).flip(2, 3), padding=(255,1)).squeeze()
        Tdk0 = toeplitz_dck(dk0, [Dh, Dw, T])  # shape of [FT, T]
        abs_Tdk0 = abs(Tdk0)
        MS0_diag = (4*abs_Tdk0.t() @ abs_Tdk0).sum(1)  # in the shape of [T]
        MS0_diag = MS0_diag + 1e-38 # make it robust for inverse
        MS0_inv = (1/MS0_diag).diag()
        b = (2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsck0).reshape(N, FT)

        torch.cuda.empty_cache()
        # print(loss_S0(2*Tdk0_t.t(), snk0, b, opts.lamb))
        S0[:, k0, :] = solv_snk0(snk0.squeeze(), MS0_diag, MS0_inv, opts.delta, 2*Tdk0, b, opts.lamb)
        # print(loss_S0(2*Tdk0_t.t(), S0[:, k0, :], b, opts.lamb))
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
    R = Func.conv2d(S0, D0.reshape(K0, 1, Dh, Dw).flip(2,3),  padding=(255,1), groups=K0).sum(1)  #shape of (N, F, T)
    Crange = torch.tensor(range(C))
    NC_1, FT = N * (C - 1), F*T

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        "DconvS is the shape of (N,CK, F, T) to (N, C, F, T)"
        DconvS = Func.conv2d(S.reshape(N, CK, 1, T), D.reshape(CK, 1, Dh, Dw).flip(2, 3),
                             padding=(255, 1), groups=CK).reshape(N, C, K, F, T).sum(2)
        dck = D[c, k, :]  # shape of [Dh, Dw]
        sck = S[:, c, k, :]  # shape of [N, 1, T]
        Tsck_core = toeplitz_sck_core(sck.squeeze(), [Dh, Dw, T])  # shape of [N, T, Dw]
        Md_core = (Tsck_core.abs().permute(0,2,1) @ Tsck_core.abs()).sum(2).sum(0)  # shape of [Dw]
        Md = Md_core.repeat(Dh)  # shape of [Dh * Dw]
        Md_inv = (Md + 1e-38) ** (-1)  # shape of [Dh * Dw]

        dck_conv_sck = Func.conv2d(sck.unsqueeze(1), dck.reshape(1, 1, Dh, Dw).flip(2, 3), padding=(255, 1)).squeeze()  # shape of [N,F,T]
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

        D[c, k, :] = solv_dck(dck, Md, Md_inv, opts.delta, Tsck_core, b)
        if torch.isinf(D).sum() > 0: print(inf_nan_happenned)
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
    """x, is the dck, shape of [Dh * Dw]
        M is with shape of [Dh * Dw], diagonal of the majorized matrix
        Minv, is Md^(-1), shape of [Dh * Dw]
        Mw, is a number, == opts.delta
        Tsck_core, is shape of [N, T, Dw]
                    is the core of Tsck, which is  [N, T*Dh, Dw*Dh]
        b is bn with all N, with shape of [N, F, T]
        """
    # for the synthetic data correction = 0.1
    N, Dh, Dw = b.shape
    maxiter, correction, threshold = 500, 0.1, 1e-4  # correction is help to make the loss monotonically decreasing
    d_til, d_old, d = x.view(-1).clone(), x.view(-1).clone(), x.view(-1).clone()
    "coef is the shape of [N, Dw*Dh, Dw*Dh], with block diagnal structure of Dw*Dw small blocks"
    coef_core = (Tsck_core.permute(0, 2, 1) @ Tsck_core.abs())  # shape of [N, Dw, Dw]
    term = (Tsck_core@b.permute(0,2,1)).permute(0,2,1).reshape(N, -1) # shape of [N, FT],permute before reshape is a must

    # loss = torch.cat((torch.tensor([], device=x.device), loss_D(Tsck_t, d, b).reshape(1)))
    for i in range(maxiter):
        d_til = d + correction*Mw*(d - d_old)  # shape of [M]
        nu = d_til - ((coef_core@ d_til.view(Dh,Dw).t()).t().reshape(N,-1) - term).sum(0) * Md_inv  # shape of [Dh * Dw]
        if torch.norm(nu) <= 1:
            d_new = nu
        else:
            d_new = acc_newton(Md, -Md*nu)  # QCQP(P, q)
        d, d_old = d_new, d
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_D(Tsck_t, d, b).reshape(1)))
        if (d - d_old).norm() / d_old.norm() < threshold: break
        if torch.isnan(d).sum() > 0: print('inf_nan_happenned')
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return d



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

