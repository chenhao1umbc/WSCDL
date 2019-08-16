##
# """This file constains all the necessary classes and functions"""
import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
# from scipy.linalg import toeplitz   # This is too slow

np.set_printoptions(linewidth=180)
torch.set_printoptions(linewidth=180)
torch.backends.cudnn.deterministic = True

##
class OPT:
    """initial c the number of classes, k0 the size of shared dictionary atoms
    miu is the coeff of low-rank term, lamb is the coeff of sparsity """
    def __init__(self, C=5, K0=10, K=20, miu=0.1, lamb=0.1,):
        self.C, self.K, self.K0 = C, K, K0
        self.miu, self.lamb = miu, lamb


def bpg_m( x, M, Mw, gradf, type=0):
    """typ1 =0 means updating dictionary itoms
        type =1 meaning updating the sparse coeff. """
    maxiter = 500
    if type == 0:
        d_old, d = x, x
        for i in range(500):
            d_til = d + Mw@(d -d_old)
            nu = d_til - np.linalg.pinv(M)@gradf(d_til)
            if norm(nu)**2 <=1 :
                d_new = nu
            else:
                d_new = prox_d(M, -M@nu) # QCQP(P, q)
            d, d_old = d_new, d
            if norm(d - d_old) < 1e-3:
                break
        return d


def prox_d(P, q):
    """proximal operator for majorized form using Accelorated Newton's method"""
    psi = 0
    dim = P.shape[0]
    maxiter = 200
    for i in range(maxiter):
        f_stroke = - 2 * np.sum(q * q * np.diag(P+psi*np.eye(dim))**(-3))
        f = np.sum(q * q * np.diag(P+psi*np.eye(dim))**(-2))
        psi_new = psi - 2 * f/f_stroke * (np.sqrt(f) - 1)
        if psi_new - psi < 1e-5: # psi_new should always larger than psi
            break
        else:
            psi = psi_new
    return psi_new


def conv(a, b):
    """an, bn are in the shape (N, Ta), (N,Tb), where N is batch_size, T is sequence length,
    an, and bn are all torch.tensor
    """
    pad = a.shape[1] if a.shape[1] < b.shape[1] else b.shape[1]
    rab = torch.nn.functional.conv1d(a.unsqueeze(1), b.flip(1).unsqueeze(1), padding=pad-1)
    return rab.squeeze()


def fconv(an, bn):
    """an, bn are numpy arrays, using fourier transform
    an \in R^T*N, bn \in R^M*N, T>M,
    using fft to do the 1-d convlution and trucate to the lenght T"""
    ifft = np.fft.ifft
    fft = np.fft.fft
    zeros = np.zeros
    l1, l2 = an.shape[0], bn.shape[0]
    if l1 < l2:
        l1, l2 = l2, l1 # make sure an is longer than bn
    l = l1 + l2
    try:
        col = an.shape[1]
    except IndexError:
        col = 1
    cn = zeros([l1, col])
    start_point = l2 // 2
    if col == 1:
        an, bn = np.squeeze(an), np.squeeze(bn)
        cn = ifft(fft(an, l) * fft(bn, l))[start_point:l1+start_point]
    else:
        for i in range(col):
            cn[:, i] = ifft(fft(an[:, i], l) * fft(bn[:, i], l))[start_point:l1+start_point]
    return cn


def toepliz(x, m=0):
    """This is a the toepliz matrx for torch.tensor
    input x has the shape of [N, T]
        M is an interger
    output tx has the shape of [N, M, T]
    """
    for i in range(x.shape[0]):
        for ii in range(m):

            pass


def updateD(DD0SS0, X, Y, opts):
    """this function is to update the distinctive D using BPG-M, updating each d_k^(c)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels

    'for better understanding the code please try:'
    a = torch.rand(3, 4, 12)  # S0
    b = torch.rand(4, 5)  # D0
    k0, m = b.shape
    s0 = a.unsqueeze(1)  # expand dimension for conv1d
    d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    for i in range(k0):
        print(F.conv1d(s0[:, :, i, :], d0[:, :, i, :]))
    'compare wth'
    print(F.conv1d(a, b.flip(1).unsqueeze(1), groups=4))
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, K0, T = S0.shape
    M =D0.shape[1]
    M_2 = int(M/2)  # dictionary atom dimension
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M-1).sum(1)[:, M_2:M_2+T]  # r is shape of [N, T)
    C, K, _ = D.shape
    D = D.flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    Crange = torch.tensor(range(C))
    for c in range(C):
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], D[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
        # D*S, not the D'*S', And here D'*S' will not be updated for each d_k,c update
        torch.cuda.empty_cache()

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tsck = toeplitz()
        dck_conv_sck = F.conv1d(sck.unsqeeze(1), dck.reshape(1,1,M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        DpconvSp = ((1- Y[:, c_prime] - Y[:, c_prime]*Y[:, c].reshape(N, 1)).unsqueeze(2)*DconvS[:, c_prime, :]).sum(1)
        b = (X - R - (DconvS.sum(1) - dck_conv_sck) - (DconvS[:, c, :] - dck_conv_sck) + DpconvSp)/2  # b is bn with all N
        torch.cuda.empty_cache()


def updateD0():
    pass


def load_data():
    pass
