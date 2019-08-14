##
# """This file constains all the necessary classes and functions"""
import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

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


def updateD0():
    pass


def updateD(DD0SS0, X, Y, opts):
    """this function is to update the distinctive D using BPG-M, updating each d_k^(c)
    input is initialed D,
    sparse coeffecient S_bar
        the structure is not a matrix for computation simplexity
        S is 4-d tensor [N,C,T,K] [samples,classes,time series, num of atoms]
        D is 3-d tensor [K,C,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
    training Data X
    training labels Y \in Z ^(N*C)
    training paremeters opts
    output is updated D_bar
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    rn = conv(D0, S0)



def load_data():
    pass
