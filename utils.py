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
    def __init__(self, C=5, K0=10, K=20, miu=0.1, lamb=0.1, delta = 0.9):
        self.C, self.K, self.K0 = C, K, K0
        self.miu, self.lamb , self.delta = miu, lamb, delta
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def bpg_m(x_M_Minv_Mw_gradf, type=0):
    """typ1 =0 means updating dictionary itoms
        type =1 meaning updating the sparse coeff.
        x_M_Minv_Mw_gradf is a list
        """
    x, M, Minv, Mw, gradf = x_M_Minv_Mw_gradf
    maxiter = 500
    if type == 0:
        d_old, d = x, x
        for i in range(500):
            d_til = d + Mw@(d -d_old)
            nu = d_til - Minv@gradf(d_til)
            if torch.norm(nu).item()**2 <=1 :
                d_new = nu
            else:
                d_new = acc_newton(M, -M@nu)  # QCQP(P, q)
            d, d_old = d_new, d
            if torch.norm(d - d_old).item() < 1e-3:
                break
        return d


def acc_newton(P, q):
    """proximal operator for majorized form using Accelorated Newton's method
    follow the solutions of `convex optimization` by boyd, exercise 4.22, solving QCQP
    where P is a square diagonal matrix, q is a vector
    for update D  q =- M_D \nu = -MD@nu, P = M_D = MD
    for update D0, q = -(M_D \nu + \rho z_k + Y_k), P = M_D + \rho I
                    q = MD@nu + rho*Z + Y[:, k] , P = MD + rho*eye(M)
    """
    psi = 0
    dim = P.shape[0]
    maxiter = 200
    for i in range(maxiter):
        f_grad = - 2 * ((P.diag()+psi*torch.eye(dim))**(-3)*q*q).sum()
        f = ((P.diag()+psi*torch.eye(dim))**(-2)*q*q).sum()
        psi_new = psi - 2 * f/f_grad * (f.sqrt() - 1)
        if psi_new.item() - psi.item() < 1e-5:  # psi_new should always larger than psi
            break
        else:
            psi = psi_new.clone()
    return psi_new


def toeplitz(x, m=0):
    """This is a the toepliz matrx for torch.tensor
    input x has the shape of [N, T]
        M is an interger
    output tx has the shape of [N, M, T]
    """
    dev = x.device
    N, T = x.shape
    x_append0 = torch.cat([x, torch.zeros(N, 2*m, device=dev)], dim=1)
    xm = x_append0.repeat(m, 1, 1).permute(1, 0, 2)  # shape of [N, m, T+2m]
    tx = torch.zeros(N, m, T, device=dev)
    m2 = int(m/2)
    for i in range(m):
        ind = range(m2 + i, m2 + i + T)
        tx[:, i, :] = xm[:, i, ind]
    return tx


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
        Tsck = toeplitz(sck)  # shape of [N, M, T]
        abs_Tsck = abs(Tsck)
        Mw = opts.delta * torch.eye(M, device=opts.dev)
        MD_diag = ((abs_Tsck @ abs_Tsck).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()
        MD = MD_diag.diag()
        MD_inu = (1/MD_diag).diag()
        nu = 0

        dck_conv_sck = F.conv1d(sck.unsqeeze(1), dck.reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        DpconvSp = ((1- Y[:, c_prime] - Y[:, c_prime]*Y[:, c].reshape(N, 1)).unsqueeze(2)*DconvS[:, c_prime, :]).sum(1)
        b = (X - R - (DconvS.sum(1) - dck_conv_sck) - (DconvS[:, c, :] - dck_conv_sck) + DpconvSp)/2  # b is shape of [N, T]
        torch.cuda.empty_cache()


def updateD0():
    pass


def load_data():
    pass
