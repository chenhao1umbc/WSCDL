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
        self.mu, self.lamb , self.delta = miu, lamb, delta
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def solv_dck(x, M, Minv, Mw, Tsck, b):
    """x, is the dck, shape of [M]
        M, is MD, with shape of [M, M], diagonal matrix
        Minv, is MD^(-1)
        Mw, is a number not diagonal matrix
        Tsck, is truncated toeplitz matrix of sck with shape of [N, M, T]
        b is bn with all N, with shape of [N, T]
        """
    maxiter = 500
    d_old, d = x, x
    coef = Minv @ (Tsck@Tsck.permute(0, 2, 1)).sum(0)
    term = Minv @ (Tsck@b.unsqueeze(2)).sum(0)
    for i in range(maxiter):
        d_til = d + Mw*(d - d_old)  # Mw is just a number for calc purpose
        nu = d_til - (coef@d_til).squeeze() + term.squeeze()  # nu is 1-d tensor
        if torch.norm(nu).item()**2 <= 1:
            d_new = nu
        else:
            d_new = acc_newton(M, -M@nu)  # QCQP(P, q)
        d, d_old = d_new, d
        if torch.norm(d - d_old).item() < 1e-4:
            break
        torch.cuda.empty_cache()
    return d


def acc_newton(P, q):
    """proximal operator for majorized form using Accelorated Newton's method
    follow the solutions of `convex optimization` by boyd, exercise 4.22, solving QCQP
    where P is a square diagonal matrix, q is a vector
    for update D  q = -M_D \nu = -MD@nu, P = M_D = MD
    for update D0, q = -(M_D \nu + \rho z_k + Y_k), P = M_D + \rho I
                    q = -(MD@nu + rho*Z + Y[:, k]), P = MD + rho*eye(M)
    """
    psi = 0
    dim = P.shape[0]
    maxiter = 200
    for i in range(maxiter):
        f_grad = - 2 * ((P.diag()+psi)**(-3)*q*q).sum()
        f = ((P.diag()+psi)**(-2)*q*q).sum()
        psi_new = psi - 2 * f/f_grad * (f.sqrt() - 1)
        if (psi_new - psi).item() < 1e-5:  # psi_new should always larger than psi
            break
        else:
            psi = psi_new.clone()
    return psi_new


def solv_dck0(x, M, Minv, Mw, Tsck, b, D0, mu):
    """x, is the dck, shape of [M]
        M, is MD, with shape of [M, M], diagonal matrix
        Minv, is MD^(-1)
        Mw, is a number not diagonal matrix
        Tsck, is truncated toeplitz matrix of sck with shape of [N, M, T]
        b is bn with all N, with shape of [N, T]
        """
    maxiter = 500
    d_old, d = x, x
    coef = Minv @ (Tsck@Tsck.permute(0, 2, 1)).sum(0)
    term = Minv @ (Tsck@b.unsqueeze(2)).sum(0)
    for i in range(maxiter):
        d_til = d + Mw*(d - d_old)  # Mw is just a number for calc purpose
        nu = d_til - (coef@d_til).squeeze() + term.squeeze()  # nu is 1-d tensor
        d_new = argmin_lowrank(M, nu, mu, D0)
        d, d_old = d_new, d
        if torch.norm(d - d_old).item() < 1e-4:
            break
        torch.cuda.empty_cache()
    return d


def argmin_lowrank(M, nu, mu, D0):
    """
    Solving the QCQP with low rank panelty term
    :param M: majorizer matrix
    :param nu: make d close to ||d-nu||_M^2
    :param mu: hyper-param of ||D0||_*
    :param D0: common dict contains all the dk0, shape of [K0, M]
    :return: dk0
    """

    return 0


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
    print(F.conv1d(a, b.flip(1).unsqueeze(1), groups=k0))
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
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], D[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
        # D*S, not the D'*S', And here D'*S' will not be updated for each d_c,k update
        torch.cuda.empty_cache()

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tsck = toeplitz(sck)  # shape of [N, M, T]
        abs_Tsck = abs(Tsck)
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD_diag = ((abs_Tsck @ abs_Tsck.permute(0, 2, 1)).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()  # shape of [M]
        MD = MD_diag.diag()
        MD_inv = (1/MD_diag).diag()

        dck_conv_sck = F.conv1d(sck.unsqeeze(1), dck.reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        # the following line is to get the sum_c'(D^(c')*S^(c'))
        DpconvSp = ((1- Y[:, c_prime] - Y[:, c_prime]*Y[:, c].reshape(N, 1)).unsqueeze(2)*DconvS[:, c_prime, :]).sum(1)
        b = (X - R - (DconvS.sum(1) - dck_conv_sck) - (DconvS[:, c, :] - dck_conv_sck) + DpconvSp)/2  # b is shape of [N, T]
        torch.cuda.empty_cache()
        D[c, k, :] = solv_dck(dck, MD, MD_inv, Mw, Tsck, b)
    return D


def updateD0(DD0SS0, X, Y, opts):
    """this function is to update the distinctive D using BPG-M, updating each d_k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, K0, T = S0.shape
    M =D0.shape[1]
    M_2 = int(M/2)  # dictionary atom dimension
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    ycDcconvSc = S[:, :, 0, :].clone()
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], D[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)
    alpha_plus_dk0 = DconvS.sum(1) + R
    beta_plus_dk0 = ycDcconvSc.sum(1) + R
    D0copy = D0.clone()

    # '''update the current dk0'''
    for k0 in range(K0):
        dk0 = D0[k0, :]
        sck0 = S0[:, k0, :]  # shape of [N, T]
        dk0convsck0 = F.conv1d(sck0.unsqueeze(1), dk0.flip().unsqueeze(0).unsqeeze(0), padding=M-1)[:, M_2:M_2 + T]
        Tsck0 = toeplitz(sck0)  # shape of [N, M, T]
        abs_Tsck0 = abs(Tsck0)
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD_diag = 4*((abs_Tsck0@abs_Tsck0.permute(0, 2, 1)).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()  # shape of [M]
        MD = MD_diag.diag()
        MD_inv = (1/MD_diag).diag()
        b = 2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsck0
        torch.cuda.empty_cache()
        D0[k0, :] = solv_dck0(dk0, MD, MD_inv, Mw, 2*Tsck0, b, D0copy, opts.mu)
    return D0
def load_data():
    pass
