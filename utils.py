##
# """This file constains all the necessary classes and functions"""
import os
import time
import torch
import torch.nn.functional as F
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
    def __init__(self, C=5, K0=10, K=20, M=20, mu=0.1, lamb=0.1, delta=0.9, maxiter=500):
        self.C, self.K, self.K0, self.M = C, K, K0, M
        self.mu, self.lamb, self.delta = mu, lamb, delta
        self.maxiter, self.plot = maxiter, False
        self.dataset = 0
        if torch.cuda.is_available():
            self.dev = 'cuda'
            print('\nGPU is available and GPU will be used')
        else:
            self.dev = 'cpu'
            print('\nGPU is not available and CPU will be used')



def init(X, Y, opts):
    """
    This function will generate the initial value for D D0 S S0 and W
    :param X: training data with shape of [N, T]
    :param Y: training labels with shape of [N, C]
    :param opts: an object with hyper-parameters
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]pyth
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
        W is a matrix [C, K], where K is per-class atoms
    :return: D, D0, S, S0, W
    """
    N, T = X.shape
    ind = list(range(N))
    np.random.shuffle(ind)
    D = torch.rand(opts.C, opts.K, opts.M, device=opts.dev)
    D0 = torch.rand(opts.K0, opts.M, device=opts.dev)
    S = torch.rand(N, opts.C, opts.K, T, device=opts.dev)
    S0 = torch.rand(N, opts.K0, T, device=opts.dev)
    W = torch.rand(opts.C, opts.K, device=opts.dev)
    return D, D0, S, S0, W


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
    dck = -((P.diag() + psi_new)**(-1)).diag() @ q
    return dck


def solv_dck(x, M, Minv, Mw, Tsck_t, b):
    """x, is the dck, shape of [M]
        M, is MD, with shape of [M, M], diagonal matrix
        Minv, is MD^(-1)
        Mw, is a number not diagonal matrix
        Tsck_t, is truncated toeplitz matrix of sck with shape of [N, M, T]
        b is bn with all N, with shape of [N, T]
        """
    maxiter = 500
    d_old, d = x.clone(), x.clone()
    coef = Minv @ (Tsck_t@Tsck_t.permute(0, 2, 1)).sum(0)
    term = Minv @ (Tsck_t@b.unsqueeze(2)).sum(0)
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


def solv_dck0(x, M, Minv, Mw, Tsck_t, b, D0, mu, k0):
    """
    :param x: is the dck, shape of [M]
    :param M: is MD, the majorizer matrix with shape of [M, M], diagonal matrix
    :param Minv: is MD^(-1)
    :param Mw: is a number not diagonal matrix
    :param Tsck_t: is truncated toeplitz matrix of sck with shape of [N, M, T], already *2
    :param b: bn with all N, with shape of [N, T]
    :param D0: is the shared dictionary
    :param mu: is the coefficient fo low-rank term
    :param k0: the current index of for loop of K0
    :return: dck0
    """
    maxiter = 500
    d_old, d = x.clone(), x.clone()
    coef = Minv @ (Tsck_t @ Tsck_t.permute(0, 2, 1)).sum(0)
    term = Minv @ (Tsck_t @ b.unsqueeze(2)).sum(0)
    for i in range(maxiter):
        d_til = d + Mw*(d - d_old)  # Mw is just a number for calc purpose
        nu = d_til - (coef@d_til).squeeze() + term.squeeze()  # nu is 1-d tensor
        d_new = argmin_lowrank(M, nu, mu, D0, k0)  # D0 will be changed, because dk0 is in D0
        d, d_old = d_new, d
        if torch.norm(d - d_old) < 1e-4:
            break
        torch.cuda.empty_cache()
    return d


def solv_snk0(x, M, Minv, Mw, Tdk0, b, lamb):
    """
    :param x: is the snk0, shape of [N, T]
    :param M: is MD, the majorizer matrix with shape of [T], diagonal of matrix
    :param Minv: is MD^(-1)
    :param Mw: is a number, not diagonal matrix
    :param Tdk0: is truncated toeplitz matrix of dk0 with shape of [M, T], already *2
    :param b: bn with all N, with shape of [N, T]
    :param lamb: sparsity hyper parameter
    :return: dck0
    """
    maxiter = 500
    snk0_old, snk0 = x.clone(), x.clone()
    coef = Minv @ Tdk0.t() @ Tdk0  # shape of [T, T]
    term = (Minv @ Tdk0.t() @b.t()).t()  # shape of [N, T]
    for i in range(maxiter):
        snk0_til = snk0 + Mw*(snk0 - snk0_old)  # Mw is just a number for calc purpose
        nu = snk0_til - (coef@snk0_til.t()).t() + term  # nu is [N, T]
        snk0_new = svt_s(M, nu, lamb)  # shape of [N, T]
        snk0, snk0_old = snk0_new, snk0
        if torch.norm(snk0 - snk0_old) < 1e-4:
            break
        torch.cuda.empty_cache()
    return snk0


def solv_sck(sc, wc, yc, Tdck, b, k, opts):
    """
    This function solves snck for all N, using BPGM
    :param sc: shape of [N, K, T]
    :param wc: shape of [K]
    :param yc: shape of [N]
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
    :param k: integer, which atom to update
    :return: sck
    """
    delta = opts.delta
    lamb = opts.lamb
    dev = opts.dev
    T = b.shape[1]
    P = torch.ones(T, device=dev)/T
    # 'skc update will lead sc change'
    sck = sc[:, k, :]  # shape of [N, T]
    sck_old = sck.clone()
    PtSncWc = sc.mean(2) @ wc  # shape of [N]
    yc_wkc = yc * wc[k]  #shape of [N]
    wkc = wc[k]  # scaler
    Tdck_t_Tdck = Tdck.t() @ Tdck  # shape of [T, T]
    term1 = (abs(8 * Tdck_t_Tdck)).sum(1)  #shape of [T]
    term2 = wkc ** 2 + yc_wkc * wkc  # scaler
    term3 = yc_wkc * wkc  # shape of [N]
    # M is the diagonal of majorization matrix with shape of [N, T]
    long = abs(term3 / PtSncWc**2 + term2 / (1-PtSncWc)**2)
    M = term1 + long.unsqueeze(1) @ P.unsqueeze(0)  # shape of [N, T]
    M_old = M.clone()

    maxiter = 5  # for test, set it to a small number
    for i in range(maxiter):
        Mw = delta * M**(-0.5) * M_old**0.5
        sck_til = sck + Mw * (sck - sck_old)
        PtSncWc = sc.mean(2) @ wc  # shape of [N]
        long = abs(term3 / PtSncWc**2 + term2 / (1-PtSncWc)**2)
        M_new = term1 + long.unsqueeze(1) @ P.unsqueeze(0)  # shape of [N, T]
        nu = sck_til - (8*Tdck_t_Tdck @ (sck.t()-b.t())).t()/M  # shape of [N, T]
        sck_new = svt_s(M, nu, lamb)  # shape of [N, T]
        sck[:], sck_old[:] = sck_new[:], sck[:]  # make sure sc is updated in each loop
        M, M_old = M_new, M
        if torch.norm(sck - sck_old) < 1e-4:
            break
        torch.cuda.empty_cache()
    return sck


def solv_wc(x, snc, yc, delta):
    """
    This fuction is using bpgm to update wc
    :param x: shape of [K], init value of wc
    :param snc: shape of [N, K, T]
    :param yc: shape of [N]
    :param delta: real number
    :return: wc
    """
    maxiter = 500
    wc_old, wc = x.clone(), x.clone()
    pt_snc = snc.mean(2)  # shape of [N, K]
    abs_pt_snc = abs(pt_snc)
    pt_snc_wc = pt_snc @ wc  # shape of [N]
    M = ((yc / (pt_snc_wc) ** 2 + (1 - yc) / (1 - pt_snc_wc) ** 2).unsqueeze(1) * abs_pt_snc * (
        abs_pt_snc.sum(1).unsqueeze(1))).sum(0)  # shape of [K]
    M_old = M.clone()
    for i in range(maxiter):
        Mw = delta * M**(-1/2) * M_old**(1/2)
        wc_til = wc + Mw*(wc - wc_old)  # Mw is just a number for calc purpose
        pt_snc_wc_til = pt_snc @ wc_til
        nu = wc_til + M**(-1) * ((yc/pt_snc_wc_til + (yc-1)/(1-pt_snc_wc_til)).unsqueeze(1)*pt_snc).sum(0)  # nu is [K]
        wc_new, M_new = gradd(abs_pt_snc, pt_snc, yc, nu, wc.clone())  # gradient descend to get wc
        wc, wc_old = wc_new, wc
        M, M_old = M_new, M
        if torch.norm(wc - wc_old) < 1e-4:
            break
        torch.cuda.empty_cache()
    return wc


def gradd(abs_pt_snc, pt_snc, yc, nu, init_wc):
    """
    This function is meant to solve 1/2||x-\nu||_M^2, where M is a function of x,
    This looks like a convex problem but it is not because M = f(x), x is part of demoninator
    :param abs_pt_snc: abs(snc(mean(2)) constant, shape of [N, K]
    :param pt_snc: snc(mean(2) constant, shape of [N, K]
    :param yc: y_n^(c) constant, shape of [N]
    :param nu: constant, shape of [M]
    :param init_wc: is a clone of wc for the initialization
    :return:
    """
    wc = init_wc.requires_grad_()
    lr = 0.005
    const = abs_pt_snc * abs_pt_snc.sum(1).unsqueeze(1)
    maxiter = 500
    loss = []
    for i in range(maxiter):
        pt_snc_wc = pt_snc @ wc
        M = ((yc / (pt_snc_wc) ** 2 + (1 - yc) / (1 - pt_snc_wc) ** 2).unsqueeze(1) * const).sum(0)  # shape of [K]
        lossfunc = 1/2*((wc-nu) * M * M * (wc-nu)).sum()
        lossfunc.backward()
        loss.append(lossfunc.detach().cpu().item())
        if abs(wc.grad).sum() < 1e-4: break  # stop criteria
        if i > 10 and abs(loss[i]-loss[i-1]) < 1e-4: break  # stop criteria
        with torch.no_grad():
            wc = wc - lr*wc.grad
            wc.requires_grad_()
        torch.cuda.empty_cache()
    return wc.detach().requires_grad_(False), M.detach().requires_grad_(False)


def argmin_lowrank(M, nu, mu, D0, k0):
    """
    Solving the QCQP with low rank panelty term. This function is using ADMM to solve dck0
    :param M: majorizer matrix
    :param nu: make d close to ||d-nu||_M^2
    :param mu: hyper-param of ||D0||_*
    :param D0: common dict contains all the dk0, shape of [K0, M]
    :return: dk0
    """
    K0, m = D0.shape
    rho = 10 * mu  # agrangian coefficients
    dev = D0.device
    Z = torch.eye(K0, m, device=dev)
    Y = torch.eye(K0, m, device=dev)  # lagrangian coefficients
    P = M + rho*torch.eye(m, device=dev)
    Mbynu = M @ nu
    maxiter = 200
    cr = []
    # begin of ADMM
    for i in range(maxiter):
        Z = svt(D0-1/rho*Y, mu/rho)
        q = -(Mbynu + rho * Z[k0, :] + Y[k0, :])
        dk0 = D0[k0, :] = acc_newton(P, q)
        cr.append(Z - D0)
        Y = Y + rho*cr[i]
        if torch.norm(cr[i]) < 1e-4 : break
        if i > 10:  # if not going anywhere
            if abs(cr[i] - cr[i-10]).sum() < 5e-5: break
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
    L = L.cpu()
    l, h = L.shape
    try:
        u, s, v = torch.svd(L)  ########## so far in version 1.2 the torch.svd for GPU could be much slower than CPU
    except:                     ########## and torch.svd may have convergence issues for GPU and CPU.
        u, s, v = torch.svd(L + 1e-3*L.mean()*torch.rand(l, h))
        print('unstable svd happened')
    s = s - tau
    s[s<0] = 0
    P = u @ s.diag() @ v.t()
    return P.to(dev)


def svt_s(M, nu, lamb):
    """
    This function is to implement the signular value thresholding, solving the following
    min_p lamb||p||_1 + 1/2||\nu-p||_M^2, p is a vector
    :param M: is used for matrix norm, shape of [N, T]
    :param lamb: is coefficient of L-1 norm, scaler
    :param nu: the the matrix to be pruned, shape of [N, T]
    :return: P the matrix after singular value thresholding
    """
    b = lamb / M  # shape of [N, T]
    P = torch.sign(nu) * F.relu(abs(nu) -b)
    return P


def toeplitz(x, m=10, T=10):
    """This is a the toepliz matrx for torch.tensor
    input x has the shape of [N, ?], ? is M or T
        M is an interger
        T is truncated length
    output tx has the shape of [N, m, T]
    """
    dev = x.device
    N, m0 = x.shape  # m0 is T for Tsck, and m0 is M for Tdck
    M = m if m < m0 else m0
    M2 = int((M - 1) / 2) + 1  # half length of M, for truncation purpose
    x_append0 = torch.cat([torch.zeros(N, m, device=dev), x, torch.zeros(N, m, device=dev)], dim=1)
    xm = x_append0.repeat(m, 1, 1).permute(1, 0, 2)  # shape of [N, m, ?+2m]
    tx = torch.zeros(N, m, T, device=dev)
    for i in range(m):
        ind = range(M2 + i, M2 + i + T)
        tx[:, i, :] = xm[:, i, ind]
    return tx.flip(1)


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
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M-1).sum(1)[:, M_2:M_2+T]  # r is shape of [N, T)
    C, K, _ = D.shape
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    Crange = torch.tensor(range(C))
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
        # D*S, not the D'*S', And here D'*S' will not be updated for each d_c,k update
        torch.cuda.empty_cache()

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tsck_t = toeplitz(sck, M, T)  # shape of [N, M, T],
        abs_Tsck_t = abs(Tsck_t)
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD_diag = ((abs_Tsck_t @ abs_Tsck_t.permute(0, 2, 1)).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()  # shape of [M]
        MD = MD_diag.diag()
        MD_inv = (1/MD_diag).diag()

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        # the following line is to get the sum_c'(D^(c')*S^(c'))
        DpconvSp = ((1- Y[:, c_prime] - Y[:, c_prime]*Y[:, c].reshape(N, 1)).unsqueeze(2)*DconvS[:, c_prime, :]).sum(1)
        b = (X - R - (DconvS.sum(1) - dck_conv_sck) - (DconvS[:, c, :] - dck_conv_sck) + DpconvSp)/2  # b is shape of [N, T]
        torch.cuda.empty_cache()
        D[c, k, :] = solv_dck(dck, MD, MD_inv, Mw, Tsck_t, b)
    return D


def updateD0(DD0SS0, X, Y, opts):
    """this function is to update the common dictionary D0 using BPG-M, updating each d_k^(0)
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
    C, K, _ = D.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    ycDcconvSc = S[:, :, 0, :].clone()
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)
    alpha_plus_dk0 = DconvS.sum(1) + R
    beta_plus_dk0 = ycDcconvSc.sum(1) + R
    D0copy = D0.clone()

    # '''update the current dk0'''
    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, T]
        dk0convsnk0 = F.conv1d(snk0.unsqueeze(1), dk0.flip(0).unsqueeze(0).unsqueeze(0), padding=M-1)[:, 0,  M_2:M_2 + T]
        Tsnk0_t = toeplitz(snk0, M, T)  # shape of [N, M, T]
        abs_Tsnk0_t = abs(Tsnk0_t)
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD_diag = 4*((abs_Tsnk0_t @ abs_Tsnk0_t.permute(0, 2, 1)).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()  # shape of [M]
        MD = MD_diag.diag()
        MD_inv = (1/MD_diag).diag()
        b = 2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsnk0
        torch.cuda.empty_cache()
        D0[k0, :] = solv_dck0(dk0, MD, MD_inv, Mw, 2*Tsnk0_t, b, D0copy, opts.mu, k0)
    return D0


def updateS0(DD0SS0, X, Y, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
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
    C, K, _ = D.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    ycDcconvSc = S[:, :, 0, :].clone()
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)
    alpha_plus_dk0 = DconvS.sum(1) + R
    beta_plus_dk0 = ycDcconvSc.sum(1) + R

    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, T]
        dk0convsck0 = F.conv1d(snk0.unsqueeze(1), dk0.flip(0).unsqueeze(0).unsqueeze(0), padding=M-1)[:, 0, M_2:M_2 + T]
        Tdk0_t = toeplitz(dk0.unsqueeze(0), m=T, T=T).squeeze()  # in shape of [m=T, T]
        abs_Tdk0 = abs(Tdk0_t).t()
        MS0_diag = (4*abs_Tdk0.t() @ abs_Tdk0).sum(1)  # in the shape of [T]
        MS0_inv = (1/MS0_diag).diag()
        b = 2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsck0
        torch.cuda.empty_cache()
        S0[:, k0, :] = solv_snk0(snk0, MS0_diag, MS0_inv, opts.delta, 2*Tdk0_t.t(), b, opts.lamb)
    return S0


def updateS(DD0SS0W, X, Y, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        W is a matrix [C, K], where K is per-class atoms
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0, W = DD0SS0W  # where DD0SS0 is a list
    N, K0, T = S0.shape
    M = D0.shape[1]  # dictionary atom dimension
    M_2 = int((M-1)/2)
    C, K, _ = D.shape
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    Crange = torch.tensor(range(C))
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    # '''update the current s_n,k^(c) '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        wc = W[c, :]  # shape of [K]
        yc = Y[:, c]  # shape of [N]
        Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        # the following line is to get the sum_c'(D^(c')*S^(c'))
        DpconvSp = ((1- Y[:, c_prime] - Y[:, c_prime]*Y[:, c].reshape(N, 1)).unsqueeze(2)*DconvS[:, c_prime, :]).sum(1)
        b = (X - R - (DconvS.sum(1) - dck_conv_sck) - (DconvS[:, c, :] - dck_conv_sck) + DpconvSp)/2  # b is shape of [N, T]
        torch.cuda.empty_cache()
        sc = S[:, c, :, :].clone()  # sc will be changed in solv_sck, so giving a clone here
        S[:, c, k, :] = solv_sck(sc, wc, yc, Tdck, b, k, opts)
    return S


def updateW(SW, Y, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        SW is a list of [S, W]
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        W is a matrix [C, K], where K is per-class atoms
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    S, W = SW
    N, C, K, T = S.shape
    for c in range(C):
        W[c, :] = solv_wc(W[c, :].clone(), S[:, c, :, :], Y[:, c], opts.delta)
    return W


def znorm(x):
    """
    This function will make the data with zero-mean, variance = 1
    :param x: input tensor with shape of [N, T]
    :return: x_z
    """
    if x.dim() == 1:
        x_z = (x-x.mean())/x.var().sqrt()
    else:
        x_z = (x-x.mean(1))/x.var(1).sqrt()
    return x_z


def load_data(opts):
    """
    So far Aug. 24. 2019, we are loading the synthetic data to see if it works, the time series length is 500
    there are 4 classes, with its mixtures but without overlapping. the features are sin waves and rectangular waves,
    500 samples will be cut into 10 fragments, each fragment contains 30 samples long feature with a random start posion from 0 to 20.
    :param opts:
    :return:
    """
    '''The common features'''
    x = torch.arange(30).float()
    feature0 = torch.sin(x*2*np.pi/30)
    feature1 = torch.sin(x * 2 * np.pi / 15) + torch.sin(x * 2 * np.pi / 10)
    feature2 = torch.sin(x * 2 * np.pi / 20) + torch.sin(x * 2 * np.pi / 5) + torch.sin(x * 2 * np.pi / 8)
    feature3 = torch.zeros(30)
    feature3[np.r_[np.arange(5), np.arange(10, 15), np.arange(20, 25)]] = 1
    feature3 = feature3 + torch.sin(x * 2 * np.pi / 13)
    feature4 = torch.zeros(30)
    feature4[np.r_[np.arange(10), np.arange(20, 30)]] = 1
    feature4 = feature4 + torch.sin(x * np.pi / 6)
    start_point = torch.randint(0, 21, (1,))
    idx_feat = torch.randint(0, 2, (1,))

    X, Y = 0, 0

    return X, Y



