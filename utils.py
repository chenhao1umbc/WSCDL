##
# """This file constains all the necessary classes and functions"""
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
import scipy.sparse as sparse

import seaborn as sns

tt = datetime.datetime.now
# torch.set_default_dtype(torch.double)
np.set_printoptions(linewidth=160)
torch.set_printoptions(linewidth=160)
torch.backends.cudnn.deterministic = True
seed = 10
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
        self.savefig = False # save plots
        self.shape = '1d' # input data is 1d or 2d, 1d could be vectorized 2d data
        if torch.cuda.is_available():
            self.dev = 'cuda'
            if not silent: print('\nRunning on GPU')
        else:
            self.dev = 'cpu'
            if not silent: print('\nRunning on CPU')


def init(X, opts):
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
    # D = l2norm(awgn(torch.cat(opts.ft[1:]).reshape(opts.C, opts.K, opts.M), -20)).to(opts.dev)
    # D0 = l2norm(awgn(opts.ft[0].reshape(opts.K0, opts.M), -20)).to(opts.dev)
    D = l2norm(torch.rand(opts.C, opts.K, opts.M, device=opts.dev))
    D0 = l2norm(torch.rand(opts.K0, opts.M, device=opts.dev))
    S = torch.zeros(N, opts.C, opts.K, T, device=opts.dev)
    S0 = torch.zeros(N, opts.K0, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K +1, device=opts.dev)

    return D, D0, S, S0, W


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


def solv_dck(x, Md, Md_inv, Mw, Tsck_t, b):
    """x, is the dck, shape of [M]
        M is with shape of [M], diagonal of the majorized matrix
        Minv, is Md^(-1), shape of [M]
        Mw, is a number, == opts.delta
        Tsck_t, is truncated toeplitz matrix of sck with shape of [N, M, T]
        b is bn with all N, with shape of [N, T]
        """
    # for the synthetic data correction = 0.1
    maxiter, correction, threshold = 500, 0.1, 1e-4  # correction is help to make the loss monotonically decreasing
    d_til, d_old, d = x.clone(), x.clone(), x.clone()
    coef = Tsck_t@Tsck_t.permute(0, 2, 1)  # shaoe of [N, M, M]
    term = (Tsck_t@b.unsqueeze(2)).squeeze()  # shape of [N, M]

    # loss = torch.cat((torch.tensor([], device=x.device), loss_D(Tsck_t, d, b).reshape(1)))
    for i in range(maxiter):
        d_til = d + correction*Mw*(d - d_old)  # shape of [M]
        nu = d_til - (coef@d_til - term).sum(0) * Md_inv  # shape of [M]
        if torch.norm(nu) <= 1:
            d_new = nu
        else:
            d_new = acc_newton(Md, -Md*nu)  # QCQP(P, q)
        d, d_old = d_new, d
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_D(Tsck_t, d, b).reshape(1)))
        if (d - d_old).norm() / d_old.norm() < threshold: break
        if torch.isnan(d).sum() > 0: print(inf_nan_happenned)
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return d


def solv_dck0(x, M, Minv, Mw, Tsck0_t, b, D0, mu, k0):
    """
    :param x: is the dck, shape of [M]
    :param M: is MD, the majorizer matrix with shape of [M], diagonal matrix
    :param Minv: is MD^(-1), shape of [M]
    :param Mw: is a number not diagonal matrix
    :param Tsck0_t: is truncated toeplitz matrix of sck with shape of [N, M, T], already *2
    :param b: bn with all N, with shape of [N, T]
    :param D0: is the shared dictionary
    :param mu: is the coefficient fo low-rank term, mu = N*mu
    :param k0: the current index of for loop of K0
    :return: dck0
    """
    # for the synthetic data correction = 0.1
    maxiter, correction, threshold = 500, 0.1, 5e-4  # correction is help to make the loss monotonically decreasing
    d_til, d_old, d = x.clone(), x.clone(), x.clone()
    coef = Tsck0_t@Tsck0_t.permute(0, 2, 1)  # shaoe of [N, M, M]
    term = (Tsck0_t@b.unsqueeze(2)).squeeze()  # shape of [N, M]

    # loss = torch.cat((torch.tensor([], device=x.device), loss_D0(Tsck0_t, d, b, D0, mu).reshape(1)))
    for i in range(maxiter):
        d_til = d + correction*Mw*(d - d_old)  # shape of [M],  Mw is just a number for calc purpose
        nu = d_til - (coef@d_til - term).sum(0) * Minv  # shape of [M]
        d_new = argmin_lowrank(M, nu, mu, D0, k0)  # D0 will be changed, because dk0 is in D0
        d, d_old = d_new, d
        if (d - d_old).norm()/d_old.norm() < threshold:break
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_D0(Tsck0_t, d, b, D0, mu).reshape(1)))
    # ll = loss[:-1] - loss[1:]
    # if ll[ll<0].shape[0] > 0: print(something_wrong)
    # plt.figure(); plt.plot(loss.cpu().numpy(), '-x')
    return d


def argmin_lowrank(M, nu, mu, D0, k0):
    """
    Solving the QCQP with low rank panelty term. This function is using ADMM to solve dck0
    :param M: majorizer matrix
    :param nu: make d close to ||d-nu||_M^2
    :param mu: hyper-param of ||D0||_*
    :param D0: common dict contains all the dk0, shape of [K0, M]
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
    return snk0


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
    T = b.shape[1]
    P = torch.ones(1, T, device=dev)/T  # shape of [1, T]
    # 'skc update will lead sc change'
    sck = sc[:, k, :]  # shape of [N, T]
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
    sc_old = sc.clone(); marker = 0

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
    return sck_old


def solv_sck_test(sc, Tdck, b, k, opts):
    """
    This function solves snck for all N, using BPGM
    :param sc: shape of [N, K, T]
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
    :param k: integer, which atom to update
    :return: sck
    """
    maxiter, correction, threshold = 500, 0.7, 1e-5
    Mw = opts.delta * correction # correction is help to make the loss monotonically decreasing
    lamb, lamb2 = opts.lamb, opts.lamb2
    dev = opts.dev
    T = b.shape[1]
    P = torch.ones(1, T, device=dev)/T  # shape of [1, T]
    # 'skc update will lead sc change'
    sck = sc[:, k, :].clone()  # shape of [N, T]
    sck_old = sck.clone()
    abs_Tdck = abs(Tdck)
    Tdck_t_Tdck = Tdck.t() @ Tdck  # shape of [T, T]
    Tdckt_bt = Tdck.t() @ b.t()  # shape of [T, N]
    M = (abs_Tdck.t() @ abs_Tdck + lamb2*torch.eye(T, device=dev)  + 1e-38).sum(1)  # M is the diagonal of majorization matrix, shape of [T]
    sc_til, sc_old, marker = sc.clone(), sc.clone(), 0 # shape of [N, K, T]

    loss = torch.cat((torch.tensor([], device=opts.dev), loss_Sck_test(Tdck, b, sc, sck, opts).reshape(1)))
    for i in range(maxiter):
        sck_til = sck + Mw * (sck - sck_old)  # shape of [N, T]
        sc_til[:, k, :] = sck_til
        nu = sck_til - (Tdck_t_Tdck@sck_til.t() - Tdckt_bt + lamb2 *sck_til.t()).t()/M  # shape of [N, T]
        sck_new = shrink(M, nu, lamb/2)  # shape of [N, T]
        sck_old[:], sck[:] = sck[:], sck_new[:]  # make sure sc is updated in each loop
        if torch.norm(sck - sck_old) / (sck.norm() + 1e-38) < threshold: break
        loss = torch.cat((loss, loss_Sck_test(Tdck, b, sc, sck, opts).reshape(1)))
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
    return sck_old


def loss_Sck(Tdck, b, sc, sck, wc, wkc, yc, opts):
    """
    This function calculates the loss func of sck
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
    :param sc: shape of [N, K, T]
    :param sck: shape of [N, T]
    :param wc: shape of [K]
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
    This function calculates the loss func of sck
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
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


def loss_Sck_test(Tdck, b, sc, sck, opts):
    """
    This function calculates the loss func of sck
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
    :param sc: shape of [N, K, T]
    :param sck: shape of [N, T]
    :param opts: for hyper parameters
    :return:
    """
    term1 = (Tdck@sck.t() -b.t()).norm()**2
    term2 = opts.lamb * sck.abs().sum()
    loss = term1 + term2
    return loss


def loss_Sck_test_spec(Tdck, b, sc, sck, opts):
    """
    This function calculates the loss func of sck
    :param Tdck: shape of [T, m=T]
    :param b: shape of [N, T]
    :param sc: shape of [N, K, T]
    :param sck: shape of [N, T]
    :param opts: for hyper parameters
    :return:
    """
    term1 = (Tdck@sck.t() -b.t()).norm()**2
    term2 = opts.lamb * sck.abs().sum()
    loss = term1 + term2
    return term1, term2


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
    M_old = M.clone()
    # print('before bpgm wc loss is : %1.3e' %loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)))

    # loss = torch.cat((torch.tensor([], device=x.device), loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)).reshape(1)))
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
    tx = torch.zeros(N, m, T, device=dev)
    indx = torch.zeros(m, T).long()
    for i in range(m):
        indx[i, :] = torch.arange(M2 + i, M2 + i + T)
    tx[:, :, :] = x_append0[:, indx]
    return tx.flip(1)


def updateD(DD0SS0W, X, Y, opts):
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
    D, D0, S, S0, W = DD0SS0W  # where DD0SS0 is a list
    N, K0, T = S0.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M-1).sum(1)[:, M_2:M_2+T]  # r is shape of [N, T)
    C, K, _ = D.shape

    # '''update the current d_c,k '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
        DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
        Crange = torch.tensor(range(C))
        for cc in range(C):
            # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
            DconvS[:, cc, :] = F.conv1d(S[:, cc, :, :], Dcopy[cc, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
            # D*S, not the D'*S', And here D'*S' will not be updated for each d_c,k update
            torch.cuda.empty_cache()

        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tsck_t = toeplitz(sck, M, T)  # shape of [N, M, T]
        abs_Tsck_t = abs(Tsck_t)
        Md = (abs_Tsck_t @ abs_Tsck_t.permute(0, 2, 1) @ torch.ones(M, device=opts.dev)).sum(0) + 1e-38 # shape of [M]
        if Md.sum() == 0: continue  # Sck is too sparse with all 0s
        Md_inv = (Md +1e-38)**(-1)

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.flip(0).reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        Dcp_conv_Sncp = DconvS[:, c, :] - dck_conv_sck
        # term 1, 2, 3 should be in the shape of [N, T]
        term1 = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        term2 = Y[:, c].reshape(N,1)*(X - R - Y[:, c].reshape(N,1)*Dcp_conv_Sncp
                - (Y[:, c_prime]*DconvS[:, c_prime, :].permute(2,0,1)).sum(2).t())
        term3 = -(1-Y[:, c]).reshape(N,1)*((1-Y[:, c]).reshape(N,1)*Dcp_conv_Sncp
                + ((1-Y[:, c_prime])*DconvS[:, c_prime, :].permute(2,0,1)).sum(2).t())
        b = (term1 + term2 + term3)/2
        torch.cuda.empty_cache()
        D[c, k, :] = solv_dck(dck, Md, Md_inv, opts.delta, Tsck_t, b)
        if torch.isinf(D).sum() > 0: print(inf_nan_happenned)
    return D


def loss_D(Tsck_t, dck, b):
    """
    calculate the loss function value for updating D, sum( norm(Tsnck*dck - bn)**2 ) , s.t. norm(dck) <=1
    :param Tsck_t: shape of [N, M, T],
    :param dck: cth, kth, atom of D, shape of [M]
    :param b: the definiation is long in the algorithm, shape of [N, T]
    :return: loss fucntion value
    """
    return 2*((Tsck_t.permute(0, 2, 1)@dck - b)**2 ).sum()


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
    # D0copy = D0.clone()  # should not use copy/clone

    # '''update the current dk0'''
    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, T]
        dk0convsnk0 = F.conv1d(snk0.unsqueeze(1), dk0.flip(0).unsqueeze(0).unsqueeze(0), padding=M-1)[:, 0,  M_2:M_2 + T]
        Tsnk0_t = toeplitz(snk0, M, T)  # shape of [N, M, T]
        abs_Tsnk0_t = abs(Tsnk0_t)   # shape of [N, M, T]
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD = 4*(abs_Tsnk0_t @ abs_Tsnk0_t.permute(0, 2, 1) @ torch.ones(M, device=opts.dev)).sum(0) + 1e-38 # shape of [M]
        if MD.sum() == 0 : continue
        MD_inv = 1/(MD)  #shape of [M]
        b = 2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsnk0
        torch.cuda.empty_cache()
        # print('D0 loss function value before update is %3.2e:' %loss_D0(2*Tsnk0_t, dk0, b, D0, opts.mu*N))
        D0[k0, :] = solv_dck0(dk0, MD, MD_inv, Mw, 2*Tsnk0_t, b, D0, opts.mu*N, k0)
        # print('D0 loss function value after update is %3.2e:' % loss_D0(2*Tsnk0_t, dk0, b, D0, opts.mu*N))
        if torch.isnan(D0).sum() + torch.isinf(D0).sum() > 0: print(inf_nan_happenned)
    return D0


def loss_D0(Tsnk0_t, dk0, b, D0, mu):
    """
    calculate the loss function value for updating D, sum( norm(Tsnck*dck - bn)**2 ) , s.t. norm(dck) <=1
    :param Tsnk0_t: shape of [N, M, T],
    :param dk0: kth atom of D0, shape of [M]
    :param b: the definiation is long in the algorithm, shape of [N, T]
    :param D0: supposed to be low_rank
    :param mu: mu = mu*N
    :return: loss fucntion value
    """
    return (((Tsnk0_t.permute(0, 2, 1)@dk0 - b)**2 ).sum()/2 + mu*D0.norm(p='nuc'))


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
        MS0_diag = MS0_diag + 1e-38 # make it robust for inverse
        MS0_inv = (1/MS0_diag).diag()
        b = 2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsck0
        torch.cuda.empty_cache()
        # print(loss_S0(2*Tdk0_t.t(), snk0, b, opts.lamb))
        S0[:, k0, :] = solv_snk0(snk0, MS0_diag, MS0_inv, opts.delta, 2*Tdk0_t.t(), b, opts.lamb)
        # print(loss_S0(2*Tdk0_t.t(), S0[:, k0, :], b, opts.lamb))
    return S0


def updateS0_test(DD0SS0, X, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y is not given
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, K0, T = S0.shape
    C, K, _ = D.shape
    M, dev = D0.shape[1], D.device
    M_2 = int((M-1)/2)  # dictionary atom dimension
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
    dconvs = DconvS.sum(1)

    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, T]
        R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:,M_2:M_2 + T]  # r is shape of [N, T)
        dk0convsck0 = F.conv1d(snk0.unsqueeze(1), dk0.flip(0).unsqueeze(0).unsqueeze(0), padding=M-1)[:, 0, M_2:M_2 + T]
        Tdk0_t = toeplitz(dk0.unsqueeze(0), m=T, T=T).squeeze()  # in shape of [m=T, T]
        abs_Tdk0 = abs(Tdk0_t).t()
        MS0_diag = (abs_Tdk0.t() @ abs_Tdk0).sum(1)  # in the shape of [T]
        # MS0_diag = MS0_diag + 1e-38 # make it robust for inverse
        MS0_diag = MS0_diag + 1e-38 + opts.lamb2*torch.eye(T, device=dev)  # make it robust for inverse
        MS0_inv = (1/MS0_diag).diag()
        b = X - dconvs - R + dk0convsck0
        torch.cuda.empty_cache()
        # print(loss_S0(2*Tdk0_t.t(), snk0, b, opts.lamb))
        # S0[:, k0, :] = solv_snk0(snk0, MS0_diag, MS0_inv, opts.delta, Tdk0_t.t(), b, opts.lamb/2)
        S0[:, k0, :] = solv_sck_test(S0, Tdk0_t.t(), b, k0, opts)
        # print(loss_S0(2*Tdk0_t.t(), S0[:, k0, :], b, opts.lamb))
    return S0


def updateS0_test_fista(DD0SS0, X, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y is not given
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, K0, T = S0.shape
    C, K, _ = D.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M-1).sum(1)[:, M_2:M_2+T]
    dconvs = DconvS.sum(1)

    for k0 in range(K0):
        dk0 = D0[k0, :]
        snk0 = S0[:, k0, :]  # shape of [N, T]
        R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:,M_2:M_2 + T]  # r is shape of [N, T)
        dk0convsck0 = F.conv1d(snk0.unsqueeze(1), dk0.flip(0).unsqueeze(0).unsqueeze(0), padding=M-1)[:, 0, M_2:M_2 + T]
        Tdk0_t = toeplitz(dk0.unsqueeze(0), m=T, T=T).squeeze()  # in shape of [m=T, T]
        b = X - dconvs - R + dk0convsck0
        torch.cuda.empty_cache()

        bb = torch.cat((b, torch.zeros(N, T, device=b.device)), 1)
        TT = torch.cat((Tdk0_t.t(), opts.lamb2 ** 0.5 * torch.eye(T, device=b.device)), 0)
        S0[:, k0, :] = fista(bb, TT, snk0, opts.lamb)

        # alpha = spams.lasso(np.asfortranarray(b.t().cpu().numpy()), D=np.asfortranarray(Tdk0_t.t().cpu().numpy()), lambda1=opts.lamb/2, lambda2=opts.lamb2)
        # a = sparse.csc_matrix.todense(alpha)
        # S0[:, k0, :] = torch.tensor(np.asarray(a).T, device=S.device)
    return S0


def loss_S0(_2Tdk0, snk0, b, lamb):
    """
    This function calculates the sub-loss function for S0
    :param _2Tdk0: shape of [T, T]
    :param snk0: shape of [N, T]
    :param b: shape of [N, T]
    :param lamb: scaler
    :return: loss
    """
    return ((_2Tdk0 @ snk0.t() - b.t())**2).sum()/2 + lamb * abs(snk0).sum()


def updateS(DD0SS0W, X, Y, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        W is a matrix [C, K+1], where K is per-class atoms
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels

        adapting for 2-D convolution, T could be replace by [H, W]
        M could be replace by [M, M], which is a square patch
    """
    D, D0, S, S0, W = DD0SS0W  # where DD0SS0 is a list
    N, K0, *T = S0.shape
    M = D0.shape[1]  # dictionary atom dimension
    M_2 = int((M-1)/2)
    C, K, _ = D.shape
    T = T[0]   # *T will make T as a list
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M-1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)
    # '''update the current s_n,k^(c) '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
        Crange = torch.tensor(range(C))
        DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
        for cc in range(C):
            # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
            DconvS[:, cc, :] = F.conv1d(S[:, cc, :, :], Dcopy[cc, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]

        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        wc = W[c, :]  # shape of [K+1], including bias
        yc = Y[:, c]  # shape of [N]
        Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.flip(0).reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        Dcp_conv_Sncp = DconvS[:, c, :] - dck_conv_sck
        # term 1, 2, 3 should be in the shape of [N, T]
        term1 = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        term2 = Y[:, c].reshape(N,1)*(X - R - Y[:, c].reshape(N,1)*Dcp_conv_Sncp - (Y[:, c_prime]*DconvS[:, c_prime, :].permute(2,0,1)).sum(2).t())
        term3 = -(1-Y[:, c]).reshape(N,1)*((1-Y[:, c]).reshape(N,1)*Dcp_conv_Sncp
                + ((1-Y[:, c_prime])*DconvS[:, c_prime, :].permute(2,0,1)).sum(2).t())
        b = (term1 + term2 + term3)/2
        torch.cuda.empty_cache()
        sc = S[:, c, :, :].clone() # sc will be changed in solv_sck, adding clone to prevent, for debugging
        # l00 = loss_fun(X, Y, D, D0, S, S0, W, opts)
        # l0 = loss_fun_special(X, Y, D, D0, S, S0, W, opts)
        # l1 = loss_Sck_special(Tdck, b, sc, sck, wc, wc[k], yc, opts)
        S[:, c, k, :] = solv_sck(sc, wc, yc, Tdck, b, k, opts)
        # ll0 = loss_fun_special(X, Y, D, D0, S, S0, W, opts)
        # ll1 = loss_Sck_special(Tdck, b, sc, sck, wc, wc[k], yc, opts)
        # print('Overall loss for fidelity, sparse, label, differences: %1.7f, %1.7f, %1.7f' %(l0[0]-ll0[0], l0[1]-ll0[1], l0[2]-ll0[2]))
        # print('Local loss for fidelity, sparse, label, differences: %1.7f, %1.7f, %1.7f' % (l1[0]-ll1[0], l1[1]-ll1[1], l1[2]-ll1[2]))
        # print('Main loss after bpgm the diff is: %1.9e' %(l00 - loss_fun(X, Y, D, D0, S, S0, W, opts)))
        # if (l00 - loss_fun(X, Y, D, D0, S, S0, W, opts)) <0 : print(bug)
        if torch.isnan(S).sum() + torch.isinf(S).sum() >0 : print(inf_nan_happenned)
    return S


def updateS_test(DD0SS0, X, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y are the labels, not given
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, K0, T = S0.shape
    M = D0.shape[1]  # dictionary atom dimension
    M_2 = int((M-1)/2)
    C, K, _ = D.shape
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)
    # '''update the current s_n,k^(c) '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
        Crange = torch.tensor(range(C))
        DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
        for cc in range(C):
            # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
            DconvS[:, cc, :] = F.conv1d(S[:, cc, :, :], Dcopy[cc, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]

        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.flip(0).reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        # term 1, 2, 3 should be in the shape of [N, T]
        b = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        torch.cuda.empty_cache()
        sc = S[:, c, :, :] # sc will be changed in solv_sck, adding clone to prevent
        # l = loss_fun_test(X, D, D0, S, S0, opts)
        # l0 = loss_fun_test_spec(X, D, D0, S, S0, opts)
        # l1 = loss_Sck_test_spec(Tdck, b, sc, sc[:, k, :], opts)
        S[:, c, k, :] = solv_sck_test(sc, Tdck, b, k, opts)
        # print('Main fidelity after bpgm the diff is: %1.9e' %(l0[0] - loss_fun_test_spec(X, D, D0, S, S0, opts)[0]))
        # print('Local fidelity after bpgm the diff is: %1.9e' % (l1[0] - loss_Sck_test_spec(Tdck, b, sc, sc[:, k, :], opts)[0]))
        # print('Main sparse after bpgm the diff is: %1.9e' %(l0[1] - loss_fun_test_spec(X, D, D0, S, S0, opts)[1]))
        # print('Local sparse after bpgm the diff is: %1.9e' % (l1[1] - loss_Sck_test_spec(Tdck, b, sc, sc[:, k, :], opts)[1]))
        # print('Main sparse after bpgm the diff is: %1.9e' %(l - loss_fun_test(X, D, D0, S, S0, opts)))
        if torch.isnan(S).sum() + torch.isinf(S).sum() >0 : print(inf_nan_happenned)
    return S


def updateS_test_fista(DD0SS0, X, opts):
    """this function is to update the sparse coefficients for common dictionary D0 using BPG-M, updating each S_n,k^(0)
    input is initialed  DD0SS0
    the data structure is not in matrix format for computation simplexity
        S is 4-d tensor [N,C,K,T] [samples,classes, num of atoms, time series,]
        D is 3-d tensor [C,K,M] [num of atoms, classes, atom size]
        S0 is 3-d tensor [N, K0, T]
        D0 is a matrix [K0, M]
        X is a matrix [N, T], training Data
        Y are the labels, not given
    """
    D, D0, S, S0 = DD0SS0  # where DD0SS0 is a list
    N, K0, T = S0.shape
    M = D0.shape[1]  # dictionary atom dimension
    M_2 = int((M-1)/2)
    C, K, _ = D.shape
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)
    # '''update the current s_n,k^(c) '''
    for c, k in [(i, j) for i in range(C) for j in range(K)]:
        Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
        Crange = torch.tensor(range(C))
        DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
        for cc in range(C):
            # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
            DconvS[:, cc, :] = F.conv1d(S[:, cc, :, :], Dcopy[cc, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]

        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.flip(0).reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        # term 1, 2, 3 should be in the shape of [N, T]
        b = X - R - (DconvS.sum(1) - dck_conv_sck)  # D'*S' = (DconvS.sum(1) - dck_conv_sck
        torch.cuda.empty_cache()

        bb = torch.cat((b, torch.zeros(N, T, device=b.device)), 1)
        TT = torch.cat((Tdck, opts.lamb2**0.5 * torch.eye(T, device=b.device)), 0)
        r1 = fista(bb, TT, sck, opts.lamb)
        S[:, c, k, :] = r1
        # print( r1.abs().sum())

        # # istead of fista using SPAMS
        # alpha = spams.lasso(np.asfortranarray(b.t().cpu().numpy()), D=np.asfortranarray(Tdck.cpu().numpy()), lambda1=opts.lamb/2, lambda2=opts.lamb2)
        # a = sparse.csc_matrix.todense(alpha)
        # r2 = torch.tensor(np.asarray(a).T, device=S.device)
        # # S[:, c, k, :] = torch.tensor(np.asarray(a).T, device=S.device)
        # print(r2.abs().sum())

        if torch.isnan(S).sum() + torch.isinf(S).sum() >0 : print(inf_nan_happenned)
    return S


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
    N, C, K, T = S.shape
    # print('the loss_W for updating W %1.3e:' %loss_W(S, W, Y))
    for c in range(C):
        # print('Before bpgm wc loss is : %1.3e' % loss_W(S[:, c, :, :].clone().unsqueeze(1), W[c, :].reshape(1, -1), Y[:, c].reshape(N, -1)))
        W[c, :] = solv_wc(W[c, :].clone(), S[:, c, :, :], Y[:, c], opts.delta)
        # print('After bpgm wc loss is : %1.3e' % loss_W(S[:, c, :, :].clone().unsqueeze(1), W[c, :].reshape(1, -1), Y[:, c].reshape(N, -1)))
        # print('the loss_W for updating W %1.3e' %loss_W(S, W, Y))
    if torch.isnan(W).sum() + torch.isinf(W).sum() > 0: print(inf_nan_happenned)
    return W


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


def znorm(x):
    """
    This function will make the data with zero-mean, variance = 1 for the last dimension
    :param x: input tensor with shape of [N, ?...?, T]
    :return: x_z
    """
    x_z = (x-x.mean())/(x.var(-1)+1e-38).sqrt().unsqueeze(-1)
    # x_z = (x-x.mean(-1).unsqueeze(-1))/(x.var(-1)+1e-38).sqrt().unsqueeze(-1)
    # x_z = (x - x.mean()) / x.var().sqrt()
    return x_z


def l2norm(x):
    """
    This function will make the data with zero-mean, variance = 1 for the last dimension
    :param x: input tensor with shape of [N, ?...?, T]
    :return: x_z
    """
    x_z = x/(x*x).sum(-1).sqrt().unsqueeze(-1)
    return x_z


def load_data(opts, data='train'):
    """
    This function will load the preprocessed AASP dataset, train and val are in one set, test is the other dataset
    :param opts: only need teh cpu or gpu info
    :return: training, validation or testing data
    """
    route = '../../data/'
    if data == 'test':  # x, y are numpy double arrays
        x, y = torch.load(route+'aasp_test_80by150.pt')
    else:
        x, y = torch.load(route + 'aasp_train_80by150.pt')
    N, T= x.shape
    if opts.shuffle:
        nn = np.arange(x.shape[0])
        np.random.shuffle(nn)
        x, y = x[nn], y[nn]
    if opts.transpose:  # true means stacking over the column
        X = x.reshape(x.shape[0], 80, 150).transpose(0, 2, 1)  # learn atom of over time

    X = torch.from_numpy(x).float().to(opts.dev)
    Y = torch.from_numpy(y).float().to(opts.dev)

    indx = torch.arange(N)
    ind, ind2 = indx[indx%4 !=0], indx[indx%4 ==0]
    xtr, ytr = l2norm(X[ind, :].reshape(ind.shape[0], -1)).reshape(ind.shape[0], T), Y[ind, :]
    xval, yval = l2norm(X[::4, :].reshape(ind2.shape[0], -1)).reshape(ind2.shape[0], T), Y[::4, :]
    if data == 'train' : return xtr, ytr
    if data == 'val' : return xval, yval   # validation
    if data == 'test': return  l2norm(X), Y  # testing


def load_toy(opts, test='train'):
    """
    So far Aug. 24. 2019, we are loading the synthetic data to see if it works, the time series length is 500
    there are 4 classes, with its mixtures but without overlapping. the features are sin waves and rectangular waves,
    ------>>
    500 samples will be cut into 10 fragments, each fragment contains 30 samples long feature with a random start posion from 0 to 20.
    <<------
    the updated version will not have fragments, and there could be bursts of several features
    :return: X, Y, the data and the label matrix
    """
    '''Generate toy data'''
    T = 1600
    if test == 'train': torch.manual_seed(seed)
    if test == 'cv' : torch.manual_seed(opts.seed)
    x = torch.arange(30).float()  # x.sin() only works for float32...
    featurec = featurec = torch.sin(x*2*np.pi/30) - torch.cos(x * 11 * np.pi /30 +1.5)  # '''The common features'''
    feature1 = torch.sin(x * 2 * np.pi / 15) + torch.sin(x * 2 * np.pi / 10)
    feature2 = torch.sin(x * 2 * np.pi / 20) + torch.cos(x * 2 * np.pi / 5) + torch.sin(x * 2 * np.pi / 8)
    feature3 = torch.zeros(30).float()
    feature3[np.r_[np.arange(5), np.arange(10, 15), np.arange(20, 25)]] = 1
    feature3 = feature3 + torch.sin(x * 2 * np.pi / 13)
    feature4 = torch.zeros(30).float()
    feature4[np.r_[np.arange(10), np.arange(20, 30)]] = 1
    feature4 = feature4 + torch.cos(x * np.pi / 6)
    n = opts.n  # number of training examples per combination
    X = torch.zeros(15*n, 2000)  # shape of [N, T+], it will be truncated
    # just the  1 feature
    for ii in range(n):  # loop through each sample
        start_point = torch.randint(0, 101, (1,))
        idx_feat = torch.randint(0, 3, (10,))  # 0 means nothing, 1 means common features, 2 means class features
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        while idx_feat[idx_feat==2].shape[0] < 1:  # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 3, (10,))
        # the procedure is random ramp size + random bursts and repeat
        for i in range(10):  # loop over the last position
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(n, 2*n):
        start_point = torch.randint(0, 101, (1,))
        idx_feat = torch.randint(0, 3, (10,))  # 0 means nothing, 1 means common features, 2 means class features
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        while idx_feat[idx_feat==2].shape[0] < 1:  # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 3, (10,))
        # the procedure is random ramp size + random bursts and repeat
        for i in range(10):  # loop over the last position
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature2.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(2*n, 3*n):
        start_point = torch.randint(0, 101, (1,))
        idx_feat = torch.randint(0, 3, (10,))  # 0 means nothing, 1 means common features, 2 means class features
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        while idx_feat[idx_feat==2].shape[0] < 1:  # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 3, (10,))
        # the procedure is random ramp size + random bursts and repeat
        for i in range(10):  # loop over the last position
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(3*n, 4*n):
        start_point = torch.randint(0, 101, (1,))
        idx_feat = torch.randint(0, 3, (10,))  # 0 means nothing, 1 means common features, 2 means class features
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        while idx_feat[idx_feat==2].shape[0] < 1:  # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 3, (10,))
        # the procedure is random ramp size + random bursts and repeat
        for i in range(10):  # loop over the last position
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature4.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    # just two features
    for ii in range(4*n, 5*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 4, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:  
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 4, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(5*n, 6*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 4, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 4, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(6*n, 7*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 4, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 4, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature4.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(7*n, 8*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 4, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 4, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature3.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(8*n, 9*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 4, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 4, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature4.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(9*n, 10*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 4, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 4, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature4.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    # three features
    for ii in range(10*n, 11*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(2, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 2 or idx_feat[idx_feat==3].shape[0] < 2 \
                or idx_feat[idx_feat==4].shape[0] < 2:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(2, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(11*n, 12*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(2, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 2 or idx_feat[idx_feat==3].shape[0] < 2 \
                or idx_feat[idx_feat==4].shape[0] < 2:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(2, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            elif idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            elif idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            elif idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            else : current_feature = feature4.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(12*n, 13*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(2, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 2 or idx_feat[idx_feat==3].shape[0] < 2 \
                or idx_feat[idx_feat==4].shape[0] < 2:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(2, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature4.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(13*n, 14*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(2, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1 \
                or idx_feat[idx_feat==4].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(2, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature4.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    # Four features
    for ii in range(14*n, 15*n):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(1, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 2 or idx_feat[idx_feat==3].shape[0] < 2 \
                or idx_feat[idx_feat==4].shape[0] < 2 or idx_feat[idx_feat==3].shape[0] < 2:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(1, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature2.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature3.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature4.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point + gap[i]: end_point] = current_feature
            start_point = end_point
    # generate labels
    Y = torch.zeros(15*n, 4)
    for i in range(4):
        current_label = torch.tensor([1, 0, 0, 0]).float()
        current_label = torch.cat((current_label[-i:], current_label[:-i]))
        Y[i*n : (i+1)*n] = current_label
    from itertools import combinations
    comb = list(combinations([0, 1, 2, 3], 2))  # this will give a list of tuples
    for i in range(4, 10):
        current_label = torch.zeros(4)
        current_label[list(comb[i-4])] = 1.0  # make tuple into list for indexing
        Y[i*n : (i+1)*n] = current_label
    for i in range(10, 14):
        current_label = torch.tensor([1, 1, 1, 0]).float()
        current_label = torch.cat((current_label[(i - 10):], current_label[:(i - 10)]))
        Y[i*n : (i+1)*n] = current_label
    current_label = torch.tensor([1, 1, 1, 1]).float()
    Y[(i+1) * n: (i + 2) * n] = current_label

    xx = X[:, :T]
    X = awgn(X[:, :T], opts.snr, opts.seed, test)  #truncation step & adding noise
    # # z-norm, the standardization, 0-mean, var-1
    X = znorm(X)
    # unit norm, norm(x) = 1
    # X = X/(X**2).sum(-1).sqrt().unsqueeze(-1)
    return X.to(opts.dev), Y.to(opts.dev), \
        [featurec, feature1, feature2, feature3, feature4], xx


def loss_fun(X, Y, D, D0, S, S0, W, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, T]
    :param Y: the input label with shape of [N, C]
    :param D: the discriminative dictionary, [C,K,M]
    :param D0: the common dictionary, [K0,M]
    :param S: the sparse coefficients, shape of [N,C,K,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0, T]
    :param W: the projection for labels, shape of [C, K+1]
    :param opts: the hyper-parameters
    :return: cost, the value of loss function
    """
    N, K0, T = S0.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    C, K, _ = D.shape
    ycDcconvSc = S[:, :, 0, :].clone()  # initialization
    ycpDcconvSc = S[:, :, 0, :].clone()  # initialization
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]  # shape of [N, C, T]
        ycpDcconvSc[:, c, :] = (1-Y[:, c].reshape(N, 1)) * DconvS[:, c, :]  # shape of [N, C, T]
        torch.cuda.empty_cache()
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    _1_Y_hat = 1 - Y_hat
    fidelity1 = torch.norm(X - R - DconvS.sum(1))**2
    fidelity2 = torch.norm(X - R - ycDcconvSc.sum(1)) ** 2
    fidelity = fidelity1 + fidelity2 + torch.norm(ycpDcconvSc.sum(1)) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum() * opts.eta
    # label = -1 * opts.eta * (Y * (Y_hat + 3e-38).log() + (1 - Y) * (_1_Y_hat + 3e-38).log()).sum()
    low_rank = N * opts.mu * D0.norm(p='nuc')
    cost = fidelity + sparse + label + low_rank
    return cost


def loss_fun_special(X, Y, D, D0, S, S0, W, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, T]
    :param Y: the input label with shape of [N, C]
    :param D: the discriminative dictionary, [C,K,M]
    :param D0: the common dictionary, [K0,M]
    :param S: the sparse coefficients, shape of [N,C,K,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0, T]
    :param W: the projection for labels, shape of [C, K+1]
    :param opts: the hyper-parameters
    :return: cost, the value of loss function
    """
    N, K0, T = S0.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    C, K, _ = D.shape
    ycDcconvSc = S[:, :, 0, :].clone()  # initialization
    ycpDcconvSc = S[:, :, 0, :].clone()  # initialization
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]  # shape of [N, C, T]
        ycpDcconvSc[:, c, :] = (1-Y[:, c].reshape(N, 1)) * DconvS[:, c, :]  # shape of [N, C, T]
        torch.cuda.empty_cache()
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    _1_Y_hat = 1 - Y_hat
    fidelity1 = torch.norm(X - R - DconvS.sum(1))**2
    fidelity2 = torch.norm(X - R - ycDcconvSc.sum(1)) ** 2
    fidelity = fidelity1 + fidelity2 + torch.norm(ycpDcconvSc.sum(1)) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum() * opts.eta
    # label = -1 * opts.eta * (Y * (Y_hat + 3e-38).log() + (1 - Y) * (_1_Y_hat + 3e-38).log()).sum()
    # print(label.item())
    low_rank = N * opts.mu * D0.norm(p='nuc')
    cost = fidelity + sparse + label + low_rank
    return fidelity.item(), sparse.item(), label.item()


def loss_fun_test(X, D, D0, S, S0, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, T]
    :param D: the discriminative dictionary, [C,K,M]
    :param D0: the common dictionary, [K0,M]
    :param S: the sparse coefficients, shape of [N,C,K,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0, T]
    :param opts: the hyper-parameters
    :return: cost, the value of loss function
    """
    N, K0, T = S0.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    C, K, _ = D.shape
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
        torch.cuda.empty_cache()
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    fidelity = torch.norm(X - R - DconvS.sum(1))**2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    l2 = opts.lamb2 * (S.norm()**2 + S0.norm()**2)
    cost = fidelity + sparse + l2
    return cost


def loss_fun_test_spec(X, D, D0, S, S0, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, T]
    :param D: the discriminative dictionary, [C,K,M]
    :param D0: the common dictionary, [K0,M]
    :param S: the sparse coefficients, shape of [N,C,K,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0, T]
    :param opts: the hyper-parameters
    :return: cost, the value of loss function
    """
    N, K0, T = S0.shape
    M = D0.shape[1]
    M_2 = int((M-1)/2)  # dictionary atom dimension
    C, K, _ = D.shape
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    for c in range(C):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
        torch.cuda.empty_cache()
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    fidelity = torch.norm(X - R - DconvS.sum(1))**2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    return fidelity, sparse


def plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts):
    sns.set(style="darkgrid")
    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'pdf.fonttype': 42  # for LaTeX compatibility
    })

    N, C = Y.shape
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)

    def save_fig(name): 
        if opts.savefig:
            plt.tight_layout()
            plt.savefig(f'figures/{name}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # 1. Sparse coefficients
    ss = S.clone().reshape(S.shape[0], -1)
    plt.figure()
    plt.imshow(ss.abs().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    save_fig('sparse_coefficients_abs')

    ss[ss != 0] = 1
    plt.figure()
    plt.imshow(ss.cpu().numpy(), aspect='auto', cmap='Greys')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    save_fig('sparse_coefficients_binary')

    # 2. Common part
    s0 = S0.clone().reshape(S0.shape[0], -1)
    plt.figure()
    plt.imshow(s0.abs().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    save_fig('common_coefficients_abs')

    s0[s0 != 0] = 1
    plt.figure()
    plt.imshow(s0.cpu().numpy(), aspect='auto', cmap='Greys')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    save_fig('common_coefficients_binary')

    # 3. Reconstructed vs Original
    DconvS = S[:, :, 0, :].clone()
    Dcopy = D.clone().flip(2).unsqueeze(2)
    K, M = D.shape[1:]
    T, M_2 = S.shape[-1], int((M - 1) / 2)
    for c in range(Y.shape[1]):
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]

    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=D0.shape[0], padding=M - 1).sum(1)[:, M_2:M_2 + T]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow((R + DconvS.sum(1)).cpu().numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    plt.subplot(1, 2, 2)
    plt.imshow(X.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    save_fig('reconstructed_data')

    # 4. Zoomed-in version
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow((R + DconvS.sum(1))[200:250, 200:250].cpu().numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    plt.subplot(1, 2, 2)
    plt.imshow(X[200:250, 200:250].cpu().numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    save_fig('reconstructed_zoomed')

    # 5. Loss plot
    if ft != 0:
        plt.figure()
        plt.plot(loss.cpu().numpy(), '-x', linewidth=2)
        plt.xlabel('Epoch index')
        plt.ylabel('Loss')
        plt.grid(True)
        save_fig('loss_curve')

        if opts.show_details:
            l = loss.clone()
            l[:] = torch.log(torch.tensor(-1.0))
            l[::5] = loss[::5]
            plt.plot(l.cpu().numpy(), '--o')
            plt.xlabel('Update Index')
            plt.legend(['Loss (epoch)', 'Loss (updates)'])
            save_fig('loss_details')

        # Common component
        plt.figure()
        plt.plot(D0.squeeze().cpu().numpy(), label='Learned', linewidth=2)
        plt.plot(ft[0] / (ft[0].norm() + 1e-38), '-x', label='Ground truth', linewidth=2)
        plt.xlabel('Time index')
        plt.ylabel('Magnitude')
        plt.legend()
        save_fig('common_feature')

        # Learned features
        for i in range(4):
            plt.figure()
            plt.plot(D[i, 0, :].cpu().numpy()/D[i, 0, :].cpu().norm().numpy(), label='Learned', linewidth=2)
            plt.plot(ft[i + 1] / ft[i + 1].norm(), '-x', label='Ground truth', linewidth=2)
            plt.xlabel('Time index')
            plt.ylabel('Magnitude')
            plt.legend()
            save_fig(f'feature_{i+1}')

    # 6. Labels
    plt.figure()
    plt.imshow(Y.cpu().numpy(), aspect='auto', cmap='Blues', interpolation='none')
    plt.ylabel('Example index')
    plt.xlabel('Label index')
    save_fig('true_labels')

    plt.figure()
    plt.imshow(Y_hat.cpu().numpy(), aspect='auto', cmap='Blues', interpolation='none')
    plt.ylabel('Example index')
    plt.xlabel('Label index')
    save_fig('reconstructed_labels')

    plt.figure()
    Y_hat[Y_hat > 0.5] = 1
    Y_hat[Y_hat <= 0.5] = 0
    plt.imshow(Y_hat.cpu().numpy(), aspect='auto', cmap='Blues', interpolation='none')
    plt.ylabel('Example index')
    plt.xlabel('Label index')
    save_fig('reconstructed_labels_thresholded')

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
    loss, threshold = torch.tensor([], device=opts.dev), 1e-5
    loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
    print('Sparse coding initial loss function value is %3.4e:' % loss[-1])
    S_numel, S0_numel = S.numel(), S0.numel()
    S, S0 = S.clone(), S0.clone()
    for i in range(opts.maxiter):
        t0 = time.time()
        Sold = S.clone()
        S = updateS_test([D, D0, S, S0], X, opts)
        loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
        if opts.show_details:
            print('check sparsity, None-zero percentage is : %1.4f' % (1-(S==0).sum().item()/S_numel))
            print('In the %1.0f epoch, the sparse coding time is :%3.2f, loss function value is :%3.4e'% (i, time.time() - t0, loss[-1]))
            t0 = time.time()
        S0 = updateS0_test([D, D0, S, S0], X, opts)
        if opts.show_details:
            loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
            print('In the %1.0f epoch, the sparse0 coding time is :%3.2f, loss function value is :%3.4e'% (i, time.time() - t0, loss[-1]))
        if opts.show_details:
            if i > 3 and abs((loss[-1] - loss[-3]) / loss[-3]) < threshold:
                print('break condition loss value diff satisfied')
                break
            if support_diff(S, Sold) < 0.005:
                print('break condition support diff satisfied')
                break
        else:
            if i > 3 and abs((loss[-1] - loss[-2]) / loss[-2]) < threshold:
                print('break condition loss value diff satisfied')
                break
            if support_diff(S, Sold) < 0.005:
                print('break condition support diff satisfied')
                break
            if i%3 == 0 : print('In the %1.0f epoch, the sparse coding time is :%3.2f' % ( i, time.time() - t0 ))
    N, C = Y.shape
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
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
    acc_all.precision = precision(Y, y_hat)
    return acc_all, 1/(1+exp_PtSnW), S, S0, loss


def test_fista(D, D0, S, S0, W, X, Y, opts):
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
    loss, threshold = torch.tensor([], device=opts.dev), 1e-4
    loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
    print('The initial loss function value is %3.4e:' % loss[-1])
    S_numel, S0_numel = S.numel(), S0.numel()
    S, S0 = S.clone(), S0.clone()
    for i in range(opts.maxiter):
        t0 = time.time()
        S = updateS_test_fista([D, D0, S, S0], X, opts)
        loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
        if opts.show_details:
            print('check sparsity, None-zero percentage is : %1.4f' % (1-(S==0).sum().item()/S_numel))
            print('In the %1.0f epoch, the sparse coding time is :%3.2f, loss function value is :%3.4e'% (i, time.time() - t0, loss[-1]))
            t0 = time.time()
        S0 = updateS0_test_fista([D, D0, S, S0], X, opts)
        if opts.show_details:
            loss = torch.cat((loss, loss_fun_test(X, D, D0, S, S0, opts).reshape(1)))
            print('In the %1.0f epoch, the sparse0 coding time is :%3.2f, loss function value is :%3.4e'% (i, time.time() - t0, loss[-1]))
        if opts.show_details:
            if i > 3 and abs((loss[-1] - loss[-3]) / loss[-3]) < threshold:
                print('break condition loss value diff satisfied')
                break
            if support_diff(S, Sold) < 0.05:
                print('break condition support diff satisfied')
                break
        else:
            if i > 3 and abs((loss[-1] - loss[-2]) / loss[-2]) < threshold:
                print('break condition loss value diff satisfied')
                break
            if support_diff(S, Sold) < 0.05:
                print('break condition support diff satisfied')
                break
            if i%3 == 0 : print('In the %1.0f epoch, the sparse coding time is :%3.2f' % ( i, time.time() - t0 ))
    N, C = Y.shape
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
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
    acc_all.precision = precision(Y, y_hat)
    return acc_all, 1/(1+exp_PtSnW), S, S0, loss


def train(D, D0, S, S0, W, X, Y, opts):
    """
    This function is the main training body of the algorithm, with showing a lot of details
    :param D: initial value, D, shape of [C, K, M]
    :param D0: pre-trained D0,  shape of [C0, K0, M]
    :param S: initial value, shape of [N,C,K,T]
    :param S0: initial value, shape of [N,K0,T]
    :param W: The pre-trained projection, shape of [C, K]
    :param X: testing data, shape of [N, T]
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


def save_results(D, D0, S, S0, W, opts, loss):
    """
    This function will save the training results
    :param D: initial value, D, shape of [C, K, M]
    :param D0: pre-trained D0,  shape of [C0, K0, M]
    :param S: initial value, shape of [N,C,K,T]
    :param S0: initial value, shape of [N,K0,T]
    :param W: The pre-trained projection, shape of [C, K]
    :param X: testing data, shape of [N, T]
    :param Y: testing Lable, ground truth, shape of [N, C]
    :param opts: options of hyper-parameters
    :param type: type == 0 synthetic data saving
                type == 1 aasp data with S and S0
                type == 2 aasp data without opts, S and S0, but with showing opts info in the file name
    """
    # if type == 0:  # synthetic data saving
    #     torch.save([D, D0, S, S0, W, opts, loss], '../toy_DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')
    # if type == 1:  # all aasp data
    #     torch.save([D, D0, S, S0, W, opts, loss], '../DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')
    # if type == 2:  # dictionaries of aasp
    param = str([opts.K, opts.K0, opts.M, opts.lamb, opts.eta , opts.mu])
    torch.save([D, D0, S, S0, W, opts, loss], '../'+param+'DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')


def awgn(x, snr, cvseed=0, test='train'):
    """
    This function is adding white guassian noise to the given signal
    :param x: the given signal with shape of [N, T]
    :param snr: a float number
    :return:
    """
    if test == 'train': np.random.seed(seed)  # seed is global variable
    if test == 'cv' : np.random.seed(cvseed)
    Esym = x.norm()**2/ x.numel()
    SNR = 10 ** (snr / 10.0)
    N0 = (Esym / SNR).item()
    noise = torch.tensor(np.sqrt(N0) * np.random.normal(0, 1, x.shape), device=x.device)
    return x+noise.to(x.dtype)


def get_perf(Y, S, W):
    """
    this function will show the performance by checking recall, accuracy, and precision
    :param Y: The true label
    :param S: The sparse matrix
    :param W: the projection vector including the bias
    :return: None
    """
    N, C = Y.shape
    S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
    exp_PtSnW = (S_tik * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    y_hat = 1 / (1 + exp_PtSnW)
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat <= 0.5] = 0
    label_diff = Y - y_hat
    acc = label_diff[label_diff==0].shape[0]/label_diff.numel()
    print('acc is : ', acc)
    print('recall is : ', recall(Y, y_hat))
    print('precision is : ', precision(Y, y_hat))


def recall(y, yh):
    """
    calculate the recall score for a matrix
    :param y: N by C
    :param yh: predicted N by C
    :return: recall_score
    """
    s = 0
    N, C = y.shape
    yc = np.array(y.cpu())
    yhc = np.array(yh.cpu())
    # for i in range(C):
    #     s = s + metrics.recall_score(yc[:,i], yhc[:,i])
    # res =s/C
    res = metrics.recall_score(yc.flatten(), yhc.flatten())
    return res


def precision(y, yh):
    """
    calculate the recall score for a matrix
    :param y: N by C
    :param yh: predicted N by C
    :return: precision_score
    """
    s = 0
    N, C = y.shape
    yc = np.array(y.cpu())
    yhc = np.array(yh.cpu())
    # for i in range(C):
    #     s = s + metrics.precision_score(yc[:,i], yhc[:,i])
    # res = s/C
    res = metrics.precision_score(yc.flatten(), yhc.flatten())
    return res


def fista(x, d, s, lamb):
    """
    Using fista to solve the lasso problem, min_s ||x-ds||_2^2 + lamb||s||_1
    :param x: input data, shape of [N, T]
    :param d: dictionary [M=T, K=T], the toeplitz matrix is used here
    :param s: spars [N, K=T]
    :param lamb: hyper parameter of sparsity level
    :return: s
    """
    maxiter, threshold = 500, 1e-5
    _2dtd = 2 * d.t()@d  # shape of [T, T]
    _2dtx = 2 * d.t()@x.t()  # shape of [T, N]
    L = _2dtd.norm()  # larger than the l2 norm but easy to code
    y, sold, told = s.clone(), s.clone(), torch.tensor(1.0, device=d.device) # y is shape of [N,T], t is the step size
    loss = torch.cat((torch.tensor([], device=x.device), loss_fista(x, d, sold, lamb).reshape(1)))

    # for i in range(maxiter):
    i = 0
    while True:
        nu = y - 1/L * (_2dtd@y.t() - _2dtx).t()  # nu is shape of [N, T]
        snew = shrink(L, nu, lamb)
        if torch.norm(snew - sold) / (sold.norm() + 1e-38) < threshold: break

        tnew = (1 + (4*told*told).sqrt())/2
        y = snew + (told-1)/tnew * (snew - sold)
        sold, told = snew.clone(), tnew.clone()

        loss = torch.cat((loss, loss_fista(x, d, sold, lamb).reshape(1)))
        i += 1
        # print(i)
    torch.cuda.empty_cache()
    # diff = loss[1:] - loss[0:-1]
    # if (diff>0).sum() > 0 : input("Loss Increases, PRESS ENTER TO CONTINUE.")
    # plt.plot(loss.cpu().numpy(), '--x')
    return sold


def loss_fista(x, d, s, lamb):
    """
    calculate the loss function value of each iter in fista
    min_s ||x-ds||_2^2 + lamb||s||_1
    :param x: input data, shape of [N, T]
    :param d: dictionary [M=T, K=T], the toeplitz matrix is used here
    :param s: spars [N, K=T]
    :param lamb: hyper parameter of sparsity level
    :return: loss
    """
    term1 = (x.t() - d@s.t()).norm()**2
    term2 = lamb * s.abs().sum()
    loss = term1 + term2
    return loss

def support_diff(S, Sold):
    """
    This function will return the percentage of the difference for the non-zero locations of the sparse coeffients
    :param S: sparse coeffients
    :param Sold: sparse coeffients
    :return: percentage of different
    """
    sf, ssf = S.flatten(), Sold.flatten()
    a, b = torch.zeros(sf.numel()), torch.zeros(ssf.numel())
    a[sf != 0] = 1
    b[ssf != 0] = 1
    return (a - b).abs().sum().item() / (b.sum().item()+ 1e-38)