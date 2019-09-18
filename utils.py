##
# """This file constains all the necessary classes and functions"""
import os
import datetime
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
tt = datetime.datetime.now
# torch.set_default_dtype(torch.double)
np.set_printoptions(linewidth=180)
torch.set_printoptions(linewidth=180)
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class OPT:
    """initial c the number of classes, k0 the size of shared dictionary atoms
    mu is the coeff of low-rank term,
    lamb is the coeff of sparsity
     nu is the coeff of cross-entropy loss
     """
    def __init__(self, C=4, K0=1, K=1, M=30, mu=0.1, eta=0.1, lamb=0.1, delta=0.9, maxiter=200):
        self.C, self.K, self.K0, self.M = C, K, K0, M
        self.mu, self.eta, self.lamb, self.delta = mu, eta, lamb, delta
        self.maxiter, self.plot = maxiter, False
        self.dataset, self.show_details, self.save_results = 0, True, True
        if torch.cuda.is_available():
            self.dev = 'cuda'
            print('\nRunning on GPU')
        else:
            self.dev = 'cpu'
            print('\nRunning on CPU')


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
    D = l2norm(torch.rand(opts.C, opts.K, opts.M, device=opts.dev))
    D0 = l2norm(torch.rand(opts.K0, opts.M, device=opts.dev))
    S = torch.rand(N, opts.C, opts.K, T, device=opts.dev)
    S0 = torch.rand(N, opts.K0, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K, device=opts.dev)

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
    psi = 0
    maxiter = 200
    qq = q*q
    for i in range(maxiter):
        f_grad = -2 * ((P+psi)**(-3) * qq).sum()
        f = ((P+psi)**(-2)*qq).sum()
        psi_new = psi - 2 * f/f_grad * (f.sqrt() - 1)
        if (psi_new - psi).item() < 1e-5:  # psi_new should always larger than psi
            break
        else:
            psi = psi_new.clone()
    dck = -((P + psi_new)**(-1)) * q
    return dck


def solv_dck(x, Md, Md_inv, Mw, Tsck_t, b):
    """x, is the dck, shape of [M]
        M is with shape of [M], diagonal of the majorized matrix
        Minv, is Md^(-1), shape of [M]
        Mw, is a number, == opts.delta
        Tsck_t, is truncated toeplitz matrix of sck with shape of [N, M, T]
        b is bn with all N, with shape of [N, T]
        """
    maxiter, correction = 500, 0.1  # correction is help to make the loss monotonically decreasing
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
        if (d - d_old).norm() / d_old.norm() < 1e-4: break
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
    maxiter, correction = 500, 0.1  # correction is help to make the loss monotonically decreasing
    d_til, d_old, d = x.clone(), x.clone(), x.clone()
    coef = Tsck0_t@Tsck0_t.permute(0, 2, 1)  # shaoe of [N, M, M]
    term = (Tsck0_t@b.unsqueeze(2)).squeeze()  # shape of [N, M]

    # loss = torch.cat((torch.tensor([], device=x.device), loss_D0(Tsck0_t, d, b, D0, mu).reshape(1)))
    for i in range(maxiter):
        d_til = d + correction*Mw*(d - d_old)  # shape of [M],  Mw is just a number for calc purpose
        nu = d_til - (coef@d_til - term).sum(0) * Minv  # shape of [M]
        d_new = argmin_lowrank(M, nu, mu, D0, k0)  # D0 will be changed, because dk0 is in D0
        d, d_old = d_new, d
        if (d - d_old).norm()/d_old.norm() < 1e-3:break
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
    K0, m = D0.shape
    rho = 10 * mu  # agrangian coefficients
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
        if i>10 and abs(cr[-1] - cr[-2])/cr[i-1] < 5e-4: break
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
    maxiter = 500
    snk0_old, snk0 = x.clone(), x.clone()
    coef = Minv @ Tdk0.t() @ Tdk0  # shape of [T, T]
    term = (Minv @ Tdk0.t() @b.t()).t()  # shape of [N, T]

    # loss = torch.cat((torch.tensor([], device=x.device), loss_S0(Tdk0, snk0, b, lamb).reshape(1)))
    for i in range(maxiter):
        snk0_til = snk0 + Mw*(snk0 - snk0_old)  # Mw is just a number for calc purpose
        nu = snk0_til - (coef@snk0_til.t()).t() + term  # nu is [N, T]
        snk0_new = shrink(M, nu, lamb)  # shape of [N, T]
        snk0, snk0_old = snk0_new, snk0
        if torch.norm(snk0 - snk0_old)/(snk0_old.norm() +1e-38) < 1e-4: break
        torch.cuda.empty_cache()
        # loss = torch.cat((loss, loss_S0(Tdk0, snk0, b, lamb).reshape(1)))
    # ll = loss[:-1] - loss[1:]
    # if ll[ll<0].shape[0] > 0: print(something_wrong)
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
    maxiter = 500
    Mw = opts.delta
    lamb = opts.lamb
    dev = opts.dev
    T = b.shape[1]
    P = torch.ones(1, T, device=dev)/T  # shape of [1, T]
    # 'skc update will lead sc change'
    sck = sc[:, k, :].clone()  # shape of [N, T]
    sck_old = sck.clone()
    wkc = wc[k]  # scaler
    Tdck_t_Tdck = Tdck.t() @ Tdck  # shape of [T, T]
    eta_wkc_square = opts.eta * wkc**2  # scaler
    _4_Tdckt_bt = 4*Tdck.t() @ b.t()  # shape of [T, N]
    term0 = (yc-1).unsqueeze(1) @ P * wkc * opts.eta  # shape of [N, T]
    term1 = (abs(4 * Tdck_t_Tdck)).sum(1)  # shape of [T]
    M = (term1 + P*eta_wkc_square + 1e-38).squeeze() # M is the diagonal of majorization matrix, shape of [T]
    sc_til = sc.clone()  # shape of [N, K, T]
    # sc_old = sc.clone(); marker = 0

    # loss = torch.cat((torch.tensor([], device=opts.dev), loss_Sck(Tdck, b, sc, sck, wc, wkc, yc, opts).reshape(1)))
    for i in range(maxiter):
        sck_til = sck + Mw * (sck - sck_old)  # shape of [N, T]
        sc_til[:, k, :] = sck_til
        exp_PtSnc_tilWc = (sc_til.mean(2) @ wc).exp()  # exp_PtSnc_tilWc should change due to sck_til changing
        exp_PtSnc_tilWc[torch.isinf(exp_PtSnc_tilWc)] = 1e38
        term = term0 + (exp_PtSnc_tilWc / (1 + exp_PtSnc_tilWc)*opts.eta*wkc ).unsqueeze(1) @ P
        nu = sck_til - (4*Tdck_t_Tdck@sck_til.t() - _4_Tdckt_bt + term.t()).t()/M  # shape of [N, T]
        sck_new = shrink(M, nu, lamb)  # shape of [N, T]
        sck_old[:], sck[:] = sck[:], sck_new[:]  # make sure sc is updated in each loop
        if exp_PtSnc_tilWc[exp_PtSnc_tilWc == 1e38].shape[0] > 0: marker = 1
        if torch.norm(sck - sck_old) / (sck.norm() + 1e-38) < 1e-4: break
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
    return sck


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
    epx_PtScWc = (sc.mean(2) @ wc).exp()  # shape of N
    epx_PtScWc[torch.isinf(epx_PtScWc)] = 1e38
    epx_PtSckWck = (sck.mean(1) * wkc).exp()
    epx_PtSckWck[torch.isinf(epx_PtSckWck)] = 1e38
    g_sck_wc = (-(1-yc)*((epx_PtSckWck+1e-38).log()) + (1+epx_PtScWc).log()).sum()
    term1 = 2*(Tdck@sck.t() -b.t()).norm()**2
    term2 = opts.lamb * sck.abs().sum()
    term3 = opts.eta * g_sck_wc
    loss = term1 + term2 + term3
    return loss


def solv_wc(x, snc, yc, Mw):
    """
    This fuction is using bpgm to update wc
    :param x: shape of [K], init value of wc
    :param snc: shape of [N, K, T]
    :param yc: shape of [N]
    :param Mw: real number, is delta
    :return: wc
    """
    maxiter, correction = 500, 0.1  # correction is help to make the loss monotonically decreasing
    wc_old, wc, wc_til = x.clone(), x.clone(), x.clone()
    pt_snc = snc.mean(2)  # shape of [N, K]
    abs_pt_snc = abs(pt_snc)  # shape of [N, K]
    const = abs_pt_snc.t() * abs_pt_snc.sum(1)  # shape of [K, N]
    M = const.sum(1)/4 + 1e-38  # shape of [K], 1e-38 for robustness
    one_min_ync = 1 - yc
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
        if torch.norm(wc - wc_old)/wc.norm() < 1e-4: break
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
    xm = x_append0.repeat(m, 1, 1).permute(1, 0, 2)  # shape of [N, m, ?+2m]
    tx = torch.zeros(N, m, T, device=dev)
    for i in range(m):
        ind = range(M2 + i, M2 + i + T)
        tx[:, i, :] = xm[:, i, ind]
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
        Tsck_t = toeplitz(sck, M, T)  # shape of [N, M, T],
        abs_Tsck_t = abs(Tsck_t)
        Md = (abs_Tsck_t @ abs_Tsck_t.permute(0, 2, 1) @ torch.ones(M, device=opts.dev)).sum(0) # shape of [M]
        Md_inv = (Md + 1e-38)**(-1)

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
        if torch.isnan(D).sum() + torch.isinf(D).sum() > 0: print(inf_nan_happenned)
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
        MD = 4*(abs_Tsnk0_t @ abs_Tsnk0_t.permute(0, 2, 1) @ torch.ones(M, device=opts.dev)).sum(0)   # shape of [M]
        MD_inv = 1/(1e-38+MD)  #shape of [M]
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
        W is a matrix [C, K], where K is per-class atoms
        X is a matrix [N, T], training Data
        Y is a matrix [N, C] \in {0,1}, training labels
    """
    D, D0, S, S0, W = DD0SS0W  # where DD0SS0 is a list
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
        wc = W[c, :]  # shape of [K]
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
        sc = S[:, c, :, :] # sc will be changed in solv_sck, adding clone to prevent
        # l0 = loss_fun(X, Y, D, D0, S, S0, W, opts)
        S[:, c, k, :] = solv_sck(sc, wc, yc, Tdck, b, k, opts)
        # print('Main loss after bpgm the diff is: %1.9e' %(l0 - loss_fun(X, Y, D, D0, S, S0, W, opts)))
        # if torch.isnan(S).sum() + torch.isinf(S).sum() >0 : print(inf_nan_happenned)
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
    :param W: shape of [C, K]
    :param Y: shape of [N, C]
    :return:
    """
    exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
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
    x_z = (x-x.mean(-1).unsqueeze(-1))/(x.var(-1)+1e-38).sqrt().unsqueeze(-1)
    return x_z


def l2norm(x):
    """
    This function will make the data with zero-mean, variance = 1 for the last dimension
    :param x: input tensor with shape of [N, ?...?, T]
    :return: x_z
    """
    x_z = x/(x*x).sum(-1).sqrt().unsqueeze(-1)
    return x_z


def load_data(opts=0):
    """
    :param opts:
    :return:
    """


def load_toy(opts):
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
    T = 1200
    x = torch.arange(30).float()  # x.sin() only works for float32...
    featurec = torch.sin(x*2*np.pi/30)  # '''The common features'''
    feature1 = torch.sin(x * 2 * np.pi / 15) + torch.sin(x * 2 * np.pi / 10)
    feature2 = torch.sin(x * 2 * np.pi / 20) + torch.cos(x * 2 * np.pi / 5) + torch.sin(x * 2 * np.pi / 8)
    feature3 = torch.zeros(30).float()
    feature3[np.r_[np.arange(5), np.arange(10, 15), np.arange(20, 25)]] = 1
    feature3 = feature3 + torch.sin(x * 2 * np.pi / 13)
    feature4 = torch.zeros(30).float()
    feature4[np.r_[np.arange(10), np.arange(20, 30)]] = 1
    feature4 = feature4 + torch.cos(x * np.pi / 6)
    X = torch.zeros(750, 2000)  # shape of [N, T+], it will be truncated
    # just the  1 feature
    for ii in range(50):  # loop through each sample
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
    for ii in range(50, 100):
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
    for ii in range(100, 150):
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
    for ii in range(150, 200):
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
    for ii in range(200, 250):
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
    for ii in range(250, 300):
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
    for ii in range(300, 350):
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
    for ii in range(350, 400):
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
    for ii in range(400, 450):
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
    for ii in range(450, 500):
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
    for ii in range(500, 550):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1 \
                or idx_feat[idx_feat==4].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(550, 600):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1 \
                or idx_feat[idx_feat==4].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature4.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(600, 650):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1 \
                or idx_feat[idx_feat==4].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature1.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature4.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    for ii in range(650, 700):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1 \
                or idx_feat[idx_feat==4].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 5, (10,))
        for i in range(10):  # loop over fragment
            if idx_feat[i].item() == 0: current_feature = torch.zeros(30).repeat(burst[i])
            if idx_feat[i].item() == 1: current_feature = featurec.repeat(burst[i])
            if idx_feat[i].item() == 2: current_feature = feature4.repeat(burst[i])
            if idx_feat[i].item() == 3: current_feature = feature2.repeat(burst[i])
            if idx_feat[i].item() == 4: current_feature = feature3.repeat(burst[i])
            end_point = start_point + current_feature.shape[0] + gap[i]
            X[ii, start_point+gap[i]: end_point] = current_feature
            start_point = end_point
    # three features
    for ii in range(700, 750):
        start_point = torch.randint(0, 51, (1,))
        burst = torch.randint(1, 6, (10,))
        gap = torch.randint(0, 51, (10,))
        idx_feat = torch.randint(0, 5, (10,))  # 0 means nothing, 1 means common features
        while idx_feat[idx_feat==2].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1 \
                or idx_feat[idx_feat==4].shape[0] < 1 or idx_feat[idx_feat==3].shape[0] < 1:
            # make sure there are discriminative features to learn
            idx_feat = torch.randint(0, 5, (10,))
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
    Y = torch.zeros(750, 4)
    for i in range(4):
        current_label = torch.tensor([1, 0, 0, 0]).float()
        current_label = torch.cat((current_label[-i:], current_label[:-i]))
        Y[i*50 : (i+1)*50] = current_label
    from itertools import combinations
    comb = list(combinations([0, 1, 2, 3], 2))  # this will give a list of tuples
    for i in range(5, 11):
        current_label = torch.zeros(4)
        current_label[list(comb[i-5])] = 1.0  # make tuple into list for indexing
        Y[i*50 : (i+1)*50] = current_label
    for i in range(11, 15):
        current_label = torch.tensor([1, 1, 1, 0]).float()
        current_label = torch.cat((current_label[-(i - 11):], current_label[:-(i - 11)]))
        Y[i*50 : (i+1)*50] = current_label
    current_label = torch.tensor([1, 1, 1, 1]).float()
    Y[i * 50: (i + 1) * 50] = current_label

    X = X[:, :T]  #truncation step
    # # z-norm, the standardization, 0-mean, var-1
    # X = znorm(X)
    # unit norm, norm(x) = 1
    # X = X/(X**2).sum(-1).sqrt().unsqueeze(-1)
    return X.to(opts.dev), Y.to(opts.dev), [featurec, feature1, feature2, feature3, feature4]


def loss_fun(X, Y, D, D0, S, S0, W, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, T]
    :param Y: the input label with shape of [N, C]
    :param D: the discriminative dictionary, [C,K,M]
    :param D0: the common dictionary, [K0,M]
    :param S: the sparse coefficients, shape of [N,C,K,T] [samples, classes, num of atoms, time series,]
    :param S0: the common coefficients, 3-d tensor [N, K0, T]
    :param W: the projection for labels, shape of [C, K]
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
    exp_PtSnW = (S.mean(3) * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    fisher1 = torch.norm(X - R - DconvS.sum(1))**2
    fisher2 = torch.norm(X - R - ycDcconvSc.sum(1)) ** 2
    fisher = fisher1 + fisher2 + torch.norm(ycpDcconvSc.sum(1)) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    # label = -1 * N * opts.eta * (Y * Y_hat.log() + (1 - Y) * (1 - Y_hat + 3e-38).log()).sum()
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum()
    low_rank = N * opts.mu * D0.norm(p='nuc')
    cost = fisher + sparse + label + low_rank
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
    :param W: the projection for labels, shape of [C, K]
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
    exp_PtSnW = (S.mean(3) * W).sum(2).exp()   # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    fisher1 = torch.norm(X - R - DconvS.sum(1))**2
    fisher2 = torch.norm(X - R - ycDcconvSc.sum(1)) ** 2
    fisher = fisher1 + fisher2 + torch.norm(ycpDcconvSc.sum(1)) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    # label = -1 * N * opts.eta * (Y * Y_hat.log() + (1 - Y) * (1 - Y_hat + 3e-38).log()).sum()
    label = (-1 * (1 - Y)*(exp_PtSnW+1e-38).log() + (exp_PtSnW + 1).log()).sum()
    low_rank = N * opts.mu * D0.norm(p='nuc')
    cost = fisher + sparse + label + low_rank
    return fisher, sparse, label


def loss_Sck_special(Tdck, b, sc, sck, wc, wkc, yc, opts):
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
    epx_PtScWc = (sc.mean(2) @ wc).exp()  # shape of N
    epx_PtScWc[torch.isinf(epx_PtScWc)] = 1e38
    epx_PtSckWck = (sck.mean(1) * wkc).exp()
    epx_PtSckWck[torch.isinf(epx_PtSckWck)] = 1e38
    g_sck_wc = (-(1-yc)*((epx_PtSckWck+1e-38).log()) + (1+epx_PtScWc).log()).sum()
    fisher = 2*(Tdck@sck.t() - b.t()).norm()**2
    sparse = opts.lamb * sck.abs().sum()
    label = opts.eta * g_sck_wc
    if label <0 :print(stop)
    return fisher, sparse, label


def plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts):
    exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)

    # reconstruction of input signal
    DconvS = S[:, :, 0, :].clone()  # to avoid zeros for cuda decision, shape of [N, C, T]
    Dcopy = D.clone().flip(2).unsqueeze(2)  # D shape is [C,K,1, M]
    K , M = D.shape[1:]
    T, M_2 =  S.shape[-1], int((M-1)/2)
    for c in range(Y.shape[1]):
        # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
        DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=D0.shape[0], padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    plt.figure()
    ss = S.clone().reshape(S.shape[0], -1)
    plt.imshow(ss.abs().cpu().numpy(), aspect='auto')
    plt.title('Absolute value of sparse coefficients')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    ss[ss!=0] = 1
    plt.figure()
    plt.imshow(ss.cpu().numpy(), aspect='auto')
    plt.title('None zeros of sparse coefficients')
    plt.xlabel('Time index')
    plt.ylabel('Example index')

    plt.figure()
    s0 = S0.clone().reshape(S0.shape[0], -1)
    plt.imshow(s0.abs().cpu().numpy(), aspect='auto')
    plt.title('Absolute value of sparse coefficients - common part')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    s0[s0!=0] = 1
    plt.figure()
    plt.imshow(s0.cpu().numpy(), aspect='auto')
    plt.title('None zeros of sparse coefficients - common part')
    plt.xlabel('Time index')
    plt.ylabel('Example index')

    plt.figure()
    plt.subplot(121)
    plt.imshow((R + DconvS.sum(1)).cpu().numpy())
    plt.title('Reconstrution of data')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    plt.subplot(122)
    plt.imshow(X.cpu().numpy())
    plt.title('Training of data')
    plt.xlabel('Time index')
    plt.ylabel('Example index')

    plt.figure()
    plt.subplot(121)
    plt.imshow((R + DconvS.sum(1))[100:200, 100:200].cpu().numpy())
    plt.title('Reconstrution of data, zoomed-in')
    plt.xlabel('Time index')
    plt.ylabel('Example index')
    plt.subplot(122)
    plt.imshow(X[100:200, 100:200].cpu().numpy())
    plt.title('Training of data, zoomed-in')
    plt.xlabel('Time index')
    plt.ylabel('Example index')

    plt.figure()
    plt.plot(loss.cpu().numpy(), '-x')
    plt.title('Loss function value')
    plt.xlabel('Epoch index')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.figure()
    plt.plot(D0.squeeze().cpu().numpy())
    plt.plot(ft[0] / (ft[0].norm()+1e-38), '-x')
    plt.title('commom component')
    plt.legend(['Learned feature', 'Ground true'])
    plt.xlabel('Time index')
    plt.ylabel('Magnitude')
    for i in range(4):
        plt.figure()
        plt.plot(D[i, 0, :].cpu().numpy()/D[i, 0, :].cpu().norm().numpy())
        plt.plot(ft[i + 1] / ft[i + 1].norm(), '-x')
        plt.title('Feature ' + str(i + 1))
        plt.legend(['Learned feature', 'Ground true'])
        plt.xlabel('Time index')
        plt.ylabel('Magnitude')
    plt.figure()
    plt.imshow(Y.cpu().numpy(), aspect='auto')
    plt.title('True labels')
    plt.ylabel('Training example index')
    plt.xlabel('Label index')
    plt.figure()
    plt.imshow(Y_hat.cpu().numpy(), aspect='auto')
    plt.title('Reconstructed labels')
    plt.ylabel('Training example index')
    plt.xlabel('Label index')


def test(D, D0, S, S0, W, X, Y, opts):
    """
    This function is made to see the test accuracy by checking the reconstrunction label
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
    loss = torch.tensor([], device=opts.dev)
    for i in range(opts.maxiter):
        t0 = time.time()
        S = updateS([D, D0, S, S0, W], X, Y, opts)
        S0 = updateS0([D, D0, S, S0], X, Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('In the %1.0f epoch, the sparse coding time is :%3.2f' % (i, time.time() - t0))
        if i > 10 and abs((loss[-1] - loss[-2]) / loss[-2]) < 5e-4: break
    exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    y_hat = 1 / (1 + exp_PtSnW)
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    label_diff = Y - y_hat
    acc = label_diff[label_diff==0].shape[0]/label_diff.numel()
    return acc, 1/(1+exp_PtSnW)


def test_details(D, D0, S, S0, W, X, Y, opts):
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
    loss = torch.tensor([], device=opts.dev)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('The initial loss function value is %3.4e:' % loss[-1])
    for i in range(opts.maxiter):
        t0 = time.time()
        S = updateS([D, D0, S, S0, W], X, Y, opts)
        S0 = updateS0([D, D0, S, S0], X, Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('check sparsity, None-zero percentage is : %1.3f' % (1 - S[S == 0].shape[0] / S.numel()))
        print('In the %1.0f epoch, the sparse coding time is :%3.2f, loss function value is :%3.4e'
              % (i, time.time() - t0, loss[-1]))
        if i > 10 and abs((loss[-1] - loss[-2]) / loss[-2]) < 5e-4: break
    exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    y_hat = 1 / (1 + exp_PtSnW)
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    label_diff = Y - y_hat
    acc = label_diff[label_diff==0].shape[0]/label_diff.numel()
    return acc, 1/(1+exp_PtSnW)


def train(D, D0, S, S0, W, X, Y, opts):
    """
    This function is the main training body of the algorithm
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
    loss = torch.tensor([], device=opts.dev)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('The initial loss function value is %3.4e:' % loss[-1])
    t = time.time()
    for i in range(opts.maxiter):
        t0 = time.time()
        D = updateD([D, D0, S, S0, W], X, Y, opts)
        D0 = updateD0([D, D0, S, S0], X, Y, opts)
        S = updateS([D, D0, S, S0, W], X, Y, opts)
        S0 = updateS0([D, D0, S, S0], X, Y, opts)
        W = updateW([S, W], Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        if i > 10 and abs((loss[-1] - loss[-2]) / loss[-2]) < 5e-4: break
        print('In the %1.0f epoch, the training time is :%3.2f' % (i, time.time() - t0))

    print('After %1.0f epochs, the loss function value is %3.4e:' % (i, loss[-1]))
    print('All done, the total running time is :%3.2f \n' % (time.time() - t))
    return D, D0, S, S0, W, loss


def train_details(D, D0, S, S0, W, X, Y, opts):
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
    loss = torch.tensor([], device=opts.dev)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('The initial loss function value is %3.4e:' % loss[-1])
    t, t1 = time.time(), time.time()
    for i in range(opts.maxiter):
        t0 = time.time()
        D = updateD([D, D0, S, S0, W], X, Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('pass D, time is %3.2f' % (time.time() - t)); t = time.time()
        print('loss function value is %3.4e:' %loss[-1])

        D0 = updateD0([D, D0, S, S0], X, Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('pass D0, time is %3.2f' % (time.time() - t)); t = time.time()
        print('loss function value is %3.4e:' %loss[-1])

        S = updateS([D, D0, S, S0, W], X, Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('pass S, time is %3.2f' % (time.time() - t)); t = time.time()
        print('loss function value is %3.4e:' %loss[-1])
        print('check sparsity, None-zero percentage is : %1.3f' % (1 - S[S == 0].shape[0] / S.numel()))

        S0 = updateS0([D, D0, S, S0], X, Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('pass S0, time is %3.2f' % (time.time() - t)); t = time.time()
        print('loss function value is %3.4e:' %loss[-1])
        print('check sparsity, None-zero percentage is : %1.3f' % (1 - S0[S0 == 0].shape[0] / S0.numel()))

        W = updateW([S, W], Y, opts)
        loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
        print('pass W, time is %3.2f' % (time.time() - t)); t = time.time()
        print('loss function value is %3.4e:' %loss[-1])

        if i > 10 and abs((loss[-1] - loss[-6]) / loss[-6]) < 5e-4: break
        print('In the %1.0f epoch, the training time is :%3.2f \n' % (i, time.time() - t0))

    print('After %1.0f epochs, the loss function value is %3.4e:' % (i, loss[-1]))
    print('All done, the total running time is :%3.2f \n' % (time.time() - t1))
    return D, D0, S, S0, W, loss


def save_results(D, D0, S, S0, W, opts, loss):
    torch.save([D, D0, S, S0, W, opts, loss], '../DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')