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


class OPT:
    """initial c the number of classes, k0 the size of shared dictionary atoms
    mu is the coeff of low-rank term,
    lamb is the coeff of sparsity
     nu is the coeff of cross-entropy loss
     """
    def __init__(self, C=4, K0=1, K=1, M=30, mu=0.1, eta=0.1, lamb=0.1, delta=0.9, maxiter=500):
        self.C, self.K, self.K0, self.M = C, K, K0, M
        self.mu, self.eta, self.lamb, self.delta = mu, eta, lamb, delta
        self.maxiter, self.plot = maxiter, False
        self.dataset = 0
        if torch.cuda.is_available():
            self.dev = 'cuda'
            print('\nGPU is available and GPU will be used')
        else:
            self.dev = 'cpu'
            print('\nGPU is not available and CPU will be used')


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
    ind = list(range(N))
    np.random.shuffle(ind)
    D = znorm(torch.rand(opts.C, opts.K, opts.M, device=opts.dev))
    D0 = znorm(torch.rand(opts.K0, opts.M, device=opts.dev))
    S = torch.rand(N, opts.C, opts.K, T, device=opts.dev)
    S0 = torch.rand(N, opts.K0, T, device=opts.dev)
    W = torch.ones(opts.C, opts.K, device=opts.dev)
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


def solv_dck0(x, M, Minv, Mw, Tsck0_t, b, D0, mu, k0):
    """
    :param x: is the dck, shape of [M]
    :param M: is MD, the majorizer matrix with shape of [M, M], diagonal matrix
    :param Minv: is MD^(-1)
    :param Mw: is a number not diagonal matrix
    :param Tsck0_t: is truncated toeplitz matrix of sck with shape of [N, M, T], already *2
    :param b: bn with all N, with shape of [N, T]
    :param D0: is the shared dictionary
    :param mu: is the coefficient fo low-rank term, mu = N*mu
    :param k0: the current index of for loop of K0
    :return: dck0
    """
    maxiter = 500
    d_old, d = x.clone(), x.clone()
    coef = Minv @ (Tsck0_t @ Tsck0_t.permute(0, 2, 1)).sum(0)
    term = Minv @ (Tsck0_t @ b.unsqueeze(2)).sum(0)
    # print('before bpgm loop loss D0 in solve_d0 is %3.2e:' % loss_D0(Tsck0_t, d, b, D0, mu))
    for i in range(maxiter):
        d_til = d + Mw*(d - d_old)  # Mw is just a number for calc purpose
        nu = d_til - (coef@d_til).squeeze() + term.squeeze()  # nu is 1-d tensor
        d_new = argmin_lowrank(M, nu, mu, D0, k0)  # D0 will be changed, because dk0 is in D0
        d, d_old = d_new, d
        if (d - d_old).norm()/d_old.norm() < 0.01:
            break
        torch.cuda.empty_cache()
        # print('loss D0 in solve_d0 is %3.2e:' %loss_D0(Tsck0_t, d, b, D0, mu))
        # print('')
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
        if torch.norm(cr[i]) < 1e-8 : break
        if i > 10:  # if not going anywhere
            if abs(cr[i] - cr[i-10]).sum() < 5e-8: break
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
    Mw = opts.delta
    lamb = opts.lamb
    dev = opts.dev
    T = b.shape[1]
    P = torch.ones(1, T, device=dev)/T  # shape of [T]
    # 'skc update will lead sc change'
    sck = sc[:, k, :]  # shape of [N, T]
    sck_old = sck.clone()
    wkc = wc[k]  # scaler
    Tdck_t_Tdck = Tdck.t() @ Tdck  # shape of [T, T]
    eta_wkc_square = (opts.eta*wkc)**2  # scaler
    Tdckt_bt = Tdck.t() @ b.t()  # shape of [T, N]
    term0 = (yc-1).unsqueeze(1) @ P * wkc * opts.eta  # shape of [N, T]
    term1 = (abs(8 * Tdck_t_Tdck)).sum(1)  # shape of [T]
    M = term1 + P*eta_wkc_square/4 + 1e-38 # M is the diagonal of majorization matrix, shape of [N, T]
    M_old = M.clone()

    maxiter = 500
    for i in range(maxiter):
        sck_til = sck + Mw * (sck - sck_old)  # shape of [N, T]
        exp_PtSnc_tilWc = (sck_til.mean(2) @ wc).exp()  # exp_PtSnc_tilWc should change due to sck_til changing
        exp_PtSnc_tilWc[torch.isinf(exp_PtSnc_tilWc)] = 1e38
        term = term0 + (exp_PtSnc_tilWc / (1 + exp_PtSnc_tilWc) * opts.eta ).unsqueeze(1) @ P
        nu = sck_til - (8*Tdck_t_Tdck@sck_til.t() - Tdckt_bt + term.t()).t()/M  # shape of [N, T]
        sck_new = svt_s(M, nu, lamb)  # shape of [N, T]
        sck[:], sck_old[:] = sck_new[:], sck[:]  # make sure sc is updated in each loop
        if torch.norm(sck - sck_old) < 1e-4:
            break
        torch.cuda.empty_cache()
    return sck


def solv_wc(x, snc, yc, Mw):
    """
    This fuction is using bpgm to update wc
    :param x: shape of [K], init value of wc
    :param snc: shape of [N, K, T]
    :param yc: shape of [N]
    :param Mw: real number, is delta
    :return: wc
    """
    maxiter = 500
    wc_old, wc = x.clone(), x.clone()
    pt_snc = snc.mean(2)  # shape of [N, K]
    abs_pt_snc = abs(pt_snc)  # shape of [N, K]
    const = abs_pt_snc.t() * abs_pt_snc.sum(1)  # shape of [K, N]
    M = const.sum(1)/4 + 1e-38  # shape of [K], 1e-38 for robustness
    one_min_ync = 1 - yc
    M_old = M.clone()
    # print('before bpgm wc loss is : %1.3e' %loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)))
    for i in range(maxiter):
        wc_til = wc + Mw*(wc - wc_old)  # Mw is just a number for calc purpose
        exp_pt_snc_wc_til = (pt_snc @ wc_til).exp()  # shape of [N]
        exp_pt_snc_wc_til[torch.isinf(exp_pt_snc_wc_til)] = 1e38
        nu = wc_til + M**(-1) * ((one_min_ync - exp_pt_snc_wc_til/(1+exp_pt_snc_wc_til))*pt_snc.t()).sum(1)  # nu is [K]
        wc, wc_old = nu.clone(), wc[:]  # gradient is not needed, nu is the best solution
        # print('torch.norm(wc - wc_old)', torch.norm(wc - wc_old).item())
        if torch.norm(wc - wc_old)/wc.norm() < 1e-2:
            break
        torch.cuda.empty_cache()
        # print('wc loss in the bpgm :%1.3e' %loss_W(snc.clone().unsqueeze(1), wc.reshape(1, -1), yc.clone().unsqueeze(-1)))
    return wc


def gradd(const, pt_snc, nu, init_wc):  # this one id abandoned, beause math part did not need it
    """
    This function is meant to solve 1/2||x-\nu||_M^2, where M is a function of x,
    This looks like a convex problem but it is not because M = f(x), x is part of demoninator
    :param const: abs_pt_snc.t()*abs_pt_snc.sum(1) , shape of [K, N]
    :param pt_snc: snc(mean(2) constant, shape of [N, K]
    :param nu: constant, shape of [M]
    :param init_wc: is a clone of wc for the initialization, shape of [K]
    :return:
    """
    wc = init_wc.requires_grad_()
    lr = 0.1
    maxiter = 500
    # print(pt_snc.max(), nu.max(), pt_snc.max())
    loss = []
    for i in range(maxiter):
        exp_pt_snc_wc = (pt_snc @ wc).exp()  # shape of [N]
        if torch.isinf(exp_pt_snc_wc).sum().item() > 0 : print('inf problem happened in gradient descent \n')
        M = (exp_pt_snc_wc/(1 + exp_pt_snc_wc)**2 * const).sum(1)  # shape of [K]
        lossfunc = 1/2*((wc-nu) * M * M * (wc-nu)).sum()
        # print('loss func in gradient descent :',i, 'iter ', lossfunc)
        lossfunc.backward()
        loss.append(lossfunc.detach().cpu().item())
        if abs(wc.grad).sum() < 1e-5: break  # stop criteria
        if i > 10 and abs(loss[i]-loss[i-1])/abs(loss[i]) < 1e-3: break  # stop criteria
        with torch.no_grad():
            # print('loss func in gradient descent :',i, 'iter ', wc.grad)
            if torch.isnan(wc.grad).item(): break  # because of too large number to calc
            wc = wc - lr*wc.grad
            wc.requires_grad_()
        torch.cuda.empty_cache()
    return wc.detach().requires_grad_(False), M.detach().requires_grad_(False)


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
    :param M: is used for matrix norm, shape of [T]
    :param lamb: is coefficient of L-1 norm, scaler
    :param nu: the the matrix to be pruned, shape of [N, T]
    :return: P the matrix after singular value thresholding
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
        for c in range(C):
            # the following line is doing, convolution, sum up C, and truncation for m/2: m/2+T
            DconvS[:, c, :] = F.conv1d(S[:, c, :, :], Dcopy[c, :, :, :], groups=K, padding=M - 1).sum(1)[:, M_2:M_2 + T]
            # D*S, not the D'*S', And here D'*S' will not be updated for each d_c,k update
            torch.cuda.empty_cache()

        dck = D[c, k, :]  # shape of [M]
        sck = S[:, c, k, :]  # shape of [N, T]
        Tsck_t = toeplitz(sck, M, T)  # shape of [N, M, T],
        abs_Tsck_t = abs(Tsck_t)
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD_diag = ((abs_Tsck_t @ abs_Tsck_t.permute(0, 2, 1)).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()  # shape of [M]
        MD_diag = MD_diag + 1e-10  # to make it robust for inverse
        MD = MD_diag.diag()
        MD_inv = (1/MD_diag).diag()

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
        # print('before updata dck : %3.2e' %loss_D(Tsck_t, D[c, k, :], b))
        # print('before updata dck : %1.5e' %loss_fun(X, Y, D, D0, S, S0, W, opts))
        D[c, k, :] = solv_dck(dck, MD, MD_inv, Mw, Tsck_t, b)
        # print('after updata dck : %1.5e' %loss_fun(X, Y, D, D0, S, S0, W, opts))
        # print('after updata dck : %3.2e' %loss_D(Tsck_t, D[c, k, :], b))
        if torch.isnan(S).sum() + torch.isinf(S).sum() > 0: print(inf_nan_happenned)
    return D


def loss_D(Tsck_t, dck, b):
    """
    calculate the loss function value for updating D, sum( norm(Tsnck*dck - bn)**2 ) , s.t. norm(dck) <=1
    :param Tsck_t: shape of [N, M, T],
    :param dck: cth, kth, atom of D, shape of [M]
    :param b: the definiation is long in the algorithm, shape of [N, T]
    :return: loss fucntion value
    """
    return (((Tsck_t.permute(0, 2, 1)*dck).sum(2) - b)**2 ).sum().item()


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
        abs_Tsnk0_t = abs(Tsnk0_t)
        Mw = opts.delta   # * torch.eye(M, device=opts.dev)
        MD_diag = 4*((abs_Tsnk0_t @ abs_Tsnk0_t.permute(0, 2, 1)).sum(0) @ torch.ones(M, 1, device=opts.dev)).squeeze()  # shape of [M]
        MD_diag = MD_diag + 1e-10  # make it robust for the M^-1
        MD = MD_diag.diag()
        MD_inv = (1/MD_diag).diag()
        b = 2*X - alpha_plus_dk0 - beta_plus_dk0 + 2*dk0convsnk0
        torch.cuda.empty_cache()
        # print('D0 loss function value before update is %3.2e:' %loss_D0(2*Tsnk0_t, dk0, b, D0, opts.mu*N))
        D0[k0, :] = solv_dck0(dk0, MD, MD_inv, Mw, 2*Tsnk0_t, b, D0, opts.mu*N, k0)
        # print('D0 loss function value after update is %3.2e:' % loss_D0(2*Tsnk0_t, dk0, b, D0, opts.mu*N))
        if torch.isnan(D0S).sum() + torch.isinf(D0).sum() > 0: print(inf_nan_happenned)
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
    return ((((Tsnk0_t.permute(0, 2, 1)*dk0).sum(2) - b)**2 ).sum()/2 + mu*D0.norm(p='nuc')).item()


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
        MS0_diag = MS0_diag + 1e-10  # make it robust for inverse
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

        dck_conv_sck = F.conv1d(sck.unsqueeze(1), dck.flip(0).reshape(1, 1, M), padding=M-1).squeeze()[:, M_2:M_2+T]  # shape of [N,T]
        c_prime = Crange[Crange != c]  # c_prime contains all the indexes
        # the following line is to get the sum_c'(D^(c')*S^(c'))
        DpconvSp = ((1- Y[:, c_prime] - Y[:, c_prime]*Y[:, c].reshape(N, 1)).unsqueeze(2)*DconvS[:, c_prime, :]).sum(1)
        b = (X - R - (DconvS.sum(1) - dck_conv_sck) - (DconvS[:, c, :] - dck_conv_sck) + DpconvSp)/2  # b is shape of [N, T]
        torch.cuda.empty_cache()
        sc = S[:, c, :, :].clone()  # sc will be changed in solv_sck, so giving a clone here
        S[:, c, k, :] = solv_sck(sc, wc, yc, Tdck, b, k, opts)
        if torch.isnan(S).sum() + torch.isinf(S).sum() >0 : print(inf_nan_happenned)
    return S


def loss_S(Tdck, b, sck, opts, wc):
    """
    This function calculates the loss func
    :param Tdck:
    :param b:
    :param sck:
    :param opts:
    :param wc:
    :return:
    """
    pass


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
    # using Y_hat is not stable because of log(), 1-Y_hat could be 0
    # exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
    # exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    # Y_hat = 1/ (1+ exp_PtSnW)
    # loss = -1 * (Y * Y_hat.log() + (1 - Y) * (1 - Y_hat + 3e-38).log()).sum()  # 1e-38 for robustness

    PtSnW = (S.mean(3) * W).sum(2)  # shape of [N, C]
    exp_PtSnW = PtSnW.exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    loss = (-1 *(1-Y)*PtSnW + (exp_PtSnW+1).log()).sum()
    return loss.item()


def znorm(x):
    """
    This function will make the data with zero-mean, variance = 1 for the last dimension
    :param x: input tensor with shape of [N, ?...?, T]
    :return: x_z
    """
    x_z = (x-x.mean(-1).unsqueeze(-1))/x.var(-1).sqrt().unsqueeze(-1)
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
    x = torch.arange(30).float()
    featurec = torch.sin(x*2*np.pi/30)  # '''The common features'''
    feature1 = torch.sin(x * 2 * np.pi / 15) + torch.sin(x * 2 * np.pi / 10)
    feature2 = torch.sin(x * 2 * np.pi / 20) + torch.cos(x * 2 * np.pi / 5) + torch.sin(x * 2 * np.pi / 8)
    feature3 = torch.zeros(30)
    feature3[np.r_[np.arange(5), np.arange(10, 15), np.arange(20, 25)]] = 1
    feature3 = feature3 + torch.sin(x * 2 * np.pi / 13)
    feature4 = torch.zeros(30)
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

    return X[:, :T].to(opts.dev), Y.to(opts.dev), [featurec, feature1, feature2, feature3, feature4]


def loss_fun(X, Y, D, D0, S, S0, W, opts):
    """
    This function will calculate the costfunction value
    :param X: the input data with shape of [N, T]
    :param Y: the input label with shape of [N, C]
    :param D: the discriminative dictionary
    :param D0: the common dictionary
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
        ycDcconvSc[:, c, :] = Y[:, c].reshape(N, 1) * DconvS[:, c, :]
        ycpDcconvSc[:, c, :] = (1-Y[:, c].reshape(N, 1)) * DconvS[:, c, :]
        torch.cuda.empty_cache()
    R = F.conv1d(S0, D0.flip(1).unsqueeze(1), groups=K0, padding=M - 1).sum(1)[:, M_2:M_2 + T]  # r is shape of [N, T)

    exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
    exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
    Y_hat = 1 / (1 + exp_PtSnW)
    fisher1 = torch.norm(X - R - DconvS.sum(1))**2
    fisher2 = torch.norm(X - R - ycDcconvSc.sum(1)) ** 2
    fisher = fisher1 + fisher2 + torch.norm(ycpDcconvSc.sum(1)) ** 2
    sparse = opts.lamb * (S.abs().sum() + S0.abs().sum())
    label = -1 * N * opts.eta * (Y*Y_hat.log() + (1-Y)*(1-Y_hat + 3e-38).log()).sum()
    low_rank = N * opts.mu * D0.norm(p='nuc')
    cost = fisher + sparse + label + low_rank
    return cost.cpu().item()

