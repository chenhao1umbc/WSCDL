"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT()
X, Y = load_toy(opts)
D, D0, S, S0, W = init(X, opts)
loss = []
loss.append(loss_fun(X, Y, D, D0, S, S0, W, opts))
print('The initial loss function value is %3.4e:' %loss[-1])
for i in range(opts.maxiter):
    t = time.time()
    D = updateD([D, D0, S, S0, W], X, Y, opts)
    # print('pass D, how mnay nan is ', torch.isnan(D).sum().item())
    print('pass D, time is %3.2f' % (time.time() -t) ); t = time.time()
    loss.append(loss_fun(X, Y, D, D0, S, S0, W, opts))
    print('loss function value is %3.4e:' %loss[-1])

    D0 = updateD0([D, D0, S, S0], X, Y, opts)
    # print('pass D0, how mnay nan is ',torch.isnan(D0).sum().item())
    print('pass D0, time is %3.2f' % (time.time() -t) ); t = time.time()
    loss.append(loss_fun(X, Y, D, D0, S, S0, W, opts))
    print('loss function value is %3.4e:' %loss[-1])

    S = updateS([D, D0, S, S0, W], X, Y, opts)
    print('check sparsity, 0 percentage is %3.2e:' % (S[S==0].shape[0]/S.numel()))
    # print('pass S, how mnay nan is ',torch.isnan(S).sum().item())
    print('pass S, time is %3.2f' % (time.time() -t) ); t = time.time()
    loss.append(loss_fun(X, Y, D, D0, S, S0, W, opts))
    print('loss function value is %3.4e:' %loss[-1])

    S0 = updateS0([D, D0, S, S0], X, Y, opts)
    # print('pass S0, how mnay nan is ',torch.isnan(S0).sum().item())
    print('pass S0, time is %3.2f' % (time.time() -t) ); t = time.time()
    loss.append(loss_fun(X, Y, D, D0, S, S0, W, opts))
    print('loss function value is %3.4e:' %loss[-1])

    W = updateW([S, W], Y, opts)
    # print('pass W, how mnay nan is ',torch.isnan(W).sum().item())
    print('pass W, time is %3.2f' % (time.time() -t) ); t = time.time()
    loss.append(loss_fun(X, Y, D, D0, S, S0, W, opts))
    print('loss function value is %3.4e:' %loss[-1])

exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
Y_hat = 1/ (1+ exp_PtSnW)