"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
opts = OPT(maxiter=50)
opts.lamb = 1  # for sparsity penalty
opts.eta = 1  # for label penalty
opts.mu= 1  # for low rank penalty
X, Y, ft= load_toy(opts)
D, D0, S, S0, W = init(X, opts)
loss = torch.tensor([], device=opts.dev)
loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
print('The initial loss function value is %3.4e:' %loss[-1])
t = time.time()
for i in range(opts.maxiter):
    t0 = time.time()
    D = updateD([D, D0, S, S0, W], X, Y, opts)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('pass D, time is %3.2f' % (time.time() -t) ); t = time.time()
    print('loss function value is %3.4e:' %loss[-1])

    D0 = updateD0([D, D0, S, S0], X, Y, opts)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('pass D0, time is %3.2f' % (time.time() -t) ); t = time.time()
    print('loss function value is %3.4e:' %loss[-1])

    S = updateS([D, D0, S, S0, W], X, Y, opts)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('check sparsity, None-zero percentage is : %1.3f' % (1-S[S==0].shape[0]/S.numel()))
    print('pass S, time is %3.2f' % (time.time() -t) ); t = time.time()
    print('loss function value is %3.4e:' %loss[-1])

    S0 = updateS0([D, D0, S, S0], X, Y, opts)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('pass S0, time is %3.2f' % (time.time() -t) ); t = time.time()
    print('loss function value is %3.4e:' %loss[-1])

    W = updateW([S, W], Y, opts)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    print('pass W, time is %3.2f' % (time.time() -t) ); t = time.time()
    print('loss function value is %3.4e:' %loss[-1])

    # if i > 10 and abs((loss[-1]-loss[-6])/loss[i-5]) < 5e-4 : break
    print('One epoch training time is :%3.2f' % (time.time() -t0) )

print('After %1.0f epochs, the loss function value is %3.4e:' %(i, loss[-1]))
print('All done, the total running time is :%3.2f \n' % (time.time() -t))
exp_PtSnW = (S.mean(3) * W).sum(2).exp()  # shape of [N, C]
exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
Y_hat = 1/ (1+ exp_PtSnW)
plt.figure();plt.plot(loss.cpu().numpy(), '-x'); plt.title('Loss fucntion value')
# plt.figure();plt.plot(D0.squeeze().cpu().numpy()); plt.plot(ft[0], '-x'); plt.title('commom component')
# plt.figure();plt.plot(D[0, 0, :].cpu().numpy()); plt.plot(ft[1], '-x'); plt.title('feature 1')
# plt.figure();plt.plot(D[1, 0, :].cpu().numpy()); plt.plot(ft[2], '-x');plt.title('feature 2')
# plt.figure();plt.plot(D[2, 0, :].cpu().numpy()); plt.plot(ft[3], '-x');plt.title('feature 3')
# plt.figure();plt.plot(D[3, 0, :].cpu().numpy()); plt.plot(ft[4], '-x');plt.title('feature 4')
# plt.figure();plt.imshow(Y.cpu().numpy(), aspect='auto'); plt.title('True labels')
# plt.figure();plt.imshow(Y_hat.cpu().numpy(), aspect='auto'); plt.title('Estimated labels')

