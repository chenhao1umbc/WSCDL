"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
opts = OPT()
opts.lamb = 0.1  # for sparsity penalty
opts.eta = 10  # for label penalty
opts.mu = 1  # for low rank penalty
X, Y, ft = load_toy(opts)
D, D0, S, S0, W = init(X, opts)
loss = torch.tensor([], device=opts.dev)
loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
print('The initial loss function value is %3.4e:' %loss[-1])
t = time.time()
for i in range(opts.maxiter):
    t0 = time.time()
    D = updateD([D, D0, S, S0, W], X, Y, opts)
    # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    # print('pass D, time is %3.2f' % (time.time() - t)); t = time.time()
    # print('loss function value is %3.4e:' %loss[-1])

    # D0 = updateD0([D, D0, S, S0], X, Y, opts)
    # # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    # # print('pass D0, time is %3.2f' % (time.time() - t)); t = time.time()
    # # print('loss function value is %3.4e:' %loss[-1])

    S = updateS([D, D0, S, S0, W], X, Y, opts)
    # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    # print('pass S, time is %3.2f' % (time.time() - t)); t = time.time()
    # print('loss function value is %3.4e:' %loss[-1])
    print('check sparsity, None-zero percentage is : %1.3f' % (1 - S[S == 0].shape[0] / S.numel()))

    S0 = updateS0([D, D0, S, S0], X, Y, opts)
    # loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    # print('pass S0, time is %3.2f' % (time.time() - t)); t = time.time()
    # print('loss function value is %3.4e:' %loss[-1])

    W = updateW([S, W], Y, opts)
    loss = torch.cat((loss, loss_fun(X, Y, D, D0, S, S0, W, opts).reshape(1)))
    # print('pass W, time is %3.2f' % (time.time() - t)); t = time.time()
    # print('loss function value is %3.4e:' %loss[-1])

    if i > 10 and abs((loss[-1]-loss[-2])/loss[-2]) < 1e-4 : break
    print('In the %1.0f epoch, the training time is :%3.2f \n' % (i, time.time() - t0))

# ll = loss[:-1]-loss[1:]
# if ll[ll<0].shape[0] >0 : print('****This result is not trustworthy because loss is not monotonically decreasing****')
print('After %1.0f epochs, the loss function value is %3.4e:' %(i, loss[-1]))
print('All done, the total running time is :%3.2f \n' % (time.time() -t))
# torch.save([D, D0, S, S0, W, opts, loss], '../DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')
plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts)


# testing section
X_test, Y_test, ft = load_toy(opts)
_, _, S, S0, _ = init(X, opts)
acc, y_hat = test_wscdl(D, D0, S, S0, W, X_test, Y_test, opts)
print('the test data accuracy is : ', acc)
plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts)

